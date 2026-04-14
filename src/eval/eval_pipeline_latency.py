"""
Benchmark end-to-end response time of the alarm-trigger pipeline, broken
down by component.

Stages measured per image:
  1. image_load_ms       — cv2.imread from disk
  2. seg_inference_ms    — scene segmentation forward pass (imgsz=640)
  3. seg_postprocess_ms  — mask resize + threshold → danger zone
  4. human_inference_ms  — human detection forward pass (imgsz=1280)
  5. human_postprocess_ms— per-instance mask resize + threshold
  6. overlap_ms          — numpy logical_and / area computation
  7. alarm_decision_ms   — threshold comparison
  8. total_ms            — sum of the above (end-to-end response time)

The total is what gets compared against benchmark methods in the report.

Output:
    eval_output/pipeline_latency/
        hardware.json        — GPU / driver / torch info for reproducibility
        results.csv          — report-ready breakdown (median/mean/p95 per stage)
        per_image.csv        — every stage time for every timed iteration
        breakdown.png        — stacked bar showing % contribution per stage

Usage:
    python -m src.eval.eval_pipeline_latency
    python -m src.eval.eval_pipeline_latency --warmup 20 --iterations 200
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO

from src.eval.eval_alarm_trigger import (
    SEG_IMGSZ,
    HUMAN_IMGSZ,
    CONF_THRESHOLD,
    OVERLAP_ROOT,
    POSITIVE_DIR,
    NEGATIVE_DIR,
    IMAGE_EXTENSIONS,
    detect_scene,
    find_cleanup,
)


# ---------------------------------------------------------------------------
# Model paths — point at the deployment copies under models/ rather than the
# raw ultralytics run folders.
# ---------------------------------------------------------------------------

MODELS_ROOT = Path("models")

SEG_MODELS = {
    "bridge": MODELS_ROOT / "scene_segmentation/bridge/bridge_hazard_yolo11s-seg/weights/best.pt",
    "railway": MODELS_ROOT
    / "scene_segmentation/railway/railway_hazard_yolo11s-seg/weights/best.pt",
    "ship": MODELS_ROOT / "scene_segmentation/ship/ship_hazard_yolo11s-seg/weights/best.pt",
}
HUMAN_MODEL_PATH = MODELS_ROOT / "human_detection/yolo11s-seg/weights/best.pt"


OUTPUT_DIR = Path("eval_output/pipeline_latency")
DEFAULT_WARMUP = 20
DEFAULT_ITERATIONS = 100
ALARM_THRESHOLD = 0.15  # matches the typical operating point


STAGE_ORDER = [
    "image_load_ms",
    "seg_inference_ms",
    "seg_postprocess_ms",
    "human_inference_ms",
    "human_postprocess_ms",
    "overlap_ms",
    "alarm_decision_ms",
]

STAGE_DISPLAY = {
    "image_load_ms": "Image load (disk → BGR)",
    "seg_inference_ms": "Scene seg forward (imgsz=640)",
    "seg_postprocess_ms": "Scene seg postprocess",
    "human_inference_ms": "Human det forward (imgsz=1280)",
    "human_postprocess_ms": "Human det postprocess",
    "overlap_ms": "Overlap computation",
    "alarm_decision_ms": "Alarm decision",
}


# ---------------------------------------------------------------------------
# Hardware info (same format as eval_latency so results are comparable)
# ---------------------------------------------------------------------------


def capture_hardware_info() -> dict:
    info: dict = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "precision": "fp32",
        "batch_size": 1,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            {
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_vram_gb": round(props.total_memory / 1e9, 2),
                "gpu_compute_capability": f"{props.major}.{props.minor}",
            }
        )
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                info["nvidia_driver"] = result.stdout.strip()
        except Exception:
            pass
    return info


# ---------------------------------------------------------------------------
# Timing primitives
# ---------------------------------------------------------------------------


class GpuTimer:
    """Wall-clock stopwatch that syncs CUDA when available.

    CUDA events only measure GPU work, but several stages in this pipeline
    are CPU (cv2.imread, numpy ops). Using synchronised wall-clock gives a
    consistent number that fairly reflects the response time a user sees.
    """

    def __init__(self, use_cuda: bool):
        self.use_cuda = use_cuda
        self.t = 0.0

    def __enter__(self):
        if self.use_cuda:
            torch.cuda.synchronize()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.use_cuda:
            torch.cuda.synchronize()
        self.t = (time.perf_counter() - self._t0) * 1000.0


def _stats(arr: list[float]) -> dict:
    if not arr:
        return {k: 0.0 for k in ("mean", "median", "std", "p95", "p99", "min", "max")}
    a = np.asarray(arr, dtype=np.float64)
    return {
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "std": float(np.std(a)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


# ---------------------------------------------------------------------------
# Pipeline stage wrappers (instrumented versions of eval_alarm_trigger ops)
# ---------------------------------------------------------------------------


def run_seg_inference(model: YOLO, source, device: str):
    return model(
        source, conf=CONF_THRESHOLD, imgsz=SEG_IMGSZ, verbose=False, save=False, device=device
    )


def run_human_inference(model: YOLO, source, device: str):
    return model(
        source, conf=CONF_THRESHOLD, imgsz=HUMAN_IMGSZ, verbose=False, save=False, device=device
    )


def postprocess_seg(results, img_w: int, img_h: int) -> np.ndarray:
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if results[0].masks is None:
        return mask
    for m in results[0].masks.data:
        resized = cv2.resize(m.cpu().numpy(), (img_w, img_h))
        mask = np.maximum(mask, (resized > 0.5).astype(np.uint8))
    return mask


def postprocess_human(results, img_w: int, img_h: int) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    if results[0].masks is None:
        return masks
    for m in results[0].masks.data:
        resized = cv2.resize(m.cpu().numpy(), (img_w, img_h))
        binary = (resized > 0.5).astype(np.uint8)
        if binary.sum() > 0:
            masks.append(binary)
    return masks


def compute_overlap(danger_zone: np.ndarray, human_masks: list[np.ndarray]) -> float:
    max_overlap = 0.0
    for person in human_masks:
        area = int(person.sum())
        if area == 0:
            continue
        overlap = float(np.logical_and(person, danger_zone).sum()) / area
        if overlap > max_overlap:
            max_overlap = overlap
    return max_overlap


def decide_alarm(max_overlap: float, threshold: float) -> bool:
    return max_overlap >= threshold


# ---------------------------------------------------------------------------
# Image collection
# ---------------------------------------------------------------------------


def collect_images() -> list[tuple[Path, Path, str]]:
    """
    Return (seg_source, human_source, scene) triples mirroring the real
    pipeline: for positives the cleanup image feeds the scene model while
    the original image feeds human detection.
    """
    triples: list[tuple[Path, Path, str]] = []
    for folder in (POSITIVE_DIR, NEGATIVE_DIR):
        if not folder.exists():
            continue
        for img_path in sorted(folder.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS or not img_path.is_file():
                continue
            scene = detect_scene(img_path.name)
            if scene is None or scene not in SEG_MODELS:
                continue
            if folder == POSITIVE_DIR:
                cleanup = find_cleanup(img_path)
                seg_source = cleanup if cleanup else img_path
            else:
                seg_source = img_path
            triples.append((seg_source, img_path, scene))
    return triples


# ---------------------------------------------------------------------------
# Benchmark driver
# ---------------------------------------------------------------------------


def benchmark(warmup: int, iterations: int) -> tuple[list[dict], dict]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_cuda = device == "cuda"

    print(f"\nLoading models on {device}...")
    seg_models = {s: YOLO(str(p)) for s, p in SEG_MODELS.items()}
    human_model = YOLO(str(HUMAN_MODEL_PATH))

    triples = collect_images()
    if not triples:
        raise RuntimeError(f"No test images found under {OVERLAP_ROOT}")
    print(f"Collected {len(triples)} images for pipeline benchmarking")

    # ── Warmup: run the full pipeline on a rotating set of images so every
    #    scene model gets its kernels compiled before timing starts.
    print(f"Warmup ({warmup} iterations)...")
    for i in range(warmup):
        seg_src, hum_src, scene = triples[i % len(triples)]
        seg_results = run_seg_inference(seg_models[scene], str(seg_src), device)
        hum_results = run_human_inference(human_model, str(hum_src), device)
        _ = postprocess_seg(seg_results, *cv2.imread(str(seg_src)).shape[1::-1])
        _ = postprocess_human(hum_results, *cv2.imread(str(hum_src)).shape[1::-1])
    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # ── Timed iterations
    print(f"Timing  ({iterations} iterations)...")
    rows: list[dict] = []

    for i in range(iterations):
        seg_src, hum_src, scene = triples[i % len(triples)]
        seg_model = seg_models[scene]

        row: dict = {
            "iteration": i,
            "scene": scene,
            "seg_source": seg_src.name,
            "human_source": hum_src.name,
        }

        # 1. image load (we load the HUMAN source since that's the "live" frame;
        #    the cleanup path is synthetic and wouldn't exist in production).
        with GpuTimer(use_cuda) as t:
            orig = cv2.imread(str(hum_src))
        row["image_load_ms"] = t.t
        if orig is None:
            continue
        h, w = orig.shape[:2]

        # If seg_src differs from hum_src (positive case with a cleanup image),
        # it represents ground-truth-only scaffolding rather than a real
        # runtime input. To keep the timing honest against production, we
        # feed seg_src to the seg model (so the measurement reflects what the
        # alarm eval sees) but we *don't* re-time a second image load.
        seg_source_for_model = str(seg_src)

        # 2. scene seg forward
        with GpuTimer(use_cuda) as t:
            seg_results = run_seg_inference(seg_model, seg_source_for_model, device)
        row["seg_inference_ms"] = t.t

        # 3. scene seg postprocess (resize mask back to original frame size)
        with GpuTimer(use_cuda) as t:
            danger_zone = postprocess_seg(seg_results, w, h)
        row["seg_postprocess_ms"] = t.t

        # 4. human det forward
        with GpuTimer(use_cuda) as t:
            hum_results = run_human_inference(human_model, str(hum_src), device)
        row["human_inference_ms"] = t.t

        # 5. human det postprocess
        with GpuTimer(use_cuda) as t:
            human_masks = postprocess_human(hum_results, w, h)
        row["human_postprocess_ms"] = t.t

        # 6. overlap
        with GpuTimer(use_cuda) as t:
            max_overlap = compute_overlap(danger_zone, human_masks)
        row["overlap_ms"] = t.t

        # 7. alarm decision
        with GpuTimer(use_cuda) as t:
            _ = decide_alarm(max_overlap, ALARM_THRESHOLD)
        row["alarm_decision_ms"] = t.t

        row["total_ms"] = sum(row[s] for s in STAGE_ORDER)
        row["max_overlap"] = max_overlap
        row["n_humans"] = len(human_masks)
        rows.append(row)

        if (i + 1) % 20 == 0:
            print(f"  iter {i + 1}/{iterations}  last total={row['total_ms']:.1f} ms")

    # ── Aggregate stats per stage + total
    agg: dict[str, dict] = {}
    for stage in STAGE_ORDER + ["total_ms"]:
        agg[stage] = _stats([r[stage] for r in rows])

    return rows, agg


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_results_csv(out_dir: Path, agg: dict) -> Path:
    path = out_dir / "results.csv"
    total_median = agg["total_ms"]["median"] or 1.0
    fields = [
        "stage",
        "display_name",
        "median_ms",
        "mean_ms",
        "p95_ms",
        "p99_ms",
        "std_ms",
        "share_of_total_pct",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for stage in STAGE_ORDER:
            s = agg[stage]
            w.writerow(
                {
                    "stage": stage,
                    "display_name": STAGE_DISPLAY[stage],
                    "median_ms": round(s["median"], 3),
                    "mean_ms": round(s["mean"], 3),
                    "p95_ms": round(s["p95"], 3),
                    "p99_ms": round(s["p99"], 3),
                    "std_ms": round(s["std"], 3),
                    "share_of_total_pct": round(100.0 * s["median"] / total_median, 1),
                }
            )
        s = agg["total_ms"]
        w.writerow(
            {
                "stage": "total_ms",
                "display_name": "TOTAL RESPONSE TIME",
                "median_ms": round(s["median"], 3),
                "mean_ms": round(s["mean"], 3),
                "p95_ms": round(s["p95"], 3),
                "p99_ms": round(s["p99"], 3),
                "std_ms": round(s["std"], 3),
                "share_of_total_pct": 100.0,
            }
        )
    return path


def save_per_image_csv(out_dir: Path, rows: list[dict]) -> Path:
    path = out_dir / "per_image.csv"
    fields = (
        ["iteration", "scene", "seg_source", "human_source"]
        + STAGE_ORDER
        + ["total_ms", "max_overlap", "n_humans"]
    )
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in r.items()
                    if k in fields
                }
            )
    return path


def print_report_table(agg: dict) -> None:
    print(f"\n{'=' * 78}")
    print(f"  PIPELINE RESPONSE TIME BREAKDOWN")
    print(f"{'=' * 78}")
    header = f"  {'Stage':<38} {'median':>9} {'p95':>9} {'share':>8}"
    print(header)
    print(f"  {'─' * (len(header) - 2)}")
    total_median = agg["total_ms"]["median"] or 1.0
    for stage in STAGE_ORDER:
        s = agg[stage]
        share = 100.0 * s["median"] / total_median
        print(
            f"  {STAGE_DISPLAY[stage]:<38} {s['median']:>7.2f}ms {s['p95']:>7.2f}ms {share:>6.1f}%"
        )
    print(f"  {'─' * (len(header) - 2)}")
    t = agg["total_ms"]
    print(f"  {'TOTAL RESPONSE TIME':<38} {t['median']:>7.2f}ms {t['p95']:>7.2f}ms {100.0:>6.1f}%")
    print(
        f"  {'(mean=%.2fms, std=%.2fms, throughput=%.1f FPS)' % (t['mean'], t['std'], 1000.0 / t['median'] if t['median'] else 0.0):<78}"
    )


def plot_breakdown(out_dir: Path, agg: dict) -> None:
    labels = [STAGE_DISPLAY[s] for s in STAGE_ORDER]
    medians = [agg[s]["median"] for s in STAGE_ORDER]
    p95s = [agg[s]["p95"] for s in STAGE_ORDER]
    total = agg["total_ms"]["median"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [2.2, 1.0]})

    # Left: horizontal bar with median + p95
    y = np.arange(len(labels))
    ax1.barh(y, medians, color="steelblue", label="median")
    ax1.barh(
        y,
        [p - m for p, m in zip(p95s, medians)],
        left=medians,
        color="lightsteelblue",
        label="p95 - median",
    )
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)
    ax1.invert_yaxis()
    ax1.set_xlabel("Latency (ms)")
    ax1.set_title(f"Per-stage latency  (total median = {total:.1f} ms)")
    ax1.grid(True, axis="x", linestyle="--", alpha=0.4)
    ax1.legend(loc="lower right")

    # Right: stacked single bar showing share of total
    bottom = 0.0
    colours = plt.get_cmap("tab10")(np.linspace(0, 1, len(labels)))
    legend_labels = []
    for label, med, c in zip(labels, medians, colours):
        share = 100.0 * med / (total or 1.0)
        ax2.bar(
            ["Pipeline"],
            [med],
            bottom=bottom,
            color=c,
            label=f"{label} — {med:.1f} ms ({share:.1f}%)",
        )
        # Only inline-label segments that are large enough not to overlap
        if share >= 8.0:
            ax2.text(
                0,
                bottom + med / 2,
                f"{med:.1f}ms ({share:.0f}%)",
                ha="center",
                va="center",
                fontsize=9,
                color="white",
                fontweight="bold",
            )
        bottom += med
    ax2.set_ylabel("Cumulative latency (ms)")
    ax2.set_title("Response time composition")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax2.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)

    fig.suptitle("Alarm-trigger pipeline response time", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "breakdown.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark end-to-end alarm-trigger pipeline latency."
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    hw = capture_hardware_info()
    with open(OUTPUT_DIR / "hardware.json", "w") as f:
        json.dump(hw, f, indent=2)
    print(f"GPU     : {hw.get('gpu_name', 'N/A')} ({hw.get('gpu_vram_gb', '?')} GB)")
    print(
        f"Torch   : {hw.get('torch_version')}  "
        f"CUDA {hw.get('cuda_version', 'N/A')}  "
        f"cuDNN {hw.get('cudnn_version', 'N/A')}"
    )
    print(f"Driver  : {hw.get('nvidia_driver', 'N/A')}")

    rows, agg = benchmark(args.warmup, args.iterations)

    print_report_table(agg)

    results_path = save_results_csv(OUTPUT_DIR, agg)
    per_image_path = save_per_image_csv(OUTPUT_DIR, rows)
    plot_breakdown(OUTPUT_DIR, agg)

    print(f"\n  Report-ready CSV   → {results_path}")
    print(f"  Per-iteration CSV  → {per_image_path}")
    print(f"  Breakdown plot     → {OUTPUT_DIR}/breakdown.png")
    print(f"  Hardware info      → {OUTPUT_DIR}/hardware.json")


if __name__ == "__main__":
    main()
