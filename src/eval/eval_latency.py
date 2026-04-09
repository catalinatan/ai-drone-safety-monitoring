"""
Benchmark YOLO model inference latency across scenes and image sizes.

Measures both inference-only time (forward pass) and end-to-end time
(preprocess + forward + postprocess), using CUDA events for accurate GPU
timing. Captures peak GPU memory, model file size, and parameter count.
The benchmarking hardware is auto-documented to hardware.json so results
remain interpretable later.

Models are discovered from:
    runs/segment/runs/segment/{scene}_hazard_*/weights/best.pt    (scenes)
    runs/segment/runs/segment/human_detection_real_*/weights/best.pt (human)

Test images are loaded from:
    data/test_dataset/images/{scene}/train/images/

Outputs are written to:
    eval_output/model_latency/{scene}_segmentation/   (or human_detection/)
        hardware.json
        results.csv
        raw_{model}_imgsz{N}.csv
        latency_distribution.png
        latency_bars.png
        memory_size.png
        pareto.png              (only with --pareto, requires eval_scene.py CSV)

Usage:
    python -m src.eval.eval_latency --scene bridge
    python -m src.eval.eval_latency --scene all --pareto
    python -m src.eval.eval_latency --scene railway --imgsz 640 1280
    python -m src.eval.eval_latency --scene ship --warmup 30 --iterations 300
"""

from __future__ import annotations

import argparse
import csv
import gc
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TEST_DATASET_ROOT = Path("data/test_dataset/images")
MODELS_ROOT       = Path("runs/segment/runs/segment")
OUTPUT_ROOT       = Path("eval_output/model_latency")
ACCURACY_ROOT     = Path("eval_output")

IMAGE_EXTENSIONS  = {".jpg", ".jpeg", ".png"}
CONF_THRESHOLD    = 0.25

DEFAULT_WARMUP    = 20
DEFAULT_ITERS     = 200
DEFAULT_IMGSZS    = [640, 1280]
MAX_TEST_IMAGES   = 50    # cap so we aren't bottlenecked on image loading

# Scene → (model glob, display folder name, accuracy CSV scene key)
SCENE_CONFIG = {
    "bridge":  ("bridge_hazard_*/weights/best.pt",          "bridge_segmentation",  "bridge"),
    "ship":    ("ship_hazard_*/weights/best.pt",            "ship_segmentation",    "ship"),
    "railway": ("railway_hazard_*/weights/best.pt",         "railway_segmentation", "railway"),
    "human":   ("human_detection_real_*/weights/best.pt",   "human_detection",      "human"),
}

MODEL_NAME_PREFIXES = (
    "bridge_hazard_",
    "ship_hazard_",
    "railway_hazard_",
    "human_detection_real_",
)


# ---------------------------------------------------------------------------
# Hardware documentation
# ---------------------------------------------------------------------------

def capture_hardware_info() -> dict:
    """Snapshot the benchmarking environment so results remain interpretable."""
    info: dict = {
        "platform":        platform.platform(),
        "processor":       platform.processor(),
        "python_version":  platform.python_version(),
        "torch_version":   torch.__version__,
        "cuda_available":  torch.cuda.is_available(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "precision":       "fp32",   # ultralytics default for .predict()
        "batch_size":      1,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update({
            "cuda_version":           torch.version.cuda,
            "cudnn_version":          torch.backends.cudnn.version(),
            "gpu_name":               torch.cuda.get_device_name(0),
            "gpu_vram_gb":            round(props.total_memory / 1e9, 2),
            "gpu_compute_capability": f"{props.major}.{props.minor}",
        })
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                info["nvidia_driver"] = result.stdout.strip()
        except Exception:
            pass
    return info


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def discover_models(scene: str) -> list[Path]:
    glob, _, _ = SCENE_CONFIG[scene]
    return sorted(MODELS_ROOT.glob(glob))


def display_model_name(model_path: Path) -> str:
    """Strip the scene prefix so we just show the YOLO variant."""
    parent = model_path.parent.parent.name
    for prefix in MODEL_NAME_PREFIXES:
        if parent.startswith(prefix):
            return parent[len(prefix):]
    return parent


def load_test_images(scene: str) -> list[np.ndarray]:
    images_dir = TEST_DATASET_ROOT / scene / "train" / "images"
    if not images_dir.exists():
        return []
    paths = sorted(p for p in images_dir.iterdir()
                   if p.suffix.lower() in IMAGE_EXTENSIONS)
    images: list[np.ndarray] = []
    for p in paths[:MAX_TEST_IMAGES]:
        img = cv2.imread(str(p))
        if img is not None:
            images.append(img)
    return images


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def _stats(arr: list[float]) -> dict:
    if not arr:
        return {k: 0.0 for k in ("mean", "median", "std", "min", "max", "p95", "p99")}
    a = np.asarray(arr, dtype=np.float64)
    return {
        "mean":   float(np.mean(a)),
        "median": float(np.median(a)),
        "std":    float(np.std(a)),
        "min":    float(np.min(a)),
        "max":    float(np.max(a)),
        "p95":    float(np.percentile(a, 95)),
        "p99":    float(np.percentile(a, 99)),
    }


def benchmark_model(model_path: Path, images: list[np.ndarray], imgsz: int,
                    warmup: int, iterations: int) -> dict:
    """Benchmark one model at one image size.

    Returns a result dict containing inference-only and end-to-end timing
    stats, peak GPU memory, model size on disk, and parameter count.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = YOLO(str(model_path))

    # Warmup — first few forward passes include kernel autotune and allocation
    for i in range(warmup):
        _ = model(images[i % len(images)], imgsz=imgsz, conf=CONF_THRESHOLD,
                  verbose=False, device=device)

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    inference_ms: list[float] = []
    e2e_ms:       list[float] = []

    for i in range(iterations):
        img = images[i % len(images)]

        if device == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)
            start.record()
            results = model(img, imgsz=imgsz, conf=CONF_THRESHOLD,
                            verbose=False, device=device)
            end.record()
            torch.cuda.synchronize()
            e2e_ms.append(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            results = model(img, imgsz=imgsz, conf=CONF_THRESHOLD,
                            verbose=False, device=device)
            e2e_ms.append((time.perf_counter() - t0) * 1000.0)

        # Ultralytics records an internal inference-only timer
        speed = getattr(results[0], "speed", None) or {}
        if "inference" in speed:
            inference_ms.append(float(speed["inference"]))

    peak_mb = (torch.cuda.max_memory_allocated() / 1e6) if device == "cuda" else 0.0
    size_mb = model_path.stat().st_size / 1e6
    try:
        params = sum(p.numel() for p in model.model.parameters())
    except Exception:
        params = 0

    result = {
        "inference_ms":  _stats(inference_ms),
        "e2e_ms":        _stats(e2e_ms),
        "fps_median":    (1000.0 / np.median(e2e_ms)) if e2e_ms else 0.0,
        "peak_gpu_mb":   peak_mb,
        "model_size_mb": size_mb,
        "params":        params,
        "raw_inference": inference_ms,
        "raw_e2e":       e2e_ms,
    }

    # Clean up so peak memory on the next model is isolated
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def save_results_csv(out_dir: Path, results: list[dict]) -> Path:
    csv_path = out_dir / "results.csv"
    fields = [
        "model", "imgsz", "params", "model_size_mb",
        "median_e2e_ms", "mean_e2e_ms", "p95_e2e_ms", "p99_e2e_ms", "std_e2e_ms",
        "median_inference_ms", "mean_inference_ms", "p95_inference_ms",
        "fps_median", "peak_gpu_mb",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "model":               r["model"],
                "imgsz":               r["imgsz"],
                "params":              r["params"],
                "model_size_mb":       round(r["model_size_mb"], 2),
                "median_e2e_ms":       round(r["e2e_ms"]["median"], 2),
                "mean_e2e_ms":         round(r["e2e_ms"]["mean"], 2),
                "p95_e2e_ms":          round(r["e2e_ms"]["p95"], 2),
                "p99_e2e_ms":          round(r["e2e_ms"]["p99"], 2),
                "std_e2e_ms":          round(r["e2e_ms"]["std"], 2),
                "median_inference_ms": round(r["inference_ms"]["median"], 2),
                "mean_inference_ms":   round(r["inference_ms"]["mean"], 2),
                "p95_inference_ms":    round(r["inference_ms"]["p95"], 2),
                "fps_median":          round(r["fps_median"], 1),
                "peak_gpu_mb":         round(r["peak_gpu_mb"], 1),
            })
    return csv_path


def save_raw_csv(out_dir: Path, results: list[dict]) -> None:
    """Per-iteration raw timings for report appendix / reproducibility."""
    for r in results:
        path = out_dir / f"raw_{r['model']}_imgsz{r['imgsz']}.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "inference_ms", "e2e_ms"])
            inf = r.get("raw_inference") or []
            e2e = r.get("raw_e2e") or []
            for i in range(max(len(inf), len(e2e))):
                writer.writerow([
                    i,
                    round(inf[i], 3) if i < len(inf) else "",
                    round(e2e[i], 3) if i < len(e2e) else "",
                ])


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_latency_distribution(out_dir: Path, results: list[dict], title: str) -> None:
    imgszs = sorted(set(r["imgsz"] for r in results))
    fig, axes = plt.subplots(1, len(imgszs),
                             figsize=(max(7, 5 * len(imgszs)), 5),
                             squeeze=False)
    for ax, sz in zip(axes[0], imgszs):
        group  = [r for r in results if r["imgsz"] == sz]
        data   = [r["raw_e2e"] for r in group]
        labels = [r["model"] for r in group]
        ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_title(f"{title} @ imgsz={sz}")
        ax.set_ylabel("End-to-end latency (ms)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "latency_distribution.png", dpi=150)
    plt.close(fig)


def plot_latency_bars(out_dir: Path, results: list[dict], title: str) -> None:
    imgszs = sorted(set(r["imgsz"] for r in results))
    models = sorted(set(r["model"] for r in results))
    x      = np.arange(len(models))
    width  = 0.8 / max(len(imgszs), 1)

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.6), 5))

    for i, sz in enumerate(imgszs):
        medians, p95s = [], []
        for m in models:
            rec = next((r for r in results if r["model"] == m and r["imgsz"] == sz), None)
            medians.append(rec["e2e_ms"]["median"] if rec else 0.0)
            p95s.append(rec["e2e_ms"]["p95"] if rec else 0.0)
        err_high = [p - med for p, med in zip(p95s, medians)]
        offset   = (i - (len(imgszs) - 1) / 2) * width
        ax.bar(x + offset, medians, width, label=f"imgsz={sz}",
               yerr=[[0] * len(err_high), err_high], capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("End-to-end latency (ms)")
    ax.set_title(f"{title} — median latency (error bars = P95)")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / "latency_bars.png", dpi=150)
    plt.close(fig)


def plot_memory_size(out_dir: Path, results: list[dict], title: str) -> None:
    imgszs = sorted(set(r["imgsz"] for r in results))
    models = sorted(set(r["model"] for r in results))
    x      = np.arange(len(models))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(12, len(models) * 2), 5))

    sizes = [next((r["model_size_mb"] for r in results if r["model"] == m), 0.0)
             for m in models]
    ax1.bar(x, sizes, color="steelblue")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=30, ha="right")
    ax1.set_ylabel("Model file size (MB)")
    ax1.set_title("Model size on disk")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.5)

    width = 0.8 / max(len(imgszs), 1)
    for i, sz in enumerate(imgszs):
        peaks = []
        for m in models:
            rec = next((r for r in results if r["model"] == m and r["imgsz"] == sz), None)
            peaks.append(rec["peak_gpu_mb"] if rec else 0.0)
        offset = (i - (len(imgszs) - 1) / 2) * width
        ax2.bar(x + offset, peaks, width, label=f"imgsz={sz}")

    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=30, ha="right")
    ax2.set_ylabel("Peak GPU memory (MB)")
    ax2.set_title("Peak GPU memory during inference")
    ax2.legend()
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_dir / "memory_size.png", dpi=150)
    plt.close(fig)


def plot_pareto(out_dir: Path, results: list[dict], scene: str, title: str) -> None:
    """Overlay latency (x) vs IoU (y) using eval_scene.py's results.csv.

    Prints a warning and skips the plot if no accuracy data is available.
    """
    accuracy_csv = ACCURACY_ROOT / scene / "results.csv"
    if not accuracy_csv.exists():
        print(f"  [pareto] no accuracy data at {accuracy_csv} — skipping")
        return

    accuracy: dict[str, float] = {}
    with open(accuracy_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            full = row.get("model", "")
            for prefix in MODEL_NAME_PREFIXES:
                if full.startswith(prefix):
                    full = full[len(prefix):]
                    break
            try:
                accuracy[full] = float(row.get("mean_iou", "nan"))
            except ValueError:
                continue

    if not accuracy:
        print(f"  [pareto] accuracy CSV had no usable rows — skipping")
        return

    imgszs = sorted(set(r["imgsz"] for r in results))
    fig, ax = plt.subplots(figsize=(9, 6))

    for sz in imgszs:
        xs, ys, labels = [], [], []
        for r in results:
            if r["imgsz"] != sz or r["model"] not in accuracy:
                continue
            xs.append(r["e2e_ms"]["median"])
            ys.append(accuracy[r["model"]])
            labels.append(r["model"])
        if not xs:
            continue
        ax.scatter(xs, ys, s=90, label=f"imgsz={sz}")
        for xv, yv, lbl in zip(xs, ys, labels):
            ax.annotate(lbl, (xv, yv), xytext=(6, 4),
                        textcoords="offset points", fontsize=8)

    ax.set_xlabel("Median end-to-end latency (ms) — lower is better")
    ax.set_ylabel("Mean IoU — higher is better")
    ax.set_title(f"{title} — accuracy vs latency")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "pareto.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-scene driver
# ---------------------------------------------------------------------------

def evaluate_scene(scene: str, imgszs: list[int], warmup: int, iterations: int,
                   make_pareto: bool) -> None:
    _, display, scene_key = SCENE_CONFIG[scene]
    out_dir = OUTPUT_ROOT / display
    out_dir.mkdir(parents=True, exist_ok=True)

    models = discover_models(scene)
    if not models:
        print(f"[{scene.upper()}] No models found under {MODELS_ROOT} — skipping.")
        return

    images = load_test_images(scene)
    if not images:
        print(f"[{scene.upper()}] No test images at "
              f"{TEST_DATASET_ROOT / scene / 'train' / 'images'} — skipping.")
        return

    print(f"\n{'=' * 70}")
    print(f"  {display.upper()}  |  {len(models)} model(s)  |  {len(images)} images")
    print(f"{'=' * 70}")

    hw = capture_hardware_info()
    with open(out_dir / "hardware.json", "w") as f:
        json.dump(hw, f, indent=2)
    print(f"  GPU:     {hw.get('gpu_name', 'N/A')} "
          f"({hw.get('gpu_vram_gb', '?')} GB)")
    print(f"  PyTorch: {hw.get('torch_version')} | "
          f"CUDA {hw.get('cuda_version', 'N/A')} | "
          f"cuDNN {hw.get('cudnn_version', 'N/A')}")
    print(f"  Driver:  {hw.get('nvidia_driver', 'N/A')}")

    results: list[dict] = []
    for model_path in models:
        name = display_model_name(model_path)
        for sz in imgszs:
            print(f"\n  [{name} @ imgsz={sz}]  "
                  f"warmup={warmup}, iterations={iterations}")
            stats = benchmark_model(model_path, images, sz, warmup, iterations)
            results.append({"model": name, "imgsz": sz, **stats})

            e2e = stats["e2e_ms"]
            print(f"    end-to-end    median={e2e['median']:.2f} ms  "
                  f"P95={e2e['p95']:.2f} ms  "
                  f"std={e2e['std']:.2f} ms  "
                  f"FPS={stats['fps_median']:.1f}")
            print(f"    inference     median={stats['inference_ms']['median']:.2f} ms")
            print(f"    peak GPU      {stats['peak_gpu_mb']:.1f} MB  "
                  f"size={stats['model_size_mb']:.1f} MB  "
                  f"params={stats['params']:,}")

    print(f"\n  Writing outputs to {out_dir}")
    save_results_csv(out_dir, results)
    save_raw_csv(out_dir, results)
    plot_latency_distribution(out_dir, results, display)
    plot_latency_bars(out_dir, results, display)
    plot_memory_size(out_dir, results, display)
    if make_pareto:
        plot_pareto(out_dir, results, scene_key, display)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark YOLO model inference latency.")
    parser.add_argument(
        "--scene",
        choices=["bridge", "ship", "railway", "human", "all"],
        default="all",
        help="Which scene/task to benchmark (default: all).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs="+",
        default=DEFAULT_IMGSZS,
        help="One or more inference image sizes. Each model is benchmarked "
             "at every size (default: 640 1280).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help=f"Warmup iterations before timing (default: {DEFAULT_WARMUP}).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERS,
        help=f"Timed iterations per model (default: {DEFAULT_ITERS}).",
    )
    parser.add_argument(
        "--pareto",
        action="store_true",
        help="Overlay latency vs IoU using eval_scene.py's results.csv "
             "(if available) and save pareto.png.",
    )
    args = parser.parse_args()

    # Match production runtime flags so latency reflects real deployment
    torch.backends.cudnn.benchmark = True

    scenes = (["bridge", "ship", "railway", "human"]
              if args.scene == "all" else [args.scene])

    for scene in scenes:
        evaluate_scene(scene, args.imgsz, args.warmup,
                       args.iterations, args.pareto)

    print(f"\nAll results under: {OUTPUT_ROOT}/")


if __name__ == "__main__":
    main()
