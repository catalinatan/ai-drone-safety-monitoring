"""
Cross-task model selection: compare YOLO variants across all segmentation
tasks and latency to recommend the best model for production.

Methodology
-----------
1.  Collect per-task accuracy metrics:
      - Scene segmentation (bridge, railway, ship): mean IoU
      - Human detection: F1 score

2.  Scene segmentation tasks are aggregated into a single score using a
    weighted harmonic mean (WHM) of IoU across bridge/railway/ship.
    Human detection F1 is reported separately — these are fundamentally
    different tasks and blending them into one number hides tradeoffs.

3.  Collect latency (median inference ms at imgsz used in production).
    Averaged across available scenes since latency is architecture-dependent,
    not scene-dependent.

4.  Plot:
      a. Grouped bar chart of per-task scores by model variant
      b. Dual Pareto frontier: scene seg score vs latency, human F1 vs latency
      c. Radar / spider chart showing the profile of each variant
      d. Summary table printed to stdout and saved as CSV

Weights for scene segmentation default to equal (bridge=1, railway=1, ship=1).
Override with --seg-weights.

Usage:
    python -m src.eval.eval_model_selection
    python -m src.eval.eval_model_selection --seg-weights 1 1 2
    python -m src.eval.eval_model_selection --imgsz 1280
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EVAL_ROOT    = Path("eval_output")
LATENCY_ROOT = EVAL_ROOT / "model_latency"
OUTPUT_DIR   = EVAL_ROOT / "model_selection"

# Scene segmentation tasks (aggregated into one score via WHM of IoU)
SEG_TASK_CONFIGS = [
    ("bridge",  EVAL_ROOT / "bridge"  / "results.csv", "mean_iou"),
    ("railway", EVAL_ROOT / "railway" / "results.csv", "mean_iou"),
    ("ship",    EVAL_ROOT / "ship"    / "results.csv", "mean_iou"),
]

# Human detection task (reported separately)
HUMAN_TASK_CONFIG = ("human", EVAL_ROOT / "human_bridge" / "results.csv", "f1")

LATENCY_SCENES = [
    LATENCY_ROOT / "bridge_segmentation" / "results.csv",
    LATENCY_ROOT / "ship_segmentation"   / "results.csv",
    # Add railway_segmentation here when available
]

# Mapping from full model name → short variant key
SCENE_PREFIXES = (
    "bridge_hazard_",
    "railway_hazard_",
    "ship_hazard_",
    "human_detection_real_",
)

VARIANT_ORDER = ["yolo11n-seg", "yolo11s-seg", "yolo11m-seg", "yolo11l-seg", "yolo26l-seg"]
VARIANT_LABELS = ["YOLO11n", "YOLO11s", "YOLO11m", "YOLO11l", "YOLO26l"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def strip_prefix(name: str) -> str:
    for p in SCENE_PREFIXES:
        if name.startswith(p):
            return name[len(p):]
    return name


def load_seg_scores() -> dict[str, dict[str, float]]:
    """Returns {variant: {scene: iou_score}} for scene segmentation tasks."""
    scores: dict[str, dict[str, float]] = {}

    for task_key, csv_path, metric_col in SEG_TASK_CONFIGS:
        if not csv_path.exists():
            print(f"  WARNING: {csv_path} not found — skipping {task_key}")
            continue
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                variant = strip_prefix(row.get("model", ""))
                try:
                    value = float(row[metric_col])
                except (KeyError, ValueError):
                    continue
                scores.setdefault(variant, {})[task_key] = value

    return scores


def load_human_scores() -> dict[str, float]:
    """Returns {variant: f1_score} for human detection."""
    task_key, csv_path, metric_col = HUMAN_TASK_CONFIG
    scores: dict[str, float] = {}

    if not csv_path.exists():
        print(f"  WARNING: {csv_path} not found — skipping human detection")
        return scores

    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            variant = strip_prefix(row.get("model", ""))
            try:
                value = float(row[metric_col])
            except (KeyError, ValueError):
                continue
            scores[variant] = value

    return scores


def load_latency(imgsz: int) -> dict[str, float]:
    """Returns {variant: median_inference_ms} averaged across available scenes."""
    latency_sums:   dict[str, float] = {}
    latency_counts: dict[str, int]   = {}

    for csv_path in LATENCY_SCENES:
        if not csv_path.exists():
            continue
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if int(row["imgsz"]) != imgsz:
                    continue
                variant = row["model"]
                try:
                    ms = float(row["median_inference_ms"])
                except (KeyError, ValueError):
                    continue
                latency_sums[variant]   = latency_sums.get(variant, 0.0) + ms
                latency_counts[variant] = latency_counts.get(variant, 0) + 1

    return {v: latency_sums[v] / latency_counts[v] for v in latency_sums}


# ---------------------------------------------------------------------------
# Aggregate scoring
# ---------------------------------------------------------------------------

def weighted_harmonic_mean(values: list[float], weights: list[float]) -> float:
    """Weighted harmonic mean — penalises weak links more than arithmetic mean."""
    if not values or any(v <= 0 for v in values):
        return 0.0
    w = np.array(weights, dtype=np.float64)
    v = np.array(values,  dtype=np.float64)
    return float(w.sum() / np.sum(w / v))


def compute_seg_aggregate(seg_scores: dict[str, dict[str, float]],
                          seg_keys: list[str],
                          weights: list[float]) -> dict[str, float]:
    """Returns {variant: WHM of scene segmentation IoUs}."""
    agg = {}
    for variant, task_scores in seg_scores.items():
        vals = [task_scores.get(t, 0.0) for t in seg_keys]
        agg[variant] = weighted_harmonic_mean(vals, weights)
    return agg


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_grouped_bars(seg_scores: dict[str, dict[str, float]],
                      seg_agg: dict[str, float],
                      human_scores: dict[str, float],
                      seg_keys: list[str],
                      out_dir: Path) -> None:
    """Two-panel bar chart: scene seg (left), human detection (right)."""
    variants = [v for v in VARIANT_ORDER if v in seg_scores or v in human_scores]
    labels   = [VARIANT_LABELS[VARIANT_ORDER.index(v)] for v in variants]
    n        = len(variants)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6),
                                    gridspec_kw={"width_ratios": [2, 1]})

    # --- Left panel: scene segmentation ---
    n_bars = len(seg_keys) + 1  # +1 for aggregate
    x     = np.arange(n)
    width = 0.8 / n_bars

    colours = plt.cm.Set2(np.linspace(0, 1, len(seg_keys)))
    for i, task in enumerate(seg_keys):
        vals   = [seg_scores.get(v, {}).get(task, 0.0) for v in variants]
        offset = (i - (n_bars - 1) / 2) * width
        bars = ax1.bar(x + offset, vals, width, label=f"{task.capitalize()} IoU",
                       color=colours[i])
        for bar, val in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    # Scene seg aggregate
    agg_vals = [seg_agg.get(v, 0.0) for v in variants]
    offset   = (len(seg_keys) - (n_bars - 1) / 2) * width
    bars = ax1.bar(x + offset, agg_vals, width, label="Seg Aggregate (WHM)",
                   color="black", alpha=0.7)
    for bar, val in zip(bars, agg_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Mean IoU")
    ax1.set_title("Scene Segmentation")
    ax1.set_ylim(0, 0.8)
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # --- Right panel: human detection ---
    human_vals = [human_scores.get(v, 0.0) for v in variants]
    bars = ax2.bar(x, human_vals, 0.5, color="steelblue")
    for bar, val in zip(bars, human_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("F1 Score")
    ax2.set_title("Human Detection")
    ax2.set_ylim(0, 0.8)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Model Comparison: Scene Segmentation vs Human Detection", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "per_task_scores.png", dpi=150)
    plt.close(fig)


def plot_radar(seg_scores: dict[str, dict[str, float]],
               human_scores: dict[str, float],
               seg_keys: list[str],
               out_dir: Path) -> None:
    """Radar chart with seg tasks + human F1 as separate axes."""
    all_keys = seg_keys + ["human F1"]
    variants = [v for v in VARIANT_ORDER if v in seg_scores or v in human_scores]
    labels   = [VARIANT_LABELS[VARIANT_ORDER.index(v)] for v in variants]

    angles = np.linspace(0, 2 * np.pi, len(all_keys), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    colours = plt.cm.tab10(np.linspace(0, 1, len(variants)))
    for vi, variant in enumerate(variants):
        vals = [seg_scores.get(variant, {}).get(t, 0.0) for t in seg_keys]
        vals.append(human_scores.get(variant, 0.0))
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=1.5, label=labels[vi],
                color=colours[vi], markersize=5)
        ax.fill(angles, vals, alpha=0.08, color=colours[vi])

    ax.set_thetagrids(np.degrees(angles[:-1]),
                      [t.capitalize() + " IoU" for t in seg_keys] + ["Human F1"])
    ax.set_ylim(0, 0.8)
    ax.set_title("Model Profile by Task", y=1.08, fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "radar_chart.png", dpi=150)
    plt.close(fig)


def _pareto_frontier(xs, ys):
    """Return indices of Pareto-optimal points (lower x, higher y)."""
    points = sorted(range(len(xs)), key=lambda i: xs[i])
    frontier = []
    best_y = -1.0
    for i in points:
        if ys[i] > best_y:
            frontier.append(i)
            best_y = ys[i]
    return frontier


def plot_dual_pareto(seg_agg: dict[str, float],
                     human_scores: dict[str, float],
                     latency: dict[str, float],
                     imgsz: int,
                     out_dir: Path) -> None:
    """Side-by-side Pareto: scene seg vs latency, human F1 vs latency."""
    variants = [v for v in VARIANT_ORDER if v in latency
                and (v in seg_agg or v in human_scores)]
    labels = [VARIANT_LABELS[VARIANT_ORDER.index(v)] for v in variants]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for ax, score_dict, ylabel, title in [
        (ax1, seg_agg,       "Scene Seg WHM (IoU)",  "Scene Segmentation vs Latency"),
        (ax2, human_scores,  "Human Detection F1",    "Human Detection vs Latency"),
    ]:
        xs = [latency[v] for v in variants]
        ys = [score_dict.get(v, 0.0) for v in variants]

        ax.scatter(xs, ys, s=120, zorder=5, c="steelblue", edgecolors="black")
        for x, y, lbl in zip(xs, ys, labels):
            ax.annotate(lbl, (x, y), xytext=(8, 6), textcoords="offset points",
                        fontsize=9, fontweight="bold")

        frontier = _pareto_frontier(xs, ys)
        if len(frontier) > 1:
            fx = [xs[i] for i in frontier]
            fy = [ys[i] for i in frontier]
            ax.plot(fx, fy, "r--", linewidth=1.5, alpha=0.7, label="Pareto frontier")
        for i in frontier:
            ax.scatter([xs[i]], [ys[i]], s=180, facecolors="none",
                       edgecolors="red", linewidths=2, zorder=6)

        ax.set_xlabel("Median inference latency (ms)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.4)
        if frontier:
            ax.legend(fontsize=9)

    fig.suptitle(f"Accuracy vs Latency  (imgsz={imgsz})", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "pareto_frontier.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-task model selection and visualisation")
    parser.add_argument(
        "--seg-weights", type=float, nargs="+",
        default=None,
        help="Scene segmentation weights in order: bridge, railway, ship. "
             "Default: equal weights (1 1 1).")
    parser.add_argument(
        "--imgsz", type=int, default=1280,
        help="Image size to use for latency comparison (default: 1280).")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    seg_scores   = load_seg_scores()
    human_scores = load_human_scores()
    latency      = load_latency(args.imgsz)

    seg_keys = [t for t, *_ in SEG_TASK_CONFIGS
                if any(t in s for s in seg_scores.values())]

    if not seg_scores and not human_scores:
        print("ERROR: No evaluation results found. Run eval_scene and eval_human first.")
        return

    seg_weights = args.seg_weights or [1.0] * len(seg_keys)
    if len(seg_weights) != len(seg_keys):
        print(f"ERROR: --seg-weights expects {len(seg_keys)} values "
              f"({seg_keys}), got {len(seg_weights)}")
        return

    # Compute scene segmentation aggregate (WHM of IoU across scenes)
    seg_agg = compute_seg_aggregate(seg_scores, seg_keys, seg_weights)

    # Collect all variants
    all_variants = set(seg_scores.keys()) | set(human_scores.keys())
    variants = [v for v in VARIANT_ORDER if v in all_variants]

    # ── Print summary ──
    print(f"\n{'=' * 90}")
    print("  MODEL SELECTION SUMMARY")
    print(f"{'=' * 90}")
    print(f"  Scene seg tasks: {seg_keys}  (weights: {dict(zip(seg_keys, seg_weights))})")
    print(f"  Scene seg aggregate: Weighted Harmonic Mean of IoU")
    print(f"  Human detection: F1 (reported separately)")
    print(f"  Latency: imgsz={args.imgsz}")

    # ── Scene Segmentation Table ──
    print(f"\n  {'─' * 80}")
    print(f"  SCENE SEGMENTATION")
    print(f"  {'─' * 80}")

    seg_header = f"  {'Model':<12}"
    for t in seg_keys:
        seg_header += f" {t.capitalize() + ' IoU':>14}"
    seg_header += f" {'Seg WHM':>10} {'Latency(ms)':>12} {'FPS':>8}"
    print(seg_header)
    print(f"  {'-' * (len(seg_header) - 2)}")

    seg_ranked = sorted(variants, key=lambda v: seg_agg.get(v, 0), reverse=True)
    for v in seg_ranked:
        lbl = VARIANT_LABELS[VARIANT_ORDER.index(v)]
        line = f"  {lbl:<12}"
        for t in seg_keys:
            val = seg_scores.get(v, {}).get(t, 0.0)
            line += f" {val:>14.4f}"
        agg = seg_agg.get(v, 0.0)
        lat = latency.get(v, float("nan"))
        fps = 1000.0 / lat if lat > 0 else 0.0
        line += f" {agg:>10.4f} {lat:>12.1f} {fps:>8.1f}"
        print(line)

    best_seg = seg_ranked[0] if seg_ranked else None

    # ── Human Detection Table ──
    print(f"\n  {'─' * 80}")
    print(f"  HUMAN DETECTION")
    print(f"  {'─' * 80}")

    hum_header = f"  {'Model':<12} {'F1':>8} {'Latency(ms)':>12} {'FPS':>8}"
    print(hum_header)
    print(f"  {'-' * (len(hum_header) - 2)}")

    hum_ranked = sorted(variants, key=lambda v: human_scores.get(v, 0), reverse=True)
    for v in hum_ranked:
        lbl = VARIANT_LABELS[VARIANT_ORDER.index(v)]
        f1  = human_scores.get(v, 0.0)
        lat = latency.get(v, float("nan"))
        fps = 1000.0 / lat if lat > 0 else 0.0
        print(f"  {lbl:<12} {f1:>8.4f} {lat:>12.1f} {fps:>8.1f}")

    best_hum = hum_ranked[0] if hum_ranked else None

    # ── Recommendation ──
    print(f"\n  {'=' * 80}")
    if best_seg:
        lbl = VARIANT_LABELS[VARIANT_ORDER.index(best_seg)]
        print(f"  Best scene segmentation:  {lbl}  (WHM={seg_agg[best_seg]:.4f})")
    if best_hum:
        lbl = VARIANT_LABELS[VARIANT_ORDER.index(best_hum)]
        print(f"  Best human detection:     {lbl}  (F1={human_scores[best_hum]:.4f})")

    if best_seg and best_hum and best_seg == best_hum:
        lbl = VARIANT_LABELS[VARIANT_ORDER.index(best_seg)]
        print(f"\n  >>> CLEAR WINNER: {lbl} — best at both tasks")
    elif best_seg and best_hum:
        print(f"\n  >>> Different models lead each task — review the Pareto plots")
        print(f"      to find the best tradeoff for your latency budget.")

    # ── Save CSV ──
    csv_path = OUTPUT_DIR / "model_selection.csv"
    fields = ["model", "variant"] + [f"{t}_iou" for t in seg_keys] + [
        "seg_whm", "human_f1", "latency_ms", "fps"]
    rows = []
    for v in seg_ranked:
        lbl = VARIANT_LABELS[VARIANT_ORDER.index(v)]
        lat = latency.get(v, float("nan"))
        row = {"model": lbl, "variant": v}
        for t in seg_keys:
            row[f"{t}_iou"] = round(seg_scores.get(v, {}).get(t, 0.0), 4)
        row["seg_whm"]    = round(seg_agg.get(v, 0.0), 4)
        row["human_f1"]   = round(human_scores.get(v, 0.0), 4)
        row["latency_ms"] = round(lat, 1)
        row["fps"]        = round(1000.0 / lat if lat > 0 else 0.0, 1)
        rows.append(row)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  CSV saved → {csv_path}")

    # ── Generate plots ──
    print(f"  Generating plots...")
    plot_grouped_bars(seg_scores, seg_agg, human_scores, seg_keys, OUTPUT_DIR)
    plot_radar(seg_scores, human_scores, seg_keys, OUTPUT_DIR)
    if latency:
        plot_dual_pareto(seg_agg, human_scores, latency, args.imgsz, OUTPUT_DIR)
    print(f"  Plots saved → {OUTPUT_DIR}/")
    print(f"\n  Files:")
    print(f"    {OUTPUT_DIR}/per_task_scores.png  — side-by-side bar charts")
    print(f"    {OUTPUT_DIR}/radar_chart.png      — model profile spider chart")
    if latency:
        print(f"    {OUTPUT_DIR}/pareto_frontier.png  — dual Pareto (seg + human vs latency)")
    print(f"    {OUTPUT_DIR}/model_selection.csv   — full results table")


if __name__ == "__main__":
    main()
