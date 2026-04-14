"""
Evaluate end-to-end alarm triggering on still images.

For each image the pipeline runs:
  1. Determine scene from filename prefix (bridge_, railway_, ship_)
  2. Scene segmentation model  → danger zone mask  (scene-specific model)
  3. Human detection model     → person masks
  4. Overlap check             → alarm triggered if any person mask overlaps
                                 the danger zone above DANGER_ZONE_OVERLAP_THRESHOLD

Ground truth:
  Positive/   — alarm SHOULD trigger  (TP if fires, FN if not)
  Negative/   — alarm should NOT fire (FP if fires, TN if not)

Positive images have a paired cleanup version in Positive/empty_ver/:
  e.g. bridge_positive_01.png  →  bridge_positive_01_cleanup.png
  The cleanup version (humans removed) is used for scene segmentation so the
  danger zone is not occluded by the person. Human detection still runs on the
  original image.

Images must be named with a scene prefix: bridge_*, railway_*, or ship_*.
The correct scene segmentation model is selected automatically per image.

Usage:
    python -m src.eval.eval_alarm_trigger
    python -m src.eval.eval_alarm_trigger --output-csv alarm_results.csv
"""

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# *** SET THESE TO YOUR BEST MODELS BEFORE RUNNING ***
# ---------------------------------------------------------------------------

# One scene segmentation model per scene type
SEG_MODELS = {
    "bridge": Path("runs/segment/runs/segment/bridge_hazard_yolo11s-seg/weights/best.pt"),
    "railway": Path("runs/segment/runs/segment/railway_hazard_yolo11s-seg/weights/best.pt"),
    "ship": Path("runs/segment/runs/segment/ship_hazard_yolo11s-seg/weights/best.pt"),
}

# Single human detection model (shared across all scenes)
HUMAN_MODEL_PATH = Path(
    "runs/segment/runs/segment/human_detection_real_yolo11s-seg/weights/best.pt"
)

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------

OVERLAP_ROOT = Path("data/test_dataset/images/overlap")
POSITIVE_DIR = OVERLAP_ROOT / "Positive"
EMPTY_VER_DIR = OVERLAP_ROOT / "Positive" / "empty_ver"
NEGATIVE_DIR = OVERLAP_ROOT / "Negative"
VIS_OUTPUT_ROOT = Path("eval_output/alarm_trigger")

CONF_THRESHOLD = 0.25
SEG_IMGSZ = 640  # Scene structures are large — 640 is sufficient
HUMAN_IMGSZ = 1280  # Humans can be small/distant — need higher resolution
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

SCENE_PREFIXES = ("bridge", "railway", "ship")


def detect_scene(filename: str) -> str | None:
    """Determine scene type from filename prefix."""
    lower = filename.lower()
    for scene in SCENE_PREFIXES:
        if lower.startswith(f"{scene}_"):
            return scene
    return None


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def get_seg_mask(model: YOLO, image_path: Path, img_w: int, img_h: int) -> np.ndarray:
    """Run scene segmentation → single combined binary danger zone mask."""
    results = model(
        str(image_path), conf=CONF_THRESHOLD, imgsz=SEG_IMGSZ, verbose=False, save=False
    )
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if results[0].masks is None:
        return mask
    for m in results[0].masks.data:
        resized = cv2.resize(m.cpu().numpy(), (img_w, img_h))
        mask = np.maximum(mask, (resized > 0.5).astype(np.uint8))
    return mask


def get_human_masks(model: YOLO, image_path: Path, img_w: int, img_h: int) -> list[np.ndarray]:
    """Run human detection → list of individual person binary masks."""
    results = model(
        str(image_path), conf=CONF_THRESHOLD, imgsz=HUMAN_IMGSZ, verbose=False, save=False
    )
    masks = []
    if results[0].masks is None:
        return masks
    for m in results[0].masks.data:
        resized = cv2.resize(m.cpu().numpy(), (img_w, img_h))
        binary = (resized > 0.5).astype(np.uint8)
        if binary.sum() > 0:
            masks.append(binary)
    return masks


def compute_max_overlap(danger_zone: np.ndarray, human_masks: list[np.ndarray]) -> float:
    """Return the maximum overlap fraction of any person inside the danger zone.

    This is the raw value that gets compared against a threshold to decide
    whether the alarm fires.  Storing it per-image lets us sweep thresholds
    without re-running inference.
    """
    max_overlap = 0.0
    for person in human_masks:
        person_area = int(person.sum())
        if person_area == 0:
            continue
        overlap = float(np.logical_and(person, danger_zone).sum()) / person_area
        max_overlap = max(max_overlap, overlap)
    return max_overlap


def alarm_fires_at(max_overlap: float, threshold: float) -> bool:
    return max_overlap >= threshold


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def make_visualisation(
    orig_img: np.ndarray,
    danger_zone: np.ndarray,
    human_masks: list[np.ndarray],
    fired: bool,
    gt_label: str,
    max_overlap: float,
) -> np.ndarray:
    """
    Single panel showing:
      - Red overlay   = danger zone
      - Green overlay = detected humans
      - Header: GT label, alarm fired/not, correct/wrong, overlap %
    """
    out = orig_img.copy().astype(np.float32)

    # Danger zone — red tint
    tint = np.zeros_like(orig_img, dtype=np.float32)
    tint[danger_zone == 1] = (0, 0, 220)
    out[danger_zone == 1] = out[danger_zone == 1] * 0.5 + tint[danger_zone == 1] * 0.5
    out = out.astype(np.uint8)
    contours, _ = cv2.findContours(danger_zone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 0, 220), 2)

    # Human masks — green tint + per-person overlap label
    for person in human_masks:
        person_area = int(person.sum())
        if person_area > 0:
            overlap = float(np.logical_and(person, danger_zone).sum()) / person_area
        else:
            overlap = 0.0

        tint = np.zeros_like(orig_img, dtype=np.float32)
        tint[person == 1] = (0, 200, 0)
        out_f = out.astype(np.float32)
        out_f[person == 1] = out_f[person == 1] * 0.5 + tint[person == 1] * 0.5
        out = out_f.astype(np.uint8)
        contours, _ = cv2.findContours(person, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 200, 0), 2)

        # Label each person with their overlap %
        if contours:
            x, y, _, _ = cv2.boundingRect(contours[0])
            cv2.putText(
                out,
                f"{overlap:.0%}",
                (x, max(y - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    # Header
    h, w = orig_img.shape[:2]
    label_h = 36
    panel = np.zeros((h + label_h, w, 3), dtype=np.uint8)
    panel[label_h:] = out

    alarm_str = "ALARM" if fired else "no alarm"
    correct = (fired and gt_label == "positive") or (not fired and gt_label == "negative")
    result_str = "CORRECT" if correct else "WRONG"
    colour = (0, 220, 0) if correct else (0, 0, 220)

    text = (
        f"GT={gt_label}  pred={alarm_str}  [{result_str}]  "
        f"overlap={max_overlap:.1%}  "
        f"zone={'yes' if danger_zone.sum() > 0 else 'no'}  "
        f"humans={len(human_masks)}"
    )
    cv2.putText(panel, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)

    return panel


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def find_cleanup(img_path: Path) -> Path | None:
    """Find the empty_ver counterpart for a positive image."""
    stem = img_path.stem
    for ext in IMAGE_EXTENSIONS:
        candidate = EMPTY_VER_DIR / f"{stem}_cleanup{ext}"
        if candidate.exists():
            return candidate
    return None


def evaluate(seg_models: dict[str, YOLO], human_model: YOLO) -> dict:
    records = []
    skipped = []

    for gt_label, folder in [("positive", POSITIVE_DIR), ("negative", NEGATIVE_DIR)]:
        image_paths = sorted(
            [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()]
        )

        for img_path in image_paths:
            scene = detect_scene(img_path.name)
            if scene is None:
                skipped.append(img_path.name)
                continue
            if scene not in seg_models:
                skipped.append(img_path.name)
                continue

            seg_model = seg_models[scene]

            orig = cv2.imread(str(img_path))
            if orig is None:
                continue
            h, w = orig.shape[:2]

            # For positives: use cleanup for seg, original for human detection
            if gt_label == "positive":
                cleanup_path = find_cleanup(img_path)
                if cleanup_path:
                    cleanup_img = cv2.imread(str(cleanup_path))
                    ch, cw = cleanup_img.shape[:2] if cleanup_img is not None else (h, w)
                    seg_mask = get_seg_mask(seg_model, cleanup_path, cw, ch)
                    # Resize mask to original dimensions if cleanup size differs
                    if (cw, ch) != (w, h):
                        seg_mask = cv2.resize(seg_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    danger_zone = seg_mask
                else:
                    danger_zone = get_seg_mask(seg_model, img_path, w, h)
            else:
                danger_zone = get_seg_mask(seg_model, img_path, w, h)

            human_masks = get_human_masks(human_model, img_path, w, h)
            max_overlap = compute_max_overlap(danger_zone, human_masks)

            records.append(
                {
                    "img_path": img_path,
                    "gt_label": gt_label,
                    "scene": scene,
                    "max_overlap": max_overlap,
                    "orig": orig,
                    "danger": danger_zone,
                    "humans": human_masks,
                }
            )

    if skipped:
        print(f"\n  WARNING: {len(skipped)} images skipped (no scene prefix detected):")
        for name in skipped[:10]:
            print(f"    {name}")
        if len(skipped) > 10:
            print(f"    ... and {len(skipped) - 10} more")

    return {"records": records}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def score_at_threshold(records: list[dict], threshold: float) -> dict:
    """Compute TP/FP/TN/FN and derived metrics at a given overlap threshold."""
    tp = fp = tn = fn = 0
    for rec in records:
        fired = alarm_fires_at(rec["max_overlap"], threshold)
        if rec["gt_label"] == "positive":
            if fired:
                tp += 1
            else:
                fn += 1
        else:
            if fired:
                fp += 1
            else:
                tn += 1

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "fpr": fpr,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_threshold_curves(
    records: list[dict],
    thresholds: list[float],
    scenes: list[str],
    scene_best: dict[str, float],
    overall_best: float,
    out_dir: Path,
) -> None:
    """Plot F1 / Precision / Recall vs threshold, one subplot per scene + overall."""
    groups = [(s, [r for r in records if r["scene"] == s]) for s in scenes]
    groups.append(("all", records))

    n = len(groups)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)

    for idx, (label, recs) in enumerate(groups):
        ax = axes[idx // cols][idx % cols]
        precs, recalls, f1s = [], [], []
        for t in thresholds:
            s = score_at_threshold(recs, t)
            precs.append(s["precision"])
            recalls.append(s["recall"])
            f1s.append(s["f1"])

        ax.plot(thresholds, f1s, "o-", label="F1", color="tab:blue", linewidth=2)
        ax.plot(thresholds, precs, "s--", label="Precision", color="tab:green", linewidth=1.5)
        ax.plot(thresholds, recalls, "^--", label="Recall", color="tab:orange", linewidth=1.5)

        # Mark best threshold
        best_t = scene_best.get(label, overall_best)
        best_s = score_at_threshold(recs, best_t)
        ax.axvline(best_t, color="red", linestyle=":", alpha=0.7)
        ax.plot(best_t, best_s["f1"], "*", color="red", markersize=14, zorder=5)
        ax.annotate(
            f"best={best_t:.2f}\nF1={best_s['f1']:.3f}",
            (best_t, best_s["f1"]),
            xytext=(8, -20),
            textcoords="offset points",
            fontsize=8,
            color="red",
        )

        n_pos = sum(1 for r in recs if r["gt_label"] == "positive")
        n_neg = sum(1 for r in recs if r["gt_label"] == "negative")
        ax.set_title(f"{label.capitalize()}  ({n_pos}pos / {n_neg}neg)", fontsize=12)
        ax.set_xlabel("Overlap threshold")
        ax.set_ylabel("Score")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    # Hide unused subplots
    for idx in range(len(groups), rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle("Alarm Trigger: Threshold vs Metrics", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "threshold_curves.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrices(
    records: list[dict],
    scenes: list[str],
    scene_best: dict[str, float],
    overall_best: float,
    out_dir: Path,
) -> None:
    """Plot confusion matrix heatmaps, one per scene + overall."""
    groups = [(s, [r for r in records if r["scene"] == s]) for s in scenes]
    groups.append(("all", records))

    n = len(groups)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows), squeeze=False)

    for idx, (label, recs) in enumerate(groups):
        ax = axes[idx // cols][idx % cols]
        thresh = scene_best.get(label, overall_best)
        s = score_at_threshold(recs, thresh)

        matrix = np.array([[s["tp"], s["fn"]], [s["fp"], s["tn"]]])
        # Normalise per row for colour intensity
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        norm = matrix / row_sums

        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="equal")

        # Annotate cells with count and percentage
        for i in range(2):
            for j in range(2):
                count = matrix[i, j]
                pct = norm[i, j]
                ax.text(
                    j,
                    i,
                    f"{count}\n({pct:.0%})",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="white" if pct > 0.5 else "black",
                    fontweight="bold",
                )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Alarm", "No Alarm"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Positive", "Negative"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{label.capitalize()}  (t={thresh:.2f}, F1={s['f1']:.3f})", fontsize=11)

    for idx in range(len(groups), rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle("Confusion Matrices (per-scene optimal thresholds)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def print_sweep_table(
    label: str, records: list[dict], thresholds: list[float]
) -> tuple[float, float]:
    """Print a threshold sweep table for a set of records. Returns (best_thresh, best_f1)."""
    n_pos = sum(1 for r in records if r["gt_label"] == "positive")
    n_neg = sum(1 for r in records if r["gt_label"] == "negative")
    print(f"\n  {label}  ({len(records)} images: {n_pos} pos, {n_neg} neg)")

    header = (
        f"  {'Thresh':>7} │ {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4} │ "
        f"{'Prec':>6} {'Recall':>6} {'F1':>6} {'Acc':>6} {'FPR':>6}"
    )
    print(header)
    print(f"  {'─' * (len(header) - 2)}")

    best_f1 = -1.0
    best_thresh = thresholds[0]
    for thresh in thresholds:
        s = score_at_threshold(records, thresh)
        marker = ""
        if s["f1"] > best_f1:
            best_f1 = s["f1"]
            best_thresh = thresh
        line = (
            f"  {s['threshold']:>7.2f} │ "
            f"{s['tp']:>4} {s['fp']:>4} {s['tn']:>4} {s['fn']:>4} │ "
            f"{s['precision']:>6.3f} {s['recall']:>6.3f} {s['f1']:>6.3f} "
            f"{s['accuracy']:>6.3f} {s['fpr']:>6.3f}"
        )
        print(line)

    # Mark the best row
    print(f"  >>> Best F1: {best_f1:.4f} at threshold={best_thresh:.2f}")
    return best_thresh, best_f1


def main():
    parser = argparse.ArgumentParser(description="Evaluate end-to-end alarm trigger")
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
        help="Overlap thresholds to evaluate",
    )
    args = parser.parse_args()

    print(f"Scene seg models:")
    for scene, path in SEG_MODELS.items():
        print(f"  {scene:>8}: {path}")
    print(f"Human det model : {HUMAN_MODEL_PATH}")
    print(f"Seg imgsz: {SEG_IMGSZ}  |  Human imgsz: {HUMAN_IMGSZ}")
    print(f"Thresholds to test: {args.thresholds}")

    # Validate all model paths exist
    for scene, path in SEG_MODELS.items():
        if not path.exists():
            print(f"ERROR: {scene} seg model not found at {path}")
            return
    if not HUMAN_MODEL_PATH.exists():
        print(f"ERROR: human model not found at {HUMAN_MODEL_PATH}")
        return

    # Load all scene segmentation models
    seg_models = {scene: YOLO(str(path)) for scene, path in SEG_MODELS.items()}
    human_model = YOLO(str(HUMAN_MODEL_PATH))

    print("\nRunning inference (once, then sweeping thresholds)...")
    result = evaluate(seg_models, human_model)
    records = result["records"]

    total_pos = sum(1 for r in records if r["gt_label"] == "positive")
    total_neg = sum(1 for r in records if r["gt_label"] == "negative")
    total = len(records)
    print(f"  {total} images evaluated ({total_pos} positive, {total_neg} negative)")

    thresholds = sorted(args.thresholds)

    # ── Per-scene threshold sweep ──
    scenes = sorted(set(r["scene"] for r in records))
    scene_best_thresh: dict[str, float] = {}

    print(f"\n{'=' * 90}")
    print(f"  PER-SCENE THRESHOLD SWEEP")
    print(f"{'=' * 90}")

    for scene in scenes:
        scene_records = [r for r in records if r["scene"] == scene]
        best_thresh, best_f1 = print_sweep_table(
            f"SCENE: {scene.upper()}", scene_records, thresholds
        )
        scene_best_thresh[scene] = best_thresh

    # ── Overall threshold sweep (single threshold for all) ──
    print(f"\n{'=' * 90}")
    print(f"  OVERALL THRESHOLD SWEEP (single threshold)")
    print(f"{'=' * 90}")
    overall_best_thresh, overall_best_f1 = print_sweep_table("ALL SCENES", records, thresholds)

    # ── Per-scene optimal: use each scene's best threshold ──
    print(f"\n{'=' * 90}")
    print(f"  PER-SCENE OPTIMAL THRESHOLDS")
    print(f"{'=' * 90}")
    print(f"\n  Best threshold per scene:")
    for scene, thresh in scene_best_thresh.items():
        print(f"    {scene:>8}: {thresh:.2f}")

    # Compute combined metrics using per-scene thresholds
    tp = fp = tn = fn = 0
    for rec in records:
        thresh = scene_best_thresh[rec["scene"]]
        fired = alarm_fires_at(rec["max_overlap"], thresh)
        if rec["gt_label"] == "positive":
            if fired:
                tp += 1
            else:
                fn += 1
        else:
            if fired:
                fp += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print(f"\n  Combined results with per-scene thresholds:")
    print(f"    TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(
        f"    Precision={precision:.3f}  Recall={recall:.3f}  "
        f"F1={f1:.3f}  Accuracy={accuracy:.3f}  FPR={fpr:.3f}"
    )

    # ── Compare: single vs per-scene ──
    print(f"\n  {'─' * 60}")
    print(f"  COMPARISON")
    print(f"  {'─' * 60}")
    print(f"    Single threshold (={overall_best_thresh:.2f}):    F1={overall_best_f1:.4f}")
    print(f"    Per-scene thresholds:              F1={f1:.4f}")
    if f1 > overall_best_f1:
        print(f"    >>> Per-scene is better by {f1 - overall_best_f1:.4f}")
    elif f1 < overall_best_f1:
        print(f"    >>> Single threshold is better by {overall_best_f1 - f1:.4f}")
    else:
        print(f"    >>> Both are equal")

    # Use per-scene thresholds for vis and CSVs
    use_per_scene = f1 >= overall_best_f1

    # ── Save visualisations ──
    out_dir = VIS_OUTPUT_ROOT
    if out_dir.exists():
        for old_file in out_dir.glob("*.jpg"):
            old_file.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    for rank, rec in enumerate(records, 1):
        if use_per_scene:
            thresh = scene_best_thresh[rec["scene"]]
        else:
            thresh = overall_best_thresh
        fired = alarm_fires_at(rec["max_overlap"], thresh)
        vis = make_visualisation(
            rec["orig"], rec["danger"], rec["humans"], fired, rec["gt_label"], rec["max_overlap"]
        )
        correct = (fired and rec["gt_label"] == "positive") or (
            not fired and rec["gt_label"] == "negative"
        )
        label = "correct" if correct else "WRONG"
        out_path = out_dir / f"{rank:03d}_{rec['scene']}_{rec['img_path'].stem}_{label}.jpg"
        cv2.imwrite(str(out_path), vis)

    thresh_desc = (
        f"per-scene: {scene_best_thresh}" if use_per_scene else f"single: {overall_best_thresh:.2f}"
    )
    print(f"\n  Visualisations saved → {out_dir}  ({thresh_desc})")

    # ── CSV: per-image results ──
    csv_path = out_dir / "results.csv"
    fields = ["image", "scene", "gt_label", "max_overlap", "alarm_fired", "correct", "threshold"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            if use_per_scene:
                thresh = scene_best_thresh[rec["scene"]]
            else:
                thresh = overall_best_thresh
            fired = alarm_fires_at(rec["max_overlap"], thresh)
            correct = (fired and rec["gt_label"] == "positive") or (
                not fired and rec["gt_label"] == "negative"
            )
            writer.writerow(
                {
                    "image": rec["img_path"].name,
                    "scene": rec["scene"],
                    "gt_label": rec["gt_label"],
                    "max_overlap": round(rec["max_overlap"], 4),
                    "alarm_fired": fired,
                    "correct": correct,
                    "threshold": thresh,
                }
            )

    # ── CSV: per-scene threshold sweep ──
    sweep_path = out_dir / "threshold_sweep.csv"
    sweep_fields = [
        "scene",
        "threshold",
        "tp",
        "fp",
        "tn",
        "fn",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "fpr",
    ]
    with open(sweep_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sweep_fields)
        writer.writeheader()
        for scene in scenes + ["all"]:
            recs = records if scene == "all" else [r for r in records if r["scene"] == scene]
            for thresh in thresholds:
                s = score_at_threshold(recs, thresh)
                row = {"scene": scene}
                row.update({k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()})
                writer.writerow(row)

    # ── CSV: summary ──
    summary_path = out_dir / "summary.csv"
    seg_model_str = ", ".join(f"{s}={p.parent.parent.name}" for s, p in SEG_MODELS.items())
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seg_models",
                "human_model",
                "mode",
                "thresholds",
                "tp",
                "fp",
                "tn",
                "fn",
                "precision",
                "recall",
                "f1",
                "accuracy",
            ],
        )
        writer.writeheader()
        # Per-scene row
        writer.writerow(
            {
                "seg_models": seg_model_str,
                "human_model": HUMAN_MODEL_PATH.parent.parent.name,
                "mode": "per_scene",
                "thresholds": str(scene_best_thresh),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "accuracy": round(accuracy, 4),
            }
        )
        # Single threshold row
        overall = score_at_threshold(records, overall_best_thresh)
        writer.writerow(
            {
                "seg_models": seg_model_str,
                "human_model": HUMAN_MODEL_PATH.parent.parent.name,
                "mode": "single",
                "thresholds": str(overall_best_thresh),
                "tp": overall["tp"],
                "fp": overall["fp"],
                "tn": overall["tn"],
                "fn": overall["fn"],
                "precision": round(overall["precision"], 4),
                "recall": round(overall["recall"], 4),
                "f1": round(overall["f1"], 4),
                "accuracy": round(overall["accuracy"], 4),
            }
        )

    print(f"\n  Per-image CSV     → {csv_path}")
    print(f"  Threshold sweep   → {sweep_path}")
    print(f"  Summary CSV       → {summary_path}")

    # ── Plots ──
    print(f"  Generating plots...")
    plot_threshold_curves(
        records, thresholds, scenes, scene_best_thresh, overall_best_thresh, out_dir
    )
    plot_confusion_matrices(records, scenes, scene_best_thresh, overall_best_thresh, out_dir)
    print(f"  threshold_curves.png    → {out_dir}/threshold_curves.png")
    print(f"  confusion_matrices.png  → {out_dir}/confusion_matrices.png")

    if args.output_csv:
        import shutil

        shutil.copy(summary_path, args.output_csv)
        print(f"  Summary also saved → {args.output_csv}")


if __name__ == "__main__":
    main()
