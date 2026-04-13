"""
Evaluate human detection segmentation models using instance-level F1 score.

Each GT instance and each predicted instance is matched individually via IoU.
Matching logic per image:
  - True Positive  (TP): GT instance matched to a prediction above IoU threshold
  - False Negative (FN): GT instance with no prediction above IoU threshold
  - False Positive (FP): predicted instance with no GT above IoU threshold

TP/FP/FN are accumulated across all images, then:
  Precision = TP / (TP + FP)
  Recall    = TP / (TP + FN)
  F1        = 2 * Precision * Recall / (Precision + Recall)

Visualisations saved to:
    eval_output/human/{model_name}/all/
Each image shows ground truth (green) and predicted (blue) instance outlines,
with TP/FP/FN counts in the header.

Test dataset structure expected:
    data/test_dataset/images/human/train/images/   (mixed scenes, all in one folder)
    data/test_dataset/images/human/train/labels/

Usage:
    python -m src.eval.eval_human
    python -m src.eval.eval_human --overlap-threshold 0.5
    python -m src.eval.eval_human --output-csv results_human.csv
"""

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

from src.human_detection.config import DANGER_ZONE_OVERLAP_THRESHOLD

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TEST_DATASET_ROOT = Path("data/test_dataset/images")
MODELS_ROOT       = Path("runs/segment/runs/segment")
VIS_OUTPUT_ROOT   = Path("eval_output")

CONF_THRESHOLD   = 0.25
INFERENCE_IMGSZ  = 1280
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------------------
# Instance mask helpers
# ---------------------------------------------------------------------------

def yolo_polygons_to_instances(label_path: Path, img_w: int, img_h: int) -> list[np.ndarray]:
    """Parse YOLO polygon label file → list of individual binary masks (one per instance)."""
    instances = []
    if not label_path.exists():
        return instances
    text = label_path.read_text().strip()
    if not text:
        return instances
    for line in text.splitlines():
        parts = line.split()
        coords = list(map(float, parts[1:]))   # skip class id
        if len(coords) < 6:
            continue
        xs = [round(coords[i]     * img_w) for i in range(0, len(coords), 2)]
        ys = [round(coords[i + 1] * img_h) for i in range(0, len(coords), 2)]
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(list(zip(xs, ys)), dtype=np.int32)], color=1)
        if mask.sum() > 0:
            instances.append(mask)
    return instances


def get_predicted_instances(model: YOLO, image_path: Path,
                             img_w: int, img_h: int) -> list[np.ndarray]:
    """Run model → list of individual binary masks (one per detected instance)."""
    results = model(str(image_path), conf=CONF_THRESHOLD,
                    imgsz=INFERENCE_IMGSZ, verbose=False, save=False)
    instances = []
    if results[0].masks is None:
        return instances
    for m in results[0].masks.data:
        resized = cv2.resize(m.cpu().numpy(), (img_w, img_h))
        binary = (resized > 0.5).astype(np.uint8)
        if binary.sum() > 0:
            instances.append(binary)
    return instances


def gt_coverage(gt: np.ndarray, pred: np.ndarray) -> float:
    """Fraction of GT mask covered by prediction — mirrors DANGER_ZONE_OVERLAP_THRESHOLD logic."""
    gt_area = int(gt.sum())
    if gt_area == 0:
        return 0.0
    return float(np.logical_and(gt, pred).sum()) / gt_area


# ---------------------------------------------------------------------------
# Instance matching
# ---------------------------------------------------------------------------

def match_instances(gt_instances: list[np.ndarray],
                    pred_instances: list[np.ndarray],
                    overlap_threshold: float) -> tuple[int, int, int]:
    """
    Optimal (Hungarian) matching of GT to predictions using GT coverage as the
    score — identical logic to DANGER_ZONE_OVERLAP_THRESHOLD in config.py.

    A GT is a TP if the best-matched prediction covers >= overlap_threshold of it.
    Unmatched GT  → FN
    Unmatched pred → FP

    Returns (TP, FP, FN).
    """
    if not gt_instances and not pred_instances:
        return 0, 0, 0

    if not gt_instances:
        return 0, len(pred_instances), 0

    if not pred_instances:
        return 0, 0, len(gt_instances)

    # Coverage matrix: rows = GT, cols = predictions
    cov_matrix = np.zeros((len(gt_instances), len(pred_instances)), dtype=np.float32)
    for i, gt in enumerate(gt_instances):
        for j, pred in enumerate(pred_instances):
            cov_matrix[i, j] = gt_coverage(gt, pred)

    # Hungarian algorithm on negated matrix (minimises cost = maximises coverage)
    row_ind, col_ind = linear_sum_assignment(-cov_matrix)

    matched_gt   = set()
    matched_pred = set()
    for i, j in zip(row_ind, col_ind):
        if cov_matrix[i, j] >= overlap_threshold:
            matched_gt.add(i)
            matched_pred.add(j)

    tp = len(matched_gt)
    fn = len(gt_instances)   - tp
    fp = len(pred_instances) - len(matched_pred)
    return tp, fp, fn


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def make_visualisation(img: np.ndarray,
                       gt_instances: list[np.ndarray],
                       pred_instances: list[np.ndarray],
                       tp: int, fp: int, fn: int) -> np.ndarray:
    """
    Side-by-side:
      Left:  original + green GT instance outlines
      Right: original + blue predicted instance outlines
    Header shows TP / FP / FN counts.
    """
    def draw_instances(base, instances, colour):
        out = base.copy()
        for mask in instances:
            tint = np.zeros_like(base, dtype=np.float32)
            tint[mask == 1] = colour
            out = out.astype(np.float32)
            out[mask == 1] = out[mask == 1] * 0.5 + tint[mask == 1] * 0.5
            out = out.astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, colour, 2)
        return out

    left  = draw_instances(img, gt_instances,   (0, 200, 0))   # green = GT
    right = draw_instances(img, pred_instances, (220, 0, 0))   # blue  = predicted

    h, w = img.shape[:2]
    label_h = 30

    def add_label(panel, text):
        labelled = np.zeros((h + label_h, w, 3), dtype=np.uint8)
        labelled[label_h:] = panel
        cv2.putText(labelled, text, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return labelled

    n_gt   = len(gt_instances)
    n_pred = len(pred_instances)
    left  = add_label(left,  f"GT (green)  n={n_gt}")
    right = add_label(right, f"Predicted (blue)  n={n_pred}  TP={tp} FP={fp} FN={fn}")

    return np.hstack([left, right])


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model_path: Path, overlap_threshold: float) -> dict:
    images_dir = TEST_DATASET_ROOT / "human" / "train" / "images"
    labels_dir = TEST_DATASET_ROOT / "human" / "train" / "labels"

    if not images_dir.exists():
        return {"error": f"No test images at {images_dir}"}

    image_paths = [p for p in sorted(images_dir.iterdir())
                   if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not image_paths:
        return {"error": "No images found"}

    model      = YOLO(str(model_path))
    model_name = model_path.parent.parent.name

    out_dir = VIS_OUTPUT_ROOT / "human" / model_name / "all"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_tp = total_fp = total_fn = 0
    n_images = 0

    for rank, img_path in enumerate(image_paths, 1):
        label_path = labels_dir / f"{img_path.stem}.txt"
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        gt_instances   = yolo_polygons_to_instances(label_path, w, h)
        pred_instances = get_predicted_instances(model, img_path, w, h)

        tp, fp, fn = match_instances(gt_instances, pred_instances, overlap_threshold)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        n_images += 1

        vis = make_visualisation(img, gt_instances, pred_instances, tp, fp, fn)
        out_path = out_dir / f"{rank:03d}_{img_path.stem}_tp{tp}_fp{fp}_fn{fn}.jpg"
        cv2.imwrite(str(out_path), vis)

    if n_images == 0:
        return {"error": "No valid images processed"}

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    print(f"    Saved {n_images} visualisations → {out_dir}")

    return {
        "n_images":  n_images,
        "tp":        total_tp,
        "fp":        total_fp,
        "fn":        total_fn,
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
    }


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

def discover_human_models() -> list[Path]:
    return sorted(MODELS_ROOT.glob("human_detection_*/weights/best.pt"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate human detection F1 on test images")
    parser.add_argument("--overlap-threshold", type=float, default=DANGER_ZONE_OVERLAP_THRESHOLD,
                        help=f"GT coverage threshold for TP matching (default: {DANGER_ZONE_OVERLAP_THRESHOLD})")
    parser.add_argument("--output-csv", type=str, default=None,
                        help="Optional path to save combined CSV of all results")
    args = parser.parse_args()

    human_models = discover_human_models()
    if not human_models:
        print(f"No human detection models found under {MODELS_ROOT}")
        return

    print(f"\n{'='*65}")
    print(f"  HUMAN DETECTION  |  {len(human_models)} model(s)  "
          f"|  overlap threshold: {args.overlap_threshold}")
    print(f"{'='*65}")

    all_results = []
    for model_path in human_models:
        model_name = model_path.parent.parent.name
        print(f"\n  [{model_name}]")

        result = evaluate_model(model_path, args.overlap_threshold)

        if "error" in result:
            print(f"  ERROR — {result['error']}")
            continue

        print(f"  F1={result['f1']}  precision={result['precision']}  "
              f"recall={result['recall']}  "
              f"TP={result['tp']} FP={result['fp']} FN={result['fn']}  "
              f"(n={result['n_images']})")
        all_results.append({"model": model_name, **result})

    if all_results:
        results_csv = VIS_OUTPUT_ROOT / "human" / "results.csv"
        results_csv.parent.mkdir(parents=True, exist_ok=True)
        fields = ["model", "f1", "precision", "recall",
                  "tp", "fp", "fn", "n_images"]
        with open(results_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(sorted(all_results, key=lambda x: -x["f1"]))
        print(f"\n  Results saved → {results_csv}")

        print(f"\n\n{'='*65}")
        print("  SUMMARY")
        print(f"{'='*65}")
        print(f"{'Model':<50} {'F1':>6} {'Prec':>7} {'Rec':>7} {'TP':>5} {'FP':>5} {'FN':>5}")
        print("-" * 65)
        for r in sorted(all_results, key=lambda x: -x["f1"]):
            print(f"{r['model']:<50} {r['f1']:>6} {r['precision']:>7} {r['recall']:>7} "
                  f"{r['tp']:>5} {r['fp']:>5} {r['fn']:>5}")

        if args.output_csv:
            with open(args.output_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(all_results)
            print(f"\nSaved to {args.output_csv}")

    print(f"\nVisualisations saved under: {VIS_OUTPUT_ROOT}/human/")


if __name__ == "__main__":
    main()
