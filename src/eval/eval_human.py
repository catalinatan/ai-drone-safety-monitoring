"""
Evaluate human detection segmentation models by comparing predicted masks
against ground-truth masks from the test dataset.

IoU (Intersection over Union) is computed per image then averaged.
GT labels are YOLO polygon format (.txt files alongside images).

Visualisations of the best and worst performing images are saved to:
    eval_output/{scene}/{model_name}/best/
    eval_output/{scene}/{model_name}/worst/

Each visualisation shows the original image with:
  - Green overlay = ground truth mask (all persons combined)
  - Blue overlay  = predicted mask  (all persons combined)

Test dataset structure expected:
    data/test_dataset/images/human_{scene}/train/images/
    data/test_dataset/images/human_{scene}/train/labels/

Usage:
    python -m src.eval.eval_human
    python -m src.eval.eval_human --scene human_bridge
    python -m src.eval.eval_human --output-csv results_human.csv
    python -m src.eval.eval_human --n-examples 5
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TEST_DATASET_ROOT = Path("data/test_dataset/images")
MODELS_ROOT       = Path("runs/segment/runs/segment")
VIS_OUTPUT_ROOT   = Path("eval_output")

CONF_THRESHOLD  = 0.25
INFERENCE_IMGSZ = 1280
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def yolo_polygons_to_mask(label_path: Path, img_w: int, img_h: int) -> np.ndarray:
    """Merge all polygon annotations (any class) into a single binary mask."""
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if not label_path.exists():
        return mask
    text = label_path.read_text().strip()
    if not text:
        return mask
    for line in text.splitlines():
        parts = line.split()
        coords = list(map(float, parts[1:]))   # skip class id
        if len(coords) < 6:
            continue
        xs = [round(coords[i]     * img_w) for i in range(0, len(coords), 2)]
        ys = [round(coords[i + 1] * img_h) for i in range(0, len(coords), 2)]
        cv2.fillPoly(mask, [np.array(list(zip(xs, ys)), dtype=np.int32)], color=1)
    return mask


def get_predicted_mask(model: YOLO, image_path: Path, img_w: int, img_h: int) -> np.ndarray:
    """Run model and combine all predicted instance masks into one binary mask."""
    results = model(str(image_path), conf=CONF_THRESHOLD,
                    imgsz=INFERENCE_IMGSZ, verbose=False, save=False)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if results[0].masks is None:
        return mask
    for m in results[0].masks.data:
        resized = cv2.resize(m.cpu().numpy(), (img_w, img_h))
        mask = np.maximum(mask, (resized > 0.5).astype(np.uint8))
    return mask


def compute_iou(gt: np.ndarray, pred: np.ndarray) -> float:
    intersection = np.logical_and(gt, pred).sum()
    union        = np.logical_or(gt, pred).sum()
    if union == 0:
        return 1.0   # both GT and prediction are empty — correct non-detection
    return float(intersection / union)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def make_visualisation(img: np.ndarray, gt_mask: np.ndarray,
                        pred_mask: np.ndarray, iou: float) -> np.ndarray:
    """
    Returns a side-by-side image:
      Left:  original + green GT mask overlay
      Right: original + blue predicted mask overlay
    """
    def overlay(base, mask, colour):
        out = base.copy().astype(np.float32)
        tint = np.zeros_like(base, dtype=np.float32)
        tint[mask == 1] = colour
        out[mask == 1] = out[mask == 1] * 0.5 + tint[mask == 1] * 0.5
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = out.astype(np.uint8)
        cv2.drawContours(result, contours, -1, colour, 2)
        return result

    left  = overlay(img, gt_mask,   (0, 200, 0))    # green = GT
    right = overlay(img, pred_mask, (220, 0, 0))    # blue  = predicted

    h, w = img.shape[:2]
    label_h = 30

    def add_label(panel, text):
        labelled = np.zeros((h + label_h, w, 3), dtype=np.uint8)
        labelled[label_h:] = panel
        cv2.putText(labelled, text, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return labelled

    left  = add_label(left,  "Ground Truth (green)")
    right = add_label(right, f"Predicted (blue)  IoU={iou:.4f}")

    return np.hstack([left, right])


def save_all_visualisations(records: list[dict], model: YOLO, model_name: str,
                            labels_dir: Path, scene: str) -> None:
    """Save visualisations for every image, sorted by IoU ascending."""
    sorted_records = sorted(records, key=lambda r: r["iou"])

    out_dir = VIS_OUTPUT_ROOT / scene / model_name / "all"
    out_dir.mkdir(parents=True, exist_ok=True)

    for rank, rec in enumerate(sorted_records, 1):
        img_path   = rec["img_path"]
        iou        = rec["iou"]
        label_path = labels_dir / f"{img_path.stem}.txt"

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        gt   = yolo_polygons_to_mask(label_path, w, h)
        pred = get_predicted_mask(model, img_path, w, h)

        vis = make_visualisation(img, gt, pred, iou)
        out_path = out_dir / f"{rank:03d}_{img_path.stem}_iou{iou:.4f}.jpg"
        cv2.imwrite(str(out_path), vis)

    print(f"    Saved {len(sorted_records)} visualisations → {out_dir}")


# ---------------------------------------------------------------------------
# Per-scene evaluation
# ---------------------------------------------------------------------------

def evaluate_scene(scene: str, model_path: Path) -> dict:
    images_dir = TEST_DATASET_ROOT / scene / "train" / "images"
    labels_dir = TEST_DATASET_ROOT / scene / "train" / "labels"

    if not images_dir.exists():
        return {"error": f"No test images at {images_dir}"}

    image_paths = [p for p in sorted(images_dir.iterdir())
                   if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not image_paths:
        return {"error": "No images found"}

    model      = YOLO(str(model_path))
    model_name = model_path.parent.parent.name
    records    = []

    for img_path in image_paths:
        label_path = labels_dir / f"{img_path.stem}.txt"
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt   = yolo_polygons_to_mask(label_path, w, h)
        pred = get_predicted_mask(model, img_path, w, h)
        trivial = int(gt.sum()) == 0 and int(pred.sum()) == 0
        records.append({"img_path": img_path, "iou": compute_iou(gt, pred), "trivial": trivial})

    if not records:
        return {"error": "No valid images processed"}

    ious = [r["iou"] for r in records]

    print(f"  Saving visualisations...")
    save_all_visualisations(records, model, model_name, labels_dir, scene=scene)

    return {
        "n_images":   len(ious),
        "mean_iou":   round(float(np.mean(ious)),   4),
        "median_iou": round(float(np.median(ious)), 4),
        "min_iou":    round(float(np.min(ious)),    4),
        "max_iou":    round(float(np.max(ious)),    4),
    }


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

def discover_human_models() -> list[Path]:
    """Find all trained human detection model weights."""
    return sorted(MODELS_ROOT.glob("human_detection_*/weights/best.pt"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    available_scenes = [
        d.name for d in TEST_DATASET_ROOT.iterdir()
        if d.is_dir() and d.name.startswith("human_")
        and (d / "train" / "images").exists()
    ] if TEST_DATASET_ROOT.exists() else []

    parser = argparse.ArgumentParser(description="Evaluate human detection IoU on test images")
    parser.add_argument("--scene", choices=available_scenes + ["all"], default="all",
                        help="Which human test scene to evaluate (default: all)")
    parser.add_argument("--output-csv", type=str, default=None,
                        help="Optional path to save combined CSV of all results")
    args = parser.parse_args()

    human_models = discover_human_models()
    if not human_models:
        print(f"No human detection models found under {MODELS_ROOT}")
        return

    scenes      = available_scenes if args.scene == "all" else [args.scene]
    all_results = []

    for scene in scenes:
        print(f"\n{'='*65}")
        print(f"  SCENE: {scene.upper()}  |  {len(human_models)} model(s)")
        print(f"{'='*65}")

        scene_results = []

        for model_path in human_models:
            model_name = model_path.parent.parent.name
            print(f"\n  [{model_name}]")

            result = evaluate_scene(scene, model_path)

            if "error" in result:
                print(f"  ERROR — {result['error']}")
            else:
                print(f"  mean IoU={result['mean_iou']}  "
                      f"median={result['median_iou']}  "
                      f"min={result['min_iou']}  max={result['max_iou']}  "
                      f"(n={result['n_images']})")
                row = {"scene": scene, "model": model_name, **result}
                scene_results.append(row)
                all_results.append(row)

        # Per-scene CSV
        if scene_results:
            scene_csv = VIS_OUTPUT_ROOT / scene / "results.csv"
            scene_csv.parent.mkdir(parents=True, exist_ok=True)
            fields = ["scene", "model", "mean_iou", "median_iou", "min_iou", "max_iou", "n_images"]
            with open(scene_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(sorted(scene_results, key=lambda x: -x["mean_iou"]))
            print(f"\n  Scene results saved → {scene_csv}")

    if all_results:
        print(f"\n\n{'='*65}")
        print("  SUMMARY")
        print(f"{'='*65}")
        print(f"{'Model':<50} {'Mean IoU':>9} {'Median':>9} {'Min':>7} {'Max':>7}")
        print("-" * 65)
        for r in sorted(all_results, key=lambda x: -x["mean_iou"]):
            print(f"{r['model']:<50} {r['mean_iou']:>9} {r['median_iou']:>9} "
                  f"{r['min_iou']:>7} {r['max_iou']:>7}")

        if args.output_csv:
            fields = ["scene", "model", "mean_iou", "median_iou", "min_iou", "max_iou", "n_images"]
            with open(args.output_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(all_results)
            print(f"\nSaved to {args.output_csv}")

    print(f"\nVisualisations saved under: {VIS_OUTPUT_ROOT}/")


if __name__ == "__main__":
    main()
