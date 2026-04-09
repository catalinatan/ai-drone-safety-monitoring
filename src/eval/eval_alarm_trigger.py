"""
Evaluate end-to-end alarm triggering on still images.

For each image the pipeline runs:
  1. Scene segmentation model  → danger zone mask
  2. Human detection model     → person masks
  3. Overlap check             → alarm triggered if any person mask overlaps
                                 the danger zone above DANGER_ZONE_OVERLAP_THRESHOLD

Ground truth:
  Positive/   — alarm SHOULD trigger  (TP if fires, FN if not)
  Negative/   — alarm should NOT fire (FP if fires, TN if not)

Positive images have a paired cleanup version in Positive/empty_ver/:
  e.g. bridge_positive_01.png  →  bridge_positive_01_cleanup.png
  The cleanup version (humans removed) is used for scene segmentation so the
  danger zone is not occluded by the person. Human detection still runs on the
  original image.

Model paths are set near the top of this file — swap them to the best models
found from eval_scene and eval_human before running.

Usage:
    python -m src.eval.eval_alarm_trigger
    python -m src.eval.eval_alarm_trigger --scene-filter bridge
    python -m src.eval.eval_alarm_trigger --output-csv alarm_results.csv
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.human_detection.config import DANGER_ZONE_OVERLAP_THRESHOLD

# ---------------------------------------------------------------------------
# *** SET THESE TO YOUR BEST MODELS BEFORE RUNNING ***
# ---------------------------------------------------------------------------

SEG_MODEL_PATH    = Path("runs/segment/runs/segment/bridge_hazard_yolo11m-seg/weights/best.pt")
HUMAN_MODEL_PATH  = Path("runs/segment/runs/segment/human_detection_real_yolo11m-seg/weights/best.pt")

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------

OVERLAP_ROOT   = Path("data/test_dataset/images/overlap")
POSITIVE_DIR   = OVERLAP_ROOT / "Positive"
EMPTY_VER_DIR  = OVERLAP_ROOT / "Positive" / "empty_ver"
NEGATIVE_DIR   = OVERLAP_ROOT / "Negative"
VIS_OUTPUT_ROOT = Path("eval_output/alarm_trigger")

CONF_THRESHOLD   = 0.25
INFERENCE_IMGSZ  = 1280
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def get_seg_mask(model: YOLO, image_path: Path, img_w: int, img_h: int) -> np.ndarray:
    """Run scene segmentation → single combined binary danger zone mask."""
    results = model(str(image_path), conf=CONF_THRESHOLD,
                    imgsz=INFERENCE_IMGSZ, verbose=False, save=False)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if results[0].masks is None:
        return mask
    for m in results[0].masks.data:
        resized = cv2.resize(m.cpu().numpy(), (img_w, img_h))
        mask = np.maximum(mask, (resized > 0.5).astype(np.uint8))
    return mask


def get_human_masks(model: YOLO, image_path: Path,
                    img_w: int, img_h: int) -> list[np.ndarray]:
    """Run human detection → list of individual person binary masks."""
    results = model(str(image_path), conf=CONF_THRESHOLD,
                    imgsz=INFERENCE_IMGSZ, verbose=False, save=False)
    masks = []
    if results[0].masks is None:
        return masks
    for m in results[0].masks.data:
        resized = cv2.resize(m.cpu().numpy(), (img_w, img_h))
        binary = (resized > 0.5).astype(np.uint8)
        if binary.sum() > 0:
            masks.append(binary)
    return masks


def alarm_fires(danger_zone: np.ndarray, human_masks: list[np.ndarray]) -> bool:
    """
    Mirrors the runtime alarm logic: alarm triggers if any person has
    >= DANGER_ZONE_OVERLAP_THRESHOLD of their mask inside the danger zone.
    """
    for person in human_masks:
        person_area = int(person.sum())
        if person_area == 0:
            continue
        overlap = float(np.logical_and(person, danger_zone).sum()) / person_area
        if overlap >= DANGER_ZONE_OVERLAP_THRESHOLD:
            return True
    return False


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def make_visualisation(orig_img: np.ndarray,
                       danger_zone: np.ndarray,
                       human_masks: list[np.ndarray],
                       fired: bool,
                       gt_label: str) -> np.ndarray:
    """
    Single panel showing:
      - Red overlay   = danger zone
      - Green overlay = detected humans
      - Header: GT label, alarm fired/not, correct/wrong
    """
    out = orig_img.copy().astype(np.float32)

    # Danger zone — red tint
    tint = np.zeros_like(orig_img, dtype=np.float32)
    tint[danger_zone == 1] = (0, 0, 220)
    out[danger_zone == 1] = out[danger_zone == 1] * 0.5 + tint[danger_zone == 1] * 0.5
    out = out.astype(np.uint8)
    contours, _ = cv2.findContours(danger_zone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 0, 220), 2)

    # Human masks — green tint
    for person in human_masks:
        tint = np.zeros_like(orig_img, dtype=np.float32)
        tint[person == 1] = (0, 200, 0)
        out_f = out.astype(np.float32)
        out_f[person == 1] = out_f[person == 1] * 0.5 + tint[person == 1] * 0.5
        out = out_f.astype(np.uint8)
        contours, _ = cv2.findContours(person, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 200, 0), 2)

    # Header
    h, w = orig_img.shape[:2]
    label_h = 36
    panel = np.zeros((h + label_h, w, 3), dtype=np.uint8)
    panel[label_h:] = out

    alarm_str  = "ALARM" if fired else "no alarm"
    correct    = (fired and gt_label == "positive") or (not fired and gt_label == "negative")
    result_str = "CORRECT" if correct else "WRONG"
    colour     = (0, 220, 0) if correct else (0, 0, 220)

    text = f"GT={gt_label}  pred={alarm_str}  [{result_str}]  " \
           f"zone={'yes' if danger_zone.sum()>0 else 'no'}  " \
           f"humans={len(human_masks)}"
    cv2.putText(panel, text, (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)

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


def evaluate(seg_model: YOLO, human_model: YOLO,
             scene_filter: str | None) -> dict:
    tp = fp = tn = fn = 0
    records = []

    for gt_label, folder in [("positive", POSITIVE_DIR), ("negative", NEGATIVE_DIR)]:
        image_paths = sorted([
            p for p in folder.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
            and p.is_file()
            and (scene_filter is None or p.stem.lower().startswith(scene_filter.lower()))
        ])

        for img_path in image_paths:
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

            human_masks  = get_human_masks(human_model, img_path, w, h)
            fired        = alarm_fires(danger_zone, human_masks)

            if gt_label == "positive":
                if fired: tp += 1
                else:     fn += 1
            else:
                if fired: fp += 1
                else:     tn += 1

            records.append({
                "img_path":  img_path,
                "gt_label":  gt_label,
                "fired":     fired,
                "orig":      orig,
                "danger":    danger_zone,
                "humans":    human_masks,
            })

    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "records": records}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate end-to-end alarm trigger")
    parser.add_argument("--scene-filter", type=str, default=None,
                        help="Only process images whose filename starts with this "
                             "(e.g. 'bridge', 'ship')")
    parser.add_argument("--output-csv", type=str, default=None)
    args = parser.parse_args()

    print(f"Scene seg model : {SEG_MODEL_PATH}")
    print(f"Human det model : {HUMAN_MODEL_PATH}")
    print(f"Overlap threshold: {DANGER_ZONE_OVERLAP_THRESHOLD}")
    if args.scene_filter:
        print(f"Scene filter    : {args.scene_filter}")

    if not SEG_MODEL_PATH.exists():
        print(f"ERROR: seg model not found at {SEG_MODEL_PATH}")
        return
    if not HUMAN_MODEL_PATH.exists():
        print(f"ERROR: human model not found at {HUMAN_MODEL_PATH}")
        return

    seg_model   = YOLO(str(SEG_MODEL_PATH))
    human_model = YOLO(str(HUMAN_MODEL_PATH))

    print("\nRunning evaluation...")
    result = evaluate(seg_model, human_model, args.scene_filter)

    tp, fp, tn, fn = result["tp"], result["fp"], result["tn"], result["fn"]
    records = result["records"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = (tp + tn) / (tp + tn + fp + fn) if records else 0.0

    print(f"\n{'='*55}")
    print(f"  RESULTS")
    print(f"{'='*55}")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  Accuracy  : {accuracy:.4f}")

    # Save visualisations
    tag = args.scene_filter or "all"
    out_dir = VIS_OUTPUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    for rank, rec in enumerate(records, 1):
        vis = make_visualisation(
            rec["orig"], rec["danger"], rec["humans"],
            rec["fired"], rec["gt_label"]
        )
        correct = (rec["fired"] and rec["gt_label"] == "positive") or \
                  (not rec["fired"] and rec["gt_label"] == "negative")
        label = "correct" if correct else "WRONG"
        out_path = out_dir / f"{rank:03d}_{rec['img_path'].stem}_{label}.jpg"
        cv2.imwrite(str(out_path), vis)

    print(f"\n  Visualisations saved → {out_dir}")

    # CSV
    csv_path = out_dir / "results.csv"
    fields = ["image", "gt_label", "alarm_fired", "correct"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            correct = (rec["fired"] and rec["gt_label"] == "positive") or \
                      (not rec["fired"] and rec["gt_label"] == "negative")
            writer.writerow({
                "image":       rec["img_path"].name,
                "gt_label":    rec["gt_label"],
                "alarm_fired": rec["fired"],
                "correct":     correct,
            })

    summary_path = out_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "seg_model", "human_model", "scene_filter",
            "tp", "fp", "tn", "fn", "precision", "recall", "f1", "accuracy"
        ])
        writer.writeheader()
        writer.writerow({
            "seg_model":    SEG_MODEL_PATH.parent.parent.name,
            "human_model":  HUMAN_MODEL_PATH.parent.parent.name,
            "scene_filter": tag,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
            "accuracy":  round(accuracy,  4),
        })

    print(f"  Per-image CSV  → {csv_path}")
    print(f"  Summary CSV    → {summary_path}")

    if args.output_csv:
        import shutil
        shutil.copy(summary_path, args.output_csv)
        print(f"  Summary also saved → {args.output_csv}")


if __name__ == "__main__":
    main()
