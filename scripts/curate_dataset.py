"""
Interactive dataset curation tool for test datasets.

Supports three modes:
  --task scene  --scene bridge|railway|ship
      Shows 1 GT panel + prediction from each model. Sorted worst IoU first.
      Removes image + label from data/test_dataset/images/{scene}/train/

  --task human
      Shows 1 GT panel + prediction from each model. Sorted worst detection first.
      Removes image + label from data/test_dataset/images/human_bridge/train/

  --task alarm
      Shows the single alarm-trigger vis per image. Sorted WRONG first.
      Removes image from data/test_dataset/images/overlap/Positive or Negative/

Controls:
    d  →  Delete (moves to removed_*/ folder for recovery)
    k  →  Keep
    u  →  Undo last decision
    q  →  Quit (progress saved, resume later)

Usage:
    python scripts/curate_dataset.py --task scene --scene bridge
    python scripts/curate_dataset.py --task scene --scene bridge --target 200
    python scripts/curate_dataset.py --task human --target 250
    python scripts/curate_dataset.py --task alarm
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EVAL_ROOT    = Path("eval_output")
DATASET_ROOT = Path("data/test_dataset/images")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Scene segmentation models
SCENE_MODELS = {
    "bridge":  ["bridge_hazard_yolo11n-seg", "bridge_hazard_yolo11s-seg",
                "bridge_hazard_yolo11m-seg", "bridge_hazard_yolo11l-seg",
                "bridge_hazard_yolo26l-seg"],
    "railway": ["railway_hazard_yolo11n-seg", "railway_hazard_yolo11s-seg",
                "railway_hazard_yolo11m-seg", "railway_hazard_yolo11l-seg",
                "railway_hazard_yolo26l-seg"],
    "ship":    ["ship_hazard_yolo11n-seg", "ship_hazard_yolo11s-seg",
                "ship_hazard_yolo11m-seg", "ship_hazard_yolo11l-seg",
                "ship_hazard_yolo26l-seg"],
}

# Human detection models
HUMAN_MODELS = [
    "human_detection_real_yolo11n-seg",
    "human_detection_real_yolo11s-seg",
    "human_detection_real_yolo11m-seg",
    "human_detection_real_yolo11l-seg",
    "human_detection_real_yolo26l-seg",
]

MODEL_SHORT = {
    "yolo11n-seg": "11n", "yolo11s-seg": "11s", "yolo11m-seg": "11m",
    "yolo11l-seg": "11l", "yolo26l-seg": "26l",
}


def get_model_short(model_name: str) -> str:
    for key, short in MODEL_SHORT.items():
        if model_name.endswith(key):
            return short
    return model_name


# ---------------------------------------------------------------------------
# Progress persistence
# ---------------------------------------------------------------------------

def progress_path(task: str, scene: str | None) -> Path:
    if task == "scene":
        return EVAL_ROOT / scene / ".curate_progress.json"
    elif task == "human":
        return EVAL_ROOT / "human_bridge" / ".curate_progress.json"
    else:
        return EVAL_ROOT / "alarm_trigger" / ".curate_progress.json"


def load_progress(task: str, scene: str | None) -> dict:
    p = progress_path(task, scene)
    if p.exists():
        return json.loads(p.read_text())
    return {"keep": [], "delete": []}


def save_progress_file(task: str, scene: str | None, progress: dict) -> None:
    p = progress_path(task, scene)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(progress, indent=2))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_image_by_stem(directory: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        p = directory / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def render_gt_panel(images_dir: Path, labels_dir: Path, stem: str,
                    target_height: int) -> np.ndarray | None:
    """Load original image + draw GT mask from YOLO label file."""
    img_path = find_image_by_stem(images_dir, stem)
    if img_path is None:
        return None

    img = cv2.imread(str(img_path))
    if img is None:
        return None

    h, w = img.shape[:2]
    label_path = labels_dir / f"{stem}.txt"

    gt_mask = np.zeros((h, w), dtype=np.uint8)
    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 7:
                    continue
                coords = list(map(float, parts[1:]))
                points = np.array(
                    [(coords[i] * w, coords[i + 1] * h)
                     for i in range(0, len(coords), 2)],
                    dtype=np.int32,
                )
                cv2.fillPoly(gt_mask, [points], 1)

    out = img.copy().astype(np.float32)
    tint = np.zeros_like(img, dtype=np.float32)
    tint[gt_mask == 1] = (0, 200, 0)
    out[gt_mask == 1] = out[gt_mask == 1] * 0.5 + tint[gt_mask == 1] * 0.5
    result = out.astype(np.uint8)
    contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 200, 0), 2)

    scale = target_height / h
    result = cv2.resize(result, (int(w * scale), target_height))
    cv2.putText(result, "GROUND TRUTH", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
    return result


def find_vis_by_stem(vis_dir: Path, stem: str, pattern: str) -> Path | None:
    """Find a vis file matching the stem using a regex pattern."""
    if not vis_dir.exists():
        return None
    for f in vis_dir.iterdir():
        if re.match(pattern.replace("{STEM}", re.escape(stem)), f.stem):
            return f
    return None


def extract_right_half(vis_path: Path, target_height: int) -> np.ndarray | None:
    """Load a side-by-side vis and return only the right half, resized."""
    img = cv2.imread(str(vis_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    half = img[:, w // 2:]
    ph, pw = half.shape[:2]
    scale = target_height / ph
    return cv2.resize(half, (int(pw * scale), target_height))


def hstack_panels(panels: list[np.ndarray]) -> np.ndarray:
    if not panels:
        return np.zeros((400, 800, 3), dtype=np.uint8)
    target_h = max(p.shape[0] for p in panels)
    aligned = []
    for p in panels:
        if p.shape[0] < target_h:
            pad = np.zeros((target_h - p.shape[0], p.shape[1], 3), dtype=np.uint8)
            p = np.vstack([p, pad])
        aligned.append(p)
    return np.hstack(aligned)


def add_header(image: np.ndarray, line1: str, line2: str = "") -> np.ndarray:
    header_h = 50 if not line2 else 70
    header = np.zeros((header_h, image.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, line1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if line2:
        cv2.putText(header, line2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (200, 200, 200), 1, cv2.LINE_AA)
    return np.vstack([header, image])


# ===================================================================
# SCENE SEGMENTATION CURATION
# ===================================================================

def scene_load_data(scene: str):
    """Returns (sorted_stems, scores_dict, models)."""
    models = SCENE_MODELS[scene]
    ious: dict[str, dict[str, float]] = {}

    for model in models:
        vis_dir = EVAL_ROOT / scene / model / "all"
        if not vis_dir.exists():
            print(f"  WARNING: {vis_dir} not found")
            continue
        for f in vis_dir.iterdir():
            m = re.match(r"^\d+_(.+)_iou([\d.]+)$", f.stem)
            if not m:
                continue
            ious.setdefault(m.group(1), {})[model] = float(m.group(2))

    mean_scores = {}
    for stem, model_ious in ious.items():
        vals = list(model_ious.values())
        mean_scores[stem] = sum(vals) / len(vals) if vals else 0.0

    sorted_stems = sorted(mean_scores.keys(), key=lambda s: mean_scores[s])
    return sorted_stems, ious, mean_scores, models


def scene_build_display(scene: str, stem: str, models: list[str],
                        ious: dict[str, float], max_h: int = 500) -> np.ndarray:
    images_dir = DATASET_ROOT / scene / "train" / "images"
    labels_dir = DATASET_ROOT / scene / "train" / "labels"
    panels = []

    gt = render_gt_panel(images_dir, labels_dir, stem, max_h)
    if gt is not None:
        panels.append(gt)

    for model in models:
        vis_path = find_vis_by_stem(
            EVAL_ROOT / scene / model / "all", stem,
            r"^\d+_{STEM}_iou[\d.]+$")
        if vis_path is None:
            p = np.zeros((max_h, 300, 3), dtype=np.uint8)
            cv2.putText(p, f"{get_model_short(model)}: N/A",
                        (10, max_h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (128, 128, 128), 1)
            panels.append(p)
            continue

        pred = extract_right_half(vis_path, max_h)
        if pred is None:
            continue
        iou = ious.get(model, 0.0)
        cv2.putText(pred, f"{get_model_short(model)}: IoU={iou:.4f}",
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(pred)

    return hstack_panels(panels)


def scene_apply_deletions(scene: str, stems: list[str], dry_run: bool):
    images_dir = DATASET_ROOT / scene / "train" / "images"
    labels_dir = DATASET_ROOT / scene / "train" / "labels"
    removed_img = images_dir.parent / "removed_images"
    removed_lbl = labels_dir.parent / "removed_labels"

    if dry_run:
        print(f"  DRY RUN — would delete {len(stems)} images")
        return

    removed_img.mkdir(exist_ok=True)
    removed_lbl.mkdir(exist_ok=True)

    moved = 0
    for stem in stems:
        p = find_image_by_stem(images_dir, stem)
        if p and p.exists():
            p.rename(removed_img / p.name)
            moved += 1
        lbl = labels_dir / f"{stem}.txt"
        if lbl.exists():
            lbl.rename(removed_lbl / lbl.name)

    remaining = sum(1 for f in images_dir.iterdir()
                    if f.suffix.lower() in IMAGE_EXTENSIONS)
    print(f"  Moved {moved} images + labels → {removed_img.parent}/removed_*/")
    print(f"  Remaining: {remaining}")


# ===================================================================
# HUMAN DETECTION CURATION
# ===================================================================

def human_load_data():
    """Returns (sorted_stems, scores_dict, mean_scores, models)."""
    models = HUMAN_MODELS
    # Filename: NNN_{stem}_tp{N}_fp{N}_fn{N}.jpg
    scores: dict[str, dict[str, tuple[int, int, int]]] = {}

    for model in models:
        vis_dir = EVAL_ROOT / "human_bridge" / model / "all"
        if not vis_dir.exists():
            print(f"  WARNING: {vis_dir} not found")
            continue
        for f in vis_dir.iterdir():
            m = re.match(r"^\d+_(.+)_tp(\d+)_fp(\d+)_fn(\d+)$", f.stem)
            if not m:
                continue
            stem = m.group(1)
            tp, fp, fn = int(m.group(2)), int(m.group(3)), int(m.group(4))
            scores.setdefault(stem, {})[model] = (tp, fp, fn)

    # Sort by mean FN across models (highest first — worst missed detections)
    mean_fn: dict[str, float] = {}
    for stem, model_scores in scores.items():
        fns = [fn for _, _, fn in model_scores.values()]
        mean_fn[stem] = sum(fns) / len(fns) if fns else 0.0

    sorted_stems = sorted(mean_fn.keys(), key=lambda s: mean_fn[s], reverse=True)
    return sorted_stems, scores, mean_fn, models


def human_build_display(stem: str, models: list[str],
                        scores: dict[str, tuple[int, int, int]],
                        max_h: int = 500) -> np.ndarray:
    images_dir = DATASET_ROOT / "human_bridge" / "train" / "images"
    labels_dir = DATASET_ROOT / "human_bridge" / "train" / "labels"
    panels = []

    gt = render_gt_panel(images_dir, labels_dir, stem, max_h)
    if gt is not None:
        panels.append(gt)

    for model in models:
        vis_path = find_vis_by_stem(
            EVAL_ROOT / "human_bridge" / model / "all", stem,
            r"^\d+_{STEM}_tp\d+_fp\d+_fn\d+$")
        if vis_path is None:
            p = np.zeros((max_h, 300, 3), dtype=np.uint8)
            cv2.putText(p, f"{get_model_short(model)}: N/A",
                        (10, max_h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (128, 128, 128), 1)
            panels.append(p)
            continue

        pred = extract_right_half(vis_path, max_h)
        if pred is None:
            continue

        tp, fp, fn = scores.get(model, (0, 0, 0))
        cv2.putText(pred, f"{get_model_short(model)}: TP={tp} FP={fp} FN={fn}",
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(pred)

    return hstack_panels(panels)


def human_apply_deletions(stems: list[str], dry_run: bool):
    images_dir = DATASET_ROOT / "human_bridge" / "train" / "images"
    labels_dir = DATASET_ROOT / "human_bridge" / "train" / "labels"
    removed_img = images_dir.parent / "removed_images"
    removed_lbl = labels_dir.parent / "removed_labels"

    if dry_run:
        print(f"  DRY RUN — would delete {len(stems)} images")
        return

    removed_img.mkdir(exist_ok=True)
    removed_lbl.mkdir(exist_ok=True)

    moved = 0
    for stem in stems:
        p = find_image_by_stem(images_dir, stem)
        if p and p.exists():
            p.rename(removed_img / p.name)
            moved += 1
        lbl = labels_dir / f"{stem}.txt"
        if lbl.exists():
            lbl.rename(removed_lbl / lbl.name)

    remaining = sum(1 for f in images_dir.iterdir()
                    if f.suffix.lower() in IMAGE_EXTENSIONS)
    print(f"  Moved {moved} images + labels → {removed_img.parent}/removed_*/")
    print(f"  Remaining: {remaining}")


# ===================================================================
# ALARM TRIGGER CURATION
# ===================================================================

def alarm_load_data():
    """Returns (sorted image info list).

    Each entry: {filename, vis_path, gt_label, scene, fired, correct}
    """
    alarm_dir = EVAL_ROOT / "alarm_trigger"
    if not alarm_dir.exists():
        return []

    entries = []
    for f in alarm_dir.iterdir():
        if f.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        # Pattern: NNN_{scene}_{img_stem}_{correct|WRONG}.jpg
        m = re.match(r"^(\d+)_(\w+?)_(.+?)_(correct|WRONG)$", f.stem)
        if not m:
            continue
        rank = int(m.group(1))
        scene = m.group(2)
        img_stem = m.group(3)
        correct = m.group(4) == "correct"
        entries.append({
            "rank": rank,
            "scene": scene,
            "img_stem": img_stem,
            "correct": correct,
            "vis_path": f,
            "key": f"{scene}_{img_stem}",
        })

    # Sort: WRONG first, then by rank
    entries.sort(key=lambda e: (e["correct"], e["rank"]))
    return entries


def alarm_build_display(entry: dict, max_h: int = 600) -> np.ndarray:
    """Just show the alarm trigger vis image (single panel)."""
    img = cv2.imread(str(entry["vis_path"]))
    if img is None:
        return np.zeros((max_h, 800, 3), dtype=np.uint8)

    h, w = img.shape[:2]
    if h > max_h:
        scale = max_h / h
        img = cv2.resize(img, (int(w * scale), max_h))
    return img


def alarm_apply_deletions(entries_to_delete: list[dict], dry_run: bool):
    """Delete alarm trigger images from overlap/Positive or Negative."""
    overlap_root = DATASET_ROOT / "overlap"
    pos_dir = overlap_root / "Positive"
    neg_dir = overlap_root / "Negative"
    removed_dir = overlap_root / "removed"

    if dry_run:
        print(f"  DRY RUN — would delete {len(entries_to_delete)} images")
        return

    removed_dir.mkdir(exist_ok=True)

    moved = 0
    for entry in entries_to_delete:
        img_stem = entry["img_stem"]
        # Determine if positive or negative from the original filename
        # Positive files: {scene}_positive_{nn}.ext
        # Negative files: {scene}_negative_{nn}.ext
        for folder in [pos_dir, neg_dir]:
            p = find_image_by_stem(folder, img_stem)
            if p and p.exists():
                p.rename(removed_dir / p.name)
                moved += 1
                # Also move empty_ver cleanup if exists
                if folder == pos_dir:
                    for ext in IMAGE_EXTENSIONS:
                        cleanup = pos_dir / "empty_ver" / f"{img_stem}_cleanup{ext}"
                        if cleanup.exists():
                            cleanup.rename(removed_dir / cleanup.name)
                break

    print(f"  Moved {moved} images → {removed_dir}/")


# ===================================================================
# GENERIC REVIEW LOOP
# ===================================================================

def review_loop(task: str, scene: str | None, items: list,
                get_key, build_display_fn, get_sort_info,
                apply_deletions_fn, target: int | None, dry_run: bool,
                window_title: str, reset: bool = False):
    """Generic interactive review loop.

    Args:
        items: ordered list of items to review
        get_key: item → unique string key for progress tracking
        build_display_fn: item → numpy image
        get_sort_info: item → string to show in status bar
        apply_deletions_fn: (deleted_items, dry_run) → None
        target: if set, stop when dataset reduced to this size
        reset: if True, clear all progress and start from scratch
    """
    if reset:
        progress = {"keep": [], "delete": []}
        save_progress_file(task, scene, progress)
        print("  Progress reset.")
    else:
        progress = load_progress(task, scene)
    already_decided = set(progress["keep"] + progress["delete"])
    remaining = [item for item in items if get_key(item) not in already_decided]

    total = len(items)
    to_remove = (total - target) if target else None

    print(f"  Total images: {total}")
    if target:
        print(f"  Target: {target}")
        print(f"  Need to remove: {to_remove}")
    print(f"  Already marked for deletion: {len(progress['delete'])}")
    print(f"  Already kept: {len(progress['keep'])}")
    print(f"  Remaining to review: {len(remaining)}")
    print(f"\n  Controls:  [d] Delete  [k] Keep  [u] Undo  [q] Quit\n")

    if to_remove is not None and to_remove <= 0:
        print(f"  Dataset already at or below target!")
        return

    history: list[tuple[str, str, object]] = []  # (key, action, item)
    i = 0
    deleted_items = []

    while i < len(remaining):
        item = remaining[i]
        key = get_key(item)
        n_deleted = len(progress["delete"])

        if to_remove is not None and n_deleted >= to_remove:
            print(f"\n  Target reached! {n_deleted} images marked for deletion.")
            break

        display = build_display_fn(item)
        sort_info = get_sort_info(item)

        n_decided = len(progress["keep"]) + len(progress["delete"])
        status = f"[{n_decided + 1}/{total}]  {sort_info}"
        if to_remove is not None:
            status += f"  |  Deleted: {n_deleted}/{to_remove}"
        else:
            status += f"  |  Deleted: {n_deleted}"
        status += "  |  [d]elete  [k]eep  [u]ndo  [q]uit"

        display = add_header(display, status, key if len(key) < 120 else key[:117] + "...")

        cv2.imshow(window_title, display)
        k = cv2.waitKey(0) & 0xFF

        if k == ord("q"):
            print(f"\n  Quit. Progress saved.")
            break

        elif k == ord("u"):
            if history:
                last_key, last_action, last_item = history.pop()
                progress[last_action].remove(last_key)
                if last_action == "delete":
                    deleted_items = [it for it in deleted_items
                                     if get_key(it) != last_key]
                remaining.insert(i, last_item)
                print(f"  Undo: {last_key[:60]} ({last_action})")
            else:
                print("  Nothing to undo.")
            continue

        elif k == ord("d"):
            progress["delete"].append(key)
            deleted_items.append(item)
            history.append((key, "delete", item))
            print(f"  DELETE  {sort_info}  {key[:60]}")
            i += 1

        elif k == ord("k"):
            progress["keep"].append(key)
            history.append((key, "keep", item))
            i += 1

        else:
            continue

        save_progress_file(task, scene, progress)

    cv2.destroyAllWindows()

    # Apply
    all_deleted_keys = progress["delete"]
    all_deleted_items = [item for item in items if get_key(item) in set(all_deleted_keys)]

    if not all_deleted_items:
        print("  No images to delete.")
        return

    print(f"\n  {len(all_deleted_items)} images marked for deletion.")
    apply_deletions_fn(all_deleted_items, dry_run)


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Curate test datasets")
    parser.add_argument("--task", required=True,
                        choices=["scene", "human", "alarm"])
    parser.add_argument("--scene", choices=["bridge", "railway", "ship"],
                        help="Required for --task scene")
    parser.add_argument("--target", type=int, default=None,
                        help="Target dataset size after removal")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reset", action="store_true",
                        help="Reset progress and start from the first image")
    args = parser.parse_args()

    if args.task == "scene":
        if not args.scene:
            parser.error("--scene is required for --task scene")
        target = args.target or 200
        scene = args.scene

        print(f"\n  Loading scene segmentation results for {scene}...")
        sorted_stems, ious, mean_scores, models = scene_load_data(scene)
        if not sorted_stems:
            print("  ERROR: No eval results found. Run eval_scene first.")
            return

        review_loop(
            task="scene",
            scene=scene,
            items=sorted_stems,
            get_key=lambda stem: stem,
            build_display_fn=lambda stem: scene_build_display(
                scene, stem, models, ious.get(stem, {})),
            get_sort_info=lambda stem: f"Mean IoU: {mean_scores.get(stem, 0):.4f}",
            apply_deletions_fn=lambda items, dr: scene_apply_deletions(
                scene, items, dr),
            target=target,
            dry_run=args.dry_run,
            window_title=f"Curate {scene} segmentation",
            reset=args.reset,
        )

    elif args.task == "human":
        target = args.target or 250

        print(f"\n  Loading human detection results...")
        sorted_stems, scores, mean_fn, models = human_load_data()
        if not sorted_stems:
            print("  ERROR: No eval results found. Run eval_human first.")
            return

        review_loop(
            task="human",
            scene=None,
            items=sorted_stems,
            get_key=lambda stem: stem,
            build_display_fn=lambda stem: human_build_display(
                stem, models, scores.get(stem, {})),
            get_sort_info=lambda stem: f"Mean FN: {mean_fn.get(stem, 0):.1f}",
            apply_deletions_fn=lambda items, dr: human_apply_deletions(items, dr),
            target=target,
            dry_run=args.dry_run,
            window_title="Curate human detection",
            reset=args.reset,
        )

    elif args.task == "alarm":
        print(f"\n  Loading alarm trigger results...")
        entries = alarm_load_data()
        if not entries:
            print("  ERROR: No alarm trigger vis found. Run eval_alarm_trigger first.")
            return

        review_loop(
            task="alarm",
            scene=None,
            items=entries,
            get_key=lambda e: e["key"],
            build_display_fn=lambda e: alarm_build_display(e),
            get_sort_info=lambda e: (
                f"{'WRONG' if not e['correct'] else 'correct'}  "
                f"scene={e['scene']}"),
            apply_deletions_fn=lambda items, dr: alarm_apply_deletions(items, dr),
            target=args.target,
            dry_run=args.dry_run,
            window_title="Curate alarm trigger",
            reset=args.reset,
        )


if __name__ == "__main__":
    main()
