"""
Visual test for hazard zone segmentation models.

Loads each available model, runs inference on sample images from the
validation set, and displays results with overlaid segmentation masks
and polygon zones.

Usage:
    python tests/test_auto_segmentation.py                  # test all available models
    python tests/test_auto_segmentation.py --scene ship     # test only ship model
    python tests/test_auto_segmentation.py --samples 10     # use 10 samples per scene
"""

import argparse
import os
import sys
import random

import cv2
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.backend.config import SEG_MODEL_PATHS, AUTO_SEG_CONFIDENCE, AUTO_SEG_SIMPLIFY_EPSILON


# Dataset paths (validation sets)
DATASET_PATHS = {
    "ship": os.path.join(PROJECT_ROOT, "data", "ship_dataset", "images", "val"),
    "railway": os.path.join(PROJECT_ROOT, "data", "railway_dataset", "images", "val"),
    "bridge": os.path.join(PROJECT_ROOT, "data", "bridge_dataset", "images", "val"),
}

# Colors for drawing (BGR)
MASK_COLOR = (0, 0, 255)      # red overlay for mask
POLYGON_COLOR = (0, 255, 0)   # green for polygon outline
MASK_ALPHA = 0.35


def get_sample_images(scene_type: str, n: int) -> list[str]:
    """Get up to n random image paths from the validation set."""
    val_dir = DATASET_PATHS.get(scene_type, "")
    if not os.path.isdir(val_dir):
        print(f"  [SKIP] No validation images found at: {val_dir}")
        return []

    images = [
        os.path.join(val_dir, f)
        for f in os.listdir(val_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not images:
        print(f"  [SKIP] No images in {val_dir}")
        return []

    random.shuffle(images)
    return images[:n]


def draw_results(frame: np.ndarray, results, scene_type: str) -> np.ndarray:
    """Draw segmentation masks and polygon zones on the frame."""
    vis = frame.copy()
    h, w = frame.shape[:2]

    if results.masks is None:
        cv2.putText(vis, "No detections", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return vis

    # Draw each mask as a colored overlay
    for i, mask_tensor in enumerate(results.masks.data):
        mask_raw = mask_tensor.cpu().numpy()
        mask_resized = cv2.resize(mask_raw, (w, h), interpolation=cv2.INTER_LINEAR)
        mask_binary = (mask_resized > 0.5).astype(np.uint8)

        # Colored overlay
        overlay = vis.copy()
        overlay[mask_binary == 1] = MASK_COLOR
        vis = cv2.addWeighted(overlay, MASK_ALPHA, vis, 1 - MASK_ALPHA, 0)

        # Draw polygon outlines (same logic as auto_segmentation.py)
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 100:
                continue
            approx = cv2.approxPolyDP(contour, AUTO_SEG_SIMPLIFY_EPSILON, True)
            if len(approx) >= 3:
                cv2.drawContours(vis, [approx], -1, POLYGON_COLOR, 2)

        # Draw confidence + class label
        if i < len(results.boxes):
            conf = float(results.boxes[i].conf[0])
            cls_id = int(results.boxes[i].cls[0])
            cls_name = results.names.get(cls_id, str(cls_id))
            label = f"{cls_name} {conf:.2f}"
            # Position label at top of mask bounding box
            ys = np.where(mask_binary)[0]
            xs = np.where(mask_binary)[1]
            if len(ys) > 0:
                tx, ty = int(xs.min()), max(int(ys.min()) - 8, 20)
                cv2.putText(vis, label, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, POLYGON_COLOR, 2)

    # Summary text
    n_masks = len(results.masks.data)
    cv2.putText(vis, f"{scene_type} | {n_masks} zone(s) detected", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return vis


def test_model(scene_type: str, n_samples: int, confidence: float):
    """Test a single scene type model."""
    model_path = SEG_MODEL_PATHS.get(scene_type)
    if not model_path:
        print(f"\n[{scene_type.upper()}] No model path configured — skipping")
        return

    if not os.path.exists(model_path):
        print(f"\n[{scene_type.upper()}] Model file not found: {model_path} — skipping")
        return

    print(f"\n{'='*60}")
    print(f"  Testing: {scene_type.upper()} model")
    print(f"  Model:   {model_path}")
    print(f"{'='*60}")

    from ultralytics import YOLO
    model = YOLO(model_path)

    # Print model info
    print(f"  Classes: {model.names}")
    print(f"  Confidence threshold: {confidence}")

    images = get_sample_images(scene_type, n_samples)
    if not images:
        return

    print(f"  Testing on {len(images)} validation images...\n")

    total_detections = 0

    for idx, img_path in enumerate(images):
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"  [{idx+1}] Could not read: {img_path}")
            continue

        results = model(frame, conf=confidence, verbose=False)[0]
        n_det = len(results.masks.data) if results.masks is not None else 0
        total_detections += n_det

        vis = draw_results(frame, results, scene_type)

        status = f"{n_det} zone(s)" if n_det > 0 else "NO DETECTIONS"
        print(f"  [{idx+1}/{len(images)}] {os.path.basename(img_path)}: {status}")

        # Resize for display if too large
        display = vis
        if vis.shape[1] > 1280:
            scale = 1280 / vis.shape[1]
            display = cv2.resize(vis, None, fx=scale, fy=scale)
        cv2.imshow(f"{scene_type} segmentation test", display)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            print("  User quit (q pressed)")
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()

    print(f"\n  Summary: {total_detections} total detections across {len(images)} images")
    print(f"  Avg detections per image: {total_detections / max(len(images), 1):.1f}")


def main():
    parser = argparse.ArgumentParser(description="Test hazard zone segmentation models")
    parser.add_argument("--scene", type=str, default=None,
                        choices=["ship", "railway", "bridge"],
                        help="Test only this scene type (default: all available)")
    parser.add_argument("--samples", type=int, default=5,
                        help="Number of sample images per scene (default: 5)")
    parser.add_argument("--conf", type=float, default=None,
                        help=f"Override confidence threshold (default: {AUTO_SEG_CONFIDENCE})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible image selection")
    args = parser.parse_args()

    random.seed(args.seed)

    confidence = args.conf if args.conf is not None else AUTO_SEG_CONFIDENCE

    print("Hazard Zone Segmentation Model Test")
    print(f"  Confidence: {confidence}")
    print(f"  Samples per scene: {args.samples}")

    scene_types = [args.scene] if args.scene else ["ship", "railway", "bridge"]

    for scene_type in scene_types:
        test_model(scene_type, args.samples, confidence)

    print("\nDone.")


if __name__ == "__main__":
    main()
