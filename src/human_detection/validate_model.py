"""
Validate the human detection model by running inference on sample images
and visualizing the output masks.

Saves results as side-by-side images (original | mask overlay) to an output directory,
and optionally displays them in a window.

Usage:
    python -m src.human_detection.validate_model
    python -m src.human_detection.validate_model --images data/human_images --output data/validation_results --show
    python -m src.human_detection.validate_model --images data/human_dataset/images/val --output data/validation_results
"""
import argparse
import cv2
import numpy as np
from pathlib import Path
from .detector import HumanDetector
from .config import MODEL_PATH, CONFIDENCE_THRESHOLD, INFERENCE_IMGSZ


def validate_model(images_dir, output_dir, show=False):
    """
    Run detection on all images in a directory and save visual results.

    For each image, produces a side-by-side comparison:
      Left:  Original image with mask overlay (green) and bounding boxes
      Right: Raw binary mask (white = detected human)

    Args:
        images_dir: Directory containing test images (.jpg, .png).
        output_dir: Directory to save result images.
        show: If True, display each result in a CV2 window.
    """
    images_path = Path(images_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    img_files = sorted(list(images_path.glob("*.jpg")) + list(images_path.glob("*.png")))
    if not img_files:
        print(f"No images found in {images_path}")
        return

    print(f"Model: {MODEL_PATH}")
    print(f"Confidence: {CONFIDENCE_THRESHOLD} | Image Size: {INFERENCE_IMGSZ}")
    print(f"Found {len(img_files)} images in {images_path}")
    print(f"Saving results to {output_path}\n")

    detector = HumanDetector()

    total_detections = 0

    for img_path in img_files:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  [SKIP] Could not read: {img_path.name}")
            continue

        h, w = frame.shape[:2]

        # Run detection
        masks = detector.get_masks(frame)
        num_detected = len(masks)
        total_detections += num_detected

        # --- Left panel: original with green mask overlay and count ---
        overlay_frame = frame.copy()
        if masks:
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            for mask in masks:
                combined_mask = np.maximum(combined_mask, mask)

            mask_overlay = np.zeros_like(frame)
            mask_overlay[combined_mask == 1] = (0, 255, 0)
            overlay_frame = cv2.addWeighted(overlay_frame, 0.6, mask_overlay, 0.4, 0)

            # Draw contours around each individual mask for clarity
            for mask in masks:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay_frame, contours, -1, (0, 255, 0), 2)

        # Draw detection count
        label = f"Detected: {num_detected}"
        cv2.putText(overlay_frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # --- Right panel: raw binary mask ---
        if masks:
            raw_mask_vis = combined_mask * 255
        else:
            raw_mask_vis = np.zeros((h, w), dtype=np.uint8)

        # Convert grayscale mask to BGR so we can stack horizontally
        raw_mask_bgr = cv2.cvtColor(raw_mask_vis, cv2.COLOR_GRAY2BGR)

        # Combine side by side
        combined = np.hstack([overlay_frame, raw_mask_bgr])

        # Save result
        result_filename = f"result_{img_path.stem}.jpg"
        result_path = output_path / result_filename
        cv2.imwrite(str(result_path), combined)

        status = "OK" if num_detected > 0 else "NO DETECTION"
        print(f"  [{status}] {img_path.name} -> {num_detected} person(s) detected")

        if show:
            cv2.imshow(f"Validation: {img_path.name}", combined)
            print("    Press any key for next, 'q' to quit display")
            if cv2.waitKey(0) & 0xFF == ord("q"):
                show = False  # Stop showing but keep saving

    if show:
        cv2.destroyAllWindows()

    print(f"\nSummary:")
    print(f"  Images processed: {len(img_files)}")
    print(f"  Total detections: {total_detections}")
    print(f"  Results saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate human detection model on sample images")
    parser.add_argument("--images", type=str, default="data/human_images",
                        help="Directory containing test images")
    parser.add_argument("--output", type=str, default="data/validation_results",
                        help="Directory to save result images")
    parser.add_argument("--show", action="store_true",
                        help="Display results in a window (press any key to advance)")
    args = parser.parse_args()
    validate_model(args.images, args.output, args.show)
