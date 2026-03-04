"""
Verify human detection dataset labels by visualizing annotations overlaid on images.
Displays polygon annotations overlaid on random sample images to catch labeling errors.

Usage:
    # Verify the default combined dataset (train split, 5 samples)
    python -m src.human_detection.verify_dataset

    # Verify a specific dataset variant
    python -m src.human_detection.verify_dataset --dataset data/human_dataset_sim

    # Verify the val split with more samples
    python -m src.human_detection.verify_dataset --dataset data/human_dataset_real --split val --samples 10

Args:
    --dataset       Path to dataset directory (default: data/human_dataset)
    --split         Which split to verify: train | val (default: train)
    --samples       Number of random images to display (default: 5)
"""
import cv2
import random
import numpy as np
from pathlib import Path


def verify_dataset(dataset_path="data/human_dataset", split="train", num_samples=5):
    """
    Visualizes polygon annotations overlaid on images to catch labeling errors.
    """
    print(f"\n--- Verifying human dataset ({split}) ---")

    img_dir = Path(dataset_path) / "images" / split
    label_dir = Path(dataset_path) / "labels" / split

    if not img_dir.exists():
        print(f"Error: Directory {img_dir} does not exist.")
        return

    img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    if not img_files:
        print(f"No images found in {img_dir}")
        return

    samples = random.sample(img_files, min(len(img_files), num_samples))

    for img_path in samples:
        print(f"Checking: {img_path.name}")
        img = cv2.imread(str(img_path))
        h, w, _ = img.shape

        label_path = label_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            print(f"  [!] Missing label file for {img_path.name}")
            continue

        # Draw polygon annotations
        with open(label_path, "r") as f:
            for line in f.readlines():
                parts = list(map(float, line.strip().split()))
                class_id = int(parts[0])
                coords = parts[1:]

                points = []
                for i in range(0, len(coords), 2):
                    points.append([int(coords[i] * w), int(coords[i + 1] * h)])

                pts = np.array(points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

                # Semi-transparent fill
                overlay = img.copy()
                cv2.fillPoly(overlay, [pts], color=(0, 255, 0))
                cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

        cv2.imshow(f"Verify: {img_path.name}", img)
        print("  Press any key for next, or 'q' to quit.")
        if cv2.waitKey(0) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Verify human detection dataset labels")
    parser.add_argument("--dataset", type=str, default="data/human_dataset",
                        help="Path to the dataset directory")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"],
                        help="Which split to verify")
    parser.add_argument("--samples", type=int, default=5,
                        help="Number of random samples to show")
    args = parser.parse_args()
    verify_dataset(args.dataset, args.split, args.samples)
