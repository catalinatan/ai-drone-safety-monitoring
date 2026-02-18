"""
Prepare the Roboflow-exported dataset into YOLO train/val split format.

Expects a flat Roboflow export at data/human_dataset/:
  data/human_dataset/
    images/   (all images in one folder)
    labels/   (all labels in one folder)

Reorganizes into:
  data/human_dataset/
    images/train/  images/val/
    labels/train/  labels/val/
    data.yaml

Usage:
    python -m src.human_detection.prepare_dataset
    python -m src.human_detection.prepare_dataset --dataset data/human_dataset --val-split 0.2
"""
import argparse
import random
import shutil
from pathlib import Path
import yaml


def prepare_dataset(dataset_dir, val_split=0.2, seed=42):
    """
    Split a flat images/labels directory into train/val sets in YOLO format.
    Moves files from the flat structure into train/val subdirectories.
    """
    random.seed(seed)

    dataset_path = Path(dataset_dir)
    images_path = dataset_path / "images"
    labels_path = dataset_path / "labels"

    if not images_path.exists() or not labels_path.exists():
        print(f"Error: Expected images/ and labels/ directories inside {dataset_path}")
        return

    # Check if already split (train/val subdirs exist with images)
    train_dir = images_path / "train"
    if train_dir.exists() and any(train_dir.glob("*.jpg")):
        print("Dataset appears to already be split into train/val. Skipping.")
        return

    # Collect image files with matching labels (from flat directory)
    image_files = sorted(list(images_path.glob("*.jpg")) + list(images_path.glob("*.png")))

    paired = []
    for img_file in image_files:
        label_file = labels_path / (img_file.stem + ".txt")
        if label_file.exists():
            paired.append((img_file, label_file))
        else:
            print(f"  [SKIP] No label for: {img_file.name}")

    if not paired:
        print("Error: No image-label pairs found!")
        return

    print(f"Found {len(paired)} image-label pairs")

    # Shuffle and split
    random.shuffle(paired)
    val_count = max(1, int(len(paired) * val_split))
    val_set = paired[:val_count]
    train_set = paired[val_count:]

    print(f"  Train: {len(train_set)}, Val: {val_count}")

    # Create subdirectories
    for split in ["train", "val"]:
        (images_path / split).mkdir(parents=True, exist_ok=True)
        (labels_path / split).mkdir(parents=True, exist_ok=True)

    # Move files into split directories
    for split_name, split_data in [("train", train_set), ("val", val_set)]:
        for img_file, label_file in split_data:
            shutil.move(str(img_file), images_path / split_name / img_file.name)
            shutil.move(str(label_file), labels_path / split_name / label_file.name)

    # Generate data.yaml
    data_yaml = {
        "path": str(dataset_path.absolute()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["person"],
    }
    yaml_path = dataset_path / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f)

    print(f"\nDataset ready at: {dataset_path}")
    print(f"  Train: {len(train_set)} images")
    print(f"  Val:   {val_count} images")
    print(f"  YAML:  {yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split Roboflow export into YOLO train/val format")
    parser.add_argument("--dataset", type=str, default="data/human_dataset",
                        help="Path to the dataset directory with images/ and labels/")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Fraction of data to use for validation (default: 0.2)")
    args = parser.parse_args()
    prepare_dataset(args.dataset, args.val_split)
