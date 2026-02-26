"""
Prepare a Roboflow-exported dataset into YOLO train/val split format.

Expects a flat Roboflow export at the dataset directory:
  data/<dataset>/
    images/   (all images in one folder)
    labels/   (all labels in one folder)

Reorganizes into:
  data/<dataset>/
    images/train/  images/val/
    labels/train/  labels/val/
    data.yaml

Works with any dataset — auto-detects class info from an existing data.yaml
(Roboflow includes one), or accepts --nc and --names overrides.

Usage:
    python -m src.human_detection.prepare_dataset --dataset data/human_dataset
    python -m src.human_detection.prepare_dataset --dataset data/bridge_dataset --val-split 0.2
    python -m src.human_detection.prepare_dataset --dataset data/my_dataset --nc 3 --names "cat,dog,bird"
"""
import argparse
import random
import shutil
from pathlib import Path
import yaml


def prepare_dataset(dataset_dir, val_split=0.2, seed=42, nc=None, names=None):
    """
    Split a flat images/labels directory into train/val sets in YOLO format.
    Moves files from the flat structure into train/val subdirectories.

    Class info (nc, names) is resolved in this order:
      1. Explicit nc/names arguments
      2. Existing data.yaml in the dataset directory
      3. Fallback: nc=1, names=["object"]
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

    # Resolve class info
    if nc is not None and names is not None:
        class_nc = nc
        class_names = names
    else:
        # Try reading existing data.yaml (Roboflow exports include one)
        existing_yaml = dataset_path / "data.yaml"
        if existing_yaml.exists():
            with open(existing_yaml) as f:
                existing = yaml.safe_load(f)
            class_nc = existing.get("nc", 1)
            class_names = existing.get("names", ["object"])
            print(f"Read class info from {existing_yaml}: nc={class_nc}, names={class_names}")
        else:
            class_nc = 1
            class_names = ["object"]
            print(f"Warning: No data.yaml found and no --nc/--names provided. Using defaults: nc={class_nc}, names={class_names}")

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
        "nc": class_nc,
        "names": class_names,
    }
    yaml_path = dataset_path / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f)

    print(f"\nDataset ready at: {dataset_path}")
    print(f"  Train: {len(train_set)} images")
    print(f"  Val:   {val_count} images")
    print(f"  Classes: {class_nc} — {class_names}")
    print(f"  YAML:  {yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split Roboflow export into YOLO train/val format")
    parser.add_argument("--dataset", type=str, default="data/human_dataset",
                        help="Path to the dataset directory with images/ and labels/")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Fraction of data to use for validation (default: 0.2)")
    parser.add_argument("--nc", type=int, default=None,
                        help="Number of classes (auto-detected from data.yaml if not provided)")
    parser.add_argument("--names", type=str, default=None,
                        help="Comma-separated class names (e.g. 'person' or 'cat,dog,bird')")
    args = parser.parse_args()

    # Parse comma-separated names if provided
    parsed_names = args.names.split(",") if args.names else None

    prepare_dataset(args.dataset, args.val_split, nc=args.nc, names=parsed_names)
