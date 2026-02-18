"""
Fine-tune YOLOv8m-seg for human detection in water/aerial scenarios.
Modeled after src/scene-segmentation/train.py

Usage:
    python -m src.human_detection.train --epochs 100
    python -m src.human_detection.train --epochs 100 --imgsz 1280 --model yolov8m-seg
    python -m src.human_detection.train --epochs 100 --model yolov8l-seg --freeze 0
"""
import logging
import argparse
from pathlib import Path
from ultralytics import YOLO
import yaml


# --- Configuration ---
DATASET_CONFIG = {
    "path": "data/human_dataset",
    "nc": 1,
    "names": ["person"],
}


def setup_logger():
    log_filename = "log_human_detection_train.log"
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_filename)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(ch)
    return logger


def prepare_dataset_yaml(logger):
    """Generate data.yaml for the human detection dataset."""
    dataset_path = Path(DATASET_CONFIG["path"])

    for split in ["train", "val"]:
        img_dir = dataset_path / "images" / split
        if not img_dir.exists():
            logger.error(f"Directory missing: {img_dir}")
            raise FileNotFoundError(f"Missing {split} images at {img_dir}")

    data_yaml = {
        "path": str(dataset_path.absolute()),
        "train": "images/train",
        "val": "images/val",
        "nc": DATASET_CONFIG["nc"],
        "names": DATASET_CONFIG["names"],
    }

    yaml_path = "data_human.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f)

    logger.info(f"Created {yaml_path}")
    return yaml_path


def train_model(
    model_name="yolov8m-seg",
    epochs=100,
    imgsz=1280,
    batch_size=-1,
    device=0,
    patience=20,
    data_yaml_path="data_human.yaml",
    freeze=10,
    logger=None,
):
    """
    Fine-tune YOLO segmentation model for human detection.

    Args:
        model_name: Base model name (e.g. yolov8m-seg, yolov8l-seg).
        epochs: Number of training epochs.
        imgsz: Training image size (should match inference size).
        batch_size: Batch size (-1 for auto).
        device: Training device (0 for GPU, 'cpu', or 'mps').
        patience: Early stopping patience.
        data_yaml_path: Path to dataset YAML config.
        freeze: Number of backbone layers to freeze (10 = freeze backbone only,
                reduces overfitting with small datasets). Set to 0 for full training.
        logger: Logger instance.
    """
    project_name = "runs/segment"
    experiment_name = "human_detection"

    logger.info("=" * 40)
    logger.info("HUMAN DETECTION FINE-TUNING")
    logger.info(f"  Base Model: {model_name}")
    logger.info(f"  Epochs: {epochs} | Image Size: {imgsz}")
    logger.info(f"  Freeze Layers: {freeze} | Device: {device}")
    logger.info("=" * 40)

    if not Path(data_yaml_path).exists():
        logger.error(f"Configuration file {data_yaml_path} missing!")
        raise FileNotFoundError(f"Missing: {data_yaml_path}")

    try:
        logger.info(f"Loading weights: {model_name}.pt")
        model = YOLO(f"{model_name}.pt")
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    logger.info("Starting training...")
    try:
        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            patience=patience,
            save=True,
            project=project_name,
            name=experiment_name,
            exist_ok=True,
            verbose=True,

            # Backbone freezing (critical for small datasets)
            freeze=freeze,

            # Augmentations tuned for aerial water scenarios
            augment=True,
            flipud=0.5,        # Vertical flip (aerial views have no fixed up)
            fliplr=0.5,        # Horizontal flip
            degrees=15.0,      # Rotation (drone orientation varies)
            translate=0.2,     # Translation (humans can appear anywhere in frame)
            scale=0.5,         # Scale variation (critical: humans range from tiny to medium)
            mosaic=1.0,        # Mosaic augmentation (combines 4 images, great for small objects)
            mixup=0.1,         # Light image blending for generalization
            hsv_h=0.015,       # Hue shift (water color variation)
            hsv_s=0.7,         # Saturation shift (water reflections, lighting)
            hsv_v=0.4,         # Brightness shift (sunlight, shadows on water)
            erasing=0.1,       # Random erasing (partial occlusion by waves)
        )

        logger.info("Training complete!")
        best_weights = f"{project_name}/{experiment_name}/weights/best.pt"
        logger.info(f"Best Weights: {best_weights}")
        logger.info("")
        logger.info("To use the fine-tuned model, update src/human_detection/config.py:")
        logger.info(f'  MODEL_PATH = "{best_weights}"')

        return results

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLO for human detection in water")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="Training image size (default: 1280)")
    parser.add_argument("--model", type=str, default="yolov8m-seg",
                        help="Base model: yolov8m-seg, yolov8l-seg, or yolov8x-seg")
    parser.add_argument("--freeze", type=int, default=10,
                        help="Number of backbone layers to freeze (0=train all, 10=freeze backbone)")
    parser.add_argument("--device", type=str, default="0",
                        help="Device: 0 (GPU), cpu, or mps")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience")

    args = parser.parse_args()

    logger = setup_logger()

    try:
        yaml_file = prepare_dataset_yaml(logger)
        train_model(
            model_name=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            data_yaml_path=yaml_file,
            freeze=args.freeze,
            device=args.device,
            patience=args.patience,
            logger=logger,
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
