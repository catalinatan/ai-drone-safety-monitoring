"""
Fine-tune YOLO segmentation for human detection in water/aerial scenarios.
Supports separate sim/real model variants and different base model sizes.

Output weights are saved to:
    runs/segment/runs/segment/human_detection_{variant}_{model}/weights/best.pt

Usage:
    # Train on combined dataset (default: yolo11n-seg, 100 epochs)
    python -m src.human_detection.train --epochs 100

    # Train sim-only variant with yolo11n
    python -m src.human_detection.train --variant sim --epochs 100

    # Train sim-only variant with yolo11s (heavier, more accurate)
    python -m src.human_detection.train --variant sim --model yolo11s-seg --epochs 100

    # Train real-only variant
    python -m src.human_detection.train --variant real --epochs 100

    # Custom dataset path (overrides the variant's default dataset)
    python -m src.human_detection.train --variant sim --dataset data/human_dataset_sim --epochs 100

    # Full control over training
    python -m src.human_detection.train --variant sim --model yolo11s-seg --epochs 200 --patience 30 --freeze 0

Args:
    --variant       sim | real | combined (default: combined)
                    Determines dataset and output folder name.
    --dataset       Override dataset path (default: auto-resolved from variant)
    --model         Base YOLO model to finetune (default: yolo11n-seg)
                    Options: yolo11n-seg (fast), yolo11s-seg (balanced)
    --epochs        Training epochs (default: 100)
    --patience      Early stopping — stops if no improvement for N epochs (default: 20)
    --imgsz         Training image size in pixels (default: 1280)
    --freeze        Backbone layers to freeze (default: 10, set 0 to train all layers)
    --device        Training device (default: 0 for GPU, or 'cpu')
"""
import logging
import argparse
from pathlib import Path
from ultralytics import YOLO
import yaml


# --- Configuration ---
VARIANT_DATASETS = {
    "sim": "data/human_dataset_sim",
    "real": "data/human_dataset_real",
    "combined": "data/human_dataset",
}

DATASET_CONFIG = {
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


def prepare_dataset_yaml(dataset_path, logger):
    """Generate data.yaml for the human detection dataset."""
    dataset_path = Path(dataset_path)

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

    yaml_path = str(dataset_path / "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f)

    logger.info(f"Created {yaml_path}")
    return yaml_path


def train_model(
    model_name="yolo11n-seg",
    epochs=100,
    imgsz=1280,
    batch_size=-1,
    device=0,
    patience=20,
    data_yaml_path="data/human_dataset/data.yaml",
    freeze=10,
    variant="combined",
    logger=None,
):
    """
    Fine-tune YOLO segmentation model for human detection.

    Args:
        model_name: Base model name (e.g. yolo11n-seg, yolo11s-seg).
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
    # e.g. human_detection_sim_yolo11n-seg, human_detection_real_yolo11s-seg
    variant_suffix = f"_{variant}" if variant != "combined" else ""
    experiment_name = f"human_detection{variant_suffix}_{model_name}"

    logger.info("=" * 40)
    logger.info(f"HUMAN DETECTION FINE-TUNING [{variant.upper()}]")
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

            amp=True,             # Mixed precision (FP16) — halves GPU memory for activations
            cache=True,           # Cache images in RAM — faster epochs, no repeated disk I/O

            # Backbone freezing (critical for small datasets)
            freeze=freeze,

            # Minimal augmentation — pretrained backbone already handles variation;
            # only horizontal flip to double effective data without distortion
            augment=True,
            fliplr=0.5,
            flipud=0.0,
            degrees=0.0,
            translate=0.0,
            scale=0.0,
            mosaic=0.0,
            mixup=0.0,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
            erasing=0.0,
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
    parser.add_argument("--variant", type=str, default="combined",
                        choices=["sim", "real", "combined"],
                        help="Model variant: sim, real, or combined (default: combined)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Override dataset path (default: auto from variant)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="Training image size (default: 1280)")
    parser.add_argument("--model", type=str, default="yolo11n-seg",
                        help="Base model: yolo11n-seg, yolov8l-seg, or yolov8x-seg")
    parser.add_argument("--freeze", type=int, default=10,
                        help="Number of backbone layers to freeze (0=train all, 10=freeze backbone)")
    parser.add_argument("--device", type=str, default="0",
                        help="Device: 0 (GPU), cpu, or mps")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience")

    args = parser.parse_args()

    # Resolve dataset path from variant or explicit override
    dataset_path = args.dataset or VARIANT_DATASETS[args.variant]

    logger = setup_logger()

    try:
        yaml_file = prepare_dataset_yaml(dataset_path, logger)
        train_model(
            model_name=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            data_yaml_path=yaml_file,
            freeze=args.freeze,
            device=args.device,
            patience=args.patience,
            variant=args.variant,
            logger=logger,
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
