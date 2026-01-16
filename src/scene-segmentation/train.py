import logging
import sys
from pathlib import Path
from ultralytics import YOLO
import yaml
from sklearn.model_selection import train_test_split
from datetime import datetime

# Setup logging with custom name
custom_log_name = sys.argv[1] if len(sys.argv) > 1 else "railway_hazard"
log_filename = f"log_{custom_log_name}.log"
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Also add console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

logger.info("="*60)
logger.info("YOLO11n-seg Railway Hazard Detection Training Script")
logger.info("="*60)

def prepare_dataset_yaml():
    """Create data.yaml for YOLO training"""
    logger.info("Preparing dataset configuration...")
    
    # Check if train/val directories exist
    dataset_dir = Path('data/railway_dataset')
    train_img_dir = dataset_dir / 'images' / 'train'
    val_img_dir = dataset_dir / 'images' / 'val'
    train_lbl_dir = dataset_dir / 'labels' / 'train'
    val_lbl_dir = dataset_dir / 'labels' / 'val'
    
    if not train_img_dir.exists():
        logger.error(f"Train images directory not found: {train_img_dir}")
        raise FileNotFoundError(f"Train directory missing: {train_img_dir}")
    
    if not val_img_dir.exists():
        logger.error(f"Val images directory not found: {val_img_dir}")
        raise FileNotFoundError(f"Val directory missing: {val_img_dir}")
    
    train_count = len(list(train_img_dir.glob('*.jpg')))
    val_count = len(list(val_img_dir.glob('*.jpg')))
    
    logger.info(f"Found {train_count} training images")
    logger.info(f"Found {val_count} validation images")
    
    if train_count == 0 or val_count == 0:
        logger.error("No images found in train or val directories!")
        raise ValueError("Dataset is empty")
    
    data_yaml = {
        'path': str(dataset_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 6,
        'names': ['rail-track', 'rail-raised', 'rail-embedded', 'tram-track', 'trackbed', 'on-rails']
    }
    
    with open('data.yaml', 'w') as f:
        yaml.dump(data_yaml, f)
    
    logger.info("data.yaml created successfully")

def train_model(
    model_name='yolov11n-seg',
    epochs=100,
    imgsz=640,
    batch_size=16,
    device=0,
    patience=20,
    data_yaml_path='data.yaml'
):
    """
    Train YOLO11n-seg model on railway hazard data
    
    Args:
        model_name: YOLO model variant (yolov11n-seg, yolov11s-seg, etc.)
        epochs: Number of training epochs
        imgsz: Input image size
        batch_size: Batch size for training
        device: GPU device id (0 for first GPU, -1 for CPU)
        patience: Early stopping patience
        data_yaml_path: Path to data.yaml
    """
    logger.info(f"Training configuration:")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Image size: {imgsz}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Patience: {patience}")
    
    # Check if data.yaml exists
    if not Path(data_yaml_path).exists():
        logger.error(f"data.yaml not found at {data_yaml_path}")
        raise FileNotFoundError(f"data.yaml missing: {data_yaml_path}")
    
    logger.info(f"Loading model: {model_name}.pt")
    
    try:
        # Load the model
        model = YOLO(f'{model_name}.pt')
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    logger.info("Starting model training...")
    
    try:
        # Train the model
        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            patience=patience,
            save=True,
            project='runs/segment',
            name='railway_hazard',
            exist_ok=True,
            verbose=True,
            augment=True,
            flipud=0.5,
            fliplr=0.5,
            degrees=10,
            translate=0.1,
            scale=0.1,
        )
        logger.info("Training completed successfully")
        logger.info(f"Best model saved at: runs/segment/railway_hazard/weights/best.pt")
        logger.info(f"Last model saved at: runs/segment/railway_hazard/weights/last.pt")
        return results
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

def evaluate_model(model_path, data_yaml_path='data.yaml'):
    """Evaluate trained model"""
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml_path)
    return metrics

def predict_single(model_path, image_path):
    """Make prediction on a single image"""
    model = YOLO(model_path)
    results = model.predict(source=image_path, conf=0.5)
    return results

if __name__ == '__main__':
    try:
        # Prepare dataset configuration
        prepare_dataset_yaml()
        
        # Train the model
        logger.info("\nStarting YOLO11n-seg training...")
        results = train_model(
            model_name='yolov11n-seg',
            epochs=100,
            imgsz=640,
            batch_size=16,
            device=0,  # Change to -1 if training on CPU
            patience=20
        )
        
        logger.info("\n" + "="*60)
        logger.info("Training complete!")
        logger.info(f"Results saved to: runs/segment/railway_hazard")
        logger.info("\nModel files:")
        logger.info(f"  - Best model: runs/segment/railway_hazard/weights/best.pt")
        logger.info(f"  - Last model: runs/segment/railway_hazard/weights/last.pt")
        logger.info("\nTo use the model for prediction:")
        logger.info(f"  from ultralytics import YOLO")
        logger.info(f"  model = YOLO('runs/segment/railway_hazard/weights/best.pt')")
        logger.info(f"  results = model.predict(source='path/to/image.jpg')")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"\n{'='*60}")
        logger.error(f"Training failed with error: {e}")
        logger.error("="*60)
        raise
