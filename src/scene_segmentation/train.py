import logging
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO
import yaml

# --- 1. Dataset Configurations ---
DATASET_CONFIGS = {
    "railway": {
        "path": "data/railway_dataset",
        "nc": 1,
        "names": ['danger-zone']
    },
    "ship": {
        "path": "data/ship_dataset",
        "nc": 1,
        "names": ['danger-zone']
    },
    "bridge": {
        "path": "data/bridge_dataset",
        "nc": 1,
        "names": ['danger-zone']
    }
}

# --- 2. Setup Logging ---
def setup_logger(name):
    log_filename = f"log_{name}.log"
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # File Handler
    fh = logging.FileHandler(log_filename)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)
    
    # Console Handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(ch)
    return logger

# --- 3. Modular Functions ---
def prepare_dataset_yaml(dataset_type, logger):
    """Generates data.yaml dynamically based on selection"""
    config = DATASET_CONFIGS.get(dataset_type)
    if not config:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    logger.info(f"Preparing YAML for {dataset_type} dataset...")
    dataset_path = Path(config['path'])
    
    # Validation
    for split in ['train', 'val']:
        img_dir = dataset_path / 'images' / split
        if not img_dir.exists():
            logger.error(f"Directory missing: {img_dir}")
            raise FileNotFoundError(f"Missing {split} images for {dataset_type}")

    data_yaml = {
        'path': str(dataset_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': config['nc'],
        'names': config['names']
    }
    
    yaml_path = str(dataset_path / "data.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f)
    
    logger.info(f"Created {yaml_path}")
    return yaml_path

def train_model(
    dataset_type,      # 'railway', 'ship', or 'bridge'
    model_name='yolo11n-seg',
    epochs=50,
    imgsz=640,
    batch_size=-1,
    device=0,
    patience=20,
    data_yaml_path='data.yaml'
):
    """
    Modular training function for multiple dataset types.
    """
    
    project_name = 'runs/segment'
    experiment_name = f'{dataset_type}_hazard'
    
    logger.info(f"="*30)
    logger.info(f"Training Task: {dataset_type.upper()}")
    logger.info(f"Configuration:")
    logger.info(f"  Model: {model_name} | Epochs: {epochs}")
    logger.info(f"  Image Size: {imgsz} | Device: {device}")
    logger.info(f"  Exp Name: {experiment_name}")
    logger.info(f"="*30)

    if not Path(data_yaml_path).exists():
        logger.error(f"Configuration file {data_yaml_path} missing!")
        raise FileNotFoundError(f"Missing: {data_yaml_path}")

    try:
        logger.info(f"Loading weights: {model_name}.pt")
        model = YOLO(f'{model_name}.pt')
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    logger.info(f"Starting {dataset_type} training session...")
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
            augment=True,
            flipud=0.5,
            fliplr=0.5,
            degrees=10.0,
            translate=0.1,
            scale=0.1,
        )
        
        logger.info(f"SUCCESS: {dataset_type} training complete.")
        logger.info(f"Best Weights: {project_name}/{experiment_name}/weights/best.pt")
        return results
        
    except Exception as e:
        logger.error(f"CRITICAL: {dataset_type} training failed: {e}")
        raise

# --- 4. Main Entry Point ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Modular YOLO11 Training")
    
    # Only two arguments are now parsed from the terminal
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['railway', 'ship', 'bridge'],
                        help="Which dataset to train on")
    
    parser.add_argument('--epochs', type=int, default=50, 
                        help="Number of training epochs")
    
    args = parser.parse_args()
    
    # Setup Logger based on the selected dataset name
    logger = setup_logger(args.dataset)

    try:
        # Step 1: Prepare the YAML (Dynamic based on dataset choice)
        yaml_file = prepare_dataset_yaml(args.dataset, logger)
        
        # Step 2: Train the Model 
        # Hardcoded imgsz=640 and patience=20 to keep things simple
        logger.info(f"\nStarting YOLO11n-seg training for: {args.dataset}...")
        results = train_model(
            dataset_type=args.dataset,
            data_yaml_path=yaml_file,
            epochs=args.epochs,   # Taken from command line
            imgsz=640,            # Hardcoded default
            patience=20,          # Hardcoded default
            device=0,             # Defaulting to GPU 0
            batch_size=-1         # Keep AutoBatch for performance
        )
        
        logger.info(f"Training for {args.dataset} successful!")
        
    except Exception as e:
        logger.error(f"Training failed for {args.dataset} with error: {e}")
        # Re-raise to see the full traceback in the terminal/log
        raise