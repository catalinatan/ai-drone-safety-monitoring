import cv2
from pathlib import Path
import random
from ultralytics import YOLO
import os

def visualize_results(model_path, image_dir, num_samples=5, conf=0.5):
    """
    Visualize predictions using the built-in Ultralytics plotting tools.
    """
    # 1. Validation checks
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        return

    # 2. Load Model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # 3. Get Images
    img_path = Path(image_dir)
    img_files = list(img_path.glob('*.jpg')) + list(img_path.glob('*.png'))
    
    if not img_files:
        print(f"ERROR: No images found in {image_dir}")
        return
    
    samples = random.sample(img_files, min(len(img_files), num_samples))
    
    print(f"Processing {len(samples)} samples at confidence {conf}...")
    print("TIP: Press any key to see the next image. Press 'q' to quit.")

    for i, img_file in enumerate(samples):
        # 4. Run Inference
        # result is a list, we take the first element [0]
        results = model.predict(source=str(img_file), conf=conf, save=False, exist_ok=True)[0]
        
        # 5. Use the built-in plot() method
        # This handles the mask overlay, labels, and boxes automatically
        annotated_frame = results.plot(
            conf=True, 
            line_width=2, 
            font_size=1,
            labels=True
        )
        
        # 6. Display using OpenCV
        window_name = f"Sample {i+1}: {img_file.name}"
        cv2.imshow(window_name, annotated_frame)
        
        # Wait for key press
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow(window_name)
        
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Visualization complete.")

if __name__ == "__main__":
    # ADJUST THESE PATHS AS NEEDED
    DATASET_CONFIGS = {
    "railway": {
        "model": "runs/segment/runs/segment/railway_hazard/weights/best.pt",
        "images": "data/railway_dataset/images/val"
    },
    "ship": {
        "model": "runs/segment/runs/segment/ship_hazard/weights/best.pt",
        "images": "data/ship_dataset/images/val"
    },
    "bridge": {
        "model": "runs/segment/runs/segment/bridge_hazard/weights/best.pt",
        "images": "data/bridge_dataset/images/val"
    }
}
    
    visualize_results(
        model_path=DATASET_CONFIGS["railway"]["model"],
        image_dir=DATASET_CONFIGS["railway"]["images"],
        num_samples=3,
        conf=0.5
    )
    visualize_results(
        model_path=DATASET_CONFIGS["bridge"]["model"],
        image_dir=DATASET_CONFIGS["bridge"]["images"],
        num_samples=3,
        conf=0.5
    )
    visualize_results(
        model_path=DATASET_CONFIGS["ship"]["model"],
        image_dir=DATASET_CONFIGS["ship"]["images"],
        num_samples=3,
        conf=0.5
    )