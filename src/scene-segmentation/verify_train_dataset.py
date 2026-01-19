import cv2
import random
from pathlib import Path
from ultralytics import YOLO

def verify_dataset(dataset_name, num_samples=3):
    """
    Visualizes images and labels from the local dataset directory 
    to ensure masks are correctly aligned.
    """
    print(f"\n--- Verifying {dataset_name} ---")
    
    # Path to your validation images
    img_dir = Path(f"data/{dataset_name}/images/train")
    
    if not img_dir.exists():
        print(f"Error: Directory {img_dir} does not exist.")
        return

    img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    if not img_files:
        print(f"No images found in {img_dir}")
        return

    # Select random samples
    samples = random.sample(img_files, min(len(img_files), num_samples))
    
    # We load a base model just to use its 'plot' utilities
    # This won't show 'ship' names yet, but it will show the POLYGONS 
    # if the labels are in the right place.
    model = YOLO("yolo11n-seg.pt")

    for img_path in samples:
        print(f"Checking: {img_path.name}")
        
        # In YOLO, to 'verify' labels without training, we can't easily 
        # use model.predict(). Instead, we'll let YOLO 'see' the dataset 
        # via a temporary mini-training batch or just draw them manually.
        
        # FASTEST WAY: Use the Ultralytics Dataset validator 
        # to see if it can parse the labels.
        img = cv2.imread(str(img_path))
        h, w, _ = img.shape
        
        # Find corresponding label
        label_path = Path(str(img_path).replace("images", "labels").replace(".jpg", ".txt").replace(".png", ".txt"))
        
        if not label_path.exists():
            print(f"  [!] Missing label file for {img_path.name}")
            continue

        # Draw the polygons manually to verify alignment
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = list(map(float, line.strip().split()))
                class_id = int(parts[0])
                coords = parts[1:]
                
                # Rescale normalized coordinates to pixel values
                points = []
                for i in range(0, len(coords), 2):
                    points.append([int(coords[i] * w), int(coords[i+1] * h)])
                
                import numpy as np
                pts = np.array(points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.fillPoly(img, [pts], color=(0, 255, 0, 128))

        # Show the result
        cv2.imshow(f"Verify {dataset_name}: {img_path.name}", img)
        print("  Press any key for next, or 'q' to quit.")
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    verify_dataset("railway_dataset")
    verify_dataset("ship_dataset")
    verify_dataset("bridge_dataset")