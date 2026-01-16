import cv2
import numpy as np
from pathlib import Path
import random

def visualize_hazardous_zones(image_dir, label_dir, num_samples=5):
    img_files = list(Path(image_dir).glob('*.jpg'))
    samples = random.sample(img_files, min(len(img_files), num_samples))
    
    # Define a single color for all hazardous areas (BGR format: Red is (0, 0, 255))
    HAZARD_COLOR = (0, 0, 255) 

    for img_path in samples:
        txt_path = Path(label_dir) / (img_path.stem + '.txt')
        if not txt_path.exists():
            print(f"No label found for {img_path.name}")
            continue

        img = cv2.imread(str(img_path))
        if img is None: continue
        
        h, w = img.shape[:2]
        overlay = img.copy()

        with open(txt_path, 'r') as f:
            for line in f.readlines():
                parts = line.split()
                if not parts: continue
                
                # We ignore class_id and treat everything in this file as 'Hazard'
                coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
                
                # Denormalize
                points = (coords * [w, h]).astype(np.int32)
                
                # Draw the polygon in RED
                cv2.fillPoly(overlay, [points], HAZARD_COLOR)
                cv2.polylines(img, [points], True, HAZARD_COLOR, 2)

        # Blend the red overlay (alpha 0.4 = 40% transparency)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        # Add a "Hazard Warning" text label on the image
        cv2.putText(img, "DANGER: HAZARD ZONE", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow(f"Hazard Visualization: {img_path.name}", img)
        print(f"Showing {img_path.name}. Press any key for next...")
        cv2.waitKey(0) 

    cv2.destroyAllWindows()

# Run it
visualize_hazardous_zones('data/railway_images', 'data/railway_labels', num_samples=3)