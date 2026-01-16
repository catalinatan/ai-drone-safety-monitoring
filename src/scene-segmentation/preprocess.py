import os
import cv2
import numpy as np
import json
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

# Load config
with open('data/railsem19/rs19-config.json', 'r') as f:
    config = json.load(f)

labels = config['labels']
class_names = [label['name'] for label in labels]
class_colors = [label['color'] for label in labels]

# For YOLO, class id is index
class_id_map = {name: i for i, name in enumerate(class_names)}
color_to_id = {tuple(color): i for i, color in enumerate(class_colors)}

# Hazardous classes: rail-track, rail-raised, rail-embedded, tram-track, trackbed, on-rails
hazardous_classes = [3, 12, 15, 16, 17, 18]

# Function to convert mask to YOLO polygons
def mask_to_yolo(mask_path, img_width, img_height):
    mask = cv2.imread(mask_path)[:,:,0]  # Take one channel, since grayscale
    yolo_lines = []
    
    for class_id in hazardous_classes:
        # Create binary mask for this class
        class_mask = (mask == class_id).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if len(contour) >= 3:  # At least 3 points for polygon
                # Approximate polygon
                epsilon = 0.01 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Normalize coordinates
                points = []
                for point in approx:
                    x, y = point[0]
                    points.extend([x / img_width, y / img_height])
                
                if points:
                    # Map to 0-based for YOLO, since only hazardous
                    yolo_class_id = hazardous_classes.index(class_id)
                    line = f"{yolo_class_id} " + " ".join(map(str, points))
                    yolo_lines.append(line)
    
    return yolo_lines

# Process val set
jpg_dir = Path('data/railway_images')
mask_dir = Path('data/railsem19/uint8/rs19_val')
output_img_dir = Path('data/railway_images')
output_label_dir = Path('data/railway_labels')

output_img_dir.mkdir(exist_ok=True)
output_label_dir.mkdir(exist_ok=True)

for jpg_file in jpg_dir.glob('*.jpg'):
    mask_file = mask_dir / (jpg_file.stem + '.png')
    if mask_file.exists():
        # Get image size
        img = cv2.imread(str(jpg_file))
        img_height, img_width = img.shape[:2]
        
        # Convert mask
        yolo_lines = mask_to_yolo(str(mask_file), img_width, img_height)
        
        # Save txt
        txt_file = output_label_dir / (jpg_file.stem + '.txt')
        with open(txt_file, 'w') as f:
            f.write('\n'.join(yolo_lines))
        
print("Conversion complete!")

# Split dataset into train and val
def split_dataset(image_dir='data/railway_images', label_dir='data/railway_labels', train_ratio=0.8):
    """
    Split dataset into train and val sets using YOLO structure:
    data/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
    """
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    
    # Create YOLO-style directory structure
    base_dir = Path('data/railway_dataset')
    train_img_dir = base_dir / 'images' / 'train'
    train_lbl_dir = base_dir / 'labels' / 'train'
    val_img_dir = base_dir / 'images' / 'val'
    val_lbl_dir = base_dir / 'labels' / 'val'
    
    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = list(image_dir.glob('*.jpg'))
    
    # Split into train/val
    train_files, val_files = train_test_split(
        image_files, 
        train_size=train_ratio, 
        random_state=42
    )
    
    # Copy train files
    for img_file in train_files:
        lbl_file = label_dir / (img_file.stem + '.txt')
        shutil.copy(img_file, train_img_dir / img_file.name)
        if lbl_file.exists():
            shutil.copy(lbl_file, train_lbl_dir / lbl_file.name)
    
    # Copy val files
    for img_file in val_files:
        lbl_file = label_dir / (img_file.stem + '.txt')
        shutil.copy(img_file, val_img_dir / img_file.name)
        if lbl_file.exists():
            shutil.copy(lbl_file, val_lbl_dir / lbl_file.name)
    
    print(f"\nDataset split complete (YOLO structure):")
    print(f"  Train: {len(train_files)} images")
    print(f"  Val: {len(val_files)} images")
    print(f"  Location: data/railway_dataset/")

# Perform the split
split_dataset(train_ratio=0.8)