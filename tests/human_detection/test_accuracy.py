import pytest
import numpy as np
import cv2
import glob
import os
from src.detection.human_detector import HumanDetector


def calculate_iou(mask1, mask2):
    """Calculates Intersection over Union between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0: return 0
    return intersection / union


def parse_yolo_label(txt_path, img_width, img_height):
    """
    Reads a YOLO format .txt file and converts it to a binary mask.
    Format: class_id x1 y1 x2 y2 ... (normalized 0-1)
    """
    mask = np.zeros((img_height, img_width), dtype=np.uint8)

    if not os.path.exists(txt_path):
        return mask  # Empty mask if no file

    with open(txt_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = list(map(float, line.strip().split()))
        class_id = int(parts[0])

        # Only process class 0 (person)
        if class_id == 0:
            # Extract polygon points (skip first number which is class_id)
            coords = parts[1:]
            points = []
            for i in range(0, len(coords), 2):
                x = int(coords[i] * img_width)
                y = int(coords[i + 1] * img_height)
                points.append([x, y])

            # Draw this polygon onto the mask
            pts = np.array(points, np.int32)
            cv2.fillPoly(mask, [pts], 1)

    return mask


@pytest.mark.integration
def test_dataset_accuracy():
    """
    Loops through all images in 'tests/data/images', finds matching .txt in 'tests/data/labels',
    and checks if IoU > 0.5
    """
    image_paths = glob.glob("data/human_images/*.jpg")
    if not image_paths:
        pytest.skip("No test images found in data/human_images/")
    detector = HumanDetector()

    scores = []

    for img_path in image_paths:
        # 1. Load Image
        frame = cv2.imread(img_path)
        h, w = frame.shape[:2]

        # 2. Get AI Prediction
        ai_masks = detector.get_masks(frame)

        # Combine all AI masks into one "AI Prediction Layer"
        ai_total_mask = np.zeros((h, w), dtype=np.uint8)
        for m in ai_masks:
            ai_total_mask = np.maximum(ai_total_mask, m)

        # 3. Get Ground Truth (Manual Label)
        label_path = img_path.replace("human_images", "human_labels").replace(".jpg", ".txt")
        gt_mask = parse_yolo_label(label_path, w, h)

        # 4. Calculate Score
        iou = calculate_iou(ai_total_mask, gt_mask)
        scores.append(iou)
        print(f"Image: {img_path} -> IoU: {iou:.2f}")

    # Final Verdict
    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"\nAverage Dataset Accuracy: {avg_score * 100:.1f}%")

    # Fail if accuracy is too low (e.g., below 70%)
    assert avg_score > 0.7, f"Model accuracy {avg_score} is too low!"