from ultralytics import YOLO
import cv2
import numpy as np
from .config import MODEL_PATH, CONFIDENCE_THRESHOLD, CLASS_ID_PERSON


class HumanDetector:
    def __init__(self):
        # Load the model once when the class is initialized
        print(f"Loading YOLO model: {MODEL_PATH}...")
        self.model = YOLO(MODEL_PATH)

    def get_masks(self, frame):
        """
        Input: A single video frame (image).
        Output: A list of binary masks (numpy arrays), one for each human found.
        """
        results = self.model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        result = results[0]

        extracted_masks = []

        if result.masks is not None:
            for i, box in enumerate(result.boxes):
                # Filter: Only process if the detected object is a Person
                if int(box.cls[0]) == CLASS_ID_PERSON:
                    # 1. Get the raw mask (usually smaller resolution)
                    mask_raw = result.masks.data[i].cpu().numpy()

                    # 2. Resize mask to match the original video frame size
                    mask_resized = cv2.resize(mask_raw, (frame.shape[1], frame.shape[0]))

                    # 3. Convert to strict binary (0 or 1)
                    # Any pixel > 0.5 becomes 1 (Human), else 0 (Background)
                    mask_binary = (mask_resized > 0.5).astype(np.uint8)

                    extracted_masks.append(mask_binary)

        return extracted_masks