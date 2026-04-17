import pytest
import numpy as np
import cv2
import os
from src.detection.human_detector import HumanDetector
from src.core.config import get_config as _get_config
MODEL_PATH = _get_config().get("detection", {}).get("model_path", "yolo11n-seg.pt")


# Marked as integration so it can be easily skipped if we want since this test may take longer
@pytest.mark.integration
class TestRealSystem:

    def test_real_model_loading(self):
        """
        Does the real YOLOv8 model file actually load without crashing?
        (This will trigger a download of the .pt file if it's missing)
        """
        print("\n[Integration] Loading real YOLO model... (this may take time)")
        detector = HumanDetector()

        # Check if the model object was actually created
        assert detector.model is not None

        # Verify it loaded the correct weights file
        # (YOLO objects usually store their name in .ckpt_path or .overrides)
        assert detector.model.ckpt_path is not None

    def test_inference_on_blank_image(self):
        """
        Pass a blank black image to the REAL model.
        It should run successfully and return an empty list (0 humans).
        """
        detector = HumanDetector()

        # Create a real 640x640 black image (standard YOLO size)
        black_frame = np.zeros((640, 640, 3), dtype=np.uint8)

        print("\n[Integration] Running inference on blank image...")
        masks = detector.get_masks(black_frame)

        # 1. It should not crash
        assert isinstance(masks, list)

        # 2. It should find NOTHING in a black void
        assert len(masks) == 0

    def test_inference_on_real_human_image(self):
        """
        Requires a file named 'tests/data/person.jpg' to exist.
        If the file exists, it proves the model can actually find a human.
        If not, it skips.
        """
        image_path = "data/human_images/suggested-hhFaD71gfpD4xdNKKULr_jpg.rf.55019e75fd6eee006c2f06d896b5bc35.jpg"

        if not os.path.exists(image_path):
            pytest.skip(f"Skipping: No test image found at {image_path}")

        # Load the image
        frame = cv2.imread(image_path)
        assert frame is not None, "Failed to read test image"

        # Run Detector
        detector = HumanDetector()
        masks = detector.get_masks(frame)

        # Assertions
        assert len(masks) > 0, "Real Model failed to detect a human in the test image!"

        # Check the mask shape matches the image
        h, w = frame.shape[:2]
        assert masks[0].shape == (h, w)

        # Check it is binary
        unique_values = np.unique(masks[0])
        # Should contain only 0s and 1s
        assert np.all(np.isin(unique_values, [0, 1]))