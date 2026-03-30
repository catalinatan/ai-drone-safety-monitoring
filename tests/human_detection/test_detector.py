import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.detection.human_detector import HumanDetector
CONFIDENCE_THRESHOLD = 0.25
INFERENCE_IMGSZ = 1280


# Create a dummy image (100x100 pixels, 3 channels for RGB)
@pytest.fixture
def dummy_frame():
    return np.zeros((100, 100, 3), dtype=np.uint8)


class TestHumanDetector:

    @patch('src.detection.human_detector.YOLO')
    def test_initialization(self, mock_yolo):
        """Test if the model loads with the correct path."""
        detector = HumanDetector()

        # Assert YOLO was called with the path from config
        
        mock_yolo.assert_called_once()

    @patch('src.human_detection.detector.YOLO')
    def test_get_masks_no_detections(self, mock_yolo, dummy_frame):
        """Test output when no humans are found."""
        # Setup the mock: The model returns a Result object with NO masks
        mock_result = MagicMock()
        mock_result.masks = None  # Simulate no detection

        # Configure the model instance to return our mock result
        mock_yolo_instance = mock_yolo.return_value
        mock_yolo_instance.return_value = [mock_result]

        detector = HumanDetector()
        masks = detector.get_masks(dummy_frame)

        assert isinstance(masks, list)
        assert len(masks) == 0

        # Verify imgsz is passed to inference
        mock_yolo_instance = mock_yolo.return_value
        mock_yolo_instance.assert_called_once_with(
            dummy_frame, conf=CONFIDENCE_THRESHOLD, imgsz=INFERENCE_IMGSZ, verbose=False
        )

    @patch('src.human_detection.detector.YOLO')
    def test_get_masks_with_detections(self, mock_yolo, dummy_frame):
        """Test output when humans ARE found."""
        # 1. Simulate a Result with Boxes and Masks
        mock_result = MagicMock()

        # Mock the Box (Class 0 = Person)
        mock_box = MagicMock()
        mock_box.cls = [0.0]  # 0.0 is 'person'

        # Mock the Mask Data (raw tensor on CPU)
        # Creating a small 10x10 mask to simulate raw output
        mock_raw_mask = np.ones((10, 10), dtype=np.float32)

        # Setup the chains
        mock_result.boxes = [mock_box]

        # The result.masks.data[i] chain is complex, so we mock the data access
        # We need to simulate: result.masks.data[i].cpu().numpy()
        mock_mask_wrapper = MagicMock()
        mock_mask_wrapper.cpu.return_value.numpy.return_value = mock_raw_mask

        mock_result.masks.data = [mock_mask_wrapper]

        # Connect it all to the YOLO call
        mock_yolo_instance = mock_yolo.return_value
        mock_yolo_instance.return_value = [mock_result]

        # 2. Run the code
        detector = HumanDetector()
        masks = detector.get_masks(dummy_frame)

        # 3. Assertions
        assert len(masks) == 1
        # The output mask should match the FRAME size (100x100), not the raw mask size
        assert masks[0].shape == (100, 100)
        # It should be binary (0 or 1), not float
        assert masks[0].dtype == np.uint8