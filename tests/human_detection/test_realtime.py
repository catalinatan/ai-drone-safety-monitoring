import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.human_detection.realtime import run_realtime_detection


@patch('src.human_detection.realtime.cv2')
@patch('src.human_detection.realtime.HumanDetector')
def test_realtime_smoke_test(mock_detector_class, mock_cv2):
    """
    Just checks if the function runs without crashing.
    Forcefully breaks the infinite loop after 1 iteration.
    """

    # --- 1. Setup Mock Camera ---
    mock_cap = MagicMock()

    # Create a dummy frame (Color image, 100x100 pixels)
    # We MUST provide a numpy array, not None, or np.zeros_like(frame) will fail later
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # Return (True, dummy_frame) when read() is called
    mock_cap.read.return_value = (True, dummy_frame)
    mock_cap.isOpened.return_value = True

    mock_cv2.VideoCapture.return_value = mock_cap

    # --- 2. Setup Mock Detector ---
    mock_detector_instance = mock_detector_class.return_value

    # CRITICAL FIX: Return an empty list [] (No humans found).
    # This prevents the code from entering the complex mask math section,
    # which avoids the "0-dimensional array" error completely.
    mock_detector_instance.get_masks.return_value = []

    # --- 3. Setup Exit Condition ---
    # Simulate pressing 'q' immediately to break the loop
    mock_cv2.waitKey.return_value = ord('q')

    # --- 4. Run it ---
    run_realtime_detection(source=0)

    # --- 5. Assertions ---
    # Did it try to open the camera?
    mock_cv2.VideoCapture.assert_called_with(0)

    # Did it try to get masks?
    mock_detector_instance.get_masks.assert_called_once_with(dummy_frame)

    # Did it clean up?
    mock_cap.release.assert_called()
    mock_cv2.destroyAllWindows.assert_called()