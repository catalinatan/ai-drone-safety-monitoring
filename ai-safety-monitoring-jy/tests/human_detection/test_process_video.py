import pytest
import numpy as np
from unittest.mock import MagicMock, patch, call
from src.human_detection.process_video import process_video


# We mock 'cv2' because we don't want to actually open windows or files
# We mock 'HumanDetector' because we already tested it separately
@patch('src.human_detection.process_video.cv2')
@patch('src.human_detection.process_video.HumanDetector')
def test_process_video_flow(mock_detector_class, mock_cv2):
    """
    Verifies that process_video:
    1. Opens the video capture
    2. initializes the VideoWriter
    3. Loops through frames
    4. Calls the detector
    5. Writes the output
    """

    # --- SETUP MOCKS ---

    # 1. Setup the Mock Detector to return a fake mask
    # This simulates finding one human
    mock_detector_instance = mock_detector_class.return_value
    fake_mask = np.zeros((100, 100), dtype=np.uint8)
    fake_mask[50:60, 50:60] = 1  # A small square representing a human
    mock_detector_instance.get_masks.return_value = [fake_mask]

    # 2. Setup Mock VideoCapture (The Input)
    mock_cap = MagicMock()
    # Configure it to return: (True, frame) once, then (False, None) to end loop
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_cap.read.side_effect = [(True, dummy_frame), (False, None)]

    # Get video properties (width, height, fps)
    mock_cap.get.side_effect = [100.0, 100.0, 30.0]  # Width, Height, FPS

    # Link our mock cap to cv2.VideoCapture
    mock_cv2.VideoCapture.return_value = mock_cap

    # 3. Setup Mock VideoWriter (The Output)
    mock_writer = MagicMock()
    mock_cv2.VideoWriter.return_value = mock_writer

    # --- RUN THE FUNCTION ---

    # We pass a fake path; thanks to mocks, it won't actually look for it
    # But we mock os.path.exists to return True so the script proceeds
    with patch('src.human_detection.process_video.os.path.exists', return_value=True):
        process_video("fake_video.mp4", "output_test.mp4")

    # --- ASSERTIONS (Did the plumbing work?) ---

    # 1. Did it try to open the video?
    mock_cv2.VideoCapture.assert_called_with("fake_video.mp4")

    # 2. Did it try to create a save file?
    mock_cv2.VideoWriter.assert_called_once()

    # 3. Did it process the frame?
    # It should have called get_masks exactly once (since we simulated 1 frame)
    mock_detector_instance.get_masks.assert_called_once_with(dummy_frame)

    # 4. Did it write the result to the output file?
    mock_writer.write.assert_called()

    # 5. Did it clean up?
    mock_cap.release.assert_called()
    mock_writer.release.assert_called()