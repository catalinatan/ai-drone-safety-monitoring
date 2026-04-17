"""Unit tests for HumanDetector — YOLO model mocked, no GPU needed."""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    """Ensure all tests run the CPU path (no TensorRT export)."""
    import src.detection.human_detector as hd
    monkeypatch.setattr(hd.torch.cuda, "is_available", lambda: False)


class TestHumanDetector:

    @patch("src.detection.human_detector.YOLO")
    def test_initialization_loads_model(self, mock_yolo):
        from src.detection.human_detector import HumanDetector
        detector = HumanDetector(model_path="dummy.pt")
        mock_yolo.assert_called_once_with("dummy.pt")

    @patch("src.detection.human_detector.YOLO")
    def test_get_masks_no_detections(self, mock_yolo):
        from src.detection.human_detector import HumanDetector

        mock_result = MagicMock()
        mock_result.masks = None
        mock_yolo.return_value.return_value = [mock_result]

        detector = HumanDetector(model_path="dummy.pt")
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        masks = detector.get_masks(frame)

        assert isinstance(masks, list)
        assert len(masks) == 0

    @patch("src.detection.human_detector.YOLO")
    def test_get_masks_person_detected(self, mock_yolo):
        import torch
        from src.detection.human_detector import HumanDetector, CLASS_ID_PERSON

        # Simulate one person detection
        mock_result = MagicMock()
        mock_box = MagicMock()
        mock_box.cls = [float(CLASS_ID_PERSON)]

        # masks.data must be a tensor so list-indexing (data[person_idx]) works
        mock_result.boxes = [mock_box]
        mock_result.masks.data = torch.ones((1, 10, 10), dtype=torch.float32)
        mock_yolo.return_value.return_value = [mock_result]

        detector = HumanDetector(model_path="dummy.pt")
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        masks = detector.get_masks(frame)

        assert len(masks) == 1
        assert masks[0].shape == (100, 100)
        assert masks[0].dtype == np.uint8

    @patch("src.detection.human_detector.YOLO")
    def test_get_masks_non_person_class_ignored(self, mock_yolo):
        import torch
        from src.detection.human_detector import HumanDetector

        mock_result = MagicMock()
        mock_box = MagicMock()
        mock_box.cls = [2.0]  # Not person (class 0)

        mock_result.boxes = [mock_box]
        mock_result.masks.data = torch.ones((1, 10, 10), dtype=torch.float32)
        mock_yolo.return_value.return_value = [mock_result]

        detector = HumanDetector(model_path="dummy.pt")
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        masks = detector.get_masks(frame)

        assert len(masks) == 0

    @patch("src.detection.human_detector.YOLO")
    def test_get_masks_batch_empty_input(self, mock_yolo):
        from src.detection.human_detector import HumanDetector

        mock_yolo.return_value.return_value = []
        detector = HumanDetector(model_path="dummy.pt")
        result = detector.get_masks_batch([])
        assert result == []

    @patch("src.detection.human_detector.YOLO")
    def test_get_masks_batch_returns_per_frame(self, mock_yolo):
        from src.detection.human_detector import HumanDetector

        # Two frames, no detections either
        mock_result = MagicMock()
        mock_result.masks = None
        mock_yolo.return_value.return_value = [mock_result, mock_result]

        detector = HumanDetector(model_path="dummy.pt")
        frames = [np.zeros((50, 50, 3), dtype=np.uint8)] * 2
        results = detector.get_masks_batch(frames)

        assert len(results) == 2
        assert all(isinstance(r, list) for r in results)
