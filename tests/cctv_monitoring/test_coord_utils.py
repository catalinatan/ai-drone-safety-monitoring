"""Unit tests for coord_utils.get_feet_from_mask — pure numpy, no AirSim needed."""

import numpy as np
from src.cctv_monitoring.coord_utils import get_feet_from_mask


class TestGetFeetFromMask:

    def test_empty_mask_returns_none(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        assert get_feet_from_mask(mask) is None

    def test_single_pixel(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[70, 50] = 1  # row=70, col=50
        result = get_feet_from_mask(mask)
        assert result == (50, 70)

    def test_vertical_person(self):
        """A tall, narrow mask — feet at the bottom."""
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[30:150, 90:110] = 1  # rows 30-149, cols 90-109
        cx, cy = get_feet_from_mask(mask)
        assert cy == 149  # bottommost row
        assert cx == 99   # center of cols 90-109

    def test_wide_mask_at_bottom(self):
        """A wide mask touching the bottom of the frame."""
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[460:480, 200:400] = 1
        cx, cy = get_feet_from_mask(mask)
        assert cy == 479  # last row
        assert cx == 299  # center of cols 200-399

    def test_irregular_shape(self):
        """Non-rectangular mask — feet should be at the widest bottom row."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        # Head (small, high)
        mask[10:20, 45:55] = 1
        # Body (wider, lower)
        mask[20:60, 40:60] = 1
        # Feet (widest, lowest)
        mask[60:70, 35:65] = 1
        cx, cy = get_feet_from_mask(mask)
        assert cy == 69  # bottommost
        assert cx == 49  # center of 35-64
