"""Unit tests for spatial.coord_utils — pure numpy, no AirSim needed."""

from __future__ import annotations

import numpy as np
import pytest

from src.spatial.coord_utils import get_feet_from_mask, unreal_to_airsim


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
        """Tall narrow mask — feet at the bottom."""
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[30:150, 90:110] = 1  # rows 30-149, cols 90-109
        cx, cy = get_feet_from_mask(mask)
        assert cy == 149        # bottommost row
        assert cx == 99         # center of cols 90-109

    def test_wide_mask_at_bottom(self):
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[460:480, 200:400] = 1
        cx, cy = get_feet_from_mask(mask)
        assert cy == 479        # last row
        assert cx == 299        # center of cols 200-399

    def test_irregular_shape(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:20, 45:55] = 1   # head
        mask[20:60, 40:60] = 1   # body
        mask[60:70, 35:65] = 1   # feet (widest, lowest)
        cx, cy = get_feet_from_mask(mask)
        assert cy == 69          # bottommost
        assert cx == 49          # center of 35-64


class TestUnrealToAirSim:

    def test_origin(self):
        result = unreal_to_airsim(0, 0, 0)
        assert result == {"X": 0.0, "Y": 0.0, "Z": 0.0}

    def test_z_flips_sign(self):
        # Z in Unreal is up; AirSim NED Z is down so it negates
        result = unreal_to_airsim(0, 0, 1000)  # 10 m up in Unreal
        assert result["Z"] == pytest.approx(-10.0)

    def test_scale_cm_to_m(self):
        result = unreal_to_airsim(100, 200, 0)
        assert result["X"] == pytest.approx(1.0)
        assert result["Y"] == pytest.approx(2.0)
