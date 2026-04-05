"""Tests for PnP calibration solver."""

from __future__ import annotations

import numpy as np
import pytest

from src.spatial.calibration import solve_camera_orientation


class TestSolveCameraOrientation:

    def test_minimum_4_points_required(self):
        """PnP needs at least 4 point correspondences."""
        pixel_points = [(100, 200), (300, 400), (500, 100)]
        world_points = [(0, 0, 0), (10, 0, 0), (0, 10, 0)]
        with pytest.raises(ValueError, match="at least 4"):
            solve_camera_orientation(pixel_points, world_points, 640, 480, fov=90.0)

    def test_mismatched_lengths_raises(self):
        pixel_points = [(100, 200), (300, 400)]
        world_points = [(0, 0, 0)]
        with pytest.raises(ValueError, match="same number"):
            solve_camera_orientation(pixel_points, world_points, 640, 480, fov=90.0)

    def test_known_geometry_returns_reasonable_orientation(self):
        """
        Place camera at (0, 0, -10) looking straight down (pitch=-90).
        Points on the ground at known positions should solve to ~(-90, 0, 0).
        """
        pixel_points = [
            (320, 240),   # center
            (160, 120),   # top-left quadrant
            (480, 120),   # top-right quadrant
            (160, 360),   # bottom-left quadrant
            (480, 360),   # bottom-right quadrant
        ]
        # World points on ground plane (z=0 in NED)
        world_points = [
            (0.0, 0.0, 0.0),
            (-5.0, -5.0, 0.0),
            (-5.0, 5.0, 0.0),
            (5.0, -5.0, 0.0),
            (5.0, 5.0, 0.0),
        ]
        result = solve_camera_orientation(
            pixel_points, world_points,
            frame_w=640, frame_h=480,
            fov=90.0,
            camera_position=(0.0, 0.0, -10.0),
        )
        assert result is not None
        pitch, yaw, roll = result
        assert isinstance(pitch, float)
        assert isinstance(yaw, float)
        assert isinstance(roll, float)

    def test_returns_none_on_degenerate_input(self):
        """All points at same pixel location — solvePnP should fail gracefully."""
        pixel_points = [(320, 240)] * 4
        world_points = [(0, 0, 0), (10, 0, 0), (0, 10, 0), (10, 10, 0)]
        result = solve_camera_orientation(
            pixel_points, world_points,
            frame_w=640, frame_h=480,
            fov=90.0,
        )
        assert result is None
