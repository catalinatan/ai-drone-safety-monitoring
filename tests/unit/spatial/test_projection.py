"""Tests for get_coords_from_lite_mono with metric depth."""

from unittest.mock import MagicMock
import math
import pytest


def _make_mock_client(cam_x, cam_y, cam_z, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
    """Create a mock AirSim client with a camera at the given pose."""
    client = MagicMock()
    info = MagicMock()
    info.pose.position.x_val = cam_x
    info.pose.position.y_val = cam_y
    info.pose.position.z_val = cam_z
    info.pose.orientation.x_val = qx
    info.pose.orientation.y_val = qy
    info.pose.orientation.z_val = qz
    info.pose.orientation.w_val = qw
    client.simGetCameraInfo.return_value = info
    return client


class TestGetCoordsFromLiteMonoMetricDepth:

    def test_zero_depth_returns_camera_position(self):
        """depth=0 should return the camera position."""
        from src.spatial.projection import get_coords_from_lite_mono
        client = _make_mock_client(10.0, 20.0, -15.0)

        result = get_coords_from_lite_mono(
            client, "0", 320, 240, 640, 480,
            ai_depth_val=0.0,
            cctv_height_meters=15.0,
        )
        assert abs(result.x_val - 10.0) < 0.01
        assert abs(result.y_val - 20.0) < 0.01

    def test_positive_depth_projects_forward(self):
        """Positive metric depth places point along the ray."""
        from src.spatial.projection import get_coords_from_lite_mono
        # Camera facing north (identity quaternion = looking along +X)
        client = _make_mock_client(0.0, 0.0, -10.0)

        result = get_coords_from_lite_mono(
            client, "0", 320, 240, 640, 480,
            ai_depth_val=25.0,
            cctv_height_meters=10.0,
        )
        # Should be in front of camera (positive x for identity orientation)
        assert result.x_val > 0.0
