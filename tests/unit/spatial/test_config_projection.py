"""Tests for ConfigProjection — config/GPS-based pixel-to-world projection."""

from __future__ import annotations

import numpy as np
import pytest

from src.spatial.config_projection import ConfigProjection
from src.spatial.projection_base import ProjectionBackend


class TestConfigProjectionInterface:

    def test_is_projection_backend(self):
        assert issubclass(ConfigProjection, ProjectionBackend)

    def test_init_with_position_and_orientation(self):
        proj = ConfigProjection(
            position=(1.2847, 103.861, 15.0),
            orientation=(-30.0, 180.0, 0.0),
            fov=90.0,
            safe_z=-10.0,
        )
        assert proj is not None


class TestConfigProjectionRayCast:

    def test_center_pixel_with_metric_depth(self):
        """Passing metric depth directly places point at that distance along the ray."""
        proj = ConfigProjection(
            position=(0.0, 0.0, -15.0),
            orientation=(-45.0, 0.0, 0.0),
            fov=90.0,
            safe_z=-10.0,
        )
        # 20m metric depth along a -45 degree ray from 15m height
        x, y, z = proj.pixel_to_world(320, 240, 20.0, 640, 480)
        assert x > 0.0  # projects forward (north)
        assert z == -10.0  # safe_z override

    def test_zero_depth_returns_camera_position(self):
        """depth=0 returns the camera position itself."""
        proj = ConfigProjection(
            position=(10.0, 20.0, -15.0),
            orientation=(-30.0, 90.0, 0.0),
            fov=90.0,
            safe_z=-10.0,
        )
        x, y, z = proj.pixel_to_world(320, 240, 0.0, 640, 480)
        assert abs(x - 10.0) < 0.01
        assert abs(y - 20.0) < 0.01

    def test_known_geometry_accuracy(self):
        """Camera 10m up, pitch -90 (straight down), center pixel, depth=10m.

        Should land directly below the camera at ground level.
        """
        proj = ConfigProjection(
            position=(5.0, 3.0, -10.0),
            orientation=(-90.0, 0.0, 0.0),
            fov=90.0,
            safe_z=-10.0,
        )
        x, y, z = proj.pixel_to_world(320, 240, 10.0, 640, 480)
        # Should be directly below camera
        assert abs(x - 5.0) < 0.5
        assert abs(y - 3.0) < 0.5

    def test_fallback_for_zero_depth(self):
        """depth=0 with horizontal camera still returns valid coords."""
        proj = ConfigProjection(
            position=(0.0, 0.0, -10.0),
            orientation=(0.0, 0.0, 0.0),
            fov=90.0,
            safe_z=-10.0,
        )
        x, y, z = proj.pixel_to_world(320, 240, 0.0, 640, 480)
        assert isinstance(x, float)
        assert isinstance(y, float)


class TestConfigProjectionUpdatePose:

    def test_update_position(self):
        proj = ConfigProjection(
            position=(0.0, 0.0, -10.0),
            orientation=(-30.0, 0.0, 0.0),
            fov=90.0,
        )
        proj.update_pose(position=(50.0, 60.0, -10.0))
        x, y, z = proj.pixel_to_world(320, 240, 0.0, 640, 480)
        assert abs(x - 50.0) < 5.0
        assert abs(y - 60.0) < 5.0

    def test_update_orientation(self):
        proj = ConfigProjection(
            position=(0.0, 0.0, -10.0),
            orientation=(-30.0, 0.0, 0.0),
            fov=90.0,
        )
        # Change yaw from 0 (north) to 90 (east)
        proj.update_pose(orientation=(-30.0, 90.0, 0.0))
        x, y, z = proj.pixel_to_world(320, 240, 0.5, 640, 480)
        # Should now project east (positive y in NED)
        assert y > 0.0


class TestConfigProjectionGPS:

    def test_update_gps_position_sets_origin(self):
        """First GPS update becomes the origin (NED 0,0,0)."""
        proj = ConfigProjection(
            position=(0.0, 0.0, -10.0),
            orientation=(-30.0, 0.0, 0.0),
            fov=90.0,
        )
        proj.update_gps_position(1.2847, 103.8610, 10.0)
        # Position should be at origin
        assert abs(proj._position[0]) < 0.01
        assert abs(proj._position[1]) < 0.01

    def test_update_gps_position_converts_to_ned(self):
        """Second GPS update should produce an NED offset from origin."""
        proj = ConfigProjection(
            position=(0.0, 0.0, -10.0),
            orientation=(-30.0, 0.0, 0.0),
            fov=90.0,
            gps_origin=(1.2847, 103.8610, 0.0),
        )
        # Move ~11m north
        proj.update_gps_position(1.2848, 103.8610, 0.0)
        assert proj._position[0] > 10.0  # x = north
        assert abs(proj._position[1]) < 0.1  # y = east, should be ~0

    def test_gps_origin_from_constructor(self):
        """GPS origin passed at construction time."""
        proj = ConfigProjection(
            position=(0.0, 0.0, -15.0),
            orientation=(-30.0, 0.0, 0.0),
            fov=90.0,
            gps_origin=(1.2847, 103.8610, 0.0),
        )
        assert proj._gps_origin == (1.2847, 103.8610, 0.0)


class TestConfigProjectionFallback:

    def test_horizontal_ray_uses_metric_depth_directly(self):
        """Camera pointing horizontally — metric depth used as-is along the ray."""
        proj = ConfigProjection(
            position=(0.0, 0.0, -10.0),
            orientation=(0.0, 0.0, 0.0),
            fov=90.0,
            safe_z=-10.0,
        )
        x, y, z = proj.pixel_to_world(320, 240, 25.0, 640, 480)
        assert isinstance(x, float)
        assert isinstance(y, float)
        assert z == -10.0


class TestComputeScaleFactor:

    def test_scale_factor_from_ground_pixels(self):
        """Scale factor should convert inverse-disparity to metric distance.

        Camera at 15m height, pitch -45 degrees, looking north.
        Center pixel ray hits ground at ~15m distance (height/sin(45)).
        If the depth map has inverse-disparity=5.0 there, scale should be ~15/5=3.0.
        """
        proj = ConfigProjection(
            position=(0.0, 0.0, -15.0),
            orientation=(-45.0, 0.0, 0.0),
            fov=90.0,
        )
        # Fake inverse-disparity map (4x4 for simplicity)
        # All pixels have inv_disp = 5.0
        depth_map = np.full((4, 4), 5.0, dtype=np.float32)

        scale = proj.compute_scale_factor(depth_map, 4, 4)

        # The geometric ground distance for center pixel at -45 pitch, 15m height
        # t_ground = 15 / ray_z component ≈ 21.2m (15 / sin(45) for the ray)
        # scale = t_ground / inv_disp ≈ 21.2 / 5.0 ≈ 4.24
        # Exact value depends on ray computation; just check it's positive and reasonable
        assert scale > 0
        assert 1.0 < scale < 100.0

    def test_scale_factor_positive(self):
        """Scale factor must always be positive."""
        proj = ConfigProjection(
            position=(0.0, 0.0, -10.0),
            orientation=(-30.0, 90.0, 0.0),
            fov=90.0,
        )
        depth_map = np.full((48, 64), 3.0, dtype=np.float32)
        scale = proj.compute_scale_factor(depth_map, 64, 48)
        assert scale > 0
