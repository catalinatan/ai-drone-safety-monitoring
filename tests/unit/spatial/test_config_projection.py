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

    def test_center_pixel_projects_along_camera_forward(self):
        """Center pixel with depth=0.5 returns a point in front of camera."""
        proj = ConfigProjection(
            position=(0.0, 0.0, -15.0),   # 15m above ground (NED)
            orientation=(-45.0, 0.0, 0.0), # pitch=-45 (looking down at 45 deg), yaw=0 (north)
            fov=90.0,
            safe_z=-10.0,
        )
        x, y, z = proj.pixel_to_world(320, 240, 0.5, 640, 480)
        # Should be somewhere in front of camera (positive x in NED for yaw=0)
        assert x > 0.0
        assert z == -10.0  # safe_z override

    def test_zero_depth_returns_near_camera(self):
        """depth=0 returns point very close to camera position."""
        proj = ConfigProjection(
            position=(10.0, 20.0, -15.0),
            orientation=(-30.0, 90.0, 0.0),
            fov=90.0,
            safe_z=-10.0,
        )
        x, y, z = proj.pixel_to_world(320, 240, 0.0, 640, 480)
        # Near camera (min_distance clamp means depth=0 still moves a bit)
        assert abs(x - 10.0) < 10.0
        assert abs(y - 20.0) < 10.0

    def test_full_depth_reaches_ground_plane(self):
        """depth=1.0 projects to the ground plane intersection."""
        proj = ConfigProjection(
            position=(0.0, 0.0, -20.0),
            orientation=(-45.0, 0.0, 0.0),
            fov=90.0,
            safe_z=-10.0,
        )
        x, y, z = proj.pixel_to_world(320, 240, 1.0, 640, 480)
        # At full depth, ground intersection should be ~20m away for 45 deg pitch at 20m height
        assert x > 10.0


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

    def test_horizontal_ray_uses_height_fallback(self):
        """Camera pointing horizontally (pitch=0) can't hit ground — uses height-based distance."""
        proj = ConfigProjection(
            position=(0.0, 0.0, -10.0),
            orientation=(0.0, 0.0, 0.0),  # looking straight ahead
            fov=90.0,
            safe_z=-10.0,
        )
        x, y, z = proj.pixel_to_world(320, 240, 0.5, 640, 480)
        # Should still produce valid coordinates
        assert isinstance(x, float)
        assert isinstance(y, float)
        assert z == -10.0
