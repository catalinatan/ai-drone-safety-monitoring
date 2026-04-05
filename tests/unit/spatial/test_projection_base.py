"""Tests for ProjectionBackend interface contract."""

from __future__ import annotations

import pytest
from src.spatial.projection_base import ProjectionBackend


class TestProjectionBackendInterface:

    def test_cannot_instantiate_directly(self):
        """Base class is abstract — can't instantiate."""
        with pytest.raises(TypeError):
            ProjectionBackend()

    def test_subclass_must_implement_pixel_to_world(self):
        """Subclass missing pixel_to_world raises TypeError."""

        class Incomplete(ProjectionBackend):
            pass

        with pytest.raises(TypeError):
            Incomplete()

    def test_concrete_subclass_can_instantiate(self):
        """Subclass implementing all abstract methods works."""

        class Concrete(ProjectionBackend):
            def pixel_to_world(self, pixel_x, pixel_y, depth, frame_w, frame_h):
                return (0.0, 0.0, 0.0)

            def update_pose(self, position=None, orientation=None):
                pass

        proj = Concrete()
        assert proj.pixel_to_world(100, 200, 0.5, 640, 480) == (0.0, 0.0, 0.0)


class TestAirSimProjectionInterface:
    """Verify AirSimProjection satisfies the interface (no AirSim needed)."""

    def test_is_projection_backend(self):
        from src.spatial.airsim_projection import AirSimProjection
        assert issubclass(AirSimProjection, ProjectionBackend)

    def test_init_without_client_uses_fallback(self):
        from src.spatial.airsim_projection import AirSimProjection

        proj = AirSimProjection(
            fallback_position=(10.0, 20.0, -5.0),
            safe_z=-10.0,
        )
        # Without an AirSim client, pixel_to_world returns fallback
        result = proj.pixel_to_world(320, 240, 0.5, 640, 480)
        assert result == (10.0, 20.0, -10.0)

    def test_update_pose_updates_fallback(self):
        from src.spatial.airsim_projection import AirSimProjection

        proj = AirSimProjection(
            fallback_position=(0.0, 0.0, 0.0),
            safe_z=-10.0,
        )
        proj.update_pose(position=(5.0, 6.0, 7.0))
        result = proj.pixel_to_world(320, 240, 0.5, 640, 480)
        assert result == (5.0, 6.0, -10.0)
