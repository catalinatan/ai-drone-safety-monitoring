"""Unit tests for src.drone_server.drone_state."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.drone_server.drone_state import DroneState, ModeRequest, GotoRequest, MoveRequest, check_safety


class TestCheckSafety:
    def test_safe_target(self):
        ok, reason = check_safety((10.0, 20.0, -5.0), max_altitude=50.0)
        assert ok is True
        assert reason == "OK"

    def test_below_ground(self):
        ok, reason = check_safety((10.0, 20.0, 5.0), max_altitude=50.0)
        assert ok is False
        assert "below ground" in reason

    def test_exceeds_max_altitude(self):
        ok, reason = check_safety((10.0, 20.0, -100.0), max_altitude=50.0)
        assert ok is False
        assert "exceeds maximum" in reason

    def test_at_max_altitude(self):
        ok, _ = check_safety((0.0, 0.0, -50.0), max_altitude=50.0)
        assert ok is True


class TestPydanticModels:
    def test_mode_request(self):
        req = ModeRequest(mode="manual")
        assert req.mode == "manual"

    def test_goto_request_defaults(self):
        req = GotoRequest(x=1.0, y=2.0)
        assert req.z == -10.0

    def test_move_request_defaults(self):
        req = MoveRequest()
        assert req.vx == 0.0
        assert req.vy == 0.0
        assert req.vz == 0.0


class TestDroneState:
    def test_initial_state(self):
        ds = DroneState()
        assert ds.get_mode() == "automatic"
        assert ds.get_target() is None
        assert ds.get_should_stop() is False
        assert ds.is_grounded() is True

    def test_set_mode(self):
        ds = DroneState()
        ds.set_mode("manual")
        assert ds.get_mode() == "manual"

    def test_manual_mode_cancels_navigation(self):
        ds = DroneState()
        ds.set_target((1.0, 2.0, -5.0))
        assert ds.is_navigating is True
        ds.set_mode("manual")
        assert ds.is_navigating is False

    def test_set_target(self):
        ds = DroneState()
        ds.set_target((10.0, 20.0, -5.0))
        assert ds.get_target() == (10.0, 20.0, -5.0)
        assert ds.is_navigating is True
        assert ds.nav_command_sent is False

    def test_clear_target(self):
        ds = DroneState()
        ds.set_target((1.0, 2.0, -3.0))
        ds.clear_target()
        assert ds.get_target() is None
        assert ds.is_navigating is False

    def test_clear_target_cancels_nav_task(self):
        ds = DroneState()
        ds.set_target((1.0, 2.0, -3.0))
        mock_task = MagicMock()
        ds.nav_task = mock_task
        ds.clear_target()
        mock_task.cancel.assert_called_once()

    def test_get_nav_snapshot(self):
        ds = DroneState()
        ds.set_target((1.0, 2.0, -3.0))
        target, navigating, sent = ds.get_nav_snapshot()
        assert target == (1.0, 2.0, -3.0)
        assert navigating is True
        assert sent is False

    def test_try_mark_nav_dispatched_success(self):
        ds = DroneState()
        ds.set_target((1.0, 2.0, -3.0))
        task = MagicMock()
        assert ds.try_mark_nav_dispatched((1.0, 2.0, -3.0), task) is True
        assert ds.nav_command_sent is True

    def test_try_mark_nav_dispatched_stale_target(self):
        ds = DroneState()
        ds.set_target((1.0, 2.0, -3.0))
        task = MagicMock()
        assert ds.try_mark_nav_dispatched((9.0, 9.0, -9.0), task) is False
        task.cancel.assert_called_once()

    def test_idle_hover(self):
        ds = DroneState()
        assert ds.get_idle_hover_sent() is False
        ds.mark_idle_hover_sent()
        assert ds.get_idle_hover_sent() is True

    def test_stop_signal(self):
        ds = DroneState()
        assert ds.get_should_stop() is False
        ds.request_stop()
        assert ds.get_should_stop() is True

    def test_camera_frames(self):
        ds = DroneState()
        assert ds.get_frame_forward() is None
        assert ds.get_frame_down() is None
        ds.set_frame_forward("fwd")
        ds.set_frame_down("down")
        assert ds.get_frame_forward() == "fwd"
        assert ds.get_frame_down() == "down"

    def test_home_position(self):
        ds = DroneState()
        assert ds.get_home() is None
        ds.set_home((0.0, 0.0, -1.0))
        assert ds.get_home() == (0.0, 0.0, -1.0)

    def test_returning_home(self):
        ds = DroneState()
        assert ds.get_returning_home() is False
        ds.set_returning_home(True)
        assert ds.get_returning_home() is True

    def test_grounded(self):
        ds = DroneState()
        assert ds.is_grounded() is True
        ds.set_grounded(False)
        assert ds.is_grounded() is False

    def test_pose(self):
        ds = DroneState()
        assert ds.get_pose() is None
        ds.set_pose((1.0, 2.0, 3.0))
        assert ds.get_pose() == (1.0, 2.0, 3.0)

    def test_manual_velocity(self):
        ds = DroneState()
        assert ds.get_manual_velocity() == (0.0, 0.0, 0.0)
        ds.set_manual_velocity(1.0, 2.0, 3.0)
        assert ds.get_manual_velocity() == (1.0, 2.0, 3.0)

    def test_set_mode_manual_cancels_nav_task(self):
        ds = DroneState()
        ds.set_target((1.0, 2.0, -3.0))
        mock_task = MagicMock()
        ds.nav_task = mock_task
        ds.set_mode("manual")
        mock_task.cancel.assert_called_once()
