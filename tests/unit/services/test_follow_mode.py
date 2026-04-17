"""Unit tests for src.services.follow_mode."""

from __future__ import annotations

import time

import numpy as np

from src.services.follow_mode import FollowModeController


def test_init_defaults():
    ctrl = FollowModeController()
    assert ctrl.target == "ship"
    assert ctrl.hover_altitude == -15.0
    assert ctrl.hover_drones is False


def test_init_custom():
    ctrl = FollowModeController(target="bridge", hover_altitude=-20.0, hover_drones=True)
    assert ctrl.target == "bridge"
    assert ctrl.hover_altitude == -20.0
    assert ctrl.hover_drones is True
    assert ctrl.tracker.target_label == "bridge"


def test_get_target_position_none_frame():
    ctrl = FollowModeController()
    assert ctrl.get_target_position(None) is None


def test_get_target_position_valid_frame():
    ctrl = FollowModeController()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = ctrl.get_target_position(frame)
    assert result is None or len(result) == 3


def test_should_update_first_call():
    ctrl = FollowModeController()
    ctrl._last_update = 0.0
    assert ctrl.should_update() is True


def test_should_update_respects_interval():
    ctrl = FollowModeController()
    ctrl._last_update = time.time()
    assert ctrl.should_update() is False


def test_compute_waypoint_none():
    ctrl = FollowModeController()
    wp = ctrl.compute_waypoint(None)
    assert wp == (0.0, 0.0, -15.0)


def test_compute_waypoint_hover():
    ctrl = FollowModeController(hover_drones=True, hover_altitude=-25.0)
    wp = ctrl.compute_waypoint((10.0, 20.0, 0.0))
    assert wp == (10.0, 20.0, -25.0)


def test_compute_waypoint_follow():
    ctrl = FollowModeController(hover_drones=False, hover_altitude=-15.0)
    wp = ctrl.compute_waypoint((5.0, 10.0, 0.0))
    assert wp == (5.0, 10.0, -15.0)
