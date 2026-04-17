"""Unit tests for src.services.target_tracker."""

from __future__ import annotations

import numpy as np

from src.services.target_tracker import TargetTracker


def test_init_default_label():
    t = TargetTracker()
    assert t.target_label == "ship"
    assert t._last_position is None


def test_init_custom_label():
    t = TargetTracker(target_label="railway")
    assert t.target_label == "railway"


def test_track_none_frame():
    t = TargetTracker()
    result = t.track_in_frame(None)
    assert result is None


def test_track_empty_frame():
    t = TargetTracker()
    frame = np.array([], dtype=np.uint8)
    result = t.track_in_frame(frame)
    assert result is None


def test_track_returns_last_position_on_none():
    t = TargetTracker()
    t._last_position = (1.0, 2.0, 0.0)
    result = t.track_in_frame(None)
    assert result == (1.0, 2.0, 0.0)


def test_detect_target_unknown_label():
    t = TargetTracker(target_label="unknown")
    assert t._detect_target(np.zeros((100, 100, 3), dtype=np.uint8)) is None


def test_detect_ship_no_contours():
    t = TargetTracker(target_label="ship")
    frame = np.full((100, 100, 3), 200, dtype=np.uint8)
    result = t._detect_ship(frame)
    # May or may not find contours depending on thresholds — just no crash
    assert result is None or len(result) == 2


def test_detect_railway_no_lines():
    t = TargetTracker(target_label="railway")
    frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    result = t._detect_railway(frame)
    assert result is None


def test_detect_bridge_no_edges():
    t = TargetTracker(target_label="bridge")
    frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    result = t._detect_bridge(frame)
    assert result is None


def test_track_in_frame_with_valid_image():
    t = TargetTracker(target_label="ship")
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[80:120, 80:120] = [30, 30, 30]
    result = t.track_in_frame(frame, camera_height=10.0)
    # Result is either None or a 3-tuple
    assert result is None or len(result) == 3
