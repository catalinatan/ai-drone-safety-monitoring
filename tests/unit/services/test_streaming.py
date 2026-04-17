"""Unit tests for src.services.streaming."""

from __future__ import annotations

import numpy as np

from src.services.streaming import encode_frame_jpeg, wrap_mjpeg_frame, render_overlay


def test_encode_frame_jpeg_returns_bytes():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = encode_frame_jpeg(frame)
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_encode_frame_jpeg_custom_quality():
    rng = np.random.RandomState(42)
    frame = rng.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    low = encode_frame_jpeg(frame, quality=10)
    high = encode_frame_jpeg(frame, quality=95)
    assert low is not None
    assert high is not None
    assert len(low) < len(high)


def test_wrap_mjpeg_frame():
    jpeg = b"\xff\xd8test_data"
    result = wrap_mjpeg_frame(jpeg)
    assert result.startswith(b"--frame\r\n")
    assert b"Content-Type: image/jpeg" in result
    assert result.endswith(b"\r\n")
    assert jpeg in result


def test_render_overlay_empty_masks():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    canvas = render_overlay(frame, [])
    assert canvas.shape == frame.shape
    assert np.all(canvas == 0)


def test_render_overlay_with_mask():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:20, 10:20] = 1
    canvas = render_overlay(frame, [mask])
    assert np.any(canvas[10:20, 10:20] != 0)
    assert np.all(canvas[0:10, 0:10] == 0)


def test_render_overlay_none_mask_skipped():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    canvas = render_overlay(frame, [None])
    assert np.all(canvas == 0)
