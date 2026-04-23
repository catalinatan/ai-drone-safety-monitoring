"""
Video routes:
  GET /video_feed/{id}  — MJPEG stream
  GET /feeds/{id}/snapshot — single JPEG frame
"""

from __future__ import annotations

import time

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response, StreamingResponse

from src.api.dependencies import get_config, get_feed_manager
from src.services.feed_manager import FeedManager

router = APIRouter()

_NO_SIGNAL_JPEG: bytes | None = None


def _make_no_signal_frame() -> bytes:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (30, 30, 30)
    text = "NO SIGNAL"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 1.5, 3)
    cv2.putText(frame, text, ((640 - tw) // 2, (480 + th) // 2), font, 1.5, (100, 100, 100), 3)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes()


def _get_no_signal() -> bytes:
    global _NO_SIGNAL_JPEG
    if _NO_SIGNAL_JPEG is None:
        _NO_SIGNAL_JPEG = _make_no_signal_frame()
    return _NO_SIGNAL_JPEG


def _generate_frames(feed_id: str, fm: FeedManager, stream_interval: float):
    """Yield MJPEG boundary-wrapped JPEG frames indefinitely."""
    last_jpeg = _get_no_signal()

    while True:
        state = fm.get_state(feed_id)
        if state is not None:
            with state.lock:
                frame = state.last_frame.copy() if state.last_frame is not None else None
                overlay = state.last_mask_overlay

            if frame is not None:
                frame_rgb = frame  # Frame is already in RGB from AirSim

                # Composite detection mask — overlay is a combined binary
                # mask (H, W) uint8 with 1 = person pixel.  Paint cyan
                # directly on masked pixels (much faster than full-frame
                # addWeighted that the old RGB overlay path required).
                if overlay is not None:
                    if overlay.shape[:2] != frame_rgb.shape[:2]:
                        overlay = cv2.resize(
                            overlay,
                            (frame_rgb.shape[1], frame_rgb.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    mask_bool = overlay.astype(bool)
                    # Blend: 60% original + 40% cyan (0, 255, 255 in RGB)
                    frame_rgb[mask_bool] = (
                        frame_rgb[mask_bool] * 0.6 + np.array([0, 255, 255], dtype=np.float32) * 0.4
                    ).astype(np.uint8)

                h, w = frame_rgb.shape[:2]

                ret, buf = cv2.imencode(".jpg", frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    last_jpeg = buf.tobytes()

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + last_jpeg + b"\r\n")
        time.sleep(stream_interval)


@router.get("/feeds/{feed_id}/snapshot")
async def get_feed_snapshot(
    feed_id: str,
    fm: FeedManager = Depends(get_feed_manager),
):
    """Return the latest frame from a feed as a JPEG."""
    state = fm.get_state(feed_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Feed {feed_id!r} not found")

    frame = fm.get_frame(feed_id)
    if frame is None:
        raise HTTPException(status_code=404, detail="No frame available")

    ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ret:
        raise HTTPException(status_code=500, detail="JPEG encoding failed")

    return Response(content=buf.tobytes(), media_type="image/jpeg")


@router.get("/video_feed/{feed_id}")
async def video_feed(
    feed_id: str,
    fm: FeedManager = Depends(get_feed_manager),
    cfg: dict = Depends(get_config),
):
    if fm.get_state(feed_id) is None:
        raise HTTPException(status_code=404, detail=f"Feed {feed_id!r} not found")

    stream_fps = cfg.get("streaming", {}).get("stream_fps", 30)
    interval = 1.0 / max(1, stream_fps)

    return StreamingResponse(
        _generate_frames(feed_id, fm, interval),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
