"""
MJPEG streaming helpers — frame encoding and multipart boundary wrapping.

No threading; these are pure transform functions used by the video route.
"""

from __future__ import annotations

from typing import Iterator, Optional

import cv2
import numpy as np


_BOUNDARY = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
_CRLF = b"\r\n"


def encode_frame_jpeg(frame: np.ndarray, quality: int = 85) -> bytes | None:
    """
    Encode a BGR or RGB numpy array as JPEG bytes.

    Returns None if encoding fails.
    """
    success, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        return None
    return buf.tobytes()


def wrap_mjpeg_frame(jpeg_bytes: bytes) -> bytes:
    """
    Wrap JPEG bytes in a multipart/x-mixed-replace boundary chunk.

    The resulting bytes are ready to be written directly to the HTTP response.
    """
    return _BOUNDARY + jpeg_bytes + _CRLF


def render_overlay(
    frame: np.ndarray,
    person_masks: list[np.ndarray],
    mask_overlay: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compose the final display frame: base frame with optional mask overlay.

    Parameters
    ----------
    frame : np.ndarray
        Base BGR (or RGB) frame.
    person_masks : list
        Raw binary person masks (used only if mask_overlay is None).
    mask_overlay : np.ndarray or None
        Pre-rendered RGB overlay (cyan silhouettes). If provided, composited
        directly onto the frame — cheaper than re-rendering per request.

    Returns
    -------
    np.ndarray  BGR frame ready for JPEG encoding.
    """
    display = frame.copy()

    if mask_overlay is not None and np.any(mask_overlay):
        # Blend pre-rendered overlay onto the frame
        nonzero = mask_overlay.sum(axis=2) > 0
        display[nonzero] = cv2.addWeighted(
            display, 0.6, mask_overlay, 0.4, 0
        )[nonzero]

    return display
