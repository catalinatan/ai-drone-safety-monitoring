"""
MJPEG streaming helpers — frame encoding and multipart boundary wrapping.

No threading; these are pure transform functions used by the video route.
"""

from __future__ import annotations

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
) -> np.ndarray:
    """
    Build a colored overlay canvas from person masks.

    Parameters
    ----------
    frame : np.ndarray
        Base frame — used only for shape/dtype reference.
    person_masks : list
        Raw binary (H, W) person masks. Each detected person is painted cyan.

    Returns
    -------
    np.ndarray  RGB canvas (same H×W as frame, zeros where no mask).
                Composited onto the live frame by the video route.
    """
    canvas = np.zeros_like(frame)
    for mask in person_masks:
        if mask is not None:
            canvas[mask.astype(bool)] = [0, 255, 255]  # cyan
    return canvas
