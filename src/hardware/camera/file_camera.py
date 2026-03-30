"""
File-based camera backend — reads frames from a video file or image.

Use cases:
- Offline testing with a known video (no AirSim required)
- Demo / validation runs
- CI fixtures
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .base import CameraBackend


class FileCamera(CameraBackend):
    """
    Plays back frames from a video file (or returns a still image repeatedly).

    Parameters
    ----------
    path : str | Path
        Path to a video file (.mp4, .avi, …) or still image (.jpg, .png, …).
    loop : bool
        If True, restart from the beginning when the file ends (default True).
    """

    def __init__(self, path: str | Path, loop: bool = True) -> None:
        self.path = str(path)
        self.loop = loop
        self._cap: cv2.VideoCapture | None = None
        self._is_image: bool = False
        self._still: np.ndarray | None = None
        self._width: int = 0
        self._height: int = 0

    # ------------------------------------------------------------------
    # CameraBackend interface
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        suffix = Path(self.path).suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png", ".bmp"}:
            self._still = cv2.imread(self.path)
            if self._still is None:
                return False
            self._is_image = True
            self._height, self._width = self._still.shape[:2]
            return True

        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            self._cap = None
            return False
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return True

    def grab_frame(self) -> np.ndarray | None:
        if self._is_image:
            return self._still.copy() if self._still is not None else None

        if self._cap is None:
            return None

        ret, frame = self._cap.read()
        if not ret:
            if self.loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
                if not ret:
                    return None
            else:
                return None
        return frame

    def disconnect(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._still = None

    @property
    def is_connected(self) -> bool:
        if self._is_image:
            return self._still is not None
        return self._cap is not None and self._cap.isOpened()

    @property
    def resolution(self) -> tuple[int, int]:
        return (self._width, self._height)
