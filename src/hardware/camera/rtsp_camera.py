"""
RTSP/IP camera backend — connects to a real camera via OpenCV VideoCapture.

Stub implementation — activate by setting camera.type = "rtsp" in feeds.yaml.
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import CameraBackend


class RTSPCamera(CameraBackend):
    """
    Reads frames from an RTSP or HTTP stream via OpenCV.

    Parameters
    ----------
    url : str
        RTSP URL, e.g. "rtsp://192.168.1.100:554/stream1"
    """

    def __init__(self, url: str) -> None:
        self.url = url
        self._cap: cv2.VideoCapture | None = None
        self._width: int = 0
        self._height: int = 0

    def connect(self) -> bool:
        self._cap = cv2.VideoCapture(self.url)
        if not self._cap.isOpened():
            self._cap = None
            return False
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return True

    def grab_frame(self) -> np.ndarray | None:
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        return frame if ret else None

    def disconnect(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def is_connected(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def resolution(self) -> tuple[int, int]:
        return (self._width, self._height)
