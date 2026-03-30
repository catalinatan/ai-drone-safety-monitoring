"""Abstract interface for all camera sources."""

from abc import ABC, abstractmethod

import numpy as np


class CameraBackend(ABC):
    """Abstract base class every camera implementation must satisfy."""

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the camera. Returns True on success."""
        ...

    @abstractmethod
    def grab_frame(self) -> np.ndarray | None:
        """Capture a single BGR frame. Returns None if unavailable."""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Release camera resources. Safe to call even if never connected."""
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """True when the camera is ready to deliver frames."""
        ...

    @property
    @abstractmethod
    def resolution(self) -> tuple[int, int]:
        """Returns (width, height) of frames produced by this backend."""
        ...
