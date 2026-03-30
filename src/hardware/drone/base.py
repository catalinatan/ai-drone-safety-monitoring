"""Abstract interface for all drone backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class DronePosition:
    x: float  # North (meters)
    y: float  # East (meters)
    z: float  # Down (meters, negative = above ground)


@dataclass
class DroneStatus:
    mode: str               # "manual" | "automatic"
    is_navigating: bool
    position: DronePosition
    is_connected: bool


class DroneBackend(ABC):
    """Abstract base class every drone implementation must satisfy."""

    @abstractmethod
    def connect(self) -> bool:
        """Connect to the drone. Returns True on success."""
        ...

    @abstractmethod
    def goto(self, position: DronePosition, speed: float) -> bool:
        """Command the drone to fly to a position. Non-blocking. Returns True if accepted."""
        ...

    @abstractmethod
    def get_status(self) -> DroneStatus:
        """Return current drone status snapshot."""
        ...

    @abstractmethod
    def set_mode(self, mode: str) -> bool:
        """Switch between 'manual' and 'automatic'. Returns True on success."""
        ...

    @abstractmethod
    def return_home(self) -> bool:
        """Command return-to-home. Returns True if accepted."""
        ...

    @abstractmethod
    def grab_frame(self) -> np.ndarray | None:
        """Capture a frame from the drone's onboard camera. Returns None if unavailable."""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Release drone resources. Safe to call even if never connected."""
        ...
