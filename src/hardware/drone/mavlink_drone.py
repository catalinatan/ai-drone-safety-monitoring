"""
MAVLink drone backend stub — for PX4/ArduPilot hardware.

Activate by setting drone.type = "mavlink" in config. Requires pymavlink or dronekit.
"""

from __future__ import annotations

import numpy as np

from .base import DroneBackend, DronePosition, DroneStatus


class MAVLinkDrone(DroneBackend):
    """Stub implementation for MAVLink-compatible drones (PX4, ArduPilot)."""

    def __init__(self, connection_string: str = "udp:127.0.0.1:14550") -> None:
        self.connection_string = connection_string
        self._vehicle = None

    def connect(self) -> bool:
        raise NotImplementedError("MAVLink backend not yet implemented. Install dronekit and implement.")

    def goto(self, position: DronePosition, speed: float) -> bool:
        raise NotImplementedError

    def get_status(self) -> DroneStatus:
        raise NotImplementedError

    def set_mode(self, mode: str) -> bool:
        raise NotImplementedError

    def return_home(self) -> bool:
        raise NotImplementedError

    def grab_frame(self) -> np.ndarray | None:
        raise NotImplementedError

    def disconnect(self) -> None:
        self._vehicle = None
