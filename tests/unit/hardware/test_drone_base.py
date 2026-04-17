"""
Hardware contract tests for DroneBackend implementations.

Since AirSim is unavailable in CI, these tests use a MockDroneBackend
to verify the interface contract and factory wiring.
"""

import numpy as np
import pytest

from src.hardware.drone.base import DroneBackend, DronePosition, DroneStatus


class MockDroneBackend(DroneBackend):
    """Minimal in-memory drone for testing."""

    def __init__(self):
        self._connected = False
        self._mode = "automatic"
        self._navigating = False
        self.goto_calls: list[tuple[DronePosition, float]] = []

    def connect(self) -> bool:
        self._connected = True
        return True

    def goto(self, position: DronePosition, speed: float) -> bool:
        if not self._connected:
            return False
        self.goto_calls.append((position, speed))
        self._navigating = True
        return True

    def get_status(self) -> DroneStatus:
        return DroneStatus(
            mode=self._mode,
            is_navigating=self._navigating,
            position=DronePosition(0.0, 0.0, -10.0),
            is_connected=self._connected,
        )

    def set_mode(self, mode: str) -> bool:
        self._mode = mode
        return True

    def return_home(self) -> bool:
        return self._connected

    def grab_frame(self) -> np.ndarray | None:
        return None

    def disconnect(self) -> None:
        self._connected = False


class DroneContractTests:
    """Mixin — every DroneBackend implementation must pass these tests."""

    def get_backend(self) -> DroneBackend:
        raise NotImplementedError

    def test_implements_abc(self):
        assert isinstance(self.get_backend(), DroneBackend)

    def test_connect_returns_bool(self):
        b = self.get_backend()
        assert isinstance(b.connect(), bool)
        b.disconnect()

    def test_get_status_returns_drone_status(self):
        b = self.get_backend()
        b.connect()
        s = b.get_status()
        assert isinstance(s, DroneStatus)
        assert isinstance(s.position, DronePosition)
        b.disconnect()

    def test_set_mode_manual(self):
        b = self.get_backend()
        b.connect()
        assert b.set_mode("manual") in (True, False)
        b.disconnect()

    def test_disconnect_is_idempotent(self):
        b = self.get_backend()
        b.disconnect()
        b.disconnect()


class TestMockDroneBackend(DroneContractTests):
    def get_backend(self) -> DroneBackend:
        return MockDroneBackend()

    def test_goto_records_call(self):
        drone = MockDroneBackend()
        drone.connect()
        pos = DronePosition(x=10.0, y=5.0, z=-10.0)
        result = drone.goto(pos, speed=5.0)
        assert result is True
        assert len(drone.goto_calls) == 1
        assert drone.goto_calls[0][0] == pos

    def test_goto_fails_if_not_connected(self):
        drone = MockDroneBackend()
        result = drone.goto(DronePosition(0, 0, 0), speed=5.0)
        assert result is False

    def test_status_navigating_after_goto(self):
        drone = MockDroneBackend()
        drone.connect()
        drone.goto(DronePosition(1, 2, -10), speed=3.0)
        assert drone.get_status().is_navigating is True
