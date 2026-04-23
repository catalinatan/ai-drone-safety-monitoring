"""
Thread-safe shared state for the drone control server.

All mutable state lives in the ``DroneState`` singleton, whose every accessor
acquires a ``threading.Lock``.  The API thread *writes* targets / mode; the
control-loop thread *reads* them via atomic snapshots (``get_nav_snapshot`` and
``try_mark_nav_dispatched``) to avoid TOCTOU races.

Coordinate system: AirSim NED (North-East-Down) in metres.
  - x → North
  - y → East
  - z → Down  (negative values = above ground)
"""

from __future__ import annotations

import threading
from typing import Optional, Tuple

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Pydantic request models (used by FastAPI route handlers in app.py)
# ---------------------------------------------------------------------------


class ModeRequest(BaseModel):
    mode: str  # "manual" or "automatic"


class GotoRequest(BaseModel):
    x: float  # North (meters)
    y: float  # East (meters)
    z: Optional[float] = -10.0  # Down (meters, negative = above ground)


class MoveRequest(BaseModel):
    vx: float = 0.0  # North velocity (m/s)
    vy: float = 0.0  # East velocity (m/s)
    vz: float = 0.0  # Down velocity (m/s, negative = up)


# ---------------------------------------------------------------------------
# Safety check
# ---------------------------------------------------------------------------


def check_safety(
    target_pos: Tuple[float, float, float],
    max_altitude: float,
) -> Tuple[bool, str]:
    """Verify whether navigation to *target_pos* is safe.

    Parameters
    ----------
    target_pos:
        (x, y, z) in NED metres.
    max_altitude:
        Maximum allowed altitude in metres (positive value).

    Returns
    -------
    (is_safe, reason)
    """
    if target_pos[2] > 0:
        return False, "Target altitude is below ground"
    if abs(target_pos[2]) > max_altitude:
        return False, f"Target altitude exceeds maximum: {abs(target_pos[2])}m > {max_altitude}m"
    return True, "OK"


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------


class DroneState:
    """Thread-safe shared state between the API server thread and the
    control-loop thread.

    Locking strategy: every public method acquires ``self.lock`` so that
    callers never need to worry about concurrent access.  For the navigation
    hot-path the control loop uses ``get_nav_snapshot()`` +
    ``try_mark_nav_dispatched()`` to avoid a TOCTOU race when a new /goto
    arrives between reading the target and issuing the AirSim command.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.mode: str = "automatic"  # "manual" | "automatic"
        self.target_position: Optional[Tuple[float, float, float]] = None
        self.home_position: Optional[Tuple[float, float, float]] = None
        self.is_navigating: bool = False
        self.nav_command_sent: bool = False
        self.nav_task = None  # AirSim future
        self.idle_hover_sent: bool = False
        self.should_stop: bool = False
        self.returning_home: bool = False
        self.grounded: bool = True  # True after RTH land / before first takeoff
        self.current_pose: Optional[Tuple[float, float, float]] = None
        self.manual_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.frame_forward = None  # forward-looking camera (camera 3)
        self.frame_down = None  # downward-looking camera (camera 0)

    # -- Mode --

    def set_mode(self, mode: str) -> None:
        """Switch mode.  Switching to manual cancels any in-flight navigation."""
        with self.lock:
            self.mode = mode
            self.idle_hover_sent = False
            if mode == "manual":
                self.is_navigating = False
                self.nav_command_sent = False
                self.returning_home = False
                if self.nav_task is not None:
                    try:
                        self.nav_task.cancel()
                    except (AttributeError, Exception) as e:
                        print(f"[WARN] Could not cancel nav_task: {e}")
                    self.nav_task = None

    def get_mode(self) -> str:
        with self.lock:
            return self.mode

    # -- Navigation target --

    def set_target(self, position: Tuple[float, float, float]) -> None:
        """Accept a new NED target.  Resets dispatch flags so the control loop
        will issue a fresh moveToPositionAsync on the next iteration."""
        with self.lock:
            self.target_position = position
            self.is_navigating = True
            self.nav_command_sent = False
            self.idle_hover_sent = False

    def get_target(self) -> Optional[Tuple[float, float, float]]:
        with self.lock:
            return self.target_position

    def clear_target(self) -> None:
        with self.lock:
            self.target_position = None
            self.is_navigating = False
            self.nav_command_sent = False
            if self.nav_task is not None:
                try:
                    self.nav_task.cancel()
                except (AttributeError, Exception) as e:
                    print(f"[WARN] Could not cancel nav_task: {e}")
                self.nav_task = None

    def get_nav_snapshot(self) -> Tuple[Optional[Tuple[float, float, float]], bool, bool]:
        """Atomically read (target, is_navigating, nav_command_sent)."""
        with self.lock:
            return self.target_position, self.is_navigating, self.nav_command_sent

    def try_mark_nav_dispatched(
        self,
        expected_target: Tuple[float, float, float],
        task,
    ) -> bool:
        """Compare-and-swap guard: commits the AirSim future only if the target
        hasn't been replaced by a newer /goto call.  If the target *has* changed,
        the stale future is cancelled immediately."""
        with self.lock:
            if self.target_position == expected_target and self.is_navigating:
                self.nav_task = task
                self.nav_command_sent = True
                return True
            else:
                try:
                    task.cancel()
                except (AttributeError, Exception) as e:
                    print(f"[WARN] Could not cancel task: {e}")
                return False

    # -- Hover sentinel --

    def get_idle_hover_sent(self) -> bool:
        with self.lock:
            return self.idle_hover_sent

    def mark_idle_hover_sent(self) -> None:
        with self.lock:
            self.idle_hover_sent = True

    # -- Stop signal --

    def request_stop(self) -> None:
        with self.lock:
            self.should_stop = True

    def get_should_stop(self) -> bool:
        with self.lock:
            return self.should_stop

    # -- Camera frames --

    def set_frame_forward(self, frame) -> None:
        with self.lock:
            self.frame_forward = frame

    def get_frame_forward(self):
        with self.lock:
            return self.frame_forward

    def set_frame_down(self, frame) -> None:
        with self.lock:
            self.frame_down = frame

    def get_frame_down(self):
        with self.lock:
            return self.frame_down

    # -- Home position --

    def set_home(self, position: Tuple[float, float, float]) -> None:
        with self.lock:
            self.home_position = position

    def get_home(self) -> Optional[Tuple[float, float, float]]:
        with self.lock:
            return self.home_position

    # -- RTH flag --

    def set_returning_home(self, value: bool) -> None:
        with self.lock:
            self.returning_home = value

    def get_returning_home(self) -> bool:
        with self.lock:
            return self.returning_home

    # -- Grounded flag --

    def set_grounded(self, value: bool) -> None:
        with self.lock:
            self.grounded = value

    def is_grounded(self) -> bool:
        with self.lock:
            return self.grounded

    # -- Pose --

    def set_pose(self, pose_xyz: Tuple[float, float, float]) -> None:
        with self.lock:
            self.current_pose = pose_xyz

    def get_pose(self) -> Optional[Tuple[float, float, float]]:
        with self.lock:
            return self.current_pose

    # -- Manual velocity --

    def set_manual_velocity(self, vx: float, vy: float, vz: float) -> None:
        with self.lock:
            self.manual_velocity = (vx, vy, vz)

    def get_manual_velocity(self) -> Tuple[float, float, float]:
        with self.lock:
            return self.manual_velocity


# Module-level singleton — shared between the API thread and the control-loop thread.
drone_state = DroneState()
