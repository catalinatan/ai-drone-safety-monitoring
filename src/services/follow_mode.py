"""
CCTV Follow Mode controller.

Manages drone following behavior for moving targets (ship, railway, bridge).
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

from src.services.target_tracker import TargetTracker


class FollowModeController:
    """Manages drone following of a target."""

    def __init__(
        self,
        target: str = "ship",
        hover_altitude: float = -15.0,
        hover_drones: bool = False,
    ):
        """
        Initialize follow mode.

        Parameters
        ----------
        target : str
            Target label ("ship", "railway", "bridge").
        hover_altitude : float
            Altitude to maintain above target (meters, NED).
        hover_drones : bool
            If True, drones hover above target. If False, drones move with target.
        """
        self.target = target
        self.hover_altitude = hover_altitude
        self.hover_drones = hover_drones
        self.tracker = TargetTracker(target_label=target)
        self._last_update = 0.0
        self._update_interval = 1.0  # Update every 1 second

    def get_target_position(
        self,
        frame,
        camera_height: float = 10.0,
    ) -> Optional[Tuple[float, float, float]]:
        """
        Detect target in frame and return its 3D position.

        Parameters
        ----------
        frame : np.ndarray
            Camera frame.
        camera_height : float
            Camera altitude.

        Returns
        -------
        (x, y, z) NED world coordinates or None.
        """
        if frame is None:
            return None

        return self.tracker.track_in_frame(frame, camera_height=camera_height)

    def should_update(self) -> bool:
        """Check if it's time to update target following."""
        now = time.time()
        if now - self._last_update >= self._update_interval:
            self._last_update = now
            return True
        return False

    def compute_waypoint(
        self,
        target_pos: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        """
        Compute waypoint for drone based on target position.

        Parameters
        ----------
        target_pos : (x, y, z)
            Target's world position (NED).

        Returns
        -------
        (x, y, z) waypoint for drone to fly to.
        """
        if not target_pos:
            return (0.0, 0.0, self.hover_altitude)

        tx, ty, tz = target_pos

        if self.hover_drones:
            # Hover directly above target
            return (tx, ty, self.hover_altitude)
        else:
            # Move with target (follow)
            return (tx, ty, self.hover_altitude)
