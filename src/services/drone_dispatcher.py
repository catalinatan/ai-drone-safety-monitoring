"""
Drone dispatcher — decides when and where to send the drone.

Receives alarm events (with 3D target coordinates) and calls the
drone API. Manages the first-auto-deploy / manual-only policy.

No direct AirSim dependency — communicates with the drone server via HTTP.
"""

from __future__ import annotations

import time
from typing import Optional


class DroneDispatcher:
    """
    Application-level drone dispatch logic.

    Policy
    ------
    - First RED-zone intrusion auto-deploys the drone (if in automatic mode
      and not already navigating).
    - All subsequent intrusions are manual-only (operator must press Deploy).
    - TRIGGER_COOLDOWN seconds must elapse between auto-deploys from the
      same feed.

    Parameters
    ----------
    drone_api_client
        Object with ``check_connection()``, ``get_status()``,
        ``set_mode(mode)``, ``goto_position(x, y, z)`` methods.
        Typically a ``DroneAPIClient`` from server.py (or a mock in tests).
    trigger_cooldown : float
        Minimum seconds between successive auto-deploys. Default 15 s.
    """

    def __init__(
        self,
        drone_api_client,
        trigger_cooldown: float = 15.0,
    ) -> None:
        self._api = drone_api_client
        self._trigger_cooldown = trigger_cooldown
        self._last_deploy_time: float = 0.0
        self._first_auto_deployed: bool = False
        self._is_navigating: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def try_auto_deploy(
        self,
        x: float,
        y: float,
        z: float,
    ) -> bool:
        """
        Attempt an automatic drone deployment to (x, y, z) NED.

        Returns True if the command was sent successfully; False otherwise.
        The caller is responsible for checking that a trigger event exists
        and that the alarm fired before calling this.
        """
        if self._api is None:
            return False

        # After the first auto-deploy, all subsequent are manual-only
        if self._first_auto_deployed:
            return False

        # Cooldown guard
        now = time.monotonic()
        if now - self._last_deploy_time < self._trigger_cooldown:
            return False

        # Check drone is ready (automatic mode, not already navigating)
        try:
            status = self._api.get_status()
        except Exception:
            return False

        if status is None:
            return False
        if status.get("mode") != "automatic":
            return False
        if status.get("is_navigating", True):
            return False

        # Issue command
        try:
            self._api.set_mode("automatic")
            success = self._api.goto_position(x, y, z)
        except Exception as e:
            print(f"[DroneDispatcher] goto failed: {e}")
            return False

        if success:
            self._last_deploy_time = now
            self._is_navigating = True
            self._first_auto_deployed = True
            print(f"[DroneDispatcher] First auto-deploy → ({x:.2f}, {y:.2f}, {z:.2f})")
        return success

    def manual_deploy(self, x: float, y: float, z: float) -> bool:
        """
        Manually deploy the drone to (x, y, z) — no policy checks.

        Returns True if the command was accepted.
        """
        if self._api is None:
            return False
        try:
            self._api.set_mode("automatic")
            success = self._api.goto_position(x, y, z)
            if success:
                self._is_navigating = True
            return success
        except Exception as e:
            print(f"[DroneDispatcher] manual_deploy failed: {e}")
            return False

    def return_home(self) -> bool:
        """Command return-to-home."""
        if self._api is None:
            return False
        try:
            import requests
            resp = requests.post(f"{self._api.base_url}/return_home", timeout=self._api.timeout)
            return resp.status_code == 200
        except Exception as e:
            print(f"[DroneDispatcher] return_home failed: {e}")
            return False

    def set_mode(self, mode: str) -> bool:
        """Pass a mode change to the drone API."""
        if self._api is None:
            return False
        return self._api.set_mode(mode)

    def poll_navigation(self) -> bool:
        """
        Check if the drone has finished navigating.

        Returns True if it was navigating and has now stopped (arrived).
        Clears the internal navigating flag in that case.
        """
        if not self._is_navigating or self._api is None:
            return False
        try:
            status = self._api.get_status()
            if status and not status.get("is_navigating", True):
                self._is_navigating = False
                return True
        except Exception:
            pass
        return False

    @property
    def is_navigating(self) -> bool:
        return self._is_navigating

    @property
    def first_auto_deployed(self) -> bool:
        return self._first_auto_deployed

    def reconnect(self) -> bool:
        """Attempt lazy reconnect to the drone API."""
        if self._api is None:
            return False
        try:
            return self._api.check_connection()
        except Exception:
            return False
