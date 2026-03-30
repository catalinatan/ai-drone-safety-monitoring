"""
HTTP client for the drone control REST API.

Extracted from server.py during Phase 5 decomposition.
"""

from __future__ import annotations

import requests


def _default_drone_url() -> str:
    try:
        from src.core.config import get_config
        return get_config().get("drone", {}).get("api_url", "http://localhost:8000")
    except Exception:
        return "http://localhost:8000"


def _default_drone_timeout() -> int:
    try:
        from src.core.config import get_config
        return int(get_config().get("drone", {}).get("api_timeout", 5))
    except Exception:
        return 5


class DroneAPIClient:
    """Client for communicating with the drone control REST API."""

    def __init__(self, base_url: str = None, timeout: int = None):
        self.base_url = base_url if base_url is not None else _default_drone_url()
        self.timeout = timeout if timeout is not None else _default_drone_timeout()

    def check_connection(self) -> bool:
        """Return True if the drone API is reachable."""
        try:
            response = requests.get(f"{self.base_url}/status", timeout=self.timeout)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def set_mode(self, mode: str) -> bool:
        """Set drone control mode ('manual' or 'automatic')."""
        try:
            response = requests.post(
                f"{self.base_url}/mode",
                json={"mode": mode},
                timeout=self.timeout,
            )
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            print(f"[DRONE] Failed to set mode: {e}")
            return False

    def goto_position(self, x: float, y: float, z: float) -> bool:
        """Send drone to target NED position."""
        try:
            response = requests.post(
                f"{self.base_url}/goto",
                json={"x": float(x), "y": float(y), "z": float(z)},
                timeout=self.timeout,
            )
            if response.status_code == 200:
                return True
            print(f"[DRONE] Goto failed: {response.json()}")
            return False
        except requests.exceptions.RequestException as e:
            print(f"[DRONE] Failed to send goto command: {e}")
            return False

    def get_status(self) -> dict | None:
        """Return current drone status dict, or None on error."""
        try:
            response = requests.get(f"{self.base_url}/status", timeout=self.timeout)
            return response.json() if response.status_code == 200 else None
        except requests.exceptions.RequestException:
            return None
