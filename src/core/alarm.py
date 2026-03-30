"""
Alarm state machine — pure business logic, no I/O.

States:
  idle      → trigger() → active
  active    → cooldown expires → idle
  active    → trigger() → active (resets cooldown)
"""

from __future__ import annotations

import time


class AlarmState:
    """
    Thread-safe alarm state with cooldown logic.

    Parameters
    ----------
    cooldown_seconds : float
        Minimum seconds between consecutive alarm activations.
    """

    def __init__(self, cooldown_seconds: float = 5.0) -> None:
        self._cooldown = cooldown_seconds
        self._last_trigger: float = 0.0   # monotonic timestamp of last trigger
        self._active: bool = False

    def trigger(self) -> bool:
        """
        Attempt to trigger the alarm.

        Returns True if the alarm fires (i.e. cooldown had elapsed).
        Returns False if still in cooldown — the caller should not dispatch
        the drone again.
        """
        now = time.monotonic()
        if now - self._last_trigger >= self._cooldown:
            self._last_trigger = now
            self._active = True
            return True
        return False

    def clear(self) -> None:
        """Manually clear the alarm (e.g. when no person is detected)."""
        self._active = False

    @property
    def is_active(self) -> bool:
        """True while the alarm is firing."""
        return self._active

    @property
    def cooldown_remaining(self) -> float:
        """Seconds until the alarm can fire again (0.0 if ready)."""
        elapsed = time.monotonic() - self._last_trigger
        remaining = self._cooldown - elapsed
        return max(0.0, remaining)

    @property
    def cooldown_seconds(self) -> float:
        return self._cooldown
