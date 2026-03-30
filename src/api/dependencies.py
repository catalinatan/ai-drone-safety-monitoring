"""
FastAPI dependency injection providers.

Holds module-level singletons (FeedManager, config, trigger store) and
exposes them as FastAPI `Depends()` callables so tests can override them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.services.feed_manager import FeedManager

# ---------------------------------------------------------------------------
# Trigger history (lives in app state, not in FeedManager)
# ---------------------------------------------------------------------------

@dataclass
class TriggerEvent:
    """A RED-zone intrusion event with snapshot and replay frames."""
    id: int
    feed_id: str
    timestamp: str
    coords: Tuple[float, float, float]
    snapshot: bytes
    replay_frames: List[Tuple[str, bytes]] = field(default_factory=list)
    replay_trigger_index: int = 0
    deployed: bool = False


class TriggerStore:
    """Thread-safe in-memory store for trigger events."""

    MAX_HISTORY = 10
    REPLAY_FPS = 10

    def __init__(self) -> None:
        self._triggers: List[TriggerEvent] = []
        self._counter: int = 0

    def add(self, event: TriggerEvent) -> None:
        self._triggers.append(event)
        if len(self._triggers) > self.MAX_HISTORY:
            self._triggers = self._triggers[-self.MAX_HISTORY:]

    def next_id(self) -> int:
        self._counter += 1
        return self._counter

    def get_by_id(self, trigger_id: int) -> Optional[TriggerEvent]:
        for t in self._triggers:
            if t.id == trigger_id:
                return t
        return None

    def latest(self) -> Optional[TriggerEvent]:
        return self._triggers[-1] if self._triggers else None

    def all(self) -> List[TriggerEvent]:
        return list(self._triggers)

    def remove(self, trigger_id: int) -> bool:
        before = len(self._triggers)
        self._triggers = [t for t in self._triggers if t.id != trigger_id]
        return len(self._triggers) < before


# ---------------------------------------------------------------------------
# Module-level singletons (overridable in tests via app.dependency_overrides)
# ---------------------------------------------------------------------------

_feed_manager: FeedManager = FeedManager()
_trigger_store: TriggerStore = TriggerStore()
_config: Dict[str, Any] = {}
_drone_api = None      # DroneAPIClient or None


def get_feed_manager() -> FeedManager:
    return _feed_manager


def get_trigger_store() -> TriggerStore:
    return _trigger_store


def get_config() -> Dict[str, Any]:
    return _config


def get_drone_api():
    return _drone_api


def set_feed_manager(fm: FeedManager) -> None:
    global _feed_manager
    _feed_manager = fm


def set_drone_api(api) -> None:
    global _drone_api
    _drone_api = api


def set_config(cfg: Dict[str, Any]) -> None:
    global _config
    _config = cfg
