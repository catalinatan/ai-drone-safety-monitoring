"""
Audit event logger — writes detection, drone, and zone events to rotating JSONL files.

Events are stored in data/events/YYYY-MM-DD.jsonl (one file per day).
Uses a module-level singleton for access throughout the application.

Usage:
    from src.services.event_logger import log_event, AuditEventType

    log_event(AuditEventType.ALARM_FIRED, feed_id="cam-1", danger_count=2)
    log_event(AuditEventType.DRONE_AUTO_DEPLOYED, feed_id="cam-1", x=10, y=20, z=-5)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional


class AuditEventType(str, Enum):
    """Types of audit events."""

    ALARM_FIRED = "alarm_fired"
    DRONE_AUTO_DEPLOYED = "drone_auto_deployed"
    DRONE_MANUAL_DEPLOYED = "drone_manual_deployed"
    ZONES_UPDATED = "zones_updated"


@dataclass
class AuditEvent:
    """Single audit event record."""

    id: int
    type: AuditEventType
    timestamp: str  # ISO 8601
    feed_id: str
    data: dict = field(default_factory=dict)


class EventLogger:
    """
    Thread-safe audit event logger writing to rotating daily JSONL files.

    Events are appended to data/events/YYYY-MM-DD.jsonl with one JSON object per line.
    """

    def __init__(self, events_dir: str | Path = "data/events") -> None:
        self.events_dir = Path(events_dir)
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self._counter: int = 0

    def log(
        self,
        event_type: AuditEventType | str,
        feed_id: str = "",
        **data: Any,
    ) -> AuditEvent:
        """
        Log an audit event.

        Args:
            event_type: Type of event (AuditEventType enum value)
            feed_id: Optional feed_id this event relates to
            **data: Arbitrary event-specific data (e.g., danger_count=2, x=10.5)

        Returns:
            The AuditEvent object that was logged
        """
        if isinstance(event_type, str):
            event_type = AuditEventType(event_type)

        self._counter += 1
        now = datetime.utcnow()
        timestamp = now.isoformat() + "Z"

        event = AuditEvent(
            id=self._counter,
            type=event_type,
            timestamp=timestamp,
            feed_id=feed_id,
            data=data,
        )

        # Write to rotating daily JSONL file
        date_str = now.strftime("%Y-%m-%d")
        file_path = self.events_dir / f"{date_str}.jsonl"

        try:
            with open(file_path, "a") as f:
                f.write(json.dumps(asdict(event), default=str) + "\n")
        except Exception as e:
            print(f"[EventLogger] Failed to write event: {e}")

        return event

    def get_recent(self, limit: int = 100) -> List[AuditEvent]:
        """
        Get the most recent N events across all log files.

        Reads all JSONL files and returns the most recent events in
        reverse chronological order (newest first).
        """
        events: List[AuditEvent] = []

        # Read all JSONL files in events_dir
        jsonl_files = sorted(self.events_dir.glob("*.jsonl"), reverse=True)

        for file_path in jsonl_files:
            try:
                with open(file_path) as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                            # Convert data dict to AuditEvent
                            event_type = data.get("type", "")
                            if event_type:
                                event = AuditEvent(
                                    id=data.get("id", 0),
                                    type=AuditEventType(event_type),
                                    timestamp=data.get("timestamp", ""),
                                    feed_id=data.get("feed_id", ""),
                                    data=data.get("data", {}),
                                )
                                events.append(event)
                        except Exception:
                            pass
            except Exception as e:
                print(f"[EventLogger] Failed to read {file_path}: {e}")

        # Sort by timestamp descending (most recent first)
        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events[:limit]


# Module-level singleton
_instance: Optional[EventLogger] = None


def get_event_logger() -> EventLogger:
    """Get or create the module-level event logger singleton."""
    global _instance
    if _instance is None:
        _instance = EventLogger()
    return _instance


def log_event(
    event_type: AuditEventType | str,
    feed_id: str = "",
    **data: Any,
) -> None:
    """
    Convenience wrapper to log an event.

    Safe to call from anywhere — returns silently if logging fails.

    Usage:
        log_event(AuditEventType.ALARM_FIRED, feed_id="cam-1", danger_count=2)
    """
    try:
        get_event_logger().log(event_type, feed_id=feed_id, **data)
    except Exception:
        pass
