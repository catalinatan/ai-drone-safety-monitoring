"""Unit tests for the audit event logger."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.services.event_logger import (
    AuditEventType,
    AuditEvent,
    EventLogger,
    log_event,
)


@pytest.fixture
def temp_events_dir():
    """Create a temporary directory for event logs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_audit_event_type_enum():
    """AuditEventType enum has expected values."""
    assert AuditEventType.ALARM_FIRED.value == "alarm_fired"
    assert AuditEventType.DRONE_AUTO_DEPLOYED.value == "drone_auto_deployed"
    assert AuditEventType.DRONE_MANUAL_DEPLOYED.value == "drone_manual_deployed"
    assert AuditEventType.ZONES_UPDATED.value == "zones_updated"


def test_event_logger_creates_directory(temp_events_dir):
    """EventLogger creates events directory if it doesn't exist."""
    events_dir = temp_events_dir / "nonexistent" / "path"
    assert not events_dir.exists()

    logger = EventLogger(events_dir=events_dir)
    assert events_dir.exists()


def test_event_logger_log_creates_file(temp_events_dir):
    """log() creates a JSONL file with the event."""
    logger = EventLogger(events_dir=temp_events_dir)

    event = logger.log(
        AuditEventType.ALARM_FIRED,
        feed_id="cam-1",
        danger_count=2,
    )

    assert event.id == 1
    assert event.type == AuditEventType.ALARM_FIRED
    assert event.feed_id == "cam-1"
    assert event.data["danger_count"] == 2

    # Check file was created
    jsonl_files = list(temp_events_dir.glob("*.jsonl"))
    assert len(jsonl_files) == 1

    # Check content
    with open(jsonl_files[0]) as f:
        line = f.readline()
        data = json.loads(line)
        assert data["id"] == 1
        assert data["type"] == "alarm_fired"
        assert data["feed_id"] == "cam-1"


def test_event_logger_multiple_events_same_day(temp_events_dir):
    """Multiple events on the same day go to the same file."""
    logger = EventLogger(events_dir=temp_events_dir)

    logger.log(AuditEventType.ALARM_FIRED, feed_id="cam-1")
    logger.log(AuditEventType.ALARM_FIRED, feed_id="cam-2")
    logger.log(AuditEventType.DRONE_AUTO_DEPLOYED, feed_id="cam-1")

    jsonl_files = list(temp_events_dir.glob("*.jsonl"))
    assert len(jsonl_files) == 1

    # Count lines
    with open(jsonl_files[0]) as f:
        lines = f.readlines()
        assert len(lines) == 3


def test_event_logger_get_recent_returns_events(temp_events_dir):
    """get_recent() returns events in reverse chronological order."""
    logger = EventLogger(events_dir=temp_events_dir)

    logger.log(AuditEventType.ALARM_FIRED, feed_id="cam-1", num=1)
    logger.log(AuditEventType.ALARM_FIRED, feed_id="cam-2", num=2)
    logger.log(AuditEventType.DRONE_AUTO_DEPLOYED, feed_id="cam-1", num=3)

    recent = logger.get_recent(limit=10)
    assert len(recent) == 3

    # Most recent should be first (num=3)
    assert recent[0].data.get("num") == 3
    assert recent[1].data.get("num") == 2
    assert recent[2].data.get("num") == 1


def test_event_logger_get_recent_respects_limit(temp_events_dir):
    """get_recent(limit=N) returns at most N events."""
    logger = EventLogger(events_dir=temp_events_dir)

    for i in range(10):
        logger.log(AuditEventType.ALARM_FIRED, feed_id=f"cam-{i}")

    recent = logger.get_recent(limit=5)
    assert len(recent) == 5


def test_log_event_convenience_function_does_not_crash():
    """log_event() convenience function doesn't crash even if logger fails."""
    # This should not raise even if logging fails
    log_event(AuditEventType.ALARM_FIRED, feed_id="cam-1", test="value")
    # If we get here, the test passed


def test_event_logger_handles_missing_files(temp_events_dir):
    """get_recent() handles missing or corrupted files gracefully."""
    logger = EventLogger(events_dir=temp_events_dir)

    # Log a valid event
    logger.log(AuditEventType.ALARM_FIRED, feed_id="cam-1")

    # Create a corrupted file
    corrupted_file = temp_events_dir / "corrupted.jsonl"
    with open(corrupted_file, "w") as f:
        f.write("not json\n")
        f.write('{"valid": "json"}\n')

    # get_recent should still work
    recent = logger.get_recent(limit=10)
    # Should at least have the valid event
    assert len(recent) >= 1


def test_event_logger_counter_increments(temp_events_dir):
    """Event ID counter increments with each log."""
    logger = EventLogger(events_dir=temp_events_dir)

    e1 = logger.log(AuditEventType.ALARM_FIRED, feed_id="cam-1")
    e2 = logger.log(AuditEventType.ALARM_FIRED, feed_id="cam-2")
    e3 = logger.log(AuditEventType.ALARM_FIRED, feed_id="cam-3")

    assert e1.id == 1
    assert e2.id == 2
    assert e3.id == 3


def test_audit_event_dataclass_fields():
    """AuditEvent dataclass has expected fields."""
    event = AuditEvent(
        id=1,
        type=AuditEventType.ALARM_FIRED,
        timestamp="2026-03-30T12:00:00Z",
        feed_id="cam-1",
        data={"test": "value"},
    )

    assert event.id == 1
    assert event.type == AuditEventType.ALARM_FIRED
    assert event.timestamp == "2026-03-30T12:00:00Z"
    assert event.feed_id == "cam-1"
    assert event.data == {"test": "value"}
