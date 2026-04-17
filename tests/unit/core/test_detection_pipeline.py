"""Unit tests for DetectionPipeline — orchestration with injected fakes."""

import numpy as np
import pytest

from src.core.alarm import AlarmState
from src.core.detection_pipeline import DetectionPipeline, DetectionResult
from src.core.models import Point, Zone
from src.core.zone_manager import ZoneManager


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeDetector:
    """Returns a configurable list of masks."""
    def __init__(self, masks=None):
        self.masks = masks or []

    def get_masks(self, frame):
        return self.masks


def make_zone(level: str, full: bool = True) -> Zone:
    pts = [(0, 0), (100, 0), (100, 100), (0, 100)] if full else [(50, 0), (100, 0), (100, 100), (50, 100)]
    return Zone(id="z1", level=level, points=[Point(x=x, y=y) for x, y in pts])


def full_person_mask(h=100, w=100) -> np.ndarray:
    m = np.ones((h, w), dtype=np.uint8)
    return m


def build_pipeline(
    detector=None,
    zones=None,
    cooldown=0.0,
    warmup_frames=0,
):
    zm = ZoneManager()
    if zones:
        zm.update_zones(zones, 100, 100)
    alarm = AlarmState(cooldown_seconds=cooldown)
    det = detector or FakeDetector()
    return DetectionPipeline(det, zm, alarm, warmup_frames=warmup_frames)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDetectionPipeline:
    def test_no_zones_returns_empty_result(self):
        pipeline = build_pipeline()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = pipeline.process_frame(frame)
        assert result.alarm_active is False
        assert result.people_count == 0

    def test_no_detections_no_alarm(self):
        pipeline = build_pipeline(zones=[make_zone("red")])
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = pipeline.process_frame(frame)
        assert result.alarm_active is False
        assert result.people_count == 0

    def test_person_in_red_zone_triggers_alarm(self):
        detector = FakeDetector([full_person_mask()])
        pipeline = build_pipeline(detector=detector, zones=[make_zone("red")])
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = pipeline.process_frame(frame)
        assert result.alarm_active is True
        assert result.danger_count == 1
        assert result.alarm_fired is True

    def test_person_outside_zones_no_alarm(self):
        # Zone covers right half only; person in left half
        right_half_zone = Zone(
            id="z1", level="red",
            points=[Point(x=50, y=0), Point(x=100, y=0), Point(x=100, y=100), Point(x=50, y=100)],
        )
        person = np.zeros((100, 100), dtype=np.uint8)
        person[:, :40] = 1  # left 40% — no overlap with right zone
        detector = FakeDetector([person])
        pipeline = build_pipeline(detector=detector, zones=[right_half_zone])
        result = pipeline.process_frame(np.zeros((100, 100, 3), dtype=np.uint8))
        assert result.alarm_active is False

    def test_person_in_yellow_zone_triggers_caution_not_alarm(self):
        detector = FakeDetector([full_person_mask()])
        pipeline = build_pipeline(detector=detector, zones=[make_zone("yellow")])
        result = pipeline.process_frame(np.zeros((100, 100, 3), dtype=np.uint8))
        assert result.alarm_active is False
        assert result.caution_active is True
        assert result.caution_count == 1

    def test_warmup_suppresses_alarm(self):
        detector = FakeDetector([full_person_mask()])
        pipeline = build_pipeline(
            detector=detector,
            zones=[make_zone("red")],
            warmup_frames=5,
        )
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        for _ in range(4):  # still in warmup
            result = pipeline.process_frame(frame)
            assert result.alarm_active is False

    def test_alarm_fires_after_warmup(self):
        detector = FakeDetector([full_person_mask()])
        pipeline = build_pipeline(
            detector=detector,
            zones=[make_zone("red")],
            warmup_frames=2,
        )
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        pipeline.process_frame(frame)  # frame 1 (warmup)
        result = pipeline.process_frame(frame)  # frame 2 (warmup complete: >= warmup_frames)
        assert result.alarm_active is True

    def test_frame_count_increments(self):
        pipeline = build_pipeline()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        pipeline.process_frame(frame)
        pipeline.process_frame(frame)
        assert pipeline.frame_count == 2
