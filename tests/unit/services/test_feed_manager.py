"""Unit tests for FeedManager — lifecycle, state management, thread safety."""

import numpy as np
import pytest

from src.core.models import Point, Zone
from src.services.feed_manager import FeedManager


class MockCamera:
    """Minimal camera stand-in — just needs to satisfy type."""
    def connect(self): return True
    def disconnect(self): pass
    def grab_frame(self): return None
    @property
    def is_connected(self): return True
    @property
    def resolution(self): return (640, 480)


def make_zone(level="red") -> Zone:
    return Zone(
        id="z1", level=level,
        points=[Point(x=0, y=0), Point(x=100, y=0), Point(x=100, y=100), Point(x=0, y=100)],
    )


@pytest.fixture
def fm():
    m = FeedManager()
    m.register_feed("feed-1", "CAM 1", "Location A", MockCamera(), scene_type="bridge")
    return m


class TestFeedManagerRegistration:
    def test_register_adds_feed(self, fm):
        assert "feed-1" in fm.feed_ids()

    def test_get_state_unknown_returns_none(self, fm):
        assert fm.get_state("feed-99") is None

    def test_register_idempotent(self, fm):
        fm.register_feed("feed-1", "CAM 1 new", "Loc", MockCamera())
        assert len(fm.feed_ids()) == 1


class TestFeedManagerFrames:
    def test_get_frame_none_before_capture(self, fm):
        assert fm.get_frame("feed-1") is None

    def test_store_and_get_frame(self, fm):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        fm.store_frame("feed-1", frame)
        result = fm.get_frame("feed-1")
        assert result is not None
        assert result.shape == frame.shape

    def test_get_frame_returns_copy(self, fm):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        fm.store_frame("feed-1", frame)
        f1 = fm.get_frame("feed-1")
        f2 = fm.get_frame("feed-1")
        assert f1 is not f2  # different objects

    def test_store_frame_increments_count(self, fm):
        state = fm.get_state("feed-1")
        assert state.frame_count == 0
        fm.store_frame("feed-1", np.zeros((10, 10, 3), dtype=np.uint8))
        assert state.frame_count == 1


class TestFeedManagerDetection:
    def test_update_detection_sets_alarm(self, fm):
        fm.update_detection("feed-1", True, False, 1, 1, 0)
        snap = fm.snapshot("feed-1")
        assert snap["alarm_active"] is True
        assert snap["people_count"] == 1

    def test_update_detection_clears_target_when_no_alarm(self, fm):
        # First set a target
        fm.update_detection("feed-1", True, False, 1, 1, 0, target_coordinates=(1.0, 2.0, -10.0))
        # Now clear alarm
        fm.update_detection("feed-1", False, False, 0, 0, 0)
        snap = fm.snapshot("feed-1")
        assert snap["target_coordinates"] is None


class TestFeedManagerZones:
    def test_update_zones_stores_zones(self, fm):
        fm.update_zones("feed-1", [make_zone("red")], 640, 480)
        zones = fm.get_zones("feed-1")
        assert len(zones) == 1
        assert zones[0].level == "red"

    def test_update_zones_unknown_feed_raises(self, fm):
        with pytest.raises(ValueError):
            fm.update_zones("feed-99", [], 640, 480)

    def test_zone_manager_red_mask_created(self, fm):
        fm.update_zones("feed-1", [make_zone("red")], 100, 100)
        state = fm.get_state("feed-1")
        assert state.zone_manager.red_mask is not None


class TestFeedManagerWarmup:
    def test_not_warmed_up_initially(self, fm):
        assert fm.is_warmed_up("feed-1", warmup_frames=5) is False

    def test_warmed_up_after_enough_frames(self, fm):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        for _ in range(5):
            fm.store_frame("feed-1", frame)
        assert fm.is_warmed_up("feed-1", warmup_frames=5) is True


class TestFeedManagerSnapshot:
    def test_snapshot_unknown_feed_returns_none(self, fm):
        assert fm.snapshot("no-such-feed") is None

    def test_snapshot_structure(self, fm):
        snap = fm.snapshot("feed-1")
        required_keys = {
            "feed_id", "alarm_active", "caution_active",
            "people_count", "danger_count", "caution_count",
        }
        assert required_keys <= snap.keys()
