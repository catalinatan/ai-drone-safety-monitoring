"""Unit tests for ZoneManager and zone_to_mask helpers."""

import numpy as np
import pytest

from src.core.models import Point, Zone
from src.core.zone_manager import ZoneManager, zones_to_mask, check_overlap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_zone(level: str, points: list[tuple[float, float]]) -> Zone:
    return Zone(id="z1", level=level, points=[Point(x=x, y=y) for x, y in points])


def full_zone(level: str = "red") -> Zone:
    """A zone that covers the full 100x100% frame."""
    return make_zone(level, [(0, 0), (100, 0), (100, 100), (0, 100)])


def half_zone(level: str = "red") -> Zone:
    """A zone covering the left half of the frame."""
    return make_zone(level, [(0, 0), (50, 0), (50, 100), (0, 100)])


# ---------------------------------------------------------------------------
# zones_to_mask()
# ---------------------------------------------------------------------------

class TestZonesToMask:
    def test_empty_zones_returns_none(self):
        assert zones_to_mask([], 100, 100) is None

    def test_full_zone_all_ones(self):
        zone = full_zone()
        mask = zones_to_mask([zone], 100, 100)
        assert mask is not None
        assert mask.shape == (100, 100)
        assert mask.min() >= 0 and mask.max() == 1

    def test_half_zone_left_half_set(self):
        zone = half_zone()
        mask = zones_to_mask([zone], 100, 100)
        assert mask is not None
        # Left half should be 1, right half 0
        assert mask[:, :50].sum() > 0
        assert mask[:, 55:].sum() == 0

    def test_mask_shape_matches_image(self):
        zone = full_zone()
        mask = zones_to_mask([zone], 640, 480)
        assert mask.shape == (480, 640)


# ---------------------------------------------------------------------------
# check_overlap()
# ---------------------------------------------------------------------------

class TestCheckOverlap:
    def test_no_zone_mask_returns_safe(self):
        person = np.ones((100, 100), dtype=np.uint8)
        is_alarm, masks = check_overlap([person], None)
        assert is_alarm is False
        assert masks == []

    def test_no_people_returns_safe(self):
        zone = np.ones((100, 100), dtype=np.uint8)
        is_alarm, masks = check_overlap([], zone)
        assert is_alarm is False
        assert masks == []

    def test_full_overlap_triggers_alarm(self):
        zone = np.ones((100, 100), dtype=np.uint8)
        person = np.ones((100, 100), dtype=np.uint8)
        is_alarm, masks = check_overlap([person], zone)
        assert is_alarm is True
        assert len(masks) == 1

    def test_zero_overlap_no_alarm(self):
        zone = np.zeros((100, 100), dtype=np.uint8)
        zone[:, 50:] = 1   # right half
        person = np.zeros((100, 100), dtype=np.uint8)
        person[:, :40] = 1  # left part
        is_alarm, masks = check_overlap([person], zone)
        assert is_alarm is False

    def test_tiny_person_ignored(self):
        zone = np.ones((100, 100), dtype=np.uint8)
        tiny = np.zeros((100, 100), dtype=np.uint8)
        tiny[0, 0] = 1  # 1 pixel — below min_person_area=2
        is_alarm, masks = check_overlap([tiny], zone, min_person_area=2)
        assert is_alarm is False


# ---------------------------------------------------------------------------
# ZoneManager
# ---------------------------------------------------------------------------

class TestZoneManager:
    def test_initially_no_masks(self):
        zm = ZoneManager()
        assert zm.red_mask is None
        assert zm.yellow_mask is None

    def test_update_zones_creates_red_mask(self):
        zm = ZoneManager()
        zm.update_zones([full_zone("red")], 100, 100)
        assert zm.red_mask is not None
        assert zm.yellow_mask is None

    def test_update_zones_creates_yellow_mask(self):
        zm = ZoneManager()
        zm.update_zones([full_zone("yellow")], 100, 100)
        assert zm.yellow_mask is not None
        assert zm.red_mask is None

    def test_check_red_alarm(self):
        zm = ZoneManager()
        zm.update_zones([full_zone("red")], 100, 100)
        person = np.ones((100, 100), dtype=np.uint8)
        is_alarm, masks = zm.check_red([person])
        assert is_alarm is True

    def test_check_yellow_excludes_red(self):
        zm = ZoneManager()
        # Full red zone + left-half yellow zone
        zm.update_zones([full_zone("red"), half_zone("yellow")], 100, 100)
        person = np.zeros((100, 100), dtype=np.uint8)
        person[:, :40] = 1  # entirely in left half
        # Yellow should not fire because left half is covered by red
        _, caution = zm.check_yellow([person], exclude_red=True)
        assert len(caution) == 0

    def test_get_zones_round_trip(self):
        zm = ZoneManager()
        zones = [full_zone("red")]
        zm.update_zones(zones, 100, 100)
        assert len(zm.get_zones()) == 1
