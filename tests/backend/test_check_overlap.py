"""Unit tests for check_danger_zone_overlap — pure numpy, no models needed."""

import numpy as np
from src.core.zone_manager import check_overlap as check_danger_zone_overlap


class TestCheckDangerZoneOverlap:

    def test_no_zone_mask_returns_safe(self):
        person_mask = np.ones((100, 100), dtype=np.uint8)
        is_alarm, danger = check_danger_zone_overlap([person_mask], None)
        assert is_alarm is False
        assert danger == []

    def test_no_people_returns_safe(self):
        zone = np.ones((100, 100), dtype=np.uint8)
        is_alarm, danger = check_danger_zone_overlap([], zone)
        assert is_alarm is False
        assert danger == []

    def test_full_overlap_triggers_alarm(self, binary_zone_mask, person_mask_in_zone):
        is_alarm, danger = check_danger_zone_overlap(
            [person_mask_in_zone], binary_zone_mask
        )
        assert is_alarm is True
        assert len(danger) == 1

    def test_no_overlap_is_safe(self, binary_zone_mask, person_mask_outside_zone):
        is_alarm, danger = check_danger_zone_overlap(
            [person_mask_outside_zone], binary_zone_mask
        )
        assert is_alarm is False
        assert danger == []

    def test_partial_overlap_below_threshold(self):
        """Person barely touches the zone — below threshold."""
        zone = np.zeros((100, 100), dtype=np.uint8)
        zone[0:50, 0:50] = 1  # top-left quadrant

        person = np.zeros((100, 100), dtype=np.uint8)
        person[45:95, 0:50] = 1  # mostly below the zone, only 5 rows overlap

        # Overlap = 5*50=250, person area = 50*50=2500, ratio = 0.1
        # Default threshold is 0.5, so this should NOT trigger
        is_alarm, danger = check_danger_zone_overlap([person], zone)
        assert is_alarm is False

    def test_partial_overlap_above_threshold(self):
        """Person is mostly inside the zone — above threshold."""
        zone = np.zeros((100, 100), dtype=np.uint8)
        zone[0:80, 0:80] = 1

        person = np.zeros((100, 100), dtype=np.uint8)
        person[10:70, 10:70] = 1  # fully inside zone

        is_alarm, danger = check_danger_zone_overlap([person], zone)
        assert is_alarm is True
        assert len(danger) == 1

    def test_tiny_person_skipped(self):
        """Person mask smaller than MIN_PERSON_AREA_PIXELS is ignored."""
        zone = np.ones((100, 100), dtype=np.uint8)

        tiny_person = np.zeros((100, 100), dtype=np.uint8)
        # Set fewer pixels than the threshold
        tiny_person[50, 50] = 1  # 1 pixel — well below MIN_PERSON_AREA_PIXELS

        is_alarm, danger = check_danger_zone_overlap([tiny_person], zone)
        assert is_alarm is False
        assert danger == []

    def test_multiple_people_mixed(self, binary_zone_mask, person_mask_in_zone,
                                   person_mask_outside_zone):
        """Two people: one in zone, one outside. Only the one in zone flagged."""
        is_alarm, danger = check_danger_zone_overlap(
            [person_mask_in_zone, person_mask_outside_zone], binary_zone_mask
        )
        assert is_alarm is True
        assert len(danger) == 1
