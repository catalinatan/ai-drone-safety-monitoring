"""
Zone management — pure business logic, no I/O.

Responsibilities:
- Store zone definitions (list of Zone objects per feed)
- Convert polygon points (0–100 percent) → binary numpy masks
- Check person mask overlap against zone masks
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from src.core.models import Zone


def zones_to_mask(
    zones: List[Zone],
    image_width: int,
    image_height: int,
) -> Optional[np.ndarray]:
    """
    Convert a list of Zone objects into a single binary mask (0/1 uint8).

    Zone polygon points are stored as percentages (0–100); this function
    scales them to pixel coordinates.

    Returns None if the zones list is empty.
    """
    if not zones:
        return None

    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    for zone in zones:
        pts = np.array(
            [[int(p.x * image_width / 100), int(p.y * image_height / 100)] for p in zone.points],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [pts], 1)
    return mask


def check_overlap(
    person_masks: List[np.ndarray],
    zone_mask: Optional[np.ndarray],
    min_person_area: int = 2,
    overlap_threshold: float = 0.5,
) -> tuple[bool, List[np.ndarray]]:
    """
    Check whether any person mask overlaps with the zone mask.

    Parameters
    ----------
    person_masks : list of binary uint8 arrays
        Each array marks pixels belonging to one detected person (1=person, 0=bg).
    zone_mask : binary uint8 array or None
        The zone to check against. If None, returns (False, []).
    min_person_area : int
        Discard person masks smaller than this many pixels (noise filter).
    overlap_threshold : float
        Fraction of the person's pixels that must fall inside the zone to trigger.

    Returns
    -------
    (is_alarm, danger_masks)
        is_alarm  — True if at least one person triggered the threshold.
        danger_masks — Subset of person_masks that triggered.
    """
    if zone_mask is None:
        return False, []

    is_alarm = False
    danger_masks: List[np.ndarray] = []

    for person_mask in person_masks:
        if np.sum(person_mask) < min_person_area:
            continue

        intersection = np.bitwise_and(person_mask, zone_mask)
        person_area = int(np.sum(person_mask))
        if person_area == 0:
            continue

        overlap_ratio = float(np.sum(intersection)) / person_area
        if overlap_ratio >= overlap_threshold:
            is_alarm = True
            danger_masks.append(person_mask)

    return is_alarm, danger_masks


class ZoneManager:
    """
    Manages zone definitions and masks for a single camera feed.

    All mask generation is lazy — call ``update_zones()`` to set zones and
    regenerate masks; call ``get_masks()`` to get the current masks.
    """

    def __init__(
        self,
        min_person_area: int = 2,
        overlap_threshold: float = 0.5,
    ) -> None:
        self._min_person_area = min_person_area
        self._overlap_threshold = overlap_threshold
        self._zones: List[Zone] = []
        self._red_mask: Optional[np.ndarray] = None
        self._yellow_mask: Optional[np.ndarray] = None

    def update_zones(
        self,
        zones: List[Zone],
        image_width: int,
        image_height: int,
    ) -> None:
        """
        Replace the current zones and regenerate binary masks.

        Manual zones have absolute pixel-level priority over auto zones.
        Where a manual zone covers a pixel, only the manual zone's level
        applies — any auto zone at that pixel is suppressed.
        """
        self._zones = list(zones)

        # Separate by source and level
        manual_red = [z for z in zones if z.source == "manual" and z.level == "red"]
        manual_yellow = [z for z in zones if z.source == "manual" and z.level == "yellow"]
        manual_green = [z for z in zones if z.source == "manual" and z.level == "green"]
        auto_red = [z for z in zones if z.source == "auto" and z.level == "red"]
        auto_yellow = [z for z in zones if z.source == "auto" and z.level == "yellow"]

        m_red = zones_to_mask(manual_red, image_width, image_height)
        m_yellow = zones_to_mask(manual_yellow, image_width, image_height)
        m_green = zones_to_mask(manual_green, image_width, image_height)
        a_red = zones_to_mask(auto_red, image_width, image_height)
        a_yellow = zones_to_mask(auto_yellow, image_width, image_height)

        # Manual zones claim pixels absolutely — auto zones only fill unclaimed pixels
        manual_any = np.zeros((image_height, image_width), dtype=np.uint8)
        for m in (m_red, m_yellow, m_green):
            if m is not None:
                manual_any = manual_any | m

        # effective_red = manual_red | (auto_red & ~manual_any)
        eff_red = np.zeros((image_height, image_width), dtype=np.uint8)
        if m_red is not None:
            eff_red = eff_red | m_red
        if a_red is not None:
            eff_red = eff_red | (a_red & ~manual_any)
        self._red_mask = eff_red if np.any(eff_red) else None

        # effective_yellow = manual_yellow | (auto_yellow & ~manual_any)
        eff_yellow = np.zeros((image_height, image_width), dtype=np.uint8)
        if m_yellow is not None:
            eff_yellow = eff_yellow | m_yellow
        if a_yellow is not None:
            eff_yellow = eff_yellow | (a_yellow & ~manual_any)
        self._yellow_mask = eff_yellow if np.any(eff_yellow) else None

    def get_zones(self) -> List[Zone]:
        return list(self._zones)

    @property
    def red_mask(self) -> Optional[np.ndarray]:
        return self._red_mask

    @property
    def yellow_mask(self) -> Optional[np.ndarray]:
        return self._yellow_mask

    def check_red(self, person_masks: List[np.ndarray]) -> tuple[bool, List[np.ndarray]]:
        """Check person overlap against RED zone mask."""
        return check_overlap(
            person_masks,
            self._red_mask,
            self._min_person_area,
            self._overlap_threshold,
        )

    def check_yellow(
        self,
        person_masks: List[np.ndarray],
        exclude_red: bool = True,
    ) -> tuple[bool, List[np.ndarray]]:
        """
        Check person overlap against YELLOW zone mask.

        If ``exclude_red=True`` (default), pixels already in the RED zone are
        subtracted so RED always takes priority over YELLOW.
        """
        effective = self._yellow_mask
        if effective is None:
            return False, []

        if exclude_red and self._red_mask is not None:
            red = self._red_mask
            if red.shape != effective.shape:
                red = cv2.resize(
                    red,
                    (effective.shape[1], effective.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            effective = effective & (~red)

        return check_overlap(
            person_masks,
            effective,
            self._min_person_area,
            self._overlap_threshold,
        )
