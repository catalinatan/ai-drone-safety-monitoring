"""
FeedManager — manages all camera feeds and their per-feed state.

Responsibilities:
- Register feeds with their CameraBackend instances
- Provide thread-safe access to the latest frame per feed
- Track per-feed detection state (alarm, people counts, zones, etc.)
- Cache the latest frame for streaming and detection

This class holds state only. Detection and streaming logic live in
DetectionPipeline and streaming.py respectively.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.core.models import Zone
from src.core.zone_manager import ZoneManager
from src.hardware.camera.base import CameraBackend


# ---------------------------------------------------------------------------
# Per-feed state
# ---------------------------------------------------------------------------

@dataclass
class FeedState:
    """All mutable runtime state for a single camera feed."""
    feed_id: str
    name: str
    location: str
    scene_type: Optional[str] = None

    # Zone state
    zones: List[Zone] = field(default_factory=list)
    zone_manager: ZoneManager = field(default_factory=ZoneManager)

    # Detection results
    alarm_active: bool = False
    caution_active: bool = False
    people_count: int = 0
    danger_count: int = 0
    caution_count: int = 0
    target_coordinates: Optional[Tuple[float, float, float]] = None
    last_detection_time: Optional[datetime] = None

    # Frame data
    last_frame: Optional[np.ndarray] = None
    last_mask_overlay: Optional[np.ndarray] = None
    position: Optional[Tuple[float, float, float]] = None  # NED (x, y, z) from camera vehicle
    frame_count: int = 0
    replay_buffer: deque = field(default_factory=lambda: deque(maxlen=200))

    # Auto-segmentation
    auto_seg_active: bool = False
    manual_zones_set: bool = False
    last_auto_seg_time: float = 0.0

    # Internal
    lock: threading.Lock = field(default_factory=threading.Lock)
    _needs_mask_regen: bool = False


# ---------------------------------------------------------------------------
# FeedManager
# ---------------------------------------------------------------------------

class FeedManager:
    """
    Central registry for all camera feeds.

    Thread safety: all public methods that touch per-feed state acquire
    ``feed.lock`` internally. The caller should NOT hold the lock.
    """

    def __init__(self) -> None:
        self._feeds: Dict[str, FeedState] = {}
        self._cameras: Dict[str, CameraBackend] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Feed registration
    # ------------------------------------------------------------------

    def register_feed(
        self,
        feed_id: str,
        name: str,
        location: str,
        camera: CameraBackend,
        scene_type: Optional[str] = None,
    ) -> None:
        """Add a feed to the registry. Idempotent — re-registering replaces the entry."""
        state = FeedState(
            feed_id=feed_id,
            name=name,
            location=location,
            scene_type=scene_type,
        )
        with self._lock:
            self._feeds[feed_id] = state
            self._cameras[feed_id] = camera

    def feed_ids(self) -> List[str]:
        with self._lock:
            return list(self._feeds.keys())

    def get_state(self, feed_id: str) -> Optional[FeedState]:
        return self._feeds.get(feed_id)

    def get_camera(self, feed_id: str) -> Optional[CameraBackend]:
        return self._cameras.get(feed_id)

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def store_frame(
        self,
        feed_id: str,
        frame: np.ndarray,
        position: Optional[Tuple[float, float, float]] = None,
        jpeg_bytes: Optional[bytes] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        """
        Store a freshly captured frame (and optional position) for a feed.

        Called from the capture thread. Thread-safe.
        """
        feed = self._feeds.get(feed_id)
        if feed is None:
            return
        with feed.lock:
            feed.last_frame = frame
            feed.frame_count += 1
            if position is not None:
                feed.position = position
            if jpeg_bytes is not None and timestamp is not None:
                feed.replay_buffer.append((timestamp, jpeg_bytes))

    def get_frame(self, feed_id: str) -> Optional[np.ndarray]:
        """Return a copy of the latest frame, or None."""
        feed = self._feeds.get(feed_id)
        if feed is None:
            return None
        with feed.lock:
            return feed.last_frame.copy() if feed.last_frame is not None else None

    # ------------------------------------------------------------------
    # Detection state updates
    # ------------------------------------------------------------------

    def update_detection(
        self,
        feed_id: str,
        alarm_active: bool,
        caution_active: bool,
        people_count: int,
        danger_count: int,
        caution_count: int,
        target_coordinates: Optional[Tuple[float, float, float]] = None,
        mask_overlay: Optional[np.ndarray] = None,
    ) -> None:
        """
        Store detection results for a feed. Called from the detection thread.
        """
        feed = self._feeds.get(feed_id)
        if feed is None:
            return
        with feed.lock:
            feed.alarm_active = alarm_active
            feed.caution_active = caution_active
            feed.people_count = people_count
            feed.danger_count = danger_count
            feed.caution_count = caution_count
            feed.last_detection_time = datetime.now()
            if not alarm_active:
                feed.target_coordinates = None
            elif target_coordinates is not None:
                feed.target_coordinates = target_coordinates
            if mask_overlay is not None:
                feed.last_mask_overlay = mask_overlay

    # ------------------------------------------------------------------
    # Zone management
    # ------------------------------------------------------------------

    def update_zones(
        self,
        feed_id: str,
        zones: List[Zone],
        image_width: int,
        image_height: int,
    ) -> None:
        """
        Replace zones for a feed and regenerate binary masks.
        """
        feed = self._feeds.get(feed_id)
        if feed is None:
            raise ValueError(f"Feed {feed_id!r} not found")
        with feed.lock:
            feed.zones = list(zones)
            feed.zone_manager.update_zones(zones, image_width, image_height)

    def get_zones(self, feed_id: str) -> List[Zone]:
        feed = self._feeds.get(feed_id)
        if feed is None:
            return []
        with feed.lock:
            return list(feed.zones)

    def is_warmed_up(self, feed_id: str, warmup_frames: int) -> bool:
        feed = self._feeds.get(feed_id)
        if feed is None:
            return False
        with feed.lock:
            return feed.frame_count >= warmup_frames

    # ------------------------------------------------------------------
    # Snapshot for API responses
    # ------------------------------------------------------------------

    def snapshot(self, feed_id: str) -> Optional[dict]:
        """
        Return a plain-dict snapshot of a feed's current detection state.
        Safe to call from any thread.
        """
        feed = self._feeds.get(feed_id)
        if feed is None:
            return None
        with feed.lock:
            target = None
            if feed.target_coordinates:
                tx, ty, tz = feed.target_coordinates
                target = {"x": tx, "y": ty, "z": tz}
            pos = None
            if feed.position:
                px, py, pz = feed.position
                pos = {"x": px, "y": py, "z": pz}
            return {
                "feed_id": feed_id,
                "alarm_active": feed.alarm_active,
                "caution_active": feed.caution_active,
                "people_count": feed.people_count,
                "danger_count": feed.danger_count,
                "caution_count": feed.caution_count,
                "target_coordinates": target,
                "last_detection_time": (
                    feed.last_detection_time.isoformat()
                    if feed.last_detection_time else None
                ),
                "position": pos,
            }
