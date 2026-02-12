"""
Backend Server for CCTV Monitoring UI Integration
==================================================

This FastAPI server bridges the React UI with AirSim and the drone control system.

Features:
- Streams CCTV video feeds from AirSim cameras
- Manages danger zones (converts UI polygons to binary masks)
- Runs human detection and overlap checking
- Calculates 3D world coordinates when humans enter RED danger zones
- Provides detection status to the UI
- Persists zones to JSON file

Zone Behavior:
- RED zones: Trigger alarm + calculate coordinates for drone deployment
- YELLOW zones: Trigger caution alert (highlight only, no drone)
- GREEN zones: Safe zones (no action)

Endpoints:
- GET  /video_feed/{feed_id}     - MJPEG video stream
- GET  /feeds                     - List all feeds with status
- GET  /feeds/{feed_id}/status    - Detection status for a feed
- POST /feeds/{feed_id}/zones     - Save zones for a feed
- GET  /health                    - Health check
"""

import airsim
import cv2
import json
import numpy as np
import os
import requests
import queue
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

# --- CENTRALIZED CONFIGURATION ---
from src.backend.config import (
    FEED_CONFIG, FEED_METADATA,
    ENCODER_PATH, DECODER_PATH,
    CCTV_HEIGHT, SAFE_Z_ALTITUDE,
    FRAME_CAPTURE_INTERVAL, STREAM_INTERVAL,
    DETECTION_INTERVAL, ALARM_COOLDOWN,
    DRONE_API_URL, DRONE_API_TIMEOUT,
    DATA_DIR, ZONES_FILE,
    BACKEND_PORT,
    FEED_SCENE_TYPE, SEG_MODEL_PATHS,
    AUTO_SEG_INTERVAL, AUTO_SEG_CONFIDENCE,
    CCTV_FOLLOW_TARGET, CCTV_FOLLOW_CAMERAS,
    CCTV_FOLLOW_INTERVAL,
)

# --- DETECTION MODULES (optional — graceful fallback) ---
# Human detection (YOLO) and depth estimation are imported separately so that
# a failure in one does not disable the other.
try:
    from src.human_detection.detector import HumanDetector
    from src.human_detection.check_overlap import check_danger_zone_overlap
    import src.human_detection.config as config
    DETECTION_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Human detection modules not available: {e}")
    DETECTION_AVAILABLE = False

try:
    from src.cctv_monitoring import depth_estimation_utils, coord_utils
    DEPTH_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Depth estimation modules not available: {e}")
    DEPTH_AVAILABLE = False

# --- AUTO-SEGMENTATION MODULE (optional) ---
try:
    from src.backend.auto_segmentation import SceneSegmenter
    AUTO_SEG_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Auto-segmentation module not available: {e}")
    AUTO_SEG_AVAILABLE = False

# ============================================================================
# PERSISTENCE FUNCTIONS
# ============================================================================

def ensure_data_dir():
    """Ensure the data directory exists."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"[STORAGE] Created data directory: {DATA_DIR}")

def load_zones_from_file() -> Dict[str, List[dict]]:
    """Load zones from JSON file."""
    ensure_data_dir()

    if not os.path.exists(ZONES_FILE):
        print("[STORAGE] No zones file found, starting fresh")
        return {}

    try:
        with open(ZONES_FILE, 'r') as f:
            data = json.load(f)
            print(f"[STORAGE] Loaded zones for {len(data)} feeds from {ZONES_FILE}")
            return data
    except Exception as e:
        print(f"[STORAGE] Error loading zones: {e}")
        return {}

def save_zones_to_file(feed_id: str, zones: List[dict]):
    """Save zones to JSON file."""
    ensure_data_dir()

    # Load existing data
    all_zones = load_zones_from_file()

    # Update with new zones
    all_zones[feed_id] = zones

    try:
        with open(ZONES_FILE, 'w') as f:
            json.dump(all_zones, f, indent=2)
        print(f"[STORAGE] Saved {len(zones)} zones for {feed_id}")
    except Exception as e:
        print(f"[STORAGE] Error saving zones: {e}")

# ============================================================================
# DRONE API CLIENT
# ============================================================================

class DroneAPIClient:
    """Client for communicating with the drone control REST API."""

    def __init__(self, base_url=DRONE_API_URL, timeout=DRONE_API_TIMEOUT):
        self.base_url = base_url
        self.timeout = timeout

    def check_connection(self):
        """Check if drone API is running."""
        try:
            response = requests.get(f"{self.base_url}/status", timeout=self.timeout)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def set_mode(self, mode):
        """Set drone control mode (manual/automatic)."""
        try:
            response = requests.post(
                f"{self.base_url}/mode",
                json={"mode": mode},
                timeout=self.timeout
            )
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            print(f"[DRONE] Failed to set mode: {e}")
            return False

    def goto_position(self, x, y, z):
        """Send drone to target position (NED coordinates)."""
        try:
            response = requests.post(
                f"{self.base_url}/goto",
                json={"x": float(x), "y": float(y), "z": float(z)},
                timeout=self.timeout
            )
            if response.status_code == 200:
                return True
            else:
                print(f"[DRONE] Goto failed: {response.json()}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"[DRONE] Failed to send goto command: {e}")
            return False

    def get_status(self):
        """Get current drone status."""
        try:
            response = requests.get(f"{self.base_url}/status", timeout=self.timeout)
            if response.status_code == 200:
                return response.json()
            return None
        except requests.exceptions.RequestException:
            return None

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class Point(BaseModel):
    x: float  # Percentage (0-100)
    y: float  # Percentage (0-100)

class Zone(BaseModel):
    id: str
    level: str  # 'red', 'yellow', 'green'
    points: List[Point]

class ZonesUpdateRequest(BaseModel):
    zones: List[Zone]

class TargetCoordinate(BaseModel):
    x: float  # North (meters)
    y: float  # East (meters)
    z: float  # Down (meters, negative = above ground)

class DetectionStatus(BaseModel):
    feed_id: str
    alarm_active: bool           # RED zone intrusion - drone deployment
    caution_active: bool         # YELLOW zone intrusion - highlight only
    people_count: int
    danger_count: int            # People in RED zones
    caution_count: int           # People in YELLOW zones
    target_coordinates: Optional[TargetCoordinate] = None
    last_detection_time: Optional[str] = None

# ============================================================================
# FEED STATE MANAGEMENT
# ============================================================================

@dataclass
class FeedState:
    """State for a single feed."""
    feed_id: str
    camera_name: str
    vehicle_name: str
    name: str
    location: str
    zones: List[Zone] = field(default_factory=list)
    red_zone_mask: Optional[np.ndarray] = None      # Binary mask for RED zones
    yellow_zone_mask: Optional[np.ndarray] = None   # Binary mask for YELLOW zones
    alarm_active: bool = False          # RED zone intrusion
    caution_active: bool = False        # YELLOW zone intrusion
    people_count: int = 0
    danger_count: int = 0               # People in RED zones
    caution_count: int = 0              # People in YELLOW zones
    target_coordinates: Optional[Tuple[float, float, float]] = None
    last_frame: Optional[np.ndarray] = None
    last_detection_time: Optional[datetime] = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    # Auto-segmentation fields
    scene_type: Optional[str] = None          # "ship", "railway", "bridge", or None
    auto_seg_active: bool = False             # Whether auto-seg is running for this feed
    manual_zones_set: bool = False            # True if user has manually edited zones (for railway/bridge)
    last_auto_seg_time: float = 0.0           # Timestamp of last auto-seg run

class FeedManager:
    """Manages all feed states."""

    def __init__(self):
        self.feeds: Dict[str, FeedState] = {}
        self.client: Optional[airsim.MultirotorClient] = None
        self.detector: Optional[HumanDetector] = None
        self.depth_model = None
        self.drone_api: Optional[DroneAPIClient] = None
        self.drone_is_navigating = False
        self.last_deployment_time = 0.0
        self.running = False
        self._init_lock = threading.Lock()
        self._airsim_lock = threading.Lock()
        self._depth_queue: queue.Queue = queue.Queue(maxsize=4)
        self.segmenter: Optional[SceneSegmenter] = None if AUTO_SEG_AVAILABLE else None

    def initialize(self):
        """Initialize AirSim connection and models."""
        with self._init_lock:
            if self.client is not None:
                return True

            try:
                # Connect to AirSim
                print("[INIT] Connecting to AirSim...")
                self.client = airsim.MultirotorClient()
                self.client.confirmConnection()
                print("[INIT] AirSim connected")

                # Initialize feeds
                for feed_id, (camera_name, vehicle_name) in FEED_CONFIG.items():
                    metadata = FEED_METADATA.get(feed_id, {"name": feed_id, "location": "Unknown"})
                    self.feeds[feed_id] = FeedState(
                        feed_id=feed_id,
                        camera_name=camera_name,
                        vehicle_name=vehicle_name,
                        name=metadata["name"],
                        location=metadata["location"]
                    )

                # Set scene type per feed from config
                for feed_id_key in self.feeds:
                    self.feeds[feed_id_key].scene_type = FEED_SCENE_TYPE.get(feed_id_key)

                # Load persisted zones
                self._load_persisted_zones()

                # Load auto-segmentation models if available
                if AUTO_SEG_AVAILABLE:
                    needed_types = set(t for t in FEED_SCENE_TYPE.values() if t)
                    needed_paths = {t: SEG_MODEL_PATHS[t] for t in needed_types if t in SEG_MODEL_PATHS}
                    if needed_paths:
                        print("[INIT] Loading scene segmentation models...")
                        self.segmenter = SceneSegmenter(needed_paths, confidence=AUTO_SEG_CONFIDENCE)
                        print(f"[INIT] Loaded {len(self.segmenter.models)} segmentation model(s)")
                    else:
                        print("[INIT] No segmentation model paths configured")

                # Load human detection model if available
                if DETECTION_AVAILABLE:
                    print("[INIT] Loading YOLO model...")
                    self.detector = HumanDetector()
                else:
                    print("[INIT] Human detection not available (import failed)")

                # Load depth model if available (independent of human detection)
                if DEPTH_AVAILABLE:
                    if os.path.exists(ENCODER_PATH) and os.path.exists(DECODER_PATH):
                        print("[INIT] Loading depth model...")
                        self.depth_model = depth_estimation_utils.load_lite_mono_model(
                            ENCODER_PATH, DECODER_PATH
                        )
                    else:
                        print("[INIT] Depth model weights not found, coordinate estimation disabled")
                else:
                    print("[INIT] Depth estimation not available (import failed)")

                # Connect to drone API
                print("[INIT] Connecting to drone API...")
                self.drone_api = DroneAPIClient()
                if self.drone_api.check_connection():
                    print("[INIT] Drone API connected")
                    if self.drone_api.set_mode("automatic"):
                        print("[INIT] Drone set to automatic mode")
                    else:
                        print("[INIT] Failed to set drone to automatic mode")
                else:
                    print("[INIT] Drone API not available, auto-deployment disabled")
                    self.drone_api = None

                return True

            except Exception as e:
                print(f"[INIT] Failed to initialize: {e}")
                self.client = None
                return False

    def _load_persisted_zones(self):
        """Load zones from file and apply to feeds."""
        saved_zones = load_zones_from_file()

        for feed_id, zones_data in saved_zones.items():
            if feed_id in self.feeds:
                # Convert dict to Zone objects
                zones = [Zone(**z) for z in zones_data]
                self.feeds[feed_id].zones = zones
                # Mark that masks need regeneration on first frame
                self.feeds[feed_id]._needs_mask_regen = True
                print(f"[INIT] Restored {len(zones)} zones for {feed_id}")

    def _regenerate_masks_if_needed(self, feed_id: str, frame_width: int, frame_height: int):
        """Regenerate binary masks from zones if needed (e.g., after server restart)."""
        if feed_id not in self.feeds:
            return

        feed = self.feeds[feed_id]

        # Check if regeneration is needed
        if not getattr(feed, '_needs_mask_regen', False):
            return

        if len(feed.zones) == 0:
            feed._needs_mask_regen = False
            return

        with feed.lock:
            # Separate zones by level
            red_zones = [z for z in feed.zones if z.level == 'red']
            yellow_zones = [z for z in feed.zones if z.level == 'yellow']

            # Create RED zone mask
            if red_zones:
                red_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                for zone in red_zones:
                    pts = np.array([
                        [int(p.x * frame_width / 100), int(p.y * frame_height / 100)]
                        for p in zone.points
                    ], dtype=np.int32)
                    cv2.fillPoly(red_mask, [pts], 1)
                feed.red_zone_mask = red_mask
                print(f"[ZONES] Regenerated {len(red_zones)} RED zone mask(s) for {feed_id}")
            else:
                feed.red_zone_mask = None

            # Create YELLOW zone mask
            if yellow_zones:
                yellow_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                for zone in yellow_zones:
                    pts = np.array([
                        [int(p.x * frame_width / 100), int(p.y * frame_height / 100)]
                        for p in zone.points
                    ], dtype=np.int32)
                    cv2.fillPoly(yellow_mask, [pts], 1)
                feed.yellow_zone_mask = yellow_mask
                print(f"[ZONES] Regenerated {len(yellow_zones)} YELLOW zone mask(s) for {feed_id}")
            else:
                feed.yellow_zone_mask = None

            feed._needs_mask_regen = False

    def get_frame(self, feed_id: str) -> Optional[np.ndarray]:
        """Get the latest frame from a feed."""
        if feed_id not in self.feeds:
            return None

        feed = self.feeds[feed_id]

        if self.client is None:
            return None

        try:
            with self._airsim_lock:
                responses = self.client.simGetImages([
                    airsim.ImageRequest(feed.camera_name, airsim.ImageType.Scene, False, False)
                ], vehicle_name=feed.vehicle_name)

            if not responses or len(responses) == 0:
                return None

            response = responses[0]
            if len(response.image_data_uint8) == 0:
                return None

            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape(response.height, response.width, 3)

            with feed.lock:
                feed.last_frame = img_rgb

            return img_rgb

        except Exception as e:
            print(f"[ERROR] Failed to get frame for {feed_id}: {e}")
            return None

    def update_zones(self, feed_id: str, zones: List[Zone], image_width: int, image_height: int):
        """Update zones for a feed and regenerate binary masks."""
        if feed_id not in self.feeds:
            raise ValueError(f"Feed {feed_id} not found")

        feed = self.feeds[feed_id]

        with feed.lock:
            feed.zones = zones

            # Separate zones by level
            red_zones = [z for z in zones if z.level == 'red']
            yellow_zones = [z for z in zones if z.level == 'yellow']

            # Create RED zone mask (triggers alarm + drone)
            if red_zones:
                red_mask = np.zeros((image_height, image_width), dtype=np.uint8)
                for zone in red_zones:
                    pts = np.array([
                        [int(p.x * image_width / 100), int(p.y * image_height / 100)]
                        for p in zone.points
                    ], dtype=np.int32)
                    cv2.fillPoly(red_mask, [pts], 1)
                feed.red_zone_mask = red_mask
                print(f"[ZONES] {feed_id}: {len(red_zones)} RED zone(s)")
            else:
                feed.red_zone_mask = None

            # Create YELLOW zone mask (triggers caution alert only)
            if yellow_zones:
                yellow_mask = np.zeros((image_height, image_width), dtype=np.uint8)
                for zone in yellow_zones:
                    pts = np.array([
                        [int(p.x * image_width / 100), int(p.y * image_height / 100)]
                        for p in zone.points
                    ], dtype=np.int32)
                    cv2.fillPoly(yellow_mask, [pts], 1)
                feed.yellow_zone_mask = yellow_mask
                print(f"[ZONES] {feed_id}: {len(yellow_zones)} YELLOW zone(s)")
            else:
                feed.yellow_zone_mask = None

        # Persist to file
        zones_data = [z.model_dump() for z in zones]
        save_zones_to_file(feed_id, zones_data)

    def run_detection(self, feed_id: str) -> DetectionStatus:
        """Run detection on a feed and update its status."""
        if feed_id not in self.feeds:
            raise ValueError(f"Feed {feed_id} not found")

        feed = self.feeds[feed_id]

        # Get latest frame
        frame = self.get_frame(feed_id)
        if frame is None:
            return self._get_status(feed)

        # Regenerate masks if needed (e.g., after server restart)
        self._regenerate_masks_if_needed(feed_id, frame.shape[1], frame.shape[0])

        # Snapshot masks under lock so detection uses a stable copy even if
        # the API thread clears them (TOCTOU race fix).
        with feed.lock:
            red_zone_mask = (feed.red_zone_mask.copy()
                            if feed.red_zone_mask is not None else None)
            yellow_zone_mask = (feed.yellow_zone_mask.copy()
                                if feed.yellow_zone_mask is not None else None)

        if self.detector is None:
            return self._get_status(feed)

        try:
            # Run human detection
            person_masks = self.detector.get_masks(frame)

            with feed.lock:
                feed.people_count = len(person_masks)
                feed.last_detection_time = datetime.now()

                # Reset counts
                feed.alarm_active = False
                feed.caution_active = False
                feed.danger_count = 0
                feed.caution_count = 0

                if len(person_masks) > 0:
                    # Check RED zone overlap (alarm + drone deployment)
                    # Uses local red_zone_mask snapshot (not feed.red_zone_mask)
                    if red_zone_mask is not None:
                        # Ensure mask matches frame dimensions
                        red_mask = red_zone_mask
                        if red_mask.shape != frame.shape[:2]:
                            red_mask = cv2.resize(
                                red_mask,
                                (frame.shape[1], frame.shape[0]),
                                interpolation=cv2.INTER_NEAREST
                            )

                        is_alarm, danger_masks = check_danger_zone_overlap(person_masks, red_mask)
                        feed.alarm_active = is_alarm
                        feed.danger_count = len(danger_masks)

                        # Enqueue depth + drone work (async — doesn't block detection)
                        if is_alarm and len(danger_masks) > 0 and self.depth_model is not None:
                            target_mask = danger_masks[0]
                            feet_pixel = coord_utils.get_feet_from_mask(target_mask)
                            if feet_pixel is not None:
                                try:
                                    job = (feed_id, frame.copy(), feet_pixel)
                                    self._depth_queue.put_nowait(job)
                                except queue.Full:
                                    pass  # drop — next cycle will retry

                    # Check YELLOW zone overlap (caution alert only - NO drone)
                    # Uses local yellow_zone_mask snapshot (not feed.yellow_zone_mask)
                    if yellow_zone_mask is not None:
                        y_mask = yellow_zone_mask
                        if y_mask.shape != frame.shape[:2]:
                            y_mask = cv2.resize(
                                y_mask,
                                (frame.shape[1], frame.shape[0]),
                                interpolation=cv2.INTER_NEAREST
                            )

                        # Subtract red zone so red always takes priority over yellow
                        effective_yellow = y_mask
                        if red_zone_mask is not None:
                            red_for_subtract = red_zone_mask
                            if red_for_subtract.shape != y_mask.shape:
                                red_for_subtract = cv2.resize(
                                    red_for_subtract,
                                    (y_mask.shape[1], y_mask.shape[0]),
                                    interpolation=cv2.INTER_NEAREST
                                )
                            effective_yellow = y_mask & (~red_for_subtract)

                        is_caution, caution_masks = check_danger_zone_overlap(person_masks, effective_yellow)
                        feed.caution_active = is_caution
                        feed.caution_count = len(caution_masks)

                # Clear target coordinates if no RED zone alarm
                if not feed.alarm_active:
                    feed.target_coordinates = None

        except Exception as e:
            print(f"[ERROR] Detection failed for {feed_id}: {e}")
            import traceback
            traceback.print_exc()

        return self._get_status(feed)

    def run_detection_only(self, feed_id: str) -> DetectionStatus:
        """Run detection on cached frame (doesn't fetch new frame from AirSim).

        Use this when frame capture runs on a separate thread.
        """
        if feed_id not in self.feeds:
            raise ValueError(f"Feed {feed_id} not found")

        feed = self.feeds[feed_id]

        # Use cached frame instead of fetching new one; snapshot masks too
        # so the detection thread works on a stable copy even if the API
        # thread clears them (TOCTOU race fix).
        with feed.lock:
            frame = feed.last_frame.copy() if feed.last_frame is not None else None
            red_zone_mask = (feed.red_zone_mask.copy()
                            if feed.red_zone_mask is not None else None)
            yellow_zone_mask = (feed.yellow_zone_mask.copy()
                                if feed.yellow_zone_mask is not None else None)

        if frame is None:
            return self._get_status(feed)

        # Regenerate masks if needed (e.g., after server restart)
        self._regenerate_masks_if_needed(feed_id, frame.shape[1], frame.shape[0])

        if self.detector is None:
            return self._get_status(feed)

        try:
            # Run human detection
            person_masks = self.detector.get_masks(frame)

            with feed.lock:
                feed.people_count = len(person_masks)
                feed.last_detection_time = datetime.now()

                # Reset counts
                feed.alarm_active = False
                feed.caution_active = False
                feed.danger_count = 0
                feed.caution_count = 0

                if len(person_masks) > 0:
                    # Check RED zone overlap (alarm + drone deployment)
                    # Uses local red_zone_mask snapshot (not feed.red_zone_mask)
                    if red_zone_mask is not None:
                        red_mask = red_zone_mask
                        if red_mask.shape != frame.shape[:2]:
                            red_mask = cv2.resize(
                                red_mask,
                                (frame.shape[1], frame.shape[0]),
                                interpolation=cv2.INTER_NEAREST
                            )

                        is_alarm, danger_masks = check_danger_zone_overlap(person_masks, red_mask)
                        feed.alarm_active = is_alarm
                        feed.danger_count = len(danger_masks)

                        # Enqueue depth + drone work (async — doesn't block detection)
                        if is_alarm and len(danger_masks) > 0 and self.depth_model is not None:
                            target_mask = danger_masks[0]
                            feet_pixel = coord_utils.get_feet_from_mask(target_mask)
                            if feet_pixel is not None:
                                try:
                                    job = (feed_id, frame.copy(), feet_pixel)
                                    self._depth_queue.put_nowait(job)
                                except queue.Full:
                                    pass  # drop — next cycle will retry

                    # Check YELLOW zone overlap (caution alert only - NO drone)
                    # Uses local yellow_zone_mask snapshot (not feed.yellow_zone_mask)
                    if yellow_zone_mask is not None:
                        y_mask = yellow_zone_mask
                        if y_mask.shape != frame.shape[:2]:
                            y_mask = cv2.resize(
                                y_mask,
                                (frame.shape[1], frame.shape[0]),
                                interpolation=cv2.INTER_NEAREST
                            )

                        # Subtract red zone so red always takes priority over yellow
                        effective_yellow = y_mask
                        if red_zone_mask is not None:
                            red_for_subtract = red_zone_mask
                            if red_for_subtract.shape != y_mask.shape:
                                red_for_subtract = cv2.resize(
                                    red_for_subtract,
                                    (y_mask.shape[1], y_mask.shape[0]),
                                    interpolation=cv2.INTER_NEAREST
                                )
                            effective_yellow = y_mask & (~red_for_subtract)

                        is_caution, caution_masks = check_danger_zone_overlap(person_masks, effective_yellow)
                        feed.caution_active = is_caution
                        feed.caution_count = len(caution_masks)

                # Clear target coordinates if no RED zone alarm
                if not feed.alarm_active:
                    feed.target_coordinates = None

        except Exception as e:
            print(f"[ERROR] Detection failed for {feed_id}: {e}")
            import traceback
            traceback.print_exc()

        return self._get_status(feed)

    def _get_status(self, feed: FeedState) -> DetectionStatus:
        """Get detection status for a feed."""
        with feed.lock:
            target = None
            if feed.target_coordinates:
                target = TargetCoordinate(
                    x=feed.target_coordinates[0],
                    y=feed.target_coordinates[1],
                    z=feed.target_coordinates[2]
                )

            return DetectionStatus(
                feed_id=feed.feed_id,
                alarm_active=feed.alarm_active,
                caution_active=feed.caution_active,
                people_count=feed.people_count,
                danger_count=feed.danger_count,
                caution_count=feed.caution_count,
                target_coordinates=target,
                last_detection_time=feed.last_detection_time.isoformat() if feed.last_detection_time else None
            )

    def run_auto_segmentation(self, feed_id: str) -> List[Zone]:
        """Run auto-segmentation on a feed's current frame and update zones.

        Returns the list of auto-generated Zone objects (empty if no detections).
        """
        if feed_id not in self.feeds:
            return []

        feed = self.feeds[feed_id]

        if not feed.scene_type or self.segmenter is None:
            return []

        # Get current cached frame
        with feed.lock:
            frame = feed.last_frame.copy() if feed.last_frame is not None else None

        if frame is None:
            return []

        # Run segmentation
        zone_dicts = self.segmenter.segment_frame(frame, feed.scene_type)

        # Always update timestamp to maintain the interval (even if no zones found)
        feed.last_auto_seg_time = time.time()
        feed.auto_seg_active = False

        if not zone_dicts:
            print(f"[AUTO-SEG] No hazard zones detected in {feed_id}")
            return []

        # Convert dicts to Zone objects
        zones = [Zone(**z) for z in zone_dicts]

        # Apply zones (generates binary masks and persists to file)
        self.update_zones(feed_id, zones, frame.shape[1], frame.shape[0])

        print(f"[AUTO-SEG] {feed_id} ({feed.scene_type}): Generated {len(zones)} zone(s)")
        return zones

# Global feed manager
feed_manager = FeedManager()

# ============================================================================
# BACKGROUND THREADS: FRAME CAPTURE + DETECTION
# ============================================================================

def frame_capture_loop():
    """High-FPS frame capture thread — grabs frames from AirSim without detection.

    This runs at FRAME_CAPTURE_FPS (default 30) to ensure smooth video streaming.
    Detection runs separately at a lower rate.
    """
    print(f"[CAPTURE] Starting frame capture loop ({1/FRAME_CAPTURE_INTERVAL:.0f} FPS)...")

    while feed_manager.running:
        for feed_id in feed_manager.feeds:
            if not feed_manager.running:
                break
            try:
                # Just grab the frame, don't run detection
                feed_manager.get_frame(feed_id)
            except Exception as e:
                print(f"[CAPTURE] Error capturing {feed_id}: {e}")

        time.sleep(FRAME_CAPTURE_INTERVAL)

    print("[CAPTURE] Frame capture loop stopped")


def detection_loop():
    """Detection thread — runs human detection at lower FPS.

    This runs at DETECTION_FPS (default 10) since detection is CPU/GPU intensive.
    Reads from cached frames populated by the capture thread.
    """
    print(f"[DETECTION] Starting detection loop ({1/DETECTION_INTERVAL:.0f} FPS)...")
    last_drone_status_check = 0.0

    while feed_manager.running:
        for feed_id in feed_manager.feeds:
            if not feed_manager.running:
                break
            try:
                feed_manager.run_detection_only(feed_id)
            except Exception as e:
                print(f"[DETECTION] Error processing {feed_id}: {e}")

        # Poll drone status (rate-limited to once per second)
        current_time = time.time()
        if (feed_manager.drone_is_navigating
                and feed_manager.drone_api is not None
                and current_time - last_drone_status_check >= 1.0):
            last_drone_status_check = current_time
            try:
                status = feed_manager.drone_api.get_status()
                if status and not status.get("is_navigating", True):
                    print("[DRONE] Drone reached target, mission complete")
                    feed_manager.drone_is_navigating = False
            except Exception as e:
                print(f"[DRONE] Error checking drone status: {e}")

        time.sleep(DETECTION_INTERVAL)

    print("[DETECTION] Detection loop stopped")


def auto_segmentation_loop():
    """Background thread for periodic auto-segmentation.

    - Ship feeds: Re-segment every AUTO_SEG_INTERVAL seconds, overwriting all zones.
    - Railway/Bridge feeds: Segment once on first frame, respect manual edits after that.
    """
    print(f"[AUTO-SEG] Starting auto-segmentation loop (ship interval: {AUTO_SEG_INTERVAL}s)...")

    # Wait for initial frames to be captured
    time.sleep(5.0)

    # Track which railway/bridge feeds have had their initial segmentation
    initial_seg_done: set = set()

    while feed_manager.running:
        if feed_manager.segmenter is None:
            time.sleep(5.0)
            continue

        for feed_id, feed in feed_manager.feeds.items():
            if not feed_manager.running:
                break

            if not feed.scene_type:
                continue

            try:
                if feed.scene_type == "ship":
                    # Ship: re-segment every AUTO_SEG_INTERVAL, always overwrite
                    current_time = time.time()
                    if current_time - feed.last_auto_seg_time >= AUTO_SEG_INTERVAL:
                        feed.auto_seg_active = True
                        feed_manager.run_auto_segmentation(feed_id)

                elif feed.scene_type in ("railway", "bridge"):
                    # Railway/Bridge: segment once initially, skip if user has set manual zones
                    if feed_id not in initial_seg_done and not feed.manual_zones_set:
                        with feed.lock:
                            has_frame = feed.last_frame is not None
                        if has_frame:
                            feed_manager.run_auto_segmentation(feed_id)
                            initial_seg_done.add(feed_id)
                            print(f"[AUTO-SEG] Initial segmentation done for {feed_id} ({feed.scene_type})")

            except Exception as e:
                print(f"[AUTO-SEG] Error processing {feed_id}: {e}")

        time.sleep(5.0)  # Poll interval (actual auto-seg interval enforced per-feed)

    print("[AUTO-SEG] Auto-segmentation loop stopped")


def depth_worker_loop():
    """Depth estimation worker — processes depth + drone dispatch off the detection thread."""
    print("[DEPTH] Starting depth worker thread...")

    while feed_manager.running:
        try:
            feed_id, frame, feet_pixel = feed_manager._depth_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        try:
            cx, cy = feet_pixel
            feed = feed_manager.feeds.get(feed_id)
            if feed is None:
                continue

            # Run depth inference (50-200ms — no longer blocking detection)
            ai_depth_map = depth_estimation_utils.run_lite_mono_inference(
                feed_manager.depth_model, frame
            )
            rel_depth = ai_depth_map[cy, cx]

            # Get camera pose (AirSim lock)
            with feed_manager._airsim_lock:
                target_pos = coord_utils.get_coords_from_lite_mono(
                    feed_manager.client, feed.camera_name, cx, cy,
                    frame.shape[1], frame.shape[0],
                    rel_depth, CCTV_HEIGHT, feed.vehicle_name
                )

            # Update coordinates
            with feed.lock:
                feed.target_coordinates = (
                    target_pos.x_val,
                    target_pos.y_val,
                    SAFE_Z_ALTITUDE
                )

            # Deploy drone (cooldown check)
            if feed_manager.drone_api is not None:
                current_time = time.time()
                cooldown_ok = (current_time - feed_manager.last_deployment_time
                               > ALARM_COOLDOWN)
                if cooldown_ok and not feed_manager.drone_is_navigating:
                    tx, ty, tz = feed.target_coordinates
                    print(f"[DRONE] Deploying to ({tx:.2f}, {ty:.2f}, {tz:.2f})"
                          f" for {feed_id}")
                    if feed_manager.drone_api.goto_position(tx, ty, tz):
                        feed_manager.last_deployment_time = current_time
                        feed_manager.drone_is_navigating = True
                        print("[DRONE] Deployment command sent successfully")
                    else:
                        print("[DRONE] Deployment command failed")

        except Exception as e:
            print(f"[DEPTH] Error processing {feed_id}: {e}")
            import traceback
            traceback.print_exc()

    print("[DEPTH] Depth worker thread stopped")


def cctv_follow_loop():
    """Follow loop — teleports CCTV drones to match Camera Actor positions.

    Uses its own AirSim client to avoid lock contention with the frame
    capture and detection threads. Each tick, reads the pose of the
    Camera Actor guide (placed in UE) and teleports the drone to that
    exact position and orientation.
    Only runs when CCTV_FOLLOW_TARGET is set via --follow flag.
    """
    import math

    # Create a dedicated AirSim client for the follow loop to avoid
    # lock contention with the frame capture / detection threads.
    try:
        print("[FOLLOW] Connecting dedicated AirSim client...")
        client = airsim.MultirotorClient()
        client.confirmConnection()
    except Exception as e:
        print(f"[FOLLOW] Failed to connect AirSim client: {e}")
        return

    # Arm and take off each CCTV drone, collecting futures
    takeoff_futures = []
    for vehicle_name in CCTV_FOLLOW_CAMERAS:
        print(f"[FOLLOW] Arming and taking off {vehicle_name}...")
        client.enableApiControl(True, vehicle_name=vehicle_name)
        client.armDisarm(True, vehicle_name=vehicle_name)
        future = client.takeoffAsync(vehicle_name=vehicle_name)
        takeoff_futures.append((vehicle_name, future))

    # Wait for all takeoffs to actually complete
    for vehicle_name, future in takeoff_futures:
        try:
            future.join()
            print(f"[FOLLOW] {vehicle_name} takeoff complete")
        except Exception as e:
            print(f"[FOLLOW] {vehicle_name} takeoff issue (may already be airborne): {e}")

    # Log camera guide mappings
    for vehicle_name, cam_name in CCTV_FOLLOW_CAMERAS.items():
        print(f"[FOLLOW] {vehicle_name} → Camera Actor '{cam_name}'")

    while feed_manager.running:
        try:
            for vehicle_name, cam_actor_name in CCTV_FOLLOW_CAMERAS.items():
                cam_pose = client.simGetObjectPose(cam_actor_name)

                if math.isnan(cam_pose.position.x_val):
                    continue

                # Teleport drone to the Camera Actor's exact pose
                client.simSetVehiclePose(cam_pose, True, vehicle_name=vehicle_name)

                # Zero out velocity so the flight controller doesn't fight back
                client.moveByVelocityAsync(0, 0, 0, 0.1, vehicle_name=vehicle_name)

        except Exception as e:
            print(f"[FOLLOW] Error: {e}")

        time.sleep(CCTV_FOLLOW_INTERVAL)

    print("[FOLLOW] Follow loop stopped")


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    if feed_manager.initialize():
        feed_manager.running = True

        # Start frame capture thread (high FPS for smooth video)
        capture_thread = threading.Thread(target=frame_capture_loop, daemon=True)
        capture_thread.start()

        # Start detection thread (lower FPS, CPU/GPU intensive)
        detection_thread = threading.Thread(target=detection_loop, daemon=True)
        detection_thread.start()

        # Start depth worker thread (async depth estimation + drone dispatch)
        if DEPTH_AVAILABLE and feed_manager.depth_model is not None:
            depth_thread = threading.Thread(target=depth_worker_loop, daemon=True)
            depth_thread.start()
            print("[SERVER] Depth worker thread started")

        # Start auto-segmentation thread (periodic scene segmentation)
        if feed_manager.segmenter is not None:
            auto_seg_thread = threading.Thread(target=auto_segmentation_loop, daemon=True)
            auto_seg_thread.start()
            print("[SERVER] Auto-segmentation thread started")

        # Start CCTV follow thread if a follow target is configured
        if CCTV_FOLLOW_TARGET:
            follow_thread = threading.Thread(target=cctv_follow_loop, daemon=True)
            follow_thread.start()
            print(f"[SERVER] CCTV follow mode active — tracking '{CCTV_FOLLOW_TARGET}'")

        print("[SERVER] Backend server started successfully")
    else:
        print("[SERVER] Backend started in limited mode (no AirSim connection)")

    yield

    # Shutdown
    feed_manager.running = False
    print("[SERVER] Backend server shutting down")


app = FastAPI(title="CCTV Monitoring Backend", version="1.0.0", lifespan=lifespan)

# Enable CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "airsim_connected": feed_manager.client is not None,
        "detection_available": DETECTION_AVAILABLE and feed_manager.detector is not None,
        "depth_available": DEPTH_AVAILABLE and feed_manager.depth_model is not None,
        "drone_api_connected": feed_manager.drone_api is not None,
        "drone_navigating": feed_manager.drone_is_navigating,
        "feeds_count": len(feed_manager.feeds)
    }

@app.get("/feeds")
async def list_feeds():
    """List all available feeds with their current status."""
    feeds = []
    for feed_id, feed in feed_manager.feeds.items():
        status = feed_manager._get_status(feed)
        feeds.append({
            "id": feed_id,
            "name": feed.name,
            "location": feed.location,
            "imageSrc": f"http://localhost:{BACKEND_PORT}/video_feed/{feed_id}",
            "zones": [z.model_dump() for z in feed.zones],
            "isLive": True,
            "status": status.model_dump(),
            "sceneType": feed.scene_type,
            "autoSegActive": feed.auto_seg_active,
        })
    return {"feeds": feeds}

def _create_no_signal_frame(width: int = 640, height: int = 480) -> bytes:
    """Create a 'NO SIGNAL' placeholder JPEG."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (30, 30, 30)
    text = "NO SIGNAL"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 1.5, 3)
    cv2.putText(frame, text, ((width - tw) // 2, (height + th) // 2),
                font, 1.5, (100, 100, 100), 3)
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes()

_NO_SIGNAL_JPEG = _create_no_signal_frame()

def generate_mjpeg_frames(feed_id: str):
    """Generate MJPEG frames for video streaming.

    Reads cached frames from feed.last_frame (populated by the detection
    thread) instead of calling AirSim directly, avoiding concurrent
    client access issues.
    """
    while True:
        feed = feed_manager.feeds.get(feed_id)

        frame = None
        if feed:
            with feed.lock:
                if feed.last_frame is not None:
                    frame = feed.last_frame.copy()

        if frame is not None:
            # Convert RGB to BGR for OpenCV encoding
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            h, w = frame_bgr.shape[:2]

            # Draw YELLOW caution border (if caution active)
            if feed.caution_active and not feed.alarm_active:
                cv2.rectangle(frame_bgr, (0, 0), (w-1, h-1), (0, 255, 255), 4)
                cv2.putText(frame_bgr, "CAUTION", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Draw RED alarm border (if alarm active - takes priority)
            if feed.alarm_active:
                cv2.rectangle(frame_bgr, (0, 0), (w-1, h-1), (0, 0, 255), 4)
                cv2.putText(frame_bgr, "ALARM", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Encode as JPEG
            ret, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            # No frame yet — yield placeholder so the stream doesn't hang
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + _NO_SIGNAL_JPEG + b'\r\n')

        time.sleep(STREAM_INTERVAL)

@app.get("/video_feed/{feed_id}")
async def video_feed(feed_id: str):
    """Stream video feed as MJPEG."""
    if feed_id not in feed_manager.feeds:
        raise HTTPException(status_code=404, detail=f"Feed {feed_id} not found")

    return StreamingResponse(
        generate_mjpeg_frames(feed_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/feeds/{feed_id}/status")
async def get_feed_status(feed_id: str):
    """Get detection status for a specific feed."""
    if feed_id not in feed_manager.feeds:
        raise HTTPException(status_code=404, detail=f"Feed {feed_id} not found")

    feed = feed_manager.feeds[feed_id]
    return feed_manager._get_status(feed).model_dump()

@app.post("/feeds/{feed_id}/zones")
async def update_zones(feed_id: str, request: ZonesUpdateRequest):
    """Update zones for a feed."""
    if feed_id not in feed_manager.feeds:
        raise HTTPException(status_code=404, detail=f"Feed {feed_id} not found")

    try:
        feed = feed_manager.feeds[feed_id]

        # Track manual edits: for railway/bridge this prevents auto-seg from overwriting
        # For ship feeds, manual_zones_set stays False (ship always auto-overwrites)
        if feed.scene_type not in ("ship",):
            feed.manual_zones_set = True

        # Try to get frame dimensions for immediate mask generation
        frame = feed_manager.get_frame(feed_id)
        if frame is not None:
            feed_manager.update_zones(feed_id, request.zones, frame.shape[1], frame.shape[0])
        else:
            # No frame yet — save zones and defer mask generation
            feed = feed_manager.feeds[feed_id]
            with feed.lock:
                feed.zones = request.zones
                feed.red_zone_mask = None
                feed.yellow_zone_mask = None
                feed._needs_mask_regen = True
            zones_data = [z.model_dump() for z in request.zones]
            save_zones_to_file(feed_id, zones_data)
            print(f"[ZONES] {feed_id}: Saved {len(request.zones)} zone(s), masks deferred until frame available")

        return {"status": "success", "zones_count": len(request.zones)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feeds/{feed_id}/auto-segment")
async def trigger_auto_segment(feed_id: str):
    """Trigger on-demand auto-segmentation for a feed."""
    if feed_id not in feed_manager.feeds:
        raise HTTPException(status_code=404, detail=f"Feed {feed_id} not found")

    feed = feed_manager.feeds[feed_id]

    if not feed.scene_type:
        raise HTTPException(status_code=400, detail=f"No scene type configured for {feed_id}")

    if feed_manager.segmenter is None:
        raise HTTPException(status_code=503, detail="Segmentation models not loaded")

    try:
        zones = feed_manager.run_auto_segmentation(feed_id)

        # For railway/bridge, reset manual flag since user explicitly requested auto-seg
        if feed.scene_type in ("railway", "bridge"):
            feed.manual_zones_set = False

        return {
            "status": "success",
            "zones_count": len(zones),
            "scene_type": feed.scene_type,
            "zones": [z.model_dump() for z in zones],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import signal

    print("=" * 60)
    print("CCTV MONITORING BACKEND SERVER")
    print("=" * 60)
    print(f"Starting on http://localhost:{BACKEND_PORT}")
    print(f"API Docs: http://localhost:{BACKEND_PORT}/docs")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    # Ensure Ctrl+C kills the process on Windows
    signal.signal(signal.SIGINT, lambda *_: os._exit(0))

    uvicorn.run(app, host="0.0.0.0", port=BACKEND_PORT, log_level="info")
