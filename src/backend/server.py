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

import asyncio
import airsim
import cv2
import json
import numpy as np
import os
import requests
import queue
import threading
from collections import deque
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

# --- CENTRALIZED CONFIGURATION ---
# Legacy flat config — kept until server.py is decomposed into src/api/ (Phase 5).
from src.backend.config import (
    FEED_CONFIG, FEED_METADATA,
    ENCODER_PATH, DECODER_PATH,
    CCTV_HEIGHT, SAFE_Z_ALTITUDE,
    FRAME_CAPTURE_INTERVAL, STREAM_INTERVAL,
    DETECTION_INTERVAL, ALARM_COOLDOWN,
    DETECTION_WARMUP_FRAMES,
    DRONE_API_URL, DRONE_API_TIMEOUT,
    DATA_DIR, ZONES_FILE,
    BACKEND_PORT,
    FEED_SCENE_TYPE, SEG_MODEL_PATHS,
    AUTO_SEG_INTERVAL, AUTO_SEG_CONFIDENCE,
    CCTV_FOLLOW_TARGET, CCTV_FOLLOW_CAMERAS,
    CCTV_FOLLOW_INTERVAL,
    CCTV_HOVER_DRONES, CCTV_HOVER_ALTITUDE,
)
# New YAML-based config — used by new modules; will fully replace above in Phase 5.
from src.core.config import get_config as _get_config, get_feeds_config as _get_feeds_config

# --- CORE MODULES ---
from src.core.zone_manager import ZoneManager, check_overlap as check_danger_zone_overlap

# --- DETECTION MODULES (optional — graceful fallback) ---
# Human detection (YOLO) and depth estimation are imported separately so that
# a failure in one does not disable the other.
try:
    from src.human_detection.detector import HumanDetector
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

# --- MASK OVERLAY TOGGLE ---
MASK_OVERLAY_ENABLED = os.getenv("DISABLE_MASK_OVERLAY", "") != "1"

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
# PYDANTIC MODELS (imported from src.core.models)
# ============================================================================

from src.core.models import (
    Point,
    Zone,
    ZonesUpdateRequest,
    TargetCoordinate,
    DetectionStatus as _DetectionStatusBase,
)

DetectionStatus = _DetectionStatusBase

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
    last_mask_overlay: Optional[np.ndarray] = None     # Pre-rendered mask image (RGB, same size as frame)
    last_detection_time: Optional[datetime] = None
    position: Optional[Tuple[float, float, float]] = None  # NED coordinates (x, y, z)
    frame_count: int = 0                                    # Total captured frames
    replay_buffer: deque = field(default_factory=lambda: deque(maxlen=200))  # Ring buffer: (timestamp, jpeg_bytes) — trimmed to 5s by time
    lock: threading.Lock = field(default_factory=threading.Lock)
    # Auto-segmentation fields
    scene_type: Optional[str] = None          # "ship", "railway", "bridge", or None
    auto_seg_active: bool = False             # Whether auto-seg is running for this feed
    manual_zones_set: bool = False            # True if user has manually edited zones (for railway/bridge)
    last_auto_seg_time: float = 0.0           # Timestamp of last auto-seg run

@dataclass
class TriggerEvent:
    """A single trigger event — recorded when a RED zone intrusion is detected."""
    id: int
    feed_id: str
    timestamp: str
    coords: Tuple[float, float, float]  # NED deploy coordinates
    snapshot: bytes                       # JPEG bytes
    replay_frames: List[Tuple[str, bytes]] = field(default_factory=list)
    replay_trigger_index: int = 0
    deployed: bool = False


MAX_TRIGGER_HISTORY = 10
TRIGGER_REPLAY_FPS = 10
TRIGGER_COOLDOWN = 15  # seconds — minimum gap between triggers from the same camera


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
        self._first_auto_deployed = False  # True after first auto-deploy; subsequent deploys are manual-only
        self.running = False
        self._init_lock = threading.Lock()
        self._airsim_lock = threading.Lock()
        self._detector_lock = threading.Lock()  # Serialize YOLO inference (CUDA isn't thread-safe)
        self._depth_queue: queue.Queue = queue.Queue(maxsize=4)
        self.segmenter: Optional[SceneSegmenter] = None if AUTO_SEG_AVAILABLE else None
        # Global settings (adjustable at runtime via PATCH /settings)
        self.global_scene_type: str = next(iter(FEED_SCENE_TYPE.values()), "bridge")
        self.auto_refresh: bool = False   # periodic auto-seg loop enabled?
        # Trigger history — stores last N trigger events
        self.trigger_history: List[TriggerEvent] = []
        self._trigger_id_counter: int = 0
        # Post-trigger capture state
        self._post_trigger: Optional[dict] = None  # {trigger_id, feed_id, end_time}
        self._replay_capture_counter: int = 0  # Downsample counter for replay buffer

    def get_trigger_by_id(self, trigger_id: int) -> Optional[TriggerEvent]:
        """Look up a trigger event by ID."""
        for t in self.trigger_history:
            if t.id == trigger_id:
                return t
        return None

    def get_latest_trigger(self) -> Optional[TriggerEvent]:
        """Get the most recent trigger event."""
        return self.trigger_history[-1] if self.trigger_history else None

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
                    print(f"[INIT] Scene types from config: {list(FEED_SCENE_TYPE.values())}")
                    print(f"[INIT] All model paths: { {k: v for k, v in SEG_MODEL_PATHS.items()} }")
                    for stype, spath in needed_paths.items():
                        import os as _os
                        print(f"[INIT] Model '{stype}': {spath} (exists={_os.path.exists(spath)})")
                    if needed_paths:
                        print("[INIT] Loading scene segmentation models...")
                        self.segmenter = SceneSegmenter(needed_paths, confidence=AUTO_SEG_CONFIDENCE)
                        print(f"[INIT] Loaded {len(self.segmenter.models)} segmentation model(s): {list(self.segmenter.models.keys())}")
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
                # Also grab the vehicle pose for coordinates
                pose = self.client.simGetVehiclePose(vehicle_name=feed.vehicle_name)

            if not responses or len(responses) == 0:
                return None

            response = responses[0]
            if len(response.image_data_uint8) == 0:
                return None

            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape(response.height, response.width, 3)

            with feed.lock:
                feed.last_frame = img_rgb
                feed.frame_count += 1
                if pose and not (pose.position.x_val != pose.position.x_val):  # NaN check
                    feed.position = (
                        round(pose.position.x_val, 2),
                        round(pose.position.y_val, 2),
                        round(pose.position.z_val, 2),
                    )

            return img_rgb

        except Exception as e:
            print(f"[ERROR] Failed to get frame for {feed_id}: {e}")
            return None

    def capture_all_frames(self):
        """Grab frames from all feeds in a single lock acquisition.

        Much faster than calling get_frame() per feed (one lock acquire
        instead of N).
        """
        if self.client is None:
            return

        self._replay_capture_counter += 1
        now_iso = datetime.now().isoformat()

        with self._airsim_lock:
            for feed_id, feed in self.feeds.items():
                try:
                    responses = self.client.simGetImages([
                        airsim.ImageRequest(feed.camera_name, airsim.ImageType.Scene, False, False)
                    ], vehicle_name=feed.vehicle_name)

                    if not responses or len(responses) == 0:
                        continue

                    response = responses[0]
                    if len(response.image_data_uint8) == 0:
                        continue

                    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                    img_rgb = img1d.reshape(response.height, response.width, 3)

                    with feed.lock:
                        feed.last_frame = img_rgb

                    # Append to replay ring buffer
                    try:
                        frame_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                        _, buf = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        feed.replay_buffer.append((now_iso, buf.tobytes()))
                    except Exception:
                        pass

                    # Post-trigger: continue capturing into the trigger event's replay (5s post-trigger)
                    if (self._post_trigger
                            and self._post_trigger["feed_id"] == feed_id):
                        try:
                            if time.time() >= self._post_trigger["end_time"]:
                                trigger_evt = self.get_trigger_by_id(self._post_trigger["trigger_id"])
                                total = len(trigger_evt.replay_frames) if trigger_evt else 0
                                print(f"[TRIGGER] Post-trigger capture complete: {total} total frames")
                                self._post_trigger = None
                            else:
                                trigger_evt = self.get_trigger_by_id(self._post_trigger["trigger_id"])
                                if trigger_evt:
                                    frame_bgr2 = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                                    _, buf2 = cv2.imencode('.jpg', frame_bgr2, [cv2.IMWRITE_JPEG_QUALITY, 70])
                                    trigger_evt.replay_frames.append((now_iso, buf2.tobytes()))
                        except Exception:
                            pass

                except Exception as e:
                    print(f"[ERROR] Failed to get frame for {feed_id}: {e}")

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

        if not self._is_feed_warmed_up(feed):
            with feed.lock:
                feed.alarm_active = False
                feed.caution_active = False
                feed.danger_count = 0
                feed.caution_count = 0
            return self._get_status(feed)

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

        # Use cached frame
        with feed.lock:
            frame = feed.last_frame.copy() if feed.last_frame is not None else None

        if frame is None:
            return self._get_status(feed)

        # Regenerate masks if needed (e.g., after server restart) — must
        # happen BEFORE the zoneless check since masks start as None.
        self._regenerate_masks_if_needed(feed_id, frame.shape[1], frame.shape[0])

        # Snapshot masks so the detection thread works on a stable copy
        with feed.lock:
            red_zone_mask = (feed.red_zone_mask.copy()
                            if feed.red_zone_mask is not None else None)
            yellow_zone_mask = (feed.yellow_zone_mask.copy()
                                if feed.yellow_zone_mask is not None else None)

        # Skip expensive YOLO inference if no zones are configured
        if red_zone_mask is None and yellow_zone_mask is None:
            return self._get_status(feed)

        # Snapshot previous state to detect changes (for WebSocket broadcast)
        with feed.lock:
            prev_alarm = feed.alarm_active
            prev_caution = feed.caution_active
            prev_people = feed.people_count
            prev_danger = feed.danger_count
            prev_caution_count = feed.caution_count

        if not self._is_feed_warmed_up(feed):
            with feed.lock:
                feed.alarm_active = False
                feed.caution_active = False
                feed.danger_count = 0
                feed.caution_count = 0
            return self._get_status(feed)

        if self.detector is None:
            return self._get_status(feed)

        try:
            # Run human detection (lock serializes CUDA inference across threads)
            t0 = time.time()
            with self._detector_lock:
                t_lock = time.time()
                person_masks = self.detector.get_masks(frame)
                t_yolo = time.time()
            print(f"[TIMING] {feed_id}: lock_wait={t_lock-t0:.3f}s  yolo={t_yolo-t_lock:.3f}s  total={t_yolo-t0:.3f}s")

            # Pre-render mask overlay (skip if disabled via --no-mask)
            overlay = None
            if MASK_OVERLAY_ENABLED and person_masks:
                h, w = frame.shape[:2]
                overlay = np.zeros((h, w, 3), dtype=np.uint8)
                for mask in person_masks:
                    m = mask if mask.shape[:2] == (h, w) else cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    overlay[m == 1] = (0, 255, 255)  # Cyan in RGB
                    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)

            with feed.lock:
                feed.people_count = len(person_masks)
                feed.last_mask_overlay = overlay
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

        status = self._get_status(feed)

        # Only broadcast over WebSocket when detection state actually changed
        changed = (
            feed.alarm_active != prev_alarm
            or feed.caution_active != prev_caution
            or feed.people_count != prev_people
            or feed.danger_count != prev_danger
            or feed.caution_count != prev_caution_count
        )
        if changed:
            _ws_broadcast_from_thread(status.model_dump_json())

        return status

    def run_detection_batch(self):
        """Run detection on all feeds with zones in a single batch YOLO call.

        Collects frames from feeds that have zones configured, runs one
        batched YOLO inference, then processes results (zone overlap,
        mask overlay, WebSocket broadcast) for each feed.
        """
        if self.detector is None:
            return

        # 1. Collect frames and metadata for feeds with zones
        batch_feed_ids = []
        batch_frames = []
        batch_zone_data = []  # (red_mask, yellow_mask) per feed
        batch_prev_state = []  # (alarm, caution, people, danger, caution_count)

        for feed_id, feed in self.feeds.items():
            with feed.lock:
                frame = feed.last_frame.copy() if feed.last_frame is not None else None

            if frame is None:
                continue

            # Regenerate masks if needed (e.g., after server restart) — must
            # happen BEFORE the zoneless check since masks start as None.
            self._regenerate_masks_if_needed(feed_id, frame.shape[1], frame.shape[0])

            with feed.lock:
                red_zone_mask = (feed.red_zone_mask.copy()
                                 if feed.red_zone_mask is not None else None)
                yellow_zone_mask = (feed.yellow_zone_mask.copy()
                                    if feed.yellow_zone_mask is not None else None)

            # Skip feeds without zones (no expensive YOLO needed)
            if red_zone_mask is None and yellow_zone_mask is None:
                continue

            # Snapshot previous state for change detection
            with feed.lock:
                prev = (feed.alarm_active, feed.caution_active,
                        feed.people_count, feed.danger_count, feed.caution_count)

            batch_feed_ids.append(feed_id)
            batch_frames.append(frame)
            batch_zone_data.append((red_zone_mask, yellow_zone_mask))
            batch_prev_state.append(prev)

        if not batch_frames:
            return

        # 2. Run batch YOLO inference (single GPU call for all frames)
        t0 = time.time()
        batch_masks = self.detector.get_masks_batch(batch_frames)
        t_yolo = time.time()
        feed_names = ", ".join(batch_feed_ids)
        print(f"[TIMING] Batch YOLO ({len(batch_frames)} frames): {t_yolo-t0:.3f}s  feeds=[{feed_names}]")

        # 3. Process results for each feed
        for idx, feed_id in enumerate(batch_feed_ids):
            feed = self.feeds[feed_id]
            frame = batch_frames[idx]
            person_masks = batch_masks[idx]
            red_zone_mask, yellow_zone_mask = batch_zone_data[idx]
            prev_alarm, prev_caution, prev_people, prev_danger, prev_caution_count = batch_prev_state[idx]

            # Pre-render mask overlay (skip if disabled via --no-mask)
            overlay = None
            if MASK_OVERLAY_ENABLED and person_masks:
                h, w = frame.shape[:2]
                overlay = np.zeros((h, w, 3), dtype=np.uint8)
                for mask in person_masks:
                    m = mask if mask.shape[:2] == (h, w) else cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    overlay[m == 1] = (0, 255, 255)  # Cyan in RGB
                    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)

            with feed.lock:
                feed.people_count = len(person_masks)
                feed.last_mask_overlay = overlay
                feed.last_detection_time = datetime.now()

                # Reset counts
                feed.alarm_active = False
                feed.caution_active = False
                feed.danger_count = 0
                feed.caution_count = 0

                if len(person_masks) > 0:
                    # Check RED zone overlap
                    if red_zone_mask is not None:
                        red_mask = red_zone_mask
                        if red_mask.shape != frame.shape[:2]:
                            red_mask = cv2.resize(red_mask, (frame.shape[1], frame.shape[0]),
                                                  interpolation=cv2.INTER_NEAREST)

                        is_alarm, danger_masks = check_danger_zone_overlap(person_masks, red_mask)
                        feed.alarm_active = is_alarm
                        feed.danger_count = len(danger_masks)

                        if is_alarm and len(danger_masks) > 0 and self.depth_model is not None:
                            target_mask = danger_masks[0]
                            feet_pixel = coord_utils.get_feet_from_mask(target_mask)
                            if feet_pixel is not None:
                                try:
                                    job = (feed_id, frame.copy(), feet_pixel)
                                    self._depth_queue.put_nowait(job)
                                except queue.Full:
                                    pass

                    # Check YELLOW zone overlap
                    if yellow_zone_mask is not None:
                        y_mask = yellow_zone_mask
                        if y_mask.shape != frame.shape[:2]:
                            y_mask = cv2.resize(y_mask, (frame.shape[1], frame.shape[0]),
                                                interpolation=cv2.INTER_NEAREST)

                        effective_yellow = y_mask
                        if red_zone_mask is not None:
                            red_for_subtract = red_zone_mask
                            if red_for_subtract.shape != y_mask.shape:
                                red_for_subtract = cv2.resize(red_for_subtract, (y_mask.shape[1], y_mask.shape[0]),
                                                              interpolation=cv2.INTER_NEAREST)
                            effective_yellow = y_mask & (~red_for_subtract)

                        is_caution, caution_masks = check_danger_zone_overlap(person_masks, effective_yellow)
                        feed.caution_active = is_caution
                        feed.caution_count = len(caution_masks)

                # Clear target coordinates if no RED zone alarm
                if not feed.alarm_active:
                    feed.target_coordinates = None

            # Broadcast via WebSocket only on state change
            changed = (
                feed.alarm_active != prev_alarm
                or feed.caution_active != prev_caution
                or feed.people_count != prev_people
                or feed.danger_count != prev_danger
                or feed.caution_count != prev_caution_count
            )
            if changed:
                status = self._get_status(feed)
                _ws_broadcast_from_thread(status.model_dump_json())

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

            pos = None
            if feed.position:
                pos = TargetCoordinate(
                    x=feed.position[0],
                    y=feed.position[1],
                    z=feed.position[2]
                )

            return DetectionStatus(
                feed_id=feed.feed_id,
                alarm_active=feed.alarm_active,
                caution_active=feed.caution_active,
                people_count=feed.people_count,
                danger_count=feed.danger_count,
                caution_count=feed.caution_count,
                target_coordinates=target,
                last_detection_time=feed.last_detection_time.isoformat() if feed.last_detection_time else None,
                position=pos,
            )

    def _is_feed_warmed_up(self, feed: FeedState) -> bool:
        """Return True once enough frames were captured for stable startup detection."""
        with feed.lock:
            return feed.frame_count >= DETECTION_WARMUP_FRAMES

    def run_auto_segmentation(self, feed_id: str) -> List[Zone]:
        """Run auto-segmentation on a feed's current frame and update zones.

        Uses progressive confidence reduction on retries: starts at the
        configured threshold and lowers it on each failed attempt so that
        partially-visible hazards still get detected.

        Returns the list of auto-generated Zone objects (empty if no detections).
        """
        if feed_id not in self.feeds:
            return []

        feed = self.feeds[feed_id]

        if not feed.scene_type or self.segmenter is None:
            print(f"[AUTO-SEG] {feed_id}: SKIPPED — scene_type={feed.scene_type!r}, segmenter={'loaded' if self.segmenter else 'None'}")
            return []

        with feed.lock:
            frame = feed.last_frame.copy() if feed.last_frame is not None else None

        if frame is None:
            print(f"[AUTO-SEG] {feed_id}: SKIPPED — no frame available")
            return []

        # AirSim frames are stored as RGB; convert to BGR for the YOLO model
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        print(f"[AUTO-SEG] {feed_id}: Running segmentation — scene_type='{feed.scene_type}', frame={frame_bgr.shape}, model_loaded={feed.scene_type in self.segmenter.models}")

        # Run segmentation
        zone_dicts = self.segmenter.segment_frame(frame_bgr, feed.scene_type)

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

# WebSocket clients for real-time status push
_ws_clients: set[WebSocket] = set()
_event_loop: Optional[asyncio.AbstractEventLoop] = None

async def _ws_broadcast(message: str):
    """Send a message to all connected WebSocket clients."""
    dead = []
    for ws in _ws_clients:
        try:
            await ws.send_text(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.discard(ws)

def _ws_broadcast_from_thread(message: str):
    """Schedule a WebSocket broadcast from a background thread."""
    if _event_loop is not None and _ws_clients:
        asyncio.run_coroutine_threadsafe(_ws_broadcast(message), _event_loop)

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
        try:
            feed_manager.capture_all_frames()
        except Exception as e:
            print(f"[CAPTURE] Error capturing frames: {e}")

        time.sleep(FRAME_CAPTURE_INTERVAL)

    print("[CAPTURE] Frame capture loop stopped")


def detection_loop():
    """Detection thread — runs human detection using batch YOLO inference.

    Collects all frames from feeds with zones, runs a single batched
    YOLO call on the GPU, then processes results per-feed. This is
    significantly faster than running YOLO sequentially per feed.
    """
    print(f"[DETECTION] Starting detection loop ({1/DETECTION_INTERVAL:.0f} FPS, batch)...")
    last_drone_status_check = 0.0

    while feed_manager.running:
        try:
            feed_manager.run_detection_batch()
        except Exception as e:
            print(f"[DETECTION] Error in batch detection: {e}")
            import traceback
            traceback.print_exc()

        # Poll drone status (rate-limited to once per second)
        current_time = time.time()
        if (feed_manager.drone_is_navigating
                and feed_manager.drone_api is not None
                and current_time - last_drone_status_check >= 1.0):
            last_drone_status_check = current_time
            try:
                status = feed_manager.drone_api.get_status()
                if status:
                    # Drone switched to manual (arrived at target) or stopped navigating
                    if not status.get("is_navigating", True):
                        if feed_manager.drone_is_navigating:
                            print("[DRONE] Drone reached target — now in user control")
                        feed_manager.drone_is_navigating = False
            except Exception as e:
                print(f"[DRONE] Error checking drone status: {e}")

        time.sleep(DETECTION_INTERVAL)

    print("[DETECTION] Detection loop stopped")


def auto_segmentation_loop():
    """Background thread for periodic auto-segmentation.

    All feed types: segment once on first frame, then periodically refresh.
    Manual edits (user saves zones) pause auto-seg until user triggers 'Reset to Auto'.
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
                # Initial segmentation: run once per feed on startup (only if auto-refresh is on)
                if feed_id not in initial_seg_done and not feed.manual_zones_set:
                    if feed_manager.auto_refresh:
                        with feed.lock:
                            has_frame = feed.last_frame is not None
                        if has_frame:
                            feed_manager.run_auto_segmentation(feed_id)
                            initial_seg_done.add(feed_id)
                            print(f"[AUTO-SEG] Initial segmentation done for {feed_id} ({feed.scene_type})")

                # Periodic refresh: only when auto_refresh is enabled and user hasn't manually edited
                elif feed_manager.auto_refresh and feed_id in initial_seg_done and not feed.manual_zones_set:
                    current_time = time.time()
                    if current_time - feed.last_auto_seg_time >= AUTO_SEG_INTERVAL:
                        feed.auto_seg_active = True
                        feed_manager.run_auto_segmentation(feed_id)

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

            # Update coordinates — also snapshot locally so the detection
            # thread can't clear them before we deploy the drone.
            deploy_coords = (
                target_pos.x_val,
                target_pos.y_val,
                SAFE_Z_ALTITUDE,
            )
            with feed.lock:
                feed.target_coordinates = deploy_coords

            # Create a new trigger event (each detection = separate ~10s recording)
            # Cooldown prevents the same camera from spamming triggers
            trigger_event = None
            now = datetime.now()
            cooldown_ok = True
            for t in reversed(feed_manager.trigger_history):
                if t.feed_id == feed_id:
                    age = (now - datetime.fromisoformat(t.timestamp)).total_seconds()
                    if age < TRIGGER_COOLDOWN:
                        cooldown_ok = False
                    break

            if cooldown_ok:
                try:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    _, snap_buf = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    snap_bytes = snap_buf.tobytes()

                    # Take only the last 5 seconds of pre-trigger frames
                    all_pre = list(feed.replay_buffer)
                    cutoff = (now - timedelta(seconds=5)).isoformat()
                    pre_frames = [(ts, data) for ts, data in all_pre if ts >= cutoff]
                    feed_manager._trigger_id_counter += 1
                    trigger_event = TriggerEvent(
                        id=feed_manager._trigger_id_counter,
                        feed_id=feed_id,
                        timestamp=now.isoformat(),
                        coords=deploy_coords,
                        snapshot=snap_bytes,
                        replay_frames=pre_frames,
                        replay_trigger_index=len(pre_frames),
                    )
                    feed_manager.trigger_history.append(trigger_event)
                    if len(feed_manager.trigger_history) > MAX_TRIGGER_HISTORY:
                        feed_manager.trigger_history = feed_manager.trigger_history[-MAX_TRIGGER_HISTORY:]

                    # Start post-trigger capture for this event
                    feed_manager._post_trigger = {"trigger_id": trigger_event.id, "feed_id": feed_id, "end_time": time.time() + 5.0}
                    print(f"[TRIGGER] Added trigger #{trigger_event.id} from {feed_id} ({len(pre_frames)} pre-trigger frames)")
                except Exception as snap_err:
                    print(f"[TRIGGER] Failed to capture trigger: {snap_err}")

            # Lazy reconnect: keep drone_api ready for deploy requests.
            if feed_manager.drone_api is None:
                try:
                    candidate = DroneAPIClient()
                    if candidate.check_connection():
                        feed_manager.drone_api = candidate
                        print("[DRONE] Lazy-connected to drone API")
                except Exception:
                    pass

            # Auto-deploy only for the very first trigger; after that it's manual-only.
            if (feed_manager.drone_api is not None
                    and not feed_manager._first_auto_deployed
                    and trigger_event is not None
                    and not trigger_event.deployed):
                current_time = time.time()
                drone_status = feed_manager.drone_api.get_status()
                drone_mode = drone_status.get("mode") if drone_status else None
                drone_nav = drone_status.get("is_navigating", True) if drone_status else True
                drone_ready = (
                    drone_status is not None
                    and drone_mode == "automatic"
                    and not drone_nav
                )
                if drone_ready:
                    tx, ty, tz = deploy_coords
                    print(f"[DRONE] First auto-deploy to ({tx:.2f}, {ty:.2f}, {tz:.2f}) for {feed_id}")
                    feed_manager.drone_api.set_mode("automatic")
                    if feed_manager.drone_api.goto_position(tx, ty, tz):
                        feed_manager.last_deployment_time = current_time
                        feed_manager.drone_is_navigating = True
                        trigger_event.deployed = True
                        feed_manager._first_auto_deployed = True
                        print("[DRONE] First auto-deploy sent successfully — subsequent deploys are manual")
                    else:
                        print("[DRONE] First auto-deploy command failed")

        except Exception as e:
            print(f"[DEPTH] Error processing {feed_id}: {e}")
            import traceback
            traceback.print_exc()

    print("[DEPTH] Depth worker thread stopped")


def hover_cctv_drones():
    """Take off and hover all CCTV drones at a fixed altitude.

    Uses the feed_manager's AirSim client. Called once during startup
    when --hover flag is set. Each drone is armed, takes off, moves to
    CCTV_HOVER_ALTITUDE, and enters hover mode.
    """
    client = feed_manager.client
    if client is None:
        print("[HOVER] No AirSim client, skipping CCTV hover")
        return

    vehicle_names = [feed.vehicle_name for feed in feed_manager.feeds.values()]
    # Deduplicate while preserving order
    seen = set()
    unique_vehicles = []
    for v in vehicle_names:
        if v not in seen:
            seen.add(v)
            unique_vehicles.append(v)

    print(f"[HOVER] Arming and taking off {len(unique_vehicles)} CCTV drones...")

    # Arm and take off all drones, collecting futures
    takeoff_futures = []
    for vehicle_name in unique_vehicles:
        try:
            client.enableApiControl(True, vehicle_name=vehicle_name)
            client.armDisarm(True, vehicle_name=vehicle_name)
            future = client.takeoffAsync(vehicle_name=vehicle_name)
            takeoff_futures.append((vehicle_name, future))
        except Exception as e:
            print(f"[HOVER] Failed to arm {vehicle_name}: {e}")

    # Wait for all takeoffs
    for vehicle_name, future in takeoff_futures:
        try:
            future.join()
            print(f"[HOVER] {vehicle_name} takeoff complete")
        except Exception as e:
            print(f"[HOVER] {vehicle_name} takeoff issue: {e}")

    # Move all drones to hover altitude
    altitude_futures = []
    for vehicle_name in unique_vehicles:
        try:
            future = client.moveToZAsync(CCTV_HOVER_ALTITUDE, 5, vehicle_name=vehicle_name)
            altitude_futures.append((vehicle_name, future))
        except Exception as e:
            print(f"[HOVER] Failed to move {vehicle_name}: {e}")

    for vehicle_name, future in altitude_futures:
        try:
            future.join()
        except Exception as e:
            print(f"[HOVER] {vehicle_name} altitude issue: {e}")

    # Enter hover mode on all drones
    for vehicle_name in unique_vehicles:
        try:
            client.hoverAsync(vehicle_name=vehicle_name).join()
        except Exception as e:
            print(f"[HOVER] {vehicle_name} hover issue: {e}")

    print(f"[HOVER] All CCTV drones hovering at altitude {CCTV_HOVER_ALTITUDE}m")


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
    global _event_loop
    _event_loop = asyncio.get_running_loop()

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
        elif CCTV_HOVER_DRONES:
            # Hover mode: take off and hold altitude (skip if follow is active,
            # since follow already handles takeoff)
            hover_cctv_drones()

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
    return {
        "feeds": feeds,
        "globalSceneType": feed_manager.global_scene_type,
        "autoRefresh": feed_manager.auto_refresh,
    }

@app.patch("/settings")
async def update_settings(request: Request):
    """Update global settings (scene type, auto-refresh)."""
    body = await request.json()
    scene_type = body.get("sceneType")
    auto_refresh = body.get("autoRefresh")

    if scene_type is not None:
        if scene_type not in ("ship", "railway", "bridge"):
            return JSONResponse({"error": f"Invalid scene type: {scene_type}"}, status_code=400)
        old_scene_type = feed_manager.global_scene_type
        feed_manager.global_scene_type = scene_type
        # Apply to all feeds
        for fid, feed in feed_manager.feeds.items():
            feed.scene_type = scene_type
            print(f"[SETTINGS] {fid}: scene_type = '{scene_type}'")
        print(f"[SETTINGS] Scene type changed: '{old_scene_type}' -> '{scene_type}'")

        # Log model path for the new scene type
        model_path = SEG_MODEL_PATHS.get(scene_type)
        print(f"[SETTINGS] Model path for '{scene_type}': {model_path}")
        if model_path:
            import os as _os
            print(f"[SETTINGS] Model file exists: {_os.path.exists(model_path)}")

        # Load the segmentation model for the new scene type if not already loaded
        if feed_manager.segmenter is not None:
            loaded_models = list(feed_manager.segmenter.models.keys())
            print(f"[SETTINGS] Currently loaded models: {loaded_models}")
            if scene_type in SEG_MODEL_PATHS:
                if scene_type not in feed_manager.segmenter.models:
                    print(f"[SETTINGS] Model for '{scene_type}' not loaded — loading now...")
                    try:
                        new_paths = {scene_type: SEG_MODEL_PATHS[scene_type]}
                        feed_manager.segmenter = SceneSegmenter(
                            {**{k: v for k, v in SEG_MODEL_PATHS.items() if k in feed_manager.segmenter.models}, **new_paths},
                            confidence=AUTO_SEG_CONFIDENCE,
                        )
                        print(f"[SETTINGS] Loaded segmentation model for '{scene_type}' — now loaded: {list(feed_manager.segmenter.models.keys())}")
                    except Exception as e:
                        print(f"[SETTINGS] Failed to load model for '{scene_type}': {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"[SETTINGS] Model for '{scene_type}' already loaded — OK")
            else:
                print(f"[SETTINGS] WARNING: No model path configured for '{scene_type}'")
        else:
            print(f"[SETTINGS] WARNING: segmenter is None — auto-segmentation not available")

    if auto_refresh is not None:
        feed_manager.auto_refresh = bool(auto_refresh)
        print(f"[SETTINGS] Auto-refresh {'enabled' if feed_manager.auto_refresh else 'disabled'}")
        if feed_manager.auto_refresh:
            print(f"[SETTINGS] Auto-seg will run every {AUTO_SEG_INTERVAL}s for scene_type='{feed_manager.global_scene_type}'")

    return {
        "globalSceneType": feed_manager.global_scene_type,
        "autoRefresh": feed_manager.auto_refresh,
    }


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

    Uses the pre-rendered frame (with detection masks baked in) when
    available, so the mask overlay always aligns with the frame it was
    computed from.  Falls back to the raw frame when no detection has
    run yet.
    """
    last_jpeg = _NO_SIGNAL_JPEG

    while True:
        feed = feed_manager.feeds.get(feed_id)

        frame = None
        mask_overlay = None
        if feed:
            with feed.lock:
                if feed.last_frame is not None:
                    frame = feed.last_frame.copy()
                if feed.last_mask_overlay is not None:
                    mask_overlay = feed.last_mask_overlay

        if frame is not None:
            # Convert RGB to BGR for OpenCV encoding
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Composite detection mask overlay (pre-rendered at detection time)
            if mask_overlay is not None:
                h, w = frame_bgr.shape[:2]
                ov = mask_overlay
                if ov.shape[:2] != (h, w):
                    ov = cv2.resize(ov, (w, h), interpolation=cv2.INTER_NEAREST)
                # Convert overlay RGB→BGR and blend where non-zero
                ov_bgr = cv2.cvtColor(ov, cv2.COLOR_RGB2BGR)
                mask_region = np.any(ov_bgr > 0, axis=2)
                frame_bgr[mask_region] = cv2.addWeighted(
                    frame_bgr, 0.6, ov_bgr, 0.4, 0
                )[mask_region]

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
                last_jpeg = buffer.tobytes()

        # Always yield a frame — last good frame, or NO SIGNAL at startup
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + last_jpeg + b'\r\n')

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

@app.websocket("/ws/status")
async def ws_status(websocket: WebSocket):
    """WebSocket endpoint for real-time detection status updates."""
    await websocket.accept()
    _ws_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()  # keep connection alive
    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients.discard(websocket)

@app.post("/feeds/{feed_id}/zones")
async def update_zones(feed_id: str, request: ZonesUpdateRequest):
    """Update zones for a feed."""
    if feed_id not in feed_manager.feeds:
        raise HTTPException(status_code=404, detail=f"Feed {feed_id} not found")

    try:
        feed = feed_manager.feeds[feed_id]

        # Track manual edits: prevents auto-seg from overwriting user-edited zones
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
def trigger_auto_segment(feed_id: str):
    """Trigger on-demand auto-segmentation for a feed (sync so it runs in a thread pool)."""
    if feed_id not in feed_manager.feeds:
        raise HTTPException(status_code=404, detail=f"Feed {feed_id} not found")

    feed = feed_manager.feeds[feed_id]

    if not feed.scene_type:
        raise HTTPException(status_code=400, detail=f"No scene type configured for {feed_id}")

    if feed_manager.segmenter is None:
        raise HTTPException(status_code=503, detail="Segmentation models not loaded")

    try:
        zones = feed_manager.run_auto_segmentation(feed_id)

        # Reset manual flag since user explicitly requested auto-seg
        feed.manual_zones_set = False

        return {
            "status": "success",
            "zones_count": len(zones),
            "scene_type": feed.scene_type,
            "zones": [z.model_dump() for z in zones],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _compute_trigger_fps(trigger: TriggerEvent) -> int:
    """Compute actual FPS from a trigger's replay frame timestamps."""
    frames = trigger.replay_frames
    if len(frames) < 2:
        return TRIGGER_REPLAY_FPS
    try:
        first_ts = datetime.fromisoformat(frames[0][0])
        last_ts = datetime.fromisoformat(frames[-1][0])
        duration = (last_ts - first_ts).total_seconds()
        if duration <= 0:
            return TRIGGER_REPLAY_FPS
        return max(1, round(len(frames) / duration))
    except Exception:
        return TRIGGER_REPLAY_FPS


# --- Multi-trigger endpoints ---

@app.get("/triggers")
async def list_triggers():
    """Return all trigger events (metadata only, no JPEG bytes)."""
    return {
        "triggers": [
            {
                "id": t.id,
                "feed_id": t.feed_id,
                "timestamp": t.timestamp,
                "deployed": t.deployed,
                "replay_frame_count": len(t.replay_frames),
                "replay_trigger_index": t.replay_trigger_index,
                "coords": list(t.coords),
                "replay_fps": _compute_trigger_fps(t),
            }
            for t in feed_manager.trigger_history
        ],
        "replay_fps": TRIGGER_REPLAY_FPS,  # default fallback
    }


@app.get("/triggers/{trigger_id}/snapshot")
async def get_trigger_snapshot_by_id(trigger_id: int):
    """Return the snapshot JPEG for a specific trigger."""
    trigger = feed_manager.get_trigger_by_id(trigger_id)
    if trigger is None:
        raise HTTPException(status_code=404, detail="Trigger not found")

    from fastapi.responses import Response
    return Response(content=trigger.snapshot, media_type="image/jpeg")


@app.get("/triggers/{trigger_id}/replay/{frame_index}")
async def get_trigger_replay_by_id(trigger_id: int, frame_index: int):
    """Return a replay frame for a specific trigger."""
    trigger = feed_manager.get_trigger_by_id(trigger_id)
    if trigger is None:
        raise HTTPException(status_code=404, detail="Trigger not found")
    if frame_index < 0 or frame_index >= len(trigger.replay_frames):
        raise HTTPException(status_code=404, detail="Frame index out of range")

    from fastapi.responses import Response
    _timestamp, jpeg_bytes = trigger.replay_frames[frame_index]
    return Response(content=jpeg_bytes, media_type="image/jpeg")


@app.post("/triggers/{trigger_id}/deploy")
async def deploy_to_trigger(trigger_id: int):
    """Deploy drone to a specific trigger's coordinates."""
    trigger = feed_manager.get_trigger_by_id(trigger_id)
    if trigger is None:
        raise HTTPException(status_code=404, detail="Trigger not found")

    if feed_manager.drone_api is None:
        raise HTTPException(status_code=503, detail="Drone API not connected")

    tx, ty, tz = trigger.coords

    # Ensure drone is in automatic mode before sending goto
    feed_manager.drone_api.set_mode("automatic")

    if feed_manager.drone_api.goto_position(tx, ty, tz):
        trigger.deployed = True
        feed_manager.drone_is_navigating = True
        feed_manager.last_deployment_time = time.time()
        print(f"[DRONE] Manual deploy to trigger #{trigger_id} ({tx:.2f}, {ty:.2f}, {tz:.2f})")
        return {"status": "deployed", "trigger_id": trigger_id, "coords": list(trigger.coords)}
    else:
        raise HTTPException(status_code=500, detail="Drone goto command failed")


@app.delete("/triggers/{trigger_id}")
async def delete_trigger(trigger_id: int):
    """Remove a trigger from the history."""
    trigger = feed_manager.get_trigger_by_id(trigger_id)
    if trigger is None:
        raise HTTPException(status_code=404, detail="Trigger not found")
    feed_manager.trigger_history = [t for t in feed_manager.trigger_history if t.id != trigger_id]
    print(f"[TRIGGER] Removed trigger #{trigger_id}")
    return {"status": "removed", "trigger_id": trigger_id}


# --- Backward-compat endpoints (use latest trigger) ---

@app.get("/trigger-snapshot")
async def get_trigger_snapshot():
    """Return the latest trigger snapshot as JPEG."""
    latest = feed_manager.get_latest_trigger()
    if latest is None:
        return JSONResponse(status_code=204, content=None)

    from fastapi.responses import Response
    return Response(content=latest.snapshot, media_type="image/jpeg")


@app.get("/trigger-info")
async def get_trigger_info():
    """Return metadata about the latest trigger (backward compat)."""
    latest = feed_manager.get_latest_trigger()
    if latest is None:
        return {
            "has_snapshot": False, "feed_id": None, "timestamp": None,
            "replay_frame_count": 0, "replay_fps": TRIGGER_REPLAY_FPS, "replay_trigger_index": 0,
        }
    return {
        "has_snapshot": True,
        "feed_id": latest.feed_id,
        "timestamp": latest.timestamp,
        "replay_frame_count": len(latest.replay_frames),
        "replay_fps": TRIGGER_REPLAY_FPS,
        "replay_trigger_index": latest.replay_trigger_index,
    }


@app.get("/trigger-replay/{frame_index}")
async def get_trigger_replay_frame(frame_index: int):
    """Return a replay frame from the latest trigger (backward compat)."""
    latest = feed_manager.get_latest_trigger()
    if latest is None or frame_index < 0 or frame_index >= len(latest.replay_frames):
        raise HTTPException(status_code=404, detail="Frame index out of range")

    from fastapi.responses import Response
    _timestamp, jpeg_bytes = latest.replay_frames[frame_index]
    return Response(content=jpeg_bytes, media_type="image/jpeg")


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
