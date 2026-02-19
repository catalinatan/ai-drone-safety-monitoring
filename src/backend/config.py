"""
Centralized configuration for the backend server.

All settings have sensible defaults and can be overridden via environment variables.
"""

import os

# ============================================================================
# PATH SETUP
# ============================================================================

_CONFIG_FILE = os.path.abspath(__file__)
_BACKEND_DIR = os.path.dirname(_CONFIG_FILE)
_SRC_DIR = os.path.dirname(_BACKEND_DIR)
_REPO_ROOT = os.path.dirname(_SRC_DIR)

CCTV_MONITORING_DIR = os.path.join(_SRC_DIR, "cctv_monitoring")

# ============================================================================
# PORTS
# ============================================================================

BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8001"))
DRONE_API_PORT = int(os.getenv("DRONE_API_PORT", "8000"))
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "5173"))

# ============================================================================
# DRONE API
# ============================================================================

DRONE_API_URL = os.getenv("DRONE_API_URL", f"http://localhost:{DRONE_API_PORT}")
DRONE_API_TIMEOUT = int(os.getenv("DRONE_API_TIMEOUT", "5"))

# ============================================================================
# FRAME CAPTURE & STREAMING
# ============================================================================

# Frame capture rate (how often we grab frames from AirSim)
FRAME_CAPTURE_FPS = int(os.getenv("FRAME_CAPTURE_FPS", "30"))
FRAME_CAPTURE_INTERVAL = 1.0 / FRAME_CAPTURE_FPS

# MJPEG streaming rate to UI
STREAM_FPS = int(os.getenv("STREAM_FPS", "30"))
STREAM_INTERVAL = 1.0 / STREAM_FPS

# ============================================================================
# DETECTION
# ============================================================================

# How often to run human detection
DETECTION_FPS = int(os.getenv("DETECTION_FPS", "30"))
DETECTION_INTERVAL = 1.0 / DETECTION_FPS
ALARM_COOLDOWN = float(os.getenv("ALARM_COOLDOWN", "5.0"))

# Startup warm-up: require a minimum number of captured frames per feed
# before detection can trigger alarms/drone actions.
DETECTION_WARMUP_FRAMES = int(os.getenv("DETECTION_WARMUP_FRAMES", "20"))

# ============================================================================
# CAMERA / AIRSIM
# ============================================================================

CCTV_HEIGHT = float(os.getenv("CCTV_HEIGHT", "15.0"))
SAFE_Z_ALTITUDE = float(os.getenv("SAFE_Z_ALTITUDE", "-10.0"))

# ============================================================================
# PERSISTENCE
# ============================================================================

DATA_DIR = os.path.join(_BACKEND_DIR, "data")
ZONES_FILE = os.path.join(DATA_DIR, "zones.json")

# ============================================================================
# MODEL PATHS
# ============================================================================

LITE_MONO_DIR = os.path.join(
    CCTV_MONITORING_DIR, "lite_mono_weights", "lite-mono-small_640x192"
)
ENCODER_PATH = os.path.join(LITE_MONO_DIR, "encoder.pth")
DECODER_PATH = os.path.join(LITE_MONO_DIR, "depth.pth")

# ============================================================================
# CCTV FOLLOW MODE (for moving environments like ship)
# ============================================================================

# Friendly label → AirSim object name mapping
FOLLOW_TARGETS = {
    "ship": "Target_Boat",
}

# Set via --follow <label> in main.py, empty string = stationary mode
_follow_label = os.getenv("CCTV_FOLLOW_TARGET", "")
CCTV_FOLLOW_TARGET = FOLLOW_TARGETS.get(_follow_label, "")

# Set via --hover in main.py — makes CCTV drones take off and hover at altitude
CCTV_HOVER_DRONES = os.getenv("CCTV_HOVER_DRONES", "") == "1"

# Altitude for CCTV hover mode (NED, so negative = up). Default -15m.
CCTV_HOVER_ALTITUDE = float(os.getenv("CCTV_HOVER_ALTITUDE", "-15.0"))

# Camera Actor guides: drone → Camera Actor name in UE World Outliner.
# The follow loop reads each Camera Actor's pose and teleports the drone
# to that exact position + orientation. Position cameras in UE to adjust views.
CCTV_FOLLOW_CAMERAS = {
    "Drone2": "CCTV1",
    "Drone3": "CCTV2",
    "Drone4": "CCTV3",
    "Drone5": "CCTV4",
}

CCTV_FOLLOW_INTERVAL = 0.01    # seconds (100 Hz — minimizes jitter)

# ============================================================================
# FEED CONFIGURATION
# ============================================================================

# AirSim camera configuration for CCTV: {feed_id: (camera_name, vehicle_name)}
# Note: Search drone cameras are served directly from the drone API (port 8000)
FEED_CONFIG = {
    "cctv-1": ("0", "Drone2"),
    "cctv-2": ("0", "Drone3"),
    "cctv-3": ("0", "Drone4"),
    "cctv-4": ("0", "Drone5"),
}

FEED_METADATA = {
    "cctv-1": {"name": "CCTV CAM 1", "location": "Aerial Overview"},
    "cctv-2": {"name": "CCTV CAM 2", "location": "Aerial Overview"},
    "cctv-3": {"name": "CCTV CAM 3", "location": "Aerial Overview"},
    "cctv-4": {"name": "CCTV CAM 4", "location": "Aerial Overview"},
}

# ============================================================================
# SCENE TYPE PER FEED
# ============================================================================

# Maps each feed to a scene type for auto-segmentation: "ship", "railway", "bridge", or None
FEED_SCENE_TYPE = {
    "cctv-1": "bridge",
    "cctv-2": "bridge",
    "cctv-3": "bridge",
    "cctv-4": "bridge",
}

# ============================================================================
# AUTO-SEGMENTATION
# ============================================================================

# YOLO segmentation model paths (per scene type)
SEG_MODEL_PATHS = {
    "railway": os.path.join(_REPO_ROOT, "runs", "segment", "runs", "segment", "railway_hazard", "weights", "best.pt"),
    "ship": os.path.join(_REPO_ROOT, "runs", "segment", "runs", "segment", "ship_hazard", "weights", "best.pt"),
    "bridge": os.path.join(_REPO_ROOT, "runs", "segment", "runs", "segment", "bridge_hazard", "weights", "best.pt"),
}

# How often to re-run auto-segmentation for ship feeds (seconds)
AUTO_SEG_INTERVAL = float(os.getenv("AUTO_SEG_INTERVAL", "60.0"))

# YOLO segmentation confidence threshold
AUTO_SEG_CONFIDENCE = float(os.getenv("AUTO_SEG_CONFIDENCE", "0.5"))

# Polygon simplification epsilon for cv2.approxPolyDP (higher = fewer points)
AUTO_SEG_SIMPLIFY_EPSILON = float(os.getenv("AUTO_SEG_SIMPLIFY_EPSILON", "2.0"))

# Minimum contour area (pixels) for auto-segment polygons
AUTO_SEG_MIN_CONTOUR_AREA = float(os.getenv("AUTO_SEG_MIN_CONTOUR_AREA", "40.0"))
