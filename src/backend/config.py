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
