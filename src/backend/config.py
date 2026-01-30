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

# How often to run human detection (can be slower than frame capture)
DETECTION_FPS = int(os.getenv("DETECTION_FPS", "10"))
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
# FEED CONFIGURATION
# ============================================================================

# AirSim camera configuration: {feed_id: (camera_name, vehicle_name)}
FEED_CONFIG = {
    "cctv-1": ("0", "Drone2"),
    "drone-cam": ("3", ""),
}

FEED_METADATA = {
    "cctv-1": {"name": "CCTV CAM 1", "location": "Aerial Overview"},
    "drone-cam": {"name": "DRONE CAM", "location": "Mobile Unit"},
}
