# AI Safety Monitoring System

A real-time safety monitoring system that uses computer vision to detect humans in restricted zones and automatically deploys drones for investigation. Built for integration with AirSim simulation environments.

## Overview

The system monitors CCTV feeds for human intrusion into designated danger zones. When a person enters a restricted area, it calculates their 3D world coordinates using monocular depth estimation and dispatches a drone to investigate.

Key capabilities:

- Real-time human detection using YOLOv8 segmentation
- Configurable danger zones (red/yellow/green) via web UI
- Monocular depth estimation for 3D coordinate calculation
- Automatic drone deployment to intrusion locations
- Multi-feed MJPEG streaming at 30 FPS

## Architecture

### High-Level Overview

```
┌──────────────┐       ┌──────────────────┐       ┌──────────────┐
│  React UI    │◄─────►│  FastAPI Backend │◄─────►│ Drone Server │
│  (Vite)      │       │  (Port 8001)     │       │ (Port 8000)  │
│  Port 5173   │       │                  │       │              │
└──────────────┘       └────────┬─────────┘       └──────┬───────┘
                                │                       │
                    ┌───────────┴───────────┐            │
                    ▼                       ▼            ▼
          ┌──────────────────┐    ┌─────────────────────────┐
          │ Hardware Layer   │    │   Drone Control API     │
          │ (Camera/Drone    │    │   (AirSim MAVLink)      │
          │  Abstraction)    │    │                         │
          └──────────────────┘    └─────────────────────────┘
```

### Module Organization

**Core Modules** (`src/core/`):
- Pure business logic — no I/O, no framework dependencies
- `models.py` — Pydantic schemas (Point, Zone, DetectionResult)
- `config.py` — YAML configuration loader
- `zone_manager.py` — Danger zone overlap detection
- `alarm.py` — Alarm state machine with cooldown
- `detection_pipeline.py` — Orchestrates detection + zone checking

**Hardware Abstraction Layer** (`src/hardware/`):
- Swappable camera/drone backends via factory pattern
- `camera/base.py` — CameraBackend ABC
- `camera/airsim_camera.py`, `file_camera.py`, `rtsp_camera.py`
- `drone/base.py` — DroneBackend ABC
- `drone/airsim_drone.py`, `mavlink_drone.py`

**Services** (`src/services/`):
- Stateful business logic — orchestrates core modules + hardware
- `feed_manager.py` — Central state store for all camera feeds
- `zone_persistence.py` — Load/save zones to JSON
- `streaming.py` — MJPEG encoding, frame overlay
- `drone_dispatcher.py` — Drone deployment logic with policies

**API Routes** (`src/api/routes/`):
- RESTful endpoints using FastAPI + dependency injection
- `health.py` — System status
- `feeds.py` — Feed list + detection status
- `zones.py` — Zone management + auto-segmentation
- `video.py` — MJPEG streaming
- `drone.py` — Trigger history + deployment

**Drone Server** (`src/drone_server/`):
- Separate FastAPI app for drone control
- Interfaces with DroneBackend for MAVLink/AirSim control

## Requirements

- Python 3.11+
- Node.js 18+ (for UI)
- AirSim simulator with Unreal Engine environment
- CUDA-capable GPU recommended for real-time detection

## Installation

### Prerequisites

- Python 3.11+ with pip
- Node.js 18+ (for React UI)
- CUDA 12.1 (optional but recommended for GPU acceleration)
- AirSim simulator (optional — system runs without it, using file/RTSP cameras)

### Steps

Clone the repository and install dependencies:

```bash
git clone <repository-url>
cd ai-safety-monitoring

# Install Python dependencies (creates editable install)
pip install -e ".[dev]"

# Optional: Install AirSim support (requires AirSim Python package)
pip install -e ".[airsim]"

# UI dependencies
cd src/ui
npm install
cd ../..
```

#### PyTorch with CUDA

The system requires PyTorch. Install with CUDA support separately:

```bash
# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Model Weights

The system uses pre-trained model weights:

1. **YOLOv8 segmentation** — Downloaded automatically by `ultralytics` on first run
2. **Lite-Mono depth model** — Place weights in:
   ```
   src/cctv_monitoring/lite_mono_weights/lite-mono-small_640x192/
   ├── encoder.pth
   └── depth.pth
   ```

## Quick Start

Launch all services with a single command:

```bash
python main.py
```

This starts:
- Backend API at http://localhost:8001
- Drone Control API at http://localhost:8000
- React UI at http://localhost:5173

Press Ctrl+C to stop all services.

### Running Services Individually

```bash
# Backend only (no UI)
python main.py --no-ui

# Or run each service separately:
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8001        # Backend API
python -m uvicorn src.drone_server.app:app --host 0.0.0.0 --port 8000  # Drone Server
cd src/ui && npm run dev                                             # React UI
```

### Development Mode with Follow/Hover

```bash
# CCTV drones follow a target object (ship, railway, etc.)
python main.py --follow ship

# CCTV drones take off and hover at altitude
python main.py --hover

# Disable mask overlay on video streams
python main.py --no-mask
```

## Configuration

Configuration is split into two YAML files:

### Global Settings: `config/default.yaml`

System-wide settings (ports, FPS, detection parameters, drone deployment policies):

```yaml
server:
  backend_port: 8001
  drone_port: 8000

streaming:
  stream_fps: 30
  capture_fps: 30

detection:
  detection_fps: 30
  warmup_frames: 20

alarm:
  cooldown_seconds: 5.0

drone:
  api_url: http://localhost:8000
  api_timeout: 5
```

### Feed Configuration: `config/feeds.yaml`

Per-feed camera and zone settings. Each feed defines its camera backend type (airsim/file/rtsp) and parameters:

```yaml
feeds:
  - feed_id: cctv-1
    name: CCTV Camera 1
    location: Zone A
    camera:
      type: airsim
      camera_name: "0"
      vehicle_name: Drone2
    scene_type: ship
    zones: []  # Populated via API
```

### Environment Variable Overrides

Settings in `config/default.yaml` can be overridden via environment variables. Any key in the YAML can be set via `CONFIG_<SECTION>_<KEY>`:

```bash
CONFIG_SERVER_BACKEND_PORT=9000 python main.py
CONFIG_ALARM_COOLDOWN_SECONDS=10.0 python main.py
```

### AirSim Camera Configuration

To enable AirSim support, install the optional dependency:

```python
FEED_CONFIG = {
    "cctv-1": ("0", "Drone2"),      # Camera 0 on vehicle Drone2
    "cctv-2": ("0", "Drone3"),      # Camera 0 on vehicle Drone3
    "cctv-3": ("0", "Drone4"),      # Camera 0 on vehicle Drone4
    "cctv-4": ("0", "Drone5"),      # Camera 0 on vehicle Drone5
}
```

Sample AirSim `settings.json`:

```json
{
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "Vehicles": {
    "Drone1": {
      "VehicleType": "SimpleFlight",
      "X": 0, "Y": 0, "Z": 0
    },
    "Drone2": {
      "VehicleType": "SimpleFlight",
      "X": 0, "Y": -10, "Z": -15,
      "Pitch": -45,
      "AllowAPIAlways": true
    },
    "Drone3": {
      "VehicleType": "SimpleFlight",
      "X": 10, "Y": -10, "Z": -15,
      "Pitch": -45,
      "AllowAPIAlways": true
    },
    "Drone4": {
      "VehicleType": "SimpleFlight",
      "X": 10, "Y": 10, "Z": -15,
      "Pitch": -45,
      "AllowAPIAlways": true
    },
    "Drone5": {
      "VehicleType": "SimpleFlight",
      "X": 0, "Y": 10, "Z": -15,
      "Pitch": -45,
      "AllowAPIAlways": true
    }
  }
}
```

## API Reference

### Backend API (Port 8001)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with component status |
| `/feeds` | GET | List all camera feeds with detection status |
| `/feeds/{feed_id}/status` | GET | Detection status for a specific feed |
| `/feeds/{feed_id}/zones` | POST | Update danger zones for a feed |
| `/video_feed/{feed_id}` | GET | MJPEG video stream |

API documentation available at http://localhost:8001/docs

### Drone Control API (Port 8000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/status` | GET | Current drone status and position |
| `/mode` | POST | Set control mode (`manual` or `automatic`) |
| `/goto` | POST | Navigate to NED coordinates (automatic mode) |
| `/return_home` | POST | Return to takeoff position |
| `/video_feed/down` | GET | Drone downward camera MJPEG stream |
| `/video_feed/forward` | GET | Drone forward camera MJPEG stream |

API documentation available at http://localhost:8000/docs

### Zone Configuration

Zones are defined as polygons with percentage-based coordinates (0-100):

```json
{
  "zones": [
    {
      "id": "zone-1",
      "level": "red",
      "points": [
        {"x": 10, "y": 10},
        {"x": 50, "y": 10},
        {"x": 50, "y": 50},
        {"x": 10, "y": 50}
      ]
    }
  ]
}
```

Zone levels:
- **red** - Triggers alarm and automatic drone deployment
- **yellow** - Triggers caution alert (visual only, no drone)
- **green** - Safe zone (no action)

## Drone Control

### Manual Mode

When in manual mode, control the drone via keyboard:

| Key | Action |
|-----|--------|
| W | Move forward (North) |
| S | Move backward (South) |
| A | Move left (West) |
| D | Move right (East) |
| Z | Move up |
| X | Move down |
| Q | Quit and land |

### Automatic Mode

In automatic mode, the drone responds to API commands:

```bash
# Switch to automatic mode
curl -X POST http://localhost:8000/mode -H "Content-Type: application/json" -d '{"mode": "automatic"}'

# Send drone to coordinates (NED frame)
curl -X POST http://localhost:8000/goto -H "Content-Type: application/json" -d '{"x": 10, "y": 5, "z": -10}'

# Return to home position
curl -X POST http://localhost:8000/return_home
```

### Safety Limits

Configured in `src/drone_control/config.yaml`:

```yaml
safety:
  max_altitude: 50        # Maximum altitude in meters
  geofence_radius: 200    # Maximum distance from origin

navigation:
  speed: 5.0              # Navigation speed in m/s
  position_tolerance: 1.0 # Arrival threshold in meters
```

## Testing

The test suite covers:
- **Unit tests** — Core logic, services, hardware abstractions (no I/O)
- **Integration tests** — FastAPI routes with dependency injection (no real hardware needed)

Run the test suite with pytest:

```bash
# All tests (excludes pre-existing failures)
pytest tests/ -v \
  --ignore=tests/human_detection/test_accuracy.py \
  --ignore=tests/human_detection/test_detector.py \
  --ignore=tests/cctv_monitoring/test_coord_utils.py

# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=html

# Specific test
pytest tests/unit/core/test_zone_manager.py::TestZoneManager::test_update_zones_creates_red_mask -v
```

**Expected result**: 146+ tests pass, 1 skipped.

## Project Structure

```
ai-safety-monitoring/
├── main.py                      # Development entrypoint (launches all services)
├── pyproject.toml               # Dependencies and tool configuration
├── Dockerfile                   # Production Docker image
├── .gitlab-ci.yml               # CI/CD pipeline
│
├── config/
│   ├── default.yaml             # Global settings (ports, FPS, detection params)
│   └── feeds.yaml               # Camera feed configurations
│
├── src/
│   ├── api/                     # RESTful API routes (FastAPI)
│   │   ├── app.py               # App factory with lifespan context manager
│   │   ├── dependencies.py      # Dependency injection (FeedManager, TriggerStore)
│   │   └── routes/
│   │       ├── health.py        # GET /health
│   │       ├── feeds.py         # GET/PATCH feed list and settings
│   │       ├── zones.py         # POST zones and auto-segmentation
│   │       ├── video.py         # GET /video_feed/{id} MJPEG streaming
│   │       └── drone.py         # Trigger CRUD + deployment
│   │
│   ├── core/                    # Pure business logic (no I/O)
│   │   ├── models.py            # Pydantic schemas
│   │   ├── config.py            # YAML config loader
│   │   ├── zone_manager.py      # Danger zone overlap detection
│   │   ├── alarm.py             # Alarm state machine with cooldown
│   │   └── detection_pipeline.py # Orchestrates detection + zone checking
│   │
│   ├── hardware/                # Swappable hardware backends (ABC pattern)
│   │   ├── __init__.py          # Factory functions
│   │   ├── camera/
│   │   │   ├── base.py          # CameraBackend ABC
│   │   │   ├── airsim_camera.py # AirSim camera implementation
│   │   │   ├── file_camera.py   # File/video camera implementation
│   │   │   └── rtsp_camera.py   # RTSP camera (stub)
│   │   └── drone/
│   │       ├── base.py          # DroneBackend ABC
│   │       ├── airsim_drone.py  # AirSim drone implementation
│   │       └── mavlink_drone.py # MAVLink drone (stub)
│   │
│   ├── services/                # Stateful business logic
│   │   ├── feed_manager.py      # Central state store for all feeds
│   │   ├── zone_persistence.py  # Load/save zones to JSON
│   │   ├── streaming.py         # MJPEG encoding and overlay
│   │   └── drone_dispatcher.py  # Drone deployment with policies
│   │
│   ├── detection/               # Detection models
│   │   ├── human_detector.py    # YOLOv8 segmentation wrapper
│   │   ├── scene_segmenter.py   # Scene type auto-segmentation
│   │   └── depth_estimator.py   # Monocular depth estimation
│   │
│   ├── spatial/                 # 3D spatial utilities
│   │   ├── coord_utils.py       # 3D coordinate calculation from masks
│   │   └── projection.py        # Camera projection matrices
│   │
│   ├── drone_server/            # Separate FastAPI for drone control
│   │   ├── app.py               # Drone control REST API
│   │   └── config.yaml          # Drone-specific settings
│   │
│   ├── cctv_monitoring/         # Lite-Mono depth model (vendored)
│   │   └── Lite-Mono/
│   │       ├── networks/        # Encoder/decoder implementations
│   │       ├── lite_mono_weights/
│   │       └── [training code]
│   │
│   ├── logger.py                # Centralized logging
│   └── ui/                      # React frontend (Vite)
│       ├── src/
│       ├── package.json
│       └── vite.config.ts
│
├── scripts/                     # Training and utility scripts
│   ├── train_human_detection.py
│   ├── train_scene_segmentation.py
│   └── prepare_dataset.py
│
├── tests/
│   ├── conftest.py              # Shared test configuration
│   ├── unit/                    # Unit tests (no I/O)
│   │   ├── core/                # Core logic tests
│   │   ├── hardware/            # Hardware abstraction tests
│   │   ├── services/            # Service layer tests
│   │   └── spatial/             # Spatial utility tests
│   ├── integration/             # API integration tests (FastAPI TestClient)
│   ├── backend/                 # Legacy zone overlap tests
│   ├── drone_control/           # DroneAPIClient tests
│   └── human_detection/         # Human detection tests
│
├── data/                        # Runtime data (persisted zones, logs)
│   └── zones.json
│
└── README.md
```

## How Detection Works

When a human is detected in a red zone:

1. YOLOv8 segmentation extracts person masks from the frame
2. The `check_overlap` function tests mask intersection with zone polygons
3. For intrusions, the feet position is estimated from the mask bottom
4. Lite-Mono depth estimation provides relative depth at that pixel
5. Ray-ground plane intersection calculates 3D world coordinates
6. Coordinates are sent to the drone API for automatic deployment

### Coordinate System

The system uses AirSim's NED (North-East-Down) coordinate frame:

- **X** - North (positive = forward)
- **Y** - East (positive = right)
- **Z** - Down (positive = below ground, negative = above ground)

## Troubleshooting

### AirSim Connection Failed

The backend starts in "limited mode" without AirSim. Video feeds will show "NO SIGNAL".

- Ensure AirSim is running before starting the backend
- Check that vehicle names in `config.py` match your `settings.json`

### Low Frame Rate

- Check GPU utilization with `nvidia-smi`
- Reduce `DETECTION_FPS` environment variable to lower GPU load
- Verify CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`

### Zones Not Persisting

Zones are saved to `src/backend/data/zones.json`. Ensure the backend has write permissions to this directory.

### Drone Not Responding to Commands

1. Verify drone API is running: `curl http://localhost:8000/status`
2. Check drone is in automatic mode for navigation commands
3. Verify target coordinates are within geofence limits
4. Check AirSim console for error messages

### Video Feed Freezing

- Frame capture and detection run on separate threads; if detection is slow, video should still be smooth
- Check backend logs for "Error capturing" messages
- Restart the backend if AirSim was restarted

## Development

### Code Style

The project uses Ruff for linting:

```bash
# Check for issues
ruff check src/

# Auto-fix issues
ruff check --fix src/
```

### Adding New Camera Feeds

1. Add the camera configuration to `FEED_CONFIG` in `src/backend/config.py`
2. Add metadata to `FEED_METADATA`
3. Restart the backend

### Environment Variables

For development, you can create a `.env` file:

```bash
FRAME_CAPTURE_FPS=30
DETECTION_FPS=5
BACKEND_PORT=8001
```

Load with: `export $(cat .env | xargs)` before running.
