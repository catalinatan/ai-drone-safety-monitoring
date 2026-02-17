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

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   React UI      │────▶│  Backend API    │────▶│  Drone Control  │
│   (Vite)        │     │  (FastAPI)      │     │  (FastAPI)      │
│   Port 5173     │     │  Port 8001      │     │  Port 8000      │
└─────────────────┘     └────────┬────────┘     └────────┬────────┘
                                 │                       │
                                 ▼                       ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │  AirSim CCTV    │     │  AirSim Drone   │
                        │  Cameras        │     │  Multirotor     │
                        └─────────────────┘     └─────────────────┘
```

The backend runs three concurrent threads:

- **Frame capture thread** (30 FPS) - Grabs frames from AirSim cameras
- **Detection thread** (30 FPS) - Runs YOLO segmentation + depth estimation
- **MJPEG streaming** (30 FPS) - Serves video feeds to the UI

## Requirements

- Python 3.11+
- Node.js 18+ (for UI)
- AirSim simulator with Unreal Engine environment
- CUDA-capable GPU recommended for real-time detection

## Installation

Clone the repository and install dependencies:

```bash
git clone <repository-url>
cd ai-safety-monitoring

# Install numpy first (required by airsim's build process)
pip install numpy

# Python dependencies (creates editable install)
pip install -e ".[dev]"

# UI dependencies
cd src/ui
npm install
cd ../..
```

### Model Weights

The system requires pre-trained model weights:

1. **YOLOv8 segmentation** - Downloaded automatically on first run
2. **Lite-Mono depth model** - Place weights in:
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
python -m src.backend.server       # Backend API
python -m src.drone_control.drone  # Drone control
cd src/ui && npm run dev           # React UI
```

## Configuration

All settings are centralized in `src/backend/config.py` and can be overridden via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `BACKEND_PORT` | 8001 | Backend API port |
| `DRONE_API_PORT` | 8000 | Drone control API port |
| `FRONTEND_PORT` | 5173 | React UI port |
| `FRAME_CAPTURE_FPS` | 30 | Frame capture rate from AirSim |
| `STREAM_FPS` | 30 | MJPEG streaming rate to UI |
| `DETECTION_FPS` | 30 | Human detection rate |
| `ALARM_COOLDOWN` | 5.0 | Seconds between drone deployments |
| `CCTV_HEIGHT` | 15.0 | Camera height in meters |
| `SAFE_Z_ALTITUDE` | -10.0 | Drone flight altitude (NED, negative = above ground) |

### AirSim Configuration

The system expects specific camera and vehicle names. Default configuration in `config.py`:

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

Run the test suite:

```bash
# All tests
pytest tests/ -v

# Unit tests only (no external dependencies)
pytest tests/ -m "not integration"

# Specific module
pytest tests/backend/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=html
```

## Project Structure

```
ai-safety-monitoring/
├── main.py                     # Development entrypoint (launches all services)
├── pyproject.toml              # Dependencies and tool configuration
├── pytest.ini                  # Test configuration
│
├── src/
│   ├── backend/
│   │   ├── server.py           # FastAPI backend server
│   │   ├── config.py           # Centralized configuration
│   │   └── data/
│   │       └── zones.json      # Persisted zone definitions
│   │
│   ├── drone_control/
│   │   ├── drone.py            # Drone control with manual/auto modes
│   │   ├── observer.py         # AirSim observer utilities
│   │   └── config.yaml         # Drone safety parameters
│   │
│   ├── cctv_monitoring/
│   │   ├── cctv_monitoring.py  # Standalone monitoring script
│   │   ├── coord_utils.py      # 3D coordinate calculation
│   │   ├── depth_estimation_utils.py
│   │   └── lite_mono_weights/  # Depth model weights
│   │
│   ├── human_detection/
│   │   ├── detector.py         # YOLOv8 segmentation wrapper
│   │   ├── check_overlap.py    # Zone overlap detection
│   │   └── config.py           # Detection thresholds
│   │
│   └── ui/                     # React frontend
│       ├── src/
│       ├── package.json
│       └── vite.config.ts
│
└── tests/
    ├── conftest.py             # Shared test fixtures
    ├── backend/                # Backend API tests
    ├── cctv_monitoring/        # Coordinate utility tests
    ├── drone_control/          # Drone API client tests
    └── human_detection/        # Detection tests
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
