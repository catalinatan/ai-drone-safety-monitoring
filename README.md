# AI Safety Monitoring

A safety monitoring system with drone control capabilities, featuring a web-based control panel and AirSim simulation integration.

## Quick Start

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install UI dependencies
cd src/ui && npm install

# Run tests
pytest
```

## Project Structure

```
ai-safety-monitoring/
├── src/
│   ├── drone-control/     # FastAPI backend + AirSim integration
│   │   ├── drone.py       # Main control system
│   │   └── config.yaml    # Safety & navigation parameters
│   └── ui/                # React frontend
│       └── src/
│           └── components/
│               ├── CommandPanel.tsx       # CCTV feed monitoring
│               └── DroneControlPanel.tsx  # Drone flight controls
├── requirements.txt       # Python dependencies
└── README.md
```

## Running Tests

```bash
# Run all Python tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_example.py
```

---

## Drone Control System

The drone control module (`src/drone-control/drone.py`) provides manual and automatic flight modes for a simulated drone via AirSim, exposed through a FastAPI server.

### Prerequisites

1. **AirSim** running on your PC desktop with a simulation environment loaded (e.g., AirSim Neighbourhood, Blocks).
2. **Python 3.10+** with dependencies installed:
3. **Node.js 18+** for the web UI

```bash
pip install -r requirements.txt
```

> Note: The `keyboard` library requires **administrator/root privileges** on Linux. On Windows, it works without elevation.

### Configuration

Edit `src/drone-control/config.yaml` before running:

```yaml
safety:
  max_altitude: 50       # Maximum flight altitude in meters
  geofence_radius: 200   # Maximum distance from origin in meters

navigation:
  speed: 5.0             # Automatic navigation speed in m/s
  position_tolerance: 1.0 # Distance in meters to consider "arrived"
```

### Running

1. Start your AirSim simulation environment and confirm the drone is visible.
2. Run the drone control system:

```bash
cd src/drone-control
python drone.py
```

On startup, the system:
- Launches a **FastAPI server** on `http://localhost:8000`
- Connects to AirSim, arms the drone, and takes off
- Enters the **main control loop** in manual mode
- Runs a **demo sequence** that exercises both modes automatically

### What the Demo Sequence Does

The built-in demo thread runs automatically after startup and exercises the full API:

| Step | Action | What to Observe |
|------|--------|-----------------|
| 1 | `GET /status` | Confirms initial state is manual mode |
| 2 | `POST /mode` → automatic | Drone switches to automatic, local window shows `Mode: AUTOMATIC` |
| 3 | `POST /goto` | Drone flies to NED position (11m north, 8m east, 10m up) |
| 4 | Poll `GET /status` | Logs distance every second until arrival |
| 5 | `POST /mode` → manual | Overrides automatic, drone hovers in place |
| 6 | 5-second pause | Keyboard controls active (w/a/s/d/z/x to move, q to quit) |
| 7 | `POST /mode` → automatic, `POST /return_home` | Drone flies back to takeoff position |
| 8 | `POST /mode` → manual | Returns to manual for continued operation |

Watch the terminal output for `[DEMO]` prefixed log lines showing each API call and response.

### Manual Keyboard Controls

While in manual mode, use the local keyboard:

| Key | Action |
|-----|--------|
| `w` | Move forward |
| `s` | Move backward |
| `a` | Move left |
| `d` | Move right |
| `z` | Move up |
| `x` | Move down |
| `q` | Quit (lands drone and exits) |

### API Endpoints

The FastAPI server exposes the following endpoints. Interactive docs are available at `http://localhost:8000/docs`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/mode` | Switch mode: `{"mode": "manual"}` or `{"mode": "automatic"}` |
| `POST` | `/goto` | Fly to NED coordinates (automatic mode only): `{"x": float, "y": float, "z": float}` |
| `POST` | `/move` | Send velocity command (manual mode only): `{"vx": float, "vy": float, "vz": float}` |
| `POST` | `/return_home` | Return to takeoff position (automatic mode only) |
| `GET` | `/status` | Current mode, navigation state, and target position |
| `GET` | `/video_feed` | MJPEG camera stream from the drone |

### Testing Manually via curl

After the demo completes, you can test the API directly:

```bash
# Check status
curl http://localhost:8000/status

# Switch to automatic
curl -X POST http://localhost:8000/mode -H "Content-Type: application/json" -d '{"mode": "automatic"}'

# Fly to a NED coordinate (11m north, 8m east, 10m above ground)
curl -X POST http://localhost:8000/goto -H "Content-Type: application/json" -d '{"x": 11.0, "y": 8.0, "z": -10.0}'

# Override back to manual
curl -X POST http://localhost:8000/mode -H "Content-Type: application/json" -d '{"mode": "manual"}'

# Send velocity command (manual mode) - move forward at 3 m/s
curl -X POST http://localhost:8000/move -H "Content-Type: application/json" -d '{"vx": 3.0, "vy": 0, "vz": 0}'

# Stop movement
curl -X POST http://localhost:8000/move -H "Content-Type: application/json" -d '{"vx": 0, "vy": 0, "vz": 0}'

# Return home
curl -X POST http://localhost:8000/mode -H "Content-Type: application/json" -d '{"mode": "automatic"}'
curl -X POST http://localhost:8000/return_home
```

### Architecture

```
┌─────────────────────────────────────────────┐
│                  Main Thread                 │
│                                              │
│  drone_control_loop()                        │
│  ├── Camera capture (both modes)             │
│  ├── Automatic: dispatch nav / check distance│
│  └── Manual: API velocity or keyboard input  │
└──────────────────┬──────────────────────────-┘
                   │ reads/writes
            ┌──────┴──────┐
            │ DroneState  │  (thread-safe, lock-protected)
            └──────┬──────┘
                   │ reads/writes
┌──────────────────┴──────────────────────────-┐
│              FastAPI Thread                   │
│                                               │
│  /mode, /goto, /move, /return_home, /status   │
│  /video_feed (MJPEG stream from DroneState)   │
└───────────────────────────────────────────────┘
                   │
                   │ HTTP (CORS enabled)
                   ▼
┌───────────────────────────────────────────────┐
│              Web UI (React)                   │
│                                               │
│  DroneControlPanel.tsx                        │
│  ├── Video feed display                       │
│  ├── Manual controls (WASD + ZX)              │
│  ├── Mode toggle (Manual/Automatic)           │
│  └── Return Home button                       │
└───────────────────────────────────────────────┘
```

---

## Web UI

The React-based web interface provides a visual control panel for drone operations.

### Running the UI

```bash
# Terminal 1: Start the drone backend (requires AirSim running)
python src/drone-control/drone.py

# Terminal 2: Start the web UI
cd src/ui
npm install   # First time only
npm run dev
```

Open http://localhost:5173 and navigate to **Drone Control**.

### UI Features

| Feature | Description |
|---------|-------------|
| **Video Feed** | Live MJPEG stream from drone camera |
| **Connection Status** | Shows CONNECTED/DISCONNECTED state |
| **Mode Toggle** | Switch between Manual and Automatic flight |
| **Movement Controls** | WASD keys or on-screen buttons (Manual mode) |
| **Altitude Controls** | Z/X keys or on-screen buttons (Manual mode) |
| **Return Home** | One-click return to takeoff position |
| **Deploy Equipment** | Placeholder for equipment deployment |

### Keyboard Shortcuts (when UI is focused)

| Key | Action |
|-----|--------|
| `W` | Move forward (North) |
| `S` | Move backward (South) |
| `A` | Move left (West) |
| `D` | Move right (East) |
| `Z` | Move up |
| `X` | Move down |
