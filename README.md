# AI Safety Monitoring

## Drone Control System

The drone control module (`src/drone-control/drone.py`) provides manual and automatic flight modes for a simulated drone via AirSim, exposed through a FastAPI server.

### Prerequisites

1. **AirSim** running on your PC desktop with a simulation environment loaded (e.g., AirSim Neighbourhood, Blocks).
2. **Python 3.10+** with dependencies installed:

```bash
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
pip install fastapi uvicorn pyyaml requests
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
│  └── Manual: read keyboard / send velocity   │
└──────────────────┬──────────────────────────-┘
                   │ reads/writes
            ┌──────┴──────┐
            │ DroneState  │  (thread-safe, lock-protected)
            └──────┬──────┘
                   │ reads/writes
┌──────────────────┴──────────────────────────-┐
│              FastAPI Thread                   │
│                                               │
│  /mode, /goto, /return_home, /status          │
│  /video_feed (MJPEG stream from DroneState)   │
└───────────────────────────────────────────────┘
```
