# AI Safety Monitoring

A dual-drone safety monitoring system with real-time CCTV feeds and flight control capabilities, featuring a web-based Command Center and AirSim simulation integration.

## Quick Start

```bash
# 1. Install Python dependencies
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt

# 2. Configure AirSim (see AirSim Setup section below)
# Edit C:\Users\<YourUsername>\Documents\AirSim\settings.json

# 3. Start AirSim simulation environment (Unreal Engine)

# 4. Start the observer camera backend (Terminal 1)
cd src/drone-control
python observer.py

# 5. Start the drone control backend (Terminal 2)
cd src/drone-control
python drone.py

# 6. Start the web UI (Terminal 3)
cd src/ui
npm install  # First time only
npm run dev

# 7. Open http://localhost:5173
```

## Project Structure

```
ai-safety-monitoring/
├── src/
│   ├── drone-control/       # FastAPI backends + AirSim integration
│   │   ├── drone.py         # Controllable drone system (port 8000)
│   │   ├── observer.py      # Static CCTV observer feed (port 8001)
│   │   └── config.yaml      # Safety & navigation parameters
│   └── ui/                  # React frontend (port 5173)
│       └── src/
│           └── components/
│               ├── CommandPanel.tsx       # CCTV feed grid (2x2)
│               ├── DroneControlPanel.tsx  # Drone flight controls
│               └── FeedCard.tsx           # Individual camera tile
├── requirements.txt         # Python dependencies (opencv-python, airsim, fastapi)
└── README.md
```

## System Architecture

This system uses **two drones** in AirSim:
- **Drone1**: Controllable search drone with manual/automatic flight modes
- **Drone2**: Static observer drone providing fixed CCTV-style aerial coverage

```
┌─────────────────────────────────────────────────────────────┐
│                    AirSim Simulation                         │
│  ┌──────────────┐              ┌──────────────┐            │
│  │   Drone1     │              │   Drone2     │            │
│  │ (Controllable)│             │ (Observer)    │            │
│  │  Camera 0    │              │  Camera 0    │            │
│  └──────┬───────┘              └──────┬───────┘            │
└─────────┼──────────────────────────────┼──────────────────-┘
          │                              │
          │ AirSim API                   │ AirSim API
          ▼                              ▼
┌─────────────────────┐        ┌─────────────────────┐
│   drone.py          │        │   observer.py       │
│   Port 8000         │        │   Port 8001         │
│                     │        │                     │
│ - Manual controls   │        │ - Camera capture    │
│ - Auto navigation   │        │ - MJPEG stream      │
│ - Camera feed       │        │ - Hovering at 3m    │
└──────────┬──────────┘        └──────────┬──────────┘
           │                              │
           │ HTTP/CORS                    │ HTTP/CORS
           └──────────────┬───────────────┘
                          ▼
              ┌───────────────────────┐
              │   React Web UI        │
              │   Port 5173           │
              │                       │
              │ - Command Center      │
              │   (CCTV Grid)         │
              │ - Drone Control Panel │
              └───────────────────────┘
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

## AirSim Setup

### Prerequisites

1. **AirSim** installed and running with an Unreal Engine environment (e.g., Neighbourhood, Blocks, LandscapeMountains)
2. **Python 3.10+** with dependencies installed
3. **Node.js 18+** for the web UI

```bash
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
```

### AirSim Configuration

Create or edit `C:\Users\<YourUsername>\Documents\AirSim\settings.json`:

```j1. Observer Camera System (`observer.py`)

Provides a static CCTV-style feed from Drone2 positioned above the scene.

**Running:**
```bash
cd src/drone-control
python observer.py
```

**What it does:**
- Connects to Drone2 in AirSim
- Enables API control and arms the drone
- Takes off to 3m altitude and holds position
- Streams live MJPEG video from camera 0 on port 8001

**Endpoints:**
- `GET /status` - Observer camera status
- `GET /video_feed/0` - MJPEG video stream (camera 0)

### 2. Controllable Drone System (`drone.py`)

Provides manual and automatic flight modes for Drone1.

**Configuration:**

Edit `src/drone-control/config.yaml` before running:

```yaml
safety:
  max_altitude: 50       # Maximum flight altitude in meters
  geofence_radius: 200   # Maximum distance from origin in meters

navigation:
  speed: 5.0             # Automatic navigation speed in m/s
  position_tolerance: 1.0 # Distance in meters to consider "arrived"
```

**Running:**      "Y": -9.76,
      "Z": -2,
      "Pitch": -50,
      "Yaw": 90,
      "AllowAPIAlways": true,
      "AutoFlyOnSpawn": false
    }
  },
  "CameraDefaults": {
    "CaptureSettings": [
      {
        "ImageType": 0,
        "Width": 1280,
        "Height": 720,
        "FOV_Degrees": 90
      }
    ]
  }
}
```

**Important**: 
- **Drone2** is positioned ~20m away from Drone1 as a static observer
- **Pitch: -50** tilts Drone2's body downward for ground coverage
- **Yaw: 90** rotates Drone2 90° clockwise for side viewing angle
- **AllowAPIAlways: true** enables programmatic hovering
- You must **restart AirSim** after editing settings.json

---

## Drone Control System

The system consists of two backend servers that control separate drones in AirSim.

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

#  Troubleshooting

### Observer camera shows static/frozen image
- Check terminal output for frame capture errors
- Verify AirSim is running and Drone2 is visible
- Restart observer.py: it should print "Captured X frames" periodically
- Confirm settings.json has correct Drone2 configuration

### Drone2 not appearing at correct position/angle
- Verify settings.json has Pitch and Yaw values set
- **Restart AirSim** completely (close Unreal Engine window)
- Settings are only read on AirSim startup, not during runtime

### CORS errors in browser console
- Verify both drone.py and observer.py are running
- Check that CORSMiddleware is enabled in both files
- Clear browser cache and reload

### Camera shows propellers/incorrect view
- Adjust Pitch angle in settings.json (range: -90 to 90)
- Negative pitch tilts the drone forward/down
- Restart AirSim after editing settings.json         └──────┬──────┘
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
┌───────────────────observer camera
cd src/drone-control
python observer.py

# Terminal 2: Start drone control
cd src/drone-control
python drone.py

# Terminal 3: Start web UI
cd src/ui
npm install   # First time only
npm run dev
```
#### Command Center (Main View)
| Feature | Description |
|---------|-------------|
| **CCTV Grid** | 2x2 grid of camera feeds |
| **Observer Feed** | Live aerial view from static Drone2 (top-left) |
| **Feed Controls** | Expand/zoom individual feeds |
| **Annotations** | Edit polygon zones on feeds |

#### Drone Control Panel
| Feature | Description |
|---------|-------------|
| **Video Feed** | Live MJPEG stream from Drone1 camera |
| **Connection Status** | Shows CONNECTED/DISCONNECTED state |
| **Mode Toggle** | Switch between Manual and Automatic flight |
| **Movement Controls** | WASD keys or on-screen buttons (Manual mode) |
| **Altitude Controls** | Z/X keys or on-screen buttons (Manual mode) |
| **Return Home** | One-click return to takeoff position
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
