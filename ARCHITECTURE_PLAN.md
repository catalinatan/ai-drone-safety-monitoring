# Architecture Refactoring Plan

**Project**: AI Safety Monitoring System
**Date**: 2026-03-30
**Status**: Proposed

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State Analysis](#2-current-state-analysis)
3. [Proposed Directory Structure](#3-proposed-directory-structure)
4. [Hardware Abstraction Layer](#4-hardware-abstraction-layer)
5. [Module Breakdown](#5-module-breakdown)
6. [Redundant Code Removal](#6-redundant-code-removal)
7. [Configuration System Redesign](#7-configuration-system-redesign)
8. [Testing Strategy](#8-testing-strategy)
9. [Migration Plan](#9-migration-plan)
10. [Future-Ready Hooks (Deferred)](#10-future-ready-hooks-deferred)

---

## 1. Executive Summary

Refactor the monolithic PoC into a modular, production-ready system where:

- **Camera and drone hardware is abstracted** behind interfaces, making the simulator swappable with real hardware via configuration alone.
- **Redundant code is eliminated** — 7 categories of dead/duplicate code are removed.
- **The monolithic `server.py` (~96KB) is decomposed** into focused modules.
- **A comprehensive test suite** validates every layer independently.
- **Core logic remains functionally identical** — same detection pipeline, same zone system, same drone dispatch behavior.

---

## 2. Current State Analysis

### What Works

- Human detection pipeline (YOLO segmentation + overlap checking)
- Zone management (red/yellow/green with binary mask overlap)
- Auto-segmentation for scene-based zone suggestion
- Depth estimation for 3D coordinate mapping
- Drone dispatch and manual handoff
- MJPEG streaming to React UI
- FastAPI endpoints for feeds, zones, status

### What's Wrong

| Problem | Impact |
|---------|--------|
| `server.py` is ~96KB with 6+ responsibilities | Impossible to test or modify safely |
| AirSim is hardcoded into camera capture, drone control, and coordinate systems | Cannot swap to real hardware |
| `cctv_monitoring.py` duplicates 80% of `server.py` logic | Confusion about which is canonical |
| `observer.py` is entirely superseded | Dead code |
| `check_overlap.py` has 2 dead functions | Misleading API surface |
| `src/utils.py` has 2 unused functions | Clutter |
| Training scripts duplicate `setup_logger()` | Inconsistent logging |
| Config is split across 3 files with different patterns | Hard to understand what controls what |

---

## 3. Proposed Directory Structure

```
ai-safety-monitoring/
├── main.py                          # Development launcher (simplified)
├── pyproject.toml
├── Dockerfile
├── .gitlab-ci.yml
├── ARCHITECTURE_PLAN.md
│
├── config/
│   ├── default.yaml                 # All settings in one place
│   └── feeds.yaml                   # Camera feed definitions (user-editable)
│
├── src/
│   ├── __init__.py
│   │
│   ├── core/                        # Pure business logic — no I/O, no frameworks
│   │   ├── __init__.py
│   │   ├── zone_manager.py          # Zone CRUD, polygon→mask, overlap checking
│   │   ├── alarm.py                 # Alarm state machine (cooldown, thresholds)
│   │   ├── detection_pipeline.py    # Orchestrates: frame → detect → check zones → alarm
│   │   └── models.py               # Pydantic models (Zone, FeedStatus, AlarmEvent, etc.)
│   │
│   ├── hardware/                    # Hardware Abstraction Layer (HAL)
│   │   ├── __init__.py
│   │   ├── camera/
│   │   │   ├── __init__.py
│   │   │   ├── base.py             # CameraBackend ABC
│   │   │   ├── airsim_camera.py    # AirSim implementation
│   │   │   ├── rtsp_camera.py      # RTSP/IP camera implementation (stub)
│   │   │   └── file_camera.py      # Video file playback (for testing/demos)
│   │   │
│   │   └── drone/
│   │       ├── __init__.py
│   │       ├── base.py             # DroneBackend ABC
│   │       ├── airsim_drone.py     # AirSim drone implementation
│   │       └── mavlink_drone.py    # MAVLink/PX4 implementation (stub)
│   │
│   ├── detection/                   # ML inference modules
│   │   ├── __init__.py
│   │   ├── human_detector.py       # HumanDetector (from current human_detection/)
│   │   ├── scene_segmenter.py      # SceneSegmenter (from current auto_segmentation.py)
│   │   └── depth_estimator.py      # LiteMonoDepthSystem (from depth_estimation_utils.py)
│   │
│   ├── spatial/                     # Coordinate transforms and geometry
│   │   ├── __init__.py
│   │   ├── coord_utils.py          # get_feet_from_mask, pixel→world transforms
│   │   └── projection.py           # Camera intrinsics, depth→3D projection
│   │
│   ├── api/                         # FastAPI application
│   │   ├── __init__.py
│   │   ├── app.py                  # FastAPI app factory, CORS, lifespan
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── health.py           # GET /health
│   │   │   ├── feeds.py            # GET /feeds, GET /feeds/{id}/status
│   │   │   ├── zones.py            # POST /feeds/{id}/zones, auto-segmentation
│   │   │   ├── video.py            # GET /video_feed/{id} (MJPEG streaming)
│   │   │   └── drone.py            # Drone command proxy endpoints
│   │   └── dependencies.py         # FastAPI dependency injection (FeedManager, etc.)
│   │
│   ├── services/                    # Application-level orchestration
│   │   ├── __init__.py
│   │   ├── feed_manager.py         # FeedManager — manages camera feeds + state
│   │   ├── drone_dispatcher.py     # Drone dispatch logic + manual handoff
│   │   ├── streaming.py            # MJPEG frame encoding and streaming
│   │   └── zone_persistence.py     # Load/save zones to disk
│   │
│   ├── drone_server/                # Standalone drone control API (port 8000)
│   │   ├── __init__.py
│   │   ├── app.py                  # FastAPI app for drone endpoints
│   │   ├── drone_state.py          # DroneState with thread-safe CAS guards
│   │   ├── control_loop.py         # Navigation + manual control thread
│   │   └── config.yaml             # Drone-specific config (speed, altitude, etc.)
│   │
│   └── ui/                          # React frontend (unchanged location)
│       ├── src/
│       ├── package.json
│       └── ...
│
├── scripts/                         # Standalone utilities (not imported by main app)
│   ├── train_human_detection.py     # Fine-tuning script
│   ├── train_scene_segmentation.py  # Scene model training
│   ├── prepare_dataset.py           # Dataset reorganization
│   └── demo_validation.py           # Offline video demo
│
├── models/                          # Model weights directory
│   ├── human_detection/
│   ├── scene_segmentation/
│   │   ├── railway/
│   │   ├── ship/
│   │   └── bridge/
│   └── depth_estimation/
│       └── lite-mono-small_640x192/
│
├── data/                            # Runtime data (persisted zones, logs)
│   ├── zones.json
│   └── events/                      # Audit log directory (deferred)
│
└── tests/
    ├── conftest.py                  # Shared fixtures
    ├── unit/
    │   ├── core/
    │   │   ├── test_zone_manager.py
    │   │   ├── test_alarm.py
    │   │   └── test_detection_pipeline.py
    │   ├── detection/
    │   │   ├── test_human_detector.py
    │   │   └── test_scene_segmenter.py
    │   ├── spatial/
    │   │   └── test_coord_utils.py
    │   ├── hardware/
    │   │   ├── test_camera_base.py
    │   │   └── test_drone_base.py
    │   └── services/
    │       ├── test_feed_manager.py
    │       ├── test_drone_dispatcher.py
    │       └── test_zone_persistence.py
    ├── integration/
    │   ├── conftest.py              # TestClient fixture
    │   ├── test_api_health.py
    │   ├── test_api_feeds.py
    │   ├── test_api_zones.py
    │   ├── test_api_video.py
    │   └── test_drone_api.py
    └── e2e/
        └── test_detection_to_alarm.py  # Full pipeline: frame → detect → alarm
```

---

## 4. Hardware Abstraction Layer

This is the most critical architectural change. The HAL defines **interfaces** that the core system programs against, with concrete implementations selected at startup via configuration.

### 4.1 Camera Backend Interface

```python
# src/hardware/camera/base.py

from abc import ABC, abstractmethod
import numpy as np


class CameraBackend(ABC):
    """Abstract interface for all camera sources."""

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the camera. Returns True on success."""
        ...

    @abstractmethod
    def grab_frame(self) -> np.ndarray | None:
        """Capture a single BGR frame. Returns None if unavailable."""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Release camera resources."""
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        ...

    @property
    @abstractmethod
    def resolution(self) -> tuple[int, int]:
        """Returns (width, height)."""
        ...
```

**Implementations**:

| Class | Source | Use Case |
|-------|--------|----------|
| `AirSimCamera` | AirSim `simGetImage()` | Simulator development |
| `RTSPCamera` | OpenCV `VideoCapture(rtsp://...)` | Production IP cameras |
| `FileCamera` | OpenCV `VideoCapture(file_path)` | Testing, demos, validation |

### 4.2 Drone Backend Interface

```python
# src/hardware/drone/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class DronePosition:
    x: float  # North (meters)
    y: float  # East (meters)
    z: float  # Down (meters, negative = up)


@dataclass
class DroneStatus:
    mode: str               # "manual" | "automatic"
    is_navigating: bool
    position: DronePosition
    is_connected: bool


class DroneBackend(ABC):
    """Abstract interface for drone control."""

    @abstractmethod
    def connect(self) -> bool:
        ...

    @abstractmethod
    def goto(self, position: DronePosition, speed: float) -> bool:
        """Command the drone to fly to a position. Non-blocking."""
        ...

    @abstractmethod
    def get_status(self) -> DroneStatus:
        ...

    @abstractmethod
    def set_mode(self, mode: str) -> bool:
        """Switch between 'manual' and 'automatic'."""
        ...

    @abstractmethod
    def return_home(self) -> bool:
        ...

    @abstractmethod
    def grab_frame(self) -> np.ndarray | None:
        """Capture frame from the drone's onboard camera."""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        ...
```

**Implementations**:

| Class | Protocol | Use Case |
|-------|----------|----------|
| `AirSimDrone` | AirSim Python API | Simulator development |
| `MAVLinkDrone` | MAVLink via `pymavlink`/`dronekit` | PX4/ArduPilot hardware |

### 4.3 Factory Pattern for Backend Selection

```python
# src/hardware/__init__.py

def create_camera_backend(config: dict) -> CameraBackend:
    """Factory: instantiate the right camera backend from config."""
    backend_type = config["type"]  # "airsim", "rtsp", "file"
    if backend_type == "airsim":
        from .camera.airsim_camera import AirSimCamera
        return AirSimCamera(**config["params"])
    elif backend_type == "rtsp":
        from .camera.rtsp_camera import RTSPCamera
        return RTSPCamera(url=config["params"]["url"])
    elif backend_type == "file":
        from .camera.file_camera import FileCamera
        return FileCamera(path=config["params"]["path"])
    raise ValueError(f"Unknown camera backend: {backend_type}")
```

Configuration-driven selection (from `config/feeds.yaml`):

```yaml
feeds:
  cctv-1:
    name: "CCTV CAM 1"
    location: "Railway Platform A"
    scene_type: "railway"
    camera:
      type: "airsim"          # Change to "rtsp" for production
      params:
        camera_name: "0"
        vehicle_name: "Drone2"

  cctv-2:
    name: "CCTV CAM 2"
    location: "Bridge Section B"
    scene_type: "bridge"
    camera:
      type: "rtsp"
      params:
        url: "rtsp://192.168.1.100:554/stream1"
```

**To swap hardware**: change `type` and `params` in the YAML. Zero code changes required.

---

## 5. Module Breakdown

### 5.1 `src/core/` — Pure Business Logic

This layer has **no I/O, no network calls, no file access**. It receives data and returns results. This makes it trivially testable.

| Module | Responsibility | Key Functions/Classes |
|--------|---------------|----------------------|
| `zone_manager.py` | Zone CRUD, polygon-to-mask conversion, overlap calculation | `ZoneManager.add_zones()`, `ZoneManager.check_overlap(person_mask)` |
| `alarm.py` | Alarm state machine with cooldown logic | `AlarmState.trigger()`, `AlarmState.is_active`, `AlarmState.cooldown_remaining` |
| `detection_pipeline.py` | Orchestrates frame → detect → check zones → alarm | `DetectionPipeline.process_frame(frame) -> DetectionResult` |
| `models.py` | Pydantic data models shared across the system | `Zone`, `FeedStatus`, `AlarmEvent`, `DetectionResult` |

#### Where current code maps to:

| Current Location | New Location |
|-----------------|-------------|
| `server.py` → zone mask creation logic | `core/zone_manager.py` |
| `server.py` → alarm cooldown tracking | `core/alarm.py` |
| `server.py` → `FeedStatus` Pydantic model | `core/models.py` |
| `check_overlap.py` → `check_danger_zone_overlap()` | `core/zone_manager.py` |
| `server.py` → detection loop body | `core/detection_pipeline.py` |

### 5.2 `src/hardware/` — Hardware Abstraction

Covered in [Section 4](#4-hardware-abstraction-layer). This replaces all direct AirSim imports in `server.py` and `drone.py`.

### 5.3 `src/detection/` — ML Inference

| Module | Current Source | Changes |
|--------|---------------|---------|
| `human_detector.py` | `human_detection/detector.py` | Unchanged logic. Remove `_try_load_tensorrt` into a private helper. |
| `scene_segmenter.py` | `backend/auto_segmentation.py` | Unchanged logic. Return structured `Zone` objects instead of raw dicts. |
| `depth_estimator.py` | `cctv_monitoring/depth_estimation_utils.py` | Consolidate `DepthDecoder` + `LiteMonoDepthSystem` + loader into one module. |

### 5.4 `src/spatial/` — Coordinate Transforms

| Module | Current Source | Changes |
|--------|---------------|---------|
| `coord_utils.py` | `cctv_monitoring/coord_utils.py` | Keep `get_feet_from_mask()`, `unreal_to_airsim()`. Remove `get_user_defined_danger_zone()` (replaced by UI). |
| `projection.py` | `cctv_monitoring/coord_utils.py` | Extract `get_coords_from_lite_mono()` and camera projection math. |

### 5.5 `src/api/` — FastAPI Backend (Port 8001)

The current 96KB `server.py` gets decomposed into focused route modules:

| Route Module | Endpoints | Lines (est.) |
|-------------|-----------|-------------|
| `health.py` | `GET /health` | ~20 |
| `feeds.py` | `GET /feeds`, `GET /feeds/{id}/status` | ~60 |
| `zones.py` | `POST /feeds/{id}/zones` | ~80 |
| `video.py` | `GET /video_feed/{id}` | ~40 |
| `drone.py` | Proxy to drone server | ~40 |
| `app.py` | App factory, lifespan, CORS | ~50 |
| `dependencies.py` | DI for FeedManager, config | ~30 |

**Total: ~320 lines across 7 files vs ~2400 lines in one file.**

### 5.6 `src/services/` — Application Orchestration

| Module | Responsibility |
|--------|---------------|
| `feed_manager.py` | Manages all camera feeds: start/stop capture threads, hold per-feed state (FeedState), provide frames to API |
| `drone_dispatcher.py` | Receives alarm events, computes 3D target position, sends goto command to drone backend, triggers manual handoff |
| `streaming.py` | MJPEG encoding: `frame → JPEG bytes → multipart boundary wrapping` |
| `zone_persistence.py` | `load_zones_from_file()` / `save_zones_to_file()` — extracted from `server.py` |

### 5.7 `src/drone_server/` — Drone Control API (Port 8000)

Extracted from current `src/drone_control/drone.py`:

| Module | Responsibility |
|--------|---------------|
| `app.py` | FastAPI app with drone-specific endpoints |
| `drone_state.py` | Thread-safe `DroneState` with CAS guards (from `drone.py`) |
| `control_loop.py` | Navigation thread + manual keyboard control (from `drone.py`) |
| `config.yaml` | Drone config — speed, altitude, tolerances (already exists) |

Key change: `drone.py` currently creates the AirSim client directly. After refactor, it receives a `DroneBackend` instance via dependency injection.

---

## 6. Redundant Code Removal

### Files to Delete Entirely

| File | Reason |
|------|--------|
| `src/drone_control/observer.py` | Superseded by `FeedManager` in backend server |
| `src/cctv_monitoring/cctv_monitoring.py` | Superseded by backend server + React UI |
| `tests/human_detection/test_check_overlap.py` | Not a test — interactive webcam runner |
| `src/scene_detection/` (entire dir) | Already deleted in git, just commit the deletion |
| `src/human_detection/process_video.py` | Already deleted in git |
| `src/human_detection/realtime.py` | Already deleted in git |
| `src/human_detection/validate_model.py` | Already deleted in git |
| `src/human_detection/verify_dataset.py` | Already deleted in git |
| `src/main.py` | Already deleted in git |

### Functions to Remove

| File | Function | Reason |
|------|----------|--------|
| `src/human_detection/check_overlap.py` | `draw_danger_annotations()` | Only called by dead `run_safety_monitoring()` |
| `src/human_detection/check_overlap.py` | `run_safety_monitoring()` | Standalone webcam loop, superseded by backend |
| `src/utils.py` | `get_device()` | Not imported anywhere |
| `src/utils.py` | `load_checkpoint()` | Not imported anywhere |
| `src/cctv_monitoring/coord_utils.py` | `get_user_defined_danger_zone()` | Manual OpenCV annotation, replaced by React UI |

### Code to Consolidate

| Duplicate | Canonical Location | Action |
|-----------|-------------------|--------|
| `setup_logger()` in `human_detection/train.py` | `src/logger.py` → `get_logger()` | Replace with import |
| `setup_logger()` in `scene_segmentation/train.py` | `src/logger.py` → `get_logger()` | Replace with import |
| `DroneAPIClient` in `cctv_monitoring.py` | `server.py` → `DroneAPIClient` | Delete the duplicate (file is being deleted) |
| `draw_alarm_overlay()` in `cctv_monitoring.py` | `server.py` → overlay logic | Delete the duplicate (file is being deleted) |

---

## 7. Configuration System Redesign

### Current State: 3 fragmented config files

- `src/backend/config.py` — backend settings (env vars)
- `src/human_detection/config.py` — detection model settings (constants)
- `src/drone_control/config.yaml` — drone settings (YAML)

### Proposed: 2 unified YAML files

**`config/default.yaml`** — all system settings with sensible defaults:

```yaml
# Server
server:
  backend_port: 8001
  drone_api_port: 8000
  frontend_port: 5173

# Detection
detection:
  model_path: "models/human_detection/yolo11n-seg.pt"
  confidence_threshold: 0.25
  inference_imgsz: 1280
  fps: 30
  warmup_frames: 20
  min_person_area_pixels: 2

# Zones
zones:
  overlap_threshold: 0.5
  alarm_cooldown_seconds: 5.0
  persistence_file: "data/zones.json"

# Auto-Segmentation
auto_segmentation:
  interval_seconds: 60.0
  confidence: 0.5
  simplify_epsilon: 2.0
  min_contour_area: 40.0
  models:
    railway: "models/scene_segmentation/railway/best.pt"
    ship: "models/scene_segmentation/ship/best.pt"
    bridge: "models/scene_segmentation/bridge/best.pt"

# Depth Estimation
depth_estimation:
  encoder_path: "models/depth_estimation/lite-mono-small_640x192/encoder.pth"
  decoder_path: "models/depth_estimation/lite-mono-small_640x192/depth.pth"

# Streaming
streaming:
  capture_fps: 30
  stream_fps: 30

# Drone
drone:
  api_url: "http://localhost:8000"
  api_timeout: 5
  safe_altitude: -10.0

# Follow Mode (AirSim-specific)
follow_mode:
  target: ""  # "ship", "railway", etc. Empty = stationary
  hover_drones: false
  hover_altitude: -15.0
  follow_interval: 0.01
  camera_mappings:
    Drone2: "CCTV1"
    Drone3: "CCTV2"
    Drone4: "CCTV3"
    Drone5: "CCTV4"
```

**`config/feeds.yaml`** — feed definitions (user-editable, per-site):

```yaml
feeds:
  cctv-1:
    name: "CCTV CAM 1"
    location: "Railway Platform A"
    scene_type: "railway"
    camera:
      type: "airsim"
      params:
        camera_name: "0"
        vehicle_name: "Drone2"

  cctv-2:
    name: "CCTV CAM 2"
    location: "Bridge Section B"
    scene_type: "bridge"
    camera:
      type: "airsim"
      params:
        camera_name: "0"
        vehicle_name: "Drone3"
```

**Config loading** (`src/core/config.py`):

- Reads YAML files at startup
- Environment variables override any YAML value (e.g., `DETECTION_FPS=15` overrides `detection.fps`)
- Exposes a frozen `AppConfig` Pydantic model validated at startup — fail fast on bad config

---

## 8. Testing Strategy

### 8.1 Test Pyramid

```
         /  E2E  \          ~5 tests    — full pipeline with mocked hardware
        / Integration \     ~20 tests   — API endpoints via TestClient
       /    Unit Tests  \   ~60 tests   — pure logic, no I/O
      -------------------
```

### 8.2 Unit Tests (`tests/unit/`)

**No mocks needed** — these test pure functions and classes.

| Test File | What It Tests | Key Cases |
|-----------|--------------|-----------|
| `test_zone_manager.py` | Polygon → mask, overlap calculation, zone CRUD | Empty zones, overlapping zones, edge-of-frame polygons |
| `test_alarm.py` | Alarm state transitions, cooldown timing | Trigger → active → cooldown → reset, rapid re-triggers |
| `test_detection_pipeline.py` | Pipeline orchestration with injected fakes | No detections, single person in red zone, multiple people in yellow zone |
| `test_human_detector.py` | Detector with mocked YOLO model | Empty frame, single person, batch processing, confidence filtering |
| `test_scene_segmenter.py` | Segmenter with mocked YOLO model | Railway/ship/bridge scenes, no detections, polygon simplification |
| `test_coord_utils.py` | `get_feet_from_mask()`, coordinate transforms | Empty mask, single pixel, vertical person, irregular shape |
| `test_camera_base.py` | Camera interface contract tests | Verifies all implementations satisfy the ABC contract |
| `test_drone_base.py` | Drone interface contract tests | Same — ensures implementations are interchangeable |
| `test_feed_manager.py` | Feed lifecycle, state management | Add/remove feeds, frame caching, concurrent access |
| `test_drone_dispatcher.py` | Dispatch logic, manual handoff | Dispatch to position, already navigating, drone unreachable |
| `test_zone_persistence.py` | Load/save zones | Round-trip, missing file, corrupted JSON |

### 8.3 Integration Tests (`tests/integration/`)

Use FastAPI `TestClient` with mocked hardware backends.

| Test File | What It Tests |
|-----------|--------------|
| `test_api_health.py` | `GET /health` returns status and feed count |
| `test_api_feeds.py` | `GET /feeds` lists all feeds with correct structure |
| `test_api_zones.py` | `POST /feeds/{id}/zones` creates masks, persists, handles errors |
| `test_api_video.py` | `GET /video_feed/{id}` returns MJPEG stream |
| `test_drone_api.py` | Drone command endpoints with mocked DroneBackend |

### 8.4 End-to-End Tests (`tests/e2e/`)

Full pipeline tests with **fake hardware backends** (FileCamera + mock drone):

```python
# tests/e2e/test_detection_to_alarm.py

def test_person_in_red_zone_triggers_alarm():
    """
    Given: A feed with a red zone configured
    When:  A frame containing a person inside the red zone is processed
    Then:  Alarm is triggered AND drone dispatch is initiated
    """
    camera = FileCamera("tests/fixtures/person_in_zone.jpg")
    drone = MockDroneBackend()
    pipeline = build_pipeline(camera=camera, drone=drone, zones=[RED_ZONE])

    result = pipeline.process_single_frame()

    assert result.alarm_active is True
    assert drone.goto_called is True

def test_person_outside_zones_no_alarm():
    """No alarm when person is detected but outside all zones."""
    ...
```

### 8.5 Hardware Contract Tests

Each hardware backend must pass the same **contract test suite**:

```python
# tests/unit/hardware/test_camera_base.py

class CameraContractTests:
    """Mixin — any CameraBackend implementation must pass these."""

    def get_backend(self) -> CameraBackend:
        raise NotImplementedError

    def test_grab_frame_returns_numpy_or_none(self):
        backend = self.get_backend()
        backend.connect()
        frame = backend.grab_frame()
        assert frame is None or isinstance(frame, np.ndarray)

    def test_resolution_matches_frame(self):
        backend = self.get_backend()
        backend.connect()
        frame = backend.grab_frame()
        if frame is not None:
            h, w = frame.shape[:2]
            assert backend.resolution == (w, h)

    def test_disconnect_is_idempotent(self):
        backend = self.get_backend()
        backend.disconnect()  # no error even if never connected
        backend.disconnect()  # still no error


class TestFileCamera(CameraContractTests):
    def get_backend(self):
        return FileCamera("tests/fixtures/test_video.mp4")


class TestAirSimCamera(CameraContractTests):
    @pytest.mark.skipif(not AIRSIM_AVAILABLE, reason="AirSim not running")
    def get_backend(self):
        return AirSimCamera(camera_name="0", vehicle_name="Drone2")
```

### 8.6 Test Fixtures

```
tests/
├── fixtures/
│   ├── frames/
│   │   ├── empty_scene.jpg         # No people
│   │   ├── person_in_zone.jpg      # Person inside red zone
│   │   └── person_outside_zone.jpg # Person outside all zones
│   ├── zones/
│   │   └── sample_zones.json       # Valid zone config
│   └── models/
│       └── README.md               # Instructions for downloading test model weights
```

### 8.7 CI Integration

```yaml
# .gitlab-ci.yml (updated)
stages:
  - lint
  - test-unit
  - test-integration
  - build

lint:
  script:
    - ruff check src/ tests/

test-unit:
  script:
    - pytest tests/unit/ -v --cov=src --cov-report=term

test-integration:
  script:
    - pytest tests/integration/ -v -m integration

build:
  script:
    - docker build -t ai-safety-monitoring .
```

---

## 9. Migration Plan

### Guiding Principles

1. **Always green.** Every step ends with all tests passing.
2. **One concern per step.** Don't mix structural moves with logic changes.
3. **Test before you move.** Write the new test first, then move/refactor the code.
4. **Feature-branch per phase.** Each phase gets its own MR for review.

---

### Phase 0: Cleanup (Pre-Refactor)

**Goal**: Remove dead code and commit pending deletions. Zero logic changes.

| Step | Action | Validation |
|------|--------|------------|
| 0.1 | Commit all pending git deletions (deleted files in git status) | `git status` clean |
| 0.2 | Delete `src/drone_control/observer.py` | Grep confirms no imports |
| 0.3 | Delete `src/cctv_monitoring/cctv_monitoring.py` | Grep confirms no imports |
| 0.4 | Delete `tests/human_detection/test_check_overlap.py` (interactive runner) | Not a real test |
| 0.5 | Remove dead functions from `check_overlap.py` (`draw_danger_annotations`, `run_safety_monitoring`) | Existing tests pass |
| 0.6 | Remove unused functions from `src/utils.py` (`get_device`, `load_checkpoint`) | Grep confirms no imports |
| 0.7 | Replace duplicate `setup_logger()` in training scripts with import from `src/logger.py` | Training scripts still runnable |

**Estimated scope**: ~200 lines deleted, 0 lines of new logic.

---

### Phase 1: Foundation — Core Models and Config

**Goal**: Create the new directory structure, config system, and data models.

| Step | Action | Validation |
|------|--------|------------|
| 1.1 | Create `config/default.yaml` and `config/feeds.yaml` from current config values | Values match current defaults |
| 1.2 | Create `src/core/models.py` — extract Pydantic models from `server.py` | Import test passes |
| 1.3 | Create `src/core/config.py` — YAML loader with env var overrides | Unit test: loads config, overrides work |
| 1.4 | Update `server.py` to import models from `src/core/models.py` | All existing tests pass |
| 1.5 | Update `server.py` to read from new YAML config (keeping `config.py` as fallback) | All existing tests pass |

---

### Phase 2: Hardware Abstraction Layer

**Goal**: Define interfaces and implement AirSim backends without changing behavior.

| Step | Action | Validation |
|------|--------|------------|
| 2.1 | Create `src/hardware/camera/base.py` (ABC) | Contract tests pass |
| 2.2 | Create `src/hardware/camera/airsim_camera.py` — extract from `server.py` | Contract tests pass |
| 2.3 | Create `src/hardware/camera/file_camera.py` — for testing | Contract tests pass |
| 2.4 | Create `src/hardware/drone/base.py` (ABC) | Contract tests pass |
| 2.5 | Create `src/hardware/drone/airsim_drone.py` — extract from `drone.py` | Contract tests pass |
| 2.6 | Create factory functions in `src/hardware/__init__.py` | Factory returns correct type |
| 2.7 | Update `FeedManager` to use `CameraBackend` instead of direct AirSim calls | All existing tests pass |

---

### Phase 3: Core Business Logic Extraction

**Goal**: Extract pure logic from `server.py` into `src/core/`.

| Step | Action | Validation |
|------|--------|------------|
| 3.1 | Create `src/core/zone_manager.py` — extract zone logic + `check_danger_zone_overlap()` | New unit tests + existing overlap tests pass |
| 3.2 | Create `src/core/alarm.py` — extract alarm state machine | New unit tests pass |
| 3.3 | Create `src/core/detection_pipeline.py` — orchestration | New unit tests pass |
| 3.4 | Update `server.py` to delegate to core modules | All integration tests pass |

---

### Phase 4: Service Layer Extraction

**Goal**: Extract application services from `server.py`.

| Step | Action | Validation |
|------|--------|------------|
| 4.1 | Create `src/services/feed_manager.py` — extract from `server.py` | New unit tests + integration tests pass |
| 4.2 | Create `src/services/drone_dispatcher.py` — extract dispatch logic | New unit tests pass |
| 4.3 | Create `src/services/streaming.py` — extract MJPEG encoding | Unit test: frame → JPEG bytes |
| 4.4 | Create `src/services/zone_persistence.py` — extract load/save | Unit test: round-trip |

---

### Phase 5: API Decomposition

**Goal**: Split the monolithic `server.py` into route modules.

| Step | Action | Validation |
|------|--------|------------|
| 5.1 | Create `src/api/app.py` — app factory with lifespan | App starts |
| 5.2 | Create `src/api/dependencies.py` — DI for FeedManager, config | Injection works |
| 5.3 | Create `src/api/routes/health.py` | Integration test passes |
| 5.4 | Create `src/api/routes/feeds.py` | Integration test passes |
| 5.5 | Create `src/api/routes/zones.py` | Integration test passes |
| 5.6 | Create `src/api/routes/video.py` | Integration test passes |
| 5.7 | Create `src/api/routes/drone.py` | Integration test passes |
| 5.8 | Delete `src/backend/server.py` | All tests pass against new routes |

---

### Phase 6: Detection Module Consolidation

**Goal**: Move ML modules into `src/detection/` and `src/spatial/`.

| Step | Action | Validation |
|------|--------|------------|
| 6.1 | Move `detector.py` → `src/detection/human_detector.py` | Tests pass |
| 6.2 | Move `auto_segmentation.py` → `src/detection/scene_segmenter.py` | Tests pass |
| 6.3 | Move depth estimation → `src/detection/depth_estimator.py` | Tests pass |
| 6.4 | Move coordinate utils → `src/spatial/coord_utils.py` + `projection.py` | Tests pass |
| 6.5 | Clean up old directories (`human_detection/`, `cctv_monitoring/`, `backend/`) | No dangling imports |

---

### Phase 7: Drone Server Refactor

**Goal**: Refactor drone control to use DroneBackend interface.

| Step | Action | Validation |
|------|--------|------------|
| 7.1 | Extract `DroneState` → `src/drone_server/drone_state.py` | Unit tests pass |
| 7.2 | Extract control loop → `src/drone_server/control_loop.py` | Tests pass |
| 7.3 | Create `src/drone_server/app.py` — inject DroneBackend | Integration tests pass |
| 7.4 | Delete `src/drone_control/drone.py` | All tests pass |

---

### Phase 8: Frontend Updates

**Goal**: Update React frontend to match any API contract changes.

| Step | Action | Validation |
|------|--------|------------|
| 8.1 | Audit API contract — identify any endpoint/payload changes | Document changes |
| 8.2 | Update TypeScript API client types if needed | Frontend builds |
| 8.3 | Update environment config (API URLs) | Frontend connects to backend |

Note: The API contract should remain **unchanged** through the refactor. This phase is a verification pass, not a rewrite. If we later add admin UI or event timeline, those are separate feature branches.

---

### Phase 9: Training Scripts and Utilities

**Goal**: Move standalone scripts out of `src/` into `scripts/`.

| Step | Action | Validation |
|------|--------|------------|
| 9.1 | Move `human_detection/train.py` → `scripts/train_human_detection.py` | Script runs |
| 9.2 | Move `scene_segmentation/train.py` → `scripts/train_scene_segmentation.py` | Script runs |
| 9.3 | Move `prepare_dataset.py` → `scripts/prepare_dataset.py` | Script runs |
| 9.4 | Move `demo_validation.py` → `scripts/demo_validation.py` | Script runs |
| 9.5 | Move model weights into `models/` directory structure | Paths in config updated |
| 9.6 | Move `data/zones.json` to top-level `data/` | Persistence still works |

---

### Phase 10: Final Cleanup and Documentation

| Step | Action | Validation |
|------|--------|------------|
| 10.1 | Delete empty old directories | Clean tree |
| 10.2 | Update `pyproject.toml` — remove `airsim` from required deps (make optional) | `pip install .` works without AirSim |
| 10.3 | Update `Dockerfile` | Docker build succeeds |
| 10.4 | Update `.gitlab-ci.yml` with new test structure | CI passes |
| 10.5 | Update `main.py` launcher | Dev startup works |
| 10.6 | Update README with new architecture | — |
| 10.7 | Full regression: run entire test suite | All green |

---

## 10. Future-Ready Hooks (Deferred)

These are **included in the architecture** but **not implemented** during this refactor. The directory structure and interfaces support them without changes.

### 10.1 Admin/Config UI

- **Where it goes**: New route module `src/api/routes/admin.py`
- **Endpoints**: `GET /config`, `PUT /config`, `GET /config/feeds`, `PUT /config/feeds`
- **Frontend**: New settings page in React
- **How it works**: Reads/writes `config/feeds.yaml` via API. Backend reloads config on change.
- **Risk**: None — purely additive.

### 10.2 Audit Event Log

- **Where it goes**: New service `src/services/event_logger.py`
- **Storage**: Rotating JSON files in `data/events/`
- **Integration point**: `DetectionPipeline` and `DroneDispatcher` emit events to the logger
- **Frontend**: Event timeline panel on the dashboard
- **Pattern**: Observer/pub-sub — core modules emit events, logger subscribes.

### 10.3 RTSP Camera Backend

- **Where it goes**: `src/hardware/camera/rtsp_camera.py` (stub already in structure)
- **Implementation**: OpenCV `VideoCapture` with RTSP URL, reconnection logic, frame buffering
- **Activation**: Change `type: "airsim"` to `type: "rtsp"` in `config/feeds.yaml`

### 10.4 MAVLink Drone Backend

- **Where it goes**: `src/hardware/drone/mavlink_drone.py` (stub already in structure)
- **Dependencies**: `pymavlink` or `dronekit` (added as optional dependency)
- **Activation**: Change drone backend type in config

---

## Appendix: File Migration Map

Quick reference showing where every current file ends up:

| Current File | New Location | Action |
|-------------|-------------|--------|
| `src/backend/server.py` | Split into `src/api/`, `src/services/`, `src/core/` | Decompose |
| `src/backend/config.py` | `config/default.yaml` + `src/core/config.py` | Replace |
| `src/backend/auto_segmentation.py` | `src/detection/scene_segmenter.py` | Move |
| `src/backend/data/zones.json` | `data/zones.json` | Move |
| `src/human_detection/detector.py` | `src/detection/human_detector.py` | Move |
| `src/human_detection/check_overlap.py` | `src/core/zone_manager.py` (only `check_danger_zone_overlap`) | Extract + delete |
| `src/human_detection/config.py` | `config/default.yaml` (detection section) | Merge |
| `src/human_detection/train.py` | `scripts/train_human_detection.py` | Move |
| `src/cctv_monitoring/coord_utils.py` | `src/spatial/coord_utils.py` + `src/spatial/projection.py` | Split |
| `src/cctv_monitoring/depth_estimation_utils.py` | `src/detection/depth_estimator.py` | Move |
| `src/cctv_monitoring/cctv_monitoring.py` | **DELETE** | Redundant |
| `src/cctv_monitoring/Lite-Mono/` | Keep as vendored dependency (or move to `vendor/`) | Keep |
| `src/drone_control/drone.py` | Split into `src/drone_server/` | Decompose |
| `src/drone_control/observer.py` | **DELETE** | Redundant |
| `src/drone_control/config.yaml` | `src/drone_server/config.yaml` | Move |
| `src/scene_segmentation/train.py` | `scripts/train_scene_segmentation.py` | Move |
| `src/utils.py` | `src/core/config.py` (only `find_project_root`) | Extract + delete |
| `src/logger.py` | `src/logger.py` (keep) | Keep |
| `src/prepare_dataset.py` | `scripts/prepare_dataset.py` | Move |
| `demo_validation.py` | `scripts/demo_validation.py` | Move |
| `main.py` | `main.py` (simplified) | Simplify |
| `tests/` | `tests/` (restructured per Section 8) | Rewrite |
