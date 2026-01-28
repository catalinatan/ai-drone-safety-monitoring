"""
Drone Control System — Manual & Automatic Modes (AirSim Simulation)

Architecture overview
---------------------
This module runs three concurrent threads from a single process:

  1. **Main thread** — `drone_control_loop()`
     Runs a ~30 FPS loop that (a) captures the camera feed from AirSim,
     (b) in *manual* mode reads keyboard input and sends velocity commands,
     or (c) in *automatic* mode dispatches / monitors a moveToPosition flight.

  2. **API thread** — `run_api_server()`
     A FastAPI (uvicorn) server on port 8000 that exposes REST endpoints for
     switching modes, issuing goto commands, returning home, querying status,
     and streaming the camera feed as MJPEG.

  3. **Demo thread** (optional) — `run_demo_sequence()`
     Exercises the API endpoints in order so you can verify the system
     end-to-end without an external client.

Thread safety
~~~~~~~~~~~~~
All mutable state lives in the `DroneState` singleton, whose every accessor
acquires a `threading.Lock`.  The API thread *writes* targets / mode; the
control-loop thread *reads* them via atomic snapshots (see `get_nav_snapshot`
and `try_mark_nav_dispatched`) to avoid TOCTOU races.

Coordinate system
~~~~~~~~~~~~~~~~~
All positions use AirSim's **NED** (North-East-Down) frame in metres.
  - x → North
  - y → East
  - z → Down  (negative values = above ground)
"""

import airsim
import time
import cv2
import numpy as np
import keyboard
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import threading
import uvicorn
from typing import Optional, Tuple
import math
import yaml
import os

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

with open(CONFIG_PATH, "r") as f:
    _cfg = yaml.safe_load(f)

# Safety parameters
MAX_ALTITUDE = _cfg["safety"]["max_altitude"]
GEOFENCE_RADIUS = _cfg["safety"]["geofence_radius"]

# Navigation parameters
POSITION_TOLERANCE = _cfg["navigation"]["position_tolerance"]
NAVIGATION_SPEED = _cfg["navigation"]["speed"]

# ============================================================================
# PYDANTIC MODELS (for FastAPI)
# ============================================================================

class ModeRequest(BaseModel):
    mode: str  # "manual" or "automatic"

class GotoRequest(BaseModel):
    x: float  # North (meters)
    y: float  # East (meters)
    z: Optional[float] = -10.0  # Down (meters, negative = above ground)

# ============================================================================
# SHARED STATE (Thread-Safe)
# ============================================================================

class DroneState:
    """Thread-safe shared state between the API server thread and the main
    control-loop thread.

    Locking strategy: every public method acquires ``self.lock`` so that
    callers never need to worry about concurrent access.  For the navigation
    hot-path the control loop uses ``get_nav_snapshot()`` +
    ``try_mark_nav_dispatched()`` to avoid a TOCTOU race when a new /goto
    arrives between reading the target and issuing the AirSim command.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.mode = "manual"              # "manual" | "automatic"
        self.target_position = None       # (x, y, z) NED metres, set by /goto
        self.home_position = None         # recorded once at takeoff
        self.is_navigating = False        # True while flying toward a target
        self.nav_command_sent = False     # True once moveToPositionAsync dispatched for current target
        self.nav_task = None              # AirSim future returned by moveToPositionAsync
        self.idle_hover_sent = False      # prevents issuing hover repeatedly when auto-mode has no target
        self.should_stop = False          # signals the control loop to land and exit
        self.current_frame = None         # latest camera frame (numpy array), served by /video_feed

    def set_mode(self, mode: str):
        """Switch mode. Switching to manual cancels any in-flight navigation."""
        with self.lock:
            self.mode = mode
            self.idle_hover_sent = False
            if mode == "manual":
                self.is_navigating = False
                self.nav_command_sent = False
                if self.nav_task is not None:
                    try:
                        self.nav_task.cancel()
                    except (AttributeError, Exception) as e:
                        print(f"[WARN] Could not cancel nav_task: {e}")
                    self.nav_task = None

    def get_mode(self) -> str:
        with self.lock:
            return self.mode

    def set_target(self, position: Tuple[float, float, float]):
        """Accept a new NED target. Resets dispatch flags so the control loop
        will issue a fresh moveToPositionAsync on the next iteration."""
        with self.lock:
            self.target_position = position
            self.is_navigating = True
            self.nav_command_sent = False
            self.idle_hover_sent = False

    def get_target(self) -> Optional[Tuple[float, float, float]]:
        with self.lock:
            return self.target_position

    def clear_target(self):
        with self.lock:
            self.target_position = None
            self.is_navigating = False
            self.nav_command_sent = False
            if self.nav_task is not None:
                try:
                    self.nav_task.cancel()
                except (AttributeError, Exception) as e:
                    print(f"[WARN] Could not cancel nav_task: {e}")
                self.nav_task = None

    def get_nav_snapshot(self) -> Tuple[Optional[Tuple[float, float, float]], bool, bool]:
        """Atomically read target, is_navigating, and nav_command_sent."""
        with self.lock:
            return self.target_position, self.is_navigating, self.nav_command_sent

    def try_mark_nav_dispatched(self, expected_target: Tuple[float, float, float], task) -> bool:
        """Compare-and-swap guard: commits the AirSim future only if the target
        hasn't been replaced by a newer /goto call between reading and
        dispatching. If the target *has* changed, the stale future is cancelled
        immediately so the drone won't fly to an outdated position."""
        with self.lock:
            if self.target_position == expected_target and self.is_navigating:
                self.nav_task = task
                self.nav_command_sent = True
                return True
            else:
                try:
                    task.cancel()
                except (AttributeError, Exception) as e:
                    print(f"[WARN] Could not cancel task: {e}")
                return False

    def get_idle_hover_sent(self) -> bool:
        with self.lock:
            return self.idle_hover_sent

    def mark_idle_hover_sent(self):
        with self.lock:
            self.idle_hover_sent = True

    def request_stop(self):
        with self.lock:
            self.should_stop = True

    def get_should_stop(self) -> bool:
        with self.lock:
            return self.should_stop

    def set_frame(self, frame):
        with self.lock:
            self.current_frame = frame

    def get_frame(self):
        with self.lock:
            return self.current_frame

    def set_home(self, position: Tuple[float, float, float]):
        with self.lock:
            self.home_position = position

    def get_home(self) -> Optional[Tuple[float, float, float]]:
        with self.lock:
            return self.home_position

# Global state instance
drone_state = DroneState()

# ============================================================================
# SAFETY CHECKS
# ============================================================================

def check_safety(target_pos: Tuple[float, float, float]) -> Tuple[bool, str]:
    """
    Verify if navigation to target position is safe.

    Returns:
        (is_safe: bool, reason: str)
    """
    # Check altitude
    if target_pos[2] > 0:  # Remember: NED, positive Z = down
        return False, "Target altitude is below ground"
    
    if abs(target_pos[2]) > MAX_ALTITUDE:
        return False, f"Target altitude exceeds maximum: {abs(target_pos[2])}m > {MAX_ALTITUDE}m"
    
    # Check geofence (distance from origin)
    distance = math.sqrt(target_pos[0]**2 + target_pos[1]**2)
    if distance > GEOFENCE_RADIUS:
        return False, f"Target outside geofence: {distance:.1f}m > {GEOFENCE_RADIUS}m"
    
    return True, "OK"

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(title="Drone Control API")

@app.post("/mode")
async def set_mode(request: ModeRequest):
    """Switch between manual and automatic control modes."""
    mode = request.mode.lower()
    
    if mode not in ["manual", "automatic"]:
        raise HTTPException(status_code=400, detail="Mode must be 'manual' or 'automatic'")
    
    drone_state.set_mode(mode)
    print(f"[API] Mode changed to: {mode}")
    
    return {"status": "success", "mode": mode}

@app.post("/goto")
async def goto_position(request: GotoRequest):
    """
    Command drone to fly to specified NED coordinates (Automatic mode only).
    """
    current_mode = drone_state.get_mode()

    if current_mode != "automatic":
        raise HTTPException(
            status_code=400,
            detail=f"Cannot navigate in {current_mode} mode. Switch to automatic first."
        )

    target_ned = (request.x, request.y, request.z)

    print(f"[API] Goto command received: NED=({request.x}, {request.y}, {request.z})")

    # Set target
    drone_state.set_target(target_ned)

    return {
        "status": "success",
        "message": "Navigation started",
        "target_ned": {
            "x": target_ned[0],
            "y": target_ned[1],
            "z": target_ned[2]
        }
    }

@app.post("/return_home")
async def return_home():
    """Command drone to return to home position."""
    home = drone_state.get_home()
    
    if home is None:
        raise HTTPException(status_code=400, detail="Home position not set")
    
    current_mode = drone_state.get_mode()
    if current_mode != "automatic":
        raise HTTPException(
            status_code=400,
            detail=f"Cannot return home in {current_mode} mode. Switch to automatic first."
        )
    
    print(f"[API] Return to home command: {home}")
    drone_state.set_target(home)
    
    return {"status": "success", "message": "Returning to home", "home_position": home}

@app.get("/status")
async def get_status():
    """Get current drone status."""
    target, is_nav, _ = drone_state.get_nav_snapshot()
    return {
        "mode": drone_state.get_mode(),
        "connected": True,  # Would check actual connection
        "is_navigating": is_nav,
        "target_position": target
    }

def generate_frames():
    """Yields MJPEG frames for the /video_feed endpoint.

    Reads the latest frame from DroneState (written by the control loop)
    and encodes it as JPEG.  The sleep keeps the stream at ~30 FPS and
    avoids busy-waiting when no frame is available yet.
    """
    while True:
        frame = drone_state.get_frame()
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033)

@app.get("/video_feed")
async def video_feed():
    """Stream video feed from drone camera."""
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# ============================================================================
# MAIN DRONE CONTROL LOOP
# ============================================================================

def drone_control_loop():
    """Main control loop — runs on the **main thread**.

    Lifecycle:
        1. Connect to AirSim, arm, and take off.
        2. Enter a ~30 FPS loop that:
           a) Grabs camera frames and publishes them for /video_feed.
           b) In *manual* mode: reads WASD/ZX keys → sends velocity commands.
           c) In *automatic* mode: dispatches a single moveToPositionAsync per
              target, then polls position until within POSITION_TOLERANCE.
        3. On exit (keyboard 'q' or should_stop flag): land, disarm, release
           API control.

    The loop intentionally does **not** block on moveToPositionAsync.join()
    so it can keep streaming camera frames and reacting to mode switches
    while the drone is in transit.
    """

    # --- AirSim initialisation ---
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)   # take control away from manual RC
    client.armDisarm(True)

    print("[DRONE] Taking off...")
    client.takeoffAsync().join()     # blocking: wait until airborne
    client.hoverAsync().join()       # stabilise before entering the loop

    # Record takeoff position so /return_home can navigate back here
    pose = client.simGetVehiclePose()
    home_pos = (pose.position.x_val, pose.position.y_val, pose.position.z_val)
    drone_state.set_home(home_pos)
    print(f"[DRONE] Home position set: {home_pos}")

    # Velocity components for manual mode (reset each iteration)
    vx = vy = vz = 0.0
    last_distance_log = 0.0  # monotonic timestamp; throttles log spam to 1 Hz

    print("[DRONE] Control loop started")
    print("Manual Mode Controls: w/a/s/d = move, z/x = up/down, q = quit")
    
    try:
        while not drone_state.get_should_stop():
            # ----- Camera feed (runs every iteration regardless of mode) -----
            # Request an uncompressed RGB scene image from camera "0"
            response = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])[0]
            if response:
                img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                img = img1d.reshape(response.height, response.width, 3)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                if img is not None:
                    # Publish a clean copy for the /video_feed MJPEG stream
                    drone_state.set_frame(img.copy())

                    # Overlay mode text on a second copy for the local OpenCV window only
                    mode_text = f"Mode: {drone_state.get_mode().upper()}"
                    cv2.putText(img, mode_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow("Drone Camera", img)

            cv2.waitKey(1)  # required by OpenCV to refresh the window
            
            # ----- Mode-specific control -----
            current_mode = drone_state.get_mode()
            
            if current_mode == "automatic":
                # --- Automatic navigation state machine ---
                # Snapshot guarantees we read a consistent (target, navigating,
                # dispatched) triple even if the API thread mutates state between
                # our reads.
                target, is_nav, cmd_sent = drone_state.get_nav_snapshot()

                if target is not None and is_nav:
                    if not cmd_sent:
                        # PHASE 1 — New target received; validate and dispatch.
                        is_safe, reason = check_safety(target)
                        if not is_safe:
                            print(f"[AUTO] Navigation aborted: {reason}")
                            drone_state.clear_target()
                        else:
                            print(f"[AUTO] Navigating to position: {target}")
                            task = client.moveToPositionAsync(
                                target[0], target[1], target[2],
                                velocity=NAVIGATION_SPEED,
                                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0)
                            )
                            # CAS guard: only commit if the target is still the
                            # same one we validated above (see docstring on
                            # try_mark_nav_dispatched).
                            drone_state.try_mark_nav_dispatched(target, task)
                    else:
                        # PHASE 2 — Command dispatched; poll position until arrival.
                        pose = client.simGetVehiclePose()
                        current = (pose.position.x_val, pose.position.y_val, pose.position.z_val)
                        distance = math.sqrt(
                            (current[0] - target[0])**2 +
                            (current[1] - target[1])**2 +
                            (current[2] - target[2])**2
                        )

                        now = time.monotonic()
                        if now - last_distance_log >= 1.0:
                            print(f"[AUTO] Distance to target: {distance:.2f}m")
                            last_distance_log = now

                        if distance < POSITION_TOLERANCE:
                            print("[AUTO] Arrived at target position")
                            drone_state.clear_target()
                            client.hoverAsync().join()
                else:
                    # PHASE 3 — Automatic mode with no target: hover in place.
                    # We only issue hover once to avoid spamming AirSim.
                    if not drone_state.get_idle_hover_sent():
                        client.hoverAsync().join()
                        drone_state.mark_idle_hover_sent()
                    
            else:
                # --- Manual mode: keyboard → velocity commands ---
                # Each key maps to a fixed velocity (m/s) along one NED axis.
                # The short duration (0.1 s) means the drone stops quickly when
                # the key is released, since the next loop iteration will send
                # zero velocity.

                if keyboard.is_pressed('q'):
                    drone_state.request_stop()
                    break

                # North / South (x-axis)
                if keyboard.is_pressed('w'):
                    vx = 3
                elif keyboard.is_pressed('s'):
                    vx = -3
                else:
                    vx = 0

                # East / West (y-axis)
                if keyboard.is_pressed('d'):
                    vy = 3
                elif keyboard.is_pressed('a'):
                    vy = -3
                else:
                    vy = 0

                # Up / Down (z-axis, NED: negative = up)
                if keyboard.is_pressed('z'):
                    vz = -2
                elif keyboard.is_pressed('x'):
                    vz = 2
                else:
                    vz = 0

                client.moveByVelocityAsync(
                    vx, vy, vz,
                    duration=0.1,
                    drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                    yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0)
                )

            # Consistent loop timing for both modes (~30 FPS)
            time.sleep(0.03)

    finally:
        # Graceful shutdown: land → disarm → release API control
        cv2.destroyAllWindows()
        print("[DRONE] Landing...")
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
        print("[DRONE] Shutdown complete")

# ============================================================================
# FASTAPI SERVER THREAD
# ============================================================================

def run_api_server():
    """Run FastAPI server in separate thread."""
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def run_demo_sequence():
    """
    Demo thread that exercises automatic and manual modes via the API.
    Runs after the server and control loop have started.
    """
    import requests

    BASE_URL = "http://localhost:8000"

    def api(method, path, **kwargs):
        resp = getattr(requests, method)(f"{BASE_URL}{path}", **kwargs)
        print(f"[DEMO] {method.upper()} {path} -> {resp.status_code}: {resp.json()}")
        return resp

    # Wait for server + control loop to be ready
    time.sleep(4)

    print("\n" + "="*60)
    print("[DEMO] Starting demo sequence")
    print("="*60)

    # 1. Check initial status (should be manual)
    api("get", "/status")

    # 2. Switch to automatic mode
    print("\n[DEMO] --- Switching to AUTOMATIC mode ---")
    api("post", "/mode", json={"mode": "automatic"})

    # 3. Send a goto command (small offset from origin so it stays in geofence)
    print("\n[DEMO] --- Sending goto command ---")
    api("post", "/goto", json={
        "x": 11.0,   # 11m north
        "y": 8.0,    # 8m east
        "z": -10.0   # 10m above ground (NED)
    })

    # 4. Monitor status while navigating
    for _ in range(10):
        time.sleep(2)
        resp = api("get", "/status")
        if not resp.json().get("is_navigating"):
            print("[DEMO] Navigation complete")
            break

    # 5. Override: switch back to manual mid-flight (or after arrival)
    print("\n[DEMO] --- Overriding to MANUAL mode ---")
    api("post", "/mode", json={"mode": "manual"})
    api("get", "/status")

    # 6. Let manual mode run for a few seconds (keyboard controls active)
    print("\n[DEMO] Manual mode active — use w/a/s/d/z/x to fly, q to quit")
    time.sleep(5)

    # 7. Switch back to automatic and return home
    print("\n[DEMO] --- Switching to AUTOMATIC and returning home ---")
    api("post", "/mode", json={"mode": "automatic"})
    api("post", "/return_home")

    for _ in range(10):
        time.sleep(2)
        resp = api("get", "/status")
        if not resp.json().get("is_navigating"):
            print("[DEMO] Returned home")
            break

    # 8. Back to manual for continued operation
    print("\n[DEMO] --- Back to MANUAL mode ---")
    api("post", "/mode", json={"mode": "manual"})

    print("\n" + "="*60)
    print("[DEMO] Demo sequence complete. Manual control active.")
    print("="*60 + "\n")


if __name__ == "__main__":
    print("="*60)
    print("DRONE CONTROL SYSTEM - Manual & Automatic Modes")
    print("="*60)
    print(f"FastAPI Server: http://localhost:8000")
    print(f"API Docs: http://localhost:8000/docs")
    print("="*60)

    # Thread 1 — REST API (daemon: exits when main thread exits)
    api_thread = threading.Thread(target=run_api_server, daemon=True)
    api_thread.start()

    # Thread 2 — automated demo that hits the API endpoints (daemon)
    demo_thread = threading.Thread(target=run_demo_sequence, daemon=True)
    demo_thread.start()

    # Give uvicorn time to bind port 8000 before the control loop starts
    time.sleep(2)

    # Main thread — the control loop blocks here until 'q' is pressed or
    # should_stop is set; on return the process exits and daemon threads die.
    drone_control_loop()