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
from fastapi.middleware.cors import CORSMiddleware
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

# Navigation parameters
POSITION_TOLERANCE = _cfg["navigation"]["position_tolerance"]
NAVIGATION_SPEED = _cfg["navigation"]["speed"]
RTH_ALTITUDE = _cfg["navigation"]["rth_altitude"]

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
        self.returning_home = False       # True when executing RTH; triggers landing on arrival
        # Camera frames for dual-view streaming
        self.frame_forward = None         # Camera 3: forward-looking view
        self.frame_down = None            # Camera 0: downward-looking view

    def set_mode(self, mode: str):
        """Switch mode. Switching to manual cancels any in-flight navigation."""
        with self.lock:
            self.mode = mode
            self.idle_hover_sent = False
            if mode == "manual":
                self.is_navigating = False
                self.nav_command_sent = False
                self.returning_home = False
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

    def set_frame_forward(self, frame):
        """Set the forward-looking camera frame (camera 3)."""
        with self.lock:
            self.frame_forward = frame

    def get_frame_forward(self):
        """Get the forward-looking camera frame (camera 3)."""
        with self.lock:
            return self.frame_forward

    def set_frame_down(self, frame):
        """Set the downward-looking camera frame (camera 0)."""
        with self.lock:
            self.frame_down = frame

    def get_frame_down(self):
        """Get the downward-looking camera frame (camera 0)."""
        with self.lock:
            return self.frame_down

    def set_home(self, position: Tuple[float, float, float]):
        with self.lock:
            self.home_position = position

    def get_home(self) -> Optional[Tuple[float, float, float]]:
        with self.lock:
            return self.home_position

    def set_returning_home(self, value: bool):
        with self.lock:
            self.returning_home = value

    def get_returning_home(self) -> bool:
        with self.lock:
            return self.returning_home

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
    
    return True, "OK"

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(title="Drone Control API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

    if drone_state.get_returning_home():
        raise HTTPException(
            status_code=409,
            detail="Drone is returning home. New goto commands are blocked until landing completes."
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
    
    # Fly directly to the exact home position; 2D arrival check + landAsync
    # handles the final descent.
    print(f"[API] Return to home command: target={home}")
    drone_state.set_returning_home(True)
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
        "returning_home": drone_state.get_returning_home(),
        "target_position": target
    }

def generate_frames_forward():
    """Yields MJPEG frames for the forward-looking camera (camera 3).

    Reads the latest frame from DroneState (written by the control loop)
    and encodes it as JPEG. The sleep keeps the stream at ~30 FPS.
    """
    while True:
        frame = drone_state.get_frame_forward()
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033)


def generate_frames_down():
    """Yields MJPEG frames for the downward-looking camera (camera 0).

    Reads the latest frame from DroneState (written by the control loop)
    and encodes it as JPEG. The sleep keeps the stream at ~30 FPS.
    """
    while True:
        frame = drone_state.get_frame_down()
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033)


@app.get("/video_feed")
async def video_feed():
    """Stream forward-looking camera (camera 3) - default view."""
    return StreamingResponse(
        generate_frames_forward(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/video_feed/forward")
async def video_feed_forward():
    """Stream forward-looking camera (camera 3)."""
    return StreamingResponse(
        generate_frames_forward(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/video_feed/down")
async def video_feed_down():
    """Stream downward-looking camera (camera 0)."""
    return StreamingResponse(
        generate_frames_down(),
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
            # ----- Camera feeds (runs every iteration regardless of mode) -----
            # Request uncompressed RGB scene images from both cameras:
            #   Camera 0: downward-looking (for ground surveillance)
            #   Camera 3: forward-looking (for navigation/obstacle avoidance)
            responses = client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),  # Down
                airsim.ImageRequest("3", airsim.ImageType.Scene, False, False),  # Forward
            ])

            # Process downward camera (camera 0)
            if responses[0] and len(responses[0].image_data_uint8) > 0:
                img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                img_down = img1d.reshape(responses[0].height, responses[0].width, 3)
                img_down = cv2.cvtColor(img_down, cv2.COLOR_RGB2BGR)
                if img_down is not None and img_down.size > 0:
                    drone_state.set_frame_down(img_down.copy())

            # Process forward camera (camera 3)
            if responses[1] and len(responses[1].image_data_uint8) > 0:
                img1d = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8)
                img_forward = img1d.reshape(responses[1].height, responses[1].width, 3)
                img_forward = cv2.cvtColor(img_forward, cv2.COLOR_RGB2BGR)
                if img_forward is not None and img_forward.size > 0:
                    drone_state.set_frame_forward(img_forward.copy())

                    # Overlay mode text on a copy for the local OpenCV window only
                    mode_text = f"Mode: {drone_state.get_mode().upper()}"
                    cv2.putText(img_forward, mode_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow("Drone Camera", img_forward)

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
                        # Skip safety check for return-to-home so the drone
                        # always flies directly to the exact home coordinates.
                        if drone_state.get_returning_home():
                            print("[AUTO] RTH — safety check bypassed")
                            fly_speed = NAVIGATION_SPEED * 3
                        else:
                            is_safe, reason = check_safety(target)
                            if not is_safe:
                                print(f"[AUTO] Navigation aborted: {reason}")
                                drone_state.clear_target()
                                continue
                            fly_speed = NAVIGATION_SPEED

                        # Re-arm if drone was dropped after a previous RTH
                        client.enableApiControl(True)
                        client.armDisarm(True)

                        print(f"[AUTO] Navigating to ({target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f}) at {fly_speed:.0f} m/s")
                        task = client.moveToPositionAsync(
                            target[0], target[1], target[2],
                            velocity=fly_speed,
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

                        # For RTH use 2D (XY) distance only — the drone just
                        # needs to be above the home spot; landAsync handles
                        # the final descent.  For normal goto use full 3D.
                        if drone_state.get_returning_home():
                            distance = math.sqrt(
                                (current[0] - target[0])**2 +
                                (current[1] - target[1])**2
                            )
                        else:
                            distance = math.sqrt(
                                (current[0] - target[0])**2 +
                                (current[1] - target[1])**2 +
                                (current[2] - target[2])**2
                            )

                        now = time.monotonic()
                        if now - last_distance_log >= 1.0:
                            print(f"[AUTO] Distance to target: {distance:.2f}m")
                            last_distance_log = now

                        rth_tolerance = POSITION_TOLERANCE * 3 if drone_state.get_returning_home() else POSITION_TOLERANCE
                        if distance < rth_tolerance:
                            print("[AUTO] Arrived at target position")
                            drone_state.clear_target()
                            if drone_state.get_returning_home():
                                print("[AUTO] Home reached — dropping")
                                drone_state.set_returning_home(False)
                                client.armDisarm(False)
                                client.enableApiControl(False)
                                # Prevent PHASE 3 from re-engaging
                                drone_state.mark_idle_hover_sent()
                            else:
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

    # Give uvicorn time to bind port 8000 before the control loop starts
    time.sleep(2)

    # Main thread — the control loop blocks here until 'q' is pressed or
    # should_stop is set; on return the process exits and daemon threads die.
    drone_control_loop()