"""
Drone control loop — runs as a background thread.

Lifecycle:
    1. Connect to AirSim, arm, and take off.
    2. Enter a ~30 FPS loop that:
       a) Grabs camera frames from both cameras and publishes them to DroneState.
       b) In *manual* mode: reads WASD/ZX keys and /move API velocity → velocity commands.
       c) In *automatic* mode: dispatches a single moveToPositionAsync per target,
          then polls position until within POSITION_TOLERANCE.
    3. On exit (keyboard 'q' or should_stop flag): land, disarm, release API control.

Architecture note (IOLoop):
    The AirSim client is created INSIDE this function (in the background thread),
    not received from the lifespan.  Pre-refactor: the loop ran on the main thread
    and created its own client — tornado's IOLoop is per-thread and .join() works
    cleanly when created in the same thread.  Reusing a client created inside
    uvicorn's asyncio lifespan causes .join() to hang (IOLoop conflict).
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import cv2
import numpy as np

try:
    import keyboard
    _KEYBOARD_AVAILABLE = True
except Exception:
    _KEYBOARD_AVAILABLE = False

from .drone_state import DroneState, check_safety

if TYPE_CHECKING:
    pass  # airsim imported lazily inside the function


def drone_control_loop(state: DroneState, client, cfg: dict) -> None:
    """Main control loop — runs on a dedicated background thread.

    Parameters
    ----------
    state : DroneState
        Shared thread-safe state singleton.
    client : ignored
        Kept for API compatibility. The loop creates its own AirSim client
        internally to avoid tornado/asyncio IOLoop conflicts.
    cfg : dict
        Parsed config.yaml (keys: safety, navigation).
    """
    import airsim  # noqa: PLC0415

    max_altitude: float = cfg["safety"]["max_altitude"]
    position_tolerance: float = cfg["navigation"]["position_tolerance"]
    navigation_speed: float = cfg["navigation"]["speed"]
    vehicle_name: str = cfg.get("vehicle_name", "")  # "" = AirSim default vehicle

    # --- AirSim connection (created in THIS thread — pre-refactor pattern) ---
    # Retry because AirSim may still be starting up when this thread launches.
    max_retries = 10
    retry_delay = 3.0
    client = None
    for attempt in range(1, max_retries + 1):
        try:
            client = airsim.MultirotorClient()
            client.confirmConnection()
            print(f"[DRONE] AirSim connected (attempt {attempt})")
            break
        except Exception as e:
            print(f"[DRONE] AirSim connection attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(retry_delay)
    if client is None:
        print("[DRONE] Could not connect to AirSim after all retries — control loop exiting")
        return

    vn = vehicle_name  # shorthand for all AirSim calls below
    vn_kw = {"vehicle_name": vn} if vn else {}
    print(f"[DRONE] Controlling vehicle: {vn or '(default)'}")

    # --- Initial arm + takeoff (matches pre-refactor exactly) ---
    client.enableApiControl(True, **vn_kw)
    client.armDisarm(True, **vn_kw)

    print("[DRONE] Taking off...")
    client.takeoffAsync(**vn_kw).join()
    client.hoverAsync(**vn_kw).join()

    pose = client.simGetVehiclePose(**vn_kw)
    home_pos = (pose.position.x_val, pose.position.y_val, pose.position.z_val)
    state.set_home(home_pos)
    state.set_grounded(False)
    print(f"[DRONE] Home position set: {home_pos}")

    last_distance_log: float = 0.0

    print("[DRONE] Control loop started")
    print("Manual Mode Controls: w/a/s/d = move, z/x = up/down, q = quit")

    try:
        while not state.get_should_stop():
            try:
                # --- Camera feeds (every iteration, both modes) ---
                responses = client.simGetImages([
                    airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
                    airsim.ImageRequest("3", airsim.ImageType.Scene, False, False),
                ], **vn_kw)

                if responses[0] and len(responses[0].image_data_uint8) > 0:
                    img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                    img_down = img1d.reshape(responses[0].height, responses[0].width, 3)
                    img_down = cv2.cvtColor(img_down, cv2.COLOR_RGB2BGR)
                    if img_down.size > 0:
                        state.set_frame_down(img_down.copy())

                if responses[1] and len(responses[1].image_data_uint8) > 0:
                    img1d = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8)
                    img_forward = img1d.reshape(responses[1].height, responses[1].width, 3)
                    img_forward = cv2.cvtColor(img_forward, cv2.COLOR_RGB2BGR)
                    if img_forward.size > 0:
                        state.set_frame_forward(img_forward.copy())
                        mode_text = f"Mode: {state.get_mode().upper()}"
                        cv2.putText(img_forward, mode_text, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow("Drone Camera", img_forward)

                cv2.waitKey(1)

                # --- Telemetry ---
                pose = client.simGetVehiclePose(**vn_kw)
                state.set_pose((pose.position.x_val, pose.position.y_val, pose.position.z_val))

                # --- Mode-specific control ---
                current_mode = state.get_mode()

                if current_mode == "automatic":
                    target, is_nav, cmd_sent = state.get_nav_snapshot()

                    if target is not None and is_nav:
                        if not cmd_sent:
                            # PHASE 1 — validate + dispatch (pre-refactor: no hover here)
                            is_rth = state.get_returning_home()
                            if is_rth:
                                fly_speed = navigation_speed * 3
                            else:
                                is_safe, reason = check_safety(target, max_altitude)
                                if not is_safe:
                                    print(f"[AUTO] Navigation aborted: {reason}")
                                    state.clear_target()
                                    time.sleep(0.03)
                                    continue
                                fly_speed = navigation_speed

                            # Re-arm/takeoff if grounded, otherwise just re-assert
                            # API control (old code did this EVERY dispatch — defensive
                            # measure in case another client or AirSim itself released it)
                            if state.is_grounded():
                                print("[AUTO] Drone grounded — re-arming and taking off")
                                client.enableApiControl(True, **vn_kw)
                                client.armDisarm(True, **vn_kw)
                                client.takeoffAsync(**vn_kw).join()
                                state.set_grounded(False)
                            else:
                                client.enableApiControl(True, **vn_kw)
                                client.armDisarm(True, **vn_kw)

                            print(f"[AUTO] Navigating to ({target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f}) "
                                  f"at {fly_speed:.0f} m/s")
                            task = client.moveToPositionAsync(
                                target[0], target[1], target[2],
                                velocity=fly_speed,
                                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0),
                                **vn_kw,
                            )
                            state.try_mark_nav_dispatched(target, task)

                        else:
                            # PHASE 2 — poll until arrival
                            pose = client.simGetVehiclePose(**vn_kw)
                            current = (pose.position.x_val, pose.position.y_val, pose.position.z_val)

                            is_rth = state.get_returning_home()
                            if is_rth:
                                # RTH: only 2D distance (altitude handled by takeoff)
                                distance = math.sqrt(
                                    (current[0] - target[0]) ** 2 +
                                    (current[1] - target[1]) ** 2
                                )
                            else:
                                distance = math.sqrt(
                                    (current[0] - target[0]) ** 2 +
                                    (current[1] - target[1]) ** 2 +
                                    (current[2] - target[2]) ** 2
                                )

                            now = time.monotonic()
                            if now - last_distance_log >= 1.0:
                                print(f"[AUTO] Distance to target: {distance:.2f}m")
                                last_distance_log = now

                            tol = position_tolerance * 3 if is_rth else position_tolerance
                            if distance < tol:
                                print("[AUTO] Arrived at target position")
                                state.clear_target()
                                if is_rth:
                                    # RTH complete — land, disarm, switch to automatic
                                    # so backend detects grounded+automatic → re-enables auto-deploy
                                    state.set_returning_home(False)
                                    client.hoverAsync(**vn_kw).join()
                                    client.landAsync(**vn_kw).join()
                                    client.armDisarm(False, **vn_kw)
                                    client.enableApiControl(False, **vn_kw)
                                    state.set_grounded(True)
                                    state.set_mode("automatic")
                                    state.mark_idle_hover_sent()
                                    print("[AUTO] Drone grounded in automatic mode — ready for next trigger")
                                else:
                                    # Normal arrival — hover and hand over to manual control.
                                    # Matches pre-refactor: operator gets manual control at the
                                    # scene. Auto-deploy is blocked until drone returns home.
                                    print("[AUTO] Switching to manual mode — operator has control")
                                    client.hoverAsync(**vn_kw).join()
                                    state.set_mode("manual")
                    else:
                        # PHASE 3 — no target: hover once then idle
                        if not state.get_idle_hover_sent():
                            client.hoverAsync(**vn_kw).join()
                            state.mark_idle_hover_sent()

                else:
                    # --- Manual mode ---
                    # Re-arm and take off if grounded after RTH (matches pre-refactor)
                    if state.is_grounded():
                        print("[MANUAL] Drone is grounded — re-arming and taking off")
                        client.enableApiControl(True, **vn_kw)
                        client.armDisarm(True, **vn_kw)
                        client.takeoffAsync(**vn_kw).join()
                        client.hoverAsync(**vn_kw).join()
                        state.set_grounded(False)

                    if _KEYBOARD_AVAILABLE and keyboard.is_pressed('q'):
                        state.request_stop()
                        break

                    vx = vy = vz = 0.0
                    if _KEYBOARD_AVAILABLE:
                        if keyboard.is_pressed('w'):
                            vx = 3.0
                        elif keyboard.is_pressed('s'):
                            vx = -3.0
                        if keyboard.is_pressed('d'):
                            vy = 3.0
                        elif keyboard.is_pressed('a'):
                            vy = -3.0
                        if keyboard.is_pressed('z'):
                            vz = -2.0
                        elif keyboard.is_pressed('x'):
                            vz = 2.0

                    api_vx, api_vy, api_vz = state.get_manual_velocity()
                    vx = vx or api_vx
                    vy = vy or api_vy
                    vz = vz or api_vz

                    client.moveByVelocityAsync(
                        vx, vy, vz,
                        duration=0.1,
                        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                        yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0),
                        **vn_kw,
                    )

                # ~30 FPS for both modes (pre-refactor had this at loop bottom)
                time.sleep(0.03)

            except Exception as e:
                print(f"[DRONE] Error: {e.__class__.__name__}: {e}")
                state.set_should_stop(True)
                break

    finally:
        cv2.destroyAllWindows()
        try:
            print("[DRONE] Landing...")
            client.landAsync(**vn_kw).join()
            client.armDisarm(False, **vn_kw)
            client.enableApiControl(False, **vn_kw)
        except Exception as cleanup_err:
            print(f"[DRONE] Cleanup error: {cleanup_err.__class__.__name__}")
        print("[DRONE] Shutdown complete")
