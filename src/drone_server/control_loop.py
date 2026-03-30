"""
Drone control loop — runs as a background thread.

Lifecycle:
    1. Connect to AirSim via the injected client, arm, and take off.
    2. Enter a ~30 FPS loop that:
       a) Grabs camera frames from both cameras and publishes them to DroneState.
       b) In *manual* mode: reads WASD/ZX keys and /move API velocity → velocity commands.
       c) In *automatic* mode: dispatches a single moveToPositionAsync per target,
          then polls position until within POSITION_TOLERANCE.
    3. On exit (keyboard 'q' or should_stop flag): land, disarm, release API control.

Dependency injection:
    ``drone_control_loop`` receives a live ``airsim.MultirotorClient`` and the
    shared ``DroneState`` singleton.  The caller (``app.py`` lifespan) is
    responsible for establishing the AirSim connection before calling this
    function, enabling the control loop to be tested with a mock client.
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
    """Main control loop — intended to run on a dedicated background thread.

    Parameters
    ----------
    state:
        The shared ``DroneState`` singleton.
    client:
        A connected ``airsim.MultirotorClient`` instance.
    cfg:
        Parsed ``config.yaml`` dict (keys: safety, navigation).
    """
    import airsim  # noqa: PLC0415 (import inside function — airsim is optional)

    max_altitude: float = cfg["safety"]["max_altitude"]
    position_tolerance: float = cfg["navigation"]["position_tolerance"]
    navigation_speed: float = cfg["navigation"]["speed"]

    # --- Initial takeoff ---
    client.enableApiControl(True)
    client.armDisarm(True)

    print("[DRONE] Taking off...")
    client.takeoffAsync().join()
    client.hoverAsync().join()

    # Record takeoff position so /return_home can navigate back here
    pose = client.simGetVehiclePose()
    home_pos = (pose.position.x_val, pose.position.y_val, pose.position.z_val)
    state.set_home(home_pos)
    state.set_grounded(False)
    print(f"[DRONE] Home position set: {home_pos}")

    last_distance_log: float = 0.0  # monotonic timestamp; throttles log spam to 1 Hz

    print("[DRONE] Control loop started")
    print("Manual Mode Controls: w/a/s/d = move, z/x = up/down, q = quit")

    try:
        while not state.get_should_stop():
            # ----- Camera feeds (runs every iteration regardless of mode) -----
            responses = client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),  # Down
                airsim.ImageRequest("3", airsim.ImageType.Scene, False, False),  # Forward
            ])

            # Downward camera (camera 0)
            if responses[0] and len(responses[0].image_data_uint8) > 0:
                img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                img_down = img1d.reshape(responses[0].height, responses[0].width, 3)
                img_down = cv2.cvtColor(img_down, cv2.COLOR_RGB2BGR)
                if img_down.size > 0:
                    state.set_frame_down(img_down.copy())

            # Forward camera (camera 3)
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

            # ----- Update telemetry -----
            pose = client.simGetVehiclePose()
            state.set_pose((pose.position.x_val, pose.position.y_val, pose.position.z_val))

            # ----- Mode-specific control -----
            current_mode = state.get_mode()

            if current_mode == "automatic":
                target, is_nav, cmd_sent = state.get_nav_snapshot()

                if target is not None and is_nav:
                    if not cmd_sent:
                        # PHASE 1 — New target; validate and dispatch.
                        if state.get_returning_home():
                            print("[AUTO] RTH — safety check bypassed")
                            fly_speed = navigation_speed * 3
                        else:
                            is_safe, reason = check_safety(target, max_altitude)
                            if not is_safe:
                                print(f"[AUTO] Navigation aborted: {reason}")
                                state.clear_target()
                                continue
                            fly_speed = navigation_speed

                        # Re-arm and take off if drone was grounded after RTH
                        if state.is_grounded():
                            print("[AUTO] Drone is grounded — re-arming and taking off")
                            client.enableApiControl(True)
                            client.armDisarm(True)
                            client.takeoffAsync().join()
                            state.set_grounded(False)
                        else:
                            client.enableApiControl(True)
                            client.armDisarm(True)

                        print(f"[AUTO] Navigating to ({target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f}) "
                              f"at {fly_speed:.0f} m/s")
                        task = client.moveToPositionAsync(
                            target[0], target[1], target[2],
                            velocity=fly_speed,
                            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                            yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0),
                        )
                        state.try_mark_nav_dispatched(target, task)

                    else:
                        # PHASE 2 — Command dispatched; poll position until arrival.
                        pose = client.simGetVehiclePose()
                        current = (pose.position.x_val, pose.position.y_val, pose.position.z_val)

                        if state.get_returning_home():
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

                        rth_tolerance = position_tolerance * 3 if state.get_returning_home() else position_tolerance
                        if distance < rth_tolerance:
                            print("[AUTO] Arrived at target position")
                            is_rth = state.get_returning_home()
                            state.clear_target()
                            if is_rth:
                                print("[AUTO] Home reached — landing and disarming")
                                state.set_returning_home(False)
                                client.hoverAsync().join()
                                client.landAsync().join()
                                client.armDisarm(False)
                                client.enableApiControl(False)
                                state.set_grounded(True)
                                state.set_mode("automatic")
                                state.mark_idle_hover_sent()
                                print("[AUTO] Drone grounded — ready for next trigger")
                            else:
                                print("[AUTO] Switching to manual mode — user has control")
                                client.hoverAsync().join()
                                state.set_mode("manual")
                else:
                    # PHASE 3 — Automatic mode with no target: hover in place (once).
                    if not state.get_idle_hover_sent():
                        client.hoverAsync().join()
                        state.mark_idle_hover_sent()

            else:
                # --- Manual mode: keyboard + API velocity → velocity commands ---
                if state.is_grounded():
                    print("[MANUAL] Drone is grounded — re-arming and taking off")
                    client.enableApiControl(True)
                    client.armDisarm(True)
                    client.takeoffAsync().join()
                    client.hoverAsync().join()
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

                # Merge with API velocity (from /move endpoint)
                api_vx, api_vy, api_vz = state.get_manual_velocity()
                vx = vx or api_vx
                vy = vy or api_vy
                vz = vz or api_vz

                client.moveByVelocityAsync(
                    vx, vy, vz,
                    duration=0.1,
                    drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                    yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0),
                )

            # Consistent loop timing (~30 FPS)
            time.sleep(0.03)

    finally:
        cv2.destroyAllWindows()
        print("[DRONE] Landing...")
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
        print("[DRONE] Shutdown complete")
