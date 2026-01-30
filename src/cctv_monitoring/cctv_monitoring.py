"""
CCTV Security Monitoring System with Drone Deployment
======================================================

This script monitors a CCTV feed from AirSim, detects humans entering
manually-defined danger zones, and automatically deploys a drone to the
person's location.

Features:
- Manual danger zone annotation at startup
- Real-time human detection using YOLO
- Depth estimation for 3D coordinate calculation
- Automatic drone deployment via REST API
- Visual feedback with alarm overlays

Usage:
    1. Start AirSim simulation (ship/bridge/railway environment)
    2. Start drone control API: python src/drone-control/drone.py
    3. Run this script: python src/cctv_monitoring/cctv_monitoring.py
    4. Draw danger zone polygon when prompted
    5. System will monitor and deploy drone when person enters zone
"""

import airsim
import time
import os
import numpy as np
import cv2
import sys
import requests
from datetime import datetime

# --- PATH SETUP ---
current_file_path = os.path.abspath(__file__)
cctv_monitoring_dir = os.path.dirname(current_file_path)
src_dir = os.path.dirname(cctv_monitoring_dir)

# Add paths for imports
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if cctv_monitoring_dir not in sys.path:
    sys.path.insert(0, cctv_monitoring_dir)

# --- IMPORTS ---
import coord_utils
import depth_estimation_utils
from human_detection.detector import HumanDetector
from human_detection.check_overlap import check_danger_zone_overlap
import human_detection.config as config

# --- CONFIGURATION ---
# AirSim camera and drone names
CCTV_NAME = "0"                 # Camera 1 - spectator camera (separate from drone)
CCTV_VEHICLE = ""               # Empty = not attached to any vehicle
DRONE_NAME = "Drone1"           # Drone vehicle name (for deployment)

# Drone deployment settings
SAFE_Z_ALTITUDE = -10.0  # meters above ground (NED: negative = up)
NAVIGATION_SPEED = 5.0   # m/s

# Camera height (adjust based on your environment)
CCTV_HEIGHT = 15.0  # meters above ground/water

# Drone API settings
DRONE_API_URL = "http://localhost:8000"
DRONE_API_TIMEOUT = 5  # seconds

# Model paths
LITE_MONO_DIR = os.path.join(cctv_monitoring_dir, "lite_mono_weights", "lite-mono-small_640x192")
ENCODER_PATH = os.path.join(LITE_MONO_DIR, "encoder.pth")
DECODER_PATH = os.path.join(LITE_MONO_DIR, "depth.pth")

# Monitoring settings
ALARM_COOLDOWN = 5.0  # seconds between drone deployments
DISPLAY_FEED = True   # Show camera feed window


class DroneAPIClient:
    """Client for communicating with the drone control REST API."""

    def __init__(self, base_url=DRONE_API_URL, timeout=DRONE_API_TIMEOUT):
        self.base_url = base_url
        self.timeout = timeout

    def check_connection(self):
        """Check if drone API is running."""
        try:
            response = requests.get(f"{self.base_url}/status", timeout=self.timeout)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def set_mode(self, mode):
        """Set drone control mode (manual/automatic)."""
        try:
            response = requests.post(
                f"{self.base_url}/mode",
                json={"mode": mode},
                timeout=self.timeout
            )
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Failed to set mode: {e}")
            return False

    def goto_position(self, x, y, z):
        """Send drone to target position (NED coordinates)."""
        try:
            response = requests.post(
                f"{self.base_url}/goto",
                json={"x": float(x), "y": float(y), "z": float(z)},
                timeout=self.timeout
            )
            if response.status_code == 200:
                return True
            else:
                print(f"[ERROR] Goto failed: {response.json()}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Failed to send goto command: {e}")
            return False

    def get_status(self):
        """Get current drone status."""
        try:
            response = requests.get(f"{self.base_url}/status", timeout=self.timeout)
            if response.status_code == 200:
                return response.json()
            return None
        except requests.exceptions.RequestException:
            return None


def log_event(message, level="INFO"):
    """Log events with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


def draw_alarm_overlay(frame, danger_masks, alarm_active):
    """Draw visual alarm indicators on the frame."""
    display_frame = frame.copy()

    if alarm_active:
        # Draw red border
        h, w = frame.shape[:2]
        cv2.rectangle(display_frame, (0, 0), (w, h), (0, 0, 255), 20)

        # Draw ALARM text
        cv2.putText(
            display_frame, "ALARM ACTIVE", (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4
        )

        # Draw bounding boxes around people in danger
        for mask in danger_masks:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                x, y, w_box, h_box = cv2.boundingRect(c)

                # Draw box
                cv2.rectangle(display_frame, (x, y), (x + w_box, y + h_box), (0, 0, 255), 3)

                # Draw warning symbol
                cv2.rectangle(display_frame, (x, y - 40), (x + 40, y), (0, 0, 255), -1)
                cv2.putText(display_frame, "!", (x + 10, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

                # Draw label
                cv2.putText(display_frame, "DANGER", (x + 50, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return display_frame


def run_security_system():
    """Main security monitoring loop."""

    log_event("Initializing CCTV Security Monitoring System", "SYSTEM")

    # --- 1. CHECK DRONE API ---
    log_event("Checking drone API connection...", "INIT")
    drone_api = DroneAPIClient()

    if not drone_api.check_connection():
        log_event(
            f"Cannot connect to drone API at {DRONE_API_URL}. "
            "Please start the drone control server first:\n"
            "    python src/drone-control/drone.py",
            "ERROR"
        )
        return

    log_event("Drone API connected successfully", "INIT")

    # Set drone to automatic mode
    log_event("Setting drone to automatic mode...", "INIT")
    if not drone_api.set_mode("automatic"):
        log_event("Failed to set drone to automatic mode", "ERROR")
        return

    # --- 2. CONNECT TO AIRSIM ---
    log_event("Connecting to AirSim...", "INIT")
    try:
        client = airsim.MultirotorClient()
        client.confirmConnection()
        log_event("AirSim connected successfully", "INIT")

        # Verify the CCTV camera exists
        if CCTV_VEHICLE:
            log_event(f"Verifying CCTV camera '{CCTV_NAME}' on vehicle '{CCTV_VEHICLE}'...", "INIT")
        else:
            log_event(f"Verifying external CCTV camera '{CCTV_NAME}'...", "INIT")

        test_response = client.simGetImages([
            airsim.ImageRequest(CCTV_NAME, airsim.ImageType.Scene, False, False)
        ], vehicle_name=CCTV_VEHICLE)

        if not test_response:
            if CCTV_VEHICLE:
                log_event(
                    f"CCTV camera '{CCTV_NAME}' not found on vehicle '{CCTV_VEHICLE}'.\n"
                    f"Please check your AirSim settings.json.",
                    "ERROR"
                )
            else:
                log_event(
                    f"External CCTV camera '{CCTV_NAME}' not found.\n"
                    f"Please check:\n"
                    f"  1. Camera '{CCTV_NAME}' exists in settings.json under 'Cameras' or 'ExternalCameras'\n"
                    f"  2. Or set CCTV_VEHICLE to the vehicle name if camera is attached to a vehicle",
                    "ERROR"
                )
            return

        log_event(f"CCTV camera '{CCTV_NAME}' verified successfully", "INIT")

    except Exception as e:
        log_event(f"Failed to connect to AirSim: {e}", "ERROR")
        log_event(
            "Please ensure:\n"
            "  1. AirSim (Unreal Engine) is running\n"
            "  2. The simulation is loaded and running\n"
            "  3. AirSim API is enabled",
            "ERROR"
        )
        return

    # --- 3. LOAD MODELS ---
    log_event("Loading AI models...", "INIT")

    # Load YOLO detector
    try:
        human_detector = HumanDetector()
        log_event(f"YOLO model loaded: {config.MODEL_PATH}", "INIT")
    except Exception as e:
        log_event(f"Failed to load YOLO model: {e}", "ERROR")
        return

    # Load depth estimation model
    log_event("Loading Lite-Mono depth model...", "INIT")
    if not os.path.exists(ENCODER_PATH):
        log_event(f"Encoder not found at: {ENCODER_PATH}", "ERROR")
        log_event("Please ensure the model weights are in the correct location", "ERROR")
        return

    if not os.path.exists(DECODER_PATH):
        log_event(f"Decoder not found at: {DECODER_PATH}", "ERROR")
        log_event("Please ensure the model weights are in the correct location", "ERROR")
        return

    try:
        depth_model = depth_estimation_utils.load_lite_mono_model(ENCODER_PATH, DECODER_PATH)
        log_event("Depth model loaded successfully", "INIT")
    except Exception as e:
        log_event(f"Failed to load depth model: {e}", "ERROR")
        return

    # --- 4. MANUAL DANGER ZONE ANNOTATION ---
    log_event("Starting danger zone annotation...", "SETUP")
    log_event("Instructions:", "SETUP")
    log_event("  1. Click points to define the danger zone polygon", "SETUP")
    log_event("  2. Press SPACE or ENTER to finish", "SETUP")
    log_event("  3. Press 'c' to clear and restart", "SETUP")

    try:
        danger_zone_mask = coord_utils.get_user_defined_danger_zone(client, CCTV_NAME, CCTV_VEHICLE)
    except Exception as e:
        log_event(f"Failed during danger zone annotation: {e}", "ERROR")
        return

    if danger_zone_mask is None or np.sum(danger_zone_mask) == 0:
        log_event("No danger zone defined. Exiting.", "ERROR")
        return

    log_event(f"Danger zone defined: {np.sum(danger_zone_mask)} pixels", "SETUP")

    # --- 5. START MONITORING ---
    log_event("="*60, "SYSTEM")
    log_event("SECURITY SYSTEM ARMED - MONITORING ACTIVE", "SYSTEM")
    log_event("="*60, "SYSTEM")
    log_event(f"Camera: {CCTV_NAME}", "SYSTEM")
    log_event(f"Overlap threshold: {config.DANGER_ZONE_OVERLAP_THRESHOLD * 100}%", "SYSTEM")
    log_event(f"Safe altitude: {SAFE_Z_ALTITUDE}m", "SYSTEM")
    log_event("Press 'q' to quit", "SYSTEM")

    # Monitoring state
    alarm_active = False
    last_deployment_time = 0
    frame_count = 0
    drone_is_navigating = False  # Track if drone is currently on a mission
    target_position = None       # Store the target position
    drone_armed = False          # Track if drone is armed and ready

    try:
        while True:
            frame_count += 1

            # --- CAPTURE FRAME FROM CCTV ---
            try:
                responses = client.simGetImages([
                    airsim.ImageRequest(CCTV_NAME, airsim.ImageType.Scene, False, False)
                ], vehicle_name=CCTV_VEHICLE)

                if not responses or len(responses) == 0:
                    log_event("No image received from CCTV camera", "WARNING")
                    time.sleep(0.1)
                    continue

                response = responses[0]
                img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                img_rgb = img1d.reshape(response.height, response.width, 3)

            except Exception as e:
                log_event(f"Error capturing frame: {e}", "ERROR")
                time.sleep(0.1)
                continue

            # --- DETECT HUMANS ---
            try:
                person_masks = human_detector.get_masks(img_rgb)
            except Exception as e:
                log_event(f"Error in human detection: {e}", "ERROR")
                continue

            # --- CHECK OVERLAP ---
            try:
                is_alarm, danger_masks = check_danger_zone_overlap(person_masks, danger_zone_mask)
            except Exception as e:
                log_event(f"Error in overlap checking: {e}", "ERROR")
                continue

            # --- HANDLE ALARM ---
            if is_alarm:
                current_time = time.time()

                if not alarm_active:
                    log_event(f"ALARM TRIGGERED! {len(danger_masks)} person(s) in danger zone", "ALARM")
                    alarm_active = True

                # Deploy drone if cooldown has passed AND drone is not already on a mission
                if current_time - last_deployment_time > ALARM_COOLDOWN and not drone_is_navigating:
                    log_event("Initiating drone deployment...", "DEPLOY")

                    try:
                        # --- ARM AND TAKEOFF (if first deployment) ---
                        if not drone_armed:
                            log_event("Arming drone...", "DEPLOY")
                            try:
                                client.armDisarm(True, vehicle_name=DRONE_NAME)
                                log_event("Taking off...", "DEPLOY")
                                client.takeoffAsync(vehicle_name=DRONE_NAME).join()
                                drone_armed = True
                                log_event("Drone is now airborne and ready", "SUCCESS")
                                time.sleep(1)  # Brief pause after takeoff
                            except Exception as e:
                                log_event(f"Failed to arm/takeoff drone: {e}", "ERROR")
                                continue

                        # --- RUN DEPTH INFERENCE ---
                        log_event("Running depth estimation...", "DEPLOY")
                        ai_depth_map = depth_estimation_utils.run_lite_mono_inference(depth_model, img_rgb)

                        # --- GET TARGET COORDINATES ---
                        # Target the first person in danger
                        target_mask = danger_masks[0]
                        feet_pixel = coord_utils.get_feet_from_mask(target_mask)

                        if feet_pixel is None:
                            log_event("Could not find feet position in mask", "WARNING")
                            continue

                        cx, cy = feet_pixel
                        log_event(f"Target pixel: ({cx}, {cy})", "DEPLOY")

                        # Get depth value at feet position
                        rel_depth = ai_depth_map[cy, cx]
                        log_event(f"Relative depth: {rel_depth:.3f}", "DEPLOY")

                        # Convert to world coordinates using ground plane intersection
                        target_pos = coord_utils.get_coords_from_lite_mono(
                            client, CCTV_NAME, cx, cy,
                            response.width, response.height,
                            rel_depth, CCTV_HEIGHT, CCTV_VEHICLE
                        )

                        log_event(
                            f"Target world coordinates: "
                            f"X={target_pos.x_val:.2f}, Y={target_pos.y_val:.2f}, Z={target_pos.z_val:.2f}",
                            "DEPLOY"
                        )

                        # --- SEND DRONE COMMAND ---
                        log_event(
                            f"Sending drone to: ({target_pos.x_val:.2f}, {target_pos.y_val:.2f}, {SAFE_Z_ALTITUDE})",
                            "DEPLOY"
                        )

                        success = drone_api.goto_position(
                            target_pos.x_val,
                            target_pos.y_val,
                            SAFE_Z_ALTITUDE
                        )

                        if success:
                            log_event("Drone deployment command sent successfully", "SUCCESS")
                            last_deployment_time = current_time
                            drone_is_navigating = True
                            target_position = (target_pos.x_val, target_pos.y_val, SAFE_Z_ALTITUDE)
                            log_event(f"Drone locked onto target. Will complete mission even if person leaves zone.", "INFO")
                        else:
                            log_event("Failed to send drone command", "ERROR")

                    except Exception as e:
                        log_event(f"Error during deployment: {e}", "ERROR")
                        import traceback
                        traceback.print_exc()

            else:
                # No alarm - all clear (but drone might still be navigating)
                if alarm_active and not drone_is_navigating:
                    log_event("Threat cleared - zone safe", "INFO")
                    alarm_active = False
                elif alarm_active and drone_is_navigating:
                    log_event("Threat cleared but drone still completing mission...", "INFO")

            # --- CHECK DRONE STATUS ---
            # Check if drone has reached target
            if drone_is_navigating and target_position:
                try:
                    status = drone_api.get_status()
                    if status and not status.get("is_navigating", True):
                        # Drone has reached the target or stopped navigating
                        log_event(f"Drone reached target position {target_position}", "SUCCESS")
                        drone_is_navigating = False
                        target_position = None
                        alarm_active = False  # Clear alarm after mission complete
                except Exception as e:
                    log_event(f"Error checking drone status: {e}", "WARNING")

            # --- DISPLAY FEED ---
            if DISPLAY_FEED:
                try:
                    # Draw alarm overlay
                    display_frame = draw_alarm_overlay(
                        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
                        danger_masks,
                        alarm_active
                    )

                    # Draw danger zone overlay (faint red tint)
                    red_overlay = np.zeros_like(display_frame)
                    red_overlay[:, :] = (0, 0, 255)
                    display_frame = np.where(
                        danger_zone_mask[..., None] == 1,
                        cv2.addWeighted(display_frame, 0.7, red_overlay, 0.3, 0),
                        display_frame
                    )

                    # Add camera label at top
                    camera_label = f"CCTV CAMERA: {CCTV_NAME} (External View)"
                    cv2.putText(
                        display_frame, camera_label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
                    )

                    # Add status text at bottom
                    status_text = f"Frame: {frame_count} | People: {len(person_masks)} | Alarm: {'ACTIVE' if alarm_active else 'CLEAR'}"
                    cv2.putText(
                        display_frame, status_text, (10, display_frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                    )

                    # Add mission status if drone is navigating
                    if drone_is_navigating:
                        mission_text = f"DRONE MISSION: Active -> {target_position}"
                        cv2.putText(
                            display_frame, mission_text, (10, display_frame.shape[0] - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                        )

                    cv2.imshow("CCTV Security Monitor - External Camera View", display_frame)

                    # Check for quit key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        log_event("User requested shutdown", "SYSTEM")
                        break

                except Exception as e:
                    log_event(f"Error displaying feed: {e}", "WARNING")

            # Small delay to prevent CPU overload
            time.sleep(0.03)  # ~30 FPS

    except KeyboardInterrupt:
        log_event("Keyboard interrupt received", "SYSTEM")

    except Exception as e:
        log_event(f"Unexpected error in monitoring loop: {e}", "ERROR")
        import traceback
        traceback.print_exc()

    finally:
        # --- CLEANUP ---
        log_event("Shutting down security system...", "SYSTEM")
        if DISPLAY_FEED:
            cv2.destroyAllWindows()
        log_event("Security system shutdown complete", "SYSTEM")


if __name__ == "__main__":
    run_security_system()
