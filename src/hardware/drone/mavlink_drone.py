"""
MAVLink drone backend — for PX4/ArduPilot hardware via pymavlink.

Activate by setting drone.type = "mavlink" in config.
Requires: pip install ".[mavlink]" (pymavlink>=2.4.40)

Connection string formats:
  - UDP: "udp:127.0.0.1:14550"   (SITL)
  - TCP: "tcp:127.0.0.1:5760"    (Hardware via TCP bridge)
  - Serial: "/dev/ttyUSB0"        (Direct serial link, e.g., "/dev/ttyUSB0:921600")
"""

from __future__ import annotations

import numpy as np

from .base import DroneBackend, DronePosition, DroneStatus


class MAVLinkDrone(DroneBackend):
    """MAVLink-compatible drone backend using pymavlink (PX4, ArduPilot)."""

    # Mode name → MAVLink mode number (ArduPilot/PX4 compatible mappings)
    _MODE_MAP = {
        "automatic": 4,    # GUIDED mode
        "manual": 0,       # STABILIZE mode
    }

    def __init__(self, connection_string: str = "udp:127.0.0.1:14550") -> None:
        """
        Initialize MAVLink drone connection.

        Args:
            connection_string: pymavlink connection string
              - "udp:127.0.0.1:14550" for SITL
              - "tcp:127.0.0.1:5760" for hardware TCP bridge
              - "/dev/ttyUSB0" or "/dev/ttyUSB0:921600" for serial
        """
        self.connection_string = connection_string
        self._vehicle = None
        self._system_id = 1
        self._component_id = 1
        self._is_connected = False
        self._is_navigating = False

    def connect(self) -> bool:
        """
        Connect to the MAVLink vehicle.

        Returns True on success, False on timeout or error.
        """
        try:
            from pymavlink import mavutil

            # Establish connection
            self._vehicle = mavutil.mavlink_connection(
                self.connection_string,
                baud=57600,
            )

            # Wait for heartbeat (10 second timeout)
            msg = self._vehicle.wait_heartbeat(timeout=10)
            if msg is None:
                self._vehicle = None
                return False

            self._system_id = self._vehicle.target_system
            self._component_id = self._vehicle.target_component
            self._is_connected = True
            return True

        except ImportError:
            print("[MAVLink] pymavlink not installed. Run: pip install '.[mavlink]'")
            return False
        except Exception as e:
            print(f"[MAVLink] Connection failed: {e}")
            self._vehicle = None
            return False

    def goto(self, position: DronePosition, speed: float) -> bool:
        """
        Command the drone to fly to NED position (non-blocking).

        Args:
            position: DronePosition with x (north), y (east), z (down) in meters
            speed: desired speed (currently unused for waypoint nav)

        Returns True if command was sent successfully.
        """
        if not self._is_connected or self._vehicle is None:
            return False

        try:
            from pymavlink import mavutil as mavutil_module

            # Send waypoint in LOCAL_NED frame
            self._vehicle.mav.mission_item_int_send(
                self._system_id,        # target_system
                self._component_id,     # target_component
                0,                      # seq (mission item index)
                mavutil_module.mavlink.MAV_FRAME_LOCAL_NED,
                mavutil_module.mavlink.MAV_CMD_NAV_WAYPOINT,
                0,                      # current
                1,                      # autocontinue
                0,                      # param1 (hold time) — 0 means default
                0, 0, 0,                # param2-4 (unused)
                int(position.x * 1e4),  # x_int (latitude in 1e-7, but LOCAL_NED uses meters × 1e4)
                int(position.y * 1e4),  # y_int
                position.z,             # z (altitude in meters, negative = above)
            )
            self._is_navigating = True
            return True

        except Exception as e:
            print(f"[MAVLink] goto failed: {e}")
            return False

    def get_status(self) -> DroneStatus:
        """
        Get current drone status.

        Returns DroneStatus with mode, is_navigating, position, is_connected.
        """
        if not self._is_connected or self._vehicle is None:
            return DroneStatus(
                mode="manual",
                is_navigating=False,
                position=DronePosition(0, 0, 0),
                is_connected=False,
            )

        try:
            # Read the latest heartbeat/status
            msg = self._vehicle.recv_match(type="HEARTBEAT", blocking=False, timeout=0.5)
            if msg:
                # Parse flight mode from custom_mode
                mode_num = msg.custom_mode
                mode_name = "manual"
                if mode_num == self._MODE_MAP.get("automatic"):
                    mode_name = "automatic"

                # Read position if available
                pos_msg = self._vehicle.recv_match(
                    type="LOCAL_POSITION_NED",
                    blocking=False,
                    timeout=0.5,
                )
                pos = DronePosition(0, 0, 0)
                if pos_msg:
                    pos = DronePosition(
                        x=pos_msg.x,
                        y=pos_msg.y,
                        z=pos_msg.z,
                    )

                return DroneStatus(
                    mode=mode_name,
                    is_navigating=self._is_navigating,
                    position=pos,
                    is_connected=True,
                )

            # Fallback if no heartbeat received
            return DroneStatus(
                mode="manual",
                is_navigating=self._is_navigating,
                position=DronePosition(0, 0, 0),
                is_connected=True,
            )

        except Exception as e:
            print(f"[MAVLink] get_status failed: {e}")
            return DroneStatus(
                mode="manual",
                is_navigating=False,
                position=DronePosition(0, 0, 0),
                is_connected=False,
            )

    def set_mode(self, mode: str) -> bool:
        """
        Set flight mode.

        Args:
            mode: "automatic" (GUIDED) or "manual" (STABILIZE)

        Returns True on success.
        """
        if not self._is_connected or self._vehicle is None:
            return False

        mode_num = self._MODE_MAP.get(mode)
        if mode_num is None:
            print(f"[MAVLink] Unknown mode: {mode!r}")
            return False

        try:
            self._vehicle.mav.set_mode_send(
                self._system_id,
                1,  # base_mode (MAV_MODE_FLAG_CUSTOM_MODE_ENABLED)
                mode_num,
            )
            return True
        except Exception as e:
            print(f"[MAVLink] set_mode failed: {e}")
            return False

    def return_home(self) -> bool:
        """
        Command return-to-home (land at home position).

        Returns True if command was sent successfully.
        """
        if not self._is_connected or self._vehicle is None:
            return False

        try:
            from pymavlink import mavutil as mavutil_module

            # Send MAV_CMD_NAV_RETURN_TO_LAUNCH
            self._vehicle.mav.command_long_send(
                self._system_id,
                self._component_id,
                mavutil_module.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
                0,  # confirmation
                0, 0, 0, 0, 0, 0, 0,  # params 1-7 (unused for RTH)
            )
            return True
        except Exception as e:
            print(f"[MAVLink] return_home failed: {e}")
            return False

    def grab_frame(self) -> np.ndarray | None:
        """
        Grab a frame from drone's onboard camera.

        MAVLink protocol doesn't natively support video streaming —
        video must be received via separate channel (RTSP, GStreamer, etc.).

        Returns None.
        """
        return None

    def disconnect(self) -> None:
        """Close the connection and release resources."""
        if self._vehicle is not None:
            try:
                self._vehicle.close()
            except Exception:
                pass
            finally:
                self._vehicle = None
        self._is_connected = False
        self._is_navigating = False
