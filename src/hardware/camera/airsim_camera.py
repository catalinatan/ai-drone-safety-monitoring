"""AirSim camera backend — captures frames from an AirSim simulation vehicle."""

from __future__ import annotations

import numpy as np

from .base import CameraBackend


class AirSimCamera(CameraBackend):
    """
    Wraps AirSim's simGetImages() as a CameraBackend.

    Parameters
    ----------
    camera_name : str
        AirSim camera name (e.g. "0", "front_center").
    vehicle_name : str
        AirSim vehicle the camera is attached to (e.g. "Drone2").
    """

    def __init__(self, camera_name: str, vehicle_name: str) -> None:
        self.camera_name = camera_name
        self.vehicle_name = vehicle_name
        self._client = None
        self._width: int = 0
        self._height: int = 0

    # ------------------------------------------------------------------
    # CameraBackend interface
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        try:
            import airsim
            import time as time_module

            print(f"[AirSimCamera {self.vehicle_name}] Attempting to connect...")

            # Create client with explicit settings
            self._client = airsim.MultirotorClient()
            self._client.enableApiControl(False)  # Release any existing control first

            print(f"[AirSimCamera {self.vehicle_name}] Client created, confirming connection...")
            # confirmConnection will block until connection is established
            self._client.confirmConnection()

            # Test that we can actually query the vehicle
            try:
                pose = self._client.simGetVehiclePose(vehicle_name=self.vehicle_name)
                print(f"[AirSimCamera {self.vehicle_name}] Verified vehicle exists at position "
                      f"({pose.position.x_val:.1f}, {pose.position.y_val:.1f}, {pose.position.z_val:.1f})")
            except Exception as ve:
                print(f"[AirSimCamera {self.vehicle_name}] WARNING: Vehicle check failed: {ve}")
                # Vehicle doesn't exist, but connection might still work for image capture
                # This can happen if vehicle names in feeds.yaml don't match AirSim setup

            print(f"[AirSimCamera {self.vehicle_name}] Connected successfully")
            return True

        except Exception as e:
            import traceback
            print(f"[AirSimCamera {self.vehicle_name}] Connection failed:")
            print(f"  Error: {e}")
            print(f"  Type: {type(e).__name__}")
            print(f"  Expected vehicle: {self.vehicle_name}")
            print(f"  Camera name: {self.camera_name}")
            print(f"\nFull traceback:")
            traceback.print_exc()
            self._client = None
            return False

    def grab_frame(self) -> np.ndarray | None:
        if self._client is None:
            return None
        try:
            import airsim
            responses = self._client.simGetImages([
                airsim.ImageRequest(self.camera_name, airsim.ImageType.Scene, False, False)
            ], vehicle_name=self.vehicle_name)

            if not responses or len(responses[0].image_data_uint8) == 0:
                return None

            r = responses[0]
            img = np.frombuffer(r.image_data_uint8, dtype=np.uint8).reshape(r.height, r.width, 3)
            self._width = r.width
            self._height = r.height
            return img
        except Exception as e:
            print(f"[AirSimCamera] grab_frame error: {e}")
            return None

    def disconnect(self) -> None:
        self._client = None

    @property
    def is_connected(self) -> bool:
        return self._client is not None

    @property
    def resolution(self) -> tuple[int, int]:
        return (self._width, self._height)

    # ------------------------------------------------------------------
    # AirSim-specific extras
    # ------------------------------------------------------------------

    def get_vehicle_position(self) -> tuple[float, float, float] | None:
        """Returns NED position (x, y, z) of the vehicle, or None on error."""
        if self._client is None:
            return None
        try:
            pose = self._client.simGetVehiclePose(vehicle_name=self.vehicle_name)
            p = pose.position
            if p.x_val != p.x_val:  # NaN guard
                return None
            return (round(p.x_val, 2), round(p.y_val, 2), round(p.z_val, 2))
        except Exception:
            return None

    @property
    def client(self):
        """Expose the underlying AirSim client for operations that need it directly."""
        return self._client
