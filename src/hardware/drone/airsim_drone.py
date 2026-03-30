"""AirSim drone backend — wraps AirSim MultirotorClient as a DroneBackend."""

from __future__ import annotations

import numpy as np

from .base import DroneBackend, DronePosition, DroneStatus


class AirSimDrone(DroneBackend):
    """
    Controls an AirSim multirotor vehicle via DroneBackend interface.

    Parameters
    ----------
    vehicle_name : str
        Name of the AirSim vehicle (e.g. "Drone1").
    camera_name : str
        Camera index/name for grab_frame() (e.g. "3" for forward camera).
    speed : float
        Default navigation speed in m/s.
    """

    def __init__(
        self,
        vehicle_name: str = "Drone1",
        camera_name: str = "3",
        speed: float = 5.0,
    ) -> None:
        self.vehicle_name = vehicle_name
        self.camera_name = camera_name
        self.speed = speed
        self._client = None
        self._mode: str = "automatic"
        self._is_navigating: bool = False

    # ------------------------------------------------------------------
    # DroneBackend interface
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        try:
            import airsim
            self._client = airsim.MultirotorClient()
            self._client.confirmConnection()
            self._client.enableApiControl(True, vehicle_name=self.vehicle_name)
            self._client.armDisarm(True, vehicle_name=self.vehicle_name)
            return True
        except Exception as e:
            print(f"[AirSimDrone] connect failed: {e}")
            self._client = None
            return False

    def goto(self, position: DronePosition, speed: float) -> bool:
        if self._client is None:
            return False
        try:
            self._client.moveToPositionAsync(
                position.x, position.y, position.z, speed,
                vehicle_name=self.vehicle_name,
            )
            self._is_navigating = True
            return True
        except Exception as e:
            print(f"[AirSimDrone] goto error: {e}")
            return False

    def get_status(self) -> DroneStatus:
        position = DronePosition(x=0.0, y=0.0, z=0.0)
        if self._client is not None:
            try:
                pose = self._client.simGetVehiclePose(vehicle_name=self.vehicle_name)
                p = pose.position
                position = DronePosition(
                    x=round(p.x_val, 2),
                    y=round(p.y_val, 2),
                    z=round(p.z_val, 2),
                )
            except Exception:
                pass
        return DroneStatus(
            mode=self._mode,
            is_navigating=self._is_navigating,
            position=position,
            is_connected=self._client is not None,
        )

    def set_mode(self, mode: str) -> bool:
        self._mode = mode
        if mode == "manual":
            self._is_navigating = False
        return True

    def return_home(self) -> bool:
        if self._client is None:
            return False
        try:
            self._client.goHomeAsync(vehicle_name=self.vehicle_name)
            return True
        except Exception as e:
            print(f"[AirSimDrone] return_home error: {e}")
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
            return np.frombuffer(r.image_data_uint8, dtype=np.uint8).reshape(r.height, r.width, 3)
        except Exception as e:
            print(f"[AirSimDrone] grab_frame error: {e}")
            return None

    def disconnect(self) -> None:
        if self._client is not None:
            try:
                self._client.enableApiControl(False, vehicle_name=self.vehicle_name)
            except Exception:
                pass
        self._client = None

    @property
    def client(self):
        """Expose the underlying AirSim client for operations that need direct access."""
        return self._client
