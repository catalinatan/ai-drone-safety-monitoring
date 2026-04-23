from .camera.base import CameraBackend
from .drone.base import DroneBackend, DronePosition, DroneStatus


def create_camera_backend(config: dict) -> CameraBackend:
    """Instantiate the correct camera backend from a feed config dict."""
    backend_type = config["type"]
    if backend_type == "airsim":
        from .camera.airsim_camera import AirSimCamera

        return AirSimCamera(**config["params"])
    elif backend_type == "rtsp":
        from .camera.rtsp_camera import RTSPCamera

        return RTSPCamera(url=config["params"]["url"])
    elif backend_type == "file":
        from .camera.file_camera import FileCamera

        return FileCamera(path=config["params"]["path"])
    raise ValueError(f"Unknown camera backend: {backend_type!r}")


def create_drone_backend(config: dict) -> DroneBackend:
    """Instantiate the correct drone backend from a config dict."""
    backend_type = config.get("type", "airsim")
    if backend_type == "airsim":
        from .drone.airsim_drone import AirSimDrone

        return AirSimDrone(**config.get("params", {}))
    elif backend_type == "mavlink":
        from .drone.mavlink_drone import MAVLinkDrone

        return MAVLinkDrone(**config.get("params", {}))
    raise ValueError(f"Unknown drone backend: {backend_type!r}")


__all__ = [
    "CameraBackend",
    "DroneBackend",
    "DronePosition",
    "DroneStatus",
    "create_camera_backend",
    "create_drone_backend",
]
