"""AirSim projection backend — uses AirSim client for camera pose queries.

Wraps the existing get_coords_from_lite_mono() function from projection.py.
Falls back to camera position if AirSim client is unavailable.
"""

from __future__ import annotations

from typing import Optional, Tuple

from src.spatial.projection_base import ProjectionBackend


class AirSimProjection(ProjectionBackend):
    """
    Projects pixels to world coordinates using AirSim camera pose.

    Parameters
    ----------
    airsim_client : optional
        AirSim MultirotorClient for querying camera extrinsics.
    camera_name : str
        AirSim camera name (e.g. "0").
    vehicle_name : str
        AirSim vehicle name (e.g. "Drone2").
    cctv_height : float
        Camera height above ground in metres.
    fallback_position : (x, y, z)
        Position to return when AirSim is unavailable.
    safe_z : float
        Override z coordinate (NED altitude for drone deployment).
    """

    def __init__(
        self,
        airsim_client=None,
        camera_name: str = "0",
        vehicle_name: str = "",
        cctv_height: float = 10.0,
        fallback_position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        safe_z: float = -10.0,
    ) -> None:
        self._client = airsim_client
        self._camera_name = camera_name
        self._vehicle_name = vehicle_name
        self._cctv_height = cctv_height
        self._fallback_position = fallback_position
        self._safe_z = safe_z

    def pixel_to_world(
        self,
        pixel_x: float,
        pixel_y: float,
        depth: float,
        frame_w: int,
        frame_h: int,
    ) -> Tuple[float, float, float]:
        if self._client is None:
            x, y, _ = self._fallback_position
            return (x, y, self._safe_z)

        try:
            from src.spatial.projection import get_coords_from_lite_mono

            world_coord = get_coords_from_lite_mono(
                self._client,
                self._camera_name,
                pixel_x,
                pixel_y,
                frame_w,
                frame_h,
                depth,
                self._cctv_height,
                vehicle_name=self._vehicle_name,
            )
            return (world_coord.x_val, world_coord.y_val, self._safe_z)
        except Exception:
            x, y, _ = self._fallback_position
            return (x, y, self._safe_z)

    def update_pose(
        self,
        position: Optional[Tuple[float, float, float]] = None,
        orientation: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        if position is not None:
            self._fallback_position = position

    def set_client(self, client) -> None:
        """Set/replace the AirSim client (e.g. after threaded connection)."""
        self._client = client
