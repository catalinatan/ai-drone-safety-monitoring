"""AirSim projection backend — uses AirSim client for camera pose queries.

Wraps the existing get_coords_from_lite_mono() function from projection.py.
Falls back to camera position if AirSim client is unavailable.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

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
        auto_height: bool = False,
    ) -> None:
        self._client = airsim_client
        self._camera_name = camera_name
        self._vehicle_name = vehicle_name
        self._cctv_height = cctv_height
        self._fallback_position = fallback_position
        self._safe_z = safe_z
        self._auto_height = auto_height
        self._height_calibrated = False

    def _try_auto_height(self) -> None:
        """Auto-calibrate cctv_height from ThirdPersonCharacter's Z position.

        Computes the height difference between the camera and the actor's
        ground level, same approach as the eval script.
        """
        if self._height_calibrated or self._client is None:
            return
        try:
            import math
            actor_pose = self._client.simGetObjectPose("ThirdPersonCharacter")
            if math.isnan(actor_pose.position.x_val):
                return
            cam_info = self._client.simGetCameraInfo(
                self._camera_name, vehicle_name=self._vehicle_name,
            )
            height = actor_pose.position.z_val - cam_info.pose.position.z_val
            if height > 0:
                self._cctv_height = float(height)
                print(f"[AirSimProjection] Auto height: {self._cctv_height:.2f}m "
                      f"(actor Z={actor_pose.position.z_val:.2f}, "
                      f"cam Z={cam_info.pose.position.z_val:.2f})")
        except Exception as e:
            print(f"[AirSimProjection] Auto height failed: {e}")
        self._height_calibrated = True

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

        if self._auto_height:
            self._try_auto_height()

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

    def compute_scale_factor(
        self,
        depth_map: "np.ndarray",
        frame_w: int,
        frame_h: int,
    ) -> float:
        """Recover metric scale by comparing ray-ground distances to model predictions.

        Uses camera pose from AirSim and cctv_height to compute geometric
        ground distances at sampled pixels, then returns median ratio.
        """
        if self._client is None:
            return 1.0

        try:
            info = self._client.simGetCameraInfo(
                self._camera_name, vehicle_name=self._vehicle_name,
            )
            cam_pos = info.pose.position
            cam_orient = info.pose.orientation

            height = self._cctv_height
            if height < 0.1:
                return 1.0

            ground_z = cam_pos.z_val + height

            fov_rad = np.deg2rad(90.0)
            focal_len = (frame_w / 2) / np.tan(fov_rad / 2)
            c_x, c_y = frame_w / 2, frame_h / 2

            r = R.from_quat([
                cam_orient.x_val, cam_orient.y_val,
                cam_orient.z_val, cam_orient.w_val,
            ])
            rot_matrix = r.as_matrix()

            # Sample pixels from bottom 20% of image (most likely ground)
            h, w = depth_map.shape
            y_start = int(h * 0.8)
            sample_rows = range(y_start, h, max(1, (h - y_start) // 5))
            sample_cols = range(0, w, max(1, w // 8))

            ratios = []
            for row in sample_rows:
                for col in sample_cols:
                    inv_disp = depth_map[row, col]
                    if inv_disp < 1e-6:
                        continue
                    px = (col / w) * frame_w
                    py = (row / h) * frame_h

                    ray_cam = np.array([
                        1.0,
                        (px - c_x) / focal_len,
                        (py - c_y) / focal_len,
                    ])
                    ray_cam /= np.linalg.norm(ray_cam)
                    ray_world = rot_matrix @ ray_cam

                    if abs(ray_world[2]) < 0.001:
                        continue

                    t_ground = (ground_z - cam_pos.z_val) / ray_world[2]
                    if t_ground <= 0:
                        continue

                    ratios.append(t_ground / inv_disp)

            if not ratios:
                return 1.0

            return float(np.median(ratios))
        except Exception as e:
            print(f"[AirSimProjection] Scale factor failed: {e}")
            return 1.0

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

    def calibrate_height(
        self,
        pixel_x: float,
        pixel_y: float,
        world_x: float,
        world_y: float,
        frame_w: int,
        frame_h: int,
    ) -> Optional[float]:
        """Back-calculate camera height from a known ground point.

        Casts a ray through the pixel and finds the height h such that
        the ray-ground intersection lands at (world_x, world_y).
        """
        if self._client is None:
            return None

        try:
            info = self._client.simGetCameraInfo(
                self._camera_name, vehicle_name=self._vehicle_name,
            )
            cam_pos = info.pose.position
            cam_orient = info.pose.orientation

            fov_rad = np.deg2rad(90.0)
            focal_len = (frame_w / 2) / np.tan(fov_rad / 2)
            c_x, c_y = frame_w / 2, frame_h / 2

            ray_cam = np.array([
                1.0,
                (pixel_x - c_x) / focal_len,
                (pixel_y - c_y) / focal_len,
            ])
            ray_cam /= np.linalg.norm(ray_cam)

            r = R.from_quat([
                cam_orient.x_val, cam_orient.y_val,
                cam_orient.z_val, cam_orient.w_val,
            ])
            ray_world = r.as_matrix() @ ray_cam

            if abs(ray_world[2]) < 0.001:
                return None  # Ray nearly horizontal, can't solve

            # We need: cam + t * ray = (world_x, world_y, ground_z)
            # Solve for t from X: t = (world_x - cam_x) / ray_x
            # Then: ground_z = cam_z + t * ray_z
            # And: cctv_height = ground_z - cam_z = t * ray_z

            # Use whichever axis (X or Y) has the larger ray component for stability
            if abs(ray_world[0]) > abs(ray_world[1]):
                t = (world_x - cam_pos.x_val) / ray_world[0]
            else:
                t = (world_y - cam_pos.y_val) / ray_world[1]

            if t <= 0:
                return None  # Point is behind the camera

            cctv_height = t * ray_world[2]
            if cctv_height <= 0:
                return None  # Ground would be above camera

            self._cctv_height = cctv_height
            print(f"[CALIBRATE] Height calibrated to {cctv_height:.2f}m")
            return cctv_height
        except Exception as e:
            print(f"[CALIBRATE] Failed: {e}")
            return None
