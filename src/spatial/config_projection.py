"""Config-based projection — uses static or live-updated camera pose.

No AirSim dependency. Camera position comes from feeds.yaml config or
live GPS updates via the API. Orientation comes from config or PnP
calibration. Uses the same ray-ground intersection math as the AirSim
projection but with pose from config instead of simGetCameraInfo().

Position can be provided as GPS (lat, lon, alt) or NED (x, y, z).
GPS coordinates are converted to local NED using a shared origin.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Optional, Tuple

from src.spatial.projection_base import ProjectionBackend
from src.spatial.gps_utils import gps_to_ned


class ConfigProjection(ProjectionBackend):
    """
    Projects pixels to world coordinates using configured camera pose.

    Parameters
    ----------
    position : (x, y, z)
        Camera position in NED coordinates (x=north, y=east, z=down).
    orientation : (pitch, yaw, roll)
        Camera orientation in degrees.
        pitch: negative = looking down (e.g. -30 means 30 deg below horizon)
        yaw: compass heading (0=north, 90=east, 180=south, 270=west)
        roll: rotation around the forward axis (usually 0)
    fov : float
        Horizontal field of view in degrees.
    safe_z : float
        Override z coordinate in output (NED altitude for drone deployment).
    gps_origin : (lat, lon, alt) or None
        GPS reference origin for converting GPS updates to NED.
    """

    def __init__(
        self,
        position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        orientation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        fov: float = 90.0,
        safe_z: float = -10.0,
        gps_origin: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        self._position = np.array(position, dtype=np.float64)
        self._safe_z = safe_z
        self._fov = fov
        self._gps_origin = gps_origin  # (lat, lon, alt)
        self._set_orientation(orientation)

    def _set_orientation(self, orientation: Tuple[float, float, float]) -> None:
        """Compute rotation matrix from (pitch, yaw, roll) in degrees."""
        pitch, yaw, roll = orientation
        self._orientation_deg = orientation
        # Build rotation: yaw around Z (down), pitch around Y (right), roll around X (forward)
        # Convention: NED frame — X=north, Y=east, Z=down
        self._rotation = R.from_euler("ZYX", [yaw, pitch, roll], degrees=True)

    def pixel_to_world(
        self,
        pixel_x: float,
        pixel_y: float,
        depth: float,
        frame_w: int,
        frame_h: int,
    ) -> Tuple[float, float, float]:
        """Project pixel to world coordinates using metric depth.

        Parameters
        ----------
        depth : float
            Metric depth in metres (distance from camera to target along the ray).
            Computed by caller via: scale_factor * inverse_disparity_at_pixel.
            Pass 0.0 to return camera position.
        """
        if depth <= 0:
            return (float(self._position[0]), float(self._position[1]), self._safe_z)

        # Camera intrinsics from FOV
        fov_rad = np.deg2rad(self._fov)
        focal_len = (frame_w / 2) / np.tan(fov_rad / 2)
        c_x, c_y = frame_w / 2, frame_h / 2

        # Ray in camera frame (camera looks along +X, Y=right, Z=down)
        ray_cam = np.array([
            1.0,
            (pixel_x - c_x) / focal_len,
            (pixel_y - c_y) / focal_len,
        ])
        ray_cam /= np.linalg.norm(ray_cam)

        # Rotate to world frame
        ray_world = self._rotation.as_matrix() @ ray_cam

        # Place point at metric depth along the ray
        point_world = self._position + depth * ray_world
        return (float(point_world[0]), float(point_world[1]), self._safe_z)

    def compute_scale_factor(
        self,
        depth_map: np.ndarray,
        frame_w: int,
        frame_h: int,
    ) -> float:
        """Recover metric scale from ground-plane pixels.

        Samples pixels from the bottom strip of the image (likely ground),
        computes geometric ray-ground distance for each, and returns the
        median ratio of geometric_distance / inverse_disparity.
        """
        fov_rad = np.deg2rad(self._fov)
        focal_len = (frame_w / 2) / np.tan(fov_rad / 2)
        c_x, c_y = frame_w / 2, frame_h / 2
        cam_z = self._position[2]
        height = abs(cam_z)

        if height < 0.1:
            return 1.0  # camera at ground level, can't calibrate

        ground_z = cam_z + height  # = 0 when cam_z is negative and height = |cam_z|

        # Sample pixels from bottom 20% of image (most likely ground)
        h, w = depth_map.shape
        y_start = int(h * 0.8)
        sample_rows = range(y_start, h, max(1, (h - y_start) // 5))
        sample_cols = range(0, w, max(1, w // 8))

        ratios = []
        rot_matrix = self._rotation.as_matrix()

        for row in sample_rows:
            for col in sample_cols:
                inv_disp = depth_map[row, col]
                if inv_disp < 1e-6:
                    continue

                # Map (col, row) to pixel coords in frame space
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

                t_ground = (ground_z - cam_z) / ray_world[2]
                if t_ground <= 0:
                    continue

                ratios.append(t_ground / inv_disp)

        if not ratios:
            return 1.0  # fallback

        return float(np.median(ratios))

    def update_pose(
        self,
        position: Optional[Tuple[float, float, float]] = None,
        orientation: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        if position is not None:
            self._position = np.array(position, dtype=np.float64)
        if orientation is not None:
            self._set_orientation(orientation)

    def update_gps_position(
        self,
        lat: float,
        lon: float,
        alt: float,
    ) -> None:
        """Update camera position from GPS coordinates.

        Converts to NED using the stored GPS origin. If no origin is set,
        this GPS position becomes the origin (NED = 0,0,0).
        """
        if self._gps_origin is None:
            self._gps_origin = (lat, lon, alt)
            self._position = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        else:
            ned = gps_to_ned(
                lat, lon, alt,
                self._gps_origin[0], self._gps_origin[1], self._gps_origin[2],
            )
            self._position = np.array(ned, dtype=np.float64)

    def set_from_calibration(
        self,
        rotation: R,
    ) -> None:
        """Set orientation from a calibration-derived rotation."""
        self._rotation = rotation
        # Extract Euler angles for reference
        self._orientation_deg = tuple(rotation.as_euler("ZYX", degrees=True))

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
        fov_rad = np.deg2rad(self._fov)
        focal_len = (frame_w / 2) / np.tan(fov_rad / 2)
        c_x, c_y = frame_w / 2, frame_h / 2

        ray_cam = np.array([
            1.0,
            (pixel_x - c_x) / focal_len,
            (pixel_y - c_y) / focal_len,
        ])
        ray_cam /= np.linalg.norm(ray_cam)

        ray_world = self._rotation.as_matrix() @ ray_cam

        if abs(ray_world[2]) < 0.001:
            return None

        cam_x, cam_y, cam_z = self._position

        if abs(ray_world[0]) > abs(ray_world[1]):
            t = (world_x - cam_x) / ray_world[0]
        else:
            t = (world_y - cam_y) / ray_world[1]

        if t <= 0:
            return None

        # ground_z = cam_z + t * ray_z, and height = ground_z - cam_z
        height = t * ray_world[2]
        if height <= 0:
            return None

        # Store as the effective height for the ground-plane intersection
        # In config projection, ground_z = 0.0 and cam_z represents height
        # So we update cam_z to match: cam_z = -height (NED: negative = above ground)
        # But actually this changes the camera position which is wrong.
        # Instead, just return the height — the caller stores it.
        print(f"[CALIBRATE] Height calibrated to {height:.2f}m")
        return height
