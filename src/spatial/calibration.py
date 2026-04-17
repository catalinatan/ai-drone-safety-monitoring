"""PnP calibration solver — determine camera orientation from known point correspondences.

Users click 4+ points in the camera view and enter their real-world GPS/NED coordinates.
This module solves for the camera's orientation (pitch, yaw, roll) using OpenCV's solvePnP.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np


def solve_camera_orientation(
    pixel_points: List[Tuple[float, float]],
    world_points: List[Tuple[float, float, float]],
    frame_w: int,
    frame_h: int,
    fov: float = 90.0,
    camera_position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Optional[Tuple[float, float, float]]:
    """
    Solve for camera orientation given pixel <-> world point correspondences.

    Parameters
    ----------
    pixel_points : list of (px, py)
        Clicked pixel coordinates in the camera image.
    world_points : list of (x, y, z)
        Corresponding real-world coordinates (NED).
    frame_w, frame_h : int
        Camera image dimensions.
    fov : float
        Horizontal field of view in degrees.
    camera_position : (x, y, z)
        Known camera position in world coordinates.

    Returns
    -------
    (pitch, yaw, roll) in degrees, or None if solving failed.
    pitch: negative = looking down
    yaw: compass heading (0=north, 90=east)
    roll: rotation around forward axis
    """
    if len(pixel_points) != len(world_points):
        raise ValueError("pixel_points and world_points must have the same number of entries")
    if len(pixel_points) < 4:
        raise ValueError("Need at least 4 point correspondences for PnP solve")

    # Camera intrinsics from FOV
    focal_len = (frame_w / 2) / np.tan(np.deg2rad(fov) / 2)
    camera_matrix = np.array(
        [
            [focal_len, 0, frame_w / 2],
            [0, focal_len, frame_h / 2],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros(4)  # assume no lens distortion

    # Convert world points to camera-relative (subtract camera position)
    cam_pos = np.array(camera_position, dtype=np.float64)
    object_points = np.array(
        [np.array(wp) - cam_pos for wp in world_points],
        dtype=np.float64,
    )
    image_points = np.array(pixel_points, dtype=np.float64)

    # Solve PnP
    try:
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error:
        return None

    if not success:
        return None

    # Convert rotation vector to rotation matrix, then to Euler angles
    from scipy.spatial.transform import Rotation as R

    rot_matrix, _ = cv2.Rodrigues(rvec)
    rotation = R.from_matrix(rot_matrix)

    # Extract ZYX Euler angles (yaw, pitch, roll)
    yaw, pitch, roll = rotation.as_euler("ZYX", degrees=True)

    return (float(pitch), float(yaw), float(roll))
