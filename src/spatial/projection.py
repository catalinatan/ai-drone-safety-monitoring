"""
Camera projection utilities — pixel + depth → 3D world coordinates.

Requires an AirSim client to query camera pose (extrinsics).
Moved here from src/cctv_monitoring/coord_utils.py during Phase 6 refactor.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def get_coords_from_ai_depth(
    client,
    camera_name: str,
    pixel_x: float,
    pixel_y: float,
    img_w: int,
    img_h: int,
    ai_depth_val: float,
    ground_scale_factor: float = 1.0,
):
    """
    Convert a pixel + AI depth value into a 3D world coordinate (AirSim NED).

    Parameters
    ----------
    client : airsim.MultirotorClient or similar
        AirSim client used to query camera pose.
    camera_name : str
        Name of the camera in AirSim.
    pixel_x, pixel_y : float
        Pixel coordinates of the target.
    img_w, img_h : int
        Image dimensions.
    ai_depth_val : float
        Raw value from the depth estimation model.
    ground_scale_factor : float
        Multiplier to convert AI depth units to real metres.

    Returns
    -------
    airsim.Vector3r with world-frame (x, y, z) NED coordinates.
    """
    import airsim

    real_dist = ai_depth_val * ground_scale_factor

    fov_rad = np.deg2rad(90.0)
    focal_len = (img_w / 2) / np.tan(fov_rad / 2)
    c_x, c_y = img_w / 2, img_h / 2

    x_local = real_dist
    y_local = real_dist * (pixel_x - c_x) / focal_len
    z_local = real_dist * (pixel_y - c_y) / focal_len
    point_local = np.array([x_local, y_local, z_local])

    info = client.simGetCameraInfo(camera_name)
    cam_pos = info.pose.position
    cam_orient = info.pose.orientation

    r = R.from_quat([cam_orient.x_val, cam_orient.y_val, cam_orient.z_val, cam_orient.w_val])
    point_world_vec = r.as_matrix() @ point_local

    return airsim.Vector3r(
        cam_pos.x_val + point_world_vec[0],
        cam_pos.y_val + point_world_vec[1],
        cam_pos.z_val + point_world_vec[2],
    )


def get_coords_from_lite_mono(
    client,
    camera_name: str,
    pixel_x: float,
    pixel_y: float,
    img_w: int,
    img_h: int,
    ai_depth_val: float,
    cctv_height_meters: float,
    vehicle_name: str = "",
):
    """
    Convert Lite-Mono relative depth (0–1) into world coordinates via ground
    plane intersection.

    Casts a ray from the camera through the pixel, intersects it with the
    ground plane (Z=0 in NED), and uses the relative depth to weight between
    the nearest and ground-intersection distances.

    Parameters
    ----------
    client : airsim client
    camera_name : str
    pixel_x, pixel_y : float
    img_w, img_h : int
    ai_depth_val : float
        Relative depth from Lite-Mono (0 = close, 1 = far).
    cctv_height_meters : float
        Camera height above ground, used as a fallback distance scale.
    vehicle_name : str
        Vehicle the camera is attached to (empty for world-space cameras).

    Returns
    -------
    airsim.Vector3r with world-frame (x, y, z) NED coordinates.
    """
    import airsim

    info = client.simGetCameraInfo(camera_name, vehicle_name=vehicle_name)
    cam_pos = info.pose.position
    cam_orient = info.pose.orientation

    fov_rad = np.deg2rad(90.0)
    focal_len = (img_w / 2) / np.tan(fov_rad / 2)
    c_x, c_y = img_w / 2, img_h / 2

    ray_cam = np.array([1.0, (pixel_x - c_x) / focal_len, (pixel_y - c_y) / focal_len])
    ray_cam /= np.linalg.norm(ray_cam)

    r = R.from_quat([cam_orient.x_val, cam_orient.y_val, cam_orient.z_val, cam_orient.w_val])
    ray_world = r.as_matrix() @ ray_cam

    ground_z = 0.0
    if abs(ray_world[2]) < 0.001:
        distance = ai_depth_val * (cctv_height_meters * 2.5)
    else:
        t_ground = (ground_z - cam_pos.z_val) / ray_world[2]
        if t_ground < 0:
            distance = ai_depth_val * (cctv_height_meters * 2.5)
        else:
            min_distance = max(1.0, cctv_height_meters * 0.5)
            max_distance = t_ground
            distance = min_distance + (max_distance - min_distance) * (ai_depth_val ** 0.5)

    point_world = np.array([
        cam_pos.x_val + distance * ray_world[0],
        cam_pos.y_val + distance * ray_world[1],
        cam_pos.z_val + distance * ray_world[2],
    ])
    point_world[2] = min(point_world[2], 0.0)

    return airsim.Vector3r(point_world[0], point_world[1], point_world[2])
