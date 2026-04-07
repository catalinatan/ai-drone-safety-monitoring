"""Abstract base class for camera projection backends.

Mirrors the CameraBackend pattern — swap implementations by config,
no detection code changes needed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple


class ProjectionBackend(ABC):
    """Convert pixel + depth into real-world 3D coordinates."""

    @abstractmethod
    def pixel_to_world(
        self,
        pixel_x: float,
        pixel_y: float,
        depth: float,
        frame_w: int,
        frame_h: int,
    ) -> Tuple[float, float, float]:
        """
        Project a pixel coordinate into world-frame (x, y, z).

        Parameters
        ----------
        pixel_x, pixel_y : float
            Pixel coordinates in the camera image.
        depth : float
            Depth estimate (interpretation depends on implementation).
        frame_w, frame_h : int
            Image dimensions for computing camera intrinsics.

        Returns
        -------
        (x, y, z) tuple in world coordinates.
        """
        ...

    @abstractmethod
    def update_pose(
        self,
        position: Optional[Tuple[float, float, float]] = None,
        orientation: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """
        Update camera pose at runtime (e.g. from live GPS feed).

        Parameters
        ----------
        position : (x, y, z) or (lat, lon, alt) depending on backend
        orientation : (pitch, yaw, roll) in degrees
        """
        ...

    def calibrate_height(
        self,
        pixel_x: float,
        pixel_y: float,
        world_x: float,
        world_y: float,
        frame_w: int,
        frame_h: int,
    ) -> Optional[float]:
        """
        Calibrate camera height above ground from a known point correspondence.

        The user clicks a point on the ground in the image and provides its
        known world X/Y (NED). The method back-calculates what cctv_height
        makes the ray through that pixel land at the given world position.

        Parameters
        ----------
        pixel_x, pixel_y : float
            Pixel coordinates of a known ground point.
        world_x, world_y : float
            Known NED X/Y of that point on the ground surface.
        frame_w, frame_h : int
            Image dimensions.

        Returns
        -------
        Calibrated height in metres, or None if calibration failed.
        """
        return None
