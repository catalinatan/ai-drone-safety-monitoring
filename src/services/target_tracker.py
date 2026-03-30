"""
Target tracker for CCTV follow mode.

Detects and tracks moving targets (e.g., "ship") in video frames,
computing their real-world position for drone following.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class TargetTracker:
    """Tracks a specific target in frames using simple color/motion detection."""

    def __init__(self, target_label: str = "ship"):
        """
        Initialize tracker.

        Parameters
        ----------
        target_label : str
            Target to track ("ship", "railway", "bridge").
        """
        self.target_label = target_label
        self._last_position: Optional[Tuple[float, float]] = None

    def track_in_frame(
        self, frame: np.ndarray, camera_height: float = 10.0
    ) -> Optional[Tuple[float, float, float]]:
        """
        Detect target in frame and estimate 3D position.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame from camera.
        camera_height : float
            Camera altitude above ground (meters).

        Returns
        -------
        (x, y, z) NED world coordinates, or None if target not found.
        """
        if frame is None or frame.size == 0:
            return self._last_position

        # Simple target detection: look for distinctive colors/features
        target_pos = self._detect_target(frame)
        if target_pos is None:
            return self._last_position

        # Convert pixel position to world coordinates
        px, py = target_pos
        h, w = frame.shape[:2]

        # Simple projection: assume target is on ground directly below camera
        # Pixel (px, py) → relative position in frame → world offset from camera nadir
        # Assumes 90° FOV (from AirSim config)
        fov_rad = np.deg2rad(90.0)
        focal_len = (w / 2) / np.tan(fov_rad / 2)
        c_x, c_y = w / 2, h / 2

        # Offset from image center
        offset_x = (px - c_x) / focal_len
        offset_y = (py - c_y) / focal_len

        # Assume target is on ground (z=0 in NED), compute distance
        distance = camera_height / np.sqrt(offset_x**2 + offset_y**2 + 1)

        # World position
        world_x = distance * offset_x
        world_y = distance * offset_y
        world_z = 0.0  # Ground level

        self._last_position = (world_x, world_y, world_z)
        return self._last_position

    def _detect_target(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Detect target in frame.

        Returns (px, py) pixel coordinates or None.
        """
        if self.target_label == "ship":
            return self._detect_ship(frame)
        elif self.target_label == "railway":
            return self._detect_railway(frame)
        elif self.target_label == "bridge":
            return self._detect_bridge(frame)
        return None

    def _detect_ship(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """Detect ship (large moving object) in frame."""
        # Ships are typically large, distinct from water
        # Look for large contours in lower part of frame
        try:
            import cv2

            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Water is typically blue-green (low saturation)
            # Ship is darker/more saturated
            lower = np.array([0, 0, 0])
            upper = np.array([180, 50, 200])
            mask = cv2.inRange(hsv, lower, upper)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None

            # Get largest contour (most likely the ship)
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] == 0:
                return None

            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            return (cx, cy)
        except Exception:
            return None

    def _detect_railway(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """Detect railway/train in frame."""
        # Railways have parallel lines (tracks)
        # Trains are moving objects
        try:
            import cv2

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            # Detect lines
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)
            if lines is None or len(lines) == 0:
                return None

            # Average position of lines
            points = lines.reshape(-1, 2)
            cx = np.mean(points[:, 0])
            cy = np.mean(points[:, 1])
            return (cx, cy)
        except Exception:
            return None

    def _detect_bridge(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """Detect bridge structure in frame."""
        # Bridges have distinctive structure (railings, supports)
        # Look for high contrast vertical/horizontal edges
        try:
            import cv2

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)

            # Center of mass of edges (structural center)
            y_coords, x_coords = np.where(edges > 0)
            if len(x_coords) == 0:
                return None

            cx = np.mean(x_coords)
            cy = np.mean(y_coords)
            return (cx, cy)
        except Exception:
            return None
