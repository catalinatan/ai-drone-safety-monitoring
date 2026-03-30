"""
Coordinate utilities — pure geometry helpers, no AirSim dependency.

Moved here from src/cctv_monitoring/coord_utils.py during Phase 6 refactor.
AirSim-specific camera projection functions live in src/spatial/projection.py.
"""

from __future__ import annotations

import numpy as np


def get_feet_from_mask(mask: np.ndarray):
    """
    Find the bottom-centre pixel of a binary person mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (0 = background, non-zero = person).

    Returns
    -------
    (cx, cy) tuple of ints, or None if the mask is empty.
    """
    rows, cols = np.where(mask > 0)
    if len(rows) == 0:
        return None

    max_y = int(np.max(rows))
    xs_at_bottom = cols[rows == max_y]
    center_x = int(np.mean(xs_at_bottom))
    return (center_x, max_y)


def unreal_to_airsim(unreal_x: float, unreal_y: float, unreal_z: float) -> dict:
    """
    Convert Unreal Engine coordinates (cm, Z-Up) to AirSim NED (m, Z-Down).

    Parameters
    ----------
    unreal_x, unreal_y, unreal_z : float
        Location values from the Unreal Details Panel (centimetres).

    Returns
    -------
    dict with keys "X", "Y", "Z" in AirSim NED metres.
    """
    x_m = unreal_x / 100.0
    y_m = unreal_y / 100.0
    z_m = unreal_z / 100.0
    return {"X": x_m, "Y": y_m, "Z": -z_m}
