"""GPS <-> NED coordinate conversion utilities.

Converts between GPS coordinates (latitude, longitude, altitude) and
local NED (North-East-Down) coordinates relative to a reference origin.

The conversion uses a flat-Earth approximation which is accurate to
within ~1% for distances up to ~10 km from the origin.
"""

from __future__ import annotations

import math
from typing import Tuple

# Metres per degree of latitude (constant at any latitude)
_METRES_PER_DEG_LAT = 111_320.0


def _metres_per_deg_lon(lat_deg: float) -> float:
    """Metres per degree of longitude at a given latitude."""
    return _METRES_PER_DEG_LAT * math.cos(math.radians(lat_deg))


def gps_to_ned(
    lat: float,
    lon: float,
    alt: float,
    origin_lat: float,
    origin_lon: float,
    origin_alt: float = 0.0,
) -> Tuple[float, float, float]:
    """
    Convert GPS (lat, lon, alt) to local NED (x, y, z) relative to an origin.

    Parameters
    ----------
    lat, lon : float
        Position in decimal degrees.
    alt : float
        Altitude in metres above sea level (positive = up).
    origin_lat, origin_lon : float
        Reference origin in decimal degrees.
    origin_alt : float
        Reference origin altitude in metres above sea level.

    Returns
    -------
    (x, y, z) : tuple of float
        NED coordinates in metres.
        x = north, y = east, z = down (negative = above ground).
    """
    x = (lat - origin_lat) * _METRES_PER_DEG_LAT
    y = (lon - origin_lon) * _metres_per_deg_lon(origin_lat)
    z = -(alt - origin_alt)  # NED: positive z = down, so negate altitude
    return (x, y, z)


def ned_to_gps(
    x: float,
    y: float,
    z: float,
    origin_lat: float,
    origin_lon: float,
    origin_alt: float = 0.0,
) -> Tuple[float, float, float]:
    """
    Convert local NED (x, y, z) back to GPS (lat, lon, alt).

    Parameters
    ----------
    x, y, z : float
        NED coordinates in metres (x=north, y=east, z=down).
    origin_lat, origin_lon : float
        Reference origin in decimal degrees.
    origin_alt : float
        Reference origin altitude in metres above sea level.

    Returns
    -------
    (lat, lon, alt) : tuple of float
        GPS coordinates (decimal degrees and metres above sea level).
    """
    lat = origin_lat + x / _METRES_PER_DEG_LAT
    lon = origin_lon + y / _metres_per_deg_lon(origin_lat)
    alt = origin_alt - z  # z is down, so altitude = origin_alt - z
    return (lat, lon, alt)
