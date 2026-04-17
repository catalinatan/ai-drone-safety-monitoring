"""Tests for GPS <-> NED conversion utilities."""

from __future__ import annotations

import math
import pytest

from src.spatial.gps_utils import gps_to_ned, ned_to_gps


class TestGpsToNed:

    def test_same_point_returns_zero(self):
        x, y, z = gps_to_ned(1.2847, 103.8610, 15.0, 1.2847, 103.8610, 15.0)
        assert abs(x) < 0.01
        assert abs(y) < 0.01
        assert abs(z) < 0.01

    def test_north_offset(self):
        """Moving 1 degree north ≈ 111,320 metres."""
        x, y, z = gps_to_ned(1.0, 103.0, 0.0, 0.0, 103.0, 0.0)
        assert abs(x - 111_320.0) < 1.0
        assert abs(y) < 0.01

    def test_east_offset(self):
        """Moving 1 degree east at equator ≈ 111,320 metres."""
        x, y, z = gps_to_ned(0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
        assert abs(x) < 0.01
        assert abs(y - 111_320.0) < 1.0  # cos(0) = 1

    def test_altitude_to_ned_z(self):
        """Higher altitude = more negative z (NED convention)."""
        x, y, z = gps_to_ned(0.0, 0.0, 20.0, 0.0, 0.0, 0.0)
        assert z == -20.0

    def test_below_origin_altitude(self):
        """Lower altitude = positive z."""
        x, y, z = gps_to_ned(0.0, 0.0, 0.0, 0.0, 0.0, 10.0)
        assert z == 10.0

    def test_small_offset_metres(self):
        """A small GPS offset should produce metres-scale NED."""
        # ~11m north, ~6m east at lat=1.28
        x, y, z = gps_to_ned(1.2848, 103.86105, 0.0, 1.2847, 103.8610, 0.0)
        assert 10.0 < x < 12.0  # ~11.1m
        assert 4.0 < y < 7.0    # ~5.6m


class TestNedToGps:

    def test_roundtrip(self):
        """gps_to_ned -> ned_to_gps should return the original GPS."""
        orig_lat, orig_lon, orig_alt = 1.2847, 103.8610, 15.0
        origin_lat, origin_lon, origin_alt = 1.2800, 103.8600, 0.0

        x, y, z = gps_to_ned(orig_lat, orig_lon, orig_alt, origin_lat, origin_lon, origin_alt)
        lat, lon, alt = ned_to_gps(x, y, z, origin_lat, origin_lon, origin_alt)

        assert abs(lat - orig_lat) < 1e-8
        assert abs(lon - orig_lon) < 1e-8
        assert abs(alt - orig_alt) < 0.01

    def test_zero_ned_returns_origin(self):
        lat, lon, alt = ned_to_gps(0, 0, 0, 1.2847, 103.8610, 15.0)
        assert abs(lat - 1.2847) < 1e-10
        assert abs(lon - 103.8610) < 1e-10
        assert abs(alt - 15.0) < 0.01
