"""
Shared Pydantic data models used across the system.

These models define the API contract and are framework-agnostic —
they carry no FastAPI or AirSim dependencies.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel


class Point(BaseModel):
    x: float  # Percentage (0–100)
    y: float  # Percentage (0–100)


class Zone(BaseModel):
    id: str
    level: str  # 'red' | 'yellow' | 'green'
    points: List[Point]
    source: str = "manual"  # 'manual' | 'auto'


class ZonesUpdateRequest(BaseModel):
    zones: List[Zone]


class TargetCoordinate(BaseModel):
    x: float  # North (meters)
    y: float  # East (meters)
    z: float  # Down (meters, negative = above ground)


class DetectionStatus(BaseModel):
    feed_id: str
    alarm_active: bool          # RED zone intrusion — drone deployment
    caution_active: bool        # YELLOW zone intrusion — highlight only
    people_count: int
    danger_count: int           # People in RED zones
    caution_count: int          # People in YELLOW zones
    target_coordinates: Optional[TargetCoordinate] = None
    last_detection_time: Optional[str] = None
    position: Optional[TargetCoordinate] = None  # CCTV camera position (NED)


class FeedInfo(BaseModel):
    """Summary info for a single feed returned by GET /feeds."""
    id: str
    name: str
    location: str
    scene_type: Optional[str] = None
