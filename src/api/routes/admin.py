"""
Admin routes:
  GET  /config               — current runtime config dict
  PUT  /config               — update runtime config (deep merge)
  GET  /config/feeds         — feeds.yaml content
  PUT  /config/feeds         — write feeds.yaml
  GET  /events?limit=100     — recent audit events
  POST /feeds/{id}/position  — live GPS position update
  POST /feeds/{id}/calibrate — PnP calibration
  POST /feeds/{id}/calibrate-height — camera height calibration
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.api.dependencies import get_config, get_event_logger, get_feed_manager
from src.services.feed_manager import FeedManager
from src.core.config import (
    PROJECT_ROOT,
    get_feeds_config,
    reset_feeds_config,
    _deep_merge,
)

router = APIRouter()


@router.get("/config")
async def get_config_endpoint(cfg: dict = Depends(get_config)):
    """Return current runtime config."""
    return cfg


@router.put("/config")
async def update_config_endpoint(
    request: Request,
    cfg: dict = Depends(get_config),
):
    """
    Update config with a partial dict (deep merge).
    Does not persist to disk — only affects runtime.
    """
    try:
        updates = await request.json()
    except Exception as e:
        return JSONResponse({"error": f"Invalid JSON: {e}"}, status_code=400)

    merged = _deep_merge(cfg, updates)
    # Note: cfg is a dict reference, so modifying it modifies the singleton
    cfg.clear()
    cfg.update(merged)

    return {"status": "ok", "config": cfg}


@router.get("/config/feeds")
async def get_feeds_config_endpoint():
    """Return feeds.yaml configuration."""
    return get_feeds_config()


@router.put("/config/feeds")
async def update_feeds_config_endpoint(request: Request):
    """
    Write feeds to feeds.yaml and reload cache.
    Expects: {"feeds": {"feed_id": {...}, ...}}
    """
    try:
        body = await request.json()
    except Exception as e:
        return JSONResponse({"error": f"Invalid JSON: {e}"}, status_code=400)

    feeds_data = body.get("feeds")
    if not isinstance(feeds_data, dict):
        return JSONResponse(
            {"error": "feeds must be a dict"},
            status_code=400,
        )

    # Write to feeds.yaml
    feeds_yaml_path = PROJECT_ROOT / "config" / "feeds.yaml"
    try:
        with open(feeds_yaml_path, "w") as f:
            yaml.dump({"feeds": feeds_data}, f, default_flow_style=False)
    except Exception as e:
        return JSONResponse(
            {"error": f"Failed to write feeds.yaml: {e}"},
            status_code=500,
        )

    # Clear cache so next get_feeds_config() reloads
    reset_feeds_config()

    return {"status": "ok", "feeds": feeds_data}


@router.get("/events")
async def get_events_endpoint(
    limit: int = 100,
    el=Depends(get_event_logger),
):
    """
    Return recent audit events.

    Query params:
        limit: max events to return (default 100)
    """
    events = el.get_recent(limit)
    return {"events": [e.model_dump() if hasattr(e, "model_dump") else vars(e) for e in events]}


# ---------------------------------------------------------------------------
# Live position + calibration
# ---------------------------------------------------------------------------

class PositionBody(BaseModel):
    latitude: float
    longitude: float
    altitude: float
    heading: Optional[float] = None


class CalibrateBody(BaseModel):
    pixel_points: List[List[float]]
    world_points: List[List[float]]  # each is [lat, lon, alt]
    frame_w: int
    frame_h: int


class CalibrateHeightBody(BaseModel):
    pixel_x: float
    pixel_y: float
    latitude: float
    longitude: float
    frame_w: int
    frame_h: int


@router.post("/feeds/{feed_id}/position")
async def update_position(
    feed_id: str,
    body: PositionBody,
    request: Request,
    fm: FeedManager = Depends(get_feed_manager),
):
    """Push a live GPS position update for a feed's camera."""
    state = fm.get_state(feed_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Feed {feed_id!r} not found")

    gps = {"latitude": body.latitude, "longitude": body.longitude, "altitude": body.altitude}

    # Update the projection backend if available
    projections = getattr(request.app.state, "projections", None)
    if projections and feed_id in projections:
        proj = projections[feed_id]
        if hasattr(proj, "update_gps_position"):
            proj.update_gps_position(body.latitude, body.longitude, body.altitude)

    # Store GPS in camera_pose
    with state.lock:
        if state.camera_pose is None:
            state.camera_pose = {"gps": gps, "orientation": (0, 0, 0), "fov": 90.0}
        else:
            state.camera_pose["gps"] = gps

        if body.heading is not None:
            orientation = state.camera_pose.get("orientation", (0, 0, 0))
            state.camera_pose["orientation"] = (orientation[0], body.heading, orientation[2])

    return {"status": "ok", "feed_id": feed_id, "gps": gps}


@router.post("/feeds/{feed_id}/calibrate")
async def calibrate_feed(
    feed_id: str,
    body: CalibrateBody,
    request: Request,
    fm: FeedManager = Depends(get_feed_manager),
):
    """Run PnP calibration to determine camera orientation from point correspondences.

    world_points should be GPS coordinates: [[lat, lon, alt], ...].
    They are converted to local NED relative to the camera's GPS position.
    """
    state = fm.get_state(feed_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Feed {feed_id!r} not found")

    if len(body.pixel_points) < 4 or len(body.world_points) < 4:
        raise HTTPException(status_code=400, detail="At least 4 point correspondences required")

    from src.spatial.calibration import solve_camera_orientation
    from src.spatial.gps_utils import gps_to_ned

    # Gather existing pose info
    fov = 90.0
    camera_gps = None
    with state.lock:
        if state.camera_pose is not None:
            fov = state.camera_pose.get("fov", 90.0)
            camera_gps = state.camera_pose.get("gps")

    # Camera position as NED origin
    if camera_gps:
        origin_lat = camera_gps["latitude"]
        origin_lon = camera_gps["longitude"]
        origin_alt = camera_gps["altitude"]
    else:
        # No GPS set — use the first world point as origin
        origin_lat, origin_lon, origin_alt = body.world_points[0]

    camera_ned = (0.0, 0.0, 0.0)
    if camera_gps:
        camera_ned = gps_to_ned(
            origin_lat, origin_lon, origin_alt,
            origin_lat, origin_lon, origin_alt,
        )  # Always (0,0,0) when camera is the origin

    # Convert world points from GPS to NED
    world_pts_ned = []
    for wp in body.world_points:
        ned = gps_to_ned(wp[0], wp[1], wp[2], origin_lat, origin_lon, origin_alt)
        world_pts_ned.append(ned)

    pixel_pts = [(p[0], p[1]) for p in body.pixel_points]

    result = solve_camera_orientation(
        pixel_points=pixel_pts,
        world_points=world_pts_ned,
        frame_w=body.frame_w,
        frame_h=body.frame_h,
        fov=fov,
        camera_position=camera_ned,
    )

    if result is None:
        raise HTTPException(status_code=422, detail="Calibration failed — solvePnP could not converge")

    pitch, yaw, roll = result
    orientation = (pitch, yaw, roll)

    with state.lock:
        if state.camera_pose is None:
            state.camera_pose = {"gps": camera_gps, "orientation": orientation, "fov": fov}
        else:
            state.camera_pose["orientation"] = orientation

    # Update projection backend orientation if available
    projections = getattr(request.app.state, "projections", None)
    if projections and feed_id in projections:
        proj = projections[feed_id]
        proj.update_pose(orientation=orientation)

    return {
        "status": "ok",
        "feed_id": feed_id,
        "orientation": {"pitch": pitch, "yaw": yaw, "roll": roll},
    }


@router.post("/feeds/{feed_id}/calibrate-height")
async def calibrate_height(
    feed_id: str,
    body: CalibrateHeightBody,
    request: Request,
    fm: FeedManager = Depends(get_feed_manager),
):
    """Calibrate camera height above ground from a single known ground point.

    The user clicks a point on the ground in the camera image and provides
    its GPS coordinates. The system back-calculates the camera height that
    makes the projection ray land at that point.
    """
    state = fm.get_state(feed_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Feed {feed_id!r} not found")

    projections = getattr(request.app.state, "projections", None)
    if not projections or feed_id not in projections:
        raise HTTPException(status_code=400, detail="No projection backend for this feed")

    proj = projections[feed_id]

    # Convert GPS to NED relative to camera position
    from src.spatial.gps_utils import gps_to_ned

    camera_gps = None
    with state.lock:
        if state.camera_pose is not None:
            camera_gps = state.camera_pose.get("gps")

    if camera_gps:
        origin_lat = camera_gps["latitude"]
        origin_lon = camera_gps["longitude"]
        origin_alt = camera_gps["altitude"]
    else:
        # Use clicked point as rough origin
        origin_lat, origin_lon, origin_alt = body.latitude, body.longitude, 0.0

    ned = gps_to_ned(body.latitude, body.longitude, 0.0, origin_lat, origin_lon, origin_alt)
    world_x, world_y = ned[0], ned[1]

    height = proj.calibrate_height(
        body.pixel_x, body.pixel_y,
        world_x, world_y,
        body.frame_w, body.frame_h,
    )

    if height is None:
        raise HTTPException(
            status_code=422,
            detail="Height calibration failed — ensure the point is visible ground",
        )

    # Store calibrated height in camera_pose
    with state.lock:
        if state.camera_pose is None:
            state.camera_pose = {}
        state.camera_pose["calibrated_height"] = round(height, 2)

    return {
        "status": "ok",
        "feed_id": feed_id,
        "calibrated_height_m": round(height, 2),
    }
