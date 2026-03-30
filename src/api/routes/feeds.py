"""
Feed routes:
  GET  /feeds               — list all feeds with status
  GET  /feeds/{id}/status   — single feed detection status
  PATCH /settings           — update global scene type / auto-refresh
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from src.api.dependencies import get_feed_manager, get_config
from src.core.models import DetectionStatus, TargetCoordinate
from src.services.feed_manager import FeedManager

router = APIRouter()


@router.get("/feeds")
async def list_feeds(
    fm: FeedManager = Depends(get_feed_manager),
    cfg: dict = Depends(get_config),
):
    backend_port = cfg.get("server", {}).get("backend_port", 8001)
    feeds = []
    for feed_id in fm.feed_ids():
        state = fm.get_state(feed_id)
        snap = fm.snapshot(feed_id) or {}
        feeds.append({
            "id": feed_id,
            "name": state.name,
            "location": state.location,
            "imageSrc": f"http://localhost:{backend_port}/video_feed/{feed_id}",
            "zones": [z.model_dump() for z in fm.get_zones(feed_id)],
            "isLive": True,
            "status": snap,
            "sceneType": state.scene_type,
            "autoSegActive": state.auto_seg_active,
        })
    return {"feeds": feeds}


@router.get("/feeds/{feed_id}/status")
async def get_feed_status(
    feed_id: str,
    fm: FeedManager = Depends(get_feed_manager),
):
    snap = fm.snapshot(feed_id)
    if snap is None:
        raise HTTPException(status_code=404, detail=f"Feed {feed_id!r} not found")
    return snap


@router.patch("/settings")
async def update_settings(
    request: Request,
    fm: FeedManager = Depends(get_feed_manager),
):
    body = await request.json()
    scene_type = body.get("sceneType")

    if scene_type is not None:
        valid = {"ship", "railway", "bridge"}
        if scene_type not in valid:
            return JSONResponse({"error": f"Invalid scene type: {scene_type!r}"}, status_code=400)
        for feed_id in fm.feed_ids():
            state = fm.get_state(feed_id)
            if state:
                state.scene_type = scene_type

    return {"status": "ok"}
