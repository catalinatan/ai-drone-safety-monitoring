"""
Zone routes:
  POST /feeds/{id}/zones          — save zone definitions + regenerate masks
  POST /feeds/{id}/auto-segment   — trigger on-demand auto-segmentation
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_feed_manager, get_scene_segmenter
from src.core.models import Zone, ZonesUpdateRequest
from src.services.feed_manager import FeedManager
from src.services.zone_persistence import save_zones

router = APIRouter()


@router.post("/feeds/{feed_id}/zones")
async def update_zones(
    feed_id: str,
    body: ZonesUpdateRequest,
    fm: FeedManager = Depends(get_feed_manager),
):
    state = fm.get_state(feed_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Feed {feed_id!r} not found")

    frame = fm.get_frame(feed_id)
    if frame is not None:
        fm.update_zones(feed_id, body.zones, frame.shape[1], frame.shape[0], source="manual")
    else:
        # No frame yet — store as manual zones and defer mask generation until first frame
        with state.lock:
            state.manual_zones = list(body.zones)
            state.zones = list(body.zones)
            state._needs_mask_regen = True

    # Persist to disk (use configured path from zones in state)
    from src.core.config import get_config

    cfg = get_config()
    zones_file = cfg.get("zones", {}).get("persistence_file", "data/zones.json")
    save_zones(zones_file, feed_id, body.zones)

    return {"status": "success", "zones_count": len(body.zones)}


@router.post("/feeds/{feed_id}/auto-segment")
def trigger_auto_segment(
    feed_id: str,
    fm: FeedManager = Depends(get_feed_manager),
    segmenter=Depends(get_scene_segmenter),
):
    state = fm.get_state(feed_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Feed {feed_id!r} not found")
    if not state.scene_type:
        raise HTTPException(status_code=400, detail=f"No scene type configured for {feed_id!r}")

    if segmenter is None:
        raise HTTPException(
            status_code=503,
            detail="Scene segmenter not loaded",
        )

    frame = fm.get_frame(feed_id)
    if frame is None:
        raise HTTPException(
            status_code=503,
            detail="No frame available yet",
        )

    try:
        zone_dicts = segmenter.segment_frame(frame, state.scene_type)
        if not zone_dicts:
            return {
                "status": "ok",
                "zones_count": 0,
                "message": "No zones detected",
            }

        # Convert dicts to Zone objects and store as auto zones.
        # Manual zones (set by user) retain higher priority — auto zones are
        # only used for detection when no manual zones exist.
        zones = [Zone(**{**z, "source": "auto"}) for z in zone_dicts]
        fm.update_zones(feed_id, zones, frame.shape[1], frame.shape[0], source="auto")
        state.auto_seg_active = True
        state.last_auto_seg_time = time.monotonic()

        # Return effective (merged) zones so the frontend gets auto + manual
        effective_zones = fm.get_zones(feed_id)
        return {
            "status": "ok",
            "zones_count": len(zones),
            "zones": [z.model_dump() for z in effective_zones],
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Segmentation failed: {str(e)}",
        )
