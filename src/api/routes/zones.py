"""
Zone routes:
  POST /feeds/{id}/zones          — save zone definitions + regenerate masks
  POST /feeds/{id}/auto-segment   — trigger on-demand auto-segmentation
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_feed_manager
from src.core.models import ZonesUpdateRequest
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

    # Mark as manually edited so auto-seg won't overwrite
    state.manual_zones_set = True

    frame = fm.get_frame(feed_id)
    if frame is not None:
        fm.update_zones(feed_id, body.zones, frame.shape[1], frame.shape[0])
    else:
        # No frame yet — store zones and defer mask generation until first frame
        import threading
        with state.lock:
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
):
    state = fm.get_state(feed_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Feed {feed_id!r} not found")
    if not state.scene_type:
        raise HTTPException(status_code=400, detail=f"No scene type configured for {feed_id!r}")

    # Auto-segmentation requires the scene segmenter — not available without model files.
    # This endpoint is a hook; the actual segmentation runs when the segmenter is loaded.
    raise HTTPException(
        status_code=503,
        detail="Auto-segmentation not available in this environment. Load scene models to enable.",
    )
