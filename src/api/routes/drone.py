"""
Drone routes — proxy to the drone control server + trigger history CRUD.

  GET    /triggers                        — list all trigger events
  GET    /triggers/{id}/snapshot          — snapshot JPEG
  GET    /triggers/{id}/replay/{frame}    — replay frame JPEG
  POST   /triggers/{id}/deploy            — manual deploy drone to trigger coords
  DELETE /triggers/{id}                   — remove trigger
  GET    /trigger-snapshot                — latest trigger snapshot (compat)
  GET    /trigger-info                    — latest trigger metadata (compat)
  GET    /trigger-replay/{frame}          — latest trigger replay frame (compat)
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, Response

from src.api.dependencies import get_drone_api, get_trigger_store, TriggerStore

router = APIRouter()

TRIGGER_REPLAY_FPS = 10


@router.get("/triggers")
async def list_triggers(store: TriggerStore = Depends(get_trigger_store)):
    return {
        "triggers": [
            {
                "id": t.id,
                "feed_id": t.feed_id,
                "timestamp": t.timestamp,
                "deployed": t.deployed,
                "replay_frame_count": len(t.replay_frames),
                "replay_trigger_index": t.replay_trigger_index,
                "coords": list(t.coords),
                "replay_fps": TRIGGER_REPLAY_FPS,
            }
            for t in store.all()
        ],
        "replay_fps": TRIGGER_REPLAY_FPS,
    }


@router.get("/triggers/{trigger_id}/snapshot")
async def get_trigger_snapshot(trigger_id: int, store: TriggerStore = Depends(get_trigger_store)):
    t = store.get_by_id(trigger_id)
    if t is None:
        raise HTTPException(status_code=404, detail="Trigger not found")
    return Response(content=t.snapshot, media_type="image/jpeg")


@router.get("/triggers/{trigger_id}/replay/{frame_index}")
async def get_trigger_replay(
    trigger_id: int,
    frame_index: int,
    store: TriggerStore = Depends(get_trigger_store),
):
    t = store.get_by_id(trigger_id)
    if t is None:
        raise HTTPException(status_code=404, detail="Trigger not found")
    if frame_index < 0 or frame_index >= len(t.replay_frames):
        raise HTTPException(status_code=404, detail="Frame index out of range")
    _ts, jpeg = t.replay_frames[frame_index]
    return Response(content=jpeg, media_type="image/jpeg")


@router.post("/triggers/{trigger_id}/deploy")
async def deploy_to_trigger(
    trigger_id: int,
    store: TriggerStore = Depends(get_trigger_store),
    drone_api=Depends(get_drone_api),
):
    t = store.get_by_id(trigger_id)
    if t is None:
        raise HTTPException(status_code=404, detail="Trigger not found")
    if drone_api is None:
        raise HTTPException(status_code=503, detail="Drone API not connected")

    tx, ty, tz = t.coords
    drone_api.set_mode("automatic")
    if drone_api.goto_position(tx, ty, tz):
        t.deployed = True
        return {"status": "deployed", "trigger_id": trigger_id, "coords": list(t.coords)}
    raise HTTPException(status_code=500, detail="Drone goto command failed")


@router.delete("/triggers/{trigger_id}")
async def delete_trigger(trigger_id: int, store: TriggerStore = Depends(get_trigger_store)):
    if not store.remove(trigger_id):
        raise HTTPException(status_code=404, detail="Trigger not found")
    return {"status": "removed", "trigger_id": trigger_id}


# ---------------------------------------------------------------------------
# Backward-compat endpoints (use latest trigger)
# ---------------------------------------------------------------------------

@router.get("/trigger-snapshot")
async def get_latest_snapshot(store: TriggerStore = Depends(get_trigger_store)):
    latest = store.latest()
    if latest is None:
        return JSONResponse(status_code=204, content=None)
    return Response(content=latest.snapshot, media_type="image/jpeg")


@router.get("/trigger-info")
async def get_latest_trigger_info(store: TriggerStore = Depends(get_trigger_store)):
    latest = store.latest()
    if latest is None:
        return {
            "has_snapshot": False, "feed_id": None, "timestamp": None,
            "replay_frame_count": 0, "replay_fps": TRIGGER_REPLAY_FPS,
            "replay_trigger_index": 0,
        }
    return {
        "has_snapshot": True,
        "feed_id": latest.feed_id,
        "timestamp": latest.timestamp,
        "replay_frame_count": len(latest.replay_frames),
        "replay_fps": TRIGGER_REPLAY_FPS,
        "replay_trigger_index": latest.replay_trigger_index,
    }


@router.get("/trigger-replay/{frame_index}")
async def get_latest_replay_frame(
    frame_index: int,
    store: TriggerStore = Depends(get_trigger_store),
):
    latest = store.latest()
    if latest is None or frame_index < 0 or frame_index >= len(latest.replay_frames):
        raise HTTPException(status_code=404, detail="Frame index out of range")
    _ts, jpeg = latest.replay_frames[frame_index]
    return Response(content=jpeg, media_type="image/jpeg")
