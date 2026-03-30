"""
Admin routes:
  GET  /config               — current runtime config dict
  PUT  /config               — update runtime config (deep merge)
  GET  /config/feeds         — feeds.yaml content
  PUT  /config/feeds         — write feeds.yaml
  GET  /events?limit=100     — recent audit events
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from src.api.dependencies import get_config, get_event_logger
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
