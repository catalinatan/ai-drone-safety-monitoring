"""
FastAPI application factory.

Creates and configures the FastAPI app, registers all routes, and manages
startup/shutdown via the lifespan context manager.

Usage (production):
    uvicorn src.api.app:app --host 0.0.0.0 --port 8001

Usage (development via main.py):
    from src.api.app import create_app
    app = create_app()
"""

from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api import dependencies as deps
from src.api.routes import drone, feeds, health, video, zones
from src.core.config import get_config, get_feeds_config
from src.hardware import create_camera_backend
from src.services.feed_manager import FeedManager


# ---------------------------------------------------------------------------
# Background threads
# ---------------------------------------------------------------------------

def _capture_loop(fm: FeedManager, cfg: Dict[str, Any]) -> None:
    """Continuously grab frames from all registered camera backends."""
    fps = cfg.get("streaming", {}).get("capture_fps", 30)
    interval = 1.0 / max(1, fps)

    while getattr(fm, "_running", False):
        for feed_id in fm.feed_ids():
            camera = fm.get_camera(feed_id)
            if camera is None or not camera.is_connected:
                continue
            try:
                frame = camera.grab_frame()
                if frame is None:
                    continue

                # Try to get vehicle position (AirSim cameras expose this)
                position = None
                if hasattr(camera, "get_vehicle_position"):
                    position = camera.get_vehicle_position()

                fm.store_frame(feed_id, frame, position=position)
            except Exception as e:
                print(f"[CAPTURE] {feed_id}: {e}")
        time.sleep(interval)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: initialise hardware and background threads. Shutdown: stop them."""
    cfg = get_config()
    feeds_cfg = get_feeds_config()

    fm = FeedManager()
    deps.set_feed_manager(fm)
    deps.set_config(cfg)

    # Register feeds from feeds.yaml and attempt camera connections
    connected_count = 0
    for feed_id, feed_def in feeds_cfg.items():
        camera_cfg = feed_def.get("camera", {})
        try:
            camera = create_camera_backend(camera_cfg)
            ok = camera.connect()
            if ok:
                connected_count += 1
        except Exception as e:
            print(f"[INIT] {feed_id}: camera backend error — {e}")
            # Use a stub (FileCamera with no path) so the feed exists in the API
            from src.hardware.camera.file_camera import FileCamera
            camera = FileCamera("/dev/null")

        fm.register_feed(
            feed_id=feed_id,
            name=feed_def.get("name", feed_id),
            location=feed_def.get("location", ""),
            camera=camera,
            scene_type=feed_def.get("scene_type"),
        )

    if connected_count > 0:
        print(f"[INIT] {connected_count}/{len(feeds_cfg)} cameras connected")
    else:
        print("[INIT] No cameras connected — running in limited mode (no AirSim)")

    # Connect to drone API
    try:
        from src.backend.config import DRONE_API_URL, DRONE_API_TIMEOUT
        from src.backend.drone_client import DroneAPIClient
        drone_api = DroneAPIClient(DRONE_API_URL, DRONE_API_TIMEOUT)
        if drone_api.check_connection():
            deps.set_drone_api(drone_api)
            print("[INIT] Drone API connected")
        else:
            print("[INIT] Drone API not available")
    except Exception as e:
        print(f"[INIT] Drone API init failed: {e}")

    # Start capture thread
    fm._running = True
    capture_thread = threading.Thread(
        target=_capture_loop, args=(fm, cfg), daemon=True, name="frame-capture"
    )
    capture_thread.start()

    print("[SERVER] Backend started")
    yield

    # Shutdown
    fm._running = False
    print("[SERVER] Backend shutting down")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Safety Monitoring Backend",
        version="2.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(feeds.router)
    app.include_router(zones.router)
    app.include_router(video.router)
    app.include_router(drone.router)

    return app


# Module-level app instance (for uvicorn src.api.app:app)
app = create_app()
