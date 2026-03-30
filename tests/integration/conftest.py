"""
Shared fixtures for integration tests.

Uses FastAPI's TestClient with dependency overrides so no real hardware,
AirSim, or camera connections are needed.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api import dependencies as deps
from src.hardware.camera.file_camera import FileCamera
from src.services.feed_manager import FeedManager


@pytest.fixture()
def feed_manager_with_feeds():
    """FeedManager pre-populated with two test feeds (no real camera)."""
    fm = FeedManager()
    cam = FileCamera("/dev/null")  # stub — never connected
    fm.register_feed("cam-1", name="Camera 1", location="Zone A", camera=cam)
    fm.register_feed("cam-2", name="Camera 2", location="Zone B", camera=cam)
    # Store a dummy frame so feed is non-empty
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    fm.store_frame("cam-1", frame)
    return fm


@pytest.fixture()
def client(feed_manager_with_feeds):
    """TestClient with overridden dependencies — no lifespan side-effects."""
    app = create_app()
    fm = feed_manager_with_feeds

    app.dependency_overrides[deps.get_feed_manager] = lambda: fm
    app.dependency_overrides[deps.get_config] = lambda: {
        "server": {"backend_port": 8001},
        "streaming": {"stream_fps": 10, "capture_fps": 10},
    }
    app.dependency_overrides[deps.get_drone_api] = lambda: None
    app.dependency_overrides[deps.get_trigger_store] = lambda: deps.TriggerStore()

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
