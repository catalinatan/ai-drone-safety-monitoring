"""Backend-specific test fixtures — provides a configured FastAPI TestClient."""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np


@pytest.fixture
def mock_airsim():
    """Mock the airsim module so server.py can be imported without AirSim installed."""
    with patch("src.backend.server.airsim") as mock:
        yield mock


@pytest.fixture
def test_client(mock_airsim):
    """FastAPI TestClient with a pre-initialized FeedManager (no AirSim needed).

    Uses httpx via Starlette's TestClient under the hood.
    """
    from starlette.testclient import TestClient
    from src.backend.server import app, feed_manager, FeedState

    # Pre-populate feeds so endpoints work without AirSim
    feed_manager.feeds = {
        "cctv-1": FeedState(
            feed_id="cctv-1",
            camera_name="0",
            vehicle_name="Drone2",
            name="CCTV CAM 1",
            location="Aerial Overview",
        ),
    }
    feed_manager.client = MagicMock()  # Mock AirSim client

    # Give the feed a cached frame so video_feed works
    feed_manager.feeds["cctv-1"].last_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    yield TestClient(app)

    # Cleanup
    feed_manager.feeds = {}
    feed_manager.client = None
