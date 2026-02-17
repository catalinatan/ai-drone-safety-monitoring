"""Integration tests for backend API endpoints using FastAPI TestClient.

These tests exercise the full HTTP layer (routing, serialization, status codes)
with a mocked AirSim backend so no simulator is required.
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock


# ============================================================================
# Health Endpoint
# ============================================================================

class TestHealthEndpoint:

    @pytest.mark.integration
    def test_health_returns_200(self, test_client):
        resp = test_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "feeds_count" in data

    @pytest.mark.integration
    def test_health_reports_feed_count(self, test_client):
        data = test_client.get("/health").json()
        assert data["feeds_count"] == 1  # conftest pre-populates one feed


# ============================================================================
# Feeds Endpoint
# ============================================================================

class TestFeedsEndpoint:

    @pytest.mark.integration
    def test_list_feeds(self, test_client):
        resp = test_client.get("/feeds")
        assert resp.status_code == 200
        feeds = resp.json()["feeds"]
        assert len(feeds) == 1
        feed = feeds[0]
        assert feed["id"] == "cctv-1"
        assert feed["name"] == "CCTV CAM 1"
        assert feed["isLive"] is True
        assert "imageSrc" in feed

    @pytest.mark.integration
    def test_feed_contains_status(self, test_client):
        feeds = test_client.get("/feeds").json()["feeds"]
        status = feeds[0]["status"]
        assert "alarm_active" in status
        assert "people_count" in status
        assert status["feed_id"] == "cctv-1"


# ============================================================================
# Feed Status Endpoint
# ============================================================================

class TestFeedStatusEndpoint:

    @pytest.mark.integration
    def test_status_known_feed(self, test_client):
        resp = test_client.get("/feeds/cctv-1/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["feed_id"] == "cctv-1"
        assert data["alarm_active"] is False
        assert data["people_count"] == 0

    @pytest.mark.integration
    def test_status_unknown_feed_returns_404(self, test_client):
        resp = test_client.get("/feeds/nonexistent/status")
        assert resp.status_code == 404


# ============================================================================
# Zones Endpoint
# ============================================================================

class TestZonesEndpoint:

    @pytest.mark.integration
    @patch("src.backend.server.feed_manager.get_frame")
    @patch("src.backend.server.save_zones_to_file")
    def test_post_zones_success(self, mock_save, mock_get_frame, test_client):
        # Provide a fake frame so the endpoint can read dimensions
        mock_get_frame.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        payload = {
            "zones": [
                {
                    "id": "z1",
                    "level": "red",
                    "points": [
                        {"x": 10, "y": 10},
                        {"x": 50, "y": 10},
                        {"x": 50, "y": 50},
                        {"x": 10, "y": 50},
                    ],
                }
            ]
        }
        resp = test_client.post("/feeds/cctv-1/zones", json=payload)
        assert resp.status_code == 200
        assert resp.json()["zones_count"] == 1

    @pytest.mark.integration
    @patch("src.backend.server.feed_manager.get_frame")
    @patch("src.backend.server.save_zones_to_file")
    def test_post_zones_creates_masks(self, mock_save, mock_get_frame, test_client):
        mock_get_frame.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        payload = {
            "zones": [
                {
                    "id": "z1",
                    "level": "red",
                    "points": [
                        {"x": 0, "y": 0},
                        {"x": 50, "y": 0},
                        {"x": 50, "y": 50},
                        {"x": 0, "y": 50},
                    ],
                },
                {
                    "id": "z2",
                    "level": "yellow",
                    "points": [
                        {"x": 60, "y": 60},
                        {"x": 90, "y": 60},
                        {"x": 90, "y": 90},
                        {"x": 60, "y": 90},
                    ],
                },
            ]
        }
        resp = test_client.post("/feeds/cctv-1/zones", json=payload)
        assert resp.status_code == 200
        assert resp.json()["zones_count"] == 2

        from src.backend.server import feed_manager
        feed = feed_manager.feeds["cctv-1"]
        assert feed.red_zone_mask is not None
        assert feed.yellow_zone_mask is not None

    @pytest.mark.integration
    def test_post_zones_unknown_feed_returns_404(self, test_client):
        payload = {"zones": []}
        resp = test_client.post("/feeds/nonexistent/zones", json=payload)
        assert resp.status_code == 404

    @pytest.mark.integration
    @patch("src.backend.server.feed_manager.get_frame", return_value=None)
    def test_post_zones_no_frame_returns_503(self, mock_get_frame, test_client):
        payload = {
            "zones": [
                {
                    "id": "z1",
                    "level": "red",
                    "points": [{"x": 0, "y": 0}, {"x": 50, "y": 0}, {"x": 50, "y": 50}],
                }
            ]
        }
        resp = test_client.post("/feeds/cctv-1/zones", json=payload)
        assert resp.status_code == 503


# ============================================================================
# Video Feed Endpoint
# ============================================================================

class TestVideoFeedEndpoint:

    @pytest.mark.integration
    @patch("src.backend.server.generate_mjpeg_frames")
    def test_video_feed_returns_mjpeg_stream(self, mock_gen, test_client):
        """Verify the stream returns MJPEG content-type and at least one frame."""
        # Mock the generator to yield a single MJPEG frame then stop
        fake_frame = (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            b'\xff\xd8\xff\xe0'  # JPEG magic bytes
            b'\r\n'
        )
        mock_gen.return_value = iter([fake_frame])

        resp = test_client.get("/video_feed/cctv-1")
        assert resp.status_code == 200
        assert "multipart/x-mixed-replace" in resp.headers["content-type"]
        assert b"--frame" in resp.content

    @pytest.mark.integration
    def test_video_feed_unknown_feed_returns_404(self, test_client):
        resp = test_client.get("/video_feed/nonexistent")
        assert resp.status_code == 404
