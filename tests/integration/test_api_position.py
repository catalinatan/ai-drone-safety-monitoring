"""Integration tests for live GPS position and calibration endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api import dependencies as deps
from src.hardware.camera.file_camera import FileCamera
from src.services.feed_manager import FeedManager


@pytest.fixture()
def client_with_pose():
    app = create_app()
    fm = FeedManager()
    cam = FileCamera("/dev/null")
    fm.register_feed(
        "cam-1", name="Camera 1", location="Bridge", camera=cam,
        camera_pose={
            "gps": {"latitude": 1.2847, "longitude": 103.8610, "altitude": 15.0},
            "orientation": (-30, 0, 0),
            "fov": 90,
        },
    )
    fm.register_feed("cam-2", name="Camera 2", location="Dock", camera=cam)

    app.dependency_overrides[deps.get_feed_manager] = lambda: fm
    app.dependency_overrides[deps.get_config] = lambda: {
        "drone": {"safe_altitude": -10.0},
        "detection": {"cctv_height_meters": 10.0},
    }
    app.dependency_overrides[deps.get_drone_api] = lambda: None
    app.dependency_overrides[deps.get_trigger_store] = lambda: deps.TriggerStore()

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c, fm


class TestLivePositionAPI:

    def test_update_position_success(self, client_with_pose):
        client, fm = client_with_pose
        resp = client.post("/feeds/cam-1/position", json={
            "latitude": 1.2850, "longitude": 103.8615, "altitude": 15.0,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["gps"]["latitude"] == 1.2850
        state = fm.get_state("cam-1")
        assert state.camera_pose["gps"]["latitude"] == 1.2850

    def test_update_position_with_heading(self, client_with_pose):
        client, fm = client_with_pose
        resp = client.post("/feeds/cam-1/position", json={
            "latitude": 1.2850, "longitude": 103.8615, "altitude": 15.0,
            "heading": 90.0,
        })
        assert resp.status_code == 200
        state = fm.get_state("cam-1")
        assert state.camera_pose["orientation"][1] == 90.0

    def test_update_position_unknown_feed_404(self, client_with_pose):
        client, _ = client_with_pose
        resp = client.post("/feeds/no-such-feed/position", json={
            "latitude": 0, "longitude": 0, "altitude": 0,
        })
        assert resp.status_code == 404

    def test_update_position_creates_pose_if_missing(self, client_with_pose):
        client, fm = client_with_pose
        resp = client.post("/feeds/cam-2/position", json={
            "latitude": 1.2847, "longitude": 103.8610, "altitude": 10.0,
        })
        assert resp.status_code == 200
        state = fm.get_state("cam-2")
        assert state.camera_pose is not None
        assert state.camera_pose["gps"]["latitude"] == 1.2847


class TestCalibrationAPI:

    def test_calibrate_success(self, client_with_pose):
        client, _ = client_with_pose
        # World points as GPS: [lat, lon, alt]
        resp = client.post("/feeds/cam-1/calibrate", json={
            "pixel_points": [[100, 200], [300, 400], [500, 100], [200, 350]],
            "world_points": [
                [1.2847, 103.8610, 0.0],
                [1.2848, 103.8610, 0.0],
                [1.2847, 103.8611, 0.0],
                [1.2848, 103.8611, 0.0],
            ],
            "frame_w": 640,
            "frame_h": 480,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "orientation" in data
        assert "pitch" in data["orientation"]

    def test_calibrate_too_few_points_400(self, client_with_pose):
        client, _ = client_with_pose
        resp = client.post("/feeds/cam-1/calibrate", json={
            "pixel_points": [[100, 200], [300, 400]],
            "world_points": [[1.2847, 103.8610, 0.0], [1.2848, 103.8610, 0.0]],
            "frame_w": 640,
            "frame_h": 480,
        })
        assert resp.status_code == 400

    def test_calibrate_unknown_feed_404(self, client_with_pose):
        client, _ = client_with_pose
        resp = client.post("/feeds/no-such/calibrate", json={
            "pixel_points": [[100, 200], [300, 400], [500, 100], [200, 350]],
            "world_points": [
                [1.2847, 103.8610, 0.0],
                [1.2848, 103.8610, 0.0],
                [1.2847, 103.8611, 0.0],
                [1.2848, 103.8611, 0.0],
            ],
            "frame_w": 640,
            "frame_h": 480,
        })
        assert resp.status_code == 404
