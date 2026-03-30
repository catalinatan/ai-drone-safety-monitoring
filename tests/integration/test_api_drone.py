"""Integration tests for drone / trigger routes."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api import dependencies as deps
from src.services.feed_manager import FeedManager
from src.hardware.camera.file_camera import FileCamera


# ---------------------------------------------------------------------------
# Shared fixture: client with a pre-populated TriggerStore
# ---------------------------------------------------------------------------

@pytest.fixture()
def store_with_trigger():
    store = deps.TriggerStore()
    import io
    import numpy as np
    import cv2

    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    jpeg = buf.tobytes()

    event = deps.TriggerEvent(
        id=store.next_id(),
        feed_id="cam-1",
        timestamp="2024-01-01T00:00:00",
        coords=(1.0, 2.0, 3.0),
        snapshot=jpeg,
        replay_frames=[("2024-01-01T00:00:00", jpeg)],
    )
    store.add(event)
    return store


@pytest.fixture()
def client_with_trigger(store_with_trigger):
    fm = FeedManager()
    cam = FileCamera("/dev/null")
    fm.register_feed("cam-1", name="Camera 1", location="Zone A", camera=cam)

    app = create_app()
    app.dependency_overrides[deps.get_feed_manager] = lambda: fm
    app.dependency_overrides[deps.get_config] = lambda: {"server": {"backend_port": 8001}}
    app.dependency_overrides[deps.get_drone_api] = lambda: None
    app.dependency_overrides[deps.get_trigger_store] = lambda: store_with_trigger

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# GET /triggers
# ---------------------------------------------------------------------------

def test_list_triggers_empty(client):
    resp = client.get("/triggers")
    assert resp.status_code == 200
    assert resp.json()["triggers"] == []


def test_list_triggers_with_event(client_with_trigger):
    data = client_with_trigger.get("/triggers").json()
    assert len(data["triggers"]) == 1
    t = data["triggers"][0]
    assert t["id"] == 1
    assert t["feed_id"] == "cam-1"
    assert t["replay_frame_count"] == 1
    assert "coords" in t


# ---------------------------------------------------------------------------
# GET /triggers/{id}/snapshot
# ---------------------------------------------------------------------------

def test_get_snapshot_returns_jpeg(client_with_trigger):
    resp = client_with_trigger.get("/triggers/1/snapshot")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"


def test_get_snapshot_not_found(client_with_trigger):
    resp = client_with_trigger.get("/triggers/999/snapshot")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /triggers/{id}/replay/{frame}
# ---------------------------------------------------------------------------

def test_get_replay_frame_valid(client_with_trigger):
    resp = client_with_trigger.get("/triggers/1/replay/0")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"


def test_get_replay_frame_out_of_range(client_with_trigger):
    resp = client_with_trigger.get("/triggers/1/replay/99")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /triggers/{id}/deploy
# ---------------------------------------------------------------------------

def test_deploy_no_drone_api(client_with_trigger):
    resp = client_with_trigger.post("/triggers/1/deploy")
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# DELETE /triggers/{id}
# ---------------------------------------------------------------------------

def test_delete_trigger(client_with_trigger):
    resp = client_with_trigger.delete("/triggers/1")
    assert resp.status_code == 200
    assert resp.json()["status"] == "removed"


def test_delete_trigger_not_found(client_with_trigger):
    resp = client_with_trigger.delete("/triggers/999")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Backward-compat endpoints
# ---------------------------------------------------------------------------

def test_trigger_info_empty(client):
    data = client.get("/trigger-info").json()
    assert data["has_snapshot"] is False


def test_trigger_info_with_event(client_with_trigger):
    data = client_with_trigger.get("/trigger-info").json()
    assert data["has_snapshot"] is True
    assert data["feed_id"] == "cam-1"


def test_trigger_snapshot_empty_returns_204(client):
    resp = client.get("/trigger-snapshot")
    assert resp.status_code == 204


def test_trigger_snapshot_with_event(client_with_trigger):
    resp = client_with_trigger.get("/trigger-snapshot")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"


def test_trigger_replay_empty_returns_404(client):
    resp = client.get("/trigger-replay/0")
    assert resp.status_code == 404
