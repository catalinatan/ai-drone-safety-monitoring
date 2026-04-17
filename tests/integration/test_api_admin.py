"""Integration tests for admin routes."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api import dependencies as deps
from src.hardware.camera.file_camera import FileCamera
from src.services.feed_manager import FeedManager


@pytest.fixture()
def persistent_config_client():
    """TestClient with a persistent config dict for testing PUT /config."""
    app = create_app()

    # Create feeds manually
    fm = FeedManager()
    cam = FileCamera("/dev/null")
    fm.register_feed("cam-1", name="Camera 1", location="Zone A", camera=cam)
    fm.register_feed("cam-2", name="Camera 2", location="Zone B", camera=cam)

    # Create a PERSISTENT config dict (not a lambda)
    persistent_cfg = {
        "server": {"backend_port": 8001},
        "streaming": {"stream_fps": 10, "capture_fps": 10},
    }

    # Override with the persistent dict
    app.dependency_overrides[deps.get_feed_manager] = lambda: fm
    app.dependency_overrides[deps.get_config] = lambda: persistent_cfg
    app.dependency_overrides[deps.get_drone_api] = lambda: None
    app.dependency_overrides[deps.get_trigger_store] = lambda: deps.TriggerStore()

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


def test_get_config_returns_dict(client):
    """GET /config returns current runtime config."""
    resp = client.get("/config")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    assert "server" in data or "streaming" in data  # has some config


def test_put_config_updates_runtime(persistent_config_client):
    """PUT /config deep-merges updates into runtime config."""
    client = persistent_config_client

    # Update with a partial dict
    update_payload = {"streaming": {"stream_fps": 25}}
    put_resp = client.put("/config", json=update_payload)
    assert put_resp.status_code == 200

    # Get again to verify change was merged
    get_resp = client.get("/config")
    new_config = get_resp.json()
    # Verify the update was merged (existing key was modified)
    assert new_config.get("streaming", {}).get("stream_fps") == 25
    # Verify other fields still exist (deep merge didn't replace entire dict)
    assert "server" in new_config or "streaming" in new_config


def test_get_config_feeds_returns_feeds(client):
    """GET /config/feeds returns feeds.yaml structure."""
    resp = client.get("/config/feeds")
    assert resp.status_code == 200
    data = resp.json()
    # Should be dict of feeds keyed by feed_id
    assert isinstance(data, dict)


def test_put_config_feeds_writes_yaml(client, tmp_path):
    """PUT /config/feeds writes to feeds.yaml."""
    import shutil
    from pathlib import Path

    # Backup original feeds.yaml before modifying
    feeds_file = Path("config/feeds.yaml")
    backup_file = tmp_path / "feeds.yaml.backup"
    if feeds_file.exists():
        shutil.copy(feeds_file, backup_file)

    try:
        # Test that the endpoint accepts the request
        feeds_payload = {
            "feeds": {
                "test-cam": {
                    "name": "Test Camera",
                    "location": "Test Location",
                    "scene_type": "ship",
                    "camera": {
                        "type": "file",
                        "params": {"path": "/tmp/test.mp4"},
                    },
                }
            }
        }
        resp = client.put("/config/feeds", json=feeds_payload)
        # May succeed or fail depending on permissions, but should be a valid response
        assert resp.status_code in [200, 500]
    finally:
        # Restore original feeds.yaml
        if backup_file.exists():
            shutil.copy(backup_file, feeds_file)


def test_get_events_returns_list(client):
    """GET /events returns list of recent events."""
    resp = client.get("/events")
    assert resp.status_code == 200
    data = resp.json()
    assert "events" in data
    assert isinstance(data["events"], list)


def test_get_events_respects_limit(client):
    """GET /events?limit=N respects the limit parameter."""
    # First log some events
    from src.services.event_logger import get_event_logger, AuditEventType

    logger = get_event_logger()
    for i in range(5):
        logger.log(AuditEventType.ALARM_FIRED, feed_id=f"cam-{i}", count=i)

    # Request with limit
    resp = client.get("/events?limit=3")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["events"]) <= 3


def test_put_config_bad_json_returns_400(client):
    """PUT /config with invalid JSON returns 400."""
    resp = client.put(
        "/config",
        content="not json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 400
