"""Integration tests for feed and settings routes."""

from __future__ import annotations


# ---------------------------------------------------------------------------
# GET /feeds
# ---------------------------------------------------------------------------

def test_list_feeds_returns_200(client):
    resp = client.get("/feeds")
    assert resp.status_code == 200


def test_list_feeds_contains_registered_feeds(client):
    data = client.get("/feeds").json()
    ids = [f["id"] for f in data["feeds"]]
    assert "cam-1" in ids
    assert "cam-2" in ids


def test_list_feeds_fields(client):
    data = client.get("/feeds").json()
    feed = data["feeds"][0]
    for key in ("id", "name", "location", "imageSrc", "zones", "isLive", "status"):
        assert key in feed, f"Missing field: {key}"


def test_list_feeds_image_src_format(client):
    data = client.get("/feeds").json()
    feed = next(f for f in data["feeds"] if f["id"] == "cam-1")
    assert "video_feed/cam-1" in feed["imageSrc"]


# ---------------------------------------------------------------------------
# GET /feeds/{feed_id}/status
# ---------------------------------------------------------------------------

def test_feed_status_known_feed(client):
    resp = client.get("/feeds/cam-1/status")
    assert resp.status_code == 200


def test_feed_status_unknown_feed(client):
    resp = client.get("/feeds/nonexistent/status")
    assert resp.status_code == 404


def test_feed_status_fields(client):
    data = client.get("/feeds/cam-1/status").json()
    for key in ("alarm_active", "caution_active", "people_count"):
        assert key in data, f"Missing field: {key}"


# ---------------------------------------------------------------------------
# PATCH /settings
# ---------------------------------------------------------------------------

def test_patch_settings_valid_scene_type(client):
    resp = client.patch("/settings", json={"sceneType": "ship"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_patch_settings_invalid_scene_type(client):
    resp = client.patch("/settings", json={"sceneType": "moon_base"})
    assert resp.status_code == 400


def test_patch_settings_no_scene_type(client):
    resp = client.patch("/settings", json={})
    assert resp.status_code == 200
