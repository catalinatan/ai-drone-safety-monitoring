"""Integration tests for zone routes."""

from __future__ import annotations


ZONES_PAYLOAD = {
    "zones": [
        {
            "id": "zone-red",
            "level": "red",
            "points": [
                {"x": 0.1, "y": 0.1},
                {"x": 0.4, "y": 0.1},
                {"x": 0.4, "y": 0.4},
                {"x": 0.1, "y": 0.4},
            ],
        }
    ]
}


def test_post_zones_returns_200(client):
    resp = client.post("/feeds/cam-1/zones", json=ZONES_PAYLOAD)
    assert resp.status_code == 200


def test_post_zones_response_fields(client):
    data = client.post("/feeds/cam-1/zones", json=ZONES_PAYLOAD).json()
    assert data["status"] == "success"
    assert data["zones_count"] == 1


def test_post_zones_unknown_feed(client):
    resp = client.post("/feeds/ghost/zones", json=ZONES_PAYLOAD)
    assert resp.status_code == 404


def test_post_zones_persists_to_feed(client, feed_manager_with_feeds):
    client.post("/feeds/cam-1/zones", json=ZONES_PAYLOAD)
    zones = feed_manager_with_feeds.get_zones("cam-1")
    assert len(zones) == 1
    assert zones[0].level == "red"


def test_auto_segment_no_scene_type_returns_400(client):
    """Feeds without a scene_type configured return 400."""
    resp = client.post("/feeds/cam-1/auto-segment")
    assert resp.status_code == 400


def test_auto_segment_with_scene_type_returns_503(client, feed_manager_with_feeds):
    """With scene_type set, the endpoint returns 503 (model unavailable)."""
    feed_manager_with_feeds.get_state("cam-1").scene_type = "ship"
    resp = client.post("/feeds/cam-1/auto-segment")
    assert resp.status_code == 503
