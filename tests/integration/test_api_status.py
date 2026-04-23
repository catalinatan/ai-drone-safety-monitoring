"""Integration tests for WebSocket /ws/status."""

from __future__ import annotations

import json


def test_websocket_status_connects(client):
    """Test that WebSocket endpoint accepts connections."""
    with client.websocket_connect("/ws/status") as websocket:
        # Just connecting should work
        pass


def test_websocket_status_sends_json(client):
    """Test that WebSocket sends valid JSON messages."""
    with client.websocket_connect("/ws/status") as websocket:
        # Receive one message
        data = websocket.receive_json()
        assert isinstance(data, dict)
        assert "feeds" in data
        assert "timestamp" in data
        assert isinstance(data["feeds"], list)


def test_websocket_status_feed_structure(client):
    """Test that feed status has expected fields."""
    with client.websocket_connect("/ws/status") as websocket:
        data = websocket.receive_json()
        feeds = data["feeds"]
        if feeds:  # If there are feeds
            feed = feeds[0]
            required_fields = ["feed_id", "alarm_active", "caution_active", "people_count", "danger_count", "caution_count"]
            for field in required_fields:
                assert field in feed