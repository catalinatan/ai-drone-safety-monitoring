"""Integration tests for GET /health."""

from __future__ import annotations


def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_shape(client):
    data = client.get("/health").json()
    assert data["status"] == "healthy"
    assert "feeds_count" in data
    assert "drone_api_connected" in data


def test_health_feeds_count_matches_registered(client):
    data = client.get("/health").json()
    assert data["feeds_count"] == 2


def test_health_drone_disconnected(client):
    data = client.get("/health").json()
    assert data["drone_api_connected"] is False
