"""Integration tests for video routes: snapshots and MJPEG streams."""

from __future__ import annotations


def test_get_feed_snapshot_known_feed(client):
    """Test GET /feeds/{feed_id}/snapshot returns JPEG for known feed."""
    resp = client.get("/feeds/cam-1/snapshot")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    # Should have some content
    assert len(resp.content) > 0


def test_get_feed_snapshot_unknown_feed(client):
    """Test GET /feeds/{feed_id}/snapshot returns 404 for unknown feed."""
    resp = client.get("/feeds/nonexistent/snapshot")
    assert resp.status_code == 404


# def test_video_feed_known_feed(client):
#     """Test GET /video_feed/{feed_id} returns streaming response for known feed."""
#     # StreamingResponse is hard to test without consuming; just check it doesn't 404
#     resp = client.get("/video_feed/cam-1", timeout=1)
#     assert resp.status_code == 200
#     assert "multipart/x-mixed-replace" in resp.headers["content-type"]
#     assert "boundary=frame" in resp.headers["content-type"]


# def test_video_feed_unknown_feed(client):
#     """Test GET /video_feed/{feed_id} returns 404 for unknown feed."""
#     resp = client.get("/video_feed/nonexistent")
#     assert resp.status_code == 404