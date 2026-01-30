"""Unit tests for backend server logic — all external deps mocked."""

import json
import numpy as np
import pytest
from unittest.mock import patch, MagicMock, mock_open

from src.backend.server import (
    Point, Zone, ZonesUpdateRequest, TargetCoordinate, DetectionStatus,
    FeedState, FeedManager,
    load_zones_from_file, save_zones_to_file, ensure_data_dir,
)


# ============================================================================
# Pydantic Models
# ============================================================================

class TestPydanticModels:

    def test_point_creation(self):
        p = Point(x=50.0, y=25.0)
        assert p.x == 50.0
        assert p.y == 25.0

    def test_zone_creation(self):
        z = Zone(
            id="zone-1",
            level="red",
            points=[Point(x=0, y=0), Point(x=100, y=0), Point(x=100, y=100)],
        )
        assert z.level == "red"
        assert len(z.points) == 3

    def test_zones_update_request(self):
        req = ZonesUpdateRequest(zones=[
            Zone(id="z1", level="yellow", points=[
                Point(x=10, y=10), Point(x=20, y=10), Point(x=20, y=20),
            ]),
        ])
        assert len(req.zones) == 1

    def test_target_coordinate(self):
        t = TargetCoordinate(x=1.0, y=2.0, z=-10.0)
        assert t.z == -10.0

    def test_detection_status_defaults(self):
        status = DetectionStatus(
            feed_id="cctv-1",
            alarm_active=False,
            caution_active=False,
            people_count=0,
            danger_count=0,
            caution_count=0,
        )
        assert status.target_coordinates is None
        assert status.last_detection_time is None


# ============================================================================
# Zone Persistence
# ============================================================================

class TestZonePersistence:

    @patch("src.backend.server.os.path.exists", return_value=False)
    @patch("src.backend.server.os.makedirs")
    def test_ensure_data_dir_creates_directory(self, mock_makedirs, mock_exists):
        ensure_data_dir()
        mock_makedirs.assert_called_once()

    @patch("src.backend.server.os.path.exists", return_value=True)
    def test_load_zones_no_file(self, mock_exists):
        # First call: data dir exists. Second call: zones file does not.
        mock_exists.side_effect = [True, False]
        result = load_zones_from_file()
        assert result == {}

    @patch("src.backend.server.os.path.exists", return_value=True)
    @patch("builtins.open", mock_open(read_data='{"cctv-1": []}'))
    def test_load_zones_from_file(self, mock_exists):
        result = load_zones_from_file()
        assert "cctv-1" in result

    @patch("src.backend.server.load_zones_from_file", return_value={})
    @patch("src.backend.server.os.path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open)
    def test_save_zones_to_file(self, mock_file, mock_exists, mock_load):
        save_zones_to_file("cctv-1", [{"id": "z1", "level": "red", "points": []}])
        mock_file.assert_called()
        written = mock_file().write.call_args_list
        assert len(written) > 0  # json.dump wrote something


# ============================================================================
# FeedState
# ============================================================================

class TestFeedState:

    def test_defaults(self):
        state = FeedState(
            feed_id="test",
            camera_name="0",
            vehicle_name="Drone2",
            name="Test Cam",
            location="Test",
        )
        assert state.alarm_active is False
        assert state.caution_active is False
        assert state.people_count == 0
        assert state.red_zone_mask is None
        assert state.yellow_zone_mask is None
        assert state.target_coordinates is None
        assert state.lock is not None


# ============================================================================
# FeedManager — Zone Updates
# ============================================================================

class TestFeedManagerZones:

    def _make_manager_with_feed(self):
        """Create a FeedManager with one test feed, no AirSim."""
        manager = FeedManager()
        manager.feeds["test-feed"] = FeedState(
            feed_id="test-feed",
            camera_name="0",
            vehicle_name="Drone2",
            name="Test",
            location="Test",
        )
        return manager

    @patch("src.backend.server.save_zones_to_file")
    def test_update_zones_creates_red_mask(self, mock_save):
        manager = self._make_manager_with_feed()
        zones = [
            Zone(id="z1", level="red", points=[
                Point(x=0, y=0), Point(x=50, y=0),
                Point(x=50, y=50), Point(x=0, y=50),
            ]),
        ]
        manager.update_zones("test-feed", zones, 100, 100)
        feed = manager.feeds["test-feed"]
        assert feed.red_zone_mask is not None
        assert feed.red_zone_mask.shape == (100, 100)
        assert feed.yellow_zone_mask is None
        mock_save.assert_called_once()

    @patch("src.backend.server.save_zones_to_file")
    def test_update_zones_creates_yellow_mask(self, mock_save):
        manager = self._make_manager_with_feed()
        zones = [
            Zone(id="z1", level="yellow", points=[
                Point(x=0, y=0), Point(x=50, y=0),
                Point(x=50, y=50), Point(x=0, y=50),
            ]),
        ]
        manager.update_zones("test-feed", zones, 100, 100)
        feed = manager.feeds["test-feed"]
        assert feed.red_zone_mask is None
        assert feed.yellow_zone_mask is not None

    @patch("src.backend.server.save_zones_to_file")
    def test_update_zones_separates_red_and_yellow(self, mock_save):
        manager = self._make_manager_with_feed()
        zones = [
            Zone(id="z1", level="red", points=[
                Point(x=0, y=0), Point(x=30, y=0),
                Point(x=30, y=30), Point(x=0, y=30),
            ]),
            Zone(id="z2", level="yellow", points=[
                Point(x=60, y=60), Point(x=90, y=60),
                Point(x=90, y=90), Point(x=60, y=90),
            ]),
        ]
        manager.update_zones("test-feed", zones, 200, 200)
        feed = manager.feeds["test-feed"]
        assert feed.red_zone_mask is not None
        assert feed.yellow_zone_mask is not None
        # Red mask should have nonzero pixels in top-left
        assert np.sum(feed.red_zone_mask[:60, :60]) > 0
        # Yellow mask should have nonzero pixels in bottom-right
        assert np.sum(feed.yellow_zone_mask[120:, 120:]) > 0

    def test_update_zones_unknown_feed_raises(self):
        manager = FeedManager()
        with pytest.raises(ValueError, match="not found"):
            manager.update_zones("nonexistent", [], 100, 100)


# ============================================================================
# Detection Status
# ============================================================================

class TestDetectionStatus:

    def test_get_status_no_alarm(self):
        manager = FeedManager()
        feed = FeedState(
            feed_id="test",
            camera_name="0",
            vehicle_name="Drone2",
            name="Test",
            location="Test",
        )
        status = manager._get_status(feed)
        assert status.alarm_active is False
        assert status.target_coordinates is None

    def test_get_status_with_coordinates(self):
        manager = FeedManager()
        feed = FeedState(
            feed_id="test",
            camera_name="0",
            vehicle_name="Drone2",
            name="Test",
            location="Test",
        )
        feed.alarm_active = True
        feed.target_coordinates = (10.0, 20.0, -5.0)
        status = manager._get_status(feed)
        assert status.alarm_active is True
        assert status.target_coordinates is not None
        assert status.target_coordinates.x == 10.0
        assert status.target_coordinates.y == 20.0
        assert status.target_coordinates.z == -5.0
