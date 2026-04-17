"""Unit tests for zone_persistence — load/save round-trip."""

import json
import pytest
from pathlib import Path

from src.core.models import Point, Zone
from src.services.zone_persistence import load_zones, save_zones


def make_zone(level="red") -> Zone:
    return Zone(
        id="z1", level=level,
        points=[Point(x=0, y=0), Point(x=100, y=0), Point(x=100, y=100), Point(x=0, y=100)],
    )


class TestLoadZones:
    def test_missing_file_returns_empty(self, tmp_path):
        result = load_zones(tmp_path / "nonexistent.json")
        assert result == {}

    def test_corrupt_file_returns_empty(self, tmp_path):
        p = tmp_path / "zones.json"
        p.write_text("not valid json")
        result = load_zones(p)
        assert result == {}

    def test_valid_file_returns_data(self, tmp_path):
        p = tmp_path / "zones.json"
        data = {"feed-1": [{"id": "z1", "level": "red", "points": []}]}
        p.write_text(json.dumps(data))
        result = load_zones(p)
        assert "feed-1" in result


class TestSaveZones:
    def test_creates_file_if_missing(self, tmp_path):
        p = tmp_path / "sub" / "zones.json"
        zones = [make_zone("red")]
        assert save_zones(p, "feed-1", zones) is True
        assert p.exists()

    def test_round_trip(self, tmp_path):
        p = tmp_path / "zones.json"
        zones = [make_zone("red"), make_zone("yellow")]
        save_zones(p, "feed-1", zones)
        loaded = load_zones(p)
        assert "feed-1" in loaded
        assert len(loaded["feed-1"]) == 2
        assert loaded["feed-1"][0]["level"] == "red"
        assert loaded["feed-1"][1]["level"] == "yellow"

    def test_preserves_other_feeds(self, tmp_path):
        p = tmp_path / "zones.json"
        save_zones(p, "feed-1", [make_zone("red")])
        save_zones(p, "feed-2", [make_zone("yellow")])
        loaded = load_zones(p)
        assert "feed-1" in loaded
        assert "feed-2" in loaded

    def test_overwrites_same_feed(self, tmp_path):
        p = tmp_path / "zones.json"
        save_zones(p, "feed-1", [make_zone("red")])
        save_zones(p, "feed-1", [])  # clear zones
        loaded = load_zones(p)
        assert loaded["feed-1"] == []
