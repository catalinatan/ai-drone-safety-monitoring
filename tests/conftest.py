"""Shared test fixtures used across all test modules."""

import numpy as np
import pytest


@pytest.fixture
def dummy_frame():
    """A 480x640 RGB frame (numpy array), simulating an AirSim camera image."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_zone_dicts():
    """Zone definitions as plain dicts (the format received over the API)."""
    return [
        {
            "id": "zone-red-1",
            "level": "red",
            "points": [
                {"x": 10.0, "y": 10.0},
                {"x": 40.0, "y": 10.0},
                {"x": 40.0, "y": 40.0},
                {"x": 10.0, "y": 40.0},
            ],
        },
        {
            "id": "zone-yellow-1",
            "level": "yellow",
            "points": [
                {"x": 60.0, "y": 60.0},
                {"x": 90.0, "y": 60.0},
                {"x": 90.0, "y": 90.0},
                {"x": 60.0, "y": 90.0},
            ],
        },
        {
            "id": "zone-green-1",
            "level": "green",
            "points": [
                {"x": 50.0, "y": 0.0},
                {"x": 60.0, "y": 0.0},
                {"x": 60.0, "y": 10.0},
                {"x": 50.0, "y": 10.0},
            ],
        },
    ]


@pytest.fixture
def binary_zone_mask():
    """A 480x640 binary mask with a filled rectangle in the top-left quadrant."""
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[48:192, 64:256] = 1  # ~10-40% region
    return mask


@pytest.fixture
def person_mask_in_zone():
    """A person-shaped binary mask that overlaps with binary_zone_mask."""
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[80:170, 100:200] = 1  # fully inside the zone
    return mask


@pytest.fixture
def person_mask_outside_zone():
    """A person-shaped binary mask that does NOT overlap with binary_zone_mask."""
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[350:450, 400:500] = 1  # bottom-right, outside zone
    return mask
