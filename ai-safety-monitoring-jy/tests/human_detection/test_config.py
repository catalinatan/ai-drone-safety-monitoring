import pytest
from src.human_detection import config


def test_config_values():
    """Ensure critical configuration constants exist and are valid."""
    assert isinstance(config.MODEL_PATH, str)
    assert len(config.MODEL_PATH) > 0

    assert isinstance(config.CONFIDENCE_THRESHOLD, float)
    assert 0.0 <= config.CONFIDENCE_THRESHOLD <= 1.0

    # Ensure the class ID for 'person' is an integer (usually 0 in COCO)
    assert isinstance(config.CLASS_ID_PERSON, int)