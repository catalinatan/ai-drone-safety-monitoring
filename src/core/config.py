"""
Application configuration loader.

Reads config/default.yaml at startup, then applies environment variable
overrides using UPPER_SNAKE_CASE keys (e.g. DETECTION_FPS=15 overrides
detection.fps). Fails fast with a clear error if required paths are missing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def _find_project_root() -> Path:
    start = Path(__file__).resolve()
    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return start.parents[-1]


PROJECT_ROOT = _find_project_root()
_DEFAULT_YAML = PROJECT_ROOT / "config" / "default.yaml"
_FEEDS_YAML = PROJECT_ROOT / "config" / "feeds.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins)."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _apply_env_overrides(cfg: dict) -> dict:
    """
    Apply environment variable overrides.
    Flat env var DETECTION_FPS maps to cfg['detection']['fps'].
    Only numeric and string values are overridden.
    """
    env_map: dict[str, tuple[list[str], type]] = {
        "BACKEND_PORT":           (["server", "backend_port"], int),
        "DRONE_API_PORT":         (["server", "drone_api_port"], int),
        "FRONTEND_PORT":          (["server", "frontend_port"], int),
        "DETECTION_FPS":          (["detection", "fps"], int),
        "DETECTION_WARMUP_FRAMES":(["detection", "warmup_frames"], int),
        "ALARM_COOLDOWN":         (["zones", "alarm_cooldown_seconds"], float),
        "AUTO_SEG_INTERVAL":      (["auto_segmentation", "interval_seconds"], float),
        "AUTO_SEG_CONFIDENCE":    (["auto_segmentation", "confidence"], float),
        "AUTO_SEG_SIMPLIFY_EPSILON": (["auto_segmentation", "simplify_epsilon"], float),
        "AUTO_SEG_MIN_CONTOUR_AREA": (["auto_segmentation", "min_contour_area"], float),
        "FRAME_CAPTURE_FPS":      (["streaming", "capture_fps"], int),
        "STREAM_FPS":             (["streaming", "stream_fps"], int),
        "DRONE_API_URL":          (["drone", "api_url"], str),
        "DRONE_API_TIMEOUT":      (["drone", "api_timeout"], int),
        "SAFE_Z_ALTITUDE":        (["drone", "safe_altitude"], float),
        "CCTV_FOLLOW_TARGET":     (["follow_mode", "target"], str),
        "CCTV_HOVER_DRONES":      (["follow_mode", "hover_drones"], lambda v: v == "1"),
        "CCTV_HOVER_ALTITUDE":    (["follow_mode", "hover_altitude"], float),
        "DETECTION_MODEL_PATH":   (["detection", "model_path"], str),
    }
    for env_key, (path, cast) in env_map.items():
        val = os.getenv(env_key)
        if val is not None:
            node = cfg
            for part in path[:-1]:
                node = node.setdefault(part, {})
            node[path[-1]] = cast(val)
    return cfg


def load_config(yaml_path: Path = _DEFAULT_YAML) -> dict[str, Any]:
    """Load default.yaml and apply env overrides. Returns a plain dict."""
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f) or {}
    return _apply_env_overrides(cfg)


def load_feeds_config(yaml_path: Path = _FEEDS_YAML) -> dict[str, Any]:
    """Load feeds.yaml. Returns the feeds dict keyed by feed_id."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f) or {}
    return data.get("feeds", {})


# Module-level singletons — loaded once at import time.
_config: dict[str, Any] | None = None
_feeds: dict[str, Any] | None = None


def get_config() -> dict[str, Any]:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def get_feeds_config() -> dict[str, Any]:
    global _feeds
    if _feeds is None:
        _feeds = load_feeds_config()
    return _feeds


def reset_feeds_config() -> None:
    """Clear the feeds config cache so next call reloads from disk."""
    global _feeds
    _feeds = None
