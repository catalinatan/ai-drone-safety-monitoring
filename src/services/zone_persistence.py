"""
Zone persistence — load/save zone definitions to/from disk.

Pure file I/O; no threading, no business logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from src.core.models import Zone


def load_zones(path: str | Path) -> Dict[str, List[dict]]:
    """
    Load zones from a JSON file.

    Returns a dict mapping feed_id → list of zone dicts.
    Returns an empty dict if the file doesn't exist or is corrupt.
    """
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return data
    except Exception as e:
        print(f"[ZonePersistence] Error loading {p}: {e}")
        return {}


def save_zones(path: str | Path, feed_id: str, zones: List[Zone]) -> bool:
    """
    Persist zones for one feed to a JSON file.

    Reads the existing file first so other feeds' zones are preserved.
    Returns True on success, False on error.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    all_zones = load_zones(p)
    all_zones[feed_id] = [z.model_dump() for z in zones]

    try:
        with open(p, "w") as f:
            json.dump(all_zones, f, indent=2)
        return True
    except Exception as e:
        print(f"[ZonePersistence] Error saving {p}: {e}")
        return False
