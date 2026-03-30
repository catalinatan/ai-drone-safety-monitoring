"""
Detection pipeline — orchestrates frame → detect → check zones → alarm.

Pure orchestration layer: no I/O, no threading. All dependencies are injected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol

import numpy as np

from src.core.alarm import AlarmState
from src.core.zone_manager import ZoneManager


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """Result of processing one frame through the pipeline."""
    people_count: int = 0
    alarm_active: bool = False     # RED zone intrusion
    caution_active: bool = False   # YELLOW zone intrusion
    danger_count: int = 0          # People in RED zones
    caution_count: int = 0         # People in YELLOW zones
    danger_masks: List[np.ndarray] = field(default_factory=list)
    caution_masks: List[np.ndarray] = field(default_factory=list)
    alarm_fired: bool = False      # True if alarm.trigger() returned True this cycle


# ---------------------------------------------------------------------------
# Detector protocol — any object with get_masks() satisfies this
# ---------------------------------------------------------------------------

class DetectorProtocol(Protocol):
    def get_masks(self, frame: np.ndarray) -> List[np.ndarray]:
        ...


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class DetectionPipeline:
    """
    Stateless (per-call) orchestrator for the detection flow.

    Parameters
    ----------
    detector : DetectorProtocol
        Object with a get_masks(frame) → List[ndarray] method.
    zone_manager : ZoneManager
        Holds current zone masks.
    alarm : AlarmState
        Manages alarm cooldown state.
    warmup_frames : int
        Minimum number of frames that must have been captured before
        alarms can fire (prevents false positives on startup).
    """

    def __init__(
        self,
        detector: DetectorProtocol,
        zone_manager: ZoneManager,
        alarm: AlarmState,
        warmup_frames: int = 20,
    ) -> None:
        self._detector = detector
        self._zone_manager = zone_manager
        self._alarm = alarm
        self._warmup_frames = warmup_frames
        self._frame_count: int = 0

    def process_frame(self, frame: np.ndarray) -> DetectionResult:
        """
        Run the full detection pipeline on one frame.

        Returns a DetectionResult with all counts and alarm flags populated.
        """
        self._frame_count += 1
        result = DetectionResult()

        # Skip alarms during warmup
        warmed_up = self._frame_count >= self._warmup_frames
        if not warmed_up:
            return result

        # No zones → nothing to check; skip expensive YOLO inference
        if self._zone_manager.red_mask is None and self._zone_manager.yellow_mask is None:
            return result

        # --- Human detection ---
        person_masks = self._detector.get_masks(frame)
        result.people_count = len(person_masks)

        if not person_masks:
            self._alarm.clear()
            return result

        # --- RED zone check (triggers alarm + drone dispatch) ---
        is_alarm, danger_masks = self._zone_manager.check_red(person_masks)
        result.alarm_active = is_alarm
        result.danger_count = len(danger_masks)
        result.danger_masks = danger_masks

        if is_alarm:
            result.alarm_fired = self._alarm.trigger()
        else:
            self._alarm.clear()

        # --- YELLOW zone check (caution only, no drone) ---
        is_caution, caution_masks = self._zone_manager.check_yellow(person_masks)
        result.caution_active = is_caution
        result.caution_count = len(caution_masks)
        result.caution_masks = caution_masks

        return result

    @property
    def frame_count(self) -> int:
        return self._frame_count
