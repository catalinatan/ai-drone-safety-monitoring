"""
Scene segmenter — YOLO-based hazard zone auto-segmentation.

Moved here from src/backend/auto_segmentation.py during Phase 6 refactor.
All configuration is now read from config/default.yaml via src.core.config.

Supported scene types: railway, ship, bridge
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

import cv2
import numpy as np


def _get_auto_seg_cfg() -> dict:
    from src.core.config import get_config

    return get_config().get("auto_segmentation", {})


class SceneSegmenter:
    """
    Loads YOLO segmentation models per scene type and converts masks to
    polygon-based Zone dicts compatible with the zone management system.

    Parameters
    ----------
    model_paths : dict
        Mapping from scene_type (str) to model weight path (str).
        If empty or None, reads paths from config/default.yaml.
    confidence : float, optional
        Detection confidence threshold. Defaults to config value.
    """

    def __init__(
        self,
        model_paths: Optional[Dict[str, str]] = None,
        confidence: Optional[float] = None,
    ):
        from ultralytics import YOLO

        cfg = _get_auto_seg_cfg()

        if model_paths is None:
            model_paths = cfg.get("models", {})

        self.confidence = confidence if confidence is not None else cfg.get("confidence", 0.5)
        self._simplify_epsilon = cfg.get("simplify_epsilon", 2.0)
        self._min_contour_area = cfg.get("min_contour_area", 40.0)

        self.models: Dict[str, object] = {}
        for scene_type, path in model_paths.items():
            if os.path.exists(path):
                print(f"[SceneSegmenter] Loading {scene_type} model: {path}")
                self.models[scene_type] = YOLO(path)
                print(f"[SceneSegmenter] {scene_type} model loaded")
            else:
                print(f"[SceneSegmenter] WARNING: Model not found for {scene_type}: {path}")

    def segment_frame(
        self,
        frame: np.ndarray,
        scene_type: str,
        confidence: Optional[float] = None,
    ) -> List[dict]:
        """
        Run segmentation on a frame and return Zone-compatible dicts.

        Parameters
        ----------
        frame : np.ndarray
            BGR or RGB image (H, W, 3).
        scene_type : str
            One of "railway", "ship", "bridge".
        confidence : float, optional
            Override confidence threshold for this call.

        Returns
        -------
        List of zone dicts::

            [{"id": str, "level": "red", "points": [{"x": float, "y": float}]}]
        """
        if scene_type not in self.models:
            print(f"[SceneSegmenter] No model loaded for scene type: {scene_type}")
            return []

        model = self.models[scene_type]
        conf = confidence if confidence is not None else self.confidence
        print(f"[SceneSegmenter] segment_frame: scene_type='{scene_type}', confidence={conf}")

        results = model(frame, conf=conf, imgsz=640, verbose=False, save=False)[0]

        num_detections = len(results.boxes) if results.boxes is not None else 0
        num_masks = len(results.masks.data) if results.masks is not None else 0
        print(f"[SceneSegmenter] {num_detections} detections, {num_masks} masks")

        if results.masks is None:
            return []

        zones = []
        h, w = frame.shape[:2]
        timestamp = int(time.time() * 1000)

        for i, mask_tensor in enumerate(results.masks.data):
            mask_raw = mask_tensor.cpu().numpy()
            mask_resized = cv2.resize(mask_raw, (w, h), interpolation=cv2.INTER_LINEAR)
            mask_binary = (mask_resized > 0.5).astype(np.uint8)

            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for j, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area < self._min_contour_area:
                    continue

                approx = cv2.approxPolyDP(contour, self._simplify_epsilon, True)
                if len(approx) < 3:
                    continue

                points = [
                    {
                        "x": round(float(pt[0][0]) / w * 100, 4),
                        "y": round(float(pt[0][1]) / h * 100, 4),
                    }
                    for pt in approx
                ]

                zones.append(
                    {
                        "id": f"auto-seg-{scene_type}-{i}-{j}-{timestamp}",
                        "level": "red",
                        "points": points,
                    }
                )

        print(f"[SceneSegmenter] {len(zones)} zone(s) returned for '{scene_type}'")
        return zones
