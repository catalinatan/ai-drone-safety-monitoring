"""
Auto-Segmentation Module
========================

Uses trained YOLO segmentation models to automatically detect hazard zones
in camera frames and convert them to polygon-based Zone objects compatible
with the existing zone management system.

Supported scene types: ship, railway, bridge
"""

import os
import time
import cv2
import numpy as np
from typing import Dict, List, Optional

from src.backend.config import AUTO_SEG_CONFIDENCE, AUTO_SEG_SIMPLIFY_EPSILON


class SceneSegmenter:
    """Loads YOLO segmentation models and converts masks to polygon zones."""

    def __init__(self, model_paths: Dict[str, str], confidence: float = AUTO_SEG_CONFIDENCE):
        from ultralytics import YOLO

        self.models: Dict[str, object] = {}
        self.confidence = confidence

        for scene_type, path in model_paths.items():
            if os.path.exists(path):
                print(f"[AUTO-SEG] Loading {scene_type} model: {path}")
                self.models[scene_type] = YOLO(path)
                print(f"[AUTO-SEG] {scene_type} model loaded")
            else:
                print(f"[AUTO-SEG] WARNING: Model not found for {scene_type}: {path}")

    def segment_frame(self, frame: np.ndarray, scene_type: str) -> List[dict]:
        """Run segmentation on a frame and return Zone-compatible dicts.

        Args:
            frame: BGR or RGB numpy array (H, W, 3)
            scene_type: One of "ship", "railway", "bridge"

        Returns:
            List of zone dicts: [{"id": str, "level": "red", "points": [{"x": float, "y": float}]}]
        """
        if scene_type not in self.models:
            print(f"[AUTO-SEG] No model loaded for scene type: {scene_type}")
            return []

        model = self.models[scene_type]
        results = model(frame, conf=self.confidence, verbose=False)[0]

        if results.masks is None:
            return []

        zones = []
        h, w = frame.shape[:2]
        timestamp = int(time.time() * 1000)

        for i, mask_tensor in enumerate(results.masks.data):
            mask_raw = mask_tensor.cpu().numpy()

            # Resize mask to match frame dimensions
            mask_resized = cv2.resize(mask_raw, (w, h), interpolation=cv2.INTER_LINEAR)

            # Binary threshold
            mask_binary = (mask_resized > 0.5).astype(np.uint8)

            # Find contours (outer contours only)
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for j, contour in enumerate(contours):
                # Filter out tiny contours (noise)
                if cv2.contourArea(contour) < 100:
                    continue

                # Simplify polygon to reduce point count
                approx = cv2.approxPolyDP(contour, AUTO_SEG_SIMPLIFY_EPSILON, True)

                # Need at least 3 points for a valid polygon
                if len(approx) < 3:
                    continue

                # Convert pixel coordinates to percentage coordinates (0-100)
                points = []
                for pt in approx:
                    px, py = pt[0]
                    points.append({
                        "x": round(float(px) / w * 100, 4),
                        "y": round(float(py) / h * 100, 4),
                    })

                zone = {
                    "id": f"auto-seg-{scene_type}-{i}-{j}-{timestamp}",
                    "level": "red",
                    "points": points,
                }
                zones.append(zone)

        return zones
