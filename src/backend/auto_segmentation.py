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

from src.backend.config import (
    AUTO_SEG_CONFIDENCE,
    AUTO_SEG_SIMPLIFY_EPSILON,
    AUTO_SEG_MIN_CONTOUR_AREA,
)


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

    def segment_frame(self, frame: np.ndarray, scene_type: str, confidence: Optional[float] = None) -> List[dict]:
        """Run segmentation on a frame and return Zone-compatible dicts.

        Args:
            frame: BGR or RGB numpy array (H, W, 3)
            scene_type: One of "ship", "railway", "bridge"
            confidence: Override confidence threshold (uses default if None)

        Returns:
            List of zone dicts: [{"id": str, "level": "red", "points": [{"x": float, "y": float}]}]
        """
        if scene_type not in self.models:
            print(f"[AUTO-SEG] No model loaded for scene type: {scene_type}")
            return []

        model = self.models[scene_type]
        conf = confidence if confidence is not None else self.confidence
        print(f"[AUTO-SEG] segment_frame: scene_type='{scene_type}', confidence={conf}, frame_shape={frame.shape}")

        results = model(frame, conf=conf, verbose=False, save=False)[0]

        # Log raw model output
        num_detections = len(results.boxes) if results.boxes is not None else 0
        num_masks = len(results.masks.data) if results.masks is not None else 0
        print(f"[AUTO-SEG] Raw model output: {num_detections} detections, {num_masks} masks")
        if results.boxes is not None and len(results.boxes) > 0:
            class_names = results.names if hasattr(results, 'names') else {}
            for i, box in enumerate(results.boxes):
                cls_id = int(box.cls[0]) if box.cls is not None else -1
                cls_name = class_names.get(cls_id, f"class_{cls_id}")
                box_conf = float(box.conf[0]) if box.conf is not None else 0
                print(f"[AUTO-SEG]   detection[{i}]: class='{cls_name}' (id={cls_id}), conf={box_conf:.3f}")

        if results.masks is None:
            print(f"[AUTO-SEG] No masks produced — returning empty")
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
            mask_pixel_count = int(mask_binary.sum())
            print(f"[AUTO-SEG]   mask[{i}]: raw_shape={mask_raw.shape}, resized={mask_resized.shape}, pixels={mask_pixel_count}")

            # Find contours (outer contours only)
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(f"[AUTO-SEG]   mask[{i}]: {len(contours)} contour(s) found")

            for j, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                # Filter out tiny contours (noise)
                if area < AUTO_SEG_MIN_CONTOUR_AREA:
                    print(f"[AUTO-SEG]   mask[{i}] contour[{j}]: area={area:.0f} < min={AUTO_SEG_MIN_CONTOUR_AREA} — SKIPPED")
                    continue

                # Simplify polygon to reduce point count
                approx = cv2.approxPolyDP(contour, AUTO_SEG_SIMPLIFY_EPSILON, True)

                # Need at least 3 points for a valid polygon
                if len(approx) < 3:
                    print(f"[AUTO-SEG]   mask[{i}] contour[{j}]: only {len(approx)} points after simplify — SKIPPED")
                    continue
                print(f"[AUTO-SEG]   mask[{i}] contour[{j}]: area={area:.0f}, points={len(approx)} — OK")

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

        print(f"[AUTO-SEG] segment_frame result: {len(zones)} zone(s) for scene_type='{scene_type}'")
        return zones
