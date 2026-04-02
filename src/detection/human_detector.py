"""
Human detector — YOLO segmentation-based person detection.

Moved here from src/human_detection/detector.py during Phase 6 refactor.
All configuration is now read from config/default.yaml via src.core.config.
"""

from __future__ import annotations

import os
from typing import List, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# YOLO COCO person class ID (fixed, not user-configurable)
CLASS_ID_PERSON = 0


def _get_cfg() -> dict:
    from src.core.config import get_config
    return get_config().get("detection", {})


def _try_load_tensorrt(model_path: str, inference_imgsz: int):
    """Try to export and load a TensorRT engine for maximum inference speed.

    The .engine file is cached on disk after the first export.
    Returns (YOLO_model, is_tensorrt) tuple.
    """
    if not torch.cuda.is_available():
        return YOLO(model_path), False

    engine_path = os.path.splitext(model_path)[0] + ".engine"

    if os.path.exists(engine_path):
        print(f"[YOLO] Loading cached TensorRT engine: {engine_path}")
        try:
            model = YOLO(engine_path)
            print("[YOLO] TensorRT engine loaded successfully")
            return model, True
        except Exception as e:
            print(f"[YOLO] Failed to load cached engine ({e}), re-exporting...")
            os.remove(engine_path)

    print("[YOLO] Exporting TensorRT engine (one-time, may take a few minutes)...")
    try:
        base_model = YOLO(model_path)
        base_model.export(
            format="engine",
            imgsz=inference_imgsz,
            half=True,
            batch=8,
            dynamic=True,
        )
        model = YOLO(engine_path)
        print(f"[YOLO] TensorRT engine exported and loaded: {engine_path}")
        return model, True
    except Exception as e:
        print(f"[YOLO] TensorRT export failed: {e}")
        print("[YOLO] Falling back to PyTorch with FP16")
        return YOLO(model_path), False


class HumanDetector:
    """
    YOLO-based human segmentation detector.

    Parameters
    ----------
    model_path : str, optional
        Path to the YOLO model weights. Defaults to config detection.model_path.
    confidence_threshold : float, optional
        Detection confidence threshold. Defaults to config detection.confidence_threshold.
    inference_imgsz : int, optional
        Inference image size. Defaults to config detection.inference_imgsz.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        inference_imgsz: Optional[int] = None,
    ):
        cfg = _get_cfg()
        self._model_path = model_path or cfg.get("model_path", "yolo11n-seg.pt")
        self._confidence = confidence_threshold if confidence_threshold is not None else cfg.get("confidence_threshold", 0.25)
        self._imgsz = inference_imgsz or cfg.get("inference_imgsz", 1280)

        print(f"[HumanDetector] Loading model: {self._model_path}")
        self._use_half = torch.cuda.is_available()
        self.model, self._is_tensorrt = _try_load_tensorrt(self._model_path, self._imgsz)

        if self._is_tensorrt:
            self._use_half = False
            print("[HumanDetector] Using TensorRT engine (FP16 baked in)")
        elif self._use_half:
            print("[HumanDetector] CUDA detected — using FP16 (half precision)")
        else:
            print("[HumanDetector] Running on CPU")

        self._warmup()

    def _warmup(self) -> None:
        """Run dummy inference to trigger CUDA kernel compilation / TensorRT plan selection.

        Without this, the first real inference is significantly slower
        (500ms–2s extra) due to JIT compilation and memory allocation.
        """
        if not torch.cuda.is_available():
            return
        print("[HumanDetector] CUDA warmup inference...")
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.model(dummy, conf=self._confidence, imgsz=self._imgsz,
                   verbose=False, half=self._use_half, save=False)
        print("[HumanDetector] Warmup complete")

    def _extract_person_masks(
        self, result, frame_shape: tuple
    ) -> List[np.ndarray]:
        """Extract binary person masks from a single YOLO result.

        Optimised: filters on GPU, batch-resizes with F.interpolate (GPU),
        thresholds on GPU, then does a single CPU transfer.
        """
        if result.masks is None or len(result.boxes) == 0:
            return []

        h, w = frame_shape[:2]

        # Filter person indices on GPU (avoids per-mask CPU roundtrips)
        person_idx = [i for i, box in enumerate(result.boxes)
                      if int(box.cls[0]) == CLASS_ID_PERSON]
        if not person_idx:
            return []

        import torch.nn.functional as F

        masks_tensor = result.masks.data[person_idx].float()       # (N, mh, mw) GPU, ensure float
        masks_resized = F.interpolate(
            masks_tensor.unsqueeze(1),                             # (N, 1, mh, mw)
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)                                               # (N, h, w)
        masks_np = (masks_resized > 0.5).byte().cpu().numpy()     # single transfer
        return [masks_np[i] for i in range(masks_np.shape[0])]

    def get_masks(self, frame: np.ndarray) -> List[np.ndarray]:
        """
        Run detection on a single frame.

        Returns a list of binary masks (one per detected person).
        """
        results = self.model(
            frame,
            conf=self._confidence,
            imgsz=self._imgsz,
            verbose=False,
            half=self._use_half,
            save=False,
        )
        return self._extract_person_masks(results[0], frame.shape)

    def get_masks_batch(self, frames: List[np.ndarray]) -> List[List[np.ndarray]]:
        """
        Run detection on multiple frames in a single batch call.

        Returns a list of lists — one list of masks per frame.
        """
        if not frames:
            return []

        results = self.model(
            frames,
            conf=self._confidence,
            imgsz=self._imgsz,
            verbose=False,
            half=self._use_half,
            save=False,
        )

        return [
            self._extract_person_masks(result, frame.shape)
            for result, frame in zip(results, frames)
        ]
