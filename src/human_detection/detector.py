from ultralytics import YOLO
import cv2
import numpy as np
import os
import torch
from .config import MODEL_PATH, CONFIDENCE_THRESHOLD, CLASS_ID_PERSON, INFERENCE_IMGSZ


def _try_load_tensorrt(model_path):
    """Try to export and load a TensorRT engine for maximum inference speed.

    The .engine file is cached on disk after the first export.
    Returns (YOLO_model, is_tensorrt) tuple.
    """
    if not torch.cuda.is_available():
        return YOLO(model_path), False

    engine_path = os.path.splitext(model_path)[0] + ".engine"

    # If engine already exists, load it directly
    if os.path.exists(engine_path):
        print(f"[YOLO] Loading cached TensorRT engine: {engine_path}")
        try:
            model = YOLO(engine_path)
            print("[YOLO] TensorRT engine loaded successfully")
            return model, True
        except Exception as e:
            print(f"[YOLO] Failed to load cached engine ({e}), re-exporting...")
            os.remove(engine_path)

    # Export to TensorRT (one-time cost)
    print("[YOLO] Exporting TensorRT engine (one-time, may take a few minutes)...")
    try:
        base_model = YOLO(model_path)
        base_model.export(
            format="engine",
            imgsz=INFERENCE_IMGSZ,
            half=True,
            batch=4,
            dynamic=True,
        )
        # Load the exported engine
        model = YOLO(engine_path)
        print(f"[YOLO] TensorRT engine exported and loaded: {engine_path}")
        return model, True
    except Exception as e:
        print(f"[YOLO] TensorRT export failed: {e}")
        print("[YOLO] Falling back to PyTorch with FP16")
        return YOLO(model_path), False


class HumanDetector:
    def __init__(self):
        print(f"Loading YOLO model: {MODEL_PATH}...")

        self._use_half = torch.cuda.is_available()
        self.model, self._is_tensorrt = _try_load_tensorrt(MODEL_PATH)

        if self._is_tensorrt:
            # TensorRT engine handles precision internally
            self._use_half = False
            print("[YOLO] Using TensorRT engine (FP16 baked in)")
        elif self._use_half:
            print("[YOLO] CUDA detected — using FP16 (half precision)")
        else:
            print("[YOLO] Running on CPU")

    def _extract_person_masks(self, result, frame_shape):
        """Extract person binary masks from a single YOLO result."""
        extracted_masks = []
        h, w = frame_shape[:2]

        if result.masks is not None:
            for i, box in enumerate(result.boxes):
                if int(box.cls[0]) == CLASS_ID_PERSON:
                    mask_raw = result.masks.data[i].cpu().numpy()
                    mask_resized = cv2.resize(mask_raw, (w, h))
                    mask_binary = (mask_resized > 0.5).astype(np.uint8)
                    extracted_masks.append(mask_binary)

        return extracted_masks

    def get_masks(self, frame):
        """
        Input: A single video frame (image).
        Output: A list of binary masks (numpy arrays), one for each human found.
        """
        results = self.model(
            frame, conf=CONFIDENCE_THRESHOLD, imgsz=INFERENCE_IMGSZ,
            verbose=False, half=self._use_half,
        )
        return self._extract_person_masks(results[0], frame.shape)

    def get_masks_batch(self, frames):
        """
        Run YOLO on multiple frames in a single batch call.

        Input: List of video frames (images).
        Output: List of lists — one list of binary masks per frame.
        """
        if not frames:
            return []

        results = self.model(
            frames, conf=CONFIDENCE_THRESHOLD, imgsz=INFERENCE_IMGSZ,
            verbose=False, half=self._use_half,
        )

        batch_masks = []
        for result, frame in zip(results, frames):
            batch_masks.append(self._extract_person_masks(result, frame.shape))

        return batch_masks
