"""
Depth estimator wrapper — loads and runs Lite-Mono depth estimation.

Provides a simple interface to estimate depth maps from frames.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from src.detection.depth_estimator import load_lite_mono_model, run_lite_mono_inference


class DepthEstimator:
    """Wraps Lite-Mono depth estimation model for inference."""

    def __init__(
        self,
        encoder_path: str = "models/depth_estimation/lite-mono-small_640x192/encoder.pth",
        decoder_path: str = "models/depth_estimation/lite-mono-small_640x192/depth.pth",
    ) -> None:
        """
        Load depth estimation model.

        Parameters
        ----------
        encoder_path : str
            Path to encoder weights.
        decoder_path : str
            Path to decoder weights.

        Raises
        ------
        FileNotFoundError
            If model files don't exist.
        RuntimeError
            If model loading fails.
        """
        enc_p = Path(encoder_path)
        dec_p = Path(decoder_path)

        if not enc_p.exists():
            raise FileNotFoundError(f"Encoder not found: {encoder_path}")
        if not dec_p.exists():
            raise FileNotFoundError(f"Decoder not found: {decoder_path}")

        print(f"[DepthEstimator] Loading from {encoder_path} and {decoder_path}")
        self._model = load_lite_mono_model(str(enc_p), str(dec_p))
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)
        print(f"[DepthEstimator] Running on {self._device}")

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """
        Estimate depth from a single frame.

        Parameters
        ----------
        frame : np.ndarray
            RGB frame (H, W, 3) uint8.

        Returns
        -------
        np.ndarray
            Inverse-disparity depth map (H, W) float32.
            Higher values = farther from camera.
            Use with compute_scale_factor() and get_metric_depth_at_pixel()
            to obtain metric depth in metres.
        """
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame
        return run_lite_mono_inference(self._model, frame)

    def get_depth_at_pixel(self, depth_map: np.ndarray, x: int, y: int) -> float:
        """
        Get raw inverse-disparity value at a specific pixel.

        For metric depth in metres, use get_metric_depth_at_pixel() instead.
        """
        h, w = depth_map.shape
        x = max(0, min(int(x), w - 1))
        y = max(0, min(int(y), h - 1))
        return float(depth_map[y, x])

    @staticmethod
    def get_metric_depth_at_pixel(
        depth_map: np.ndarray, x: int, y: int, scale_factor: float,
    ) -> float:
        """
        Get metric depth (metres) at a pixel.

        Parameters
        ----------
        depth_map : np.ndarray
            Inverse-disparity map from estimate() — values proportional to depth.
        x, y : int
            Pixel coordinates (clamped to bounds).
        scale_factor : float
            Per-frame scale factor from projection backend's compute_scale_factor().

        Returns
        -------
        float
            Metric depth in metres at the given pixel.
        """
        h, w = depth_map.shape
        x = max(0, min(int(x), w - 1))
        y = max(0, min(int(y), h - 1))
        return float(scale_factor * depth_map[y, x])
