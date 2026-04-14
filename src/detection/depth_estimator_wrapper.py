"""
Depth estimator wrapper — loads and runs Lite-Mono depth estimation.

Provides a simple interface to estimate depth maps from frames.
"""

from __future__ import annotations

from pathlib import Path

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
            Normalized depth map (H, W) float32, values in [0, 1].
            0 = closest, 1 = farthest.
        """
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame
        return run_lite_mono_inference(self._model, frame)

    def get_depth_at_pixel(self, depth_map: np.ndarray, x: int, y: int) -> float:
        """
        Get depth value at a specific pixel.

        Parameters
        ----------
        depth_map : np.ndarray
            Depth map from estimate().
        x, y : int
            Pixel coordinates (clipped to bounds).

        Returns
        -------
        float
            Depth value at pixel, normalized to [0, 1].
        """
        h, w = depth_map.shape
        x = max(0, min(int(x), w - 1))
        y = max(0, min(int(y), h - 1))
        return float(depth_map[y, x])
