"""
Lite-Mono depth estimator.

Moved here from src/cctv_monitoring/depth_estimation_utils.py during Phase 6.
The Lite-Mono vendored code remains at src/cctv_monitoring/Lite-Mono/.
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Resolve Lite-Mono vendor path (in models/lite-mono/)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LITEMONO_ROOT = str(_PROJECT_ROOT / "models" / "lite-mono")
if _LITEMONO_ROOT not in sys.path:
    sys.path.insert(0, _LITEMONO_ROOT)

from networks.depth_encoder import LiteMono  # noqa: E402 (vendor import)

# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


class _Conv3x3(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_refl: bool = True):
        super().__init__()
        self.pad = nn.ReflectionPad2d(1) if use_refl else nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        return self.conv(self.pad(x))


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = _Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        return self.nonlin(self.conv(x))


def _upsample(x):
    return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


class DepthDecoder(nn.Module):
    def __init__(
        self,
        num_ch_enc,
        scales=range(4),
        num_output_channels: int = 1,
        use_skips: bool = True,
        num_ch_dec=None,
    ):
        super().__init__()
        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.scales = scales
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 16, 24, 40, 64] if num_ch_dec is None else num_ch_dec)

        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = _ConvBlock(num_ch_in, num_ch_out)

            num_ch_in = self.num_ch_dec[i]
            skip_ch = 0
            if use_skips:
                if i == 4 and len(self.num_ch_enc) >= 2:
                    skip_ch = self.num_ch_enc[-2]
                elif i == 3 and len(self.num_ch_enc) >= 3:
                    skip_ch = self.num_ch_enc[-3]
            num_ch_in += skip_ch
            self.convs[("upconv", i, 1)] = _ConvBlock(num_ch_in, self.num_ch_dec[i])

        for s in self.scales:
            self.convs[("dispconv", s)] = _Conv3x3(self.num_ch_dec[s], num_output_channels)

        self.decoder_modules = nn.ModuleList(self.convs.values())

    def forward(self, input_features):
        self.outputs = {}
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [_upsample(x)]
            if self.use_skips:
                if i == 4 and len(input_features) >= 2:
                    x += [input_features[-2]]
                elif i == 3 and len(input_features) >= 3:
                    x += [input_features[-3]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = torch.sigmoid(self.convs[("dispconv", i)](x))
        return self.outputs


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------


class LiteMonoDepthSystem(nn.Module):
    def __init__(self, encoder, decoder, skip_adapters=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.skip_adapters = skip_adapters

    def forward(self, x):
        features = self.encoder.forward_features(x)
        if self.skip_adapters is not None:
            features = [
                self.skip_adapters[i](f) if i < len(self.skip_adapters) else f
                for i, f in enumerate(features)
            ]
        outputs = self.decoder(features)
        return outputs[("disp", 0)]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_lite_mono_model(
    encoder_path: str,
    decoder_path: str,
) -> LiteMonoDepthSystem:
    """Load encoder + decoder weights and return an assembled, eval-mode model."""
    print("[DepthEstimator] Analyzing weight files...")

    dec_dict = torch.load(decoder_path, map_location="cpu")
    if "model_state_dict" in dec_dict:
        dec_dict = dec_dict["model_state_dict"]

    enc_last_dim = dec_dict["decoder.0.conv.conv.weight"].shape[1]
    layer1_in = dec_dict["decoder.1.conv.conv.weight"].shape[1]
    layer0_out = dec_dict["decoder.0.conv.conv.weight"].shape[0]
    skip_at_level4 = layer1_in - layer0_out

    layer3_in = dec_dict["decoder.3.conv.conv.weight"].shape[1]
    layer2_out = (
        dec_dict["decoder.2.conv.conv.weight"].shape[0]
        if "decoder.2.conv.conv.weight" in dec_dict
        else 40
    )
    skip_at_level3 = layer3_in - layer2_out

    enc_dims = [skip_at_level3, skip_at_level4, enc_last_dim]

    if enc_dims == [48, 80, 128]:
        model_variant = "lite-mono-small"
    elif enc_dims == [32, 64, 128]:
        model_variant = "lite-mono-tiny"
    elif enc_dims == [64, 128, 224]:
        model_variant = "lite-mono-8m"
    else:
        model_variant = "lite-mono-small"

    print(f"[DepthEstimator] Creating {model_variant} encoder with dims {enc_dims}")

    encoder = LiteMono(model=model_variant, width=640, height=192)
    enc_dict = torch.load(encoder_path, map_location="cpu")
    if "model_state_dict" in enc_dict:
        enc_dict = enc_dict["model_state_dict"]
    clean_enc = {k: v for k, v in enc_dict.items() if "head" not in k and "total" not in k}
    encoder.load_state_dict(clean_enc, strict=False)

    decoder = DepthDecoder(
        num_ch_enc=enc_dims,
        scales=range(4),
        num_ch_dec=[16, 16, 24, 40, 64],
    )

    new_dec = {}
    for k, v in dec_dict.items():
        if "total_ops" in k or "total_params" in k:
            continue
        if k.startswith("decoder."):
            k = k.replace("decoder.", "decoder_modules.")
        new_dec[k] = v
    decoder.load_state_dict(new_dec, strict=False)

    model = LiteMonoDepthSystem(encoder, decoder, skip_adapters=None)
    model.eval()
    print("[DepthEstimator] Model assembled successfully.")
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def run_lite_mono_inference(model: LiteMonoDepthSystem, img_rgb: np.ndarray) -> np.ndarray:
    """
    Run depth estimation on an RGB image.

    Returns an inverse-disparity map (float32) matching the input image's
    spatial dimensions. Values are proportional to true depth: higher values
    mean farther from the camera. Use with a per-frame scale factor to
    recover metric depth in metres.
    """
    input_h, input_w = 192, 640
    img_resized = cv2.resize(img_rgb, (input_w, input_h))
    img_tensor = torch.from_numpy(img_resized).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

    # Move input to same device as model weights
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        pred_disp = model(img_tensor)

    pred_disp = F.interpolate(
        pred_disp,
        size=(img_rgb.shape[0], img_rgb.shape[1]),
        mode="bilinear",
        align_corners=False,
    )
    disp_map = pred_disp.squeeze().cpu().numpy()
    return 1.0 / (disp_map + 1e-6)
