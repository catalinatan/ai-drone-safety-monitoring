import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import numpy as np
import cv2
from collections import OrderedDict

# --- 1. SETUP PATHS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
# Use the networks folder which has the COMPLETE implementation
litemono_root = os.path.join(current_dir, "Lite-Mono")
if litemono_root not in sys.path:
    sys.path.insert(0, litemono_root)

# Import from networks.depth_encoder (the complete implementation)
from networks.depth_encoder import LiteMono

# --- 2. ROBUST DECODER ---
class Conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()
        self.pad = nn.ReflectionPad2d(1) if use_refl else nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)
    def forward(self, x): return self.conv(self.pad(x))

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)
    def forward(self, x): return self.nonlin(self.conv(x))

def upsample(x):
    return nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, num_ch_dec=None):
        super(DepthDecoder, self).__init__()
        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.scales = scales
        self.num_ch_enc = num_ch_enc
        
        # Structure derived from your error logs
        if num_ch_dec is None:
            self.num_ch_dec = np.array([16, 16, 24, 40, 64])
        else:
            self.num_ch_dec = np.array(num_ch_dec)

        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # 1. Input Channels
            if i == 4:
                num_ch_in = self.num_ch_enc[-1]
            else:
                num_ch_in = self.num_ch_dec[i + 1]
            
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # 2. Skip Connection Channels
            num_ch_in = self.num_ch_dec[i]
            
            # --- SKIP CONNECTION LOGIC ---
            # The encoder returns features from low-res to high-res
            # Decoder starts from the deepest (last) feature and works upward
            # At level i=4, add second-to-last encoder feature as skip
            # At level i=3, add third-to-last encoder feature as skip
            # i=2 has no skip (intentional)
            skip_ch_to_add = 0
            if self.use_skips:
                if i == 4 and len(self.num_ch_enc) >= 2:
                    skip_ch_to_add = self.num_ch_enc[-2]  # Second-to-last feature
                elif i == 3 and len(self.num_ch_enc) >= 3:
                    skip_ch_to_add = self.num_ch_enc[-3]  # Third-to-last feature
                # i=2 is skipped intentionally (no skip connection)
            
            num_ch_in += skip_ch_to_add
            
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder_modules = nn.ModuleList(self.convs.values())

    def forward(self, input_features):
        self.outputs = {}
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            
            # Match the __init__ logic exactly - use relative indexing from the back
            if self.use_skips:
                if i == 4 and len(input_features) >= 2:
                    x += [input_features[-2]]  # Second-to-last feature
                elif i == 3 and len(input_features) >= 3:
                    x += [input_features[-3]]  # Third-to-last feature
                # i=2 skipped (no skip connection)
            
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = torch.sigmoid(self.convs[("dispconv", i)](x))
        return self.outputs

# --- 3. ADAPTIVE SYSTEM ---
class LiteMonoDepthSystem(nn.Module):
    def __init__(self, encoder, decoder, skip_adapters=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.skip_adapters = skip_adapters 

    def forward(self, x):
        features = self.encoder.forward_features(x)

        # Apply adapters if they exist
        if self.skip_adapters is not None:
            new_features = []
            for i, feat in enumerate(features):
                if i < len(self.skip_adapters):
                    new_features.append(self.skip_adapters[i](feat))
                else:
                    new_features.append(feat)
            features = new_features

        outputs = self.decoder(features)
        return outputs[("disp", 0)]

# --- 4. SMART LOADER ---
def load_lite_mono_model(encoder_path, decoder_path):
    print(f"[LOADER] Analyzing Weights Files...")

    # A. DETECT ENCODER DIMENSIONS FROM DECODER
    # The decoder skip connections tell us what encoder dimensions were used
    dec_dict = torch.load(decoder_path, map_location='cpu')
    if 'model_state_dict' in dec_dict: dec_dict = dec_dict['model_state_dict']

    # Analyze decoder layer shapes to determine encoder dimensions
    # decoder.0 expects the LAST encoder feature directly
    # decoder.1 expects: output_from_decoder.0 + skip_connection
    # decoder.1.conv.conv.weight shape: [out_channels, in_channels, 3, 3]

    # Get the last encoder dimension from decoder.0
    if 'decoder.0.conv.conv.weight' in dec_dict:
        enc_last_dim = dec_dict['decoder.0.conv.conv.weight'].shape[1]
        print(f"[LOADER] Detected encoder last dimension: {enc_last_dim}")

    # Get skip at level 4 from decoder.1
    if 'decoder.1.conv.conv.weight' in dec_dict:
        layer1_in = dec_dict['decoder.1.conv.conv.weight'].shape[1]
        layer0_out = dec_dict['decoder.0.conv.conv.weight'].shape[0]
        skip_at_level4 = layer1_in - layer0_out
        print(f"[LOADER] Detected skip at level 4 (encoder[2]): {skip_at_level4}")

    # Get skip at level 3 from decoder.3
    if 'decoder.3.conv.conv.weight' in dec_dict:
        layer3_in = dec_dict['decoder.3.conv.conv.weight'].shape[1]
        layer2_out = dec_dict['decoder.2.conv.conv.weight'].shape[0] if 'decoder.2.conv.conv.weight' in dec_dict else 40
        skip_at_level3 = layer3_in - layer2_out
        print(f"[LOADER] Detected skip at level 3 (encoder[1]): {skip_at_level3}")

    # Encoder dimensions: [first, second, last]
    # The decoder uses: encoder[-3]=48, encoder[-2]=80, encoder[-1]=128
    # With 3 features: [48, 80, 128]
    # - encoder[0] / encoder[-3] = 48 (used as skip at level 3)
    # - encoder[1] / encoder[-2] = 80 (used as skip at level 4)
    # - encoder[2] / encoder[-1] = 128 (used as initial input)
    enc_dims = [skip_at_level3, skip_at_level4, enc_last_dim]  # [48, 80, 128]
    print(f"[LOADER] Using encoder dimensions: {enc_dims}")

    # B. LOAD ENCODER
    enc_dict = torch.load(encoder_path, map_location='cpu')
    if 'model_state_dict' in enc_dict: enc_dict = enc_dict['model_state_dict']

    # Determine model variant based on dimensions
    # lite-mono-small: [48, 80, 128]
    # lite-mono: [48, 80, 128] (same dims, different depth)
    # lite-mono-tiny: [32, 64, 128]
    # lite-mono-8m: [64, 128, 224]
    if enc_dims == [48, 80, 128]:
        model_variant = "lite-mono-small"
    elif enc_dims == [32, 64, 128]:
        model_variant = "lite-mono-tiny"
    elif enc_dims == [64, 128, 224]:
        model_variant = "lite-mono-8m"
    else:
        model_variant = "lite-mono-small"  # default

    print(f"[LOADER] Creating {model_variant} encoder with dims {enc_dims}")

    encoder = LiteMono(model=model_variant, width=640, height=192)

    # The encoder has its own internal dimensions - we just load the weights
    clean_enc = {k: v for k, v in enc_dict.items() if "head" not in k and "total" not in k}
    missing, unexpected = encoder.load_state_dict(clean_enc, strict=False)

    if missing:
        print(f"[LOADER] Warning: Missing encoder keys: {len(missing)}")
    if unexpected:
        print(f"[LOADER] Warning: Unexpected encoder keys: {len(unexpected)}")

    # C. CREATE DECODER (with detected encoder dimensions)
    custom_dec_channels = [16, 16, 24, 40, 64]

    print(f"[LOADER] Creating decoder with encoder dims: {enc_dims}")

    decoder = DepthDecoder(
        num_ch_enc=enc_dims,
        scales=range(4),
        num_ch_dec=custom_dec_channels
    )

    # Prepare decoder state dict
    new_dec = {}
    for k, v in dec_dict.items():
        if "total_ops" in k or "total_params" in k:
            continue
        if k.startswith("decoder."):
            k = k.replace("decoder.", "decoder_modules.")
        new_dec[k] = v

    # Load decoder weights
    missing_keys, unexpected_keys = decoder.load_state_dict(new_dec, strict=False)

    if missing_keys:
        print(f"[LOADER] Warning: Missing decoder keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"[LOADER] Warning: Unexpected decoder keys: {len(unexpected_keys)}")

    # D. COMBINE
    model = LiteMonoDepthSystem(encoder, decoder, skip_adapters=None)
    model.eval()
    print("[LOADER] System Assembled Successfully.")
    return model

# --- 5. INFERENCE ---
def run_lite_mono_inference(model, img_rgb):
    input_h, input_w = 192, 640
    img_resized = cv2.resize(img_rgb, (input_w, input_h))
    img_tensor = torch.from_numpy(img_resized).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    
    with torch.no_grad():
        pred_disp = model(img_tensor)
    
    pred_disp = F.interpolate(pred_disp, size=(img_rgb.shape[0], img_rgb.shape[1]), mode="bilinear", align_corners=False)
    depth_map = pred_disp.squeeze().cpu().numpy()
    depth_min, depth_max = depth_map.min(), depth_map.max()
    depth_norm = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)
    return depth_norm