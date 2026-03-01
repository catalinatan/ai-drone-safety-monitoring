# Model Settings
# Variant: "sim" or "real"
# Base model: "yolo11n-seg" or "yolo11s-seg"
HUMAN_MODEL_VARIANT = "sim"
HUMAN_BASE_MODEL = "yolo11n-seg"

import os
# Resolves to e.g. runs/segment/runs/segment/human_detection_sim_yolo11n-seg/weights/best.pt
_variant_suffix = f"_{HUMAN_MODEL_VARIANT}" if HUMAN_MODEL_VARIANT != "combined" else ""
_finetuned_path = f"runs/segment/runs/segment/human_detection{_variant_suffix}_{HUMAN_BASE_MODEL}/weights/best.pt"
MODEL_PATH = _finetuned_path if os.path.exists(_finetuned_path) else f"{HUMAN_BASE_MODEL}.pt"
CONFIDENCE_THRESHOLD = 0.25    # Lowered from 0.5 for small/distant humans in water
CLASS_ID_PERSON = 0            # YOLO specific ID for 'person'

# Inference Settings
INFERENCE_IMGSZ = 1280         # Increased from default 640 for better small object detection

# --- SAFETY ZONE CONFIGURATION ---
# How much of the PERSON'S body must be inside the zone to trigger an alarm?
# 0.1 = 10% (Triggers if they just step in)
# 0.5 = 50% (Triggers if half their body is in)
DANGER_ZONE_OVERLAP_THRESHOLD = 0.5

# Optional: Minimum pixels to consider a person "real" before checking overlap
# This prevents tiny specks of noise from triggering alarms.
MIN_PERSON_AREA_PIXELS = 2