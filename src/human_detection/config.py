# Model Settings
# MODEL_PATH = "runs/segment/runs/segment/human_detection/weights/best.pt"
MODEL_PATH = "yolov8n-seg.pt"
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