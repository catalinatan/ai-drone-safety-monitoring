# Model Settings
MODEL_PATH = "yolov8m-seg.pt"  # Use 'm' (medium) or 'x' (large) for better mask accuracy
CONFIDENCE_THRESHOLD = 0.5     # Only detect humans if 50% sure
CLASS_ID_PERSON = 0            # YOLO specific ID for 'person'

# --- SAFETY ZONE CONFIGURATION ---
# How much of the PERSON'S body must be inside the zone to trigger an alarm?
# 0.1 = 10% (Triggers if they just step in)
# 0.5 = 50% (Triggers if half their body is in)
DANGER_ZONE_OVERLAP_THRESHOLD = 0.5

# Optional: Minimum pixels to consider a person "real" before checking overlap
# This prevents tiny specks of noise from triggering alarms.
MIN_PERSON_AREA_PIXELS = 100