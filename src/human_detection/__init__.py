# Expose the main detector class
from .detector import HumanDetector

# Expose configuration constants (optional, but useful if main.py needs to see settings)
from .config import MODEL_PATH, CONFIDENCE_THRESHOLD