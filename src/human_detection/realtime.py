import cv2
import time
import numpy as np
from .detector import HumanDetector


def run_realtime_detection(source=0):
    """
    source:
      - 0 (default) for your local webcam
      - "rtsp://username:password@ip_address..." for an IP Camera/CCTV
      - "data/video.mp4" if you want to test the loop with a file
    """

    # 1. Initialize Source
    print(f"Connecting to source: {source}...")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # 2. Load Model
    detector = HumanDetector()
    print("Model loaded. Starting stream... (Press 'q' to quit)")

    # Performance tracking
    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or failed to read frame.")
            break

        # ---------------------------------------------------------
        # STEP A: Get the Masks (The "Output")
        # ---------------------------------------------------------
        masks = detector.get_masks(frame)

        # ---------------------------------------------------------
        # STEP B: The Hand-off (Where logic happens)
        # ---------------------------------------------------------
        if masks:
            # Combine individual masks into one 'Total Human Area'
            combined_mask = np.zeros_like(masks[0])
            for mask in masks:
                combined_mask = np.maximum(combined_mask, mask)

            # [FUTURE INTEGRATION POINT]
            # This is exactly where you will send 'combined_mask' to
            # your teammate's forbidden zone checker.
            # e.g., is_alarm = check_zone_overlap(combined_mask)
        else:
            combined_mask = None

        # ---------------------------------------------------------
        # STEP C: Visualization (For your verification)
        # ---------------------------------------------------------
        # Calculate FPS to ensure we are running "Real Time"
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # Draw the visual overlay
        if combined_mask is not None:
            # Create green overlay
            overlay = np.zeros_like(frame)
            overlay[combined_mask == 1] = (0, 255, 0)  # Green

            # Blend it
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Draw FPS counter on screen
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the live window ("Pretty version")
        cv2.imshow("Live Human Detection", frame)

        # CHECK 2: Show the "Raw" version (what the code uses)
        if combined_mask is not None:
            # Multiply by 255 because 0 and 1 are too dark to see
            cv2.imshow("DEBUG: Raw Integration Data", combined_mask * 255)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Use 0 for Webcam, or replace with CCTV URL
    run_realtime_detection(0)

    # testing with test video
    # run_realtime_detection("data/test_video.mp4")