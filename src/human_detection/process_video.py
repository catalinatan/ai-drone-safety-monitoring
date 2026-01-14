import cv2
import os
import numpy as np
from .detector import HumanDetector


def process_video(video_path, output_path="output_result.mp4"):
    # 1. Check if input file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)

    # Get video properties (to ensure output matches input)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 2. Setup Video Writer (MP4 format)
    # 'mp4v' is a standard codec that works on Mac and Linux
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    detector = HumanDetector()
    print(f"Processing: {video_path} -> Saving to: {output_path}")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # 3. Get the binary masks from your detector
        masks = detector.get_masks(frame)

        # 4. Create the Overlay
        if masks:
            # Create a black image of the same size
            mask_overlay = np.zeros_like(frame)

            # Combine all human masks into one
            combined_mask = np.zeros_like(masks[0])
            for mask in masks:
                combined_mask = np.maximum(combined_mask, mask)

            # Color the mask area GREEN (B, G, R format in OpenCV)
            # Where combined_mask is 1, set color to (0, 255, 0)
            mask_overlay[combined_mask == 1] = (0, 255, 0)

            # Blend original frame with mask overlay
            # 0.6 = 60% Original Video, 0.4 = 40% Green Mask
            frame = cv2.addWeighted(frame, 0.6, mask_overlay, 0.4, 0)

        # 5. Write the processed frame to the video file
        out.write(frame)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print("Done! Video saved.")


if __name__ == "__main__":
    # Remember to point this to your actual file in the 'data' folder
    INPUT_VIDEO = "data/test_video.mp4"
    process_video(INPUT_VIDEO)