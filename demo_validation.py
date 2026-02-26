"""
Demo script to validate the complete safety monitoring system on real-world videos.

Combines:
  - Human detection (fine-tuned model)
  - Scene-specific hazard zone segmentation (railway/bridge/ship)
  - Zone overlap detection with alarm visualization

Usage:
    # Process all videos in data/test_videos/ship/
    python demo_validation.py --scene ship

    # Process all videos in data/test_videos/railway/
    python demo_validation.py --scene railway

    # Process specific video file
    python demo_validation.py --scene bridge --video data/my_video.mp4

    # Save outputs to file
    python demo_validation.py --scene ship --output-dir outputs/
"""
import argparse
import cv2
import numpy as np
import time
from pathlib import Path
from ultralytics import YOLO

from src.human_detection.detector import HumanDetector
from src.human_detection.check_overlap import check_danger_zone_overlap
from src.human_detection.config import CONFIDENCE_THRESHOLD, INFERENCE_IMGSZ


# Model paths for scene segmentation
SCENE_MODELS = {
    "railway": "runs/segment/railway_hazard/weights/best.pt",
    "bridge": "runs/segment/bridge_hazard/weights/best.pt",
    "ship": "runs/segment/ship_hazard/weights/best.pt",
}


def get_hazard_mask_from_result(result, frame_shape):
    """
    Extract hazard zone mask from YOLO segmentation result.
    Combines all detected hazard classes into a single binary mask.
    """
    h, w = frame_shape[:2]
    hazard_mask = np.zeros((h, w), dtype=np.uint8)

    if result.masks is None:
        return hazard_mask

    for i, mask_data in enumerate(result.masks.data):
        # Extract and resize mask
        mask_raw = mask_data.cpu().numpy()
        mask_resized = cv2.resize(mask_raw, (w, h))
        mask_binary = (mask_resized > 0.5).astype(np.uint8)

        # Combine into single hazard mask
        hazard_mask = np.maximum(hazard_mask, mask_binary)

    return hazard_mask


def draw_overlays(frame, person_masks, hazard_mask, alarm_active, danger_masks):
    """
    Draw all visualization overlays on the frame.

    - Hazard zones: Red semi-transparent overlay
    - Detected humans: Green contours when safe, Red when in danger zone
    - Alarm: Large red "ALARM" text at top
    """
    overlay = frame.copy()

    # 1. Draw hazard zone (red tint)
    if hazard_mask is not None and np.any(hazard_mask):
        red_tint = np.zeros_like(frame)
        red_tint[:, :] = (0, 0, 200)
        overlay = np.where(hazard_mask[..., None] == 1,
                          cv2.addWeighted(overlay, 0.7, red_tint, 0.3, 0),
                          overlay)

    # 2. Draw person masks - differentiate between safe (green) and danger (red)
    if person_masks:
        # Helper function to check if two masks are the same
        def masks_match(mask1, mask2):
            return np.array_equal(mask1, mask2)

        # Draw each person with appropriate color
        for person_mask in person_masks:
            # Check if this person is in danger
            is_in_danger = any(masks_match(person_mask, danger_mask) for danger_mask in danger_masks)

            contours, _ = cv2.findContours(person_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Color based on individual danger status
            color = (0, 0, 255) if is_in_danger else (0, 255, 0)  # Red if in danger, green if safe
            thickness = 3 if is_in_danger else 2

            cv2.drawContours(overlay, contours, -1, color, thickness)

            # If in danger, draw bounding box and warning
            if is_in_danger:
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 3)

                    # Warning indicator
                    cv2.rectangle(overlay, (x, y - 40), (x + 40, y), (0, 0, 255), -1)
                    cv2.putText(overlay, "!", (x + 10, y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        # Add semi-transparent overlay for each person
        for person_mask in person_masks:
            is_in_danger = any(masks_match(person_mask, danger_mask) for danger_mask in danger_masks)
            tint_color = (0, 0, 255) if is_in_danger else (0, 255, 0)

            tint = np.zeros_like(frame)
            tint[:, :] = tint_color
            overlay = np.where(person_mask[..., None] == 1,
                              cv2.addWeighted(overlay, 0.85, tint, 0.15, 0),
                              overlay)

    # 3. Alarm banner
    if alarm_active:
        # Large red banner at top
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (0, 0, 255), -1)
        cv2.putText(overlay, "ALARM: PERSON IN DANGER ZONE", (20, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)

    return overlay


def process_video(video_path, scene_type, output_path=None, show=True, human_detector=None, hazard_model=None):
    """
    Process a single video with human detection and scene-specific hazard segmentation.
    """
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Error: Could not open video {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n  Processing: {video_path.name}")
    print(f"    Resolution: {width}x{height} @ {fps} FPS | Frames: {total_frames}")

    # Setup output writer if requested
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"    Saving to: {output_path}")

    frame_count = 0
    alarm_frames = 0

    # Calculate target frame time for real-time playback
    target_frame_time = 1.0 / fps if fps > 0 else 0.033  # Fallback to ~30fps

    # Hazard zone segmentation settings
    hazard_mask = None
    last_hazard_frame = None
    HAZARD_REFRESH_FRAMES = 60  # Re-segment every 60 frames for ship scenes

    while True:
        frame_start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        # Run hazard zone detection:
        # - First frame: always detect
        # - Ship scene: re-detect every 60 frames (water movement, ship orientation changes)
        # - Bridge/Railway: only once (truly static)
        should_detect_hazard = (
            frame_count == 0 or
            (scene_type == "ship" and last_hazard_frame is not None and
             frame_count - last_hazard_frame >= HAZARD_REFRESH_FRAMES)
        )

        if should_detect_hazard:
            if frame_count == 0:
                print(f"    Detecting hazard zones (first frame)...")
            else:
                print(f"    Re-detecting hazard zones (ship scene, every {HAZARD_REFRESH_FRAMES} frames)...")
            hazard_results = hazard_model(frame, conf=CONFIDENCE_THRESHOLD, imgsz=INFERENCE_IMGSZ, verbose=False)
            hazard_mask = get_hazard_mask_from_result(hazard_results[0], frame.shape)
            last_hazard_frame = frame_count

        # Run human detection on every frame
        person_masks = human_detector.get_masks(frame)

        # Check for danger
        alarm_active, danger_masks = check_danger_zone_overlap(person_masks, hazard_mask)
        if alarm_active:
            alarm_frames += 1

        # Visualize
        display_frame = draw_overlays(frame, person_masks, hazard_mask, alarm_active, danger_masks)

        # Add stats overlay (including real-time FPS)
        processing_time = time.time() - frame_start_time
        actual_fps = 1.0 / processing_time if processing_time > 0 else 0
        stats_text = f"Frame {frame_count}/{total_frames} | People: {len(person_masks)} | Alarm: {'YES' if alarm_active else 'NO'} | FPS: {actual_fps:.1f}"
        cv2.putText(display_frame, stats_text, (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show/save
        if show:
            cv2.imshow(f"Demo - {scene_type.title()} Scene - {video_path.name}", display_frame)

            # Calculate delay to match original video FPS
            elapsed = time.time() - frame_start_time
            delay_needed = target_frame_time - elapsed

            # Wait at least 1ms, or the calculated delay for real-time playback
            wait_time = max(1, int(delay_needed * 1000)) if delay_needed > 0 else 1

            if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                print("    Stopped by user")
                break

        if out:
            out.write(display_frame)

        frame_count += 1

    # Cleanup
    cap.release()
    if out:
        out.release()

    # Summary
    print(f"    ✓ Processed {frame_count} frames | Alarms: {alarm_frames} ({100*alarm_frames/max(frame_count,1):.1f}%)")


def run_demo(scene_type, video_path=None, output_dir=None, show=True):
    """
    Main entry point - processes video(s) for a given scene type.
    """
    # Validate scene type
    if scene_type not in SCENE_MODELS:
        print(f"Error: Unknown scene type '{scene_type}'. Choose from: {list(SCENE_MODELS.keys())}")
        return

    scene_model_path = SCENE_MODELS[scene_type]
    if not Path(scene_model_path).exists():
        print(f"Error: Scene model not found at {scene_model_path}")
        print(f"Train the {scene_type} model first using: python -m src.scene-segmentation.train --dataset {scene_type}")
        return

    # Load models once (reuse for multiple videos)
    print(f"\n{'='*60}")
    print(f"Loading models for {scene_type.upper()} scene validation")
    print(f"{'='*60}")
    print(f"  Human detection: fine-tuned model")
    print(f"  Scene hazard: {scene_model_path}")
    print(f"  Confidence: {CONFIDENCE_THRESHOLD} | Image size: {INFERENCE_IMGSZ}")

    human_detector = HumanDetector()
    hazard_model = YOLO(scene_model_path)

    # Determine which videos to process
    if video_path:
        # Single video specified
        videos = [Path(video_path)]
        if not videos[0].exists():
            print(f"Error: Video not found at {video_path}")
            return
    else:
        # Find all videos in data/test_videos/{scene_type}/
        video_dir = Path(f"data/test_videos/{scene_type}")
        if not video_dir.exists():
            print(f"Error: Directory not found: {video_dir}")
            print(f"Create the directory and add test videos, or use --video to specify a single file")
            return

        videos = sorted(list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi")) + list(video_dir.glob("*.mov")))
        if not videos:
            print(f"No videos found in {video_dir}")
            return

    print(f"\nFound {len(videos)} video(s) to process")
    print(f"{'='*60}\n")

    # Create output directory if needed
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None

    # Process each video
    for i, video in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {video.name}")

        # Determine output file path
        if output_path:
            out_file = output_path / f"{video.stem}_annotated.mp4"
        else:
            out_file = None

        process_video(video, scene_type, out_file, show, human_detector, hazard_model)

    if show:
        cv2.destroyAllWindows()

    print(f"\n{'='*60}")
    print(f"Demo complete! Processed {len(videos)} video(s)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate safety monitoring system on real-world videos")
    parser.add_argument("--scene", type=str, required=True,
                       choices=["railway", "bridge", "ship"],
                       help="Scene type (determines which hazard model to use)")
    parser.add_argument("--video", type=str, default=None,
                       help="Path to specific video file (optional - defaults to all videos in data/test_videos/{scene}/)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Directory to save annotated output videos (optional)")
    parser.add_argument("--no-show", action="store_true",
                       help="Don't display video in real-time")

    args = parser.parse_args()

    run_demo(args.scene, args.video, args.output_dir, show=not args.no_show)
