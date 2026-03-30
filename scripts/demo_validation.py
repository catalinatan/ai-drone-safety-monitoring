"""
Demo script to validate the complete safety monitoring system on real-world videos.

Combines:
  - Human detection (fine-tuned model)
  - Scene-specific hazard zone segmentation (railway/bridge/ship)
  - Zone overlap detection with alarm visualization

Has two modes: demo (default) and compare (latency benchmark).

=== DEMO MODE ===
Runs the full pipeline on video(s) with visualization. Press 'q' to stop.

Usage:
    python scripts/demo_validation.py --scene ship
    python scripts/demo_validation.py --scene bridge --video data/my_video.mp4
    python scripts/demo_validation.py --scene ship --output-dir demo_output/ship/ --no-show

Args (demo mode):
    --scene             railway | bridge | ship (required)
    --video             Path to a specific video file (optional)
    --output-dir        Directory to save annotated output videos (optional)
    --no-show           Don't display video in real-time

=== COMPARE MODE ===
Benchmarks multiple human detection models on the same video.

Usage:
    python scripts/demo_validation.py compare --video data/test_videos/ship/video.mp4
    python scripts/demo_validation.py compare --video data/test_videos/ship/video.mp4 --frames 200

Args (compare mode):
    --video             Path to test video (required)
    --models            Models to compare (default: yolo11n-seg yolo11s-seg)
    --variant           sim | real | combined (default: sim)
    --frames            Number of frames to benchmark (default: 100)
"""
import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.core.zone_manager import check_overlap
from src.core.config import get_config
from src.detection.human_detector import HumanDetector


# Confidence / image size from config (with fallbacks matching old defaults)
_cfg = get_config().get("detection", {})
CONFIDENCE_THRESHOLD = _cfg.get("confidence_threshold", 0.25)
INFERENCE_IMGSZ = _cfg.get("inference_imgsz", 1280)

# Model paths for scene segmentation
SCENE_MODELS = {
    "railway": "runs/segment/railway_hazard_yolo11s-seg/weights/best.pt",
    "bridge": "runs/segment/bridge_hazard_yolo11s-seg/weights/best.pt",
    "ship": "runs/segment/ship_hazard_yolo11s-seg/weights/best.pt",
}


def get_hazard_mask_from_result(result, frame_shape):
    """Extract hazard zone mask from YOLO segmentation result."""
    h, w = frame_shape[:2]
    hazard_mask = np.zeros((h, w), dtype=np.uint8)

    if result.masks is None:
        return hazard_mask

    for mask_data in result.masks.data:
        mask_raw = mask_data.cpu().numpy()
        mask_resized = cv2.resize(mask_raw, (w, h))
        mask_binary = (mask_resized > 0.5).astype(np.uint8)
        hazard_mask = np.maximum(hazard_mask, mask_binary)

    return hazard_mask


def draw_overlays(frame, person_masks, hazard_mask, alarm_active, danger_masks):
    """Draw visualization overlays on the frame."""
    overlay = frame.copy()

    # Hazard zone: red tint
    if hazard_mask is not None and np.any(hazard_mask):
        red_tint = np.zeros_like(frame)
        red_tint[:, :] = (0, 0, 200)
        overlay = np.where(
            hazard_mask[..., None] == 1,
            cv2.addWeighted(overlay, 0.7, red_tint, 0.3, 0),
            overlay,
        )

    # Person masks — green (safe) or red (danger)
    if person_masks:
        def masks_match(m1, m2):
            return np.array_equal(m1, m2)

        for person_mask in person_masks:
            is_in_danger = any(masks_match(person_mask, dm) for dm in danger_masks)
            contours, _ = cv2.findContours(person_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            color = (0, 0, 255) if is_in_danger else (0, 255, 0)
            thickness = 3 if is_in_danger else 2
            cv2.drawContours(overlay, contours, -1, color, thickness)

            if is_in_danger:
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 3)
                    cv2.rectangle(overlay, (x, y - 40), (x + 40, y), (0, 0, 255), -1)
                    cv2.putText(overlay, "!", (x + 10, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        for person_mask in person_masks:
            is_in_danger = any(masks_match(person_mask, dm) for dm in danger_masks)
            tint_color = (0, 0, 255) if is_in_danger else (0, 255, 0)
            tint = np.zeros_like(frame)
            tint[:, :] = tint_color
            overlay = np.where(
                person_mask[..., None] == 1,
                cv2.addWeighted(overlay, 0.85, tint, 0.15, 0),
                overlay,
            )

    # Alarm banner
    if alarm_active:
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (0, 0, 255), -1)
        cv2.putText(overlay, "ALARM: PERSON IN DANGER ZONE", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)

    return overlay


def process_video(video_path, scene_type, output_path=None, show=True, human_detector=None, hazard_model=None):
    """Process a single video with human detection and scene-specific hazard segmentation."""
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

    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"    Saving to: {output_path}")

    frame_count = 0
    alarm_frames = 0
    hazard_mask = None
    last_hazard_frame = None
    HAZARD_REFRESH_FRAMES = 60
    target_frame_time = 1.0 / fps if fps > 0 else 0.033

    while True:
        frame_start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        should_detect_hazard = (
            frame_count == 0
            or (
                scene_type == "ship"
                and last_hazard_frame is not None
                and frame_count - last_hazard_frame >= HAZARD_REFRESH_FRAMES
            )
        )

        if should_detect_hazard:
            hazard_results = hazard_model(
                frame, conf=CONFIDENCE_THRESHOLD, imgsz=INFERENCE_IMGSZ, verbose=False, save=False
            )
            hazard_mask = get_hazard_mask_from_result(hazard_results[0], frame.shape)
            last_hazard_frame = frame_count

        if frame_count == 0 and hazard_mask is not None:
            out_dir = Path("demo_output") / scene_type / "first_frames"
            out_dir.mkdir(parents=True, exist_ok=True)
            first_frame_viz = frame.copy()
            if np.any(hazard_mask):
                overlay = first_frame_viz.copy()
                overlay[hazard_mask > 0] = [0, 0, 255]
                cv2.addWeighted(overlay, 0.4, first_frame_viz, 0.6, 0, first_frame_viz)
                contours, _ = cv2.findContours(hazard_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(first_frame_viz, contours, -1, (0, 0, 255), 2)
            first_frame_path = out_dir / f"{video_path.stem}_first_frame_zones.jpg"
            cv2.imwrite(str(first_frame_path), first_frame_viz)

        person_masks = human_detector.get_masks(frame)
        alarm_active, danger_masks = check_overlap(person_masks, hazard_mask)
        if alarm_active:
            alarm_frames += 1

        display_frame = draw_overlays(frame, person_masks, hazard_mask, alarm_active, danger_masks)

        processing_time = time.time() - frame_start_time
        actual_fps = 1.0 / processing_time if processing_time > 0 else 0
        stats_text = (
            f"Frame {frame_count}/{total_frames} | People: {len(person_masks)} | "
            f"Alarm: {'YES' if alarm_active else 'NO'} | FPS: {actual_fps:.1f}"
        )
        cv2.putText(display_frame, stats_text, (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if show:
            cv2.imshow(f"Demo - {scene_type.title()} Scene - {video_path.name}", display_frame)
            elapsed = time.time() - frame_start_time
            delay_needed = target_frame_time - elapsed
            wait_time = max(1, int(delay_needed * 1000)) if delay_needed > 0 else 1
            if cv2.waitKey(wait_time) & 0xFF == ord("q"):
                print("    Stopped by user")
                break

        if out:
            out.write(display_frame)

        frame_count += 1

    cap.release()
    if out:
        out.release()

    print(f"    Processed {frame_count} frames | Alarms: {alarm_frames} "
          f"({100 * alarm_frames / max(frame_count, 1):.1f}%)")


def run_demo(scene_type, video_path=None, output_dir=None, show=True, model_variant=None, base_model=None):
    """Main entry point — processes video(s) for a given scene type."""
    if scene_type not in SCENE_MODELS:
        print(f"Error: Unknown scene type '{scene_type}'. Choose from: {list(SCENE_MODELS.keys())}")
        return

    scene_model_path = SCENE_MODELS[scene_type]
    if not Path(scene_model_path).exists():
        print(f"Error: Scene model not found at {scene_model_path}")
        print(f"Train the {scene_type} model first using:")
        print(f"  python scripts/train_scene_segmentation.py --dataset {scene_type}")
        return

    # Build model path for human detection
    detection_cfg = get_config().get("detection", {})
    default_model = detection_cfg.get("model_path", "yolo11n-seg.pt")

    if model_variant or base_model:
        variant = model_variant or "combined"
        bm = base_model or "yolo11n-seg"
        variant_suffix = f"_{variant}" if variant != "combined" else ""
        finetuned = f"runs/segment/human_detection{variant_suffix}_{bm}/weights/best.pt"
        model_path = finetuned if os.path.exists(finetuned) else f"{bm}.pt"
    else:
        model_path = default_model

    print(f"\n{'=' * 60}")
    print(f"Loading models for {scene_type.upper()} scene validation")
    print(f"{'=' * 60}")
    print(f"  Human detection: {model_path}")
    print(f"  Scene hazard:    {scene_model_path}")

    human_detector = HumanDetector(model_path=model_path)
    hazard_model = YOLO(scene_model_path)

    if video_path:
        videos = [Path(video_path)]
        if not videos[0].exists():
            print(f"Error: Video not found at {video_path}")
            return
    else:
        video_dir = Path(f"data/test_videos/{scene_type}")
        if not video_dir.exists():
            print(f"Error: Directory not found: {video_dir}")
            return
        videos = sorted(
            list(video_dir.glob("*.mp4"))
            + list(video_dir.glob("*.avi"))
            + list(video_dir.glob("*.mov"))
        )
        if not videos:
            print(f"No videos found in {video_dir}")
            return

    print(f"\nFound {len(videos)} video(s) to process")
    print(f"{'=' * 60}\n")

    output_path = Path(output_dir) if output_dir else None
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)

    for i, video in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {video.name}")
        out_file = (output_path / f"{video.stem}_annotated.mp4") if output_path else None
        process_video(video, scene_type, out_file, show, human_detector, hazard_model)

    if show:
        cv2.destroyAllWindows()

    print(f"\n{'=' * 60}")
    print(f"Demo complete! Processed {len(videos)} video(s)")
    print(f"{'=' * 60}\n")


def compare_models(video_path, models=None, variant="sim", num_frames=100):
    """Benchmark multiple human detection models on the same video."""
    if models is None:
        models = ["yolo11n-seg", "yolo11s-seg"]

    video_path = Path(video_path)
    if not video_path.exists():
        print(f"Error: Video not found at {video_path}")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frames = []
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        print("Error: No frames read from video")
        return

    print(f"\n{'=' * 60}")
    print(f"MODEL LATENCY COMPARISON")
    print(f"  Video: {video_path.name} | Frames: {len(frames)} | Variant: {variant}")
    print(f"{'=' * 60}\n")

    results = []

    for model_name in models:
        variant_suffix = f"_{variant}" if variant != "combined" else ""
        finetuned_path = f"runs/segment/human_detection{variant_suffix}_{model_name}/weights/best.pt"
        if os.path.exists(finetuned_path):
            model_path = finetuned_path
            source = "finetuned"
        else:
            model_path = f"{model_name}.pt"
            source = "pretrained"

        print(f"  Benchmarking: {model_name} ({source}: {model_path})")
        detector = HumanDetector(model_path=model_path)

        for frame in frames[:3]:
            detector.get_masks(frame)

        times = []
        total_detections = 0
        for frame in frames:
            t0 = time.time()
            masks = detector.get_masks(frame)
            t1 = time.time()
            times.append(t1 - t0)
            total_detections += len(masks)

        avg_ms = np.mean(times) * 1000
        min_ms = np.min(times) * 1000
        max_ms = np.max(times) * 1000
        fps = 1000.0 / avg_ms if avg_ms > 0 else 0

        results.append({
            "model": model_name,
            "source": source,
            "avg_ms": avg_ms,
            "min_ms": min_ms,
            "max_ms": max_ms,
            "fps": fps,
            "detections": total_detections,
        })

        print(f"    Avg: {avg_ms:.1f}ms | Min: {min_ms:.1f}ms | Max: {max_ms:.1f}ms | "
              f"FPS: {fps:.1f} | Detections: {total_detections}")

    print(f"\n{'=' * 60}")
    print(f"{'Model':<20} {'Source':<12} {'Avg (ms)':<10} {'FPS':<8} {'Detections':<12}")
    print(f"{'-' * 60}")
    for r in results:
        print(f"{r['model']:<20} {r['source']:<12} {r['avg_ms']:<10.1f} {r['fps']:<8.1f} {r['detections']:<12}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate safety monitoring system on real-world videos")
    subparsers = parser.add_subparsers(dest="command")

    parser.add_argument("--scene", type=str, choices=["railway", "bridge", "ship"])
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--model-variant", type=str, choices=["sim", "real", "combined"], default=None)
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--video", type=str, required=True)
    compare_parser.add_argument("--models", type=str, nargs="+", default=["yolo11n-seg", "yolo11s-seg"])
    compare_parser.add_argument("--variant", type=str, default="sim",
                                choices=["sim", "real", "combined"])
    compare_parser.add_argument("--frames", type=int, default=100)

    args = parser.parse_args()

    if args.command == "compare":
        compare_models(args.video, args.models, args.variant, args.frames)
    else:
        if not args.scene:
            parser.error("--scene is required for demo mode")
        run_demo(
            args.scene,
            args.video,
            args.output_dir,
            show=not args.no_show,
            model_variant=args.model_variant,
            base_model=args.base_model,
        )
