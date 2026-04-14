"""
Evaluate depth estimation + coordinate projection accuracy against AirSim ground truth.

For each camera feed, runs YOLO detection and Lite-Mono depth estimation on live
AirSim frames, computes estimated 3D world coordinates via ray-ground intersection,
and compares against the true position of 'ThirdPersonCharacter' from AirSim.

Results are saved to eval_output/depth_estimation/{env}/:
  - results.csv               — per-sample raw data
  - summary.csv               — aggregate statistics
  - error_vs_distance.png     — error as a function of ground-truth distance
  - scatter_xy.png            — bird's-eye estimated vs ground-truth positions
  - error_histogram.png       — distribution of 2D errors
  - error_by_camera.png       — per-camera error comparison

After running on all environments, use --aggregate to combine results:
  - eval_output/depth_estimation/aggregate/combined_results.csv
  - eval_output/depth_estimation/aggregate/comparison_table.csv
  - eval_output/depth_estimation/aggregate/error_by_environment.png

Usage:
    1. Start AirSim with the environment and ThirdPersonCharacter
    2. Run per environment:
       python -m src.eval.eval_depth_estimation --env bridge
       python -m src.eval.eval_depth_estimation --env ship --samples 100
       python -m src.eval.eval_depth_estimation --env railway --cameras cctv-1 cctv-2
    3. Add --setup to auto-position drones around the actor (useful for new envs):
       python -m src.eval.eval_depth_estimation --env ship --setup
    4. Add --show to open a live window showing detections during evaluation:
       python -m src.eval.eval_depth_estimation --env bridge --show
       (press 'q' in the window to stop early)
    5. Aggregate all results:
       python -m src.eval.eval_depth_estimation --aggregate
"""

from __future__ import annotations

import argparse
import csv
import math
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_ROOT = Path("eval_output/depth_estimation")
ACTOR_NAME = "ThirdPersonCharacter_2"

# Camera configs matching feeds.yaml
CAMERA_FEEDS = {
    "cctv-1": {"camera_name": "0", "vehicle_name": "Drone2"},
    "cctv-2": {"camera_name": "0", "vehicle_name": "Drone3"},
    "cctv-3": {"camera_name": "0", "vehicle_name": "Drone4"},
    "cctv-4": {"camera_name": "0", "vehicle_name": "Drone5"},
}

CCTV_HEIGHT_METERS = 15.0

# If the best detection-to-ground-truth 2D distance exceeds this, assume
# the target actor was not detected and skip the sample.
MATCH_DISTANCE_THRESHOLD_M = 10.0

# Minimum mask area as a fraction of the frame — filters out small blobs/false positives
MIN_MASK_AREA_FRACTION = 0.001


# ---------------------------------------------------------------------------
# AirSim helpers
# ---------------------------------------------------------------------------


def connect_airsim():
    """Connect to AirSim and return the client."""
    import airsim

    client = airsim.MultirotorClient()
    client.confirmConnection()
    return client


def get_camera_frame(client, camera_name: str, vehicle_name: str) -> Optional[np.ndarray]:
    """Grab a single RGB frame from an AirSim camera."""
    import airsim

    responses = client.simGetImages(
        [airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)],
        vehicle_name=vehicle_name,
    )
    if not responses or responses[0].width == 0:
        return None
    img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
    img = img1d.reshape(responses[0].height, responses[0].width, 3)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_ground_truth_position(client) -> Optional[tuple]:
    """Get the true NED position of the target actor."""
    pose = client.simGetObjectPose(ACTOR_NAME)
    if math.isnan(pose.position.x_val):
        return None
    return (pose.position.x_val, pose.position.y_val, pose.position.z_val)


def start_follow_loop(client, cameras: dict) -> threading.Event:
    """Start a background thread that continuously teleports drones to Camera Actor poses.

    Same logic as _follow_mode_loop in app.py — reads CCTV1-4 Camera Actor poses
    from the UE scene and teleports the corresponding drones to match.

    Returns a stop_event that can be set to stop the loop.
    """
    import airsim

    # camera_mappings: {vehicle_name: cam_actor_name}
    from src.core.config import get_config

    follow_cfg = get_config().get("follow_mode", {})
    camera_mappings = follow_cfg.get(
        "camera_mappings",
        {
            "Drone2": "CCTV1",
            "Drone3": "CCTV2",
            "Drone4": "CCTV3",
            "Drone5": "CCTV4",
        },
    )
    interval = follow_cfg.get("follow_interval", 0.01)

    print(
        f"[FOLLOW] Starting follow loop — teleporting {list(camera_mappings.keys())} "
        f"to Camera Actors {list(camera_mappings.values())}"
    )

    # Arm and takeoff all drones
    takeoff_futures = []
    for vehicle_name in camera_mappings:
        try:
            client.enableApiControl(True, vehicle_name=vehicle_name)
            client.armDisarm(True, vehicle_name=vehicle_name)
            future = client.takeoffAsync(vehicle_name=vehicle_name)
            takeoff_futures.append((vehicle_name, future))
            print(f"[FOLLOW] {vehicle_name} takeoff initiated")
        except Exception as e:
            print(f"[FOLLOW] {vehicle_name} arm/takeoff failed: {e}")

    for vehicle_name, future in takeoff_futures:
        try:
            future.join()
            print(f"[FOLLOW] {vehicle_name} takeoff complete")
        except Exception as e:
            print(f"[FOLLOW] {vehicle_name} takeoff join: {e}")

    print("[FOLLOW] All drones airborne — starting teleport loop")

    stop_event = threading.Event()

    def _loop():
        # Dedicated client for the follow thread
        follow_client = airsim.MultirotorClient()
        follow_client.confirmConnection()

        while not stop_event.is_set():
            try:
                for vehicle_name, cam_actor_name in camera_mappings.items():
                    cam_pose = follow_client.simGetObjectPose(cam_actor_name)
                    if math.isnan(cam_pose.position.x_val):
                        continue
                    follow_client.simSetVehiclePose(cam_pose, True, vehicle_name=vehicle_name)
                    follow_client.moveByVelocityAsync(0, 0, 0, 0.1, vehicle_name=vehicle_name)
            except Exception as e:
                print(f"[FOLLOW] Error: {e}")
            time.sleep(interval)

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
    print("[FOLLOW] Background follow thread started")

    return stop_event


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def run_evaluation(
    n_samples: int = 50,
    camera_ids: list[str] | None = None,
    use_simulator_model: bool = False,
    env: str = "default",
    show: bool = False,
    setup: bool = False,
) -> list[dict]:
    """
    Run the evaluation loop.

    For each sample:
      1. Grab frame from each AirSim camera
      2. Run YOLO detection — skip if no person found
      3. Run Lite-Mono depth estimation on the detected person's mask center
      4. Project pixel + depth to 3D world coordinates
      5. Query AirSim for the person's true position
      6. Compute the error
    """
    from src.detection.depth_estimator_wrapper import DepthEstimator
    from src.detection.human_detector import HumanDetector
    from src.spatial.projection import get_coords_from_lite_mono

    cameras = {k: v for k, v in CAMERA_FEEDS.items() if camera_ids is None or k in camera_ids}

    if not cameras:
        print("[ERROR] No valid cameras specified")
        return []

    # Load models
    model_path = None
    if use_simulator_model:
        model_path = "yolo11n-seg.pt"
    print("[EVAL] Loading YOLO detector...")
    detector = HumanDetector(model_path=model_path)
    print("[EVAL] Loading depth estimator...")
    depth_est = DepthEstimator()

    # Connect to AirSim
    print("[EVAL] Connecting to AirSim...")
    client = connect_airsim()
    print("[EVAL] Connected")

    # Optionally start follow mode (teleport drones to Camera Actor poses)
    follow_stop = None
    if setup:
        follow_stop = start_follow_loop(client, cameras)
        print("[EVAL] Waiting 2s for drones to settle...")
        time.sleep(2.0)

    records = []
    sample_count = 0
    attempt = 0
    max_attempts = n_samples * 10  # avoid infinite loop if person is never detected

    print(f"[EVAL] Collecting {n_samples} samples across {len(cameras)} camera(s)...")
    print(f"[EVAL] Cameras: {list(cameras.keys())}")
    print()

    skip_stats = {
        "no_frame": 0,
        "no_detection": 0,
        "no_person": 0,
        "no_gt": 0,
        "no_projection": 0,
        "too_far": 0,
    }
    gt_z_printed = False

    while sample_count < n_samples and attempt < max_attempts:
        attempt += 1

        # Periodic progress update
        if attempt % 10 == 0:
            print(
                f"  [DEBUG] Attempt {attempt} | Samples so far: {sample_count}/{n_samples} | "
                f"Skips: frame={skip_stats['no_frame']} det={skip_stats['no_detection']} "
                f"person={skip_stats['no_person']} gt={skip_stats['no_gt']} "
                f"proj={skip_stats['no_projection']} far={skip_stats['too_far']}"
            )

        for feed_id, cam_cfg in cameras.items():
            if sample_count >= n_samples:
                break

            cam_name = cam_cfg["camera_name"]
            veh_name = cam_cfg["vehicle_name"]

            # 1. Grab frame
            frame = get_camera_frame(client, cam_name, veh_name)
            if frame is None:
                skip_stats["no_frame"] += 1
                print(f"  [DEBUG] {feed_id}: no frame received")
                if show:
                    # Show blank placeholder
                    blank = np.zeros((240, 320, 3), dtype=np.uint8)
                    cv2.putText(
                        blank,
                        f"{feed_id} — NO FRAME",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )
                    cv2.imshow(feed_id, blank)
                continue

            # 2. Detect humans
            results = detector.model(
                frame,
                conf=0.25,
                imgsz=1280,
                verbose=False,
                half=detector._use_half,
                save=False,
            )

            # Build visualisation frame for this camera
            vis = frame.copy() if show else None

            if not results or results[0].masks is None:
                skip_stats["no_detection"] += 1
                if show:
                    cv2.putText(
                        vis,
                        "NO DETECTIONS",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                    cv2.imshow(feed_id, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                continue

            # Filter person detections (class 0)
            n_total_det = len(results[0].boxes)
            person_indices = [i for i, box in enumerate(results[0].boxes) if int(box.cls[0]) == 0]
            if not person_indices:
                skip_stats["no_person"] += 1
                print(f"  [DEBUG] {feed_id}: {n_total_det} detections but 0 persons")
                if show:
                    cv2.putText(
                        vis,
                        f"{n_total_det} det, 0 persons",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 165, 255),
                        2,
                    )
                    cv2.imshow(feed_id, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                continue

            print(
                f"  [DEBUG] {feed_id}: {len(person_indices)} person(s) detected "
                f"(out of {n_total_det} total)"
            )

            # 3. Ground truth (get early so we can match detections)
            gt = get_ground_truth_position(client)
            if gt is None:
                skip_stats["no_gt"] += 1
                print(
                    f"  [DEBUG] {feed_id}: ground truth unavailable "
                    f"(ThirdPersonCharacter not found)"
                )
                if show:
                    cv2.putText(
                        vis,
                        "GT UNAVAILABLE",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                    cv2.imshow(feed_id, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                continue

            if not gt_z_printed:
                cam_info_dbg = client.simGetCameraInfo(cam_name, vehicle_name=veh_name)
                actual_height = gt[2] - cam_info_dbg.pose.position.z_val
                print(
                    f"  [DEBUG] Actor Z = {gt[2]:.2f}, "
                    f"Camera Z = {cam_info_dbg.pose.position.z_val:.2f}"
                )
                print(
                    f"  [DEBUG] Actual camera-to-ground height = {actual_height:.2f}m  "
                    f"(configured CCTV_HEIGHT_METERS = {CCTV_HEIGHT_METERS})"
                )
                gt_z_printed = True

            # Compute actual camera height above ground from AirSim poses
            cam_info_cur = client.simGetCameraInfo(cam_name, vehicle_name=veh_name)
            cctv_height_actual = gt[2] - cam_info_cur.pose.position.z_val

            # 4. Depth estimation (once per frame, shared across all detections)
            depth_map = depth_est.estimate(frame)

            # 5. Project ALL person detections and find the one closest to ground truth
            best_est = None
            best_err_2d = float("inf")
            best_center = None
            best_depth_val = None

            for idx in person_indices:
                mask = results[0].masks.data[idx]
                mask_np = mask.cpu().numpy()
                mask_resized = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))
                mask_binary = (mask_resized > 0.5).astype(np.uint8)

                y_indices, x_indices = mask_binary.nonzero()
                if len(y_indices) == 0:
                    continue

                # Skip small blobs (false positives)
                mask_area = len(y_indices)
                frame_area = frame.shape[0] * frame.shape[1]
                if mask_area < frame_area * MIN_MASK_AREA_FRACTION:
                    print(
                        f"  [DEBUG] {feed_id}: detection {idx} skipped — mask too small "
                        f"({mask_area}/{frame_area} = {mask_area / frame_area:.4f})"
                    )
                    continue

                # Use bottom-center of mask (feet), not centroid (torso),
                # so the ground-plane intersection lands where the person stands.
                cx = float(np.mean(x_indices))
                cy = float(np.max(y_indices))
                dv = depth_est.get_depth_at_pixel(depth_map, int(cx), int(cy))

                try:
                    est_pos = get_coords_from_lite_mono(
                        client,
                        cam_name,
                        cx,
                        cy,
                        frame.shape[1],
                        frame.shape[0],
                        dv,
                        cctv_height_actual,
                        vehicle_name=veh_name,
                    )
                    ex, ey = est_pos.x_val, est_pos.y_val
                except Exception as e:
                    print(f"  [DEBUG] {feed_id}: projection failed for detection {idx}: {e}")
                    continue

                d2d = math.sqrt((ex - gt[0]) ** 2 + (ey - gt[1]) ** 2)
                if d2d < best_err_2d:
                    best_err_2d = d2d
                    best_est = (est_pos.x_val, est_pos.y_val, est_pos.z_val)
                    best_center = (cx, cy)
                    best_depth_val = dv

            if best_est is None:
                skip_stats["no_projection"] += 1
                print(f"  [DEBUG] {feed_id}: all projections failed")
                if show:
                    cv2.putText(
                        vis,
                        "PROJECTION FAILED",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                    cv2.imshow(feed_id, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                continue

            # Skip if best match is too far — likely not the target actor
            if best_err_2d > MATCH_DISTANCE_THRESHOLD_M:
                skip_stats["too_far"] += 1
                print(
                    f"  [DEBUG] {feed_id}: best match {best_err_2d:.1f}m away "
                    f"(threshold {MATCH_DISTANCE_THRESHOLD_M}m) — skipping as likely not target"
                )
                if show:
                    for idx in person_indices:
                        m = results[0].masks.data[idx].cpu().numpy()
                        m = cv2.resize(m, (vis.shape[1], vis.shape[0]))
                        vis[m > 0.5] = (vis[m > 0.5] * 0.5 + np.array([0, 0, 255]) * 0.5).astype(
                            np.uint8
                        )
                    cv2.putText(
                        vis,
                        f"TOO FAR ({best_err_2d:.1f}m) — SKIPPED",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                    cv2.imshow(feed_id, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                continue

            est = best_est
            center_x, center_y = best_center
            depth_val = best_depth_val

            # 6. Compute errors (X/Y only — Z is not relevant for drone dispatch)
            err_x = est[0] - gt[0]
            err_y = est[1] - gt[1]
            err_2d = best_err_2d

            # Camera-to-person 2D distance
            cam_info = client.simGetCameraInfo(cam_name, vehicle_name=veh_name)
            cam_pos = cam_info.pose.position
            cam_to_person = math.sqrt(
                (cam_pos.x_val - gt[0]) ** 2 + (cam_pos.y_val - gt[1]) ** 2,
            )

            record = {
                "sample": sample_count + 1,
                "env": env,
                "depth_val": round(depth_val, 4),
                "pixel_x": round(center_x, 1),
                "pixel_y": round(center_y, 1),
                "est_x": round(est[0], 3),
                "est_y": round(est[1], 3),
                "gt_x": round(gt[0], 3),
                "gt_y": round(gt[1], 3),
                "err_x": round(err_x, 3),
                "err_y": round(err_y, 3),
                "err_2d": round(err_2d, 3),
                "cam_to_person_dist": round(cam_to_person, 3),
            }
            records.append(record)
            sample_count += 1

            print(
                f"  [{sample_count:3d}/{n_samples}] {feed_id}  "
                f"2D err={err_2d:6.2f}m  "
                f"depth={depth_val:.3f}  cam_dist={cam_to_person:.1f}m"
            )

            # --- Live visualisation (per-camera window) ---
            if show:
                # Draw all person masks in blue
                for idx in person_indices:
                    m = results[0].masks.data[idx].cpu().numpy()
                    m = cv2.resize(m, (vis.shape[1], vis.shape[0]))
                    vis[m > 0.5] = (vis[m > 0.5] * 0.5 + np.array([255, 150, 0]) * 0.5).astype(
                        np.uint8
                    )

                # Highlight matched detection in green
                cx_i, cy_i = int(center_x), int(center_y)
                cv2.circle(vis, (cx_i, cy_i), 8, (0, 255, 0), -1)
                cv2.putText(
                    vis,
                    "MATCHED",
                    (cx_i + 12, cy_i - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                # Info overlay
                info_lines = [
                    f"Sample: {sample_count}/{n_samples}",
                    f"2D err: {err_2d:.2f}m",
                    f"Depth: {depth_val:.3f}  Cam dist: {cam_to_person:.1f}m",
                    f"Detections: {len(person_indices)}",
                ]
                for li, line in enumerate(info_lines):
                    cv2.putText(
                        vis,
                        line,
                        (10, 25 + li * 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

                cv2.imshow(feed_id, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        # Check for quit key (shared across all windows)
        if show:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\n[EVAL] Stopped by user (q)")
                cv2.destroyAllWindows()
                if follow_stop is not None:
                    follow_stop.set()
                return records

        # Small delay to avoid hammering AirSim
        time.sleep(0.1)

    # Final debug summary
    print(f"\n[EVAL] Finished: {sample_count}/{n_samples} samples in {attempt} attempts")
    print(
        f"[EVAL] Skip reasons: "
        f"no_frame={skip_stats['no_frame']}  "
        f"no_detection={skip_stats['no_detection']}  "
        f"no_person={skip_stats['no_person']}  "
        f"no_gt={skip_stats['no_gt']}  "
        f"no_projection={skip_stats['no_projection']}  "
        f"too_far={skip_stats['too_far']}"
    )

    if sample_count < n_samples:
        print(
            f"\n[WARN] Only collected {sample_count}/{n_samples} samples "
            f"after {max_attempts} attempts"
        )

    if show:
        cv2.destroyAllWindows()

    if follow_stop is not None:
        follow_stop.set()

    return records


# ---------------------------------------------------------------------------
# Output — CSV and plots
# ---------------------------------------------------------------------------


def save_results(records: list[dict], output_dir: Path) -> None:
    """Save raw CSV, summary CSV, and visualisation plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if not records:
        print("[EVAL] No records to save")
        return

    # --- Raw CSV ---
    csv_path = output_dir / "results.csv"
    fields = list(records[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(f"\n  Raw results   → {csv_path}")

    # --- Compute summary stats ---
    err_2d = [r["err_2d"] for r in records]

    summary = {
        "n_samples": len(records),
        "mean_2d_error_m": round(float(np.mean(err_2d)), 3),
        "median_2d_error_m": round(float(np.median(err_2d)), 3),
        "std_2d_error_m": round(float(np.std(err_2d)), 3),
        "max_2d_error_m": round(float(np.max(err_2d)), 3),
        "min_2d_error_m": round(float(np.min(err_2d)), 3),
    }

    # --- Summary CSV ---
    summary_path = output_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for k, v in summary.items():
            writer.writerow([k, v])
    print(f"  Summary       → {summary_path}")

    # --- Print summary table ---
    print(f"\n{'=' * 65}")
    print("  DEPTH ESTIMATION EVALUATION RESULTS")
    print(f"{'=' * 65}")
    print(f"  Samples collected:  {summary['n_samples']}")
    print(f"  Mean 2D error:      {summary['mean_2d_error_m']:.3f} m")
    print(f"  Median 2D error:    {summary['median_2d_error_m']:.3f} m")
    print(f"  Std 2D error:       {summary['std_2d_error_m']:.3f} m")
    print(f"  Max 2D error:       {summary['max_2d_error_m']:.3f} m")
    print(f"  Min 2D error:       {summary['min_2d_error_m']:.3f} m")
    print(f"{'=' * 65}")

    # --- Generate plots ---
    _generate_plots(records, summary, output_dir)


def _generate_plots(records: list[dict], summary: dict, output_dir: Path) -> None:
    """Generate all visualisation plots using matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  [WARN] matplotlib not installed — skipping plots")
        return

    # --- 1. Error vs distance from camera ---
    fig, ax = plt.subplots(figsize=(8, 5))
    dists = [r["cam_to_person_dist"] for r in records]
    errs = [r["err_2d"] for r in records]
    ax.scatter(dists, errs, alpha=0.6, s=30, color="#2196F3")
    ax.set_xlabel("Camera-to-Person Distance (m)")
    ax.set_ylabel("2D Position Error (m)")
    ax.set_title("Projection Error vs Distance from Camera")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "error_vs_distance.png", dpi=150)
    plt.close(fig)
    print(f"  Plot          → {output_dir / 'error_vs_distance.png'}")

    # --- 2. Bird's-eye scatter: estimated vs ground truth ---
    fig, ax = plt.subplots(figsize=(8, 8))
    gt_x = [r["gt_x"] for r in records]
    gt_y = [r["gt_y"] for r in records]
    est_x = [r["est_x"] for r in records]
    est_y = [r["est_y"] for r in records]
    ax.scatter(gt_y, gt_x, marker="o", alpha=0.5, s=30, color="#4CAF50", label="Ground Truth")
    ax.scatter(est_y, est_x, marker="x", alpha=0.5, s=30, color="#F44336", label="Estimated")
    for g_x, g_y, e_x, e_y in zip(gt_x, gt_y, est_x, est_y):
        ax.plot([g_y, e_y], [g_x, e_x], color="grey", alpha=0.15, linewidth=0.8)
    ax.set_xlabel("East (Y) — metres")
    ax.set_ylabel("North (X) — metres")
    ax.set_title("Bird's-Eye View: Ground Truth (o) vs Estimated (x)")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "scatter_xy.png", dpi=150)
    plt.close(fig)
    print(f"  Plot          → {output_dir / 'scatter_xy.png'}")

    # --- 3. Error histogram ---
    fig, ax = plt.subplots(figsize=(8, 5))
    err_2d = [r["err_2d"] for r in records]
    ax.hist(err_2d, bins=30, edgecolor="black", alpha=0.7, color="#2196F3")
    ax.axvline(
        np.mean(err_2d),
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean = {np.mean(err_2d):.2f} m",
    )
    ax.axvline(
        np.median(err_2d),
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label=f"Median = {np.median(err_2d):.2f} m",
    )
    ax.set_xlabel("2D Position Error (m)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of 2D Projection Errors")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "error_histogram.png", dpi=150)
    plt.close(fig)
    print(f"  Plot          → {output_dir / 'error_histogram.png'}")


# ---------------------------------------------------------------------------
# Aggregation — combine results from multiple environments
# ---------------------------------------------------------------------------


def aggregate_results() -> None:
    """Combine results.csv from all environment subdirectories into one report."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    agg_dir = OUTPUT_ROOT / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)

    # Find all per-env results
    all_records = []
    env_dirs = [
        d
        for d in OUTPUT_ROOT.iterdir()
        if d.is_dir() and d.name != "aggregate" and (d / "results.csv").exists()
    ]

    if not env_dirs:
        print("[AGGREGATE] No environment results found. Run evaluations first.")
        return

    for env_dir in sorted(env_dirs):
        csv_path = env_dir / "results.csv"
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["env"] = env_dir.name
                # Convert numeric fields
                for k in [
                    "err_2d",
                    "cam_to_person_dist",
                    "depth_val",
                    "est_x",
                    "est_y",
                    "gt_x",
                    "gt_y",
                ]:
                    if k in row:
                        row[k] = float(row[k])
                all_records.append(row)

    if not all_records:
        print("[AGGREGATE] No records found in results files")
        return

    # Combined CSV
    combined_path = agg_dir / "combined_results.csv"
    fields = list(all_records[0].keys())
    with open(combined_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_records)
    print(f"  Combined CSV  → {combined_path}")

    # Comparison table
    envs = sorted(set(r["env"] for r in all_records))
    comparison = []
    for env_name in envs:
        env_records = [r for r in all_records if r["env"] == env_name]
        errs_2d = [r["err_2d"] for r in env_records]
        comparison.append(
            {
                "Environment": env_name,
                "N": len(env_records),
                "Mean 2D Error (m)": round(float(np.mean(errs_2d)), 3),
                "Median 2D Error (m)": round(float(np.median(errs_2d)), 3),
                "Std 2D Error (m)": round(float(np.std(errs_2d)), 3),
                "Max 2D Error (m)": round(float(np.max(errs_2d)), 3),
            }
        )

    # Add overall row
    all_2d = [r["err_2d"] for r in all_records]
    comparison.append(
        {
            "Environment": "OVERALL",
            "N": len(all_records),
            "Mean 2D Error (m)": round(float(np.mean(all_2d)), 3),
            "Median 2D Error (m)": round(float(np.median(all_2d)), 3),
            "Std 2D Error (m)": round(float(np.std(all_2d)), 3),
            "Max 2D Error (m)": round(float(np.max(all_2d)), 3),
        }
    )

    comp_path = agg_dir / "comparison_table.csv"
    comp_fields = list(comparison[0].keys())
    with open(comp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=comp_fields)
        writer.writeheader()
        writer.writerows(comparison)
    print(f"  Comparison    → {comp_path}")

    # Print table
    print(f"\n{'=' * 80}")
    print("  CROSS-ENVIRONMENT COMPARISON")
    print(f"{'=' * 80}")
    print(f"  {'Environment':<15} {'N':>5} {'Mean 2D':>10} {'Median 2D':>10} {'Std':>8} {'Max':>8}")
    print(f"  {'-' * 58}")
    for row in comparison:
        print(
            f"  {row['Environment']:<15} {row['N']:>5} "
            f"{row['Mean 2D Error (m)']:>9.3f}m {row['Median 2D Error (m)']:>9.3f}m "
            f"{row['Std 2D Error (m)']:>7.3f}m {row['Max 2D Error (m)']:>7.3f}m"
        )

    # Plot: box plot by environment
    if plt is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        env_data = []
        env_labels = []
        for env_name in envs:
            errs = [r["err_2d"] for r in all_records if r["env"] == env_name]
            env_data.append(errs)
            env_labels.append(env_name)

        cmap = plt.cm.Set2
        bp = ax.boxplot(env_data, labels=env_labels, patch_artist=True)
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(cmap(i / max(len(envs) - 1, 1)))
            patch.set_alpha(0.7)
        ax.set_xlabel("Environment")
        ax.set_ylabel("2D Position Error (m)")
        ax.set_title("Depth Estimation Error by Environment")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        plot_path = agg_dir / "error_by_environment.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  Plot          → {plot_path}")

        # Scatter: error vs distance, coloured by env
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, env_name in enumerate(envs):
            env_r = [r for r in all_records if r["env"] == env_name]
            dists = [r["cam_to_person_dist"] for r in env_r]
            errs = [r["err_2d"] for r in env_r]
            ax.scatter(
                dists, errs, alpha=0.5, s=25, label=env_name, color=cmap(i / max(len(envs) - 1, 1))
            )
        ax.set_xlabel("Camera-to-Person Distance (m)")
        ax.set_ylabel("2D Position Error (m)")
        ax.set_title("Projection Error vs Distance (All Environments)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plot_path = agg_dir / "error_vs_distance_all.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  Plot          → {plot_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate depth estimation accuracy against AirSim ground truth",
    )
    parser.add_argument(
        "--samples", type=int, default=50, help="Number of samples to collect (default: 50)"
    )
    parser.add_argument(
        "--cameras", nargs="+", default=None, help="Camera feed IDs to evaluate (default: all)"
    )
    parser.add_argument(
        "--simulator", action="store_true", help="Use stock yolo11n-seg model instead of fine-tuned"
    )
    parser.add_argument(
        "--env", type=str, default="default", help="Environment name (e.g. bridge, ship, railway)"
    )
    parser.add_argument(
        "--show", action="store_true", help="Show live detection window during evaluation"
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Auto-position drones around ThirdPersonCharacter before starting",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Aggregate results from all environments (no AirSim needed)",
    )
    args = parser.parse_args()

    if args.aggregate:
        print("=" * 65)
        print("  AGGREGATING RESULTS ACROSS ENVIRONMENTS")
        print("=" * 65)
        aggregate_results()
        print("\nDone.")
        return

    output_dir = OUTPUT_ROOT / args.env

    print("=" * 65)
    print(f"  DEPTH ESTIMATION EVALUATION — {args.env.upper()}")
    print("=" * 65)

    records = run_evaluation(
        n_samples=args.samples,
        camera_ids=args.cameras,
        use_simulator_model=args.simulator,
        env=args.env,
        show=args.show,
        setup=args.setup,
    )

    if records:
        save_results(records, output_dir)
    else:
        print("\n[EVAL] No samples collected — is AirSim running with ThirdPersonCharacter?")

    print("\nDone.")


if __name__ == "__main__":
    main()
