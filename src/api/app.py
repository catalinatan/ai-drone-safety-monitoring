"""
FastAPI application factory.

Creates and configures the FastAPI app, registers all routes, and manages
startup/shutdown via the lifespan context manager.

Usage (production):
    uvicorn src.api.app:app --host 0.0.0.0 --port 8001

Usage (development via main.py):
    from src.api.app import create_app
    app = create_app()
"""

from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api import dependencies as deps
from src.api.routes import admin, drone, feeds, health, status, video, zones
from src.core.config import get_config, get_feeds_config
from src.hardware import create_camera_backend
from src.services.feed_manager import FeedManager


# ---------------------------------------------------------------------------
# Background threads
# ---------------------------------------------------------------------------

def _capture_loop(fm: FeedManager, cfg: Dict[str, Any]) -> None:
    """Continuously grab frames from all registered camera backends."""
    fps = cfg.get("streaming", {}).get("capture_fps", 30)
    interval = 1.0 / max(1, fps)

    while getattr(fm, "_running", False):
        for feed_id in fm.feed_ids():
            camera = fm.get_camera(feed_id)
            if camera is None or not camera.is_connected:
                continue
            try:
                frame = camera.grab_frame()
                if frame is None:
                    continue

                # Try to get vehicle position (AirSim cameras expose this)
                position = None
                if hasattr(camera, "get_vehicle_position"):
                    position = camera.get_vehicle_position()

                fm.store_frame(feed_id, frame, position=position)
            except Exception as e:
                print(f"[CAPTURE] {feed_id}: {e}")
        time.sleep(interval)


def _detection_loop(
    fm: FeedManager,
    cfg: Dict[str, Any],
    pipelines: Dict[str, Any],
    store: Any,
    depth_estimator: Any = None,
) -> None:
    """Run human detection on every frame from all feeds."""
    from datetime import datetime
    from src.services.streaming import render_overlay

    det_cfg = cfg.get("detection", {})
    fps = det_cfg.get("fps", 10)
    interval = 1.0 / max(1, fps)

    def encode_frame_jpeg(frame):
        """Encode numpy array as JPEG bytes."""
        import cv2
        _, buf = cv2.imencode(".jpg", frame)
        return buf.tobytes() if buf is not None else b""

    def get_person_coords(
        fm, feed_id, frame, person_masks, depth_estimator, airsim_client
    ):
        """
        Estimate 3D world coordinates for the detected person.

        Falls back to camera position if estimation fails.
        """
        state = fm.get_state(feed_id)
        camera = fm.get_camera(feed_id)

        # Without depth estimator or camera that doesn't support AirSim, use camera position
        if not depth_estimator or not person_masks:
            return state.position or (0.0, 0.0, -10.0)

        # Only AirSim cameras can provide world coordinates
        if not hasattr(camera, "camera_name") or not hasattr(camera, "vehicle_name"):
            return state.position or (0.0, 0.0, -10.0)

        # If no AirSim client available, use camera position
        if not airsim_client:
            return state.position or (0.0, 0.0, -10.0)

        try:
            # Get depth map
            depth_map = depth_estimator.estimate(frame)

            # Find centroid of first person mask
            person_mask = person_masks[0]
            y_indices, x_indices = person_mask.nonzero()
            if len(y_indices) == 0:
                return state.position or (0.0, 0.0, -10.0)

            center_x = float(np.mean(x_indices))
            center_y = float(np.mean(y_indices))
            depth_val = depth_estimator.get_depth_at_pixel(depth_map, center_x, center_y)

            # Convert pixel + depth to world coordinates
            from src.spatial.projection import get_coords_from_lite_mono

            camera_name = camera.camera_name
            vehicle_name = camera.vehicle_name
            cctv_height = cfg.get("detection", {}).get("cctv_height_meters", 10.0)

            world_coord = get_coords_from_lite_mono(
                airsim_client,
                camera_name,
                center_x,
                center_y,
                frame.shape[1],
                frame.shape[0],
                depth_val,
                cctv_height,
                vehicle_name=vehicle_name,
            )
            return (world_coord.x_val, world_coord.y_val, world_coord.z_val)
        except Exception as e:
            print(f"[DETECTION] Depth estimation failed: {e}, using camera position")
            return state.position or (0.0, 0.0, -10.0)

    while getattr(fm, "_running", False):
        for feed_id in fm.feed_ids():
            state = fm.get_state(feed_id)
            if state is None:
                continue

            frame = fm.get_frame(feed_id)
            if frame is None:
                continue

            # Skip if not warmed up yet
            warmup = det_cfg.get("warmup_frames", 20)
            if not fm.is_warmed_up(feed_id, warmup):
                continue

            # Run detection on this frame
            pipeline = pipelines.get(feed_id)
            if pipeline is None:
                continue

            try:
                result = pipeline.process_frame(frame)

                # Render overlay if masks exist
                all_masks = result.danger_masks + result.caution_masks
                if all_masks:
                    overlay = render_overlay(frame, all_masks)
                else:
                    overlay = None

                # Update detection state
                fm.update_detection(
                    feed_id,
                    alarm_active=result.alarm_active,
                    caution_active=result.caution_active,
                    mask_overlay=overlay,
                )

                # If alarm fired, add to trigger store with 3D coordinates
                if result.alarm_fired:
                    snapshot_jpeg = encode_frame_jpeg(frame)
                    replay = list(state.replay_buffer)
                    trigger_idx = len(replay) - 1

                    # Get AirSim client from camera if available
                    camera = fm.get_camera(feed_id)
                    airsim_client = getattr(camera, "client", None) if camera else None

                    # Compute 3D coordinates using depth estimation
                    coords = get_person_coords(
                        fm,
                        feed_id,
                        frame,
                        result.danger_masks,
                        depth_estimator,
                        airsim_client,
                    )

                    event = deps.TriggerEvent(
                        id=store.next_id(),
                        feed_id=feed_id,
                        timestamp=datetime.utcnow().isoformat() + "Z",
                        coords=coords,
                        snapshot=snapshot_jpeg or b"",
                        replay_frames=replay,
                        replay_trigger_index=max(0, trigger_idx),
                    )
                    store.add(event)

            except Exception as e:
                print(f"[DETECTION] {feed_id}: {e}")

        time.sleep(interval)


def _auto_seg_loop(
    fm: FeedManager,
    cfg: Dict[str, Any],
    segmenter: Any,
) -> None:
    """Periodically auto-segment zones on every feed."""
    from src.core.models import Zone

    seg_cfg = cfg.get("auto_segmentation", {})
    interval_seconds = seg_cfg.get("interval_seconds", 60.0)

    while getattr(fm, "_running", False):
        now = time.monotonic()
        for feed_id in fm.feed_ids():
            state = fm.get_state(feed_id)
            if state is None:
                continue

            # Skip if user set zones manually
            if state.manual_zones_set:
                continue

            # Skip if scene_type not configured
            if not state.scene_type:
                continue

            # Skip if too soon since last auto-seg
            last_seg_time = getattr(state, "last_auto_seg_time", 0)
            if now - last_seg_time < interval_seconds:
                continue

            # Get frame and segment
            frame = fm.get_frame(feed_id)
            if frame is None:
                continue

            try:
                zone_dicts = segmenter.segment_frame(frame, state.scene_type)
                if zone_dicts:
                    # Convert dicts to Zone objects
                    zones = [Zone(**z) for z in zone_dicts]
                    fm.update_zones(feed_id, zones, frame.shape[1], frame.shape[0])
                    state.auto_seg_active = True
                    state.last_auto_seg_time = now
            except Exception as e:
                print(f"[AUTO-SEG] {feed_id}: {e}")

        time.sleep(5.0)  # Check every 5 seconds


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: initialise hardware and background threads. Shutdown: stop them."""
    cfg = get_config()
    feeds_cfg = get_feeds_config()

    fm = FeedManager()
    deps.set_feed_manager(fm)
    deps.set_config(cfg)

    # Register feeds from feeds.yaml and attempt camera connections
    # Retry logic for AirSim connections (may need time to start)
    connected_count = 0
    for feed_id, feed_def in feeds_cfg.items():
        camera_cfg = feed_def.get("camera", {})
        camera = None

        try:
            camera = create_camera_backend(camera_cfg)

            # Retry AirSim connections more aggressively
            is_airsim = camera_cfg.get("type") == "airsim"
            max_retries = 5 if is_airsim else 1
            retry_delay = 2.0 if is_airsim else 0.5

            for attempt in range(max_retries):
                print(f"[INIT] {feed_id}: connection attempt {attempt + 1}/{max_retries}...")
                ok = camera.connect()
                if ok:
                    connected_count += 1
                    print(f"[INIT] {feed_id}: ✓ connected")
                    break
                if attempt < max_retries - 1:
                    print(f"[INIT] {feed_id}: waiting {retry_delay}s before retry...")
                    time.sleep(retry_delay)
                else:
                    print(f"[INIT] {feed_id}: ✗ failed after {max_retries} attempts")

        except Exception as e:
            print(f"[INIT] {feed_id}: camera backend error — {e}")

        # Raise exception instead of using stub - force real diagnosis
        if camera is None or not camera.is_connected:
            raise RuntimeError(
                f"[INIT] Camera {feed_id} failed to connect to AirSim. "
                f"Make sure AirSim is running and accessible. "
                f"Check the detailed error messages above."
            )

        fm.register_feed(
            feed_id=feed_id,
            name=feed_def.get("name", feed_id),
            location=feed_def.get("location", ""),
            camera=camera,
            scene_type=feed_def.get("scene_type"),
        )

    if connected_count > 0:
        print(f"[INIT] {connected_count}/{len(feeds_cfg)} cameras connected")
    else:
        print("[INIT] No cameras connected — running in limited mode (no AirSim)")

    # Connect to drone API
    try:
        from src.backend.drone_client import DroneAPIClient
        drone_api = DroneAPIClient()  # reads url/timeout from config/default.yaml
        if drone_api.check_connection():
            deps.set_drone_api(drone_api)
            print("[INIT] Drone API connected")
        else:
            print("[INIT] Drone API not available")
    except Exception as e:
        print(f"[INIT] Drone API init failed: {e}")

    # Load shared human detector
    detector = None
    try:
        from src.detection.human_detector import HumanDetector
        detector = HumanDetector()
        print("[INIT] Human detector loaded")
    except Exception as e:
        print(f"[INIT] Human detector failed to load: {e}")

    # Load shared scene segmenter
    segmenter = None
    try:
        from src.detection.scene_segmenter import SceneSegmenter
        segmenter = SceneSegmenter()
        deps.set_scene_segmenter(segmenter)
        print("[INIT] Scene segmenter loaded")
    except Exception as e:
        print(f"[INIT] Scene segmenter failed to load: {e}")

    # Load depth estimator (if configured)
    depth_estimator = None
    try:
        from src.detection.depth_estimator_wrapper import DepthEstimator
        enc_path = cfg.get("depth_estimation", {}).get("encoder_path")
        dec_path = cfg.get("depth_estimation", {}).get("decoder_path")
        if enc_path and dec_path:
            depth_estimator = DepthEstimator(encoder_path=enc_path, decoder_path=dec_path)
            deps.set_depth_estimator(depth_estimator)
            print("[INIT] Depth estimator loaded")
    except Exception as e:
        print(f"[INIT] Depth estimator failed to load: {e}")

    # Create per-feed detection pipelines
    pipelines = {}
    if detector is not None:
        try:
            from src.core.detection_pipeline import DetectionPipeline
            from src.core.alarm import AlarmState

            cooldown = cfg.get("zones", {}).get("alarm_cooldown_seconds", 5.0)
            warmup = cfg.get("detection", {}).get("warmup_frames", 20)
            for feed_id in fm.feed_ids():
                state = fm.get_state(feed_id)
                alarm = AlarmState(cooldown_seconds=cooldown)
                pipelines[feed_id] = DetectionPipeline(
                    detector=detector,
                    zone_manager=state.zone_manager,
                    alarm=alarm,
                    warmup_frames=warmup,
                )
        except Exception as e:
            print(f"[INIT] Detection pipeline setup failed: {e}")

    # Start background threads
    fm._running = True
    capture_thread = threading.Thread(
        target=_capture_loop, args=(fm, cfg), daemon=True, name="frame-capture"
    )
    capture_thread.start()

    if pipelines:
        detection_thread = threading.Thread(
            target=_detection_loop,
            args=(fm, cfg, pipelines, deps.get_trigger_store(), depth_estimator),
            daemon=True,
            name="detection",
        )
        detection_thread.start()

    if segmenter is not None:
        auto_seg_thread = threading.Thread(
            target=_auto_seg_loop,
            args=(fm, cfg, segmenter),
            daemon=True,
            name="auto-seg",
        )
        auto_seg_thread.start()

    print("[SERVER] Backend started")
    yield

    # Shutdown
    fm._running = False
    print("[SERVER] Backend shutting down")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Safety Monitoring Backend",
        version="2.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(feeds.router)
    app.include_router(zones.router)
    app.include_router(video.router)
    app.include_router(status.router)
    app.include_router(drone.router)
    app.include_router(admin.router)

    return app


# Module-level app instance (for uvicorn src.api.app:app)
app = create_app()
