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

import os
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
    import cv2
    from datetime import datetime

    fps = cfg.get("streaming", {}).get("capture_fps", 30)
    interval = 1.0 / max(1, fps)
    _last_reconnect: dict = {}  # feed_id → monotonic time of last retry

    while getattr(fm, "_running", False):
        now = time.monotonic()
        for feed_id in fm.feed_ids():
            camera = fm.get_camera(feed_id)
            if camera is None or not camera.is_connected:
                # Retry connection every 5s (AirSim may have started since last attempt)
                if camera is not None and now - _last_reconnect.get(feed_id, 0) >= 5.0:
                    _last_reconnect[feed_id] = now
                    try:
                        if camera.connect():
                            print(f"[CAPTURE] {feed_id}: reconnected to AirSim")
                    except Exception:
                        pass
                continue
            try:
                frame = camera.grab_frame()
                if frame is None:
                    continue

                # Try to get vehicle position (AirSim cameras expose this)
                position = None
                if hasattr(camera, "get_vehicle_position"):
                    position = camera.get_vehicle_position()

                # Encode JPEG for replay buffer (pre-refactor stored frames for replay)
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                jpeg_bytes = buf.tobytes() if buf is not None else None
                timestamp = datetime.utcnow().isoformat() + "Z"

                fm.store_frame(feed_id, frame, position=position,
                               jpeg_bytes=jpeg_bytes, timestamp=timestamp)
            except Exception as e:
                print(f"[CAPTURE] {feed_id}: {e}")
        time.sleep(interval)


def _detection_loop(
    fm: FeedManager,
    cfg: Dict[str, Any],
    pipelines: Dict[str, Any],
    store: Any,
    depth_estimator: Any = None,
    drone_api: Any = None,
) -> None:
    """Run human detection on every frame from all feeds."""
    from datetime import datetime
    from src.services.streaming import render_overlay

    # Create a dedicated AirSim client for coordinate lookups (simGetCameraInfo).
    # Using the camera's shared client from the lifespan context causes an
    # "IOLoop is already running" error because that client was created inside
    # uvicorn's asyncio loop. A client created here (in the background thread,
    # outside asyncio) avoids the conflict — same pattern as the follow mode loop.
    _depth_airsim_client = None
    if depth_estimator is not None:
        try:
            import airsim as _airsim
            _depth_airsim_client = _airsim.MultirotorClient()
            _depth_airsim_client.confirmConnection()
            print("[DETECTION] Dedicated AirSim client for depth coordinate lookup connected")
        except Exception as _e:
            print(f"[DETECTION] Could not create dedicated AirSim client: {_e} — will use camera position fallback")

    det_cfg = cfg.get("detection", {})
    fps = det_cfg.get("fps", 10)
    interval = 1.0 / max(1, fps)
    cooldown = cfg.get("zones", {}).get("alarm_cooldown_seconds", 5.0)

    # Drone navigation state — mirrors pre-refactor FeedManager.drone_is_navigating
    nav = {
        "is_navigating": False,
        "last_deployment_time": 0.0,
        "last_status_check": 0.0,
    }

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
        safe_z = cfg.get("drone", {}).get("safe_altitude", -10.0)

        def _fallback():
            x, y, _ = state.position if state.position else (0.0, 0.0, 0.0)
            return (x, y, safe_z)

        # Without depth estimator or camera that doesn't support AirSim, use camera position
        if not depth_estimator or not person_masks:
            return _fallback()

        # Only AirSim cameras can provide world coordinates
        if not hasattr(camera, "camera_name") or not hasattr(camera, "vehicle_name"):
            return _fallback()

        # If no AirSim client available, use camera position
        if not airsim_client:
            return _fallback()

        try:
            # Get depth map
            depth_map = depth_estimator.estimate(frame)

            # Find centroid of first person mask
            person_mask = person_masks[0]
            y_indices, x_indices = person_mask.nonzero()
            if len(y_indices) == 0:
                return _fallback()

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
            # Always use safe_altitude for z — pre-refactor always overrode z with
            # SAFE_Z_ALTITUDE rather than the raw depth-estimated z (which can be
            # positive / below-ground in AirSim NED coordinates)
            safe_z = cfg.get("drone", {}).get("safe_altitude", -10.0)
            return (world_coord.x_val, world_coord.y_val, safe_z)
        except Exception as e:
            print(f"[DETECTION] Depth estimation failed: {e}, using camera position")
            return _fallback()

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

                # Render overlay for all detected people (unless --no-mask was passed)
                disable_overlay = os.environ.get("DISABLE_MASK_OVERLAY") == "1"
                if result.person_masks and not disable_overlay:
                    overlay = render_overlay(frame, result.person_masks)
                else:
                    overlay = None

                # Update detection state
                fm.update_detection(
                    feed_id,
                    alarm_active=result.alarm_active,
                    caution_active=result.caution_active,
                    people_count=result.people_count,
                    danger_count=result.danger_count,
                    caution_count=result.caution_count,
                    mask_overlay=overlay,
                )

                # If alarm fired, record trigger and auto-deploy search drone
                if result.alarm_fired:
                    snapshot_jpeg = encode_frame_jpeg(frame)
                    replay = list(state.replay_buffer)
                    trigger_idx = len(replay) - 1

                    # Compute 3D coordinates using depth estimation.
                    # Use the dedicated AirSim client (created in this thread) to
                    # avoid IOLoop conflicts with uvicorn's asyncio event loop.
                    coords = get_person_coords(
                        fm,
                        feed_id,
                        frame,
                        result.danger_masks,
                        depth_estimator,
                        _depth_airsim_client,
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

                    # Auto-deploy search drone (same as pre-refactor depth_worker_loop)
                    if drone_api is not None and not nav["is_navigating"]:
                        now = time.time()
                        if now - nav["last_deployment_time"] > cooldown:
                            tx, ty, tz = coords
                            print(f"[DRONE] Auto-deploying to ({tx:.2f}, {ty:.2f}, {tz:.2f}) for {feed_id}")
                            if drone_api.goto_position(tx, ty, tz):
                                nav["last_deployment_time"] = now
                                nav["is_navigating"] = True
                                event.deployed = True
                                print("[DRONE] Deployment command sent")
                            else:
                                print("[DRONE] Deployment command failed")

            except Exception as e:
                print(f"[DETECTION] {feed_id}: {e}")

        # Poll drone status once per second to detect navigation completion
        now = time.time()
        if nav["is_navigating"] and drone_api is not None and now - nav["last_status_check"] >= 1.0:
            nav["last_status_check"] = now
            try:
                status = drone_api.get_status()
                if status and not status.get("is_navigating", True):
                    print("[DRONE] Drone reached target, mission complete")
                    nav["is_navigating"] = False
            except Exception as e:
                print(f"[DRONE] Status check failed: {e}")

        time.sleep(interval)


def _auto_seg_loop(
    fm: FeedManager,
    cfg: Dict[str, Any],
    segmenter: Any,
) -> None:
    """Periodically auto-segment zones on every feed.

    Scene-type behaviour (matches pre-refactor auto_segmentation_loop):
    - ship   : Re-segment every interval_seconds, always overwrite all zones
               (ship deck changes as vessel moves — auto wins over manual).
    - railway/bridge : Segment once on first available frame; afterwards
               skip if user has saved manual zones.
    """
    from src.core.models import Zone

    seg_cfg = cfg.get("auto_segmentation", {})
    interval_seconds = seg_cfg.get("interval_seconds", 60.0)

    # Track which static-scene feeds have already had their initial segmentation
    initial_seg_done: set = set()

    # Wait for initial frames before starting
    time.sleep(5.0)

    while getattr(fm, "_running", False):
        now = time.monotonic()
        for feed_id in fm.feed_ids():
            state = fm.get_state(feed_id)
            if state is None or not state.scene_type:
                continue

            if state.scene_type == "ship":
                # Ship: always re-segment on interval, overwrite everything
                last_seg_time = getattr(state, "last_auto_seg_time", 0)
                if now - last_seg_time < interval_seconds:
                    continue

                frame = fm.get_frame(feed_id)
                if frame is None:
                    continue

                try:
                    zone_dicts = segmenter.segment_frame(frame, state.scene_type)
                    if zone_dicts:
                        zones = [Zone(**z) for z in zone_dicts]
                        # Clear manual zones so auto-seg always wins for ship
                        with state.lock:
                            state.manual_zones = []
                        fm.update_zones(feed_id, zones, frame.shape[1], frame.shape[0], source="auto")
                        state.auto_seg_active = True
                        state.last_auto_seg_time = now
                except Exception as e:
                    print(f"[AUTO-SEG] {feed_id}: {e}")

            else:
                # Railway / Bridge: segment once initially; skip if manual zones exist
                if feed_id in initial_seg_done or state.manual_zones:
                    continue

                frame = fm.get_frame(feed_id)
                if frame is None:
                    continue

                try:
                    zone_dicts = segmenter.segment_frame(frame, state.scene_type)
                    if zone_dicts:
                        zones = [Zone(**z) for z in zone_dicts]
                        fm.update_zones(feed_id, zones, frame.shape[1], frame.shape[0], source="auto")
                        state.auto_seg_active = True
                        state.last_auto_seg_time = now
                        initial_seg_done.add(feed_id)
                        print(f"[AUTO-SEG] Initial segmentation done for {feed_id} ({state.scene_type})")
                except Exception as e:
                    print(f"[AUTO-SEG] {feed_id}: {e}")

        time.sleep(5.0)  # Check every 5 seconds


def _follow_mode_loop(
    fm: FeedManager,
    cfg: Dict[str, Any],
    drone_api,
) -> None:
    """Teleport CCTV drones to match Camera Actor positions in the UE scene.

    Camera Actors (CCTV1-4) are placed in the Unreal Engine World Outliner
    and move with the ship/target.  Each tick this loop reads each Camera
    Actor's pose via simGetObjectPose() and teleports the corresponding drone
    to that exact pose with simSetVehiclePose(), then zeroes its velocity so
    the flight controller does not drift.

    Only runs when follow_mode.target is non-empty (set via --follow <label>).
    """
    import math

    follow_cfg = cfg.get("follow_mode", {})
    target = follow_cfg.get("target", "")
    if not target:
        print("[FOLLOW] Follow mode disabled (no target configured)")
        return

    # {vehicle_name: cam_actor_name} e.g. {"Drone2": "CCTV1", ...}
    camera_mappings: dict = follow_cfg.get("camera_mappings", {})
    if not camera_mappings:
        print("[FOLLOW] No camera_mappings configured — follow mode skipped")
        return

    interval = follow_cfg.get("follow_interval", 0.01)

    print(f"[FOLLOW] Following '{target}' — teleporting {list(camera_mappings.keys())} "
          f"to Camera Actors {list(camera_mappings.values())} @ {interval:.3f}s interval")

    # Dedicated AirSim client to avoid IOLoop conflicts with the capture thread
    try:
        import airsim
        client = airsim.MultirotorClient()
        client.confirmConnection()
        print("[FOLLOW] Dedicated AirSim client connected")
    except Exception as e:
        print(f"[FOLLOW] Could not connect AirSim client: {e}")
        return

    # Arm and take off every CCTV drone, collecting futures so we can join all at once
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
            print(f"[FOLLOW] {vehicle_name} takeoff join (may already be airborne): {e}")

    print("[FOLLOW] All drones airborne — starting teleport loop")

    while getattr(fm, "_running", False):
        try:
            for vehicle_name, cam_actor_name in camera_mappings.items():
                cam_pose = client.simGetObjectPose(cam_actor_name)

                if math.isnan(cam_pose.position.x_val):
                    continue

                # Teleport drone to the Camera Actor's exact pose
                client.simSetVehiclePose(cam_pose, True, vehicle_name=vehicle_name)

                # Zero out velocity so the flight controller doesn't fight back
                client.moveByVelocityAsync(0, 0, 0, 0.1, vehicle_name=vehicle_name)

        except Exception as e:
            print(f"[FOLLOW] Error: {e}")

        time.sleep(interval)


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

    # Register feeds from feeds.yaml — cameras are created but NOT connected here.
    # AirSim RPC calls (enableApiControl, confirmConnection, etc.) are blocking
    # and use tornado IOLoop internally, which deadlocks when called inside
    # uvicorn's asyncio event loop (the lifespan runs in that loop).
    # The capture loop (background thread, outside asyncio) handles connection.
    for feed_id, feed_def in feeds_cfg.items():
        camera_cfg = feed_def.get("camera", {})
        camera = None

        try:
            camera = create_camera_backend(camera_cfg)
        except Exception as e:
            print(f"[INIT] {feed_id}: camera backend error — {e}")

        fm.register_feed(
            feed_id=feed_id,
            name=feed_def.get("name", feed_id),
            location=feed_def.get("location", ""),
            camera=camera,
            scene_type=feed_def.get("scene_type"),
        )
        print(f"[INIT] {feed_id}: registered (connection deferred to capture loop)")

    print(f"[INIT] {len(feeds_cfg)} feed(s) registered — connections deferred to background threads")

    # Restore manually-saved zones from disk (without frame dimensions — masks
    # are regenerated on first real frame via _needs_mask_regen flag)
    try:
        from src.core.models import Zone as _Zone
        from src.services.zone_persistence import load_zones as _load_zones

        zones_file = cfg.get("zones", {}).get("persistence_file", "data/zones.json")
        persisted = _load_zones(zones_file)
        for feed_id, zone_dicts in persisted.items():
            state = fm.get_state(feed_id)
            if state is None:
                continue
            zones = [_Zone(**z) for z in zone_dicts]
            if zones:
                with state.lock:
                    state.manual_zones = zones
                    state.zones = zones
                    state._needs_mask_regen = True
                print(f"[INIT] {feed_id}: restored {len(zones)} manual zone(s) from disk")
    except Exception as e:
        print(f"[INIT] Zone restore failed: {e}")

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
            args=(fm, cfg, pipelines, deps.get_trigger_store(), depth_estimator, deps.get_drone_api()),
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

    # Start follow mode if target is configured
    follow_target = cfg.get("follow_mode", {}).get("target", "")
    if follow_target:
        follow_thread = threading.Thread(
            target=_follow_mode_loop,
            args=(fm, cfg, deps.get_drone_api()),
            daemon=True,
            name="follow-mode",
        )
        follow_thread.start()

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
