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
from concurrent.futures import ThreadPoolExecutor, Future
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
        loop_start = time.monotonic()
        for feed_id in fm.feed_ids():
            camera = fm.get_camera(feed_id)
            if camera is None or not camera.is_connected:
                # Retry connection every 5s (AirSim may have started since last attempt)
                if camera is not None and loop_start - _last_reconnect.get(feed_id, 0) >= 5.0:
                    _last_reconnect[feed_id] = loop_start
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
        # Deadline-based sleep: account for time spent capturing
        elapsed = time.monotonic() - loop_start
        remaining = interval - elapsed
        if remaining > 0:
            time.sleep(remaining)


def _detection_loop(
    fm: FeedManager,
    cfg: Dict[str, Any],
    pipelines: Dict[str, Any],
    store: Any,
    depth_estimator: Any = None,
    drone_api: Any = None,
    gpu_lock: threading.Lock | None = None,
) -> None:
    """Run human detection on every frame from all feeds.

    Uses batched YOLO inference (single GPU call for all feeds) matching
    the pre-refactor design for maximum throughput.
    """
    from datetime import datetime

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
    warmup = det_cfg.get("warmup_frames", 20)
    disable_overlay = os.environ.get("DISABLE_MASK_OVERLAY") == "1"

    # Get the shared detector from any pipeline (all share the same instance)
    detector = None
    for p in pipelines.values():
        detector = p._detector
        break

    # Drone navigation state — mirrors pre-refactor FeedManager.drone_is_navigating
    nav = {
        "is_navigating": False,
        "first_auto_deployed": False,   # True after first auto-deploy; blocks further auto-deploys
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
        loop_start = time.monotonic()

        # --- 1. Collect frames from all eligible feeds ---
        batch_feed_ids = []
        batch_frames = []
        batch_pipelines = []

        for feed_id in fm.feed_ids():
            state = fm.get_state(feed_id)
            if state is None:
                continue
            if not state.detection_enabled:
                continue
            if not fm.is_warmed_up(feed_id, warmup):
                continue
            pipeline = pipelines.get(feed_id)
            if pipeline is None:
                continue

            frame = fm.get_frame(feed_id)
            if frame is None:
                continue

            # Skip feeds without zones (no detection needed) — matches old server
            zm = pipeline._zone_manager
            if zm.red_mask is None and zm.yellow_mask is None:
                continue

            batch_feed_ids.append(feed_id)
            batch_frames.append(frame)
            batch_pipelines.append(pipeline)

        # --- 2. Run batched YOLO inference (single GPU call) ---
        if batch_frames and detector is not None:
            try:
                if gpu_lock is not None:
                    gpu_lock.acquire()
                try:
                    batch_masks = detector.get_masks_batch(batch_frames)
                finally:
                    if gpu_lock is not None:
                        gpu_lock.release()
            except Exception as e:
                print(f"[DETECTION] Batch YOLO failed: {e}")
                batch_masks = [[] for _ in batch_frames]

            # --- 3. Process results per feed (zone checks, alarms, overlays) ---
            for idx, feed_id in enumerate(batch_feed_ids):
                pipeline = batch_pipelines[idx]
                frame = batch_frames[idx]
                person_masks = batch_masks[idx]
                state = fm.get_state(feed_id)

                try:
                    # Zone overlap checks (CPU-only, fast)
                    zm = pipeline._zone_manager
                    alarm = pipeline._alarm
                    pipeline._frame_count += 1

                    people_count = len(person_masks)
                    is_alarm = False
                    is_caution = False
                    danger_count = 0
                    caution_count = 0
                    danger_masks = []
                    caution_masks = []
                    alarm_fired = False

                    if person_masks:
                        is_alarm, danger_masks = zm.check_red(person_masks)
                        danger_count = len(danger_masks)
                        if is_alarm:
                            alarm_fired = alarm.trigger()
                        else:
                            alarm.clear()
                        is_caution, caution_masks = zm.check_yellow(person_masks)
                        caution_count = len(caution_masks)
                    else:
                        alarm.clear()

                    # Build combined binary mask (single uint8 array) instead of
                    # a full RGB overlay canvas — cheaper to create here and to
                    # composite in the video route.
                    combined_mask = None
                    if person_masks and not disable_overlay:
                        combined_mask = person_masks[0].copy()
                        for m in person_masks[1:]:
                            np.maximum(combined_mask, m, out=combined_mask)

                    fm.update_detection(
                        feed_id,
                        alarm_active=is_alarm,
                        caution_active=is_caution,
                        people_count=people_count,
                        danger_count=danger_count,
                        caution_count=caution_count,
                        mask_overlay=combined_mask,
                    )

                    # If alarm fired, record trigger and auto-deploy search drone
                    if alarm_fired:
                        snapshot_jpeg = encode_frame_jpeg(frame)
                        replay = list(state.replay_buffer)
                        trigger_idx = len(replay) - 1

                        coords = get_person_coords(
                            fm, feed_id, frame, danger_masks,
                            depth_estimator, _depth_airsim_client,
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

                        # Auto-deploy only for the very first trigger.
                        # After that, only manual "Redirect to this Location" can move the drone.
                        # Auto-deploy re-enables only after RTH + landing (grounded + automatic).
                        if (drone_api is not None
                                and not nav["is_navigating"]
                                and not nav["first_auto_deployed"]):
                            now = time.time()
                            if now - nav["last_deployment_time"] > cooldown:
                                tx, ty, tz = coords
                                print(f"[DRONE] First auto-deploy to ({tx:.2f}, {ty:.2f}, {tz:.2f}) for {feed_id}")
                                if drone_api.goto_position(tx, ty, tz):
                                    nav["last_deployment_time"] = now
                                    nav["is_navigating"] = True
                                    nav["first_auto_deployed"] = True
                                    event.deployed = True
                                    print("[DRONE] Deployment command sent — subsequent deploys are manual only")
                                else:
                                    print("[DRONE] Deployment command failed")

                except Exception as e:
                    print(f"[DETECTION] {feed_id}: {e}")

        # Poll drone status once per second to detect navigation completion
        now = time.time()
        if drone_api is not None and now - nav["last_status_check"] >= 1.0:
            nav["last_status_check"] = now
            try:
                status = drone_api.get_status()
                if status:
                    if nav["is_navigating"] and not status.get("is_navigating", True):
                        print("[DRONE] Drone reached target, mission complete")
                        nav["is_navigating"] = False
                    # Reset auto-deploy gate only when drone is grounded at home
                    # (after RTH + landing). This is the ONLY path back to auto-deploy.
                    if (nav["first_auto_deployed"]
                            and status.get("grounded", False)
                            and status.get("mode") == "automatic"):
                        print("[DRONE] Drone grounded at home — auto-deploy re-enabled")
                        nav["first_auto_deployed"] = False
            except Exception as e:
                print(f"[DRONE] Status check failed: {e}")

        # Deadline-based sleep: account for time spent on inference + processing
        elapsed = time.monotonic() - loop_start
        remaining = interval - elapsed
        if remaining > 0:
            time.sleep(remaining)


def _auto_seg_loop(
    fm: FeedManager,
    cfg: Dict[str, Any],
    segmenter: Any,
    gpu_lock: threading.Lock | None = None,
) -> None:
    """Periodically auto-segment zones on every feed.

    Scene-type behaviour:
    - ship   : Re-segment every interval_seconds. Auto zones are merged with
               manual zones (manual zones layer on top with higher priority).
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
                # Ship: re-segment on interval only when auto-refresh is enabled
                if not state.auto_refresh_enabled:
                    continue
                last_seg_time = getattr(state, "last_auto_seg_time", 0)
                if now - last_seg_time < interval_seconds:
                    continue

                frame = fm.get_frame(feed_id)
                if frame is None:
                    continue

                try:
                    if gpu_lock is not None:
                        gpu_lock.acquire()
                    try:
                        zone_dicts = segmenter.segment_frame(frame, state.scene_type)
                    finally:
                        if gpu_lock is not None:
                            gpu_lock.release()
                    if zone_dicts:
                        zones = [Zone(**z) for z in zone_dicts]
                        # Auto zones are merged with (not replacing) manual zones.
                        # Manual zones layer on top with higher priority.
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
                    if gpu_lock is not None:
                        gpu_lock.acquire()
                    try:
                        zone_dicts = segmenter.segment_frame(frame, state.scene_type)
                    finally:
                        if gpu_lock is not None:
                            gpu_lock.release()
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

    # Start capture thread EARLY so cameras connect and accumulate warmup
    # frames while models load (models can take minutes on first run).
    fm._running = True
    capture_thread = threading.Thread(
        target=_capture_loop, args=(fm, cfg), daemon=True, name="frame-capture"
    )
    capture_thread.start()
    print("[INIT] Capture thread started (cameras connecting in background)")

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

    # Load all ML models in parallel — each model load is independent and
    # the sequential loading was the main startup bottleneck.
    print("[INIT] Loading ML models in parallel...")
    _load_start = time.monotonic()

    def _load_human_detector():
        from src.detection.human_detector import HumanDetector
        return HumanDetector()

    def _load_scene_segmenter():
        from src.detection.scene_segmenter import SceneSegmenter
        return SceneSegmenter()

    def _load_depth_estimator():
        from src.detection.depth_estimator_wrapper import DepthEstimator
        enc_path = cfg.get("depth_estimation", {}).get("encoder_path")
        dec_path = cfg.get("depth_estimation", {}).get("decoder_path")
        if enc_path and dec_path:
            return DepthEstimator(encoder_path=enc_path, decoder_path=dec_path)
        return None

    detector = None
    segmenter = None
    depth_estimator = None

    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="model-load") as pool:
        fut_detector: Future = pool.submit(_load_human_detector)
        fut_segmenter: Future = pool.submit(_load_scene_segmenter)
        fut_depth: Future = pool.submit(_load_depth_estimator)

        try:
            detector = fut_detector.result()
            print("[INIT] Human detector loaded")
        except Exception as e:
            print(f"[INIT] Human detector failed to load: {e}")

        try:
            segmenter = fut_segmenter.result()
            if segmenter is not None:
                deps.set_scene_segmenter(segmenter)
            print("[INIT] Scene segmenter loaded")
        except Exception as e:
            print(f"[INIT] Scene segmenter failed to load: {e}")

        try:
            depth_estimator = fut_depth.result()
            if depth_estimator is not None:
                deps.set_depth_estimator(depth_estimator)
                print("[INIT] Depth estimator loaded")
        except Exception as e:
            print(f"[INIT] Depth estimator failed to load: {e}")

    print(f"[INIT] All models loaded in {time.monotonic() - _load_start:.1f}s")

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

    # Shared GPU lock — serializes CUDA inference across detection and
    # auto-segmentation threads (CUDA isn't thread-safe).
    gpu_lock = threading.Lock()

    # Start remaining background threads (capture thread already running)
    if pipelines:
        detection_thread = threading.Thread(
            target=_detection_loop,
            args=(fm, cfg, pipelines, deps.get_trigger_store(), depth_estimator, deps.get_drone_api(), gpu_lock),
            daemon=True,
            name="detection",
        )
        detection_thread.start()

    if segmenter is not None:
        auto_seg_thread = threading.Thread(
            target=_auto_seg_loop,
            args=(fm, cfg, segmenter, gpu_lock),
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
