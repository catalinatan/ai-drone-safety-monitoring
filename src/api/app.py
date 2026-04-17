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
from concurrent.futures import Future, ThreadPoolExecutor
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
# Projection factory
# ---------------------------------------------------------------------------


def _create_projection(feed_id: str, feed_def: dict, cfg: dict):
    """
    Create the appropriate projection backend for a feed.

    AirSim cameras get AirSimProjection (client set later from detection thread).
    Config-based cameras get ConfigProjection using position/orientation from feeds.yaml.
    """

    camera_type = feed_def.get("camera", {}).get("type", "")
    safe_z = cfg.get("drone", {}).get("safe_altitude", -10.0)
    cctv_height = cfg.get("detection", {}).get("cctv_height_meters", 10.0)

    if camera_type == "airsim":
        from src.spatial.airsim_projection import AirSimProjection

        params = feed_def.get("camera", {}).get("params", {})
        return AirSimProjection(
            camera_name=params.get("camera_name", "0"),
            vehicle_name=params.get("vehicle_name", ""),
            cctv_height=cctv_height,
            safe_z=safe_z,
            auto_height=True,
        )
    else:
        from src.spatial.config_projection import ConfigProjection

        pos_cfg = feed_def.get("position", {})
        ori_cfg = feed_def.get("orientation", {})
        orientation = (
            ori_cfg.get("pitch", 0.0),
            ori_cfg.get("yaw", 0.0),
            ori_cfg.get("roll", 0.0),
        )
        fov = feed_def.get("fov", 90.0)

        # Position: accept GPS (lat/lon/alt) or legacy NED (x/y/z)
        gps_origin = None
        if "latitude" in pos_cfg and "longitude" in pos_cfg:
            lat = pos_cfg.get("latitude", 0.0)
            lon = pos_cfg.get("longitude", 0.0)
            alt = pos_cfg.get("altitude", 0.0)
            gps_origin = (lat, lon, alt)
            # Use this camera's GPS as its own origin → NED (0, 0, 0)
            # The origin will be shared across feeds via _gps_origin
            position = (0.0, 0.0, 0.0)
        else:
            position = (
                pos_cfg.get("x", 0.0),
                pos_cfg.get("y", 0.0),
                pos_cfg.get("z", 0.0),
            )

        return ConfigProjection(
            position=position,
            orientation=orientation,
            fov=fov,
            safe_z=safe_z,
            gps_origin=gps_origin,
        )


# ---------------------------------------------------------------------------
# Background threads
# ---------------------------------------------------------------------------


def _capture_loop(fm: FeedManager, cfg: Dict[str, Any]) -> None:
    """Continuously grab frames from all registered camera backends."""
    from datetime import datetime

    import cv2

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

                fm.store_frame(
                    feed_id, frame, position=position, jpeg_bytes=jpeg_bytes, timestamp=timestamp
                )
            except Exception as e:
                print(f"[CAPTURE] {feed_id}: {e}")
        # Deadline-based sleep: account for time spent capturing
        elapsed = time.monotonic() - loop_start
        remaining = interval - elapsed
        if remaining > 0:
            time.sleep(remaining)


def _drone_status_loop(
    fm: FeedManager,
    drone_api: Any,
    nav: Dict[str, Any],
) -> None:
    """Poll drone status once per second in a dedicated thread.

    Detects navigation completion and re-enables auto-deploy after RTH + landing.
    Runs independently of the detection loop to avoid blocking inference.
    """
    while getattr(fm, "_running", False):
        try:
            status = drone_api.get_status()
            if status:
                if nav["is_navigating"] and not status.get("is_navigating", True):
                    print("[DRONE] Drone reached target, mission complete")
                    nav["is_navigating"] = False
                # Reset auto-deploy gate only when drone is grounded at home
                # (after RTH + landing). This is the ONLY path back to auto-deploy.
                if (
                    nav["first_auto_deployed"]
                    and status.get("grounded", False)
                    and status.get("mode") == "automatic"
                ):
                    print("[DRONE] Drone grounded at home — auto-deploy re-enabled")
                    nav["first_auto_deployed"] = False
        except Exception as e:
            print(f"[DRONE] Status check failed: {e}")
        time.sleep(1.0)


def _detection_loop(
    fm: FeedManager,
    cfg: Dict[str, Any],
    pipelines: Dict[str, Any],
    store: Any,
    depth_estimator: Any = None,
    drone_api: Any = None,
    gpu_lock: threading.Lock | None = None,
    nav: Dict[str, Any] | None = None,
    projections: Dict[str, Any] | None = None,
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
            # Wire the AirSim client into any AirSimProjection backends
            if projections:
                from src.spatial.airsim_projection import AirSimProjection

                for _proj in projections.values():
                    if isinstance(_proj, AirSimProjection):
                        _proj.set_client(_depth_airsim_client)
        except Exception as _e:
            print(
                f"[DETECTION] Could not create dedicated AirSim client: {_e} "
                "— will use camera position fallback"
            )

    det_cfg = cfg.get("detection", {})
    fps = det_cfg.get("fps", 10)
    interval = 1.0 / max(1, fps)
    print(f"[DETECTION] Target FPS: {fps} (interval: {interval:.3f}s)")
    cooldown = cfg.get("zones", {}).get("alarm_cooldown_seconds", 5.0)
    warmup = det_cfg.get("warmup_frames", 20)
    disable_overlay = os.environ.get("DISABLE_MASK_OVERLAY") == "1"

    # Get the shared detector from any pipeline (all share the same instance)
    detector = None
    for p in pipelines.values():
        detector = p._detector
        break

    # Drone navigation state — shared with _drone_status_loop thread
    if nav is None:
        nav = {
            "is_navigating": False,
            "first_auto_deployed": False,
            "last_deployment_time": 0.0,
        }

    def encode_frame_jpeg(frame):
        """Encode numpy array as JPEG bytes."""
        import cv2

        _, buf = cv2.imencode(".jpg", frame)
        return buf.tobytes() if buf is not None else b""

    def get_person_coords(fm, feed_id, frame, person_masks, depth_estimator, projections):
        """
        Estimate 3D world coordinates for the detected person.
        Uses the feed's projection backend (AirSim or Config-based).
        """
        state = fm.get_state(feed_id)
        safe_z = cfg.get("drone", {}).get("safe_altitude", -10.0)

        def _fallback():
            x, y, _ = state.position if state.position else (0.0, 0.0, 0.0)
            return (x, y, safe_z)

        projection = (projections or {}).get(feed_id)
        if projection is None:
            return _fallback()

        # Update projection's fallback position from latest camera pose
        if state.position:
            projection.update_pose(position=state.position)

        if not depth_estimator or not person_masks:
            return _fallback()

        try:
            depth_map = depth_estimator.estimate(frame)
            person_mask = person_masks[0]
            y_indices, x_indices = person_mask.nonzero()
            if len(y_indices) == 0:
                return _fallback()

            center_x = float(np.mean(x_indices))
            center_y = float(np.mean(y_indices))
            depth_val = depth_estimator.get_depth_at_pixel(depth_map, center_x, center_y)

            return projection.pixel_to_world(
                center_x,
                center_y,
                depth_val,
                frame.shape[1],
                frame.shape[0],
            )
        except Exception as e:
            print(f"[DETECTION] Projection failed: {e}, using camera position")
            return _fallback()

    # Timing instrumentation — accumulate and log every 100 cycles
    _t_cycles = 0
    _t_total = 0.0
    _t_collect = 0.0
    _t_lock = 0.0
    _t_infer = 0.0
    _t_post = 0.0
    _t_zone = 0.0
    _t_mask = 0.0
    _t_update = 0.0
    _t_alarm = 0.0
    _t_drone = 0.0

    while getattr(fm, "_running", False):
        loop_start = time.monotonic()

        # --- 1. Collect frames from all eligible feeds ---
        _t0 = time.monotonic()
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
        _t1 = time.monotonic()
        _t_collect += _t1 - _t0
        if batch_frames and detector is not None:
            try:
                _t_lock_start = time.monotonic()
                if gpu_lock is not None:
                    if not gpu_lock.acquire(timeout=0.005):
                        _t_lock += time.monotonic() - _t_lock_start
                        print("[DETECTION] GPU lock contention, skipping cycle")
                        continue
                _t_lock_end = time.monotonic()
                _t_lock += _t_lock_end - _t_lock_start
                try:
                    batch_masks = detector.get_masks_batch(batch_frames)
                finally:
                    _t_infer += time.monotonic() - _t_lock_end
                    if gpu_lock is not None:
                        gpu_lock.release()
            except Exception as e:
                print(f"[DETECTION] Batch YOLO failed: {e}")
                batch_masks = [[] for _ in batch_frames]

            # --- 3. Process results per feed (zone checks, alarms, overlays) ---
            _t_post_start = time.monotonic()
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

                    _t_zone_start = time.monotonic()
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
                    _t_zone += time.monotonic() - _t_zone_start

                    # Build combined binary mask (single uint8 array) instead of
                    # a full RGB overlay canvas — cheaper to create here and to
                    # composite in the video route.
                    _t_mask_start = time.monotonic()
                    combined_mask = None
                    if person_masks and not disable_overlay:
                        combined_mask = person_masks[0].copy()
                        for m in person_masks[1:]:
                            np.maximum(combined_mask, m, out=combined_mask)
                    _t_mask += time.monotonic() - _t_mask_start

                    _t_update_start = time.monotonic()
                    fm.update_detection(
                        feed_id,
                        alarm_active=is_alarm,
                        caution_active=is_caution,
                        people_count=people_count,
                        danger_count=danger_count,
                        caution_count=caution_count,
                        mask_overlay=combined_mask,
                    )
                    _t_update += time.monotonic() - _t_update_start

                    # If alarm fired, record trigger and auto-deploy search drone
                    _t_alarm_start = time.monotonic()
                    if alarm_fired:
                        import cv2 as _cv2

                        # Burn the danger-only mask onto the trigger frame so
                        # the replay highlights the person who caused the alarm,
                        # regardless of the --no-mask flag.
                        trigger_frame = _cv2.cvtColor(frame, _cv2.COLOR_RGB2BGR)
                        if danger_masks:
                            _dmask = danger_masks[0].copy()
                            for _dm in danger_masks[1:]:
                                np.maximum(_dmask, _dm, out=_dmask)
                            if _dmask.shape[:2] != trigger_frame.shape[:2]:
                                _dmask = _cv2.resize(
                                    _dmask,
                                    (trigger_frame.shape[1], trigger_frame.shape[0]),
                                    interpolation=_cv2.INTER_NEAREST,
                                )
                            _mb = _dmask.astype(bool)
                            trigger_frame[_mb] = (
                                trigger_frame[_mb] * 0.6
                                + np.array([0, 0, 255], dtype=np.float32) * 0.4
                            ).astype(np.uint8)

                        snapshot_jpeg = encode_frame_jpeg(trigger_frame)
                        raw_replay = list(state.replay_buffer)
                        trigger_idx = len(raw_replay) - 1

                        # Burn person masks onto ALL replay frames so the
                        # replay shows the person's trajectory leading up to
                        # the danger-zone entry.  Cyan for tracking, red for
                        # the trigger frame (danger masks).
                        replay = []
                        for _ri, _entry in enumerate(raw_replay):
                            _ts, _jpeg = _entry[0], _entry[1]
                            _mask = _entry[2] if len(_entry) > 2 else None

                            if _ri == trigger_idx:
                                # Trigger frame — use the danger-mask overlay
                                # (red, already composited onto trigger_frame).
                                _, _buf = _cv2.imencode(
                                    ".jpg",
                                    trigger_frame,
                                    [_cv2.IMWRITE_JPEG_QUALITY, 70],
                                )
                                if _buf is not None:
                                    _jpeg = _buf.tobytes()
                            elif _mask is not None:
                                # Earlier frame — burn person mask in cyan
                                _dec = _cv2.imdecode(
                                    np.frombuffer(_jpeg, np.uint8),
                                    _cv2.IMREAD_COLOR,
                                )
                                if _dec is not None:
                                    if _mask.shape[:2] != _dec.shape[:2]:
                                        _mask = _cv2.resize(
                                            _mask,
                                            (_dec.shape[1], _dec.shape[0]),
                                            interpolation=_cv2.INTER_NEAREST,
                                        )
                                    _mb2 = _mask.astype(bool)
                                    _dec[_mb2] = (
                                        _dec[_mb2] * 0.6
                                        + np.array([255, 255, 0], dtype=np.float32) * 0.4
                                    ).astype(np.uint8)
                                    _, _buf = _cv2.imencode(
                                        ".jpg",
                                        _dec,
                                        [_cv2.IMWRITE_JPEG_QUALITY, 70],
                                    )
                                    if _buf is not None:
                                        _jpeg = _buf.tobytes()

                            replay.append((_ts, _jpeg))

                        coords = get_person_coords(
                            fm,
                            feed_id,
                            frame,
                            danger_masks,
                            depth_estimator,
                            projections,
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
                        if (
                            drone_api is not None
                            and not nav["is_navigating"]
                            and not nav["first_auto_deployed"]
                        ):
                            now = time.time()
                            if now - nav["last_deployment_time"] > cooldown:
                                tx, ty, tz = coords
                                print(
                                    f"[DRONE] First auto-deploy to "
                                    f"({tx:.2f}, {ty:.2f}, {tz:.2f}) for {feed_id}"
                                )
                                if drone_api.goto_position(tx, ty, tz):
                                    nav["last_deployment_time"] = now
                                    nav["is_navigating"] = True
                                    nav["first_auto_deployed"] = True
                                    event.deployed = True
                                    print(
                                        "[DRONE] Deployment command sent "
                                    "— subsequent deploys are manual only"
                                    )
                                else:
                                    print("[DRONE] Deployment command failed")

                    _t_alarm += time.monotonic() - _t_alarm_start

                except Exception as e:
                    print(f"[DETECTION] {feed_id}: {e}")

        _t_drone_start = time.monotonic()
        _t_drone += time.monotonic() - _t_drone_start

        # --- Timing report ---
        _t_cycle_end = time.monotonic()
        if batch_frames and detector is not None:
            _t_post += _t_cycle_end - _t_post_start
        _t_total += _t_cycle_end - loop_start
        _t_cycles += 1
        if _t_cycles >= 100:
            avg_total = (_t_total / _t_cycles) * 1000
            avg_infer = (_t_infer / _t_cycles) * 1000
            avg_post = (_t_post / _t_cycles) * 1000
            avg_lock = (_t_lock / _t_cycles) * 1000
            avg_fps = _t_cycles / _t_total if _t_total > 0 else 0
            avg_zone = (_t_zone / _t_cycles) * 1000
            avg_mask = (_t_mask / _t_cycles) * 1000
            avg_update = (_t_update / _t_cycles) * 1000
            avg_alarm = (_t_alarm / _t_cycles) * 1000
            avg_drone = (_t_drone / _t_cycles) * 1000
            print(
                f"[DETECTION] FPS report (last {_t_cycles} cycles): "
                f"avg {avg_fps:.1f} FPS | "
                f"inference {avg_infer:.1f}ms | "
                f"post {avg_post:.1f}ms "
                f"[zone {avg_zone:.1f} | mask {avg_mask:.1f} | update {avg_update:.1f} | "
                f"alarm {avg_alarm:.1f} | drone {avg_drone:.1f}] | "
                f"lock-wait {avg_lock:.1f}ms | "
                f"total {avg_total:.1f}ms"
            )
            _t_cycles = 0
            _t_total = 0.0
            _t_collect = 0.0
            _t_lock = 0.0
            _t_infer = 0.0
            _t_post = 0.0
            _t_zone = 0.0
            _t_mask = 0.0
            _t_update = 0.0
            _t_alarm = 0.0
            _t_drone = 0.0

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

    Behaviour is controlled by ``auto_segmentation`` in the runtime config:
    - ``enabled``  — when False the loop idles (no re-segmentation).
    - ``scene_type`` — the scene model to use for all feeds.
    - ``interval_seconds`` — how often to re-segment (applies to all scenes).

    On first run (or when auto-refresh is disabled), each feed gets one
    initial segmentation pass.  Subsequent passes only happen while
    ``enabled`` is True.
    """
    from src.core.models import Zone

    # Track which feeds have had their initial segmentation
    initial_seg_done: set = set()
    last_scene_type: str | None = None

    # Wait for initial frames before starting
    time.sleep(5.0)

    while getattr(fm, "_running", False):
        seg_cfg = cfg.get("auto_segmentation", {})
        enabled = seg_cfg.get("enabled", False)
        scene_type = seg_cfg.get("scene_type", "bridge")
        interval_seconds = seg_cfg.get("interval_seconds", 60.0)

        # If scene type changed, reset initial segmentation so all feeds re-segment
        if scene_type != last_scene_type:
            if last_scene_type is not None:
                print(
                    f"[AUTO-SEG] Scene type changed: {last_scene_type} "
                    f"-> {scene_type}, re-segmenting all feeds"
                )
                initial_seg_done.clear()
            last_scene_type = scene_type

        now = time.monotonic()
        for feed_id in fm.feed_ids():
            state = fm.get_state(feed_id)
            if state is None:
                continue

            # Keep feed scene_type in sync with global config
            state.scene_type = scene_type

            # Initial segmentation — run once per feed regardless of enabled flag
            if feed_id not in initial_seg_done:
                frame = fm.get_frame(feed_id)
                if frame is None:
                    continue
                try:
                    if gpu_lock is not None:
                        gpu_lock.acquire()
                    try:
                        zone_dicts = segmenter.segment_frame(frame, scene_type)
                    finally:
                        if gpu_lock is not None:
                            gpu_lock.release()
                    if zone_dicts:
                        zones = [Zone(**z) for z in zone_dicts]
                        fm.update_zones(
                            feed_id, zones, frame.shape[1], frame.shape[0], source="auto"
                        )
                        state.auto_seg_active = True
                        state.last_auto_seg_time = now
                    initial_seg_done.add(feed_id)
                    print(f"[AUTO-SEG] Initial segmentation done for {feed_id} ({scene_type})")
                except Exception as e:
                    print(f"[AUTO-SEG] {feed_id}: {e}")
                continue

            # Periodic refresh — only when enabled
            if not enabled:
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
                    zone_dicts = segmenter.segment_frame(frame, scene_type)
                finally:
                    if gpu_lock is not None:
                        gpu_lock.release()
                if zone_dicts:
                    zones = [Zone(**z) for z in zone_dicts]
                    fm.update_zones(feed_id, zones, frame.shape[1], frame.shape[0], source="auto")
                    state.auto_seg_active = True
                    state.last_auto_seg_time = now
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

    print(
        f"[FOLLOW] Following '{target}' — teleporting {list(camera_mappings.keys())} "
        f"to Camera Actors {list(camera_mappings.values())} @ {interval:.3f}s interval"
    )

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

        # Extract camera pose from config (for real-world cameras)
        camera_pose = None
        pos_cfg = feed_def.get("position")
        ori_cfg = feed_def.get("orientation")
        if pos_cfg or ori_cfg:
            # Store GPS coordinates in camera_pose if available
            gps = None
            if pos_cfg and "latitude" in pos_cfg:
                gps = {
                    "latitude": pos_cfg.get("latitude", 0.0),
                    "longitude": pos_cfg.get("longitude", 0.0),
                    "altitude": pos_cfg.get("altitude", 0.0),
                }
            camera_pose = {
                "gps": gps,
                "orientation": (
                    ori_cfg.get("pitch", 0.0) if ori_cfg else 0.0,
                    ori_cfg.get("yaw", 0.0) if ori_cfg else 0.0,
                    ori_cfg.get("roll", 0.0) if ori_cfg else 0.0,
                ),
                "fov": feed_def.get("fov", 90.0),
            }

        # Use per-feed scene_type if defined, else fall back to global config
        global_scene_type = cfg.get("auto_segmentation", {}).get("scene_type", "bridge")
        fm.register_feed(
            feed_id=feed_id,
            name=feed_def.get("name", feed_id),
            location=feed_def.get("location", ""),
            camera=camera,
            scene_type=feed_def.get("scene_type") or global_scene_type,
            camera_pose=camera_pose,
        )
        print(f"[INIT] {feed_id}: registered (connection deferred to capture loop)")

    # Create per-feed projection backends
    projections = {}
    for feed_id, feed_def in feeds_cfg.items():
        projections[feed_id] = _create_projection(feed_id, feed_def, cfg)
    print(f"[INIT] {len(projections)} projection backend(s) created")

    # Store on app.state so routes can access projection backends
    app.state.projections = projections

    print(
        f"[INIT] {len(feeds_cfg)} feed(s) registered — connections deferred to background threads"
    )

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
            from src.core.alarm import AlarmState
            from src.core.detection_pipeline import DetectionPipeline

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

    # Shared drone navigation state — accessed by detection loop (writes)
    # and drone status polling thread (reads/clears).
    nav = {
        "is_navigating": False,
        "first_auto_deployed": False,
        "last_deployment_time": 0.0,
    }

    # Start remaining background threads (capture thread already running)
    drone_api = deps.get_drone_api()
    if pipelines:
        detection_thread = threading.Thread(
            target=_detection_loop,
            args=(
                fm,
                cfg,
                pipelines,
                deps.get_trigger_store(),
                depth_estimator,
                drone_api,
                gpu_lock,
                nav,
                projections,
            ),
            daemon=True,
            name="detection",
        )
        detection_thread.start()

    if drone_api is not None:
        drone_poll_thread = threading.Thread(
            target=_drone_status_loop,
            args=(fm, drone_api, nav),
            daemon=True,
            name="drone-poll",
        )
        drone_poll_thread.start()

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
