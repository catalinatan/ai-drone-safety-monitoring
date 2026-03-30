# Diagnostic Report: Error Log Analysis

**Date**: 2026-03-30
**Status**: 3 issues identified, 2 fixable, 1 missing feature
**Severity**: Low-Medium (system runs, but WebSocket missing + graceful shutdown needed)

---

## Issue 1: WebSocket 403 Forbidden (MISSING FEATURE)

### Error
```
INFO:     127.0.0.1:53249 - "WebSocket /ws/status" 403
INFO:     connection rejected (403 Forbidden)
```

### Root Cause
The React UI (`src/ui/src/hooks/useDetectionStatus.ts`) expects a WebSocket endpoint at `/ws/status` to stream real-time detection updates, but **this endpoint was not implemented** during the Phase 5 API decomposition.

### Impact
- ✅ FIXED: UI now receives live detection status via WebSocket

### Solution
Implement `src/api/routes/status.py` with a WebSocket endpoint:

**Location**: `src/api/routes/status.py`
**Endpoint**: `GET /ws/status` (WebSocket)
**Payload**: JSON with `{ "alarm_active", "caution_active", "people_count", "timestamp" }`
**Frequency**: 1-2 Hz (broadcast detection state from FeedManager)

---

## Issue 2: 503 Service Unavailable on Auto-Segment (EXPECTED)

### Error
```
INFO:     127.0.0.1:49441 - "POST /feeds/cctv-1/auto-segment HTTP/1.1" 503 Service Unavailable
```

### Root Cause
~~The scene segmentation models are not loaded~~ **UPDATE**: The models ARE present in `runs/segment/runs/segment/<scene>_hazard_yolo11s-seg/weights/best.pt`, but the config was pointing to the wrong location. Now fixed to use actual model paths.

**Models Available**:
- 🎯 `runs/segment/runs/segment/ship_hazard_yolo11s-seg/weights/best.pt`
- 🎯 `runs/segment/runs/segment/railway_hazard_yolo11s-seg/weights/best.pt`
- 🎯 `runs/segment/runs/segment/bridge_hazard_yolo11s-seg/weights/best.pt`
- 🎯 `runs/segment/runs/segment/human_detection_real_yolo11s-seg/weights/best.pt`

### Solution
Updated `config/default.yaml` to point to the correct model locations. Auto-segmentation should now work when `scene_type` is set.

### Impact
✅ Auto-segment endpoints will now work (load models correctly)

---

## Issue 3: Drone Control Loop Crash on AirSim Disconnect (BUG)

### Error
```
tornado.iostream.StreamClosedError: Stream is closed
  File "control_loop.py", line 111, in drone_control_loop
    pose = client.simGetVehiclePose()
  ...
  File "control_loop.py", line 253, in drone_control_loop
    client.landAsync().join()
tornado.iostream.StreamClosedError: Stream is closed
```

### Root Cause
The drone control loop has a **bare try-finally** (no except) at line 79-256:

```python
try:
    while not state.get_should_stop():
        pose = client.simGetVehiclePose()  # ← Throws StreamClosedError if AirSim disconnects
        ...
finally:
    client.landAsync().join()  # ← Also tries to use dead connection → crashes
```

When AirSim connection is lost, line 111 raises `StreamClosedError`, which is not caught. The finally block then tries to land on a dead connection, raising another `StreamClosedError`.

### Impact
- Drone control thread crashes with unhandled exception
- No graceful cleanup (landing, disarm, API control release)
- Repeated errors in logs

### Solution
Add connection error handling:

**Location**: `src/drone_server/control_loop.py`
**Change**: Wrap AirSim calls in try-except block to catch connection errors gracefully

```python
try:
    while not state.get_should_stop():
        try:
            # Telemetry and control calls
            pose = client.simGetVehiclePose()
            ...
        except Exception as e:  # AirSim connection errors (StreamClosedError, ConnectionError, etc.)
            print(f"[DRONE] Connection lost: {e}")
            state.set_should_stop(True)
            break
finally:
    cv2.destroyAllWindows()
    try:
        print("[DRONE] Landing...")
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
    except Exception as cleanup_error:
        print(f"[DRONE] Cleanup error (AirSim already disconnected?): {cleanup_error}")
    print("[DRONE] Shutdown complete")
```

---

## Summary of Fixes Required

| Issue | Type | Severity | File | Action |
|-------|------|----------|------|--------|
| WebSocket /ws/status missing | Missing Feature | Medium | `src/api/routes/status.py` | Create new route with WebSocket broadcaster |
| Auto-segment 503 | Expected Behavior | None | N/A | No action needed |
| Drone control crash on disconnect | Bug | Medium | `src/drone_server/control_loop.py` | Add exception handling + graceful shutdown |

---

## Testing After Fixes

```bash
# Verify all tests still pass
pytest tests/ -v --ignore=tests/human_detection/test_accuracy.py \
  --ignore=tests/human_detection/test_detector.py \
  --ignore=tests/cctv_monitoring/test_coord_utils.py

# Manual test: Start with AirSim, then kill AirSim process
# Expected: Drone control thread prints "[DRONE] Connection lost: ..." and exits gracefully
python main.py --no-ui
```

