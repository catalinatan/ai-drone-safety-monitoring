# Detection FPS Bottleneck — Diagnostic & Fix Design

**Date:** 2026-04-02
**Problem:** Human mask overlays appear at low FPS / laggy relative to the video stream.
**Goal:** Identify and fix the detection pipeline bottleneck so masks update at real-time rates (~30 FPS), matching the capture and stream FPS.

## Context

The detection pipeline flows:

```
Camera (30 FPS) → store_frame() → detection thread → YOLO batch inference → zone checks → mask overlay → MJPEG stream (30 FPS)
```

- **Capture FPS:** 30 (`streaming.capture_fps`)
- **Detection FPS:** configured as 30 (`detection.fps`), but code defaults to 10 if config key is missing (`app.py:120`)
- **Stream FPS:** 30 (`streaming.stream_fps`)
- **GPU:** Dedicated with headroom
- **Feeds:** 4 active, should support more
- **Model:** YOLO11s-seg at 1280px, TensorRT-accelerated
- **Resolution:** Must stay at 1280px (no reduction)

The mask overlay is composited in the MJPEG stream route (`video.py:55-70`), blending the latest `last_mask_overlay` onto the latest `last_frame`. If detection runs slower than the stream, masks appear stale/laggy.

## Design

### 1. Verify Resolved Detection FPS (Fix)

**File:** `src/api/app.py` (detection loop startup, around line 120)

Add a log line after resolving the detection FPS from config:

```python
fps = det_cfg.get("fps", 10)
print(f"[DETECTION] Target FPS: {fps} (interval: {1.0 / max(1, fps):.3f}s)")
```

This confirms whether the config is actually loading or the code is silently falling back to 10 FPS. This is a one-line addition.

### 2. Timing Instrumentation (Diagnostic)

**File:** `src/api/app.py` (inside `_detection_loop`)

Add lightweight timing around each phase of the detection cycle. Log a summary every 100 cycles to avoid log spam.

**Phases to measure:**
1. **Frame collection** — iterating feeds, calling `fm.get_frame()` (lines 218-246)
2. **GPU lock acquisition** — time spent waiting for the lock (lines 251-252)
3. **YOLO batch inference** — `detector.get_masks_batch()` call (line 254)
4. **Post-processing** — zone checks, mask combining, alarm logic, detection state update (lines 262-388)
5. **Total cycle time** — full loop iteration

**Output format** (every 100 cycles):
```
[DETECTION] FPS report (last 100 cycles): avg 28.3 FPS | inference 22.1ms | post 4.2ms | lock-wait 0.3ms | total 35.3ms
```

**Implementation:**
- Use `time.monotonic()` (already used for deadline scheduling)
- Accumulate times in local variables, reset every 100 cycles
- Calculate effective FPS from total cycle times
- No new dependencies, no threads, no external state

### 3. GPU Lock Timeout (Fix)

**File:** `src/api/app.py` (line 252)

Replace blocking `gpu_lock.acquire()` with a timeout:

```python
# Before (blocks indefinitely):
gpu_lock.acquire()

# After (skip cycle if auto-seg holds lock):
if not gpu_lock.acquire(timeout=0.005):  # 5ms
    print("[DETECTION] GPU lock contention, skipping cycle")
    continue
```

**Why:** The auto-seg thread holds the GPU lock during `segmenter.segment_frame()`, which can take 100-500ms. During that time, the detection thread is completely stalled. With a 5ms timeout, detection skips one cycle (33ms at 30 FPS) rather than stalling for the full segmentation duration. At 30 FPS this means dropping 1 frame instead of dropping 3-15 frames.

The `continue` skips to the next loop iteration, which re-checks the deadline timer and sleeps appropriately.

### 4. TensorRT Batch Size (Fix)

**File:** `src/detection/human_detector.py` (line 56)

Change the TensorRT export max batch size from 4 to 8:

```python
# Before:
base_model.export(format="engine", imgsz=inference_imgsz, half=True, batch=4, dynamic=True)

# After:
base_model.export(format="engine", imgsz=inference_imgsz, half=True, batch=8, dynamic=True)
```

**Why:** The current `batch=4` hard-caps inference at 4 frames. With `dynamic=True`, smaller batches (1-4) are handled efficiently, so increasing to 8 has no cost for the current 4-feed setup but enables future scaling.

**Note:** Changing this requires deleting the cached `.engine` file so it re-exports on next startup. The export is one-time and takes a few minutes.

## Files Modified

| File | Change | Type |
|------|--------|------|
| `src/api/app.py` (~line 120) | Log resolved detection FPS | Fix |
| `src/api/app.py` (detection loop body) | Timing instrumentation | Diagnostic |
| `src/api/app.py` (~line 252) | GPU lock timeout | Fix |
| `src/detection/human_detector.py` (~line 56) | TensorRT batch=8 | Fix |

## What This Tells Us

After deploying, the FPS report log will reveal:

| Scenario | Meaning | Next Step |
|----------|---------|-----------|
| FPS ~30, inference <30ms | Pipeline is fine, problem is elsewhere (display?) | Investigate MJPEG stream / frontend |
| FPS ~10, inference <30ms | Config not loading, stuck on fallback FPS | Fix config loading path |
| FPS <20, inference >40ms | GPU-bound at 1280px with 4 feeds | Consider model optimization or async pipeline |
| Lock-wait spikes visible | Auto-seg contention confirmed | GPU lock timeout fix handles this |
| Post-processing >10ms | Zone checks or mask combining too slow | Profile zone overlap logic |

## Out of Scope

- Frame-mask synchronization (deferred — may not be needed if FPS is sufficient)
- Resolution reduction (user requires 1280px)
- Per-feed detection threading (unnecessary complexity for 4 feeds)
- Frontend/MJPEG display changes
