# Detection FPS Bottleneck — Diagnostic & Fix Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Diagnose and fix the detection pipeline FPS bottleneck causing laggy human mask overlays.

**Architecture:** Add timing instrumentation to the detection loop to identify the bottleneck, fix a potential config/code mismatch where detection silently defaults to 10 FPS, add GPU lock timeout to prevent auto-seg stalls, and increase TensorRT max batch size for future scalability.

**Tech Stack:** Python, threading, time.monotonic(), YOLO/Ultralytics, TensorRT

---

### Task 1: Log Resolved Detection FPS on Startup

**Files:**
- Modify: `src/api/app.py:120-121`

- [ ] **Step 1: Add FPS log line**

After `app.py:121` (`interval = 1.0 / max(1, fps)`), add a log line that prints the resolved value. This is the most important diagnostic — it confirms whether you're running at 10 or 30 FPS.

In `src/api/app.py`, find:

```python
    det_cfg = cfg.get("detection", {})
    fps = det_cfg.get("fps", 10)
    interval = 1.0 / max(1, fps)
```

Replace with:

```python
    det_cfg = cfg.get("detection", {})
    fps = det_cfg.get("fps", 10)
    interval = 1.0 / max(1, fps)
    print(f"[DETECTION] Target FPS: {fps} (interval: {interval:.3f}s)")
```

- [ ] **Step 2: Commit**

```bash
git add src/api/app.py
git commit -m "feat: log resolved detection FPS on startup"
```

---

### Task 2: Add Detection Loop Timing Instrumentation

**Files:**
- Modify: `src/api/app.py:215-414` (inside `_detection_loop` while-loop)

- [ ] **Step 1: Add timing accumulators before the while loop**

In `src/api/app.py`, find:

```python
    while getattr(fm, "_running", False):
        loop_start = time.monotonic()
```

Replace with:

```python
    # Timing instrumentation — accumulate and log every 100 cycles
    _t_cycles = 0
    _t_total = 0.0
    _t_collect = 0.0
    _t_lock = 0.0
    _t_infer = 0.0
    _t_post = 0.0

    while getattr(fm, "_running", False):
        loop_start = time.monotonic()
```

- [ ] **Step 2: Wrap frame collection phase with timing**

In `src/api/app.py`, find:

```python
        # --- 1. Collect frames from all eligible feeds ---
        batch_feed_ids = []
        batch_frames = []
        batch_pipelines = []

        for feed_id in fm.feed_ids():
```

Replace with:

```python
        # --- 1. Collect frames from all eligible feeds ---
        _t0 = time.monotonic()
        batch_feed_ids = []
        batch_frames = []
        batch_pipelines = []

        for feed_id in fm.feed_ids():
```

- [ ] **Step 3: Add timing around GPU lock acquisition and YOLO inference**

In `src/api/app.py`, find:

```python
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
```

Replace with:

```python
        # --- 2. Run batched YOLO inference (single GPU call) ---
        _t1 = time.monotonic()
        _t_collect += _t1 - _t0
        if batch_frames and detector is not None:
            try:
                _t_lock_start = time.monotonic()
                if gpu_lock is not None:
                    gpu_lock.acquire()
                _t_lock_end = time.monotonic()
                _t_lock += _t_lock_end - _t_lock_start
                try:
                    batch_masks = detector.get_masks_batch(batch_frames)
                finally:
                    _t_infer += time.monotonic() - _t_lock_end
                    if gpu_lock is not None:
                        gpu_lock.release()
```

- [ ] **Step 4: Add timing around post-processing and the FPS report logger**

In `src/api/app.py`, find:

```python
            # --- 3. Process results per feed (zone checks, alarms, overlays) ---
            for idx, feed_id in enumerate(batch_feed_ids):
```

Replace with:

```python
            # --- 3. Process results per feed (zone checks, alarms, overlays) ---
            _t_post_start = time.monotonic()
            for idx, feed_id in enumerate(batch_feed_ids):
```

Then find:

```python
        # Deadline-based sleep: account for time spent on inference + processing
        elapsed = time.monotonic() - loop_start
```

Replace with:

```python
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
            print(
                f"[DETECTION] FPS report (last {_t_cycles} cycles): "
                f"avg {avg_fps:.1f} FPS | "
                f"inference {avg_infer:.1f}ms | "
                f"post {avg_post:.1f}ms | "
                f"lock-wait {avg_lock:.1f}ms | "
                f"total {avg_total:.1f}ms"
            )
            _t_cycles = 0
            _t_total = 0.0
            _t_collect = 0.0
            _t_lock = 0.0
            _t_infer = 0.0
            _t_post = 0.0

        # Deadline-based sleep: account for time spent on inference + processing
        elapsed = time.monotonic() - loop_start
```

- [ ] **Step 5: Verify the detection loop still runs correctly**

Run the app and check that:
1. The `[DETECTION] Target FPS: ...` line appears on startup
2. After ~3-4 seconds (100 cycles at 30 FPS), the FPS report line appears
3. The video stream still works normally

```bash
python -m src.api.app
```

Check console output for lines like:
```
[DETECTION] Target FPS: 30 (interval: 0.033s)
[DETECTION] FPS report (last 100 cycles): avg 28.3 FPS | inference 22.1ms | post 4.2ms | lock-wait 0.3ms | total 35.3ms
```

- [ ] **Step 6: Commit**

```bash
git add src/api/app.py
git commit -m "feat: add timing instrumentation to detection loop"
```

---

### Task 3: GPU Lock Timeout

**Files:**
- Modify: `src/api/app.py:251-252`

- [ ] **Step 1: Replace blocking lock with timeout**

In `src/api/app.py`, find:

```python
                _t_lock_start = time.monotonic()
                if gpu_lock is not None:
                    gpu_lock.acquire()
                _t_lock_end = time.monotonic()
```

Replace with:

```python
                _t_lock_start = time.monotonic()
                if gpu_lock is not None:
                    if not gpu_lock.acquire(timeout=0.005):
                        _t_lock += time.monotonic() - _t_lock_start
                        print("[DETECTION] GPU lock contention, skipping cycle")
                        continue
                _t_lock_end = time.monotonic()
```

Note: The `continue` here goes back to the `while getattr(fm, "_running", False):` loop, which will re-check the deadline timer and sleep. This means one skipped detection cycle (~33ms at 30 FPS) instead of blocking for potentially hundreds of milliseconds while auto-seg holds the lock.

- [ ] **Step 2: Verify lock contention is now logged**

Run the app with auto-segmentation enabled. On the first segmentation pass (5 seconds after startup), you should see at most one `[DETECTION] GPU lock contention, skipping cycle` log line rather than a multi-frame stall.

```bash
python -m src.api.app
```

- [ ] **Step 3: Commit**

```bash
git add src/api/app.py
git commit -m "fix: use timeout on GPU lock to prevent detection stalls from auto-seg"
```

---

### Task 4: Increase TensorRT Max Batch Size

**Files:**
- Modify: `src/detection/human_detector.py:56`

- [ ] **Step 1: Change batch size from 4 to 8**

In `src/detection/human_detector.py`, find:

```python
        base_model.export(
            format="engine",
            imgsz=inference_imgsz,
            half=True,
            batch=4,
            dynamic=True,
        )
```

Replace with:

```python
        base_model.export(
            format="engine",
            imgsz=inference_imgsz,
            half=True,
            batch=8,
            dynamic=True,
        )
```

- [ ] **Step 2: Run existing detector tests**

```bash
pytest tests/unit/detection/test_human_detector.py -v
```

Expected: All 6 tests pass. These tests mock YOLO and don't trigger TensorRT export, so the batch size change doesn't affect them. The change only takes effect on next TensorRT re-export (when the cached `.engine` file is deleted).

- [ ] **Step 3: Commit**

```bash
git add src/detection/human_detector.py
git commit -m "feat: increase TensorRT max batch size from 4 to 8 for multi-feed scaling"
```

---

### Task 5: Manual Verification

- [ ] **Step 1: Delete cached TensorRT engine (if exists)**

The batch size change requires re-exporting. Find and delete the cached `.engine` file:

```bash
find models/ -name "*.engine" -type f
```

Delete any found `.engine` files. On next startup, YOLO will re-export with `batch=8` (one-time, takes a few minutes).

- [ ] **Step 2: Start the application and verify diagnostics**

```bash
python -m src.api.app
```

Watch for these log lines in order:
1. `[DETECTION] Target FPS: 30 (interval: 0.033s)` — confirms config is loaded
2. `[YOLO] Exporting TensorRT engine...` — re-export with new batch size
3. `[DETECTION] FPS report (last 100 cycles): ...` — shows actual performance

- [ ] **Step 3: Interpret the FPS report**

| Observation | Diagnosis | Action |
|-------------|-----------|--------|
| FPS ~30, inference <30ms | Pipeline is healthy, latency may be in display | Investigate MJPEG stream/frontend |
| FPS ~10, inference <30ms | Config not loading, stuck on fallback | Debug config loading in `src/core/config.py` |
| FPS <20, inference >40ms | GPU-bound at 1280px with 4 feeds | Consider async pipeline or model swap |
| lock-wait spikes >5ms | Auto-seg contention (now handled by timeout) | Timeout fix is working as intended |
| post >10ms | Zone checks or mask combining slow | Profile `zone_manager.check_red/check_yellow` |

- [ ] **Step 4: Commit any remaining changes and document findings**

If the FPS report reveals the bottleneck, note it for the next round of optimization.
