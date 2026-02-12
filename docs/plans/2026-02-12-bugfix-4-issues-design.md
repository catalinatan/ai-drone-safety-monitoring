# Design: Fix 4 Edit Zones & Detection Issues

**Date:** 2026-02-12
**Branch:** ui

## Issues

### 1. Tiny CCTV Feed in Edit Zones Page
**Root cause:** PolygonCanvas inner wrapper uses `inline-block max-w-full max-h-full`, sizing to intrinsic image dimensions instead of expanding to fill container.

**Fix:** Change inner wrapper to `w-full h-full flex items-center justify-center`. Keep image with `object-contain` to preserve aspect ratio. Track rendered image bounds so the SVG overlay aligns with the actual image area, not the full wrapper.

**Files:** `src/ui/src/components/PolygonCanvas.tsx`

### 2. Mask Cache Race Condition
**Root cause:** Detection thread reads `feed.red_zone_mask` / `feed.yellow_zone_mask` without lock protection, while API thread sets them to `None` under lock. TOCTOU race causes deleted zones to keep triggering alarms.

**Fix:** Snapshot frame + masks in a single `with feed.lock:` block at the start of each detection cycle. Use local copies for the entire cycle. Remove lines that write resized masks back to feed state.

**Files:** `src/backend/server.py` (run_detection, run_detection_only)

### 3. Auto Segment Button Obstruction
**Root cause:** Canvas container uses `absolute inset-0` which can overlap the toolbar below. PolygonCanvas click handlers intercept clicks intended for toolbar buttons.

**Fix:** Remove the intermediate `absolute inset-0` div in EditFeedPage. Let PolygonCanvas fill its flex parent naturally. Add error feedback when auto-segment API call fails silently.

**Files:** `src/ui/src/components/EditFeedPage.tsx`, `src/ui/src/components/PolygonCanvas.tsx`

### 4. Detection Latency (Decouple Alarm from Depth)
**Root cause:** Depth estimation (50-200ms) runs synchronously in the detection thread, blocking all frame processing while calculating coordinates.

**Fix:** Split into two phases:
- **Phase 1 (fast):** YOLO + overlap check -> immediately set alarm_active. ~30-185ms.
- **Phase 2 (async):** Offload depth inference + drone dispatch to a dedicated worker thread via `queue.Queue(maxsize=4)`. Detection thread is never blocked.

Alarm fires ~100-200ms faster. Drone dispatch takes same total time but is async.

**Files:** `src/backend/server.py` (run_detection, run_detection_only, new depth_worker_loop, FeedManager.__init__, lifespan)

## Approach
Approach A for all 4 issues: targeted fixes with minimal architectural change.
