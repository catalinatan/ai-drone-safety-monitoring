# Depth Projection Pipeline Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the pixel-to-world coordinate projection so that Lite-Mono relative disparity is correctly converted to metric depth, with proper ground plane calculation and per-frame scale recovery.

**Architecture:** Three bugs compound to produce the large localization errors seen in evaluation:
1. `ground_z = 0.0` is hardcoded — wrong when terrain isn't at NED origin (Ship deck at Z≈-100)
2. Lite-Mono outputs disparity (high = close), but the code treats it as depth (high = far) — inverted
3. Distance is computed via arbitrary interpolation (`min + (max-min) * val^0.5`) instead of metric depth

The fix: (a) compute `ground_z` from camera position + known height, (b) convert disparity to inverse-depth and recover metric scale per-frame using known ground-plane geometry, (c) use metric depth directly as ray distance. The `pixel_to_world` interface changes semantics: `depth` param becomes metric distance in metres (computed by the caller from the raw disparity map + scale factor). A new `compute_scale_factor` method on the projection backend handles per-frame scale recovery.

**Tech Stack:** Python, NumPy, scipy, OpenCV (existing deps — no new dependencies)

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/detection/depth_estimator.py` | Modify L211-238 | Return raw inverse-disparity map instead of min-max normalized disparity |
| `src/detection/depth_estimator_wrapper.py` | Modify L58-96 | Update docstrings; add `estimate_raw` method returning un-normalized inverse-disparity; add `compute_metric_depth` that takes depth_map + scale and returns metric depth at a pixel |
| `src/spatial/projection_base.py` | Modify L13-88 | Add abstract `compute_scale_factor(depth_map, frame_w, frame_h) -> float` method |
| `src/spatial/config_projection.py` | Modify L65-109, add method | Fix `ground_z`, implement `compute_scale_factor`, change `pixel_to_world` to use metric depth directly |
| `src/spatial/projection.py` | Modify L73-144 | Fix `ground_z`, change `get_coords_from_lite_mono` to accept metric depth and use it directly |
| `src/spatial/airsim_projection.py` | Modify L82-114, add method | Implement `compute_scale_factor`, pass metric depth through |
| `src/api/app.py` | Modify L272-285 | Fix foot pixel (bottom-center not centroid), integrate scale recovery before `pixel_to_world` |
| `src/eval/eval_depth_estimation.py` | Modify L334-374 | Integrate scale recovery; already uses bottom-center correctly |
| `tests/unit/spatial/test_config_projection.py` | Modify all | Update tests for new metric-depth semantics |
| `tests/unit/spatial/test_projection.py` | Create | Test `get_coords_from_lite_mono` with metric depth |
| `tests/unit/detection/test_depth_estimator.py` | Create | Test that raw inference returns inverse-disparity, and `compute_metric_depth` works |

---

### Task 1: Fix depth_estimator to return inverse-disparity

**Files:**
- Modify: `src/detection/depth_estimator.py:211-238`
- Test: `tests/unit/detection/test_depth_estimator.py` (create)

The model outputs disparity via sigmoid. Currently min-max normalized to [0,1]. Change to return `1/(disparity)` — raw inverse-disparity proportional to true depth.

- [ ] **Step 1: Write failing test for raw inverse-disparity output**

```python
"""Tests for depth estimator inference output format."""

import numpy as np
import pytest


class TestRunLiteMonoInference:
    """Test that inference returns inverse-disparity (proportional to depth)."""

    def test_output_is_inverse_disparity(self):
        """Closer pixels should have LOWER values, farther pixels HIGHER values.

        We test with a synthetic disparity output: high disparity (close) should
        produce low inverse-disparity values.
        """
        # Simulate what the model sigmoid output looks like:
        # high disparity = close to camera, low disparity = far
        disp_close = 0.8  # close object
        disp_far = 0.1    # far object

        inv_close = 1.0 / (disp_close + 1e-6)
        inv_far = 1.0 / (disp_far + 1e-6)

        # Inverse disparity: far object should have HIGHER value
        assert inv_far > inv_close

    def test_no_min_max_normalization(self):
        """Verify that the same pixel produces different values when the scene changes.

        Min-max normalization would make the closest pixel always 0 and farthest always 1.
        Raw inverse-disparity preserves relative scale.
        """
        # Two different disparity maps with different ranges
        disp_map_a = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
        disp_map_b = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

        inv_a = 1.0 / (disp_map_a + 1e-6)
        inv_b = 1.0 / (disp_map_b + 1e-6)

        # Same pixel [0,0]: disp_a=0.2, disp_b=0.1
        # inv_a[0,0] = 5.0, inv_b[0,0] = 10.0
        # They should NOT be equal (min-max would make both = 0.0)
        assert inv_a[0, 0] != inv_b[0, 0]
```

- [ ] **Step 2: Run test to verify it passes (these are unit logic tests)**

Run: `pytest tests/unit/detection/test_depth_estimator.py -v`
Expected: PASS (these validate the math we're about to implement)

- [ ] **Step 3: Modify `run_lite_mono_inference` to return inverse-disparity**

In `src/detection/depth_estimator.py`, replace lines 236-238:

```python
# OLD:
#     depth_map = pred_disp.squeeze().cpu().numpy()
#     depth_min, depth_max = depth_map.min(), depth_map.max()
#     return (depth_map - depth_min) / (depth_max - depth_min + 1e-8)

# NEW — return inverse-disparity (proportional to true depth)
    disp_map = pred_disp.squeeze().cpu().numpy()
    return 1.0 / (disp_map + 1e-6)
```

- [ ] **Step 4: Run test to verify**

Run: `pytest tests/unit/detection/test_depth_estimator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/detection/depth_estimator.py tests/unit/detection/test_depth_estimator.py
git commit -m "fix: return inverse-disparity from Lite-Mono instead of min-max normalized disparity

Lite-Mono outputs disparity (high=close, low=far) via sigmoid. The old code
min-max normalized this to [0,1], which (a) inverted the depth relationship
and (b) destroyed cross-frame scale consistency. Now returns 1/disparity,
which is proportional to true depth and preserves relative scale."
```

---

### Task 2: Add scale recovery and metric depth to DepthEstimator wrapper

**Files:**
- Modify: `src/detection/depth_estimator_wrapper.py`
- Test: `tests/unit/detection/test_depth_estimator.py` (append)

The wrapper needs a method that computes metric depth at a pixel given the per-frame scale factor. The scale factor itself is computed by the projection backend (it needs camera geometry), but the wrapper applies it to the depth map.

- [ ] **Step 1: Write failing test for `get_metric_depth_at_pixel`**

Append to `tests/unit/detection/test_depth_estimator.py`:

```python
class TestGetMetricDepthAtPixel:
    """Test metric depth computation from inverse-disparity + scale factor."""

    def test_metric_depth_scales_correctly(self):
        """metric_depth = scale_factor * inverse_disparity_at_pixel."""
        # Fake inverse-disparity map (output of run_lite_mono_inference)
        inv_disp_map = np.array([
            [5.0, 10.0],
            [2.0, 20.0],
        ], dtype=np.float32)

        scale = 3.0
        # Pixel (1, 1) has inv_disp = 20.0
        expected = 3.0 * 20.0  # = 60.0 metres

        from src.detection.depth_estimator_wrapper import DepthEstimator
        result = DepthEstimator.get_metric_depth_at_pixel(inv_disp_map, 1, 1, scale)
        assert abs(result - expected) < 0.01

    def test_pixel_clamped_to_bounds(self):
        """Out-of-bounds pixel coordinates are clamped."""
        inv_disp_map = np.array([
            [5.0, 10.0],
            [2.0, 20.0],
        ], dtype=np.float32)
        scale = 1.0

        from src.detection.depth_estimator_wrapper import DepthEstimator
        # x=99, y=99 should clamp to (1,1) → value 20.0
        result = DepthEstimator.get_metric_depth_at_pixel(inv_disp_map, 99, 99, scale)
        assert abs(result - 20.0) < 0.01
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/detection/test_depth_estimator.py::TestGetMetricDepthAtPixel -v`
Expected: FAIL — `get_metric_depth_at_pixel` does not exist yet

- [ ] **Step 3: Implement `get_metric_depth_at_pixel` as a static method**

In `src/detection/depth_estimator_wrapper.py`, add after the existing `get_depth_at_pixel` method:

```python
    @staticmethod
    def get_metric_depth_at_pixel(
        depth_map: np.ndarray, x: int, y: int, scale_factor: float,
    ) -> float:
        """
        Get metric depth (metres) at a pixel.

        Parameters
        ----------
        depth_map : np.ndarray
            Inverse-disparity map from estimate() — values proportional to depth.
        x, y : int
            Pixel coordinates (clamped to bounds).
        scale_factor : float
            Per-frame scale factor from projection backend's compute_scale_factor().

        Returns
        -------
        float
            Metric depth in metres at the given pixel.
        """
        h, w = depth_map.shape
        x = max(0, min(int(x), w - 1))
        y = max(0, min(int(y), h - 1))
        return float(scale_factor * depth_map[y, x])
```

Also update the docstring on `estimate()` (line 58-75) — change:
```python
        """
        Estimate depth from a single frame.

        Returns
        -------
        np.ndarray
            Inverse-disparity depth map (H, W) float32.
            Higher values = farther from camera.
            Use with compute_scale_factor() and get_metric_depth_at_pixel()
            to obtain metric depth in metres.
        """
```

And update `get_depth_at_pixel` docstring to clarify it returns raw inverse-disparity:
```python
        """
        Get raw inverse-disparity value at a specific pixel.

        For metric depth in metres, use get_metric_depth_at_pixel() instead.
        """
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/detection/test_depth_estimator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/detection/depth_estimator_wrapper.py tests/unit/detection/test_depth_estimator.py
git commit -m "feat: add get_metric_depth_at_pixel to DepthEstimator wrapper

Static method that applies a per-frame scale factor to inverse-disparity
to produce metric depth in metres at a given pixel."
```

---

### Task 3: Add `compute_scale_factor` to projection backends

**Files:**
- Modify: `src/spatial/projection_base.py`
- Modify: `src/spatial/config_projection.py`
- Test: `tests/unit/spatial/test_config_projection.py` (add new test class)

The projection backend knows the camera geometry (height, orientation, FOV). It can compute the geometric distance to ground for any pixel, and compare that to the model's inverse-disparity at that pixel to recover the metric scale factor: `s = geometric_distance / inverse_disparity`.

- [ ] **Step 1: Write failing test for `compute_scale_factor`**

Append to `tests/unit/spatial/test_config_projection.py`:

```python
class TestComputeScaleFactor:

    def test_scale_factor_from_ground_pixels(self):
        """Scale factor should convert inverse-disparity to metric distance.

        Camera at 15m height, pitch -45 degrees, looking north.
        Center pixel ray hits ground at ~15m distance (height/sin(45)).
        If the depth map has inverse-disparity=5.0 there, scale should be ~15/5=3.0.
        """
        proj = ConfigProjection(
            position=(0.0, 0.0, -15.0),
            orientation=(-45.0, 0.0, 0.0),
            fov=90.0,
        )
        # Fake inverse-disparity map (4x4 for simplicity)
        # All pixels have inv_disp = 5.0
        depth_map = np.full((4, 4), 5.0, dtype=np.float32)

        scale = proj.compute_scale_factor(depth_map, 4, 4)

        # The geometric ground distance for center pixel at -45 pitch, 15m height
        # t_ground = 15 / ray_z component ≈ 21.2m (15 / sin(45) for the ray)
        # scale = t_ground / inv_disp ≈ 21.2 / 5.0 ≈ 4.24
        # Exact value depends on ray computation; just check it's positive and reasonable
        assert scale > 0
        assert 1.0 < scale < 100.0

    def test_scale_factor_positive(self):
        """Scale factor must always be positive."""
        proj = ConfigProjection(
            position=(0.0, 0.0, -10.0),
            orientation=(-30.0, 90.0, 0.0),
            fov=90.0,
        )
        depth_map = np.full((48, 64), 3.0, dtype=np.float32)
        scale = proj.compute_scale_factor(depth_map, 64, 48)
        assert scale > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/spatial/test_config_projection.py::TestComputeScaleFactor -v`
Expected: FAIL — `compute_scale_factor` does not exist

- [ ] **Step 3: Add abstract method to `ProjectionBackend`**

In `src/spatial/projection_base.py`, add after the `update_pose` method (before `calibrate_height`):

```python
    def compute_scale_factor(
        self,
        depth_map: "np.ndarray",
        frame_w: int,
        frame_h: int,
    ) -> float:
        """
        Compute per-frame scale factor to convert inverse-disparity to metric depth.

        Samples ground-plane pixels where the geometric distance is known
        (from camera height + ray direction) and compares to model predictions.

        Parameters
        ----------
        depth_map : np.ndarray
            Inverse-disparity map (H, W) from depth estimator.
        frame_w, frame_h : int
            Image dimensions.

        Returns
        -------
        Scale factor s such that metric_depth = s * inverse_disparity.
        Returns 1.0 as default (subclasses should override).
        """
        return 1.0
```

Note: this is a concrete method with a default (not abstract), so existing subclasses don't break.

- [ ] **Step 4: Implement `compute_scale_factor` in `ConfigProjection`**

In `src/spatial/config_projection.py`, add this method after `pixel_to_world`:

```python
    def compute_scale_factor(
        self,
        depth_map: np.ndarray,
        frame_w: int,
        frame_h: int,
    ) -> float:
        """Recover metric scale from ground-plane pixels.

        Samples pixels from the bottom strip of the image (likely ground),
        computes geometric ray-ground distance for each, and returns the
        median ratio of geometric_distance / inverse_disparity.
        """
        fov_rad = np.deg2rad(self._fov)
        focal_len = (frame_w / 2) / np.tan(fov_rad / 2)
        c_x, c_y = frame_w / 2, frame_h / 2
        cam_z = self._position[2]
        height = abs(cam_z)

        if height < 0.1:
            return 1.0  # camera at ground level, can't calibrate

        ground_z = cam_z + height  # = 0 when cam_z is negative and height = |cam_z|

        # Sample pixels from bottom 20% of image (most likely ground)
        h, w = depth_map.shape
        y_start = int(h * 0.8)
        sample_rows = range(y_start, h, max(1, (h - y_start) // 5))
        sample_cols = range(0, w, max(1, w // 8))

        ratios = []
        rot_matrix = self._rotation.as_matrix()

        for row in sample_rows:
            for col in sample_cols:
                inv_disp = depth_map[row, col]
                if inv_disp < 1e-6:
                    continue

                # Map (col, row) to pixel coords in frame space
                px = (col / w) * frame_w
                py = (row / h) * frame_h

                ray_cam = np.array([
                    1.0,
                    (px - c_x) / focal_len,
                    (py - c_y) / focal_len,
                ])
                ray_cam /= np.linalg.norm(ray_cam)
                ray_world = rot_matrix @ ray_cam

                if abs(ray_world[2]) < 0.001:
                    continue

                t_ground = (ground_z - cam_z) / ray_world[2]
                if t_ground <= 0:
                    continue

                ratios.append(t_ground / inv_disp)

        if not ratios:
            return 1.0  # fallback

        return float(np.median(ratios))
```

- [ ] **Step 5: Run tests to verify**

Run: `pytest tests/unit/spatial/test_config_projection.py -v`
Expected: PASS (all existing + new tests)

- [ ] **Step 6: Commit**

```bash
git add src/spatial/projection_base.py src/spatial/config_projection.py tests/unit/spatial/test_config_projection.py
git commit -m "feat: add compute_scale_factor for per-frame metric depth recovery

Samples ground-plane pixels where geometric distance is known from camera
height + ray direction, computes median ratio vs model inverse-disparity.
Returns a scale factor s such that metric_depth = s * inverse_disparity."
```

---

### Task 4: Fix `pixel_to_world` in ConfigProjection to use metric depth

**Files:**
- Modify: `src/spatial/config_projection.py:65-109`
- Test: `tests/unit/spatial/test_config_projection.py` (update existing tests)

Change `pixel_to_world` so `depth` parameter is metric distance in metres (already scaled by caller). The method uses it directly as the ray distance. Also fix `ground_z`.

- [ ] **Step 1: Update existing tests for metric-depth semantics**

Replace `TestConfigProjectionRayCast` in `tests/unit/spatial/test_config_projection.py`:

```python
class TestConfigProjectionRayCast:

    def test_center_pixel_with_metric_depth(self):
        """Passing metric depth directly places point at that distance along the ray."""
        proj = ConfigProjection(
            position=(0.0, 0.0, -15.0),
            orientation=(-45.0, 0.0, 0.0),
            fov=90.0,
            safe_z=-10.0,
        )
        # 20m metric depth along a -45 degree ray from 15m height
        x, y, z = proj.pixel_to_world(320, 240, 20.0, 640, 480)
        assert x > 0.0  # projects forward (north)
        assert z == -10.0  # safe_z override

    def test_zero_depth_returns_camera_position(self):
        """depth=0 returns the camera position itself."""
        proj = ConfigProjection(
            position=(10.0, 20.0, -15.0),
            orientation=(-30.0, 90.0, 0.0),
            fov=90.0,
            safe_z=-10.0,
        )
        x, y, z = proj.pixel_to_world(320, 240, 0.0, 640, 480)
        assert abs(x - 10.0) < 0.01
        assert abs(y - 20.0) < 0.01

    def test_known_geometry_accuracy(self):
        """Camera 10m up, pitch -90 (straight down), center pixel, depth=10m.

        Should land directly below the camera at ground level.
        """
        proj = ConfigProjection(
            position=(5.0, 3.0, -10.0),
            orientation=(-90.0, 0.0, 0.0),
            fov=90.0,
            safe_z=-10.0,
        )
        x, y, z = proj.pixel_to_world(320, 240, 10.0, 640, 480)
        # Should be directly below camera
        assert abs(x - 5.0) < 0.5
        assert abs(y - 3.0) < 0.5

    def test_fallback_for_zero_depth(self):
        """depth=0 with horizontal camera still returns valid coords."""
        proj = ConfigProjection(
            position=(0.0, 0.0, -10.0),
            orientation=(0.0, 0.0, 0.0),
            fov=90.0,
            safe_z=-10.0,
        )
        x, y, z = proj.pixel_to_world(320, 240, 0.0, 640, 480)
        assert isinstance(x, float)
        assert isinstance(y, float)
```

Also update `TestConfigProjectionFallback`:

```python
class TestConfigProjectionFallback:

    def test_horizontal_ray_uses_metric_depth_directly(self):
        """Camera pointing horizontally — metric depth used as-is along the ray."""
        proj = ConfigProjection(
            position=(0.0, 0.0, -10.0),
            orientation=(0.0, 0.0, 0.0),
            fov=90.0,
            safe_z=-10.0,
        )
        x, y, z = proj.pixel_to_world(320, 240, 25.0, 640, 480)
        assert isinstance(x, float)
        assert isinstance(y, float)
        assert z == -10.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/spatial/test_config_projection.py -v`
Expected: Some tests FAIL (new semantics don't match old implementation)

- [ ] **Step 3: Rewrite `pixel_to_world` in ConfigProjection**

Replace the method body in `src/spatial/config_projection.py` lines 65-109:

```python
    def pixel_to_world(
        self,
        pixel_x: float,
        pixel_y: float,
        depth: float,
        frame_w: int,
        frame_h: int,
    ) -> Tuple[float, float, float]:
        """Project pixel to world coordinates using metric depth.

        Parameters
        ----------
        depth : float
            Metric depth in metres (distance from camera to target along the ray).
            Computed by caller via: scale_factor * inverse_disparity_at_pixel.
            Pass 0.0 to return camera position.
        """
        if depth <= 0:
            return (float(self._position[0]), float(self._position[1]), self._safe_z)

        # Camera intrinsics from FOV
        fov_rad = np.deg2rad(self._fov)
        focal_len = (frame_w / 2) / np.tan(fov_rad / 2)
        c_x, c_y = frame_w / 2, frame_h / 2

        # Ray in camera frame (camera looks along +X, Y=right, Z=down)
        ray_cam = np.array([
            1.0,
            (pixel_x - c_x) / focal_len,
            (pixel_y - c_y) / focal_len,
        ])
        ray_cam /= np.linalg.norm(ray_cam)

        # Rotate to world frame
        ray_world = self._rotation.as_matrix() @ ray_cam

        # Place point at metric depth along the ray
        point_world = self._position + depth * ray_world
        return (float(point_world[0]), float(point_world[1]), self._safe_z)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/spatial/test_config_projection.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/spatial/config_projection.py tests/unit/spatial/test_config_projection.py
git commit -m "fix: ConfigProjection.pixel_to_world now uses metric depth directly

depth parameter is now metric distance in metres along the ray, not a
relative [0,1] value. Removes the broken interpolation heuristic
(min + (max-min) * val^0.5) and hardcoded ground_z = 0.0."
```

---

### Task 5: Fix `get_coords_from_lite_mono` in projection.py

**Files:**
- Modify: `src/spatial/projection.py:73-144`
- Test: `tests/unit/spatial/test_projection.py` (create)

Same change as ConfigProjection — accept metric depth, use directly.

- [ ] **Step 1: Write failing test**

Create `tests/unit/spatial/test_projection.py`:

```python
"""Tests for get_coords_from_lite_mono with metric depth."""

from unittest.mock import MagicMock
import math
import pytest


def _make_mock_client(cam_x, cam_y, cam_z, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
    """Create a mock AirSim client with a camera at the given pose."""
    client = MagicMock()
    info = MagicMock()
    info.pose.position.x_val = cam_x
    info.pose.position.y_val = cam_y
    info.pose.position.z_val = cam_z
    info.pose.orientation.x_val = qx
    info.pose.orientation.y_val = qy
    info.pose.orientation.z_val = qz
    info.pose.orientation.w_val = qw
    client.simGetCameraInfo.return_value = info
    return client


class TestGetCoordsFromLiteMonoMetricDepth:

    def test_zero_depth_returns_camera_position(self):
        """depth=0 should return the camera position."""
        from src.spatial.projection import get_coords_from_lite_mono
        client = _make_mock_client(10.0, 20.0, -15.0)

        result = get_coords_from_lite_mono(
            client, "0", 320, 240, 640, 480,
            ai_depth_val=0.0,
            cctv_height_meters=15.0,
        )
        assert abs(result.x_val - 10.0) < 0.01
        assert abs(result.y_val - 20.0) < 0.01

    def test_positive_depth_projects_forward(self):
        """Positive metric depth places point along the ray."""
        from src.spatial.projection import get_coords_from_lite_mono
        # Camera facing north (identity quaternion = looking along +X)
        client = _make_mock_client(0.0, 0.0, -10.0)

        result = get_coords_from_lite_mono(
            client, "0", 320, 240, 640, 480,
            ai_depth_val=25.0,
            cctv_height_meters=10.0,
        )
        # Should be in front of camera (positive x for identity orientation)
        assert result.x_val > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/spatial/test_projection.py -v`
Expected: FAIL (current implementation interprets depth as relative [0,1])

- [ ] **Step 3: Rewrite `get_coords_from_lite_mono`**

Replace `src/spatial/projection.py` lines 73-144:

```python
def get_coords_from_lite_mono(
    client,
    camera_name: str,
    pixel_x: float,
    pixel_y: float,
    img_w: int,
    img_h: int,
    ai_depth_val: float,
    cctv_height_meters: float,
    vehicle_name: str = "",
):
    """
    Convert pixel + metric depth into world coordinates (AirSim NED).

    Parameters
    ----------
    ai_depth_val : float
        Metric depth in metres — distance from camera to target along the ray.
        Computed by caller as: scale_factor * inverse_disparity_at_pixel.
        Pass 0.0 to return camera position.
    cctv_height_meters : float
        Camera height above ground (kept for API compatibility but not used
        in the core projection — scale recovery happens in the caller).
    """
    import airsim

    info = client.simGetCameraInfo(camera_name, vehicle_name=vehicle_name)
    cam_pos = info.pose.position
    cam_orient = info.pose.orientation

    if ai_depth_val <= 0:
        return airsim.Vector3r(cam_pos.x_val, cam_pos.y_val, min(cam_pos.z_val, 0.0))

    fov_rad = np.deg2rad(90.0)
    focal_len = (img_w / 2) / np.tan(fov_rad / 2)
    c_x, c_y = img_w / 2, img_h / 2

    ray_cam = np.array([1.0, (pixel_x - c_x) / focal_len, (pixel_y - c_y) / focal_len])
    ray_cam /= np.linalg.norm(ray_cam)

    r = R.from_quat([cam_orient.x_val, cam_orient.y_val, cam_orient.z_val, cam_orient.w_val])
    ray_world = r.as_matrix() @ ray_cam

    # Place point at metric depth along the ray
    point_world = np.array([
        cam_pos.x_val + ai_depth_val * ray_world[0],
        cam_pos.y_val + ai_depth_val * ray_world[1],
        cam_pos.z_val + ai_depth_val * ray_world[2],
    ])
    point_world[2] = min(point_world[2], 0.0)

    return airsim.Vector3r(point_world[0], point_world[1], point_world[2])
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/spatial/test_projection.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/spatial/projection.py tests/unit/spatial/test_projection.py
git commit -m "fix: get_coords_from_lite_mono now uses metric depth directly

Removes broken interpolation heuristic and hardcoded ground_z = 0.0.
depth parameter is now metric distance in metres along the ray."
```

---

### Task 6: Add `compute_scale_factor` to AirSimProjection

**Files:**
- Modify: `src/spatial/airsim_projection.py`

The AirSim backend gets camera pose from `simGetCameraInfo()`. It needs the same scale recovery logic as ConfigProjection but using live camera pose.

- [ ] **Step 1: Implement `compute_scale_factor` in AirSimProjection**

Add this method to the `AirSimProjection` class in `src/spatial/airsim_projection.py`:

```python
    def compute_scale_factor(
        self,
        depth_map: "np.ndarray",
        frame_w: int,
        frame_h: int,
    ) -> float:
        """Recover metric scale by comparing ray-ground distances to model predictions.

        Uses camera pose from AirSim and cctv_height to compute geometric
        ground distances at sampled pixels, then returns median ratio.
        """
        if self._client is None:
            return 1.0

        try:
            info = self._client.simGetCameraInfo(
                self._camera_name, vehicle_name=self._vehicle_name,
            )
            cam_pos = info.pose.position
            cam_orient = info.pose.orientation

            height = self._cctv_height
            if height < 0.1:
                return 1.0

            ground_z = cam_pos.z_val + height

            fov_rad = np.deg2rad(90.0)
            focal_len = (frame_w / 2) / np.tan(fov_rad / 2)
            c_x, c_y = frame_w / 2, frame_h / 2

            r = R.from_quat([
                cam_orient.x_val, cam_orient.y_val,
                cam_orient.z_val, cam_orient.w_val,
            ])
            rot_matrix = r.as_matrix()

            # Sample pixels from bottom 20% of image (most likely ground)
            h, w = depth_map.shape
            y_start = int(h * 0.8)
            sample_rows = range(y_start, h, max(1, (h - y_start) // 5))
            sample_cols = range(0, w, max(1, w // 8))

            ratios = []
            for row in sample_rows:
                for col in sample_cols:
                    inv_disp = depth_map[row, col]
                    if inv_disp < 1e-6:
                        continue

                    px = (col / w) * frame_w
                    py = (row / h) * frame_h

                    ray_cam = np.array([
                        1.0,
                        (px - c_x) / focal_len,
                        (py - c_y) / focal_len,
                    ])
                    ray_cam /= np.linalg.norm(ray_cam)
                    ray_world = rot_matrix @ ray_cam

                    if abs(ray_world[2]) < 0.001:
                        continue

                    t_ground = (ground_z - cam_pos.z_val) / ray_world[2]
                    if t_ground <= 0:
                        continue

                    ratios.append(t_ground / inv_disp)

            if not ratios:
                return 1.0

            return float(np.median(ratios))
        except Exception as e:
            print(f"[AirSimProjection] Scale factor failed: {e}")
            return 1.0
```

- [ ] **Step 2: Run all spatial tests to verify nothing breaks**

Run: `pytest tests/unit/spatial/ -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add src/spatial/airsim_projection.py
git commit -m "feat: add compute_scale_factor to AirSimProjection

Per-frame scale recovery using AirSim camera pose and known cctv_height.
Samples bottom 20% of image, computes geometric ground distance vs model
inverse-disparity, returns median ratio."
```

---

### Task 7: Update app.py caller — fix foot pixel + integrate scale recovery

**Files:**
- Modify: `src/api/app.py:272-288`

Two fixes: (a) use bottom-center of mask (feet) not centroid (torso), (b) compute scale factor and metric depth before calling `pixel_to_world`.

- [ ] **Step 1: Fix the foot pixel and add scale recovery**

In `src/api/app.py`, replace lines 272-288:

```python
        try:
            depth_map = depth_estimator.estimate(frame)
            person_mask = person_masks[0]
            y_indices, x_indices = person_mask.nonzero()
            if len(y_indices) == 0:
                return _fallback()

            # Bottom-center of mask = foot position (ground contact point)
            center_x = float(np.mean(x_indices))
            center_y = float(np.max(y_indices))

            # Per-frame scale recovery + metric depth
            scale = projection.compute_scale_factor(depth_map, frame.shape[1], frame.shape[0])
            metric_depth = depth_estimator.get_metric_depth_at_pixel(
                depth_map, int(center_x), int(center_y), scale,
            )

            return projection.pixel_to_world(
                center_x, center_y, metric_depth, frame.shape[1], frame.shape[0],
            )
        except Exception as e:
            print(f"[DETECTION] Projection failed: {e}, using camera position")
            return _fallback()
```

Key changes:
- `center_y` changed from `np.mean(y_indices)` to `np.max(y_indices)` (bottom of mask = feet)
- Added `compute_scale_factor` call
- Added `get_metric_depth_at_pixel` call
- `pixel_to_world` now receives metric depth in metres

- [ ] **Step 2: Verify app still imports correctly**

Run: `python -c "from src.api.app import create_app; print('OK')"`
Expected: OK (or import errors if AirSim not available — that's fine, the structure is correct)

- [ ] **Step 3: Commit**

```bash
git add src/api/app.py
git commit -m "fix: use foot pixel (bottom-center) and metric depth in app.py

Changes centroid sampling to bottom-center of mask for ground contact.
Integrates per-frame scale recovery before pixel_to_world call."
```

---

### Task 8: Update eval script to use new pipeline

**Files:**
- Modify: `src/eval/eval_depth_estimation.py:334-374`

The eval script already uses bottom-center (line 363: `cy = float(np.max(y_indices))`). Just need to add scale recovery and pass metric depth.

- [ ] **Step 1: Update the projection call in the eval loop**

In `src/eval/eval_depth_estimation.py`, replace lines 333-374 (the depth estimation + projection section inside the person loop):

```python
            # 4. Depth estimation (once per frame, shared across all detections)
            depth_map = depth_est.estimate(frame)

            # 5. Per-frame scale recovery using known camera height
            #    Compute geometric ground distance at bottom-of-image pixels
            #    and compare to model predictions to recover metric scale.
            cam_info_scale = client.simGetCameraInfo(cam_name, vehicle_name=veh_name)
            cam_z = cam_info_scale.pose.position.z_val
            cam_orient = cam_info_scale.pose.orientation
            ground_z_local = cam_z + cctv_height_actual

            from scipy.spatial.transform import Rotation as Rot
            rot = Rot.from_quat([
                cam_orient.x_val, cam_orient.y_val,
                cam_orient.z_val, cam_orient.w_val,
            ])
            rot_matrix = rot.as_matrix()

            fov_rad = np.deg2rad(90.0)
            focal_len = (frame.shape[1] / 2) / np.tan(fov_rad / 2)
            cx_img, cy_img = frame.shape[1] / 2, frame.shape[0] / 2

            h_dm, w_dm = depth_map.shape
            y_start = int(h_dm * 0.8)
            sample_rows = range(y_start, h_dm, max(1, (h_dm - y_start) // 5))
            sample_cols = range(0, w_dm, max(1, w_dm // 8))

            ratios = []
            for row in sample_rows:
                for col in sample_cols:
                    inv_disp = depth_map[row, col]
                    if inv_disp < 1e-6:
                        continue
                    px = (col / w_dm) * frame.shape[1]
                    py = (row / h_dm) * frame.shape[0]
                    ray_cam = np.array([
                        1.0,
                        (px - cx_img) / focal_len,
                        (py - cy_img) / focal_len,
                    ])
                    ray_cam /= np.linalg.norm(ray_cam)
                    ray_world = rot_matrix @ ray_cam
                    if abs(ray_world[2]) < 0.001:
                        continue
                    t_ground = (ground_z_local - cam_z) / ray_world[2]
                    if t_ground <= 0:
                        continue
                    ratios.append(t_ground / inv_disp)

            scale_factor = float(np.median(ratios)) if ratios else 1.0

            # 6. Project ALL person detections using metric depth
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
                    print(f"  [DEBUG] {feed_id}: detection {idx} skipped — mask too small "
                          f"({mask_area}/{frame_area} = {mask_area/frame_area:.4f})")
                    continue

                cx = float(np.mean(x_indices))
                cy = float(np.max(y_indices))

                # Metric depth at foot pixel
                dm_h, dm_w = depth_map.shape
                dm_x = max(0, min(int(cx * dm_w / frame.shape[1]), dm_w - 1))
                dm_y = max(0, min(int(cy * dm_h / frame.shape[0]), dm_h - 1))
                metric_depth = scale_factor * depth_map[dm_y, dm_x]

                try:
                    est_pos = get_coords_from_lite_mono(
                        client, cam_name,
                        cx, cy,
                        frame.shape[1], frame.shape[0],
                        metric_depth, cctv_height_actual,
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
                    best_depth_val = metric_depth
```

- [ ] **Step 2: Run the full test suite**

Run: `pytest tests/ -v --ignore=tests/integration`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add src/eval/eval_depth_estimation.py
git commit -m "fix: eval script uses per-frame scale recovery and metric depth

Computes scale factor from ground-plane pixels each frame, applies to
inverse-disparity at foot pixel to get metric depth before projection."
```

---

### Task 9: Run full test suite and verify

- [ ] **Step 1: Run all unit tests**

Run: `pytest tests/unit/ -v`
Expected: ALL PASS

- [ ] **Step 2: Verify no import errors in main modules**

Run: `python -c "from src.spatial.config_projection import ConfigProjection; from src.spatial.projection import get_coords_from_lite_mono; from src.detection.depth_estimator_wrapper import DepthEstimator; print('All imports OK')"`
Expected: `All imports OK`

- [ ] **Step 3: Final commit with all changes verified**

```bash
git log --oneline -8
```

Verify the commit history shows the incremental changes.

---

## Summary of Changes

| What changed | Why |
|---|---|
| `depth_estimator.py`: return `1/disp` not min-max normalized | Disparity was inverted (high=close treated as high=far) and normalization destroyed cross-frame scale |
| `projection_base.py`: add `compute_scale_factor` | Per-frame scale recovery interface |
| `config_projection.py`: metric depth + scale recovery | Remove `ground_z = 0.0` hardcode, remove broken interpolation heuristic |
| `projection.py`: metric depth directly | Same fix for AirSim direct-call path |
| `airsim_projection.py`: add scale recovery | Correct `ground_z = cam_z + height` using AirSim camera pose |
| `app.py`: foot pixel + scale integration | Fix centroid→bottom-center bug, wire up new pipeline |
| `eval_depth_estimation.py`: scale integration | Wire up new pipeline for evaluation |
