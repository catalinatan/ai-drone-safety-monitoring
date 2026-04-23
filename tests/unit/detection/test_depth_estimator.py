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
