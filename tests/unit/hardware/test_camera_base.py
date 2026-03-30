"""
Hardware contract tests for CameraBackend implementations.

Any class that implements CameraBackend must pass CameraContractTests.
Only FileCamera is exercised here (no AirSim dependency needed).
"""

import numpy as np
import pytest

from src.hardware.camera.base import CameraBackend
from src.hardware.camera.file_camera import FileCamera


class CameraContractTests:
    """Mixin — every CameraBackend implementation must pass these tests."""

    def get_backend(self) -> CameraBackend:
        raise NotImplementedError

    def test_implements_abc(self):
        backend = self.get_backend()
        assert isinstance(backend, CameraBackend)

    def test_grab_frame_returns_numpy_or_none(self):
        backend = self.get_backend()
        backend.connect()
        frame = backend.grab_frame()
        assert frame is None or isinstance(frame, np.ndarray)
        backend.disconnect()

    def test_resolution_type(self):
        backend = self.get_backend()
        backend.connect()
        w, h = backend.resolution
        assert isinstance(w, int) and isinstance(h, int)
        backend.disconnect()

    def test_disconnect_is_idempotent(self):
        backend = self.get_backend()
        backend.disconnect()   # safe even if never connected
        backend.disconnect()   # still no error

    def test_is_connected_false_before_connect(self):
        backend = self.get_backend()
        # Fresh instance — not yet connected
        assert not backend.is_connected

    def test_is_connected_true_after_connect(self):
        backend = self.get_backend()
        result = backend.connect()
        if result:
            assert backend.is_connected
        backend.disconnect()

    def test_is_connected_false_after_disconnect(self):
        backend = self.get_backend()
        backend.connect()
        backend.disconnect()
        assert not backend.is_connected


class TestFileCamera(CameraContractTests):
    """Run the full contract suite against FileCamera using a test fixture."""

    FIXTURE = "tests/fixtures/frames/empty_scene.jpg"

    def get_backend(self) -> CameraBackend:
        return FileCamera(self.FIXTURE)

    def test_still_image_returns_same_shape_every_call(self):
        cam = FileCamera(self.FIXTURE)
        cam.connect()
        f1 = cam.grab_frame()
        f2 = cam.grab_frame()
        assert f1 is not None and f2 is not None
        assert f1.shape == f2.shape
        cam.disconnect()

    def test_resolution_matches_frame(self):
        cam = FileCamera(self.FIXTURE)
        cam.connect()
        frame = cam.grab_frame()
        if frame is not None:
            h, w = frame.shape[:2]
            assert cam.resolution == (w, h)
        cam.disconnect()

    def test_connect_returns_false_for_missing_file(self):
        cam = FileCamera("/nonexistent/path/image.jpg")
        result = cam.connect()
        assert result is False
        assert not cam.is_connected
