"""Unit tests for DroneAPIClient — all HTTP calls mocked."""

import pytest
from unittest.mock import patch, MagicMock
from src.backend.drone_client import DroneAPIClient


class TestDroneAPIClient:

    def _make_client(self):
        return DroneAPIClient(base_url="http://localhost:9999", timeout=1)

    @patch("src.backend.drone_client.requests.get")
    def test_check_connection_success(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200)
        client = self._make_client()
        assert client.check_connection() is True
        mock_get.assert_called_once()

    @patch("src.backend.drone_client.requests.get")
    def test_check_connection_failure(self, mock_get):
        import requests as req
        mock_get.side_effect = req.exceptions.ConnectionError("refused")
        client = self._make_client()
        assert client.check_connection() is False

    @patch("src.backend.drone_client.requests.post")
    def test_set_mode_success(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        client = self._make_client()
        assert client.set_mode("automatic") is True
        call_kwargs = mock_post.call_args
        assert call_kwargs[1]["json"] == {"mode": "automatic"}

    @patch("src.backend.drone_client.requests.post")
    def test_set_mode_failure(self, mock_post):
        mock_post.return_value = MagicMock(status_code=500)
        client = self._make_client()
        assert client.set_mode("automatic") is False

    @patch("src.backend.drone_client.requests.post")
    def test_goto_position_success(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        client = self._make_client()
        assert client.goto_position(10.0, 20.0, -5.0) is True
        call_kwargs = mock_post.call_args
        assert call_kwargs[1]["json"] == {"x": 10.0, "y": 20.0, "z": -5.0}

    @patch("src.backend.drone_client.requests.post")
    def test_goto_position_failure(self, mock_post):
        mock_post.return_value = MagicMock(status_code=400, json=MagicMock(return_value={"detail": "bad"}))
        client = self._make_client()
        assert client.goto_position(10.0, 20.0, -5.0) is False

    @patch("src.backend.drone_client.requests.get")
    def test_get_status_success(self, mock_get):
        expected = {"mode": "automatic", "is_navigating": False}
        mock_get.return_value = MagicMock(status_code=200, json=MagicMock(return_value=expected))
        client = self._make_client()
        result = client.get_status()
        assert result == expected

    @patch("src.backend.drone_client.requests.get")
    def test_get_status_unreachable(self, mock_get):
        import requests as req
        mock_get.side_effect = req.exceptions.ConnectionError("refused")
        client = self._make_client()
        assert client.get_status() is None
