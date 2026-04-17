"""Unit tests for MAVLink drone backend."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock pymavlink before importing MAVLinkDrone
sys.modules["pymavlink"] = MagicMock()
sys.modules["pymavlink.mavutil"] = MagicMock()

from src.hardware.drone.base import DronePosition, DroneStatus
from src.hardware.drone.mavlink_drone import MAVLinkDrone


class TestMAVLinkDroneConnect:
    """Tests for MAVLinkDrone.connect()."""

    def test_connect_success(self):
        """connect() returns True on successful connection."""
        drone = MAVLinkDrone()

        # Mock successful connection
        mock_conn = MagicMock()
        mock_conn.wait_heartbeat.return_value = MagicMock()  # Mock heartbeat message
        mock_conn.target_system = 1
        mock_conn.target_component = 1

        sys.modules["pymavlink"].mavutil.mavlink_connection.return_value = mock_conn

        result = drone.connect()
        assert result is True
        assert drone._is_connected is True

    def test_connect_heartbeat_timeout(self):
        """connect() returns False when heartbeat times out."""
        drone = MAVLinkDrone()

        mock_conn = MagicMock()
        mock_conn.wait_heartbeat.return_value = None  # Timeout
        sys.modules["pymavlink"].mavutil.mavlink_connection.return_value = mock_conn

        result = drone.connect()
        assert result is False
        assert drone._is_connected is False

    def test_connect_exception(self):
        """connect() returns False on any exception."""
        drone = MAVLinkDrone()

        sys.modules["pymavlink"].mavutil.mavlink_connection.side_effect = Exception("Connection failed")

        result = drone.connect()
        assert result is False

        # Reset for other tests
        sys.modules["pymavlink"].mavutil.mavlink_connection.side_effect = None


class TestMAVLinkDroneGoto:
    """Tests for MAVLinkDrone.goto()."""

    def test_goto_success(self):
        """goto() returns True and sets _is_navigating."""
        drone = MAVLinkDrone()
        drone._is_connected = True

        mock_vehicle = MagicMock()
        drone._vehicle = mock_vehicle

        # Mock the mavutil module for goto()
        with patch("pymavlink.mavutil.mavlink") as mock_mavlink:
            mock_mavlink.MAV_FRAME_LOCAL_NED = 1
            mock_mavlink.MAV_CMD_NAV_WAYPOINT = 16

            pos = DronePosition(x=10.0, y=20.0, z=-5.0)
            result = drone.goto(pos, speed=5.0)

            assert result is True
            assert drone._is_navigating is True
            # Verify mav.mission_item_int_send was called
            mock_vehicle.mav.mission_item_int_send.assert_called_once()

    def test_goto_not_connected(self):
        """goto() returns False if not connected."""
        drone = MAVLinkDrone()
        drone._is_connected = False

        pos = DronePosition(x=10.0, y=20.0, z=-5.0)
        result = drone.goto(pos, speed=5.0)

        assert result is False

    def test_goto_exception(self):
        """goto() returns False on exception."""
        drone = MAVLinkDrone()
        drone._is_connected = True

        mock_vehicle = MagicMock()
        mock_vehicle.mav.mission_item_int_send.side_effect = Exception("MAV error")
        drone._vehicle = mock_vehicle

        pos = DronePosition(x=10.0, y=20.0, z=-5.0)
        result = drone.goto(pos, speed=5.0)

        assert result is False


class TestMAVLinkDroneGetStatus:
    """Tests for MAVLinkDrone.get_status()."""

    def test_get_status_not_connected(self):
        """get_status() returns disconnected status when not connected."""
        drone = MAVLinkDrone()
        drone._is_connected = False

        status = drone.get_status()
        assert status.is_connected is False
        assert status.mode == "manual"

    def test_get_status_connected_with_heartbeat(self):
        """get_status() parses heartbeat and returns status."""
        drone = MAVLinkDrone()
        drone._is_connected = True

        mock_vehicle = MagicMock()

        # Mock heartbeat message
        mock_heartbeat = MagicMock()
        mock_heartbeat.custom_mode = 4  # GUIDED mode
        mock_vehicle.recv_match.side_effect = [mock_heartbeat, None]  # Heartbeat then no position

        drone._vehicle = mock_vehicle

        status = drone.get_status()
        assert status.is_connected is True
        assert status.mode == "automatic"  # GUIDED maps to "automatic"

    def test_get_status_with_position(self):
        """get_status() includes position when available."""
        drone = MAVLinkDrone()
        drone._is_connected = True

        mock_vehicle = MagicMock()

        # Mock position message
        mock_pos = MagicMock()
        mock_pos.x = 10.0
        mock_pos.y = 20.0
        mock_pos.z = -5.0
        mock_vehicle.recv_match.return_value = mock_pos

        drone._vehicle = mock_vehicle

        status = drone.get_status()
        assert status.position.x == 10.0
        assert status.position.y == 20.0
        assert status.position.z == -5.0


class TestMAVLinkDroneSetMode:
    """Tests for MAVLinkDrone.set_mode()."""

    def test_set_mode_automatic(self):
        """set_mode('automatic') sends GUIDED mode command."""
        drone = MAVLinkDrone()
        drone._is_connected = True

        mock_vehicle = MagicMock()
        drone._vehicle = mock_vehicle

        result = drone.set_mode("automatic")
        assert result is True
        mock_vehicle.mav.set_mode_send.assert_called_once()

    def test_set_mode_manual(self):
        """set_mode('manual') sends STABILIZE mode command."""
        drone = MAVLinkDrone()
        drone._is_connected = True

        mock_vehicle = MagicMock()
        drone._vehicle = mock_vehicle

        result = drone.set_mode("manual")
        assert result is True
        mock_vehicle.mav.set_mode_send.assert_called_once()

    def test_set_mode_unknown(self):
        """set_mode() with unknown mode returns False."""
        drone = MAVLinkDrone()
        drone._is_connected = True
        drone._vehicle = MagicMock()

        result = drone.set_mode("unknown_mode")
        assert result is False

    def test_set_mode_not_connected(self):
        """set_mode() returns False if not connected."""
        drone = MAVLinkDrone()
        drone._is_connected = False

        result = drone.set_mode("automatic")
        assert result is False


class TestMAVLinkDroneReturnHome:
    """Tests for MAVLinkDrone.return_home()."""

    def test_return_home_success(self):
        """return_home() sends RTH command."""
        drone = MAVLinkDrone()
        drone._is_connected = True

        mock_vehicle = MagicMock()
        drone._vehicle = mock_vehicle

        with patch("pymavlink.mavutil.mavlink") as mock_mavlink:
            mock_mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH = 20

            result = drone.return_home()
            assert result is True
            mock_vehicle.mav.command_long_send.assert_called_once()

    def test_return_home_not_connected(self):
        """return_home() returns False if not connected."""
        drone = MAVLinkDrone()
        drone._is_connected = False

        result = drone.return_home()
        assert result is False

    def test_return_home_exception(self):
        """return_home() returns False on exception."""
        drone = MAVLinkDrone()
        drone._is_connected = True

        mock_vehicle = MagicMock()
        mock_vehicle.mav.command_long_send.side_effect = Exception("MAV error")
        drone._vehicle = mock_vehicle

        result = drone.return_home()
        assert result is False


class TestMAVLinkDroneGrabFrame:
    """Tests for MAVLinkDrone.grab_frame()."""

    def test_grab_frame_returns_none(self):
        """grab_frame() always returns None (MAVLink doesn't support video)."""
        drone = MAVLinkDrone()
        drone._is_connected = True
        drone._vehicle = MagicMock()

        frame = drone.grab_frame()
        assert frame is None


class TestMAVLinkDroneDisconnect:
    """Tests for MAVLinkDrone.disconnect()."""

    def test_disconnect_success(self):
        """disconnect() closes vehicle connection."""
        drone = MAVLinkDrone()
        drone._is_connected = True
        drone._is_navigating = True

        mock_vehicle = MagicMock()
        drone._vehicle = mock_vehicle

        drone.disconnect()

        assert drone._is_connected is False
        assert drone._is_navigating is False
        assert drone._vehicle is None
        mock_vehicle.close.assert_called_once()

    def test_disconnect_no_vehicle(self):
        """disconnect() safely handles None vehicle."""
        drone = MAVLinkDrone()
        drone._vehicle = None

        # Should not crash
        drone.disconnect()
        assert drone._is_connected is False

    def test_disconnect_close_exception(self):
        """disconnect() handles close() exceptions gracefully."""
        drone = MAVLinkDrone()
        drone._is_connected = True

        mock_vehicle = MagicMock()
        mock_vehicle.close.side_effect = Exception("Close failed")
        drone._vehicle = mock_vehicle

        # Should not raise
        drone.disconnect()
        assert drone._is_connected is False
        assert drone._vehicle is None


class TestMAVLinkDroneInitialization:
    """Tests for MAVLinkDrone initialization."""

    def test_init_default_connection_string(self):
        """MAVLinkDrone initializes with default SITL connection string."""
        drone = MAVLinkDrone()
        assert drone.connection_string == "udp:127.0.0.1:14550"
        assert drone._is_connected is False
        assert drone._is_navigating is False

    def test_init_custom_connection_string(self):
        """MAVLinkDrone accepts custom connection string."""
        drone = MAVLinkDrone(connection_string="tcp:192.168.1.1:5760")
        assert drone.connection_string == "tcp:192.168.1.1:5760"

    def test_init_mode_map(self):
        """MAVLinkDrone mode map contains expected modes."""
        drone = MAVLinkDrone()
        assert drone._MODE_MAP["automatic"] == 4  # GUIDED
        assert drone._MODE_MAP["manual"] == 0  # STABILIZE
