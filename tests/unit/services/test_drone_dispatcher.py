"""Unit tests for DroneDispatcher — dispatch logic and policy checks."""

import pytest

from src.services.drone_dispatcher import DroneDispatcher


class MockDroneAPI:
    def __init__(self, connected=True, mode="automatic", navigating=False):
        self.connected = connected
        self.mode = mode
        self.navigating = navigating
        self.goto_calls: list = []
        self.base_url = "http://localhost:8000"
        self.timeout = 5

    def check_connection(self):
        return self.connected

    def get_status(self):
        if not self.connected:
            return None
        return {"mode": self.mode, "is_navigating": self.navigating}

    def set_mode(self, mode):
        self.mode = mode
        return True

    def goto_position(self, x, y, z):
        self.goto_calls.append((x, y, z))
        return True


class TestDroneDispatcher:
    def test_auto_deploy_sends_goto(self):
        api = MockDroneAPI()
        dispatcher = DroneDispatcher(api, trigger_cooldown=0.0)
        result = dispatcher.try_auto_deploy(10.0, 5.0, -10.0)
        assert result is True
        assert len(api.goto_calls) == 1
        assert api.goto_calls[0] == (10.0, 5.0, -10.0)

    def test_auto_deploy_only_fires_once(self):
        api = MockDroneAPI()
        dispatcher = DroneDispatcher(api, trigger_cooldown=0.0)
        dispatcher.try_auto_deploy(1.0, 2.0, -10.0)
        result = dispatcher.try_auto_deploy(3.0, 4.0, -10.0)
        assert result is False
        assert len(api.goto_calls) == 1

    def test_auto_deploy_blocked_during_navigation(self):
        api = MockDroneAPI(navigating=True)
        dispatcher = DroneDispatcher(api, trigger_cooldown=0.0)
        result = dispatcher.try_auto_deploy(1.0, 2.0, -10.0)
        assert result is False
        assert len(api.goto_calls) == 0

    def test_auto_deploy_blocked_in_manual_mode(self):
        api = MockDroneAPI(mode="manual")
        dispatcher = DroneDispatcher(api, trigger_cooldown=0.0)
        result = dispatcher.try_auto_deploy(1.0, 2.0, -10.0)
        assert result is False

    def test_auto_deploy_no_api(self):
        dispatcher = DroneDispatcher(None, trigger_cooldown=0.0)
        assert dispatcher.try_auto_deploy(1.0, 2.0, -10.0) is False

    def test_manual_deploy_always_works(self):
        api = MockDroneAPI()
        dispatcher = DroneDispatcher(api)
        dispatcher.try_auto_deploy(1.0, 2.0, -10.0)  # lock first auto-deploy
        result = dispatcher.manual_deploy(5.0, 6.0, -10.0)
        assert result is True
        assert len(api.goto_calls) == 2

    def test_is_navigating_after_deploy(self):
        api = MockDroneAPI()
        dispatcher = DroneDispatcher(api, trigger_cooldown=0.0)
        dispatcher.try_auto_deploy(1.0, 2.0, -10.0)
        assert dispatcher.is_navigating is True

    def test_first_auto_deployed_flag(self):
        api = MockDroneAPI()
        dispatcher = DroneDispatcher(api, trigger_cooldown=0.0)
        assert dispatcher.first_auto_deployed is False
        dispatcher.try_auto_deploy(1.0, 2.0, -10.0)
        assert dispatcher.first_auto_deployed is True
