"""Unit tests for AlarmState — state machine and cooldown logic."""

import time
import pytest

from src.core.alarm import AlarmState


class TestAlarmState:
    def test_initially_not_active(self):
        alarm = AlarmState(cooldown_seconds=1.0)
        assert alarm.is_active is False

    def test_trigger_activates_alarm(self):
        alarm = AlarmState(cooldown_seconds=1.0)
        fired = alarm.trigger()
        assert fired is True
        assert alarm.is_active is True

    def test_trigger_returns_false_during_cooldown(self):
        alarm = AlarmState(cooldown_seconds=10.0)
        alarm.trigger()  # first trigger — should fire
        fired = alarm.trigger()  # too soon — cooldown not elapsed
        assert fired is False

    def test_clear_deactivates_alarm(self):
        alarm = AlarmState(cooldown_seconds=1.0)
        alarm.trigger()
        alarm.clear()
        assert alarm.is_active is False

    def test_cooldown_remaining_decreases(self):
        alarm = AlarmState(cooldown_seconds=5.0)
        alarm.trigger()
        r1 = alarm.cooldown_remaining
        time.sleep(0.05)
        r2 = alarm.cooldown_remaining
        assert r1 >= r2

    def test_cooldown_remaining_zero_before_first_trigger(self):
        alarm = AlarmState(cooldown_seconds=5.0)
        # Cooldown starts at 0 (never triggered = elapsed time is infinite)
        assert alarm.cooldown_remaining == 0.0

    def test_trigger_resets_cooldown(self):
        alarm = AlarmState(cooldown_seconds=0.05)
        alarm.trigger()
        time.sleep(0.06)  # cooldown expires
        fired = alarm.trigger()
        assert fired is True  # should fire again

    def test_cooldown_property(self):
        alarm = AlarmState(cooldown_seconds=3.14)
        assert alarm.cooldown_seconds == 3.14
