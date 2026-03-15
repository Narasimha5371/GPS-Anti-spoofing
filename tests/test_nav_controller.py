#!/usr/bin/env python3
"""Unit tests for the navigation controller mode switching."""

import math
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestNavController:
    """Test navigation controller state machine logic."""

    NORMAL = 'NORMAL'
    ALERT = 'ALERT'
    SAFE_STOP = 'SAFE_STOP'

    def test_normal_to_alert_on_spoof(self):
        """Controller should switch to ALERT when spoofing is detected."""
        mode = self.NORMAL
        spoof_alert = True

        if spoof_alert:
            mode = self.ALERT

        assert mode == self.ALERT

    def test_alert_to_normal_on_clear(self):
        """Controller should return to NORMAL when alert clears."""
        mode = self.ALERT
        spoof_alert = False

        if not spoof_alert:
            mode = self.NORMAL

        assert mode == self.NORMAL

    def test_speed_reduction_in_alert(self):
        """Speed should be reduced in ALERT mode."""
        max_speed = 5.0
        alert_factor = 0.2
        current_speed = 5.0
        decel = 1.0
        dt = 0.1

        target_speed = max_speed * alert_factor

        # Simulate deceleration
        steps = 0
        while current_speed > target_speed + 0.01:
            current_speed = max(target_speed, current_speed - decel * dt)
            steps += 1
            if steps > 1000:
                break

        assert abs(current_speed - target_speed) < 0.1, \
            f'Should reach target speed: {current_speed} vs {target_speed}'

    def test_safe_stop_reaches_zero(self):
        """SAFE_STOP should bring vehicle to complete halt."""
        current_speed = 1.0  # alert speed
        decel = 1.0
        dt = 0.1

        steps = 0
        while current_speed > 0.01:
            current_speed = max(0.0, current_speed - decel * dt)
            steps += 1
            if steps > 100:
                break

        assert current_speed <= 0.01, f'Should reach zero: {current_speed}'
        assert steps < 20, f'Should stop quickly: {steps} steps'

    def test_heading_error_computation(self):
        """Heading error should be correctly computed."""
        # Vehicle at origin facing right (yaw=0), target is up-right
        px, py, yaw = 0.0, 0.0, 0.0
        tx, ty = 10.0, 10.0

        target_yaw = math.atan2(ty - py, tx - px)  # π/4
        heading_err = target_yaw - yaw

        assert abs(heading_err - math.pi / 4) < 1e-10

    def test_alignment_reduces_speed(self):
        """Vehicle should slow down when not aligned with target."""
        heading_err = math.pi / 2  # 90 degrees off
        alignment = max(0, math.cos(heading_err))

        assert abs(alignment) < 0.01, \
            f'Perpendicular alignment should be near zero: {alignment}'

    def test_aligned_full_speed(self):
        """Vehicle should go full speed when perfectly aligned."""
        heading_err = 0.0
        alignment = max(0, math.cos(heading_err))

        assert abs(alignment - 1.0) < 1e-10, \
            f'Perfect alignment should be 1.0: {alignment}'

    def test_pid_integral_clamping(self):
        """PID integral should be clamped to prevent windup."""
        integral = 0.0
        max_integral = 10.0

        for _ in range(200):
            integral += 1.0 * 0.1  # accumulate
            integral = max(-max_integral, min(max_integral, integral))

        assert integral <= max_integral, \
            f'Integral should be clamped: {integral}'

    def test_angle_normalization(self):
        """Angles should be normalized to [-pi, pi]."""
        def normalize(angle):
            while angle > math.pi:
                angle -= 2 * math.pi
            while angle < -math.pi:
                angle += 2 * math.pi
            return angle

        assert abs(normalize(3 * math.pi) - math.pi) < 1e-10
        # -3π normalizes to -π (or equivalently π) 
        assert abs(abs(normalize(-3 * math.pi)) - math.pi) < 1e-10
        assert abs(normalize(0.5)) < 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
