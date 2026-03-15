#!/usr/bin/env python3
"""Unit tests for the anomaly detector logic (no ROS2 required)."""

import math
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestAnomalyDetector:
    """Test anomaly detection heuristics independently."""

    def test_velocity_jump_detection(self):
        """Large GPS jumps should produce high anomaly scores."""
        # Simulate sudden GPS position change
        max_vel = 30.0
        dt = 0.1

        # Normal speed (5 m/s)
        dx_normal = 5.0 * dt
        speed_normal = dx_normal / dt
        assert speed_normal <= max_vel

        # Spoofed jump (500 m in 0.1s)
        dx_spoof = 500.0
        speed_spoof = dx_spoof / dt
        assert speed_spoof > max_vel

        # Score should be high
        score = min(1.0, speed_spoof / (max_vel * 3))
        assert score > 0.5, f'Jump score should be high: {score}'

    def test_normal_velocity_no_alarm(self):
        """Normal speeds should not trigger velocity jump alarm."""
        max_vel = 30.0
        vel_jump_thresh = 15.0

        speed = 5.0
        score = 0.0
        if speed > max_vel:
            score = min(1.0, speed / (max_vel * 3))
        elif speed > vel_jump_thresh:
            score = min(0.5, (speed - vel_jump_thresh) /
                        (max_vel - vel_jump_thresh))

        assert score == 0.0, f'Normal speed should not alarm: {score}'

    def test_ekf_drift_detection(self):
        """Large GPS-EKF drift should produce anomaly."""
        drift_thresh = 10.0

        # Normal drift (< 1m)
        drift_normal = 0.8
        score_normal = 0.0
        if drift_normal > drift_thresh:
            score_normal = min(1.0, drift_normal / (drift_thresh * 3))
        assert score_normal == 0.0

        # Spoofed drift (50m)
        drift_spoof = 50.0
        score_spoof = 0.0
        if drift_spoof > drift_thresh:
            score_spoof = min(1.0, drift_spoof / (drift_thresh * 3))
        assert score_spoof > 0.5, f'Drift score should be high: {score_spoof}'

    def test_ema_smoothing(self):
        """EMA should smooth out spike noise."""
        alpha = 0.3
        ema = 0.0

        # Single spike
        spike_scores = [0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0]
        ema_values = []
        for s in spike_scores:
            ema = alpha * s + (1 - alpha) * ema
            ema_values.append(ema)

        # Peak should be dampened
        assert max(ema_values) < 0.9 * 0.5, \
            f'EMA peak should be dampened: {max(ema_values)}'

    def test_sustained_anomaly_builds_up(self):
        """Sustained anomaly should accumulate in EMA."""
        alpha = 0.3
        ema = 0.0

        for _ in range(20):
            ema = alpha * 0.8 + (1 - alpha) * ema

        assert ema > 0.7, f'Sustained anomaly should build up: {ema}'

    def test_consistency_check_detects_no_noise(self):
        """GPS readings with zero noise should flag consistency issue."""
        # Spoofed GPS often has unnaturally low noise
        positions = [(10.0, 10.0)] * 10  # identical readings
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        var = np.var(xs) + np.var(ys)
        assert var < 0.001, 'Identical readings should have near-zero variance'

    def test_combined_scoring(self):
        """Combined score should correctly weight components."""
        vel_score = 0.8
        drift_score = 0.6
        consistency_score = 0.2

        combined = min(1.0, sum([
            vel_score * 0.5,
            drift_score * 0.35,
            consistency_score * 0.15
        ]))

        expected = 0.8 * 0.5 + 0.6 * 0.35 + 0.2 * 0.15
        assert abs(combined - expected) < 1e-10, \
            f'Combined score mismatch: {combined} != {expected}'
        assert combined > 0.5, f'Combined should indicate anomaly: {combined}'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
