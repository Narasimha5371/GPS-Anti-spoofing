#!/usr/bin/env python3
"""Unit tests for the ML spoof classifier heuristic logic."""

import math
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestMLClassifier:
    """Test the heuristic fallback classifier."""

    def _make_window(self, n=20, drift=0.5, anomaly=0.0, speed=5.0,
                     drift_trend=0.0):
        """Create a synthetic feature window."""
        window = []
        for i in range(n):
            d = drift + drift_trend * i
            window.append([
                100.0 + i * 0.5,      # gps_x
                50.0 + i * 0.3,       # gps_y
                100.0 + i * 0.5,      # ekf_x (close to GPS normally)
                50.0 + i * 0.3,       # ekf_y
                d,                     # drift
                anomaly,               # anomaly_score
                speed + np.random.normal(0, 0.5),  # speed
            ])
        return window

    def _heuristic_inference(self, window):
        """Replicate the heuristic classifier logic."""
        features = np.array(window)
        drifts = features[:, 4]
        anomalies = features[:, 5]
        speeds = features[:, 6]

        mean_drift = np.mean(drifts)
        drift_trend = np.polyfit(range(len(drifts)), drifts, 1)[0]
        mean_anomaly = np.mean(anomalies)
        speed_std = np.std(speeds)

        score = 0.0
        if mean_drift > 5.0:
            score += min(0.4, mean_drift / 25.0)
        if drift_trend > 0.1:
            score += min(0.2, drift_trend)
        score += mean_anomaly * 0.3
        if speed_std > 10.0:
            score += min(0.1, speed_std / 100.0)

        return min(1.0, score)

    def test_normal_conditions_low_score(self):
        """Normal driving should produce low spoof probability."""
        window = self._make_window(drift=0.5, anomaly=0.0)
        score = self._heuristic_inference(window)
        assert score < 0.2, f'Normal should be low: {score}'

    def test_high_drift_detected(self):
        """High GPS-EKF drift should flag spoofing."""
        window = self._make_window(drift=15.0, anomaly=0.3)
        score = self._heuristic_inference(window)
        assert score > 0.3, f'High drift should flag: {score}'

    def test_rising_drift_trend(self):
        """Increasing drift over time should flag gradual spoofing."""
        window = self._make_window(drift=1.0, drift_trend=2.0, anomaly=0.1)
        score = self._heuristic_inference(window)
        assert score > 0.2, f'Rising drift should flag: {score}'

    def test_high_anomaly_score(self):
        """High anomaly score should contribute to detection."""
        window = self._make_window(drift=0.5, anomaly=0.8)
        score = self._heuristic_inference(window)
        assert score > 0.2, f'High anomaly should flag: {score}'

    def test_combined_indicators(self):
        """Multiple indicators together should produce high confidence."""
        window = self._make_window(
            drift=20.0, anomaly=0.7, drift_trend=1.0
        )
        score = self._heuristic_inference(window)
        assert score > 0.5, f'Combined indicators should be high: {score}'

    def test_window_size_respected(self):
        """Feature window should be correct size."""
        window = self._make_window(n=20)
        assert len(window) == 20
        assert len(window[0]) == 7


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
