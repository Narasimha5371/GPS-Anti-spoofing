#!/usr/bin/env python3
"""Unit tests for the EKF fusion logic (no ROS2 required)."""

import math
import sys
import os
import numpy as np
import pytest

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestEKF:
    """Test the EKF math independently of ROS2."""

    def setup_method(self):
        """Set up a minimal EKF state."""
        self.x = np.zeros(6)       # [x, y, vx, vy, yaw, yaw_rate]
        self.P = np.eye(6) * 10.0
        self.Q = np.diag([0.1, 0.1, 0.5, 0.5, 0.01, 0.001])

    def predict(self, dt):
        """EKF predict step."""
        x = self.x.copy()
        self.x[0] = x[0] + x[2] * dt
        self.x[1] = x[1] + x[3] * dt
        self.x[4] = x[4] + x[5] * dt

        F = np.eye(6)
        F[0, 2] = dt
        F[1, 3] = dt
        F[4, 5] = dt
        self.P = F @ self.P @ F.T + self.Q * dt

    def update(self, z, H, R):
        """EKF update step."""
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P

    def test_predict_position_from_velocity(self):
        """State prediction: position should propagate from velocity."""
        self.x = np.array([0.0, 0.0, 5.0, 3.0, 0.0, 0.0])
        self.predict(1.0)

        assert abs(self.x[0] - 5.0) < 1e-10, f'x={self.x[0]}'
        assert abs(self.x[1] - 3.0) < 1e-10, f'y={self.x[1]}'

    def test_predict_yaw_from_yaw_rate(self):
        """Yaw should propagate from yaw rate."""
        self.x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
        self.predict(2.0)

        assert abs(self.x[4] - 1.0) < 1e-10, f'yaw={self.x[4]}'

    def test_covariance_grows_with_predict(self):
        """Covariance should increase during prediction."""
        P_before = self.P.copy()
        self.predict(1.0)

        for i in range(6):
            assert self.P[i, i] >= P_before[i, i], \
                f'P[{i},{i}] should grow: {P_before[i, i]} -> {self.P[i, i]}'

    def test_update_reduces_covariance(self):
        """Measurement update should reduce uncertainty."""
        self.predict(1.0)
        P_before_update = self.P.copy()

        z = np.array([1.0, 1.0])
        H = np.zeros((2, 6))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        R = np.eye(2) * 0.5

        self.update(z, H, R)

        assert self.P[0, 0] < P_before_update[0, 0], 'x covariance should decrease'
        assert self.P[1, 1] < P_before_update[1, 1], 'y covariance should decrease'

    def test_update_moves_state_toward_measurement(self):
        """State should move toward measurement after update."""
        self.x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        z = np.array([10.0, 20.0])
        H = np.zeros((2, 6))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        R = np.eye(2) * 1.0

        self.update(z, H, R)

        assert self.x[0] > 0, f'x should move toward 10: {self.x[0]}'
        assert self.x[1] > 0, f'y should move toward 20: {self.x[1]}'

    def test_convergence_with_repeated_measurements(self):
        """State should converge to measurement after many updates."""
        true_pos = np.array([50.0, 30.0])
        H = np.zeros((2, 6))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        R = np.eye(2) * 1.0

        for _ in range(100):
            self.predict(0.1)
            z = true_pos + np.random.normal(0, 0.5, 2)
            self.update(z, H, R)

        assert abs(self.x[0] - 50.0) < 3.0, f'x should converge: {self.x[0]}'
        assert abs(self.x[1] - 30.0) < 3.0, f'y should converge: {self.x[1]}'

    def test_multiple_sensor_fusion(self):
        """Fusing GPS + odometry should reduce covariance faster than either alone."""
        # GPS-only
        x1 = np.zeros(6)
        P1 = np.eye(6) * 10.0

        # GPS + odom
        x2 = np.zeros(6)
        P2 = np.eye(6) * 10.0

        H_gps = np.zeros((2, 6))
        H_gps[0, 0] = 1.0
        H_gps[1, 1] = 1.0
        R_gps = np.eye(2) * 2.0

        H_odom = np.zeros((4, 6))
        H_odom[0, 0] = 1.0
        H_odom[1, 1] = 1.0
        H_odom[2, 2] = 1.0
        H_odom[3, 3] = 1.0
        R_odom = np.eye(4) * 0.5

        z_gps = np.array([10.0, 10.0])
        z_odom = np.array([10.0, 10.0, 1.0, 1.0])

        for _ in range(20):
            # GPS only
            F = np.eye(6)
            P1 = F @ P1 @ F.T + self.Q * 0.1
            y1 = z_gps - H_gps @ x1
            S1 = H_gps @ P1 @ H_gps.T + R_gps
            K1 = P1 @ H_gps.T @ np.linalg.inv(S1)
            x1 = x1 + K1 @ y1
            P1 = (np.eye(6) - K1 @ H_gps) @ P1

            # GPS + odom
            P2 = F @ P2 @ F.T + self.Q * 0.1
            y2 = z_gps - H_gps @ x2
            S2 = H_gps @ P2 @ H_gps.T + R_gps
            K2 = P2 @ H_gps.T @ np.linalg.inv(S2)
            x2 = x2 + K2 @ y2
            P2 = (np.eye(6) - K2 @ H_gps) @ P2

            y3 = z_odom - H_odom @ x2
            S3 = H_odom @ P2 @ H_odom.T + R_odom
            K3 = P2 @ H_odom.T @ np.linalg.inv(S3)
            x2 = x2 + K3 @ y3
            P2 = (np.eye(6) - K3 @ H_odom) @ P2

        assert P2[0, 0] < P1[0, 0], 'Dual-sensor should have lower uncertainty'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
