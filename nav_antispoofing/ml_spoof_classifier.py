#!/usr/bin/env python3
"""
ML Spoofing Classifier Node — LSTM-based spoofing detector.

Collects a sliding window of sensor features and runs an LSTM model
to classify whether the current GPS signal is being spoofed.

Features per timestep:
  [gps_x, gps_y, ekf_x, ekf_y, drift, anomaly_score, speed]

If no trained model is available, falls back to a threshold-based
heuristic classifier.
"""

import math
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64
import numpy as np

# Try to import PyTorch — graceful fallback if not available
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ────────────────────────────────────────────────────────────────
# LSTM Model Definition
# ────────────────────────────────────────────────────────────────

if HAS_TORCH:
    class SpoofDetectorLSTM(nn.Module):
        """Lightweight LSTM for spoofing classification."""

        def __init__(self, input_size=7, hidden_size=32, num_layers=2):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size, hidden_size, num_layers,
                batch_first=True, dropout=0.2
            )
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, 16),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            # x: (batch, seq_len, features)
            lstm_out, _ = self.lstm(x)
            last_hidden = lstm_out[:, -1, :]  # last timestep
            return self.classifier(last_hidden).squeeze(-1)


# ────────────────────────────────────────────────────────────────
# ROS2 Node
# ────────────────────────────────────────────────────────────────

class MLSpoofClassifier(Node):
    def __init__(self):
        super().__init__('ml_spoof_classifier')

        # Parameters
        self.declare_parameter('origin_lat', 12.9716)
        self.declare_parameter('origin_lon', 77.5946)
        self.declare_parameter('window_size', 20)
        self.declare_parameter('model_path', '')
        self.declare_parameter('classification_threshold', 0.6)
        self.declare_parameter('inference_rate', 5.0)       # Hz

        self.origin_lat = self.get_parameter('origin_lat').value
        self.origin_lon = self.get_parameter('origin_lon').value
        self.window_size = self.get_parameter('window_size').value
        self.threshold = self.get_parameter('classification_threshold').value
        model_path = self.get_parameter('model_path').value
        rate = self.get_parameter('inference_rate').value

        # Feature buffer: each entry is [gps_x, gps_y, ekf_x, ekf_y,
        #                                drift, anomaly_score, speed]
        self.feature_buffer = []
        self.max_buffer = self.window_size * 2

        # Latest sensor values
        self.latest_gps = None    # (x, y, time)
        self.latest_ekf = None    # (x, y)
        self.latest_anomaly = 0.0
        self.prev_gps = None

        # Load model
        self.model = None
        self.using_ml = False
        if HAS_TORCH and model_path and os.path.exists(model_path):
            try:
                self.model = SpoofDetectorLSTM()
                self.model.load_state_dict(
                    torch.load(model_path, map_location='cpu')
                )
                self.model.eval()
                self.using_ml = True
                self.get_logger().info(f'Loaded LSTM model from {model_path}')
            except Exception as e:
                self.get_logger().warn(f'Failed to load model: {e}. Using heuristic.')
        else:
            if not HAS_TORCH:
                self.get_logger().info(
                    'PyTorch not available — using heuristic classifier'
                )
            else:
                self.get_logger().info(
                    'No model file specified — using heuristic classifier'
                )

        # Publishers
        self.detection_pub = self.create_publisher(
            Float64, '/spoofing/detection', 10
        )

        # Subscribers
        self.create_subscription(NavSatFix, '/gps/fix', self.gps_callback, 10)
        self.create_subscription(
            PoseStamped, '/fused_pose', self.fused_callback, 10
        )
        self.create_subscription(
            Float64, '/spoofing/anomaly_score', self.anomaly_callback, 10
        )

        # Inference timer
        self.timer = self.create_timer(1.0 / rate, self.run_inference)

        self.get_logger().info(
            f'ML Spoof Classifier started — '
            f'ml={self.using_ml}, window={self.window_size}'
        )

    def _latlon_to_xy(self, lat, lon):
        x = (lon - self.origin_lon) * 111320.0 * math.cos(
            math.radians(self.origin_lat)
        )
        y = (lat - self.origin_lat) * 111320.0
        return x, y

    def gps_callback(self, msg: NavSatFix):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        x, y = self._latlon_to_xy(msg.latitude, msg.longitude)
        self.prev_gps = self.latest_gps
        self.latest_gps = (x, y, t)
        self._update_features()

    def fused_callback(self, msg: PoseStamped):
        self.latest_ekf = (msg.pose.position.x, msg.pose.position.y)

    def anomaly_callback(self, msg: Float64):
        self.latest_anomaly = msg.data

    def _update_features(self):
        """Build a feature vector from latest sensor data."""
        if self.latest_gps is None or self.latest_ekf is None:
            return

        gps_x, gps_y, gps_t = self.latest_gps
        ekf_x, ekf_y = self.latest_ekf

        drift = math.sqrt((gps_x - ekf_x)**2 + (gps_y - ekf_y)**2)

        speed = 0.0
        if self.prev_gps is not None:
            px, py, pt = self.prev_gps
            dt = gps_t - pt
            if dt > 0.001:
                speed = math.sqrt((gps_x - px)**2 + (gps_y - py)**2) / dt

        feature = [gps_x, gps_y, ekf_x, ekf_y, drift,
                   self.latest_anomaly, speed]
        self.feature_buffer.append(feature)

        if len(self.feature_buffer) > self.max_buffer:
            self.feature_buffer = self.feature_buffer[-self.max_buffer:]

    def run_inference(self):
        """Run spoofing classification."""
        if len(self.feature_buffer) < self.window_size:
            # Not enough data yet
            msg = Float64()
            msg.data = 0.0
            self.detection_pub.publish(msg)
            return

        window = self.feature_buffer[-self.window_size:]

        if self.using_ml:
            probability = self._ml_inference(window)
        else:
            probability = self._heuristic_inference(window)

        msg = Float64()
        msg.data = float(probability)
        self.detection_pub.publish(msg)

    def _ml_inference(self, window):
        """Run LSTM model inference."""
        try:
            x = torch.tensor([window], dtype=torch.float32)
            with torch.no_grad():
                prob = self.model(x).item()
            return prob
        except Exception as e:
            self.get_logger().warn(f'ML inference failed: {e}')
            return self._heuristic_inference(window)

    def _heuristic_inference(self, window):
        """
        Enhanced heuristic classifier with strong gradual drift detection.

        Uses statistical analysis of the feature window:
        - Drift magnitude and trend (slope + acceleration)
        - Drift monotonicity (consistently increasing = suspicious)
        - Anomaly score accumulation
        - Speed irregularities
        """
        features = np.array(window)

        drifts = features[:, 4]      # drift column
        anomalies = features[:, 5]   # anomaly_score column
        speeds = features[:, 6]      # speed column

        # ── Drift analysis ──
        mean_drift = np.mean(drifts)
        max_drift = np.max(drifts)
        recent_drift = np.mean(drifts[-5:]) if len(drifts) >= 5 else mean_drift

        # Drift trend (slope via linear regression)
        x_vals = np.arange(len(drifts))
        drift_trend = np.polyfit(x_vals, drifts, 1)[0]

        # Drift acceleration (is the trend getting steeper?)
        if len(drifts) >= 10:
            first_half = drifts[:len(drifts)//2]
            second_half = drifts[len(drifts)//2:]
            drift_accel = np.mean(second_half) - np.mean(first_half)
        else:
            drift_accel = 0.0

        # Monotonicity check (what fraction of samples show increasing drift)
        if len(drifts) >= 3:
            increasing = sum(1 for i in range(1, len(drifts))
                           if drifts[i] > drifts[i-1])
            monotonic_ratio = increasing / (len(drifts) - 1)
        else:
            monotonic_ratio = 0.0

        mean_anomaly = np.mean(anomalies)
        speed_std = np.std(speeds)

        # ── Weighted scoring ──
        score = 0.0

        # 1. Drift magnitude (lowered threshold for earlier detection)
        if mean_drift > 1.5:
            score += min(0.3, mean_drift / 10.0)
        if max_drift > 3.0:
            score += min(0.15, max_drift / 20.0)

        # 2. Rising drift trend (key for gradual attacks)
        if drift_trend > 0.01:
            score += min(0.35, drift_trend * 3.0)

        # 3. Drift acceleration
        if drift_accel > 0.1:
            score += min(0.1, drift_accel / 2.0)

        # 4. Monotonically increasing drift
        if monotonic_ratio > 0.6 and mean_drift > 0.5:
            score += min(0.2, monotonic_ratio * 0.25)

        # 5. Anomaly score accumulation
        score += mean_anomaly * 0.35

        # 6. Erratic speed
        if speed_std > 5.0:
            score += min(0.1, speed_std / 50.0)

        return min(1.0, score)


def main(args=None):
    rclpy.init(args=args)
    node = MLSpoofClassifier()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
