#!/usr/bin/env python3
"""
Anomaly Detector Node — Physics-based spoofing detection.

Compares raw GPS readings against the EKF fused pose to detect
inconsistencies that indicate GPS spoofing. Uses five checks:

1. Velocity jump detection — sudden GPS position changes beyond physical limits
2. GPS vs EKF drift — divergence between raw GPS and fused estimate
3. Drift trend — detects gradually increasing drift over time (key for gradual attacks)
4. Cross-sensor velocity — compares GPS-implied velocity vs odometry velocity
5. Position consistency — detects unnaturally low GPS noise (spoofed signals)
"""

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64, String
import numpy as np
import json


class AnomalyDetectorNode(Node):
    def __init__(self):
        super().__init__('anomaly_detector_node')

        # Parameters
        self.declare_parameter('origin_lat', 12.9716)
        self.declare_parameter('origin_lon', 77.5946)
        self.declare_parameter('max_velocity_mps', 30.0)
        self.declare_parameter('drift_threshold_m', 3.0)          # lowered from 10m
        self.declare_parameter('velocity_jump_threshold', 15.0)
        self.declare_parameter('drift_trend_window', 30)          # samples for trend
        self.declare_parameter('drift_trend_threshold', 0.05)     # m/sample slope
        self.declare_parameter('velocity_mismatch_threshold', 2.0)  # m/s
        self.declare_parameter('history_window', 50)              # increased from 20
        self.declare_parameter('anomaly_ema_alpha', 0.3)

        self.origin_lat = self.get_parameter('origin_lat').value
        self.origin_lon = self.get_parameter('origin_lon').value
        self.max_vel = self.get_parameter('max_velocity_mps').value
        self.drift_thresh = self.get_parameter('drift_threshold_m').value
        self.vel_jump_thresh = self.get_parameter('velocity_jump_threshold').value
        self.drift_trend_window = self.get_parameter('drift_trend_window').value
        self.drift_trend_thresh = self.get_parameter('drift_trend_threshold').value
        self.vel_mismatch_thresh = self.get_parameter('velocity_mismatch_threshold').value
        self.history_window = self.get_parameter('history_window').value
        self.ema_alpha = self.get_parameter('anomaly_ema_alpha').value

        # State
        self.gps_history = []        # list of (time, x, y)
        self.drift_history = []      # list of drift values for trend analysis
        self.last_fused_pose = None  # (x, y)
        self.last_odom_vel = None    # (vx, vy)
        self.anomaly_score_ema = 0.0

        # Publishers
        self.score_pub = self.create_publisher(
            Float64, '/spoofing/anomaly_score', 10
        )
        self.detail_pub = self.create_publisher(
            String, '/spoofing/anomaly_detail', 10
        )

        # Subscribers
        self.create_subscription(NavSatFix, '/gps/fix', self.gps_callback, 10)
        self.create_subscription(
            PoseStamped, '/fused_pose', self.fused_callback, 10
        )
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        self.get_logger().info('Anomaly Detector started (enhanced drift detection)')

    def _latlon_to_xy(self, lat, lon):
        """Convert lat/lon to local XY meters."""
        x = (lon - self.origin_lon) * 111320.0 * math.cos(
            math.radians(self.origin_lat)
        )
        y = (lat - self.origin_lat) * 111320.0
        return x, y

    def fused_callback(self, msg: PoseStamped):
        self.last_fused_pose = (
            msg.pose.position.x,
            msg.pose.position.y
        )

    def odom_callback(self, msg: Odometry):
        self.last_odom_vel = (
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y
        )

    def gps_callback(self, msg: NavSatFix):
        now_sec = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        gps_x, gps_y = self._latlon_to_xy(msg.latitude, msg.longitude)

        scores = {}

        # --- Check 1: Velocity jump ---
        velocity_jump_score = 0.0
        gps_vx, gps_vy = 0.0, 0.0
        if len(self.gps_history) > 0:
            prev_t, prev_x, prev_y = self.gps_history[-1]
            dt = now_sec - prev_t
            if dt > 0.001:
                dx = gps_x - prev_x
                dy = gps_y - prev_y
                gps_vx = dx / dt
                gps_vy = dy / dt
                speed = math.sqrt(dx**2 + dy**2) / dt

                if speed > self.max_vel:
                    velocity_jump_score = min(1.0, speed / (self.max_vel * 3))
                elif speed > self.vel_jump_thresh:
                    velocity_jump_score = min(
                        0.5,
                        (speed - self.vel_jump_thresh) /
                        (self.max_vel - self.vel_jump_thresh)
                    )
        scores['velocity_jump'] = velocity_jump_score

        # --- Check 2: GPS vs EKF drift (absolute) ---
        drift = 0.0
        drift_score = 0.0
        if self.last_fused_pose is not None:
            fx, fy = self.last_fused_pose
            drift = math.sqrt((gps_x - fx)**2 + (gps_y - fy)**2)
            if drift > self.drift_thresh:
                drift_score = min(1.0, drift / (self.drift_thresh * 3))
            elif drift > self.drift_thresh * 0.3:
                drift_score = min(
                    0.4,
                    (drift - self.drift_thresh * 0.3) /
                    (self.drift_thresh * 0.7)
                )
        scores['ekf_drift'] = drift_score

        # --- Check 3: Drift TREND (new — key for gradual attacks) ---
        self.drift_history.append(drift)
        if len(self.drift_history) > self.history_window:
            self.drift_history = self.drift_history[-self.history_window:]

        drift_trend_score = 0.0
        if len(self.drift_history) >= self.drift_trend_window:
            recent_drifts = self.drift_history[-self.drift_trend_window:]
            # Linear regression: slope of drift over time
            x_vals = np.arange(len(recent_drifts))
            slope = np.polyfit(x_vals, recent_drifts, 1)[0]

            if slope > self.drift_trend_thresh:
                # Drift is consistently increasing — strong spoofing signal
                drift_trend_score = min(1.0, slope / (self.drift_trend_thresh * 5))
            elif slope > self.drift_trend_thresh * 0.3:
                drift_trend_score = min(
                    0.3,
                    (slope - self.drift_trend_thresh * 0.3) /
                    (self.drift_trend_thresh * 0.7)
                )

            # Also check if drift is monotonically increasing
            if len(recent_drifts) >= 10:
                increasing_count = sum(
                    1 for i in range(1, len(recent_drifts))
                    if recent_drifts[i] > recent_drifts[i-1]
                )
                monotonic_ratio = increasing_count / (len(recent_drifts) - 1)
                if monotonic_ratio > 0.7 and slope > 0:
                    drift_trend_score = max(drift_trend_score, monotonic_ratio * 0.6)

        scores['drift_trend'] = drift_trend_score

        # --- Check 4: Cross-sensor velocity mismatch (new) ---
        velocity_mismatch_score = 0.0
        if self.last_odom_vel is not None and len(self.gps_history) > 0:
            odom_speed = math.sqrt(
                self.last_odom_vel[0]**2 + self.last_odom_vel[1]**2
            )
            gps_speed = math.sqrt(gps_vx**2 + gps_vy**2)

            speed_diff = abs(gps_speed - odom_speed)
            if speed_diff > self.vel_mismatch_thresh:
                velocity_mismatch_score = min(
                    1.0, speed_diff / (self.vel_mismatch_thresh * 5)
                )
            elif speed_diff > self.vel_mismatch_thresh * 0.5:
                velocity_mismatch_score = min(
                    0.3,
                    (speed_diff - self.vel_mismatch_thresh * 0.5) /
                    (self.vel_mismatch_thresh * 0.5)
                )
        scores['velocity_mismatch'] = velocity_mismatch_score

        # --- Check 5: Position consistency (low noise = suspicious) ---
        consistency_score = 0.0
        if len(self.gps_history) >= 5:
            recent = self.gps_history[-5:]
            xs = [p[1] for p in recent]
            ys = [p[2] for p in recent]
            total_var = np.var(xs) + np.var(ys)
            if total_var < 0.001:
                consistency_score = 0.4
        scores['consistency'] = consistency_score

        # --- Combine scores (reweighted for gradual drift sensitivity) ---
        raw_score = min(1.0, sum([
            scores['velocity_jump']      * 0.25,
            scores['ekf_drift']          * 0.20,
            scores['drift_trend']        * 0.25,   # high weight for trend
            scores['velocity_mismatch']  * 0.20,   # new cross-sensor check
            scores['consistency']        * 0.10,
        ]))

        # EMA smoothing
        self.anomaly_score_ema = (
            self.ema_alpha * raw_score +
            (1 - self.ema_alpha) * self.anomaly_score_ema
        )

        # Update history
        self.gps_history.append((now_sec, gps_x, gps_y))
        if len(self.gps_history) > self.history_window:
            self.gps_history.pop(0)

        # Publish score
        score_msg = Float64()
        score_msg.data = self.anomaly_score_ema
        self.score_pub.publish(score_msg)

        # Publish detail
        detail = {
            'anomaly_score': round(self.anomaly_score_ema, 4),
            'raw_score': round(raw_score, 4),
            'velocity_jump': round(scores['velocity_jump'], 4),
            'ekf_drift': round(scores['ekf_drift'], 4),
            'drift_trend': round(scores['drift_trend'], 4),
            'velocity_mismatch': round(scores['velocity_mismatch'], 4),
            'consistency': round(scores['consistency'], 4),
            'drift_m': round(drift, 3),
            'gps_xy': [round(gps_x, 2), round(gps_y, 2)],
            'fused_xy': (
                [round(self.last_fused_pose[0], 2),
                 round(self.last_fused_pose[1], 2)]
                if self.last_fused_pose else None
            )
        }
        detail_msg = String()
        detail_msg.data = json.dumps(detail)
        self.detail_pub.publish(detail_msg)


def main(args=None):
    rclpy.init(args=args)
    node = AnomalyDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
