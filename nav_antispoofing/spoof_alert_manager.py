#!/usr/bin/env python3
"""
Spoof Alert Manager — Fuses anomaly + ML detection into alerts.

Subscribes to:
  /spoofing/anomaly_score  (Float64)  — physics-based anomaly score
  /spoofing/detection      (Float64)  — ML classifier probability

Publishes:
  /spoofing/alert   (Bool)    — True when spoofing is detected
  /spoofing/status  (String)  — JSON with detailed status

Implements alert escalation, cooldown, and hysteresis to prevent
false positives and flapping.
"""

import json
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float64, String


class SpoofAlertManager(Node):
    """
    Alert states:
      CLEAR     — no spoofing detected
      SUSPECT   — anomaly detected, waiting for confirmation
      ALERT     — spoofing confirmed
      COOLDOWN  — recently cleared, suppressing re-trigger
    """

    CLEAR = 'CLEAR'
    SUSPECT = 'SUSPECT'
    ALERT = 'ALERT'
    COOLDOWN = 'COOLDOWN'

    def __init__(self):
        super().__init__('spoof_alert_manager')

        # Parameters
        self.declare_parameter('anomaly_weight', 0.4)
        self.declare_parameter('ml_weight', 0.6)
        self.declare_parameter('suspect_threshold', 0.35)
        self.declare_parameter('alert_threshold', 0.55)
        self.declare_parameter('clear_threshold', 0.15)
        self.declare_parameter('suspect_confirm_sec', 2.0)
        self.declare_parameter('cooldown_sec', 5.0)
        self.declare_parameter('publish_rate', 10.0)

        self.w_anomaly = self.get_parameter('anomaly_weight').value
        self.w_ml = self.get_parameter('ml_weight').value
        self.thresh_suspect = self.get_parameter('suspect_threshold').value
        self.thresh_alert = self.get_parameter('alert_threshold').value
        self.thresh_clear = self.get_parameter('clear_threshold').value
        self.confirm_time = self.get_parameter('suspect_confirm_sec').value
        self.cooldown_time = self.get_parameter('cooldown_sec').value
        rate = self.get_parameter('publish_rate').value

        # State
        self.state = self.CLEAR
        self.suspect_start = None
        self.cooldown_start = None
        self.latest_anomaly = 0.0
        self.latest_ml = 0.0
        self.combined_score = 0.0
        self.alert_count = 0

        # Publishers
        self.alert_pub = self.create_publisher(Bool, '/spoofing/alert', 10)
        self.status_pub = self.create_publisher(String, '/spoofing/status', 10)

        # Subscribers
        self.create_subscription(
            Float64, '/spoofing/anomaly_score', self._anomaly_cb, 10
        )
        self.create_subscription(
            Float64, '/spoofing/detection', self._ml_cb, 10
        )

        # Main loop
        self.timer = self.create_timer(1.0 / rate, self.update)

        self.get_logger().info(
            f'Spoof Alert Manager started — '
            f'suspect>{self.thresh_suspect}, alert>{self.thresh_alert}'
        )

    def _anomaly_cb(self, msg: Float64):
        self.latest_anomaly = msg.data

    def _ml_cb(self, msg: Float64):
        self.latest_ml = msg.data

    def update(self):
        """State machine update."""
        now = time.time()

        # Fused score
        self.combined_score = (
            self.w_anomaly * self.latest_anomaly +
            self.w_ml * self.latest_ml
        )

        prev_state = self.state

        # ── State transitions ──
        if self.state == self.CLEAR:
            if self.combined_score >= self.thresh_alert:
                self.state = self.ALERT
                self.alert_count += 1
                self.get_logger().warn(
                    f'⚠️  SPOOFING ALERT #{self.alert_count} — '
                    f'score={self.combined_score:.3f}'
                )
            elif self.combined_score >= self.thresh_suspect:
                self.state = self.SUSPECT
                self.suspect_start = now

        elif self.state == self.SUSPECT:
            if self.combined_score >= self.thresh_alert:
                self.state = self.ALERT
                self.alert_count += 1
                self.get_logger().warn(
                    f'⚠️  SPOOFING ALERT #{self.alert_count} — '
                    f'score={self.combined_score:.3f}'
                )
            elif self.combined_score < self.thresh_suspect:
                self.state = self.CLEAR
                self.suspect_start = None
            elif (now - self.suspect_start) > self.confirm_time:
                # Sustained suspicion → escalate
                self.state = self.ALERT
                self.alert_count += 1
                self.get_logger().warn(
                    f'⚠️  SPOOFING ALERT #{self.alert_count} (escalated) — '
                    f'score={self.combined_score:.3f}'
                )

        elif self.state == self.ALERT:
            if self.combined_score < self.thresh_clear:
                self.state = self.COOLDOWN
                self.cooldown_start = now
                self.get_logger().info(
                    '✅ Spoofing alert cleared — entering cooldown'
                )

        elif self.state == self.COOLDOWN:
            if self.combined_score >= self.thresh_suspect:
                self.state = self.ALERT
                self.alert_count += 1
                self.get_logger().warn(
                    f'⚠️  SPOOFING RE-ALERT #{self.alert_count} — '
                    f'score={self.combined_score:.3f}'
                )
            elif (now - self.cooldown_start) > self.cooldown_time:
                self.state = self.CLEAR
                self.cooldown_start = None

        # Publish alert
        alert_msg = Bool()
        alert_msg.data = (self.state == self.ALERT)
        self.alert_pub.publish(alert_msg)

        # Publish status
        status = {
            'state': self.state,
            'combined_score': round(self.combined_score, 4),
            'anomaly_score': round(self.latest_anomaly, 4),
            'ml_score': round(self.latest_ml, 4),
            'alert_count': self.alert_count,
        }
        status_msg = String()
        status_msg.data = json.dumps(status)
        self.status_pub.publish(status_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SpoofAlertManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
