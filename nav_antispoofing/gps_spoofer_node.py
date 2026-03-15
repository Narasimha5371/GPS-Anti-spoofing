#!/usr/bin/env python3
"""
GPS Spoofer Node — Injects fake GPS signals for testing.

Publishes spoofing offsets to /gps/spoof_offset. Supports three
attack modes:
  1. gradual_drift  — slowly shifts GPS position over time
  2. sudden_jump    — teleports GPS to a fake location
  3. replay         — replays a circular trajectory offset

Activated via ROS2 parameters. Can be triggered with a delay.
"""

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
from std_msgs.msg import String
import json


class GPSSpooferNode(Node):
    def __init__(self):
        super().__init__('gps_spoofer_node')

        # Parameters
        self.declare_parameter('attack_mode', 'gradual_drift')
        self.declare_parameter('activation_delay', 10.0)     # seconds
        self.declare_parameter('attack_duration', 30.0)      # seconds
        self.declare_parameter('drift_rate', 0.0000005)      # deg/sec (~0.05m/s)
        self.declare_parameter('jump_lat_offset', 0.001)     # ~111m
        self.declare_parameter('jump_lon_offset', 0.001)     # ~85m
        self.declare_parameter('replay_radius', 0.0005)      # ~55m
        self.declare_parameter('replay_speed', 0.5)          # rad/s
        self.declare_parameter('publish_rate', 10.0)

        self.attack_mode = self.get_parameter('attack_mode').value
        self.delay = self.get_parameter('activation_delay').value
        self.duration = self.get_parameter('attack_duration').value
        self.drift_rate = self.get_parameter('drift_rate').value
        self.jump_lat = self.get_parameter('jump_lat_offset').value
        self.jump_lon = self.get_parameter('jump_lon_offset').value
        self.replay_r = self.get_parameter('replay_radius').value
        self.replay_spd = self.get_parameter('replay_speed').value
        rate = self.get_parameter('publish_rate').value

        # State
        self.active = False
        self.attack_started = False
        self.start_time = self.get_clock().now()
        self.attack_start_time = None

        # Publishers
        self.offset_pub = self.create_publisher(
            Vector3, '/gps/spoof_offset', 10
        )
        self.status_pub = self.create_publisher(
            String, '/spoofer/status', 10
        )

        self.timer = self.create_timer(1.0 / rate, self.update)

        self.get_logger().info(
            f'GPS Spoofer initialized — mode={self.attack_mode}, '
            f'delay={self.delay}s, duration={self.duration}s'
        )

    def update(self):
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9

        offset = Vector3()
        status = 'WAITING'

        if not self.attack_started and elapsed >= self.delay:
            self.attack_started = True
            self.attack_start_time = self.get_clock().now()
            self.get_logger().warn(
                f'🔴 SPOOFING ATTACK STARTED — mode={self.attack_mode}'
            )

        if self.attack_started:
            attack_elapsed = (
                self.get_clock().now() - self.attack_start_time
            ).nanoseconds / 1e9

            if attack_elapsed > self.duration:
                self.attack_started = False
                status = 'COMPLETED'
                self.get_logger().info('🟢 Spoofing attack completed')
            else:
                self.active = True
                status = 'ACTIVE'

                if self.attack_mode == 'gradual_drift':
                    offset = self._gradual_drift(attack_elapsed)
                elif self.attack_mode == 'sudden_jump':
                    offset = self._sudden_jump(attack_elapsed)
                elif self.attack_mode == 'replay':
                    offset = self._replay_attack(attack_elapsed)
                else:
                    self.get_logger().error(
                        f'Unknown attack mode: {self.attack_mode}'
                    )

        self.offset_pub.publish(offset)

        # Publish status
        status_msg = String()
        status_msg.data = json.dumps({
            'mode': self.attack_mode,
            'status': status,
            'elapsed': round(elapsed, 1),
            'offset_lat': round(offset.x, 8),
            'offset_lon': round(offset.y, 8),
        })
        self.status_pub.publish(status_msg)

    def _gradual_drift(self, t):
        """Slowly shift GPS position — hard to detect."""
        offset = Vector3()
        offset.x = self.drift_rate * t       # latitude drift
        offset.y = self.drift_rate * t * 0.7  # longitude drift (different rate)
        offset.z = 0.0
        return offset

    def _sudden_jump(self, t):
        """Instant teleport to fake position."""
        offset = Vector3()
        # Ramp up quickly in first 0.5 seconds for a sharp jump
        ramp = min(1.0, t / 0.5)
        offset.x = self.jump_lat * ramp
        offset.y = self.jump_lon * ramp
        offset.z = 0.0
        return offset

    def _replay_attack(self, t):
        """Replay a circular offset pattern — sophisticated attack."""
        offset = Vector3()
        theta = self.replay_spd * t
        offset.x = self.replay_r * math.sin(theta)
        offset.y = self.replay_r * math.cos(theta)
        offset.z = 0.0
        return offset


def main(args=None):
    rclpy.init(args=args)
    node = GPSSpooferNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
