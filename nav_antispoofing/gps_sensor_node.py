#!/usr/bin/env python3
"""
GPS Sensor Node — Simulated GPS publisher.

Publishes NavSatFix messages at 10 Hz simulating a vehicle moving
along a predefined path. Supports external spoofing injection via
the /gps/spoof_offset topic.
"""

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, NavSatStatus
from geometry_msgs.msg import Vector3
import numpy as np


class GPSSensorNode(Node):
    def __init__(self):
        super().__init__('gps_sensor_node')

        # Parameters
        self.declare_parameter('publish_rate', 10.0)
        self.declare_parameter('origin_lat', 12.9716)    # Bangalore, India
        self.declare_parameter('origin_lon', 77.5946)
        self.declare_parameter('gps_noise_std', 0.000002)  # ~0.2m in lat/lon
        self.declare_parameter('speed_mps', 5.0)           # 5 m/s forward speed
        self.declare_parameter('path_radius', 100.0)       # circular path radius (m)

        rate = self.get_parameter('publish_rate').value
        self.origin_lat = self.get_parameter('origin_lat').value
        self.origin_lon = self.get_parameter('origin_lon').value
        self.noise_std = self.get_parameter('gps_noise_std').value
        self.speed = self.get_parameter('speed_mps').value
        self.radius = self.get_parameter('path_radius').value

        # Publishers
        self.gps_pub = self.create_publisher(NavSatFix, '/gps/fix', 10)

        # Subscriber for spoofing offset injection
        self.spoof_offset = np.array([0.0, 0.0, 0.0])
        self.create_subscription(
            Vector3, '/gps/spoof_offset', self._spoof_offset_cb, 10
        )

        # Internal state
        self.start_time = self.get_clock().now()
        self.timer = self.create_timer(1.0 / rate, self.publish_gps)

        self.get_logger().info(
            f'GPS Sensor started — origin=({self.origin_lat}, {self.origin_lon}), '
            f'rate={rate} Hz, noise_std={self.noise_std}'
        )

    def _spoof_offset_cb(self, msg: Vector3):
        self.spoof_offset = np.array([msg.x, msg.y, msg.z])

    def _get_true_position(self, elapsed: float):
        """Simulate circular path motion."""
        angular_speed = self.speed / self.radius
        theta = angular_speed * elapsed

        # Position in meters from origin
        x = self.radius * math.cos(theta)
        y = self.radius * math.sin(theta)

        # Convert meters to lat/lon offset (rough approximation)
        # 1 degree lat ≈ 111,320 m, 1 degree lon ≈ 111,320 * cos(lat) m
        lat_offset = y / 111320.0
        lon_offset = x / (111320.0 * math.cos(math.radians(self.origin_lat)))

        return (
            self.origin_lat + lat_offset,
            self.origin_lon + lon_offset,
            0.0  # altitude
        )

    def publish_gps(self):
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9

        true_lat, true_lon, true_alt = self._get_true_position(elapsed)

        # Add noise + any spoofing offset
        noisy_lat = true_lat + np.random.normal(0, self.noise_std) + self.spoof_offset[0]
        noisy_lon = true_lon + np.random.normal(0, self.noise_std) + self.spoof_offset[1]
        noisy_alt = true_alt + np.random.normal(0, 0.5) + self.spoof_offset[2]

        msg = NavSatFix()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'gps_link'

        msg.status.status = NavSatStatus.STATUS_FIX
        msg.status.service = NavSatStatus.SERVICE_GPS

        msg.latitude = noisy_lat
        msg.longitude = noisy_lon
        msg.altitude = noisy_alt

        # Position covariance (diagonal, in m^2)
        msg.position_covariance = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 4.0
        ]
        msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN

        self.gps_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = GPSSensorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
