#!/usr/bin/env python3
"""
IMU Sensor Node — Simulated IMU publisher.

Publishes Imu messages at 50 Hz with realistic noise characteristics.
Simulates angular velocity and linear acceleration consistent with
the circular path motion of the GPS sensor.
"""

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Quaternion
import numpy as np


def euler_to_quaternion(roll, pitch, yaw):
    """Convert Euler angles to quaternion."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)

    return Quaternion(
        x=sr * cp * cy - cr * sp * sy,
        y=cr * sp * cy + sr * cp * sy,
        z=cr * cp * sy - sr * sp * cy,
        w=cr * cp * cy + sr * sp * sy
    )


class IMUSensorNode(Node):
    def __init__(self):
        super().__init__('imu_sensor_node')

        # Parameters
        self.declare_parameter('publish_rate', 50.0)
        self.declare_parameter('gyro_noise_std', 0.01)      # rad/s
        self.declare_parameter('accel_noise_std', 0.05)      # m/s^2
        self.declare_parameter('gyro_bias', 0.001)           # rad/s
        self.declare_parameter('speed_mps', 5.0)
        self.declare_parameter('path_radius', 100.0)

        rate = self.get_parameter('publish_rate').value
        self.gyro_noise = self.get_parameter('gyro_noise_std').value
        self.accel_noise = self.get_parameter('accel_noise_std').value
        self.gyro_bias = self.get_parameter('gyro_bias').value
        self.speed = self.get_parameter('speed_mps').value
        self.radius = self.get_parameter('path_radius').value

        # Derived
        self.angular_speed = self.speed / self.radius  # rad/s
        self.centripetal_accel = self.speed ** 2 / self.radius  # m/s^2

        # Publisher
        self.imu_pub = self.create_publisher(Imu, '/imu/data', 10)

        # Internal state
        self.start_time = self.get_clock().now()
        self.timer = self.create_timer(1.0 / rate, self.publish_imu)

        self.get_logger().info(
            f'IMU Sensor started — rate={rate} Hz, '
            f'gyro_noise={self.gyro_noise}, accel_noise={self.accel_noise}'
        )

    def publish_imu(self):
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        theta = self.angular_speed * elapsed

        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'imu_link'

        # Orientation (yaw changes as vehicle goes around circle)
        yaw = theta + math.pi / 2  # tangent to circle
        msg.orientation = euler_to_quaternion(0.0, 0.0, yaw)
        msg.orientation_covariance = [
            0.001, 0.0, 0.0,
            0.0, 0.001, 0.0,
            0.0, 0.0, 0.005
        ]

        # Angular velocity (constant yaw rate for circular motion)
        msg.angular_velocity.x = 0.0 + np.random.normal(0, self.gyro_noise)
        msg.angular_velocity.y = 0.0 + np.random.normal(0, self.gyro_noise)
        msg.angular_velocity.z = (
            self.angular_speed + self.gyro_bias +
            np.random.normal(0, self.gyro_noise)
        )
        msg.angular_velocity_covariance = [
            self.gyro_noise**2, 0.0, 0.0,
            0.0, self.gyro_noise**2, 0.0,
            0.0, 0.0, self.gyro_noise**2
        ]

        # Linear acceleration
        # Centripetal acceleration pointing toward center + gravity on z
        ax = -self.centripetal_accel * math.cos(theta) + np.random.normal(0, self.accel_noise)
        ay = -self.centripetal_accel * math.sin(theta) + np.random.normal(0, self.accel_noise)
        az = 9.81 + np.random.normal(0, self.accel_noise)

        msg.linear_acceleration.x = ax
        msg.linear_acceleration.y = ay
        msg.linear_acceleration.z = az
        msg.linear_acceleration_covariance = [
            self.accel_noise**2, 0.0, 0.0,
            0.0, self.accel_noise**2, 0.0,
            0.0, 0.0, self.accel_noise**2
        ]

        self.imu_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = IMUSensorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
