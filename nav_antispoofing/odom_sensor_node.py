#!/usr/bin/env python3
"""
Odometry Sensor Node — Simulated wheel odometry publisher.

Publishes Odometry messages at 20 Hz. Simulates wheel encoder data
consistent with circular path motion, with configurable noise.
"""

import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, TransformStamped
import numpy as np

try:
    from tf2_ros import TransformBroadcaster
    HAS_TF2 = True
except ImportError:
    HAS_TF2 = False


def euler_to_quaternion(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return Quaternion(
        x=sr * cp * cy - cr * sp * sy,
        y=cr * sp * cy + sr * cp * sy,
        z=cr * cp * sy - sr * sp * cy,
        w=cr * cp * cy + sr * sp * sy
    )


class OdomSensorNode(Node):
    def __init__(self):
        super().__init__('odom_sensor_node')

        # Parameters
        self.declare_parameter('publish_rate', 20.0)
        self.declare_parameter('odom_noise_std', 0.02)    # meters
        self.declare_parameter('speed_mps', 5.0)
        self.declare_parameter('path_radius', 100.0)
        self.declare_parameter('publish_tf', True)

        rate = self.get_parameter('publish_rate').value
        self.noise_std = self.get_parameter('odom_noise_std').value
        self.speed = self.get_parameter('speed_mps').value
        self.radius = self.get_parameter('path_radius').value
        self.publish_tf = self.get_parameter('publish_tf').value

        self.angular_speed = self.speed / self.radius

        # Publisher
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)

        # TF broadcaster
        self.tf_broadcaster = None
        if self.publish_tf and HAS_TF2:
            self.tf_broadcaster = TransformBroadcaster(self)

        # Internal state — accumulated odometry with drift
        self.drift_x = 0.0
        self.drift_y = 0.0
        self.drift_rate = 0.001  # slow drift in m per update

        self.start_time = self.get_clock().now()
        self.timer = self.create_timer(1.0 / rate, self.publish_odom)

        self.get_logger().info(
            f'Odometry Sensor started — rate={rate} Hz, noise_std={self.noise_std}'
        )

    def publish_odom(self):
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        theta = self.angular_speed * elapsed

        # True position in local frame (meters)
        true_x = self.radius * math.cos(theta)
        true_y = self.radius * math.sin(theta)
        true_yaw = theta + math.pi / 2

        # Accumulate drift
        self.drift_x += np.random.normal(0, self.drift_rate)
        self.drift_y += np.random.normal(0, self.drift_rate)

        # Noisy odometry
        odom_x = true_x + self.drift_x + np.random.normal(0, self.noise_std)
        odom_y = true_y + self.drift_y + np.random.normal(0, self.noise_std)
        odom_yaw = true_yaw + np.random.normal(0, 0.005)

        # Velocities
        vx = -self.radius * self.angular_speed * math.sin(theta)
        vy = self.radius * self.angular_speed * math.cos(theta)

        now = self.get_clock().now().to_msg()

        # Build Odometry message
        msg = Odometry()
        msg.header.stamp = now
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_link'

        msg.pose.pose.position.x = odom_x
        msg.pose.pose.position.y = odom_y
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation = euler_to_quaternion(0, 0, odom_yaw)

        # Pose covariance (6x6 diagonal subset)
        cov = [0.0] * 36
        cov[0] = self.noise_std ** 2
        cov[7] = self.noise_std ** 2
        cov[14] = 0.01
        cov[21] = 0.001
        cov[28] = 0.001
        cov[35] = 0.005
        msg.pose.covariance = cov

        msg.twist.twist.linear.x = vx + np.random.normal(0, 0.05)
        msg.twist.twist.linear.y = vy + np.random.normal(0, 0.05)
        msg.twist.twist.angular.z = self.angular_speed + np.random.normal(0, 0.005)

        twist_cov = [0.0] * 36
        twist_cov[0] = 0.01
        twist_cov[7] = 0.01
        twist_cov[35] = 0.005
        msg.twist.covariance = twist_cov

        self.odom_pub.publish(msg)

        # Broadcast TF: odom -> base_link
        if self.tf_broadcaster:
            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = 'odom'
            t.child_frame_id = 'base_link'
            t.transform.translation.x = odom_x
            t.transform.translation.y = odom_y
            t.transform.translation.z = 0.0
            t.transform.rotation = euler_to_quaternion(0, 0, odom_yaw)
            self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = OdomSensorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
