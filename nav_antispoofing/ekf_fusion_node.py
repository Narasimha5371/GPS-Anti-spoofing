#!/usr/bin/env python3
"""
Extended Kalman Filter Fusion Node.

Fuses GPS, IMU, and odometry into a single best-estimate pose.

State vector: [x, y, vx, vy, yaw, yaw_rate]
Publishes PoseStamped on /fused_pose and a detailed FusedState on /fused_state.
"""

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray
import numpy as np


def quaternion_to_yaw(q):
    """Extract yaw from quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def euler_to_quaternion_msg(yaw):
    """Convert yaw to geometry_msgs Quaternion (roll=pitch=0)."""
    from geometry_msgs.msg import Quaternion
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return Quaternion(x=0.0, y=0.0, z=sy, w=cy)


class EKFFusionNode(Node):
    """
    6-state Extended Kalman Filter for vehicle localization.

    State: [x, y, vx, vy, yaw, yaw_rate]
    """

    def __init__(self):
        super().__init__('ekf_fusion_node')

        # Parameters
        self.declare_parameter('origin_lat', 12.9716)
        self.declare_parameter('origin_lon', 77.5946)
        self.declare_parameter('process_noise_pos', 0.1)
        self.declare_parameter('process_noise_vel', 0.5)
        self.declare_parameter('process_noise_yaw', 0.01)
        self.declare_parameter('gps_noise', 1.0)           # meters
        self.declare_parameter('odom_noise_pos', 0.1)
        self.declare_parameter('odom_noise_vel', 0.2)
        self.declare_parameter('imu_yaw_noise', 0.05)
        self.declare_parameter('publish_rate', 50.0)

        self.origin_lat = self.get_parameter('origin_lat').value
        self.origin_lon = self.get_parameter('origin_lon').value

        # State vector and covariance
        self.x = np.zeros(6)  # [x, y, vx, vy, yaw, yaw_rate]
        self.P = np.eye(6) * 10.0  # initial uncertainty

        # Process noise
        q_pos = self.get_parameter('process_noise_pos').value
        q_vel = self.get_parameter('process_noise_vel').value
        q_yaw = self.get_parameter('process_noise_yaw').value
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel, q_yaw, q_yaw * 0.1])

        # Measurement noise
        gps_n = self.get_parameter('gps_noise').value
        odom_n_p = self.get_parameter('odom_noise_pos').value
        odom_n_v = self.get_parameter('odom_noise_vel').value
        imu_n = self.get_parameter('imu_yaw_noise').value

        self.R_gps = np.diag([gps_n ** 2, gps_n ** 2])
        self.R_odom = np.diag([odom_n_p ** 2, odom_n_p ** 2,
                               odom_n_v ** 2, odom_n_v ** 2])
        self.R_imu = np.array([[imu_n ** 2, 0.0],
                               [0.0, (imu_n * 0.5) ** 2]])

        # Timing
        self.last_predict_time = None

        # Publishers
        rate = self.get_parameter('publish_rate').value
        self.pose_pub = self.create_publisher(PoseStamped, '/fused_pose', 10)
        self.state_pub = self.create_publisher(
            Float64MultiArray, '/fused_state', 10
        )

        # Subscribers
        self.create_subscription(NavSatFix, '/gps/fix', self.gps_callback, 10)
        self.create_subscription(Imu, '/imu/data', self.imu_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # Publish fused pose at high rate
        self.timer = self.create_timer(1.0 / rate, self.publish_state)

        self.get_logger().info('EKF Fusion Node started')

    def _predict(self, dt):
        """Predict step using constant-velocity + yaw-rate model."""
        if dt <= 0 or dt > 1.0:
            return

        x, y, vx, vy, yaw, yr = self.x

        # State transition
        self.x[0] = x + vx * dt
        self.x[1] = y + vy * dt
        self.x[4] = yaw + yr * dt

        # Normalize yaw
        self.x[4] = (self.x[4] + math.pi) % (2 * math.pi) - math.pi

        # Jacobian of state transition
        F = np.eye(6)
        F[0, 2] = dt
        F[1, 3] = dt
        F[4, 5] = dt

        # Update covariance
        self.P = F @ self.P @ F.T + self.Q * dt

    def _update(self, z, H, R):
        """Generic EKF update step."""
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            self.get_logger().warn('Singular innovation matrix — skipping update')
            return
        self.x = self.x + K @ y
        I = np.eye(6)
        self.P = (I - K @ H) @ self.P

        # Normalize yaw
        self.x[4] = (self.x[4] + math.pi) % (2 * math.pi) - math.pi

    def _get_dt(self):
        """Get time delta since last predict."""
        now = self.get_clock().now()
        if self.last_predict_time is None:
            self.last_predict_time = now
            return 0.0
        dt = (now - self.last_predict_time).nanoseconds / 1e9
        self.last_predict_time = now
        return dt

    def gps_callback(self, msg: NavSatFix):
        """GPS measurement update — measures [x, y] in local frame."""
        dt = self._get_dt()
        self._predict(dt)

        # Convert lat/lon to local XY
        x = (msg.longitude - self.origin_lon) * 111320.0 * math.cos(
            math.radians(self.origin_lat)
        )
        y = (msg.latitude - self.origin_lat) * 111320.0

        z = np.array([x, y])
        H = np.zeros((2, 6))
        H[0, 0] = 1.0
        H[1, 1] = 1.0

        self._update(z, H, self.R_gps)

    def imu_callback(self, msg: Imu):
        """IMU measurement update — measures [yaw, yaw_rate]."""
        dt = self._get_dt()
        self._predict(dt)

        yaw = quaternion_to_yaw(msg.orientation)
        yaw_rate = msg.angular_velocity.z

        z = np.array([yaw, yaw_rate])
        H = np.zeros((2, 6))
        H[0, 4] = 1.0
        H[1, 5] = 1.0

        self._update(z, H, self.R_imu)

    def odom_callback(self, msg: Odometry):
        """Odometry measurement update — measures [x, y, vx, vy]."""
        dt = self._get_dt()
        self._predict(dt)

        z = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y
        ])
        H = np.zeros((4, 6))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        H[3, 3] = 1.0

        self._update(z, H, self.R_odom)

    def publish_state(self):
        """Publish fused pose and full state."""
        # PoseStamped
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'odom'
        pose_msg.pose.position.x = float(self.x[0])
        pose_msg.pose.position.y = float(self.x[1])
        pose_msg.pose.position.z = 0.0
        pose_msg.pose.orientation = euler_to_quaternion_msg(float(self.x[4]))
        self.pose_pub.publish(pose_msg)

        # Full state as Float64MultiArray [x, y, vx, vy, yaw, yaw_rate]
        state_msg = Float64MultiArray()
        state_msg.data = self.x.tolist()
        self.state_pub.publish(state_msg)


def main(args=None):
    rclpy.init(args=args)
    node = EKFFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
