#!/usr/bin/env python3
"""
Navigation Controller Node — Spoof-aware vehicle motion controller.

Operates in two modes:
  NORMAL — follows waypoints using fused pose, publishes cmd_vel
  ALERT  — on spoof detection, reduces speed, ignores GPS, initiates safe stop

Uses a simple PID controller for heading and speed control.
"""

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PointStamped, Twist, Quaternion
from std_msgs.msg import Bool, String
import json


def quaternion_to_yaw(q: Quaternion):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle):
    """Normalize angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


class NavControllerNode(Node):
    """
    PID-based navigation controller with spoofing awareness.
    """

    NORMAL = 'NORMAL'
    ALERT = 'ALERT'
    SAFE_STOP = 'SAFE_STOP'

    def __init__(self):
        super().__init__('nav_controller_node')

        # Parameters
        self.declare_parameter('max_linear_speed', 5.0)           # m/s
        self.declare_parameter('max_angular_speed', 1.0)          # rad/s
        self.declare_parameter('alert_speed_factor', 0.2)         # reduce to 20%
        self.declare_parameter('safe_stop_decel', 1.0)            # m/s^2
        self.declare_parameter('pid_linear_kp', 0.5)
        self.declare_parameter('pid_linear_ki', 0.01)
        self.declare_parameter('pid_linear_kd', 0.05)
        self.declare_parameter('pid_angular_kp', 2.0)
        self.declare_parameter('pid_angular_ki', 0.0)
        self.declare_parameter('pid_angular_kd', 0.1)
        self.declare_parameter('control_rate', 20.0)
        self.declare_parameter('safe_stop_timeout', 5.0)          # seconds

        self.max_lin = self.get_parameter('max_linear_speed').value
        self.max_ang = self.get_parameter('max_angular_speed').value
        self.alert_factor = self.get_parameter('alert_speed_factor').value
        self.decel = self.get_parameter('safe_stop_decel').value
        self.stop_timeout = self.get_parameter('safe_stop_timeout').value
        rate = self.get_parameter('control_rate').value

        # PID gains
        self.kp_lin = self.get_parameter('pid_linear_kp').value
        self.ki_lin = self.get_parameter('pid_linear_ki').value
        self.kd_lin = self.get_parameter('pid_linear_kd').value
        self.kp_ang = self.get_parameter('pid_angular_kp').value
        self.ki_ang = self.get_parameter('pid_angular_ki').value
        self.kd_ang = self.get_parameter('pid_angular_kd').value

        # PID state
        self.lin_integral = 0.0
        self.lin_prev_err = 0.0
        self.ang_integral = 0.0
        self.ang_prev_err = 0.0


        # Navigation state
        self.mode = self.NORMAL
        self.current_pose = None   # (x, y, yaw)
        self.target_point = None   # (x, y)
        self.spoof_alert = False
        self.current_speed = 0.0
        self.safe_stop_start = None

        # Other cars (simulated obstacles)
        self.other_cars = []  # List of (x, y, speed)

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.mode_pub = self.create_publisher(String, '/nav/mode', 10)

        # Subscribers
        self.create_subscription(
            PoseStamped, '/fused_pose', self.pose_callback, 10
        )
        self.create_subscription(
            PointStamped, '/nav/target_waypoint', self.target_callback, 10
        )
        self.create_subscription(
            Bool, '/spoofing/alert', self.alert_callback, 10
        )


        # Subscribe to other cars (simulated obstacles)
        from geometry_msgs.msg import PoseArray
        self.create_subscription(
            PoseArray, '/other_cars', self.other_cars_callback, 10
        )

        # Control loop
        self.dt = 1.0 / rate
        self.timer = self.create_timer(self.dt, self.control_loop)
    def other_cars_callback(self, msg):
        # Update list of other cars: [(x, y, speed), ...]
        self.other_cars = []
        for pose in msg.poses:
            # If speed is not available, set to 0
            self.other_cars.append((pose.position.x, pose.position.y, 0.0))

        self.get_logger().info(
            f'Navigation Controller started — '
            f'max_speed={self.max_lin} m/s, rate={rate} Hz'
        )

    def pose_callback(self, msg: PoseStamped):
        yaw = quaternion_to_yaw(msg.pose.orientation)
        self.current_pose = (
            msg.pose.position.x,
            msg.pose.position.y,
            yaw
        )

    def target_callback(self, msg: PointStamped):
        self.target_point = (msg.point.x, msg.point.y)

    def alert_callback(self, msg: Bool):
        was_alert = self.spoof_alert
        self.spoof_alert = msg.data

        if msg.data and not was_alert:
            self.get_logger().warn('🚨 Spoofing detected — entering ALERT mode')
            self.mode = self.ALERT
            self.safe_stop_start = None
        elif not msg.data and was_alert:
            self.get_logger().info('✅ Spoofing cleared — resuming NORMAL mode')
            self.mode = self.NORMAL
            self._reset_pid()

    def _reset_pid(self):
        self.lin_integral = 0.0
        self.lin_prev_err = 0.0
        self.ang_integral = 0.0
        self.ang_prev_err = 0.0

    def control_loop(self):
        """Main control loop."""
        cmd = Twist()

        if self.mode == self.NORMAL:
            cmd = self._normal_control()
        elif self.mode == self.ALERT:
            cmd = self._alert_control()
        elif self.mode == self.SAFE_STOP:
            cmd = self._safe_stop_control()

        self.cmd_pub.publish(cmd)

        # Publish mode
        mode_msg = String()
        mode_msg.data = json.dumps({
            'mode': self.mode,
            'spoofing': self.spoof_alert,
            'speed': round(self.current_speed, 2)
        })
        self.mode_pub.publish(mode_msg)

    def _normal_control(self):
        """Normal waypoint-following control with backend collision avoidance."""
        cmd = Twist()

        if self.current_pose is None or self.target_point is None:
            return cmd

        px, py, yaw = self.current_pose
        tx, ty = self.target_point

        # Distance and heading to target
        dx = tx - px
        dy = ty - py
        dist = math.sqrt(dx**2 + dy**2)
        target_yaw = math.atan2(dy, dx)
        heading_err = normalize_angle(target_yaw - yaw)

        # Linear PID
        self.lin_integral += dist * self.dt
        self.lin_integral = max(-10, min(10, self.lin_integral))
        lin_deriv = (dist - self.lin_prev_err) / self.dt
        self.lin_prev_err = dist

        linear_speed = (
            self.kp_lin * dist +
            self.ki_lin * self.lin_integral +
            self.kd_lin * lin_deriv
        )
        linear_speed = max(0, min(self.max_lin, linear_speed))

        # Reduce speed when not aligned
        alignment = max(0, math.cos(heading_err))
        linear_speed *= alignment

        # --- Backend collision avoidance: slow/stop and steer if car ahead ---
        steering_offset = 0.0
        avoidance_active = False
        
        for ox, oy, ospeed in self.other_cars:
            odx = ox - px
            ody = oy - py
            odist = math.sqrt(odx**2 + ody**2)
            angle_to_car = math.atan2(ody, odx)
            angle_diff = normalize_angle(angle_to_car - yaw)
            
            # If obstacle is ahead (within 12m and within 45 degrees in front)
            if odist < 12.0 and abs(angle_diff) < math.pi / 4:
                # Slow down based on distance
                speed_scale = max(0.0, (odist - 3.0) / 9.0) # 0 at 3m, 1 at 12m
                linear_speed = min(linear_speed, self.max_lin * speed_scale)
                
                # Steer away if too close (within 8m)
                if odist < 8.0:
                    avoidance_active = True
                    # Steer away from obstacle center
                    side = -1.0 if angle_diff > 0 else 1.0
                    # Force increases as we get closer
                    force = (8.0 - odist) / 5.0 * (math.pi / 3) # Up to 60 deg offset
                    steering_offset += side * force

        target_yaw = math.atan2(dy, dx) + steering_offset
        heading_err = normalize_angle(target_yaw - yaw)

        # Angular PID
        self.ang_integral += heading_err * self.dt
        self.ang_integral = max(-5, min(5, self.ang_integral))
        ang_deriv = (heading_err - self.ang_prev_err) / self.dt
        self.ang_prev_err = heading_err

        angular_speed = (
            self.kp_ang * heading_err +
            self.ki_ang * self.ang_integral +
            self.kd_ang * ang_deriv
        )
        angular_speed = max(-self.max_ang, min(self.max_ang, angular_speed))

        cmd.linear.x = linear_speed
        cmd.angular.z = angular_speed
        self.current_speed = linear_speed

        return cmd

    def _alert_control(self):
        """
        Alert mode — GPS is unreliable.
        Reduce speed and prepare for safe stop.
        """
        cmd = Twist()

        # Gradually reduce speed
        target_speed = self.max_lin * self.alert_factor
        if self.current_speed > target_speed:
            self.current_speed = max(
                target_speed,
                self.current_speed - self.decel * self.dt
            )

        cmd.linear.x = self.current_speed
        cmd.angular.z = 0.0  # maintain current heading

        # Transition to safe stop after timeout
        if self.safe_stop_start is None:
            self.safe_stop_start = self.get_clock().now()
        else:
            elapsed = (
                self.get_clock().now() - self.safe_stop_start
            ).nanoseconds / 1e9
            if elapsed > self.stop_timeout:
                self.mode = self.SAFE_STOP
                self.get_logger().warn(
                    '🛑 Spoofing persists — initiating SAFE STOP'
                )

        return cmd

    def _safe_stop_control(self):
        """Safe stop — bring vehicle to complete halt."""
        cmd = Twist()

        self.current_speed = max(
            0.0,
            self.current_speed - self.decel * self.dt
        )
        cmd.linear.x = self.current_speed
        cmd.angular.z = 0.0

        if self.current_speed <= 0.01:
            self.current_speed = 0.0
            if not self.spoof_alert:
                self.mode = self.NORMAL
                self.get_logger().info('Resumed NORMAL after safe stop')
                self._reset_pid()

        return cmd


def main(args=None):
    rclpy.init(args=args)
    node = NavControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
