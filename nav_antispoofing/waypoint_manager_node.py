#!/usr/bin/env python3
"""
Waypoint Manager Node — Manages navigation waypoints.

Loads a list of GPS waypoints and publishes the current target.
Advances to the next waypoint when the vehicle reaches proximity.
"""

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PointStamped
from std_msgs.msg import Int32, Bool
import json


class WaypointManagerNode(Node):
    def __init__(self):
        super().__init__('waypoint_manager_node')

        # Parameters
        self.declare_parameter('origin_lat', 12.9716)
        self.declare_parameter('origin_lon', 77.5946)
        self.declare_parameter('arrival_radius', 5.0)  # meters
        self.declare_parameter('loop_waypoints', True)
        self.declare_parameter('waypoints_json', json.dumps([
            # Default: square route around origin
            {"lat": 12.9720, "lon": 77.5950},
            {"lat": 12.9720, "lon": 77.5942},
            {"lat": 12.9712, "lon": 77.5942},
            {"lat": 12.9712, "lon": 77.5950},
        ]))

        self.origin_lat = self.get_parameter('origin_lat').value
        self.origin_lon = self.get_parameter('origin_lon').value
        self.arrival_radius = self.get_parameter('arrival_radius').value
        self.loop = self.get_parameter('loop_waypoints').value

        # Parse waypoints
        wp_json = self.get_parameter('waypoints_json').value
        raw_waypoints = json.loads(wp_json)
        self.waypoints = []
        for wp in raw_waypoints:
            x = (wp['lon'] - self.origin_lon) * 111320.0 * math.cos(
                math.radians(self.origin_lat)
            )
            y = (wp['lat'] - self.origin_lat) * 111320.0
            self.waypoints.append((x, y))

        self.current_idx = 0
        self.mission_complete = False

        # Publishers
        self.target_pub = self.create_publisher(
            PointStamped, '/nav/target_waypoint', 10
        )
        self.idx_pub = self.create_publisher(
            Int32, '/nav/waypoint_index', 10
        )
        self.complete_pub = self.create_publisher(
            Bool, '/nav/mission_complete', 10
        )

        # Subscriber
        self.create_subscription(
            PoseStamped, '/fused_pose', self.pose_callback, 10
        )

        # Publish at 2 Hz
        self.timer = self.create_timer(0.5, self.publish_target)

        self.get_logger().info(
            f'Waypoint Manager started — {len(self.waypoints)} waypoints, '
            f'arrival_radius={self.arrival_radius}m'
        )

    def pose_callback(self, msg: PoseStamped):
        if self.mission_complete:
            return

        vx = msg.pose.position.x
        vy = msg.pose.position.y

        if self.current_idx < len(self.waypoints):
            tx, ty = self.waypoints[self.current_idx]
            dist = math.sqrt((vx - tx)**2 + (vy - ty)**2)

            if dist < self.arrival_radius:
                self.get_logger().info(
                    f'Reached waypoint {self.current_idx} '
                    f'(dist={dist:.1f}m) — advancing'
                )
                self.current_idx += 1

                if self.current_idx >= len(self.waypoints):
                    if self.loop:
                        self.current_idx = 0
                        self.get_logger().info('Looping back to waypoint 0')
                    else:
                        self.mission_complete = True
                        self.get_logger().info('🏁 Mission complete!')

    def publish_target(self):
        # Waypoint index
        idx_msg = Int32()
        idx_msg.data = self.current_idx
        self.idx_pub.publish(idx_msg)

        # Mission complete
        comp_msg = Bool()
        comp_msg.data = self.mission_complete
        self.complete_pub.publish(comp_msg)

        # Target waypoint
        if not self.mission_complete and self.current_idx < len(self.waypoints):
            tx, ty = self.waypoints[self.current_idx]
            target_msg = PointStamped()
            target_msg.header.stamp = self.get_clock().now().to_msg()
            target_msg.header.frame_id = 'odom'
            target_msg.point.x = tx
            target_msg.point.y = ty
            target_msg.point.z = 0.0
            self.target_pub.publish(target_msg)


def main(args=None):
    rclpy.init(args=args)
    node = WaypointManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
