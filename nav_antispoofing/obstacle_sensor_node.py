#!/usr/bin/env python3
"""
Obstacle Sensor Node — Simulates virtual obstacles on the road.
Publishes a PoseArray of obstacle positions.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
import math

class ObstacleSensorNode(Node):
    def __init__(self):
        super().__init__('obstacle_sensor_node')

        # Parameters
        self.declare_parameter('publish_rate', 10.0)
        # List of obstacles: [[x, y], [x, y], ...]
        # We use a JSON string or simpler list if possible. 
        # For now, let's hardcode some on the path or use a list of values.
        self.declare_parameter('obstacle_positions', [
            -100.0, -144.0,  # obstacle 1
            -27.0, -100.0,   # obstacle 2
             30.0, 30.0      # obstacle 3
        ])

        rate = self.get_parameter('publish_rate').value
        self.obs_raw = self.get_parameter('obstacle_positions').value
        
        self.obstacles = []
        for i in range(0, len(self.obs_raw), 2):
            if i + 1 < len(self.obs_raw):
                self.obstacles.append((self.obs_raw[i], self.obs_raw[i+1]))

        # Publisher
        self.obs_pub = self.create_publisher(PoseArray, '/other_cars', 10)

        # Timer
        self.timer = self.create_timer(1.0 / rate, self.publish_obstacles)

        self.get_logger().info(f'Obstacle Sensor started — publishing {len(self.obstacles)} obstacles at {rate} Hz')

    def publish_obstacles(self):
        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'

        for x, y in self.obstacles:
            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = 0.0
            # No rotation for now
            msg.poses.append(pose)

        self.obs_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ObstacleSensorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
