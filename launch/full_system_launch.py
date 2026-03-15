#!/usr/bin/env python3
"""Launch file for the full anti-spoofing navigation system."""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Config file
    pkg_dir = get_package_share_directory('nav_antispoofing')
    config_file = os.path.join(pkg_dir, 'config', 'nav_params.yaml')

    return LaunchDescription([
        # ── Sensor Nodes ──
        Node(
            package='nav_antispoofing',
            executable='gps_sensor',
            name='gps_sensor_node',
            parameters=[config_file],
            output='screen',
        ),
        Node(
            package='nav_antispoofing',
            executable='imu_sensor',
            name='imu_sensor_node',
            parameters=[config_file],
            output='screen',
        ),
        Node(
            package='nav_antispoofing',
            executable='odom_sensor',
            name='odom_sensor_node',
            parameters=[config_file],
            output='screen',
        ),

        # ── Fusion ──
        Node(
            package='nav_antispoofing',
            executable='ekf_fusion',
            name='ekf_fusion_node',
            parameters=[config_file],
            output='screen',
        ),

        # ── Anti-Spoofing Detection ──
        Node(
            package='nav_antispoofing',
            executable='anomaly_detector',
            name='anomaly_detector_node',
            parameters=[config_file],
            output='screen',
        ),
        Node(
            package='nav_antispoofing',
            executable='ml_classifier',
            name='ml_spoof_classifier',
            parameters=[config_file],
            output='screen',
        ),
        Node(
            package='nav_antispoofing',
            executable='spoof_alert',
            name='spoof_alert_manager',
            parameters=[config_file],
            output='screen',
        ),

        # ── Navigation ──
        Node(
            package='nav_antispoofing',
            executable='waypoint_manager',
            name='waypoint_manager_node',
            parameters=[config_file],
            output='screen',
        ),
        Node(
            package='nav_antispoofing',
            executable='nav_controller',
            name='nav_controller_node',
            parameters=[config_file],
            output='screen',
        ),
    ])
