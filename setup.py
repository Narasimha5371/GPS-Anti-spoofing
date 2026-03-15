from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'nav_antispoofing'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='developer',
    maintainer_email='dev@nav.local',
    description='Anti-Spoofing Navigation Control System for Autonomous Cars',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gps_sensor = nav_antispoofing.gps_sensor_node:main',
            'imu_sensor = nav_antispoofing.imu_sensor_node:main',
            'odom_sensor = nav_antispoofing.odom_sensor_node:main',
            'ekf_fusion = nav_antispoofing.ekf_fusion_node:main',
            'anomaly_detector = nav_antispoofing.anomaly_detector_node:main',
            'ml_classifier = nav_antispoofing.ml_spoof_classifier:main',
            'spoof_alert = nav_antispoofing.spoof_alert_manager:main',
            'waypoint_manager = nav_antispoofing.waypoint_manager_node:main',
            'nav_controller = nav_antispoofing.nav_controller_node:main',
            'gps_spoofer = nav_antispoofing.gps_spoofer_node:main',
        ],
    },
)
