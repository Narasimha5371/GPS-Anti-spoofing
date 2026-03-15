# Anti-Spoofing Navigation Control System — Detailed Project Guide

## Overview
This project is a ROS2-based anti-spoofing navigation system for autonomous vehicles. It simulates a car navigating a city using GPS, IMU, and odometry sensors, detects GPS spoofing attacks using both physics-based and machine learning methods, and automatically switches to safe navigation modes when spoofing is detected. The system is fully simulated and includes a 3D web-based city visualizer.

---

## System Architecture

**Key Components:**
- **Sensor Nodes:** Simulate GPS, IMU, and odometry data.
- **EKF Fusion Node:** Fuses sensor data for accurate pose estimation.
- **Anomaly Detector Node:** Detects abnormal sensor behavior (e.g., GPS jumps).
- **ML Spoof Classifier:** Uses an LSTM or heuristic to classify spoofing likelihood.
- **Spoof Alert Manager:** State machine that manages spoofing alerts and transitions.
- **Waypoint Manager Node:** Publishes navigation waypoints.
- **Navigation Controller Node:** Controls vehicle speed and direction, adapts to spoofing alerts.
- **GPS Spoofer Node:** Simulates spoofing attacks for testing.
- **Web Visualizer:** 3D city simulation in the browser, visualizes car, route, and alerts.

---

## Code Structure

- **nav_antispoofing/** — Main ROS2 package with all nodes
  - `gps_sensor_node.py`, `imu_sensor_node.py`, `odom_sensor_node.py`: Simulate sensors
  - `ekf_fusion_node.py`: Extended Kalman Filter for pose
  - `anomaly_detector_node.py`: Physics-based spoof detection
  - `ml_spoof_classifier.py`: ML-based spoof detection
  - `spoof_alert_manager.py`: Alert state machine
  - `waypoint_manager_node.py`: Publishes waypoints
  - `nav_controller_node.py`: Navigation logic (normal, alert, safe stop)
  - `gps_spoofer_node.py`: Simulates spoofing attacks
- **config/nav_params.yaml** — All parameters (sensor noise, waypoints, thresholds)
- **launch/** — Launch files for normal and spoofing test runs
- **scripts/** — Data generation and ML model training
- **simulation/index.html** — 3D web-based city and car visualizer
- **tests/** — Unit tests for core logic

---

## How the Code Works

### 1. Sensor Simulation
- Each sensor node publishes simulated data at a configurable rate and noise level.
- Parameters like `origin_lat`, `origin_lon`, and `path_radius` define the simulated environment.

### 2. Data Fusion
- The EKF node subscribes to all sensor topics and fuses them into a single `/fused_pose` output.

### 3. Spoofing Detection
- The anomaly detector checks for sudden jumps or inconsistencies in sensor data.
- The ML classifier (if enabled) uses a trained LSTM model to predict spoofing probability.
- Both scores are combined in the alert manager, which manages the spoofing alert state.

### 4. Navigation
- The waypoint manager publishes the next target waypoint.
- The navigation controller computes velocity commands to follow waypoints.
- If spoofing is detected, the controller reduces speed or initiates a safe stop.

### 5. Visualization
- The web visualizer connects via rosbridge and roslibjs to ROS2 topics.
- It displays the car, route, and spoofing alerts in a 3D city.

---

## Customizing the System

### Change the Navigation Route
- Edit `waypoints_json` in `config/nav_params.yaml` to set custom GPS waypoints.
- The backend will follow these waypoints; the frontend route is currently static (see below).

### Change Sensor Noise or Rates
- Adjust `gps_noise_std`, `gyro_noise_std`, etc., in `config/nav_params.yaml` for more/less realistic sensors.

### Tune Spoofing Detection
- Change thresholds like `anomaly_detector_node.drift_threshold_m` or `spoof_alert_manager.alert_threshold` in the YAML config.
- Replace the ML model by training a new one with `scripts/train_model.py` and updating the model path in the config.

### Modify Navigation Behavior
- Edit `nav_controller_node.py` to change how the car responds to spoofing (e.g., more aggressive safe stop, different speed reduction).
- Add new logic for obstacle or pedestrian avoidance in this node.

### Change Visualization
- Edit `simulation/index.html` to:
  - Change the city layout or visual route (edit the `WAYPOINTS` array and 3D objects)
  - Add new visual indicators for alerts or status
  - Integrate dynamic route loading from ROS2 (advanced)

### Add New Sensors or Features
- Add new sensor nodes in `nav_antispoofing/` and update the EKF and anomaly detector to use them.
- Extend the state machine in `spoof_alert_manager.py` for more nuanced spoofing responses.

---

## Example: What Happens If You...
- **Change `waypoints_json`**: The car will follow a new route, but the visualizer will still show the default route unless you update it too.
- **Increase `gps_noise_std`**: The car's estimated position will be less accurate, possibly triggering more spoofing alerts.
- **Lower `alert_threshold`**: The system will be more sensitive to spoofing, possibly causing more false positives.
- **Edit `nav_controller_node.py` to stop for obstacles**: The car will slow or stop for obstacles, improving safety.
- **Edit `simulation/index.html` to add new buildings**: The city will look different, but navigation logic is unaffected.

---

## Tips for Extending the Project
- Keep backend (ROS2) logic and frontend (visualizer) logic separate for clarity.
- Use ROS2 topics for all inter-node communication.
- Test changes with `ros2 topic echo` and the web visualizer.
- Use unit tests in `tests/` to validate core logic without running the full simulation.

---

## Further Reading
- See the main `README.md` for quickstart, topic list, and parameter documentation.
- Explore each Python node for detailed logic and comments.
- For advanced integration, consider synchronizing frontend and backend waypoints via a ROS2 service or topic.

---

MIT License
