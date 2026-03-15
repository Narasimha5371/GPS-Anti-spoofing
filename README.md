# Anti-Spoofing Navigation System

## Overview

This project implements and simulates an anti-spoofing navigation system for autonomous vehicles. It features a ROS2-based backend for sensor fusion, anomaly detection, and spoofing response, along with a 3D city simulation frontend for visualization and testing.

---

## Features

- **Sensor Fusion:** Combines GPS, IMU, and odometry data using an Extended Kalman Filter (EKF).
- **Spoofing Detection:** Detects GPS spoofing attacks using anomaly detection and machine learning classifiers.
- **Attack Simulation:** Supports gradual drift, sudden jump, and replay GPS spoofing attacks.
- **3D Visualization:** Interactive city simulation with real-time vehicle tracking and route display.
- **Modular ROS2 Nodes:** Each function (sensors, fusion, detection, control) is a separate ROS2 node for flexibility and testing.

---

## Project Structure

```
nav/
├── config/
│   └── nav_params.yaml         # Configuration for sensors and navigation
├── launch/
│   ├── full_system_launch.py   # Launches all system nodes (no attack)
│   └── test_spoofing_launch.py # Launches system with GPS spoofer
├── nav_antispoofing/
│   ├── anomaly_detector_node.py
│   ├── ekf_fusion_node.py
│   ├── gps_sensor_node.py
│   ├── gps_spoofer_node.py
│   ├── imu_sensor_node.py
│   ├── ml_spoof_classifier.py
│   ├── nav_controller_node.py
│   ├── odom_sensor_node.py
│   ├── spoof_alert_manager.py
│   └── waypoint_manager_node.py
├── resource/
├── scripts/
│   ├── evaluate_performance.py
│   ├── generate_training_data.py
│   └── train_model.py
├── simulation/
│   └── index.html              # 3D city simulation frontend
├── tests/
│   ├── test_anomaly_detector.py
│   ├── test_ekf.py
│   ├── test_ml_classifier.py
│   └── test_nav_controller.py
├── performance_results.txt     # Performance report
├── README.md                   # This file
└── ...
```

---

## Getting Started

### Prerequisites

- ROS2 Humble
- Python 3.10+
- Node.js (for running a local web server, optional)
- [rosbridge_server](http://wiki.ros.org/rosbridge_suite) (for WebSocket communication)
- Modern web browser (Chrome, Firefox, Edge)

### Installation

1. **Clone the repository:**
        ```
        git clone <repo-url>
        cd nav
        ```

2. **Install Python dependencies:**
        ```
        pip install -r requirements.txt
        ```

3. **Build ROS2 packages:**
        ```
        colcon build
        source install/setup.bash
        ```

4. **(Optional) Start rosbridge_server:**
        ```
        ros2 launch rosbridge_server rosbridge_websocket_launch.xml
        ```

---

## Running the Simulation

### 1. **Start the Backend**

- **Full System (no attack):**
       ```
       ros2 launch launch/full_system_launch.py
       ```

- **With GPS Spoofing Attack:**
       ```
       ros2 launch launch/test_spoofing_launch.py attack_mode:=gradual_drift activation_delay:=10.0
       ```

### 2. **Start rosbridge_server**

- In a new terminal:
       ```
       ros2 launch rosbridge_server rosbridge_websocket_launch.xml
       ```

### 3. **Open the Frontend**

- Open `simulation/index.html` in your browser.
- For best results, use a local web server (e.g., `python -m http.server` in the simulation directory).

---

## Customization

### Changing Waypoints

- Edit the `WAYPOINTS` array in `simulation/index.html` for frontend visualization.
- For backend navigation, update the `waypoints_json` parameter in `waypoint_manager_node.py` or set it in your launch/config files.

### Adding New Attacks

- Implement new attack logic in `gps_spoofer_node.py`.
- Update the frontend and launch files as needed.

---

## Testing

- Unit tests are in the `tests/` directory.
- Run with:
       ```
       pytest tests/
       ```

---

## Performance

See `performance_results.txt` for detailed results of spoofing detection and navigation accuracy under various attack scenarios.

---

## License

[MIT License](LICENSE)

---

## Acknowledgments

- ROS2 Community
- Three.js for 3D visualization
- Contributors to open-source anti-spoofing research

---

## Contact

For questions or contributions, please open an issue or contact the maintainer.
# 🛡️ Anti-Spoofing Navigation Control System

A ROS2-based anti-spoofing navigation system for autonomous cars that detects GPS spoofing attacks using multi-sensor fusion and machine learning, and automatically falls back to safe navigation when spoofing is detected.

## Architecture

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  GPS Sensor  │  │  IMU Sensor  │  │  Odometry   │
│  (10 Hz)     │  │  (50 Hz)     │  │  (20 Hz)    │
└──────┬───────┘  └──────┬───────┘  └──────┬──────┘
       │                 │                 │
       └─────────┬───────┴────────┬────────┘
                 │                │
          ┌──────▼──────┐  ┌─────▼────────────┐
          │  EKF Fusion  │  │ Anomaly Detector  │
          │  (6-state)   │  │ (physics-based)   │
          └──────┬───────┘  └─────┬────────────┘
                 │                │
                 │         ┌──────▼──────────┐
                 │         │ ML Classifier    │
                 │         │ (LSTM / heuristic)│
                 │         └──────┬───────────┘
                 │                │
                 │         ┌──────▼──────────┐
                 │         │ Alert Manager    │
                 │         │ (state machine)  │
                 │         └──────┬───────────┘
                 │                │
          ┌──────▼────────────────▼──────┐
          │    Navigation Controller      │
          │  NORMAL → ALERT → SAFE_STOP  │
          └──────────────┬───────────────┘
                         │
                    ┌────▼────┐
                    │ cmd_vel │
                    └─────────┘
```

## Features

- **Extended Kalman Filter** — 6-state EKF fusing GPS, IMU, and odometry
- **Physics-Based Anomaly Detection** — velocity jumps, GPS-EKF drift, consistency checks
- **ML Spoofing Classifier** — LSTM neural network with heuristic fallback
- **Alert State Machine** — CLEAR → SUSPECT → ALERT → COOLDOWN with hysteresis
- **Spoof-Aware Navigation** — PID controller with NORMAL / ALERT / SAFE_STOP modes
- **GPS Spoofer** — Test tool with gradual drift, sudden jump, and replay attacks
- **Fully Simulated** — No hardware required; all sensors are simulated

## Prerequisites

- **Ubuntu 22.04** with **ROS2 Humble**
- **Python 3.10+**
- **NumPy** (`pip install numpy`)
- **PyTorch** (optional, for ML model training: `pip install torch`)
- **pytest** (for testing: `pip install pytest`)

## Setup

### 1. Create a ROS2 Workspace

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
ln -s /path/to/nav nav_antispoofing
```

### 2. Build

```bash
cd ~/ros2_ws
colcon build --packages-select nav_antispoofing
source install/setup.bash
```

### 3. Run the System

**Normal operation (no spoofing):**
```bash
ros2 launch nav_antispoofing full_system_launch.py
```

**With spoofing test:**
```bash
ros2 launch nav_antispoofing test_spoofing_launch.py attack_mode:=sudden_jump
```

Attack modes: `gradual_drift`, `sudden_jump`, `replay`

### 4. Monitor Topics

```bash
# Spoofing alert
ros2 topic echo /spoofing/alert

# Detailed status
ros2 topic echo /spoofing/status

# Vehicle commands
ros2 topic echo /cmd_vel

# Navigation mode
ros2 topic echo /nav/mode

# Anomaly details
ros2 topic echo /spoofing/anomaly_detail
```

## Simulation Route Visualization

**Note:** The 3D simulation visualizer (`simulation/index.html`) displays a hardcoded route and waypoints for visualization, which may appear as a central or default path in the city. This visual route is not dynamically loaded from the backend and may not match the actual navigation route used by the backend controller.

- The backend navigation route is defined by the `waypoints_json` parameter in `config/nav_params.yaml`. You can customize the car's real route by editing this parameter with your desired list of GPS waypoints.
- The frontend visualizer currently draws its own static route for display. The car's position is updated live from the `/fused_pose` topic, but the visual route and markers are not automatically synchronized with backend waypoints.
- To make the visual route match the backend, further integration is needed to load waypoints from ROS2 or a shared config.

If you see the car navigating through the center of the map, it is due to the frontend's hardcoded route. The backend navigation logic and anti-spoofing operate on the waypoints you set in the YAML config.

## ML Model Training (Optional)

The system works out of the box with a heuristic classifier. To train the LSTM model:

```bash
# Generate synthetic training data
cd scripts
python generate_training_data.py -o ../data/training_data.csv

# Train the model
python train_model.py --data ../data/training_data.csv -o ../models/spoof_detector.pt --epochs 50

# Update config to use the model
# In config/nav_params.yaml, set ml_spoof_classifier.model_path to the model path
```

## Testing

Run unit tests (no ROS2 required):
```bash
cd /path/to/nav
python -m pytest tests/ -v
```

## ROS2 Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/gps/fix` | `NavSatFix` | Raw GPS readings |
| `/imu/data` | `Imu` | IMU measurements |
| `/odom` | `Odometry` | Wheel odometry |
| `/fused_pose` | `PoseStamped` | EKF fused position |
| `/spoofing/anomaly_score` | `Float64` | Physics-based anomaly score (0–1) |
| `/spoofing/detection` | `Float64` | ML classifier probability (0–1) |
| `/spoofing/alert` | `Bool` | Active spoofing alert |
| `/spoofing/status` | `String` | JSON status detail |
| `/cmd_vel` | `Twist` | Vehicle velocity command |
| `/nav/mode` | `String` | Navigation mode (NORMAL/ALERT/SAFE_STOP) |
| `/nav/target_waypoint` | `PointStamped` | Current target waypoint |

## Configuration

All parameters are in `config/nav_params.yaml`. Key tuning parameters:

- **`anomaly_detector_node.drift_threshold_m`** — GPS-EKF drift threshold (default: 10m)
- **`spoof_alert_manager.alert_threshold`** — Combined score to trigger alert (default: 0.55)
- **`nav_controller_node.alert_speed_factor`** — Speed reduction on alert (default: 0.2 = 20%)
- **`nav_controller_node.safe_stop_timeout`** — Seconds before initiating safe stop (default: 5s)

## Project Structure

```
nav/
├── nav_antispoofing/          # ROS2 Python package
│   ├── __init__.py
│   ├── gps_sensor_node.py     # Simulated GPS
│   ├── imu_sensor_node.py     # Simulated IMU
│   ├── odom_sensor_node.py    # Simulated odometry
│   ├── ekf_fusion_node.py     # Extended Kalman Filter
│   ├── anomaly_detector_node.py # Physics-based detection
│   ├── ml_spoof_classifier.py # LSTM classifier
│   ├── spoof_alert_manager.py # Alert state machine
│   ├── waypoint_manager_node.py # Waypoint navigation
│   ├── nav_controller_node.py # Spoof-aware controller
│   └── gps_spoofer_node.py    # Test spoofing tool
├── launch/
│   ├── full_system_launch.py
│   └── test_spoofing_launch.py
├── config/
│   └── nav_params.yaml
├── scripts/
│   ├── generate_training_data.py
│   └── train_model.py
├── tests/
│   ├── test_ekf.py
│   ├── test_anomaly_detector.py
│   ├── test_ml_classifier.py
│   └── test_nav_controller.py
├── models/                    # Trained ML models
├── package.xml
├── setup.py
├── setup.cfg
└── README.md
```

## License

MIT
   
      