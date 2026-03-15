#!/usr/bin/env python3
"""
Performance Evaluation — Anti-Spoofing Navigation System

Runs offline benchmarks measuring:
  1. Detection accuracy (precision, recall, F1) per attack type
  2. Detection latency (time-to-detect in samples)
  3. Computational throughput (EKF, anomaly detector, classifier)
  4. False positive rate under normal conditions
  5. EKF tracking error vs. ground truth

No ROS2 required — runs standalone.
"""

import math
import time
import sys
import os
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ═══════════════════════════════════════════════════════════════
# Simulators (mirror the ROS2 node logic)
# ═══════════════════════════════════════════════════════════════

class VehicleSimulator:
    """Simulate circular path vehicle motion."""
    def __init__(self, radius=100.0, speed=5.0):
        self.radius = radius
        self.speed = speed
        self.omega = speed / radius

    def true_position(self, t):
        x = self.radius * math.cos(self.omega * t)
        y = self.radius * math.sin(self.omega * t)
        yaw = self.omega * t + math.pi / 2
        return x, y, yaw


class SensorSimulator:
    """Simulate GPS, IMU, odometry readings."""
    def __init__(self, vehicle, gps_noise=0.5, imu_noise=0.01, odom_noise=0.02):
        self.vehicle = vehicle
        self.gps_noise = gps_noise
        self.imu_noise = imu_noise
        self.odom_noise = odom_noise
        self.odom_drift_x = 0.0
        self.odom_drift_y = 0.0

    def gps(self, t, spoof_offset=(0, 0)):
        x, y, _ = self.vehicle.true_position(t)
        return (
            x + np.random.normal(0, self.gps_noise) + spoof_offset[0],
            y + np.random.normal(0, self.gps_noise) + spoof_offset[1]
        )

    def imu_yaw(self, t):
        _, _, yaw = self.vehicle.true_position(t)
        return yaw + np.random.normal(0, self.imu_noise)

    def odom(self, t):
        x, y, _ = self.vehicle.true_position(t)
        self.odom_drift_x += np.random.normal(0, 0.001)
        self.odom_drift_y += np.random.normal(0, 0.001)
        return (
            x + self.odom_drift_x + np.random.normal(0, self.odom_noise),
            y + self.odom_drift_y + np.random.normal(0, self.odom_noise)
        )


class EKF:
    """Minimal EKF (mirrors ekf_fusion_node logic)."""
    def __init__(self):
        self.x = np.zeros(6)
        self.P = np.eye(6) * 10.0
        self.Q = np.diag([0.1, 0.1, 0.5, 0.5, 0.01, 0.001])

    def predict(self, dt):
        if dt <= 0: return
        self.x[0] += self.x[2] * dt
        self.x[1] += self.x[3] * dt
        self.x[4] += self.x[5] * dt
        F = np.eye(6)
        F[0, 2] = dt
        F[1, 3] = dt
        F[4, 5] = dt
        self.P = F @ self.P @ F.T + self.Q * dt

    def update(self, z, H, R):
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P

    def update_gps(self, gx, gy):
        H = np.zeros((2, 6)); H[0,0]=1; H[1,1]=1
        self.update(np.array([gx, gy]), H, np.eye(2) * 1.0)

    def update_odom(self, ox, oy):
        H = np.zeros((2, 6)); H[0,0]=1; H[1,1]=1
        self.update(np.array([ox, oy]), H, np.eye(2) * 0.1)

    def update_imu(self, yaw):
        H = np.zeros((1, 6)); H[0,4]=1
        self.update(np.array([yaw]), H, np.array([[0.05**2]]))


class AnomalyDetector:
    """Mirrors enhanced anomaly_detector_node logic."""
    def __init__(self, max_vel=30.0, drift_thresh=3.0, vel_jump_thresh=15.0,
                 drift_trend_thresh=0.05, vel_mismatch_thresh=2.0,
                 history_window=50, alpha=0.3):
        self.max_vel = max_vel
        self.drift_thresh = drift_thresh
        self.vel_jump_thresh = vel_jump_thresh
        self.drift_trend_thresh = drift_trend_thresh
        self.vel_mismatch_thresh = vel_mismatch_thresh
        self.history_window = history_window
        self.alpha = alpha
        self.ema = 0.0
        self.prev_gps = None
        self.drift_history = []

    def update(self, gps_x, gps_y, ekf_x, ekf_y, dt, odom_vx=None, odom_vy=None):
        scores = {}

        # Velocity jump + GPS velocity
        vel_score = 0.0
        gps_vx, gps_vy = 0.0, 0.0
        if self.prev_gps is not None and dt > 0.001:
            dx = gps_x - self.prev_gps[0]
            dy = gps_y - self.prev_gps[1]
            gps_vx, gps_vy = dx / dt, dy / dt
            speed = math.sqrt(dx**2 + dy**2) / dt
            if speed > self.max_vel:
                vel_score = min(1.0, speed / (self.max_vel * 3))
            elif speed > self.vel_jump_thresh:
                vel_score = min(0.5, (speed - self.vel_jump_thresh) / (self.max_vel - self.vel_jump_thresh))
        scores['velocity_jump'] = vel_score

        # Absolute drift
        drift = math.sqrt((gps_x - ekf_x)**2 + (gps_y - ekf_y)**2)
        drift_score = 0.0
        if drift > self.drift_thresh:
            drift_score = min(1.0, drift / (self.drift_thresh * 3))
        elif drift > self.drift_thresh * 0.3:
            drift_score = min(0.4, (drift - self.drift_thresh * 0.3) / (self.drift_thresh * 0.7))
        scores['ekf_drift'] = drift_score

        # Drift TREND (key for gradual attacks)
        self.drift_history.append(drift)
        if len(self.drift_history) > self.history_window:
            self.drift_history = self.drift_history[-self.history_window:]

        drift_trend_score = 0.0
        trend_window = min(30, len(self.drift_history))
        if trend_window >= 10:
            recent_drifts = self.drift_history[-trend_window:]
            x_vals = np.arange(len(recent_drifts))
            slope = np.polyfit(x_vals, recent_drifts, 1)[0]
            if slope > self.drift_trend_thresh:
                drift_trend_score = min(1.0, slope / (self.drift_trend_thresh * 5))
            elif slope > self.drift_trend_thresh * 0.3:
                drift_trend_score = min(0.3, (slope - self.drift_trend_thresh * 0.3) / (self.drift_trend_thresh * 0.7))

            # Monotonicity boost
            increasing = sum(1 for i in range(1, len(recent_drifts)) if recent_drifts[i] > recent_drifts[i-1])
            monotonic_ratio = increasing / (len(recent_drifts) - 1)
            if monotonic_ratio > 0.7 and slope > 0:
                drift_trend_score = max(drift_trend_score, monotonic_ratio * 0.6)
        scores['drift_trend'] = drift_trend_score

        # Cross-sensor velocity mismatch
        vel_mismatch_score = 0.0
        if odom_vx is not None and odom_vy is not None:
            odom_speed = math.sqrt(odom_vx**2 + odom_vy**2)
            gps_speed = math.sqrt(gps_vx**2 + gps_vy**2)
            speed_diff = abs(gps_speed - odom_speed)
            if speed_diff > self.vel_mismatch_thresh:
                vel_mismatch_score = min(1.0, speed_diff / (self.vel_mismatch_thresh * 5))
            elif speed_diff > self.vel_mismatch_thresh * 0.5:
                vel_mismatch_score = min(0.3, (speed_diff - self.vel_mismatch_thresh * 0.5) / (self.vel_mismatch_thresh * 0.5))
        scores['velocity_mismatch'] = vel_mismatch_score

        # Combined (reweighted for gradual drift)
        raw = min(1.0, sum([
            scores['velocity_jump']     * 0.25,
            scores['ekf_drift']         * 0.20,
            scores['drift_trend']       * 0.25,
            scores['velocity_mismatch'] * 0.20,
        ]))
        self.ema = self.alpha * raw + (1 - self.alpha) * self.ema
        self.prev_gps = (gps_x, gps_y)
        return self.ema, drift


class HeuristicClassifier:
    """Mirrors enhanced ml_spoof_classifier heuristic logic."""
    def __init__(self, window_size=20):
        self.window_size = window_size
        self.buffer = []

    def update(self, gps_x, gps_y, ekf_x, ekf_y, drift, anomaly, speed):
        self.buffer.append([gps_x, gps_y, ekf_x, ekf_y, drift, anomaly, speed])
        if len(self.buffer) > self.window_size * 2:
            self.buffer = self.buffer[-self.window_size * 2:]
        if len(self.buffer) < self.window_size:
            return 0.0
        window = np.array(self.buffer[-self.window_size:])
        drifts = window[:, 4]
        anomalies = window[:, 5]
        speeds = window[:, 6]

        mean_drift = np.mean(drifts)
        max_drift = np.max(drifts)
        x_vals = np.arange(len(drifts))
        drift_trend = np.polyfit(x_vals, drifts, 1)[0]

        # Drift acceleration
        if len(drifts) >= 10:
            first_half = drifts[:len(drifts)//2]
            second_half = drifts[len(drifts)//2:]
            drift_accel = np.mean(second_half) - np.mean(first_half)
        else:
            drift_accel = 0.0

        # Monotonicity
        if len(drifts) >= 3:
            increasing = sum(1 for i in range(1, len(drifts)) if drifts[i] > drifts[i-1])
            monotonic_ratio = increasing / (len(drifts) - 1)
        else:
            monotonic_ratio = 0.0

        mean_anomaly = np.mean(anomalies)
        speed_std = np.std(speeds)

        score = 0.0
        # Drift magnitude (lowered thresholds)
        if mean_drift > 1.5:
            score += min(0.3, mean_drift / 10.0)
        if max_drift > 3.0:
            score += min(0.15, max_drift / 20.0)
        # Drift trend (key for gradual)
        if drift_trend > 0.01:
            score += min(0.35, drift_trend * 3.0)
        # Drift acceleration
        if drift_accel > 0.1:
            score += min(0.1, drift_accel / 2.0)
        # Monotonicity
        if monotonic_ratio > 0.6 and mean_drift > 0.5:
            score += min(0.2, monotonic_ratio * 0.25)
        # Anomaly accumulation
        score += mean_anomaly * 0.35
        # Speed irregularity
        if speed_std > 5.0:
            score += min(0.1, speed_std / 50.0)

        return min(1.0, score)


class AlertManager:
    """Mirrors spoof_alert_manager logic."""
    def __init__(self, w_a=0.4, w_m=0.6, suspect=0.35, alert=0.55, clear=0.15):
        self.w_a = w_a
        self.w_m = w_m
        self.thresh_suspect = suspect
        self.thresh_alert = alert
        self.thresh_clear = clear
        self.state = 'CLEAR'
        self.suspect_start = None

    def update(self, anomaly_score, ml_score, t):
        combined = self.w_a * anomaly_score + self.w_m * ml_score
        if self.state == 'CLEAR':
            if combined >= self.thresh_alert:
                self.state = 'ALERT'
            elif combined >= self.thresh_suspect:
                self.state = 'SUSPECT'
                self.suspect_start = t
        elif self.state == 'SUSPECT':
            if combined >= self.thresh_alert:
                self.state = 'ALERT'
            elif combined < self.thresh_suspect:
                self.state = 'CLEAR'
                self.suspect_start = None
            elif (t - self.suspect_start) > 2.0:
                self.state = 'ALERT'
        elif self.state == 'ALERT':
            if combined < self.thresh_clear:
                self.state = 'CLEAR'
        return self.state == 'ALERT', combined


# ═══════════════════════════════════════════════════════════════
# Attack Generators
# ═══════════════════════════════════════════════════════════════

def no_attack(t, attack_t):
    return (0.0, 0.0)

def gradual_drift(t, attack_t):
    rate = 0.5  # m/s drift
    return (rate * attack_t, rate * 0.7 * attack_t)

def sudden_jump(t, attack_t):
    ramp = min(1.0, attack_t / 0.5)
    return (50.0 * ramp, 30.0 * ramp)

def replay_attack(t, attack_t):
    return (20.0 * math.sin(0.5 * attack_t), 20.0 * math.cos(0.5 * attack_t))


ATTACKS = {
    'none': no_attack,
    'gradual_drift': gradual_drift,
    'sudden_jump': sudden_jump,
    'replay': replay_attack,
}


# ═══════════════════════════════════════════════════════════════
# Simulation Runner
# ═══════════════════════════════════════════════════════════════

def run_simulation(attack_name, duration=60.0, dt=0.1, attack_start=20.0):
    """Run one full simulation and return metrics."""
    vehicle = VehicleSimulator()
    sensors = SensorSimulator(vehicle)
    ekf = EKF()
    anomaly_det = AnomalyDetector()
    classifier = HeuristicClassifier()
    alert_mgr = AlertManager()
    attack_fn = ATTACKS[attack_name]

    results = {
        'timestamps': [],
        'true_labels': [],
        'predicted_alerts': [],
        'combined_scores': [],
        'ekf_errors': [],
        'drifts': [],
        'anomaly_scores': [],
        'ml_scores': [],
    }

    detection_time = None
    prev_speed = 5.0

    steps = int(duration / dt)
    for i in range(steps):
        t = i * dt
        is_attack = (attack_name != 'none' and t >= attack_start)
        attack_t = t - attack_start if is_attack else 0.0

        spoof_offset = attack_fn(t, attack_t) if is_attack else (0.0, 0.0)

        # Sensor readings
        gps_x, gps_y = sensors.gps(t, spoof_offset)
        imu_yaw = sensors.imu_yaw(t)
        odom_x, odom_y = sensors.odom(t)

        # EKF
        ekf.predict(dt)
        ekf.update_gps(gps_x, gps_y)
        ekf.update_odom(odom_x, odom_y)
        ekf.update_imu(imu_yaw)

        ekf_x, ekf_y = ekf.x[0], ekf.x[1]
        true_x, true_y, _ = vehicle.true_position(t)
        ekf_err = math.sqrt((ekf_x - true_x)**2 + (ekf_y - true_y)**2)

        # Anomaly detection (with cross-sensor velocity)
        omega = vehicle.omega
        theta = omega * t
        odom_vx = -vehicle.radius * omega * math.sin(theta) + np.random.normal(0, 0.05)
        odom_vy = vehicle.radius * omega * math.cos(theta) + np.random.normal(0, 0.05)
        anomaly_score, drift = anomaly_det.update(
            gps_x, gps_y, ekf_x, ekf_y, dt,
            odom_vx=odom_vx, odom_vy=odom_vy
        )

        # Speed estimate
        if i > 0:
            speed = math.sqrt((gps_x - results.get('_prev_gps', (gps_x, gps_y))[0])**2 +
                            (gps_y - results.get('_prev_gps', (gps_x, gps_y))[1])**2) / dt
        else:
            speed = 5.0
        results['_prev_gps'] = (gps_x, gps_y)

        # ML classifier
        ml_score = classifier.update(gps_x, gps_y, ekf_x, ekf_y, drift, anomaly_score, speed)

        # Alert manager
        alert, combined = alert_mgr.update(anomaly_score, ml_score, t)

        # Track detection latency
        if is_attack and alert and detection_time is None:
            detection_time = t - attack_start

        results['timestamps'].append(t)
        results['true_labels'].append(1 if is_attack else 0)
        results['predicted_alerts'].append(1 if alert else 0)
        results['combined_scores'].append(combined)
        results['ekf_errors'].append(ekf_err)
        results['drifts'].append(drift)
        results['anomaly_scores'].append(anomaly_score)
        results['ml_scores'].append(ml_score)

    results.pop('_prev_gps', None)
    results['detection_latency'] = detection_time
    return results


def compute_metrics(results):
    """Compute precision, recall, F1, false positive rate."""
    y_true = np.array(results['true_labels'])
    y_pred = np.array(results['predicted_alerts'])

    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-10, precision + recall)
    fpr = fp / max(1, fp + tn)
    accuracy = (tp + tn) / max(1, tp + fp + fn + tn)

    return {
        'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn),
        'precision': precision, 'recall': recall, 'f1': f1,
        'fpr': fpr, 'accuracy': accuracy,
        'detection_latency': results['detection_latency'],
        'mean_ekf_error': np.mean(results['ekf_errors']),
        'max_ekf_error': np.max(results['ekf_errors']),
    }


# ═══════════════════════════════════════════════════════════════
# Throughput Benchmark
# ═══════════════════════════════════════════════════════════════

def benchmark_throughput(n_iterations=10000):
    """Measure computational throughput of each component."""
    vehicle = VehicleSimulator()
    ekf = EKF()
    anomaly_det = AnomalyDetector()
    classifier = HeuristicClassifier()

    # Warm up
    for i in range(100):
        ekf.predict(0.1)
        ekf.update_gps(float(i), float(i))

    timings = {}

    # EKF predict+update
    start = time.perf_counter()
    for i in range(n_iterations):
        ekf.predict(0.1)
        ekf.update_gps(100.0 + i * 0.01, 50.0 + i * 0.01)
    timings['ekf_predict_update'] = (time.perf_counter() - start) / n_iterations

    # Anomaly detector
    start = time.perf_counter()
    for i in range(n_iterations):
        anomaly_det.update(100.0 + i * 0.01, 50.0 + i * 0.01,
                          100.0, 50.0, 0.1)
    timings['anomaly_detector'] = (time.perf_counter() - start) / n_iterations

    # Heuristic classifier
    start = time.perf_counter()
    for i in range(n_iterations):
        classifier.update(100.0, 50.0, 100.0, 50.0, 0.5, 0.1, 5.0)
    timings['heuristic_classifier'] = (time.perf_counter() - start) / n_iterations

    return timings


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    np.random.seed(42)

    print("=" * 72)
    print("  ANTI-SPOOFING NAVIGATION SYSTEM -- PERFORMANCE EVALUATION")
    print("=" * 72)

    # ── 1. Detection Accuracy per Attack Type ──
    print("\n" + "-" * 72)
    print("  1. DETECTION ACCURACY BY ATTACK TYPE")
    print("-" * 72)
    print(f"  {'Attack':<18} {'Prec':>7} {'Recall':>7} {'F1':>7} "
          f"{'FPR':>7} {'Acc':>7} {'Latency':>10}")
    print("  " + "-" * 68)

    all_metrics = {}
    for attack in ['none', 'gradual_drift', 'sudden_jump', 'replay']:
        results = run_simulation(attack, duration=60.0, dt=0.1)
        metrics = compute_metrics(results)
        all_metrics[attack] = metrics

        lat_str = (f"{metrics['detection_latency']:.1f}s"
                   if metrics['detection_latency'] is not None else "N/A")
        print(f"  {attack:<18} {metrics['precision']:>7.3f} "
              f"{metrics['recall']:>7.3f} {metrics['f1']:>7.3f} "
              f"{metrics['fpr']:>7.4f} {metrics['accuracy']:>7.3f} "
              f"{lat_str:>10}")

    # ── 2. EKF Tracking Performance ──
    print("\n" + "-" * 72)
    print("  2. EKF TRACKING PERFORMANCE (vs Ground Truth)")
    print("-" * 72)
    print(f"  {'Scenario':<18} {'Mean Error (m)':>15} {'Max Error (m)':>15}")
    print("  " + "-" * 50)
    for attack in ['none', 'gradual_drift', 'sudden_jump', 'replay']:
        m = all_metrics[attack]
        print(f"  {attack:<18} {m['mean_ekf_error']:>15.3f} "
              f"{m['max_ekf_error']:>15.3f}")

    # ── 3. Confusion Matrix (sudden_jump as example) ──
    print("\n" + "-" * 72)
    print("  3. CONFUSION MATRIX -- sudden_jump attack")
    print("-" * 72)
    m = all_metrics['sudden_jump']
    print(f"                    Predicted")
    print(f"                  Normal  Spoofed")
    print(f"  Actual Normal  | {m['TN']:>5} | {m['FP']:>5} |")
    print(f"  Actual Spoofed | {m['FN']:>5} | {m['TP']:>5} |")

    # ── 4. False Positive Analysis (no attack) ──
    print("\n" + "-" * 72)
    print("  4. FALSE POSITIVE ANALYSIS (no attack scenario)")
    print("-" * 72)
    m = all_metrics['none']
    total = m['TP'] + m['FP'] + m['FN'] + m['TN']
    print(f"  Total samples:        {total}")
    print(f"  False alarms:         {m['FP']}")
    print(f"  False positive rate:  {m['fpr']:.4f} ({m['fpr']*100:.2f}%)")

    # ── 5. Computational Throughput ──
    print("\n" + "-" * 72)
    print("  5. COMPUTATIONAL THROUGHPUT")
    print("-" * 72)
    timings = benchmark_throughput()
    print(f"  {'Component':<25} {'Time/iter':>12} {'Throughput':>15}")
    print("  " + "-" * 55)
    for name, t in timings.items():
        throughput = 1.0 / t if t > 0 else float('inf')
        print(f"  {name:<25} {t*1e6:>10.1f} us {throughput:>12,.0f} Hz")

    # ── 6. Detection Latency Summary ──
    print("\n" + "-" * 72)
    print("  6. DETECTION LATENCY (time from attack start to alert)")
    print("-" * 72)
    for attack in ['gradual_drift', 'sudden_jump', 'replay']:
        lat = all_metrics[attack]['detection_latency']
        bar = ""
        if lat is not None:
            bar_len = min(40, int(lat * 2))
            bar = "#" * bar_len
            print(f"  {attack:<18} {lat:>6.1f}s  {bar}")
        else:
            print(f"  {attack:<18}   N/A   (not detected)")

    # ── 7. Overall System Rating ──
    print("\n" + "=" * 72)
    print("  OVERALL SYSTEM PERFORMANCE SUMMARY")
    print("=" * 72)

    # Average F1 across attack types
    attack_f1s = [all_metrics[a]['f1'] for a in ['gradual_drift', 'sudden_jump', 'replay']]
    avg_f1 = np.mean(attack_f1s)
    avg_recall = np.mean([all_metrics[a]['recall'] for a in ['gradual_drift', 'sudden_jump', 'replay']])
    avg_precision = np.mean([all_metrics[a]['precision'] for a in ['gradual_drift', 'sudden_jump', 'replay']])
    fpr_none = all_metrics['none']['fpr']
    avg_latency = np.mean([all_metrics[a]['detection_latency'] for a in ['gradual_drift', 'sudden_jump', 'replay']
                           if all_metrics[a]['detection_latency'] is not None])

    print(f"  Average Precision:     {avg_precision:.3f}")
    print(f"  Average Recall:        {avg_recall:.3f}")
    print(f"  Average F1 Score:      {avg_f1:.3f}")
    print(f"  False Positive Rate:   {fpr_none:.4f}")
    print(f"  Avg Detection Latency: {avg_latency:.1f}s")
    print(f"  EKF Tracking Error:    {all_metrics['none']['mean_ekf_error']:.3f}m (normal)")

    ekf_hz = 1.0 / timings['ekf_predict_update']
    total_hz = 1.0 / sum(timings.values())
    print(f"  EKF Throughput:        {ekf_hz:,.0f} Hz")
    print(f"  Full Pipeline:         {total_hz:,.0f} Hz (real-time capable @ 50 Hz)")

    # Grade
    if avg_f1 > 0.9 and fpr_none < 0.05:
        grade = "A+"
    elif avg_f1 > 0.8 and fpr_none < 0.10:
        grade = "A"
    elif avg_f1 > 0.7:
        grade = "B"
    elif avg_f1 > 0.5:
        grade = "C"
    else:
        grade = "D"

    print(f"\n  System Grade: {grade}")
    print("=" * 72)


if __name__ == '__main__':
    main()
