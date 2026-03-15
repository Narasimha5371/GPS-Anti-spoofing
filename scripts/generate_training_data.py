#!/usr/bin/env python3
"""
Generate synthetic training data for the LSTM spoof detector.

Creates CSV with normal and spoofed GPS scenarios:
  - Normal: vehicle follows circular path with noise
  - Spoofed: gradual drift, sudden jump, or replay attacks

Output columns:
  timestamp, gps_x, gps_y, ekf_x, ekf_y, drift, anomaly_score, speed, label
"""

import math
import csv
import argparse
import os
import numpy as np


def simulate_circular_path(t, radius=100.0, speed=5.0):
    """True position on circular path."""
    omega = speed / radius
    x = radius * math.cos(omega * t)
    y = radius * math.sin(omega * t)
    return x, y


def generate_normal_data(n_samples=1000, dt=0.1):
    """Generate clean GPS data with normal noise."""
    data = []
    for i in range(n_samples):
        t = i * dt
        true_x, true_y = simulate_circular_path(t)

        # GPS with noise
        gps_x = true_x + np.random.normal(0, 0.5)
        gps_y = true_y + np.random.normal(0, 0.5)

        # EKF tracks closely
        ekf_x = true_x + np.random.normal(0, 0.1)
        ekf_y = true_y + np.random.normal(0, 0.1)

        drift = math.sqrt((gps_x - ekf_x)**2 + (gps_y - ekf_y)**2)

        # Speed
        if i > 0:
            prev_x = data[-1]['gps_x']
            prev_y = data[-1]['gps_y']
            speed = math.sqrt((gps_x - prev_x)**2 + (gps_y - prev_y)**2) / dt
        else:
            speed = 5.0

        anomaly = min(1.0, drift / 20.0)

        data.append({
            'timestamp': round(t, 3),
            'gps_x': round(gps_x, 4),
            'gps_y': round(gps_y, 4),
            'ekf_x': round(ekf_x, 4),
            'ekf_y': round(ekf_y, 4),
            'drift': round(drift, 4),
            'anomaly_score': round(anomaly, 4),
            'speed': round(speed, 4),
            'label': 0  # normal
        })

    return data


def generate_spoofed_data(n_samples=1000, dt=0.1, attack_type='gradual_drift'):
    """Generate spoofed GPS data."""
    data = []
    attack_start = n_samples // 3  # attack starts at 1/3

    for i in range(n_samples):
        t = i * dt
        true_x, true_y = simulate_circular_path(t)

        # EKF still tracks truth (uses IMU + odom too)
        ekf_x = true_x + np.random.normal(0, 0.1)
        ekf_y = true_y + np.random.normal(0, 0.1)

        label = 0
        if i >= attack_start:
            label = 1
            attack_t = (i - attack_start) * dt

            if attack_type == 'gradual_drift':
                offset_x = 0.5 * attack_t
                offset_y = 0.3 * attack_t
            elif attack_type == 'sudden_jump':
                offset_x = 50.0 if attack_t > 0.5 else attack_t * 100.0
                offset_y = 30.0 if attack_t > 0.5 else attack_t * 60.0
            elif attack_type == 'replay':
                offset_x = 20.0 * math.sin(0.5 * attack_t)
                offset_y = 20.0 * math.cos(0.5 * attack_t)
            else:
                offset_x = offset_y = 0.0

            gps_x = true_x + offset_x + np.random.normal(0, 0.3)
            gps_y = true_y + offset_y + np.random.normal(0, 0.3)
        else:
            gps_x = true_x + np.random.normal(0, 0.5)
            gps_y = true_y + np.random.normal(0, 0.5)

        drift = math.sqrt((gps_x - ekf_x)**2 + (gps_y - ekf_y)**2)

        if i > 0:
            prev_x = data[-1]['gps_x']
            prev_y = data[-1]['gps_y']
            speed = math.sqrt((gps_x - prev_x)**2 + (gps_y - prev_y)**2) / dt
        else:
            speed = 5.0

        anomaly = min(1.0, drift / 20.0)

        data.append({
            'timestamp': round(t, 3),
            'gps_x': round(gps_x, 4),
            'gps_y': round(gps_y, 4),
            'ekf_x': round(ekf_x, 4),
            'ekf_y': round(ekf_y, 4),
            'drift': round(drift, 4),
            'anomaly_score': round(anomaly, 4),
            'speed': round(speed, 4),
            'label': label
        })

    return data


def main():
    parser = argparse.ArgumentParser(
        description='Generate training data for spoof detector'
    )
    parser.add_argument('--output', '-o', default='training_data.csv',
                        help='Output CSV file path')
    parser.add_argument('--normal-samples', type=int, default=2000,
                        help='Number of normal samples')
    parser.add_argument('--attack-samples', type=int, default=1500,
                        help='Samples per attack type')
    parser.add_argument('--dt', type=float, default=0.1,
                        help='Time step in seconds')
    args = parser.parse_args()

    print('Generating training data...')

    all_data = []

    # Normal data
    print(f'  Normal: {args.normal_samples} samples')
    all_data.extend(generate_normal_data(args.normal_samples, args.dt))

    # Spoofed data — all attack types
    for attack in ['gradual_drift', 'sudden_jump', 'replay']:
        print(f'  {attack}: {args.attack_samples} samples')
        all_data.extend(
            generate_spoofed_data(args.attack_samples, args.dt, attack)
        )

    # Shuffle
    np.random.shuffle(all_data)

    # Write CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_data[0].keys())
        writer.writeheader()
        writer.writerows(all_data)

    total = len(all_data)
    spoofed = sum(1 for d in all_data if d['label'] == 1)
    print(f'\nSaved {total} samples to {args.output}')
    print(f'  Normal: {total - spoofed} ({(total - spoofed)/total*100:.1f}%)')
    print(f'  Spoofed: {spoofed} ({spoofed/total*100:.1f}%)')


if __name__ == '__main__':
    main()
