#!/usr/bin/env python3
"""
Train the LSTM spoofing detector model.

Usage:
  python train_model.py --data training_data.csv --output ../models/spoof_detector.pt

Requires PyTorch.
"""

import argparse
import csv
import os
import sys
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print('ERROR: PyTorch required. Install with: pip install torch')
    sys.exit(1)


# ────────────────────────────────────────────────────────────────
# Model Definition (same as in ml_spoof_classifier.py)
# ────────────────────────────────────────────────────────────────

class SpoofDetectorLSTM(nn.Module):
    def __init__(self, input_size=7, hidden_size=32, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=0.2
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.classifier(last_hidden).squeeze(-1)


# ────────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────────

class SpoofDataset(Dataset):
    """Sliding window dataset from CSV."""

    FEATURE_COLS = ['gps_x', 'gps_y', 'ekf_x', 'ekf_y',
                    'drift', 'anomaly_score', 'speed']

    def __init__(self, csv_path, window_size=20):
        self.window_size = window_size
        self.windows = []
        self.labels = []

        # Read CSV
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Build sliding windows
        features = []
        labels = []
        for row in rows:
            feat = [float(row[c]) for c in self.FEATURE_COLS]
            features.append(feat)
            labels.append(int(row['label']))

        features = np.array(features, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)

        # Normalize features
        self.mean = features.mean(axis=0)
        self.std = features.std(axis=0) + 1e-8
        features = (features - self.mean) / self.std

        # Create windows
        for i in range(len(features) - window_size):
            window = features[i:i + window_size]
            # Label = majority label in window, or last label
            label = labels[i + window_size - 1]
            self.windows.append(window)
            self.labels.append(label)

        self.windows = np.array(self.windows)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.windows[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )


# ────────────────────────────────────────────────────────────────
# Training
# ────────────────────────────────────────────────────────────────

def train(args):
    print(f'Loading data from {args.data}...')
    dataset = SpoofDataset(args.data, window_size=args.window_size)
    print(f'  Total windows: {len(dataset)}')
    print(f'  Spoofed ratio: {dataset.labels.mean():.2%}')

    # Train/val split
    n = len(dataset)
    n_train = int(n * 0.8)
    n_val = n - n_train
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    # Model
    model = SpoofDetectorLSTM(
        input_size=7,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    print(f'\nTraining for {args.epochs} epochs...')
    print(f'  Model params: {sum(p.numel() for p in model.parameters()):,}')

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        # ── Train ──
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y)
            train_correct += ((pred > 0.5).float() == y).sum().item()
            train_total += len(y)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y in val_loader:
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item() * len(y)
                val_correct += ((pred > 0.5).float() == y).sum().item()
                val_total += len(y)

        train_acc = train_correct / max(1, train_total)
        val_acc = val_correct / max(1, val_total)

        print(
            f'  Epoch {epoch + 1:3d}/{args.epochs} | '
            f'Train Loss: {train_loss / max(1, train_total):.4f} '
            f'Acc: {train_acc:.4f} | '
            f'Val Loss: {val_loss / max(1, val_total):.4f} '
            f'Acc: {val_acc:.4f}'
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(os.path.abspath(args.output)),
                        exist_ok=True)
            torch.save(model.state_dict(), args.output)

    print(f'\nBest validation accuracy: {best_val_acc:.4f}')
    print(f'Model saved to {args.output}')

    # Save normalization stats for inference
    stats_path = args.output.replace('.pt', '_stats.npz')
    np.savez(stats_path, mean=dataset.mean, std=dataset.std)
    print(f'Normalization stats saved to {stats_path}')


def main():
    parser = argparse.ArgumentParser(description='Train LSTM spoof detector')
    parser.add_argument('--data', required=True, help='Training CSV path')
    parser.add_argument('--output', '-o', default='../models/spoof_detector.pt',
                        help='Output model path')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden-size', type=int, default=32)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--window-size', type=int, default=20)
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
