# ============================================================
# DriveGuard AI — Synthetic Driving Data Generator
# ============================================================
# Generates realistic driving sensor data for 6 behaviour classes
# using physics-based profiles with controlled noise injection.
# ============================================================

import numpy as np
import pandas as pd
import os
from tqdm import tqdm

import config


def _normal_profile(n):
    """Generate 'Normal Driving' sensor data."""
    return {
        "speed":            np.random.normal(65, 12, n).clip(30, 100),
        "acceleration":     np.random.normal(0.5, 0.3, n).clip(0, 2),
        "braking":          np.random.normal(0.3, 0.2, n).clip(0, 1.5),
        "steering_angle":   np.random.normal(0, 5, n).clip(-15, 15),
        "lane_deviation":   np.random.normal(0.05, 0.04, n).clip(0, 0.2),
        "eye_aspect_ratio": np.random.normal(0.75, 0.08, n).clip(0.35, 1.0),
        "head_yaw":         np.random.normal(0, 4, n).clip(-10, 10),
        "phone_usage":      np.zeros(n),
    }


def _aggressive_profile(n):
    """Generate 'Aggressive Driving' sensor data — high speed, hard steering."""
    return {
        "speed":            np.random.normal(110, 18, n).clip(80, 160),
        "acceleration":     np.random.normal(3.5, 1.2, n).clip(1.5, 8),
        "braking":          np.random.normal(1.0, 0.8, n).clip(0, 4),
        "steering_angle":   np.random.normal(0, 14, n).clip(-40, 40),
        "lane_deviation":   np.random.normal(0.15, 0.1, n).clip(0, 0.5),
        "eye_aspect_ratio": np.random.normal(0.70, 0.10, n).clip(0.30, 1.0),
        "head_yaw":         np.random.normal(0, 7, n).clip(-18, 18),
        "phone_usage":      np.zeros(n),
    }


def _distracted_profile(n):
    """Generate 'Distracted Driving' — head turned, lane drift, possible phone."""
    phone = np.zeros(n)
    phone[np.random.choice(n, size=int(n * 0.4), replace=False)] = 1.0
    return {
        "speed":            np.random.normal(60, 15, n).clip(20, 100),
        "acceleration":     np.random.normal(0.6, 0.4, n).clip(0, 3),
        "braking":          np.random.normal(0.5, 0.3, n).clip(0, 2),
        "steering_angle":   np.random.normal(0, 10, n).clip(-30, 30),
        "lane_deviation":   np.random.normal(0.25, 0.12, n).clip(0, 0.7),
        "eye_aspect_ratio": np.random.normal(0.65, 0.12, n).clip(0.25, 1.0),
        "head_yaw":         np.random.normal(15, 8, n).clip(-35, 35),
        "phone_usage":      phone,
    }


def _drowsy_profile(n):
    """Generate 'Drowsy Driving' — low EAR, slow reactions, lane drift."""
    return {
        "speed":            np.random.normal(55, 12, n).clip(20, 85),
        "acceleration":     np.random.normal(0.3, 0.2, n).clip(0, 1.5),
        "braking":          np.random.normal(0.2, 0.15, n).clip(0, 1),
        "steering_angle":   np.random.normal(0, 6, n).clip(-18, 18),
        "lane_deviation":   np.random.normal(0.20, 0.12, n).clip(0, 0.6),
        "eye_aspect_ratio": np.random.normal(0.22, 0.06, n).clip(0.10, 0.35),
        "head_yaw":         np.random.normal(5, 8, n).clip(-25, 25),
        "phone_usage":      np.zeros(n),
    }


def _unsafe_braking_profile(n):
    """Generate 'Unsafe Braking' — sudden hard stops."""
    return {
        "speed":            np.random.normal(70, 20, n).clip(20, 120),
        "acceleration":     np.random.normal(0.4, 0.3, n).clip(0, 2),
        "braking":          np.random.normal(5.5, 1.5, n).clip(3, 10),
        "steering_angle":   np.random.normal(0, 8, n).clip(-25, 25),
        "lane_deviation":   np.random.normal(0.15, 0.1, n).clip(0, 0.5),
        "eye_aspect_ratio": np.random.normal(0.72, 0.10, n).clip(0.30, 1.0),
        "head_yaw":         np.random.normal(0, 5, n).clip(-12, 12),
        "phone_usage":      np.zeros(n),
    }


def _unsafe_accel_profile(n):
    """Generate 'Unsafe Acceleration' — sudden strong forward thrust."""
    return {
        "speed":            np.random.normal(90, 25, n).clip(40, 150),
        "acceleration":     np.random.normal(5.0, 1.5, n).clip(3, 9),
        "braking":          np.random.normal(0.2, 0.15, n).clip(0, 1),
        "steering_angle":   np.random.normal(0, 6, n).clip(-20, 20),
        "lane_deviation":   np.random.normal(0.10, 0.08, n).clip(0, 0.4),
        "eye_aspect_ratio": np.random.normal(0.74, 0.09, n).clip(0.35, 1.0),
        "head_yaw":         np.random.normal(0, 4, n).clip(-10, 10),
        "phone_usage":      np.zeros(n),
    }


# Profile lookup by class name
PROFILE_GENERATORS = {
    "Normal":              _normal_profile,
    "Aggressive":          _aggressive_profile,
    "Distracted":          _distracted_profile,
    "Drowsy":              _drowsy_profile,
    "Unsafe Braking":      _unsafe_braking_profile,
    "Unsafe Acceleration": _unsafe_accel_profile,
}


def generate_dataset(samples_per_class=None, save=True):
    """
    Generate the full synthetic dataset.

    Returns:
        pd.DataFrame with columns = config.FEATURES + ['label']
    """
    spc = samples_per_class or config.SAMPLES_PER_CLASS
    all_rows = []

    print("=" * 60)
    print("  DriveGuard AI — Generating Synthetic Driving Dataset")
    print("=" * 60)

    for cls_name in config.CLASSES:
        gen = PROFILE_GENERATORS[cls_name]
        data = gen(spc)

        # Add Gaussian noise
        for feat in config.FEATURES:
            if feat == "phone_usage":
                continue  # keep binary
            data[feat] += np.random.normal(0, config.NOISE_LEVEL * np.std(data[feat]), spc)

        # Build rows
        for i in tqdm(range(spc), desc=f"  Generating [{cls_name:>20s}]", ncols=80):
            row = {feat: float(data[feat][i]) for feat in config.FEATURES}
            row["label"] = cls_name
            all_rows.append(row)

    df = pd.DataFrame(all_rows)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    if save:
        csv_path = os.path.join(config.DATA_DIR, "driving_dataset.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Dataset saved to {csv_path}")
        print(f"  Total samples : {len(df)}")
        print(f"  Features      : {config.NUM_FEATURES}")
        print(f"  Classes       : {config.NUM_CLASSES}")
        print(f"  Per class     : {spc}")

        # Print class distribution
        print("\n  Class Distribution:")
        for cls, count in df["label"].value_counts().items():
            print(f"    {cls:>22s}: {count:>6d}")

    return df


def generate_sequences(df, sequence_length=None):
    """
    Convert flat samples into overlapping sequences for CNN/LSTM.

    For each class, creates overlapping windows of shape
    (sequence_length, num_features).

    Returns:
        X: np.ndarray of shape (num_sequences, seq_len, num_features)
        y: np.ndarray of shape (num_sequences,) — integer labels
    """
    seq_len = sequence_length or config.SEQUENCE_LENGTH
    X_sequences = []
    y_labels = []

    print("\n  Creating sequences (window=%d)..." % seq_len)

    for cls_name in config.CLASSES:
        cls_data = df[df["label"] == cls_name][config.FEATURES].values
        cls_idx = config.CLASS_TO_IDX[cls_name]

        # Sliding window with stride = seq_len // 2 (50% overlap)
        stride = max(1, seq_len // 2)
        for start in range(0, len(cls_data) - seq_len + 1, stride):
            window = cls_data[start:start + seq_len]
            X_sequences.append(window)
            y_labels.append(cls_idx)

    X = np.array(X_sequences, dtype=np.float32)
    y = np.array(y_labels, dtype=np.int32)

    # Shuffle
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    print(f"  ✓ Sequences: {X.shape[0]}  |  Shape: {X.shape}")
    return X, y


# ── CLI Entry Point ───────────────────────────────────────────
if __name__ == "__main__":
    df = generate_dataset()
    X, y = generate_sequences(df)
    print(f"\n  Final X shape: {X.shape}")
    print(f"  Final y shape: {y.shape}")
    print(f"  Class counts : {np.bincount(y)}")
    print("\n✓ Data generation complete.")
