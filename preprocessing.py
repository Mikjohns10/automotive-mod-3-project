# ============================================================
# DriveGuard AI — Data Preprocessing & Feature Engineering
# ============================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

import config


def load_dataset(csv_path=None):
    """Load the generated CSV dataset."""
    path = csv_path or os.path.join(config.DATA_DIR, "driving_dataset.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run data_generator.py first."
        )
    df = pd.read_csv(path)
    print(f"✓ Loaded dataset: {len(df)} samples, {df.shape[1]} columns")
    return df


def add_engineered_features(df):
    """
    Add derived features that help the model:
      - speed_accel_ratio  : speed / (acceleration + 1)
      - braking_intensity  : braking * speed / 100
      - head_deviation_abs : |head_yaw|
      - eye_risk           : 1 - EAR (high = more closure)
      - steering_abs       : |steering_angle|
    """
    df = df.copy()
    df["speed_accel_ratio"]  = df["speed"] / (df["acceleration"] + 1.0)
    df["braking_intensity"]  = df["braking"] * df["speed"] / 100.0
    df["head_deviation_abs"] = df["head_yaw"].abs()
    df["eye_risk"]           = 1.0 - df["eye_aspect_ratio"]
    df["steering_abs"]       = df["steering_angle"].abs()
    return df


# Combined feature list (original 8 + 5 engineered)
EXTENDED_FEATURES = config.FEATURES + [
    "speed_accel_ratio",
    "braking_intensity",
    "head_deviation_abs",
    "eye_risk",
    "steering_abs",
]


def preprocess_for_training(df, use_engineered=True):
    """
    Full preprocessing pipeline:
      1. Add engineered features (optional)
      2. Scale features with StandardScaler
      3. Encode labels to integers
      4. Split into train/val sets
      5. Create sequences for CNN/LSTM

    Returns:
        X_train, X_val, y_train, y_val, scaler
    """
    # Feature engineering
    if use_engineered:
        df = add_engineered_features(df)
        feature_cols = EXTENDED_FEATURES
    else:
        feature_cols = config.FEATURES

    print(f"  Using {len(feature_cols)} features: {feature_cols}")

    # Extract features and labels
    X_flat = df[feature_cols].values.astype(np.float32)
    y_flat = df["label"].map(config.CLASS_TO_IDX).values.astype(np.int32)

    # Scale
    scaler = StandardScaler()
    X_flat = scaler.fit_transform(X_flat)

    # Save scaler
    joblib.dump(scaler, config.SCALER_PATH)
    print(f"  ✓ Scaler saved to {config.SCALER_PATH}")

    # Create per-class DataFrames for sequence generation
    X_sequences = []
    y_labels = []
    seq_len = config.SEQUENCE_LENGTH
    stride = max(1, seq_len // 2)

    for cls_idx in range(config.NUM_CLASSES):
        mask = y_flat == cls_idx
        cls_data = X_flat[mask]

        for start in range(0, len(cls_data) - seq_len + 1, stride):
            window = cls_data[start:start + seq_len]
            X_sequences.append(window)
            y_labels.append(cls_idx)

    X = np.array(X_sequences, dtype=np.float32)
    y = np.array(y_labels, dtype=np.int32)

    # Shuffle
    perm = np.random.RandomState(42).permutation(len(X))
    X, y = X[perm], y[perm]

    # Train/Val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=(1 - config.TRAIN_SPLIT),
        random_state=42,
        stratify=y,
    )

    print(f"  ✓ Sequences  : {X.shape[0]} total")
    print(f"  ✓ Train      : {X_train.shape[0]}")
    print(f"  ✓ Validation : {X_val.shape[0]}")
    print(f"  ✓ Seq shape  : {X_train.shape[1:]}")

    return X_train, X_val, y_train, y_val, scaler


def preprocess_realtime(raw_features, scaler=None):
    """
    Preprocess a single frame or a sequence for real-time inference.

    Args:
        raw_features: dict or list of dicts with the 8 sensor readings
        scaler: fitted StandardScaler (loaded from disk if None)

    Returns:
        np.ndarray ready for model.predict()
    """
    if scaler is None:
        if not os.path.exists(config.SCALER_PATH):
            raise FileNotFoundError("Scaler not found. Train the model first.")
        scaler = joblib.load(config.SCALER_PATH)

    if isinstance(raw_features, dict):
        raw_features = [raw_features]

    rows = []
    for frame in raw_features:
        row = [frame.get(f, 0.0) for f in config.FEATURES]
        # Engineered features
        speed = frame.get("speed", 0)
        accel = frame.get("acceleration", 0)
        braking = frame.get("braking", 0)
        head_yaw = frame.get("head_yaw", 0)
        ear = frame.get("eye_aspect_ratio", 0.7)
        steering = frame.get("steering_angle", 0)

        row.append(speed / (accel + 1.0))         # speed_accel_ratio
        row.append(braking * speed / 100.0)        # braking_intensity
        row.append(abs(head_yaw))                  # head_deviation_abs
        row.append(1.0 - ear)                      # eye_risk
        row.append(abs(steering))                  # steering_abs
        rows.append(row)

    X = np.array(rows, dtype=np.float32)
    X = scaler.transform(X)

    return X


# ── CLI ───────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_dataset()
    X_tr, X_v, y_tr, y_v, sc = preprocess_for_training(df)
    print(f"\n✓ Preprocessing complete.")
    print(f"  X_train: {X_tr.shape}  y_train: {y_tr.shape}")
    print(f"  X_val:   {X_v.shape}   y_val:   {y_v.shape}")
