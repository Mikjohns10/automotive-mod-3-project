# ============================================================
# DriveGuard AI — System Configuration
# ============================================================

import os

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(BASE_DIR, "data")
MODEL_DIR      = os.path.join(BASE_DIR, "saved_models")
LOG_DIR        = os.path.join(BASE_DIR, "logs")
SCALER_PATH    = os.path.join(MODEL_DIR, "scaler.pkl")
MODEL_PATH     = os.path.join(MODEL_DIR, "driveguard_cnn_lstm.keras")
HISTORY_PATH   = os.path.join(MODEL_DIR, "training_history.pkl")

# Ensure directories exist
for d in [DATA_DIR, MODEL_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Sensor Parameters ─────────────────────────────────────────
# The 8 input features captured from camera + vehicle sensors
FEATURES = [
    "speed",            # km/h       — Vehicle speed
    "acceleration",     # m/s²       — Longitudinal acceleration
    "braking",          # m/s²       — Braking deceleration force
    "steering_angle",   # degrees    — Steering wheel angle
    "lane_deviation",   # meters     — Deviation from lane center
    "eye_aspect_ratio", # ratio 0–1  — Eye closure (EAR from camera)
    "head_yaw",         # degrees    — Head yaw angle (camera)
    "phone_usage",      # binary 0/1 — Phone detected near face
]

NUM_FEATURES = len(FEATURES)

# ── Behaviour Classes ─────────────────────────────────────────
CLASSES = [
    "Normal",
    "Aggressive",
    "Distracted",
    "Drowsy",
    "Unsafe Braking",
    "Unsafe Acceleration",
]

NUM_CLASSES = len(CLASSES)

# Class-to-index mapping
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(CLASSES)}

# ── Data Generation Parameters ────────────────────────────────
SAMPLES_PER_CLASS     = 8000        # Samples per class to generate
SEQUENCE_LENGTH       = 30          # Frames per sequence (for LSTM)
TRAIN_SPLIT           = 0.8        # Train / validation split
NOISE_LEVEL           = 0.05       # Gaussian noise std added to data

# ── Model Hyperparameters ─────────────────────────────────────
# CNN branch
CNN_FILTERS_1         = 32
CNN_FILTERS_2         = 64
CNN_KERNEL_SIZE       = 3
CNN_POOL_SIZE         = 2

# LSTM branch
LSTM_UNITS_1          = 128
LSTM_UNITS_2          = 64
LSTM_DROPOUT          = 0.3

# Dense head
DENSE_UNITS           = 128
DENSE_DROPOUT         = 0.4

# Training
LEARNING_RATE         = 1e-4
BATCH_SIZE            = 64
EPOCHS                = 120
EARLY_STOP_PATIENCE   = 15
REDUCE_LR_PATIENCE    = 7
REDUCE_LR_FACTOR      = 0.5

# ── Risk Score Weights ────────────────────────────────────────
# Each class contributes a base risk when detected as primary
CLASS_RISK_WEIGHTS = {
    "Normal":              5,
    "Aggressive":          55,
    "Distracted":          60,
    "Drowsy":              75,
    "Unsafe Braking":      65,
    "Unsafe Acceleration": 50,
}

# Parameter-specific risk contributions (additive)
PARAM_RISK_RULES = {
    "speed":            {"threshold": 100, "weight": 0.3},    # above 100 km/h
    "acceleration":     {"threshold": 4.0, "weight": 5.0},    # above 4 m/s²
    "braking":          {"threshold": 4.0, "weight": 6.0},    # above 4 m/s²
    "eye_aspect_ratio": {"threshold": 0.25, "weight": 150, "inverted": True},  # below 0.25
    "lane_deviation":   {"threshold": 0.30, "weight": 40},    # above 0.30m
    "head_yaw":         {"threshold": 20,   "weight": 1.5, "absolute": True},  # |yaw| > 20°
    "phone_usage":      {"threshold": 0.5,  "weight": 35},    # phone present
}

# ── Server Configuration ──────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 5000
CORS_ORIGINS = "*"
# Auto-detect async mode: use gevent in production (Docker/Render), threading locally
try:
    import gevent          # noqa: F401
    SOCKETIO_ASYNC_MODE = "gevent"
except ImportError:
    SOCKETIO_ASYNC_MODE = "threading"

# ── Alert Thresholds (defaults) ───────────────────────────────
DEFAULT_THRESHOLDS = {
    "ear_drowsy":      0.25,
    "max_speed":       120,     # km/h
    "max_accel":       5.0,     # m/s²
    "max_lane_dev":    0.30,    # meters
    "risk_alert":      60,      # score
}
