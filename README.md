# 🚗 DriveGuard AI — Driver Behaviour, Fatigue & Risk Detection System

> **AI/ML Technologies**: CNN, LSTM, TensorFlow, OpenCV, MediaPipe  
> **Stack**: Python (ML Backend) + Flask API + HTML/CSS/JS Dashboard

---

## 🎯 Project Overview

DriveGuard AI is an end-to-end system that **learns driving behaviour** from multimodal sensor data (camera + vehicle) and **identifies potentially unsafe situations** in real-time. The system classifies driver behaviour into 6 categories, computes a dynamic risk score (0–100), and generates alerts when thresholds are exceeded.

### Architecture

```
Camera + Vehicle Sensors
        ↓
  Feature Extraction
        ↓
    CNN / LSTM
        ↓
  Driver Behavior Classification
        ↓
    Risk Score (0–100)
```

### Behaviour Classes

| Class               | Description                                    |
|---------------------|------------------------------------------------|
| ✅ Normal            | Safe driving within all parameters             |
| 🔴 Aggressive        | High speed, hard steering, rapid acceleration  |
| 🟡 Distracted        | Head turned, lane drift, phone usage detected  |
| 🟣 Drowsy            | Low eye closure ratio (EAR), slow reactions    |
| 🟠 Unsafe Braking    | Sudden hard stops, high deceleration force     |
| 🟠 Unsafe Accel.     | Sudden strong forward thrust                   |

### Input Parameters (8 Sensors)

| Parameter       | Unit      | Source           |
|-----------------|-----------|------------------|
| Speed           | km/h      | Vehicle sensor   |
| Acceleration    | m/s²      | IMU              |
| Braking         | m/s²      | IMU              |
| Steering Angle  | degrees   | Steering sensor  |
| Lane Deviation  | meters    | Lane sensor      |
| Eye Closure     | ratio 0–1 | Camera (EAR)     |
| Head Position   | degrees   | Camera (yaw)     |
| Phone Usage     | binary    | Camera (detect)  |

---

## 📁 Project Structure

```
automotive mod 3 project/
│
├── config.py              # Central configuration (all hyperparameters)
├── data_generator.py      # Synthetic driving data generator (48K samples)
├── preprocessing.py       # Feature engineering, scaling, sequencing
├── model.py               # CNN + LSTM model architecture (TensorFlow)
├── train.py               # Full training pipeline + evaluation
├── inference.py            # Real-time inference engine + risk scoring
├── camera_detector.py     # OpenCV + MediaPipe facial detection (EAR, head pose)
├── api.py                  # Flask REST API + WebSocket server
│
├── index.html              # Dashboard UI (6 sections)
├── style.css               # Dark cyberpunk design system
├── app.js                  # Frontend logic + Socket.IO client
│
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
├── data/                   # Generated datasets (auto-created)
│   └── driving_dataset.csv
│
├── saved_models/           # Trained models (auto-created)
│   ├── driveguard_cnn_lstm.keras
│   ├── scaler.pkl
│   ├── metrics.json
│   └── training_history.pkl
│
└── logs/                   # TensorBoard logs (auto-created)
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Data + Train Model

```bash
python train.py
```

This will:
- Generate 48,000 synthetic driving samples (8,000 per class)
- Engineer 5 additional features (13 total)
- Create sliding-window sequences (length=30, 50% overlap)
- Build and train the CNN-LSTM model for 120 epochs
- Evaluate on validation set and save metrics

**Expected output:**
```
✓ Training Complete!
✓ Best val acc    : ~94%
✓ Model saved     : saved_models/driveguard_cnn_lstm.keras
```

### 3. Run the Full System

```bash
python api.py
```

Open your browser at **http://localhost:5000** — the dashboard will connect to the backend via WebSocket and display real-time CNN-LSTM predictions.

### 4. Standalone Mode (No Python)

Simply open `index.html` directly in a browser — the dashboard runs with built-in JavaScript simulation when no server is detected.

---

## 🧠 Model Architecture

```
┌────────────────────────────────────────────────────────────┐
│  Input: (batch, 30, 13) — 30-frame sequences, 13 features │
│                                                            │
│  ┌─── CNN Branch (Spatial Features) ───────────┐           │
│  │  Conv1D(32, kernel=3) → BatchNorm → ReLU    │           │
│  │  Conv1D(64, kernel=3) → BatchNorm → ReLU    │           │
│  │  MaxPool1D(2) → Dropout(0.25)               │           │
│  └─────────────────────────────────────────────┘           │
│                       ↓                                    │
│  ┌─── LSTM Branch (Temporal) ──────────────────┐           │
│  │  LSTM(128, return_sequences=True, drop=0.3)  │           │
│  │  LSTM(64, return_sequences=False, drop=0.3)  │           │
│  └─────────────────────────────────────────────┘           │
│                       ↓                                    │
│  Dense(128) → BatchNorm → ReLU → Dropout(0.4)             │
│  Dense(6, softmax) → [Normal, Aggr, Distr, Drowsy, UB, UA]│
└────────────────────────────────────────────────────────────┘
```

**Total Parameters**: ~2.4M  
**Training**: Adam optimizer (lr=1e-4), sparse categorical crossentropy  
**Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

---

## 🌐 API Endpoints

| Endpoint                | Method | Description                                |
|-------------------------|--------|--------------------------------------------|
| `/`                     | GET    | Dashboard UI                               |
| `/api/status`           | GET    | System status + model info                 |
| `/api/predict`          | POST   | Single frame prediction                    |
| `/api/predict/batch`    | POST   | Batch prediction (multiple frames)         |
| `/api/session`          | GET    | Current session summary                    |
| `/api/risk-history`     | GET    | Risk score history                         |
| `/api/metrics`          | GET    | Model training metrics + confusion matrix  |
| `/api/config`           | GET    | System configuration                       |
| `/api/thresholds`       | POST   | Update alert thresholds                    |

### WebSocket Events

| Event              | Direction      | Description                         |
|--------------------|----------------|-------------------------------------|
| `sensor_update`    | Server → Client | Real-time prediction broadcast     |
| `sensor_frame`     | Client → Server | Submit custom sensor data           |
| `prediction`       | Server → Client | Response to sensor_frame            |
| `update_thresholds`| Client → Server | Change alert thresholds             |

### Example: Single Frame Prediction

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "speed": 72,
    "acceleration": 0.8,
    "braking": 0.2,
    "steering_angle": 8,
    "lane_deviation": 0.12,
    "eye_aspect_ratio": 0.82,
    "head_yaw": 2.1,
    "phone_usage": 0
  }'
```

---

## 📊 Dashboard Sections

| Section        | Features                                                       |
|----------------|----------------------------------------------------------------|
| **Dashboard**  | Risk gauge, CNN/LSTM class probabilities, telemetry chart, 8 sparklines |
| **Camera**     | Face tracking SVG, EAR gauge, head pose visualization, detection log |
| **Sensors**    | 6 arc gauges, multi-axis radar chart vs safe thresholds        |
| **History**    | Risk score timeline, incident events table                     |
| **AI Model**   | Architecture diagram, accuracy bars, confusion matrix, stats   |
| **Alerts**     | Live alert feed, configurable thresholds, channel toggles      |

---

## 📷 Camera Detection Module

When a webcam is available, `camera_detector.py` provides:

- **Face Detection** via MediaPipe Face Mesh (468 landmarks)
- **Eye Aspect Ratio (EAR)** — measures eye closure for drowsiness
- **Head Pose Estimation** — PnP solve for yaw/pitch/roll
- **Phone Detection** — face occlusion heuristic

```bash
# Test camera independently
python camera_detector.py
```

---

## ⚙️ Configuration

All hyperparameters are in `config.py`:

- **Model**: CNN filters, LSTM units, dropout, learning rate
- **Data**: samples per class, sequence length, noise level
- **Risk**: per-class risk weights, parameter threshold rules
- **Alerts**: EAR threshold, speed limit, lane deviation limit

---

## 📈 Training Plots

After training, find in `saved_models/`:
- `training_plot.png` — Accuracy & loss curves
- `confusion_matrix.png` — Heatmap visualization

```bash
# Generate plots from saved history
python train.py --plot
```

---

## 🛠️ Development

### Regenerate Dataset
```bash
python train.py --regenerate
```

### Run Data Generator Only
```bash
python data_generator.py
```

### Test Inference Engine
```bash
python inference.py
```

### Test Model Architecture
```bash
python model.py
```
