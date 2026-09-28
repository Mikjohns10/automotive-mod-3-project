# ============================================================
# DriveGuard AI — Flask REST API + WebSocket Server
# ============================================================
# Serves the frontend dashboard and provides real-time
# inference via REST endpoints and Socket.IO events.
# ============================================================

import os
import json
import time
import threading
import numpy as np
from flask import Flask, jsonify, request, send_from_directory, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit

import config
from inference import DriverBehaviorClassifier

# ── App Setup ─────────────────────────────────────────────────
app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app, origins=config.CORS_ORIGINS)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode=config.SOCKETIO_ASYNC_MODE)

# ── Initialize Classifier ────────────────────────────────────
classifier = DriverBehaviorClassifier()

# ── Simulation Thread ─────────────────────────────────────────
# Generates simulated sensor data when no real sensors are connected.
simulation_active = True
simulation_interval = 0.3  # seconds


def simulate_driving():
    """Background thread that simulates driving sensor data."""
    np.random.seed(int(time.time()) % 10000)

    # State variables with smooth transitions
    speed     = 65.0
    accel     = 0.5
    braking   = 0.2
    steering  = 0.0
    lane_dev  = 0.05
    ear       = 0.78
    head_yaw  = 0.0
    phone     = 0.0

    tick = 0

    while simulation_active:
        tick += 1

        # Smooth random walk
        speed    = np.clip(speed    + np.random.normal(0, 2.0),   0, 145)
        accel    = np.clip(accel    + np.random.normal(0, 0.15),  0, 8)
        braking  = np.clip(braking  + np.random.normal(0, 0.1),   0, 7)
        steering = np.clip(steering + np.random.normal(0, 1.0),  -40, 40)
        lane_dev = np.clip(lane_dev + np.random.normal(0, 0.02),   0, 0.8)
        ear      = np.clip(ear      + np.random.normal(0, 0.015),  0.12, 1.0)
        head_yaw = np.clip(head_yaw + np.random.normal(0, 0.6),  -40, 40)

        # Occasional events (every ~40 ticks on average)
        if np.random.random() < 0.025:
            event = np.random.choice(["drowsy", "phone", "brake", "accel", "lane"])
            if event == "drowsy":
                ear = np.random.uniform(0.12, 0.22)
            elif event == "phone":
                phone = 1.0
            elif event == "brake":
                braking = np.random.uniform(4.5, 7)
            elif event == "accel":
                accel = np.random.uniform(5, 8)
            elif event == "lane":
                lane_dev = np.random.uniform(0.35, 0.6)

        # Reset phone after some ticks
        if phone > 0 and np.random.random() < 0.15:
            phone = 0.0

        # Build sensor frame
        sensor_data = {
            "speed":            round(float(speed), 2),
            "acceleration":     round(float(accel), 2),
            "braking":          round(float(braking), 2),
            "steering_angle":   round(float(steering), 2),
            "lane_deviation":   round(float(lane_dev), 3),
            "eye_aspect_ratio": round(float(ear), 4),
            "head_yaw":         round(float(head_yaw), 2),
            "phone_usage":      float(phone),
        }

        # Run through classifier
        result = classifier.process_frame(sensor_data)

        # Broadcast via WebSocket
        socketio.emit("sensor_update", result)

        time.sleep(simulation_interval)


# ── Static File Serving ───────────────────────────────────────
@app.route("/")
def serve_index():
    """Serve the main dashboard."""
    return send_file(os.path.join(config.BASE_DIR, "index.html"))


@app.route("/<path:filename>")
def serve_static(filename):
    """Serve static files (CSS, JS, images)."""
    return send_from_directory(config.BASE_DIR, filename)


# ── REST API Endpoints ────────────────────────────────────────

@app.route("/api/status")
def api_status():
    """System status endpoint."""
    model_exists = os.path.exists(config.MODEL_PATH)
    return jsonify({
        "status": "running",
        "model_loaded": classifier.model is not None,
        "model_path": config.MODEL_PATH if model_exists else None,
        "scaler_loaded": classifier.scaler is not None,
        "classes": config.CLASSES,
        "features": config.FEATURES,
        "sequence_length": config.SEQUENCE_LENGTH,
        "frame_count": classifier.frame_count,
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Single-frame prediction endpoint.

    POST JSON body:
    {
        "speed": 72,
        "acceleration": 0.8,
        "braking": 0.2,
        "steering_angle": 8,
        "lane_deviation": 0.12,
        "eye_aspect_ratio": 0.82,
        "head_yaw": 2.1,
        "phone_usage": 0
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    result = classifier.process_frame(data)
    return jsonify(result)


@app.route("/api/predict/batch", methods=["POST"])
def api_predict_batch():
    """
    Batch prediction — process multiple frames at once.

    POST JSON body: { "frames": [ {...}, {...}, ... ] }
    """
    data = request.get_json()
    frames = data.get("frames", [])
    if not frames:
        return jsonify({"error": "No frames provided"}), 400

    results = []
    for frame in frames:
        result = classifier.process_frame(frame)
        results.append(result)

    return jsonify({"results": results, "count": len(results)})


@app.route("/api/session")
def api_session():
    """Get current session summary."""
    return jsonify(classifier.get_session_summary())


@app.route("/api/risk-history")
def api_risk_history():
    """Get risk score history."""
    return jsonify({
        "risk_history": list(classifier.risk_history),
        "count": len(classifier.risk_history),
    })


@app.route("/api/metrics")
def api_metrics():
    """Get trained model metrics (if available)."""
    metrics_path = os.path.join(config.MODEL_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            return jsonify(json.load(f))
    return jsonify({"error": "No metrics found. Train the model first."}), 404


@app.route("/api/config")
def api_config():
    """Get system configuration."""
    return jsonify({
        "features": config.FEATURES,
        "classes": config.CLASSES,
        "num_features": config.NUM_FEATURES,
        "num_classes": config.NUM_CLASSES,
        "sequence_length": config.SEQUENCE_LENGTH,
        "risk_weights": config.CLASS_RISK_WEIGHTS,
        "param_risk_rules": {k: {kk: vv for kk, vv in v.items()} for k, v in config.PARAM_RISK_RULES.items()},
        "default_thresholds": config.DEFAULT_THRESHOLDS,
    })


@app.route("/api/thresholds", methods=["POST"])
def api_update_thresholds():
    """Update alert thresholds dynamically."""
    data = request.get_json()
    for key, value in data.items():
        if key in config.DEFAULT_THRESHOLDS:
            config.DEFAULT_THRESHOLDS[key] = float(value)
    return jsonify({"status": "updated", "thresholds": config.DEFAULT_THRESHOLDS})


# ── WebSocket Events ──────────────────────────────────────────

@socketio.on("connect")
def handle_connect():
    """Client connected."""
    print(f"✓ Client connected")
    emit("status", {
        "message": "Connected to DriveGuard AI",
        "model_loaded": classifier.model is not None,
    })


@socketio.on("disconnect")
def handle_disconnect():
    print(f"  Client disconnected")


@socketio.on("sensor_frame")
def handle_sensor_frame(data):
    """
    Receive a sensor frame from the client and return prediction.
    Used when frontend sends its own simulated data.
    """
    result = classifier.process_frame(data)
    emit("prediction", result)


@socketio.on("update_thresholds")
def handle_update_thresholds(data):
    """Update thresholds via WebSocket."""
    for key, value in data.items():
        if key in config.DEFAULT_THRESHOLDS:
            config.DEFAULT_THRESHOLDS[key] = float(value)
    emit("thresholds_updated", config.DEFAULT_THRESHOLDS)


# ── Main Entry Point ──────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  DriveGuard AI — Server Starting")
    print("=" * 60)
    print(f"  Model loaded  : {classifier.model is not None}")
    print(f"  Scaler loaded : {classifier.scaler is not None}")
    print(f"  Features      : {config.NUM_FEATURES}")
    print(f"  Classes       : {config.NUM_CLASSES}")
    print(f"  Dashboard     : http://localhost:{config.API_PORT}")
    print(f"  API Base      : http://localhost:{config.API_PORT}/api")
    print("=" * 60)

    # Start simulation thread
    sim_thread = threading.Thread(target=simulate_driving, daemon=True)
    sim_thread.start()
    print("  ✓ Simulation thread started")

    # Run server
    socketio.run(
        app,
        host=config.API_HOST,
        port=config.API_PORT,
        debug=False,
        allow_unsafe_werkzeug=True,
    )
