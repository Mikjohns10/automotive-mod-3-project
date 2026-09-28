# ============================================================
# DriveGuard AI — Real-Time Inference Engine
# ============================================================
# Handles model loading, sequence buffering, prediction,
# risk score computation, and alert generation.
# ============================================================

import numpy as np
import os
import json
import time
import joblib
from collections import deque

import config


class DriverBehaviorClassifier:
    """
    Real-time driver behaviour classifier.

    Maintains a sliding window of sensor frames and performs
    CNN-LSTM inference to classify behaviour + compute risk.
    """

    def __init__(self, model_path=None, scaler_path=None):
        self.model = None
        self.scaler = None
        self.sequence_buffer = deque(maxlen=config.SEQUENCE_LENGTH)
        self.prediction_history = deque(maxlen=120)   # last 2 minutes
        self.risk_history = deque(maxlen=600)          # last 10 minutes
        self.alert_history = []
        self.frame_count = 0
        self.last_class = "Normal"
        self.last_confidence = 1.0
        self.risk_score = 0.0
        self._load_model(model_path, scaler_path)

    def _load_model(self, model_path=None, scaler_path=None):
        """Load trained model and scaler."""
        mp = model_path or config.MODEL_PATH
        sp = scaler_path or config.SCALER_PATH

        if os.path.exists(mp):
            import tensorflow as tf
            self.model = tf.keras.models.load_model(mp)
            print(f"✓ Model loaded from {mp}")
        else:
            print(f"⚠ Model not found at {mp}. Using rule-based fallback.")

        if os.path.exists(sp):
            self.scaler = joblib.load(sp)
            print(f"✓ Scaler loaded from {sp}")
        else:
            print(f"⚠ Scaler not found at {sp}. No scaling applied.")

    def _engineer_features(self, raw):
        """Compute 5 engineered features from raw 8."""
        speed = raw.get("speed", 0)
        accel = raw.get("acceleration", 0)
        braking = raw.get("braking", 0)
        head_yaw = raw.get("head_yaw", 0)
        ear = raw.get("eye_aspect_ratio", 0.7)
        steering = raw.get("steering_angle", 0)

        return {
            "speed_accel_ratio":  speed / (accel + 1.0),
            "braking_intensity":  braking * speed / 100.0,
            "head_deviation_abs": abs(head_yaw),
            "eye_risk":           1.0 - ear,
            "steering_abs":       abs(steering),
        }

    def _to_feature_vector(self, raw):
        """Convert a raw sensor dict to a 13-element feature vector."""
        eng = self._engineer_features(raw)
        vec = [raw.get(f, 0.0) for f in config.FEATURES]
        vec += [
            eng["speed_accel_ratio"],
            eng["braking_intensity"],
            eng["head_deviation_abs"],
            eng["eye_risk"],
            eng["steering_abs"],
        ]
        return vec

    def process_frame(self, sensor_data):
        """
        Process a single sensor frame.

        Args:
            sensor_data: dict with keys matching config.FEATURES
                {
                    "speed": 72,
                    "acceleration": 0.8,
                    "braking": 0.2,
                    "steering_angle": 8,
                    "lane_deviation": 0.12,
                    "eye_aspect_ratio": 0.82,
                    "head_yaw": 2.1,
                    "phone_usage": 0,
                }

        Returns:
            dict with prediction results:
                {
                    "predicted_class": "Normal",
                    "confidence": 0.942,
                    "all_probabilities": {...},
                    "risk_score": 28,
                    "risk_level": "LOW",
                    "alerts": [...],
                    "parameters": {...},
                }
        """
        self.frame_count += 1
        feature_vec = self._to_feature_vector(sensor_data)
        self.sequence_buffer.append(feature_vec)

        # Default result (before buffer is full)
        result = {
            "predicted_class": "Normal",
            "confidence": 1.0,
            "all_probabilities": {c: 0.0 for c in config.CLASSES},
            "risk_score": 0,
            "risk_level": "LOW",
            "alerts": [],
            "parameters": sensor_data,
            "frame": self.frame_count,
            "buffer_fill": len(self.sequence_buffer),
            "buffer_required": config.SEQUENCE_LENGTH,
        }
        result["all_probabilities"]["Normal"] = 1.0

        # Need full sequence before CNN-LSTM inference
        if len(self.sequence_buffer) < config.SEQUENCE_LENGTH:
            return result

        # ── Model Inference ──
        sequence = np.array(list(self.sequence_buffer), dtype=np.float32)

        if self.scaler is not None:
            sequence = self.scaler.transform(sequence)

        if self.model is not None:
            # CNN-LSTM prediction
            input_batch = np.expand_dims(sequence, axis=0)  # (1, seq_len, features)
            probabilities = self.model.predict(input_batch, verbose=0)[0]
            predicted_idx = int(np.argmax(probabilities))
            predicted_class = config.IDX_TO_CLASS[predicted_idx]
            confidence = float(probabilities[predicted_idx])

            result["all_probabilities"] = {
                config.IDX_TO_CLASS[i]: float(probabilities[i])
                for i in range(config.NUM_CLASSES)
            }
        else:
            # Rule-based fallback (when model isn't trained yet)
            predicted_class, confidence, probs = self._rule_based_classify(sensor_data)
            result["all_probabilities"] = probs

        result["predicted_class"] = predicted_class
        result["confidence"] = confidence
        self.last_class = predicted_class
        self.last_confidence = confidence

        # ── Risk Score ──
        risk = self._compute_risk(sensor_data, predicted_class, confidence)
        result["risk_score"] = risk
        result["risk_level"] = (
            "LOW" if risk < 30 else
            "MODERATE" if risk < 65 else
            "HIGH"
        )
        self.risk_score = risk
        self.risk_history.append(risk)

        # ── Alert Generation ──
        alerts = self._generate_alerts(sensor_data, predicted_class, risk)
        result["alerts"] = alerts

        # Save to prediction history
        self.prediction_history.append({
            "time": time.time(),
            "class": predicted_class,
            "confidence": confidence,
            "risk": risk,
        })

        return result

    def _rule_based_classify(self, data):
        """Fallback classification using threshold rules (no ML model)."""
        speed   = data.get("speed", 0)
        accel   = data.get("acceleration", 0)
        braking = data.get("braking", 0)
        ear     = data.get("eye_aspect_ratio", 0.7)
        head    = data.get("head_yaw", 0)
        lane    = data.get("lane_deviation", 0)
        phone   = data.get("phone_usage", 0)

        # Priority-based rules
        if ear < 0.25:
            cls, conf = "Drowsy", 0.88 + np.random.uniform(0, 0.08)
        elif phone > 0.5:
            cls, conf = "Distracted", 0.85 + np.random.uniform(0, 0.10)
        elif braking > 4:
            cls, conf = "Unsafe Braking", 0.82 + np.random.uniform(0, 0.12)
        elif accel > 4.5:
            cls, conf = "Unsafe Acceleration", 0.80 + np.random.uniform(0, 0.12)
        elif speed > 100 or accel > 3:
            cls, conf = "Aggressive", 0.78 + np.random.uniform(0, 0.15)
        elif abs(head) > 22 or lane > 0.3:
            cls, conf = "Distracted", 0.75 + np.random.uniform(0, 0.15)
        else:
            cls, conf = "Normal", 0.90 + np.random.uniform(0, 0.08)

        # Distribute remaining probability
        probs = {}
        remaining = 1.0 - conf
        for c in config.CLASSES:
            if c == cls:
                probs[c] = conf
            else:
                probs[c] = remaining / (config.NUM_CLASSES - 1)

        return cls, conf, probs

    def _compute_risk(self, data, predicted_class, confidence):
        """
        Compute risk score (0–100) from:
          - Base class risk weight
          - Parameter-specific threshold violations
          - Confidence weighting
        """
        # Base risk from predicted class
        base_risk = config.CLASS_RISK_WEIGHTS.get(predicted_class, 10)
        risk = base_risk * confidence

        # Add parameter-specific risks
        for param, rules in config.PARAM_RISK_RULES.items():
            value = data.get(param, 0)
            threshold = rules["threshold"]
            weight = rules["weight"]

            if rules.get("absolute"):
                value = abs(value)

            if rules.get("inverted"):
                # Alert when BELOW threshold (e.g., EAR)
                if value < threshold:
                    risk += (threshold - value) * weight
            else:
                # Alert when ABOVE threshold
                if value > threshold:
                    risk += (value - threshold) * weight

        return min(100, max(0, round(risk, 1)))

    def _generate_alerts(self, data, predicted_class, risk):
        """Generate alert messages based on current state."""
        alerts = []

        ear = data.get("eye_aspect_ratio", 0.7)
        speed = data.get("speed", 0)
        accel = data.get("acceleration", 0)
        braking = data.get("braking", 0)
        lane = data.get("lane_deviation", 0)
        phone = data.get("phone_usage", 0)

        if ear < config.DEFAULT_THRESHOLDS["ear_drowsy"]:
            alerts.append({
                "type": "critical",
                "title": "Drowsiness Alert",
                "message": f"Eye closure ratio (EAR={ear:.3f}) below threshold",
                "icon": "😴",
            })

        if speed > config.DEFAULT_THRESHOLDS["max_speed"]:
            alerts.append({
                "type": "warning",
                "title": "Speed Warning",
                "message": f"Speed {speed:.0f} km/h exceeds {config.DEFAULT_THRESHOLDS['max_speed']} km/h",
                "icon": "🚗",
            })

        if accel > config.DEFAULT_THRESHOLDS["max_accel"]:
            alerts.append({
                "type": "warning",
                "title": "Aggressive Acceleration",
                "message": f"Acceleration {accel:.1f} m/s² exceeds threshold",
                "icon": "⚡",
            })

        if braking > 4.0:
            alerts.append({
                "type": "critical",
                "title": "Hard Braking Detected",
                "message": f"Braking force {braking:.1f} m/s² — sudden stop detected",
                "icon": "🛑",
            })

        if lane > config.DEFAULT_THRESHOLDS["max_lane_dev"]:
            alerts.append({
                "type": "warning",
                "title": "Lane Departure Warning",
                "message": f"Lane deviation {lane:.2f}m exceeds {config.DEFAULT_THRESHOLDS['max_lane_dev']}m",
                "icon": "🛣️",
            })

        if phone > 0.5:
            alerts.append({
                "type": "critical",
                "title": "Phone Usage Detected",
                "message": "Device detected near driver face region",
                "icon": "📱",
            })

        if risk > config.DEFAULT_THRESHOLDS["risk_alert"]:
            alerts.append({
                "type": "critical",
                "title": "High Risk Alert",
                "message": f"Overall risk score {risk:.0f}/100 — exercise caution",
                "icon": "⚠️",
            })

        return alerts

    def get_session_summary(self):
        """Return a summary of the current driving session."""
        if not self.prediction_history:
            return {"status": "No data yet"}

        preds = list(self.prediction_history)
        risks = list(self.risk_history)

        class_counts = {}
        for p in preds:
            c = p["class"]
            class_counts[c] = class_counts.get(c, 0) + 1

        return {
            "total_frames": self.frame_count,
            "avg_risk": round(np.mean(risks), 2) if risks else 0,
            "max_risk": round(max(risks), 2) if risks else 0,
            "min_risk": round(min(risks), 2) if risks else 0,
            "class_distribution": class_counts,
            "total_alerts": len(self.alert_history),
            "current_class": self.last_class,
            "current_risk": self.risk_score,
        }


# ── CLI Test ──────────────────────────────────────────────────
if __name__ == "__main__":
    classifier = DriverBehaviorClassifier()

    # Simulate 60 frames of normal driving
    print("\n  Simulating 60 frames of sensor data...\n")
    for i in range(60):
        frame = {
            "speed":            65 + np.random.normal(0, 5),
            "acceleration":     0.5 + np.random.normal(0, 0.2),
            "braking":          0.2 + np.random.normal(0, 0.1),
            "steering_angle":   np.random.normal(0, 3),
            "lane_deviation":   0.05 + np.random.uniform(0, 0.1),
            "eye_aspect_ratio": 0.75 + np.random.normal(0, 0.05),
            "head_yaw":         np.random.normal(0, 3),
            "phone_usage":      0,
        }
        result = classifier.process_frame(frame)

        if i % 10 == 0:
            print(f"  Frame {i:3d}: Class={result['predicted_class']:>20s}  "
                  f"Conf={result['confidence']:.3f}  "
                  f"Risk={result['risk_score']:5.1f}  "
                  f"Level={result['risk_level']}")

    # Simulate drowsiness event
    print("\n  --- Simulating Drowsiness Event ---\n")
    for i in range(30):
        frame = {
            "speed":            55 + np.random.normal(0, 3),
            "acceleration":     0.3 + np.random.normal(0, 0.1),
            "braking":          0.1,
            "steering_angle":   np.random.normal(0, 4),
            "lane_deviation":   0.20 + np.random.uniform(0, 0.1),
            "eye_aspect_ratio": 0.18 + np.random.normal(0, 0.03),  # drowsy!
            "head_yaw":         np.random.normal(5, 6),
            "phone_usage":      0,
        }
        result = classifier.process_frame(frame)

        if i % 5 == 0:
            print(f"  Frame {60+i:3d}: Class={result['predicted_class']:>20s}  "
                  f"Conf={result['confidence']:.3f}  "
                  f"Risk={result['risk_score']:5.1f}  "
                  f"Level={result['risk_level']}")
            if result["alerts"]:
                for a in result["alerts"]:
                    print(f"             🔔 [{a['type'].upper()}] {a['title']}")

    print("\n  Session Summary:")
    summary = classifier.get_session_summary()
    for k, v in summary.items():
        print(f"    {k}: {v}")
