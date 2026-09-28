# ============================================================
# DriveGuard AI — Camera-Based Detection Module
# ============================================================
# Uses OpenCV + MediaPipe for real-time facial landmark
# detection, eye aspect ratio (EAR), head pose estimation,
# and phone/object detection from webcam feed.
# ============================================================

import numpy as np
import time

try:
    import cv2
    import mediapipe as mp
    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False
    print("⚠ OpenCV/MediaPipe not available. Camera features disabled.")

import config


# ── Eye Aspect Ratio Computation ──────────────────────────────
# Using the 6 facial landmarks around each eye from MediaPipe.

# MediaPipe Face Mesh indices for eyes (468 landmarks)
LEFT_EYE_INDICES  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33,  160, 158, 133, 153, 144]

# Head pose estimation landmark indices
NOSE_TIP    = 1
CHIN        = 199
LEFT_EAR_PT = 454
RIGHT_EAR_PT = 234
LEFT_EYE_CT = 468   # approximate
RIGHT_EYE_CT = 473  # approximate
FOREHEAD    = 10


def compute_ear(eye_landmarks):
    """
    Compute Eye Aspect Ratio (EAR).

    EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)

    Args:
        eye_landmarks: 6 (x, y) points around the eye.

    Returns:
        float: EAR value. High = open, Low = closed.
    """
    p = np.array(eye_landmarks)

    # Vertical distances
    v1 = np.linalg.norm(p[1] - p[5])
    v2 = np.linalg.norm(p[2] - p[4])

    # Horizontal distance
    h = np.linalg.norm(p[0] - p[3])

    if h < 1e-6:
        return 0.0

    ear = (v1 + v2) / (2.0 * h)
    return float(ear)


def compute_head_pose(landmarks, frame_shape):
    """
    Estimate head yaw and pitch from facial landmarks.

    Uses PnP solve with a generic 3D face model.

    Returns:
        (yaw_degrees, pitch_degrees, roll_degrees)
    """
    h, w = frame_shape[:2]

    # 2D image points from detected landmarks
    image_points = np.array([
        [landmarks[NOSE_TIP].x * w,     landmarks[NOSE_TIP].y * h],
        [landmarks[CHIN].x * w,         landmarks[CHIN].y * h],
        [landmarks[LEFT_EAR_PT].x * w,  landmarks[LEFT_EAR_PT].y * h],
        [landmarks[RIGHT_EAR_PT].x * w, landmarks[RIGHT_EAR_PT].y * h],
        [landmarks[33].x * w,           landmarks[33].y * h],   # left eye
        [landmarks[263].x * w,          landmarks[263].y * h],  # right eye
    ], dtype="double")

    # 3D model points (generic face model)
    model_points = np.array([
        (0.0,    0.0,   0.0),     # Nose tip
        (0.0,   -63.6, -12.5),    # Chin
        (-43.3,  32.7, -26.0),    # Left ear
        (43.3,   32.7, -26.0),    # Right ear
        (-28.9,  28.9, -24.1),    # Left eye
        (28.9,   28.9, -24.1),    # Right eye
    ])

    # Camera internals (approximate)
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0,            center[0]],
        [0,            focal_length, center[1]],
        [0,            0,            1],
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return 0.0, 0.0, 0.0

    # Convert rotation vector to rotation matrix → Euler angles
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    proj_matrix = np.hstack((rotation_matrix, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(
        np.vstack((proj_matrix, [0, 0, 0, 1]))[:3]
    )

    pitch = float(euler_angles[0])
    yaw   = float(euler_angles[1])
    roll  = float(euler_angles[2])

    return yaw, pitch, roll


class CameraDetector:
    """
    Real-time camera-based driver monitoring.

    Captures webcam feed and extracts:
      - Face detection confidence
      - Eye Aspect Ratio (EAR) → drowsiness
      - Head pose (yaw, pitch) → distraction
      - Basic phone/object detection via face occlusion
    """

    def __init__(self, camera_index=0):
        if not CAMERA_AVAILABLE:
            self.active = False
            return

        self.active = True
        self.camera_index = camera_index
        self.cap = None
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.ear_buffer = []
        self.ear_avg_window = 5
        self.consecutive_drowsy_frames = 0
        self.drowsy_threshold = config.DEFAULT_THRESHOLDS["ear_drowsy"]

    def start(self):
        """Start camera capture."""
        if not self.active:
            return False
        self.cap = cv2.VideoCapture(self.camera_index)
        return self.cap.isOpened()

    def stop(self):
        """Stop camera capture."""
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def process_frame(self):
        """
        Capture and process a single camera frame.

        Returns:
            dict with detection results:
                {
                    "face_detected": True,
                    "face_confidence": 0.99,
                    "eye_aspect_ratio": 0.82,
                    "left_ear": 0.83,
                    "right_ear": 0.81,
                    "head_yaw": 2.1,
                    "head_pitch": -1.3,
                    "is_drowsy": False,
                    "drowsy_frames": 0,
                    "phone_detected": False,
                    "fps": 30,
                }
        """
        if not self.active or not self.cap or not self.cap.isOpened():
            return self._default_result()

        ret, frame = self.cap.read()
        if not ret:
            return self._default_result()

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        result = self._default_result()

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            result["face_detected"] = True
            result["face_confidence"] = 0.95  # MediaPipe doesn't expose this directly

            # ── EAR Computation ──
            left_eye_pts  = [(landmarks[i].x, landmarks[i].y) for i in LEFT_EYE_INDICES]
            right_eye_pts = [(landmarks[i].x, landmarks[i].y) for i in RIGHT_EYE_INDICES]

            left_ear  = compute_ear(left_eye_pts)
            right_ear = compute_ear(right_eye_pts)
            avg_ear   = (left_ear + right_ear) / 2.0

            # Smooth EAR
            self.ear_buffer.append(avg_ear)
            if len(self.ear_buffer) > self.ear_avg_window:
                self.ear_buffer.pop(0)
            smoothed_ear = np.mean(self.ear_buffer)

            result["left_ear"]  = round(left_ear, 4)
            result["right_ear"] = round(right_ear, 4)
            result["eye_aspect_ratio"] = round(smoothed_ear, 4)

            # ── Drowsiness Detection ──
            if smoothed_ear < self.drowsy_threshold:
                self.consecutive_drowsy_frames += 1
            else:
                self.consecutive_drowsy_frames = 0

            result["is_drowsy"] = self.consecutive_drowsy_frames > 10  # sustained closure
            result["drowsy_frames"] = self.consecutive_drowsy_frames

            # ── Head Pose ──
            try:
                yaw, pitch, roll = compute_head_pose(landmarks, frame.shape)
                result["head_yaw"]   = round(yaw, 2)
                result["head_pitch"] = round(pitch, 2)
            except Exception:
                pass

            # ── Phone Detection (simple heuristic) ──
            # If face is partially occluded (landmarks at unusual positions)
            nose = landmarks[NOSE_TIP]
            if nose.x < 0.2 or nose.x > 0.8:
                result["phone_detected"] = True

        return result

    def _default_result(self):
        """Default result when no face is detected."""
        return {
            "face_detected": False,
            "face_confidence": 0.0,
            "eye_aspect_ratio": 0.0,
            "left_ear": 0.0,
            "right_ear": 0.0,
            "head_yaw": 0.0,
            "head_pitch": 0.0,
            "is_drowsy": False,
            "drowsy_frames": 0,
            "phone_detected": False,
        }

    def __del__(self):
        self.stop()


# ── CLI Test ──────────────────────────────────────────────────
if __name__ == "__main__":
    if not CAMERA_AVAILABLE:
        print("Camera libraries not installed. Run: pip install opencv-python mediapipe")
        exit(1)

    detector = CameraDetector()
    if not detector.start():
        print("Failed to open camera.")
        exit(1)

    print("Camera detection running. Press Ctrl+C to stop.\n")
    try:
        while True:
            result = detector.process_frame()
            if result["face_detected"]:
                print(f"  EAR: {result['eye_aspect_ratio']:.4f}  "
                      f"Yaw: {result['head_yaw']:+6.1f}°  "
                      f"Pitch: {result['head_pitch']:+6.1f}°  "
                      f"Drowsy: {result['is_drowsy']}  "
                      f"Phone: {result['phone_detected']}")
            else:
                print("  No face detected")
            time.sleep(0.033)  # ~30 FPS
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        detector.stop()
