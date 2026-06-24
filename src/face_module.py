import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import logging
import os
import urllib.request
from src.config import settings

logger = logging.getLogger(__name__)

class FaceLivenessDetector:
    def __init__(self):
        """
        Initialize the MediaPipe Face Landmarker (Tasks API).
        State is externalized for thread safety.
        """
        model_path = settings.FACE_MODEL_PATH
        if not os.path.exists(model_path):
            logger.info(f"Downloading MediaPipe face_landmarker.task model to {model_path}...")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", 
                model_path
            )

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=2,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
        # EAR thresholds
        self.EAR_THRESHOLD = 0.21
        self.BLINK_CONSEC_FRAMES = 2
        
        # Eye landmarks indices
        self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

    def create_default_state(self):
        """Return a fresh state dictionary for temporal tracking."""
        return {
            "blink_counter": 0,
            "blink_total": 0,
            "window_start_time": time.time()
        }

    def _extract_roi(self, frame, face_landmarks) -> np.ndarray:
        h, w, _ = frame.shape
        x_coords = [int(lm.x * w) for lm in face_landmarks]
        y_coords = [int(lm.y * h) for lm in face_landmarks]
        pad = 20
        x_min, y_min = max(0, min(x_coords) - pad), max(0, min(y_coords) - pad)
        x_max, y_max = min(w, max(x_coords) + pad), min(h, max(y_coords) + pad)
        return frame[y_min:y_max, x_min:x_max]

    def _calculate_laplacian_variance(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _detect_moire_patterns(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        mag = 20 * np.log(np.abs(fshift) + 1e-8)
        rows, cols = gray.shape
        mag[rows//2-15:rows//2+15, cols//2-15:cols//2+15] = 0
        return np.var(mag)

    def _calculate_ear(self, face_landmarks, h, w) -> float:
        def pt(idx):
            lm = face_landmarks[idx]
            return np.array([lm.x * w, lm.y * h])
        def build_ear(indices):
            p = [pt(idx) for idx in indices]
            return (np.linalg.norm(p[1]-p[5]) + np.linalg.norm(p[2]-p[4])) / (2.0 * np.linalg.norm(p[0]-p[3]))
        return (build_ear(self.LEFT_EYE_INDICES) + build_ear(self.RIGHT_EYE_INDICES)) / 2.0

    def analyze_face_with_telemetry(self, frame: np.ndarray, state: dict = None) -> tuple[float, dict]:
        """
        Main pipeline. Now thread-safe by accepting external state.
        """
        if state is None:
            state = self.create_default_state()
            
        telemetry = {"movement_status": "Focused", "multiple_faces": False, "no_face": False, "warning": "", "raw_features": {}}
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.detector.detect(mp_image)

        if not detection_result.face_landmarks:
            telemetry.update({"no_face": True, "warning": "No Face Detected", "movement_status": "Absent"})
            return 0.0, telemetry
            
        if len(detection_result.face_landmarks) > 1:
            telemetry.update({"multiple_faces": True, "warning": "Multiple Faces Detected"})
            return 0.0, telemetry
            
        face_landmarks = detection_result.face_landmarks[0]
        
        # Head Pose
        nose, l_cheek, r_cheek = face_landmarks[1], face_landmarks[234], face_landmarks[454]
        d_l = ((nose.x - l_cheek.x)**2 + (nose.y - l_cheek.y)**2)**0.5
        d_r = ((nose.x - r_cheek.x)**2 + (nose.y - r_cheek.y)**2)**0.5
        if d_l > 1.8 * d_r: telemetry["movement_status"] = "Looking Right"
        elif d_r > 1.8 * d_l: telemetry["movement_status"] = "Looking Left"
            
        roi = self._extract_roi(frame, face_landmarks)
        if roi.size == 0: return 0.0, telemetry

        blur_score = min(max((self._calculate_laplacian_variance(roi) - 50.0) / 200.0, 0.0), 1.0) 
        moire_score = max(0.0, 1.0 - (self._detect_moire_patterns(roi) / 5000.0)) 

        # Temporal Analysis
        curr_time = time.time()
        if (curr_time - state["window_start_time"]) > 3.0:
            state["blink_total"] = 0
            state["window_start_time"] = curr_time

        ear = self._calculate_ear(face_landmarks, h, w)
        if ear < self.EAR_THRESHOLD:
            state["blink_counter"] += 1
            if telemetry["movement_status"] == "Focused": telemetry["movement_status"] = "Blinking"
        else:
            if state["blink_counter"] >= self.BLINK_CONSEC_FRAMES:
                state["blink_total"] += 1
            state["blink_counter"] = 0

        temporal_score = 1.0 if state["blink_total"] >= 1 else 0.5 
        telemetry["raw_features"] = {"blur_score": float(blur_score), "moire_score": float(moire_score), "ear": float(ear), "blink_count": int(state["blink_total"])}

        final_liveness = (blur_score * 0.35) + (moire_score * 0.35) + (temporal_score * 0.30)
        final_score = max(0.0, min(1.0, final_liveness))
        
        if final_score < 0.5 and not telemetry["warning"]:
             telemetry["warning"] = "Suspicious Behavior Detected"
             
        return final_score, telemetry

    def get_liveness_score(self, frame: np.ndarray) -> float:
        score, _ = self.analyze_face_with_telemetry(frame)
        return score
