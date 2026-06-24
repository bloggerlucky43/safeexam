"""
Synthetic Integration Test for Liveness Detection Pipeline.
Creates dummy video frames and audio buffers to verify the end-to-end mathematical models.
"""

import numpy as np
import cv2
import librosa
from src.face_module import FaceLivenessDetector
from src.voice_module import VoiceLivenessDetector
from src.fusion import fuse_scores

def test_pipeline():
    print("--- 1. Initializing Modules ---")
    try:
        face_detector = FaceLivenessDetector()
        voice_detector = VoiceLivenessDetector()
        print("[SUCCESS] Modules initialized.")
    except Exception as e:
        print(f"[ERROR] Module initialization failed: {e}")
        return

    print("\n--- 2. Testing Face Module (Synthetic Data) ---")
    # Create a dummy image (e.g., pure black) which will fail MediaPipe detection
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    try:
        face_score = face_detector.get_liveness_score(dummy_frame)
        print(f"[EXPECTED SPOOF] Face Liveness Score: {face_score:.2f}")
    except Exception as e:
        print(f"[ERROR] Face execution failed: {e}")
        face_score = 0.0

    print("\n--- 3. Testing Voice Module (Synthetic Data) ---")
    # Create 3 seconds of white noise at 44.1kHz
    sr = 44100
    dummy_audio = np.random.uniform(-1, 1, sr * 3)
    try:
        voice_score = voice_detector.analyze_audio(dummy_audio, original_sr=sr)
        print(f"[EXPECTED SPOOF/NOISE] Voice Liveness Score: {voice_score:.2f}")
    except Exception as e:
        print(f"[ERROR] Voice execution failed: {e}")
        voice_score = 0.0

    print("\n--- 4. Testing Score Fusion ---")
    final_score, status = fuse_scores(face_score, voice_score)
    print(f"Final Score: {final_score:.2f} | Status: {status}")

if __name__ == "__main__":
    test_pipeline()
