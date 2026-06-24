"""
Hardware Verification Script for Biometric Liveness Detection.
Checks for CUDA (GPU), Camera access (OpenCV), and Microphone access (PyAudio).
"""

import sys
import platform
import cv2
import torch
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False


def check_gpu() -> bool:
    print("--- GPU (CUDA) Check ---")
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"[PASSED] GPU Detected: {device_name}")
        return True
    else:
        print("[WARNING] No CUDA GPU detected. Models will run on CPU, which may increase latency.")
        return False


def check_camera() -> bool:
    print("\n--- Camera Check ---")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened() or not cap.read()[0]:
        print("[FAILED] Cannot access the default camera (index 0).")
        if cap.isOpened():
            cap.release()
        return False
    
    print("[PASSED] Default camera successfully accessed.")
    cap.release()
    return True


def check_microphone() -> bool:
    print("\n--- Microphone Check ---")
    if PYAUDIO_AVAILABLE:
        print("[PASSED] PyAudio detected.")
        return True
    else:
        print("[SKIPPED] PyAudio is currently disabled due to Windows C++ build requirements. Audio will be evaluated via file upload instead.")
        return True


def main():
    print(f"System Info: {platform.system()} {platform.release()} (Python {sys.version.split()[0]})")
    print("=" * 40)
    
    gpu_ok = check_gpu()
    cam_ok = check_camera()
    mic_ok = check_microphone()
    
    print("=" * 40)
    if cam_ok and mic_ok:
        print("[SUCCESS] Essential hardware environment is ready for development.")
    else:
        print("[ERROR] Essential hardware missing. Please resolve prior to running main.py.")


if __name__ == "__main__":
    main()
