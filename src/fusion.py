"""
Decision level score fusion logic
"""
import logging
import os
import numpy as np

logger = logging.getLogger(__name__)

# Try to load the trained AI model globally inside this module
ML_MODEL_PATH = "models/liveness_ai.joblib"
classifier = None

try:
    if os.path.exists(ML_MODEL_PATH):
        import joblib
        classifier = joblib.load(ML_MODEL_PATH)
        logger.info("Successfully loaded ML Random Forest model for inference.")
except Exception as e:
    logger.error(f"Failed to load ML model: {e}")

def fuse_scores(face_score: float, voice_score: float, raw_features: dict = None) -> tuple[float, str]:
    """
    Fuses the face and voice liveness scores.
    Uses Machine Learning inference if available, else falls back to heuristics.
    """
    # 0.0 generally implies definitive spoof due to missing face/silent audio handling
    if face_score == 0.0 or voice_score == 0.0:
        logger.warning("Critical spoof threshold hit on one of the sensors. Denying access.")
        return 0.0, "FAILED - Critical Spoof Detected"

    # 1. Machine Learning Inference (Phase 8 Upgrade)
    if classifier is not None and raw_features is not None:
        try:
            # Expected feature order mapping from training pipeline
            features = np.array([[
                raw_features.get('blur_score', 0.5),
                raw_features.get('moire_score', 0.5),
                raw_features.get('ear', 0.25),
                raw_features.get('blink_count', 0),
                raw_features.get('hnr_score', 0.5),
                raw_features.get('hf_score', 0.5),
                raw_features.get('mfcc_variance', 0.5),
                raw_features.get('spectral_score', 0.5)
            ]])
            
            # Predict probability of class 1 (Live)
            ai_score = float(classifier.predict_proba(features)[0][1])
            if ai_score >= 0.8:
                return ai_score, "SUCCESS - AI Validated"
            else:
                return ai_score, "FAILED - AI Low Confidence"
        except Exception as e:
            logger.error(f"AI Model Inference failed: {e}. Falling back to Mathematical Fusion.")

    # 2. Mathematical Fallback (Pre-Phase 8 Legacy)
    w_f = 0.60
    w_v = 0.40
    
    final_score = (face_score * w_f) + (voice_score * w_v)
    final_score = max(0.0, min(1.0, final_score))
    
    # 80% confidence threshold for "SUCCESS"
    if final_score >= 0.8:
        return final_score, "SUCCESS"
    else:
        return final_score, "FAILED - Low Confidence"

