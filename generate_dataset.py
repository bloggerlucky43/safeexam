import pandas as pd
import numpy as np
import os

def generate_multi_modal_dataset():
    os.makedirs('data', exist_ok=True)

    # Generate 500 Live samples
    np.random.seed(42)
    live_features = {
        'blur_score': np.random.normal(0.8, 0.1, 500).clip(0, 1),
        'moire_score': np.random.normal(0.9, 0.05, 500).clip(0, 1),
        'ear': np.random.normal(0.25, 0.03, 500),
        'blink_count': np.random.poisson(1.5, 500),
        'hnr_score': np.random.normal(0.85, 0.1, 500).clip(0, 1),
        'hf_score': np.random.normal(0.7, 0.2, 500).clip(0, 1),
        'mfcc_variance': np.random.normal(0.8, 0.15, 500).clip(0, 1),
        'spectral_score': np.random.normal(0.9, 0.1, 500).clip(0, 1),
        'label': 1
    }

    # Generate 500 Spoof samples (printed photos, screen replays, compressed audio)
    spoof_features = {
        'blur_score': np.random.normal(0.3, 0.2, 500).clip(0, 1), # Blurry photos
        'moire_score': np.random.normal(0.2, 0.15, 500).clip(0, 1), # High moire from screens
        'ear': np.random.normal(0.28, 0.01, 500), # Static open eyes
        'blink_count': np.zeros(500), # No blinks
        'hnr_score': np.random.normal(0.4, 0.2, 500).clip(0, 1), # Noisy audio replay
        'hf_score': np.random.normal(0.1, 0.1, 500).clip(0, 1), # Compressed audio lacks HF
        'mfcc_variance': np.random.normal(0.2, 0.1, 500).clip(0, 1), # Static audio/noise
        'spectral_score': np.random.normal(0.3, 0.2, 500).clip(0, 1), # Tinny speaker shape
        'label': 0
    }

    df_live = pd.DataFrame(live_features)
    df_spoof = pd.DataFrame(spoof_features)
    df = pd.concat([df_live, df_spoof]).sample(frac=1).reset_index(drop=True)

    df.to_csv('data/biometric_features.csv', index=False)
    print("Successfully generated 1000 synthetic multi-modal biometric feature sets in data/biometric_features.csv")

if __name__ == "__main__":
    generate_multi_modal_dataset()
