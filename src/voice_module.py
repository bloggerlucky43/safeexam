import librosa
import numpy as np
import logging

logger = logging.getLogger(__name__)

class VoiceLivenessDetector:
    def __init__(self, target_sr=16000):
        """
        Initialize the Audio Signal Processing module.
        target_sr: 16kHz standard for human voice feature processing.
        """
        self.target_sr = target_sr
        self.HF_THRESHOLD = 16000  # High-Frequency check threshold (Hz)
        
    def _extract_mfcc(self, y: np.ndarray, sr: int) -> float:
        """
        Extract MFCCs to capture the vocal tract signature.
        Returns the variance of the first 13 MFCCs.
        Live humans typically display a dynamic vocal tract signature.
        """
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        # Calculate variance across time frames to measure dynamic speech
        variance_score = np.mean(np.var(mfcc, axis=1))
        return variance_score

    def _analyze_spectral_shape(self, y: np.ndarray, sr: int) -> tuple[float, float]:
        """
        Calculate Spectral Centroid and Spectral Roll-off.
        Cheap speakers emphasize higher mids and thin out bass and extreme highs.
        Returns average centroid and average rolloff.
        """
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        
        return np.mean(centroid), np.mean(rolloff)

    def _calculate_hnr(self, y: np.ndarray) -> float:
        """
        Estimate Harmonic-to-Noise Ratio (HNR).
        Live human voices have higher harmonicity than compressed digital replays.
        """
        # Separate harmonic and percussive (noise-like) components
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # Calculate power (energy)
        power_h = np.sum(y_harmonic ** 2)
        power_p = np.sum(y_percussive ** 2)
        
        # Avoid division by zero
        if power_p == 0:
            return 100.0  # Max hypothetical HNR

        # HNR calculation in dB
        hnr = 10 * np.log10(power_h / power_p)
        return hnr

    def _check_high_frequencies(self, y: np.ndarray, original_sr: int) -> float:
        """
        Search for frequencies above 16kHz before downsampling.
        Many recording devices and compression algorithms (MP3/WhatsApp) truncate at 16kHz.
        If original audio has energy > 16kHz, it's more likely live (uncompressed).
        """
        # Nyquist frequency is original_sr / 2
        if original_sr < self.HF_THRESHOLD * 2:
            logger.warning("Low Sample Rate. Cannot verify >16kHz presence.")
            return 0.0 
            
        # Compute STFT magnitude
        stft = np.abs(librosa.stft(y))
        
        # Get frequencies corresponding to STFT bins
        freqs = librosa.fft_frequencies(sr=original_sr)
        
        # Find indices where frequency > 16kHz
        hf_indices = np.where(freqs > self.HF_THRESHOLD)[0]
        
        if len(hf_indices) == 0:
            return 0.0
            
        # Sum energy in high frequency bins vs total energy
        hf_energy = np.sum(stft[hf_indices, :])
        total_energy = np.sum(stft)
        
        if total_energy == 0:
            return 0.0
            
        return hf_energy / total_energy

    def analyze_audio(self, audio_buffer: np.ndarray, original_sr: int = 44100) -> float:
        """
        Process the audio buffer for liveness detection (3-5 seconds handling).
        
        Args:
            audio_buffer: numpy array of original audio (1D)
            original_sr: sample rate of the input buffer
            
        Returns:
            float: Confidence score between 0.0 (Spoof) and 1.0 (Live)
        """
        try:
            # Handle empty/invalid buffer lengths
            if audio_buffer is None or len(audio_buffer) == 0:
                logger.warning("[FAILED] Empty audio buffer.")
                return 0.0

            # Crop to the first 5 seconds to avoid slow execution and memory limits on long exam audio
            max_samples = int(5 * original_sr)
            if len(audio_buffer) > max_samples:
                audio_buffer = audio_buffer[:max_samples]

            # 1. High-Frequency Analysis (Must be done before standardizing to 16kHz)
            hf_ratio = self._check_high_frequencies(audio_buffer, original_sr)
            # Normalize HF score: uncompressed speech usually has a small but present HF ratio
            hf_score = min(max(hf_ratio * 100.0, 0.0), 1.0) 

            # Standardize sampling rate to 16kHz for remaining vocal tract analysis
            if original_sr != self.target_sr:
                y_16k = librosa.resample(y=audio_buffer, orig_sr=original_sr, target_sr=self.target_sr)
            else:
                y_16k = audio_buffer

            # 2. Extract MFCCs (Vocal Tract Signature)
            # High variance = dynamic range of real live speech
            mfcc_variance = self._extract_mfcc(y_16k, self.target_sr)
            # Map arbitrarily based on empirical expected variance
            mfcc_score = min(max((mfcc_variance - 30.0) / 100.0, 0.0), 1.0) 

            # 3. Spectral Shape (Centroid and Roll-off)
            centroid, rolloff = self._analyze_spectral_shape(y_16k, self.target_sr)
            if centroid > 3500:
                # "Tinny" replay attack characteristic
                spectral_score = 0.2
            elif centroid < 500:
                # Unnaturally muffled/low-passed
                spectral_score = 0.1
            else:
                # Normal human baseline (approx 1000Hz - 2500Hz)
                spectral_score = 0.9 
                
            # 4. Harmonic-to-Noise Ratio
            hnr_db = self._calculate_hnr(y_16k)
            # Typical live speech has clear harmonic peaks over noise
            hnr_score = min(max((hnr_db) / 15.0, 0.0), 1.0) 

            # Score Level Fusion Logic
            # Weights: 30% HNR, 30% Uncompressed HF Evidence, 20% MFCC Dynamic, 20% Spectral Balance
            final_liveness = (hnr_score * 0.30) + (hf_score * 0.30) + (mfcc_score * 0.20) + (spectral_score * 0.20)
            
            return max(0.0, min(1.0, final_liveness))
            
        except Exception as e:
            logger.error(f"Voice analysis failed: {str(e)}")
            return 0.0
