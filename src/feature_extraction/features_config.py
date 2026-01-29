"""
Configuration for feature extraction
"""

# Audio preprocessing parameters
SAMPLE_RATE = 16000  # 16 kHz
TARGET_DURATION = 3.0  # 3 seconds
TRIM_SILENCE = True
SILENCE_THRESHOLD_DB = 20

# MFCC parameters
N_MFCC = 13  # Number of MFCC coefficients
N_FFT = 2048  # FFT window size
HOP_LENGTH = 512  # Hop length for STFT

# Pitch detection parameters
PITCH_FMIN = 85  # Minimum frequency for female voice (Hz)
PITCH_FMAX = 300  # Maximum frequency for female voice (Hz)

# Feature dimensions
MFCC_DIM = 26  # 13 mean + 13 std
PITCH_DIM = 4  # mean, std, min, max
SPECTRAL_DIM = 6  # centroid, rolloff, bandwidth (mean + std each)
TEMPORAL_DIM = 4  # ZCR, RMS (mean + std each)
CHROMA_DIM = 12  # 12 pitch classes

TOTAL_FEATURE_DIM = MFCC_DIM + PITCH_DIM + SPECTRAL_DIM + TEMPORAL_DIM + CHROMA_DIM  # 52
