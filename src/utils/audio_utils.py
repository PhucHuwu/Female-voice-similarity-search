"""
Audio utility functions for voice similarity search system
"""
import librosa
import soundfile as sf
import numpy as np
from typing import Tuple


def load_audio(file_path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load audio file and resample to target sample rate
    
    Args:
        file_path: Path to audio file
        sr: Target sample rate (default 16000 Hz)
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    audio, sample_rate = librosa.load(file_path, sr=sr)
    return audio, sample_rate


def trim_silence(audio: np.ndarray, top_db: int = 20) -> np.ndarray:
    """
    Remove silence from beginning and end of audio
    
    Args:
        audio: Audio data array
        top_db: Threshold in decibels below reference to consider as silence
        
    Returns:
        Trimmed audio array
    """
    trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed_audio


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio to have peak amplitude of 1.0
    
    Args:
        audio: Audio data array
        
    Returns:
        Normalized audio array
    """
    return librosa.util.normalize(audio)


def fix_length(audio: np.ndarray, target_length: int) -> np.ndarray:
    """
    Fix audio to target length by cropping or padding
    
    Args:
        audio: Audio data array
        target_length: Target length in samples
        
    Returns:
        Fixed-length audio array
    """
    if len(audio) > target_length:
        # Crop from center
        start = (len(audio) - target_length) // 2
        return audio[start:start + target_length]
    else:
        # Pad with zeros
        return np.pad(audio, (0, target_length - len(audio)), mode='constant')


def preprocess_audio(
    file_path: str,
    target_sr: int = 16000,
    target_duration: float = 3.0,
    trim: bool = True
) -> np.ndarray:
    """
    Complete preprocessing pipeline for audio file
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate in Hz
        target_duration: Target duration in seconds
        trim: Whether to trim silence
        
    Returns:
        Preprocessed audio array
    """
    # Load audio
    audio, sr = load_audio(file_path, sr=target_sr)
    
    # Trim silence
    if trim:
        audio = trim_silence(audio)
    
    # Normalize
    audio = normalize_audio(audio)
    
    # Fix length
    target_length = int(target_duration * target_sr)
    audio = fix_length(audio, target_length)
    
    return audio


def save_audio(audio: np.ndarray, file_path: str, sr: int = 16000) -> None:
    """
    Save audio to file
    
    Args:
        audio: Audio data array
        file_path: Output file path
        sr: Sample rate
    """
    sf.write(file_path, audio, sr)
