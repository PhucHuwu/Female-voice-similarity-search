"""
Feature extraction module for voice similarity search
Extracts MFCC, pitch, spectral, and temporal features from audio
"""
import librosa
import numpy as np
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class AudioFeatureExtractor:
    """Extract audio features for voice similarity comparison"""
    
    def __init__(
        self,
        sr: int = 16000,
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 512
    ):
        """
        Initialize feature extractor
        
        Args:
            sr: Sample rate
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
        """
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features (voice timbre)
        
        Args:
            audio: Audio signal array
            
        Returns:
            MFCC feature vector (mean and std of each coefficient)
        """
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Calculate mean and std for each coefficient
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        
        return np.concatenate([mfcc_mean, mfcc_std])
    
    def extract_pitch_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract pitch (fundamental frequency) features
        
        Args:
            audio: Audio signal array
            
        Returns:
            Pitch feature vector [mean, std, min, max]
        """
        # Extract pitch using piptrack
        pitches, magnitudes = librosa.piptrack(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Get pitch values (filter out zeros)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if len(pitch_values) == 0:
            return np.zeros(4)
        
        pitch_values = np.array(pitch_values)
        
        return np.array([
            np.mean(pitch_values),
            np.std(pitch_values),
            np.min(pitch_values),
            np.max(pitch_values)
        ])
    
    def extract_spectral_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract spectral features (brightness, rolloff, flux)
        
        Args:
            audio: Audio signal array
            
        Returns:
            Spectral feature vector
        """
        # Spectral centroid (brightness)
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )[0]
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )[0]
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )[0]
        
        return np.array([
            np.mean(spectral_centroids),
            np.std(spectral_centroids),
            np.mean(spectral_rolloff),
            np.std(spectral_rolloff),
            np.mean(spectral_bandwidth),
            np.std(spectral_bandwidth)
        ])
    
    def extract_temporal_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract temporal features (ZCR, RMS energy)
        
        Args:
            audio: Audio signal array
            
        Returns:
            Temporal feature vector
        """
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(
            y=audio,
            frame_length=self.n_fft,
            hop_length=self.hop_length
        )[0]
        
        # RMS energy
        rms = librosa.feature.rms(
            y=audio,
            frame_length=self.n_fft,
            hop_length=self.hop_length
        )[0]
        
        return np.array([
            np.mean(zcr),
            np.std(zcr),
            np.mean(rms),
            np.std(rms)
        ])
    
    def extract_chroma_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract chroma features (pitch class energy distribution)
        
        Args:
            audio: Audio signal array
            
        Returns:
            Chroma feature vector (12 pitch classes)
        """
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Mean of each pitch class
        return np.mean(chroma, axis=1)
    
    def extract_all_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract all features and concatenate into single vector
        
        Args:
            audio: Audio signal array
            
        Returns:
            Complete feature vector
        """
        mfcc_features = self.extract_mfcc(audio)  # 26 features (13 mean + 13 std)
        pitch_features = self.extract_pitch_features(audio)  # 4 features
        spectral_features = self.extract_spectral_features(audio)  # 6 features
        temporal_features = self.extract_temporal_features(audio)  # 4 features
        chroma_features = self.extract_chroma_features(audio)  # 12 features
        
        # Concatenate all features
        all_features = np.concatenate([
            mfcc_features,
            pitch_features,
            spectral_features,
            temporal_features,
            chroma_features
        ])
        
        return all_features
    
    def extract_from_file(self, file_path: str) -> np.ndarray:
        """
        Extract features directly from audio file
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Feature vector
        """
        audio, _ = librosa.load(file_path, sr=self.sr)
        return self.extract_all_features(audio)
    
    def get_feature_dimension(self) -> int:
        """Get total dimension of feature vector"""
        return 26 + 4 + 6 + 4 + 12  # 52 dimensions total


def extract_all_features(audio_path: str, sr: int = 16000) -> np.ndarray:
    """
    Convenience function to extract all features from audio file
    
    Args:
        audio_path: Path to audio file
        sr: Sample rate
        
    Returns:
        Feature vector (52 dimensions)
    """
    extractor = AudioFeatureExtractor(sr=sr)
    return extractor.extract_from_file(audio_path)
