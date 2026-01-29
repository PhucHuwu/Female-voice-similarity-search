"""
Generate synthetic audio files for testing
Useful when external datasets are not accessible
"""
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List
import pandas as pd


def generate_synthetic_voice(
    duration: float = 3.0,
    sample_rate: int = 16000,
    base_frequency: float = 200.0,
    frequency_variation: float = 50.0
) -> np.ndarray:
    """
    Generate synthetic voice-like audio
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        base_frequency: Base fundamental frequency (pitch)
        frequency_variation: Random variation in frequency
        
    Returns:
        Audio signal array
    """
    n_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, n_samples)
    
    # Random frequency variation to simulate voice
    freq = base_frequency + np.random.randn() * frequency_variation
    
    # Generate harmonic series (voice has multiple harmonics)
    signal = np.zeros(n_samples)
    
    # Add fundamental and harmonics with decreasing amplitude
    for harmonic in range(1, 6):
        amplitude = 0.5 / harmonic  # Decreasing amplitude
        signal += amplitude * np.sin(2 * np.pi * freq * harmonic * t)
    
    # Add formant-like resonances (simplified)
    formant1 = 800 + np.random.randn() * 100
    formant2 = 1200 + np.random.randn() * 150
    signal += 0.3 * np.sin(2 * np.pi * formant1 * t)
    signal += 0.2 * np.sin(2 * np.pi * formant2 * t)
    
    # Add slight noise (breathiness)
    noise = np.random.randn(n_samples) * 0.05
    signal += noise
    
    # Apply envelope (attack, sustain, release)
    envelope = np.ones(n_samples)
    attack_samples = int(0.1 * sample_rate)
    release_samples = int(0.2 * sample_rate)
    
    # Attack
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    # Release
    envelope[-release_samples:] = np.linspace(1, 0, release_samples)
    
    signal *= envelope
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8
    
    return signal.astype(np.float32)


def generate_dataset(
    num_samples: int = 50,
    output_dir: str = "data/raw",
    duration: float = 3.0,
    sample_rate: int = 16000
) -> None:
    """
    Generate a synthetic dataset of voice-like audio
    
    Args:
        num_samples: Number of audio files to generate
        output_dir: Output directory
        duration: Duration of each file in seconds
        sample_rate: Sample rate
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_samples} synthetic voice samples...")
    print(f"Duration: {duration}s, Sample rate: {sample_rate}Hz")
    print(f"Output: {output_dir}")
    
    metadata = []
    
    # Female voice frequency range: ~165-255 Hz
    # We'll create variations within this range
    base_frequencies = np.random.uniform(165, 255, num_samples)
    
    for i in range(num_samples):
        # Generate audio
        audio = generate_synthetic_voice(
            duration=duration,
            sample_rate=sample_rate,
            base_frequency=base_frequencies[i],
            frequency_variation=20.0
        )
        
        # Save file
        file_name = f"synthetic_{i:05d}.wav"
        file_path = output_path / file_name
        sf.write(file_path, audio, sample_rate)
        
        # Track metadata
        metadata.append({
            'file_path': str(file_path),
            'source': 'synthetic',
            'index': i,
            'base_frequency': base_frequencies[i],
            'duration': duration,
            'sample_rate': sample_rate
        })
        
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{num_samples} files...")
    
    # Save metadata
    df = pd.DataFrame(metadata)
    metadata_path = "data/metadata.csv"
    df.to_csv(metadata_path, index=False)
    
    print(f"\nGenerated {num_samples} synthetic voice files")
    print(f"Saved to {output_dir}")
    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    # Generate 50 samples for testing
    generate_dataset(num_samples=50)
