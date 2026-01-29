"""
Script to view features from database
"""
import numpy as np
import pandas as pd

# Load features
features = np.load('database/features.npy')
print(f"Features shape: {features.shape}")
print(f"Total audio files: {features.shape[0]}")
print(f"Features per file: {features.shape[1]}")

# Show first audio file's features
print("\n" + "="*60)
print("Example: Features của file đầu tiên")
print("="*60)

feature_names = [
    # MFCC (26)
    *[f"MFCC_{i}_mean" for i in range(1, 14)],
    *[f"MFCC_{i}_std" for i in range(1, 14)],
    
    # Pitch (4)
    "Pitch_mean", "Pitch_std", "Pitch_min", "Pitch_max",
    
    # Spectral (6)
    "Centroid_mean", "Centroid_std",
    "Rolloff_mean", "Rolloff_std",
    "Bandwidth_mean", "Bandwidth_std",
    
    # Temporal (4)
    "ZCR_mean", "ZCR_std",
    "RMS_mean", "RMS_std",
    
    # Chroma (12)
    *[f"Chroma_{i}" for i in range(12)]
]

# Create DataFrame for first sample
df = pd.DataFrame([features[0]], columns=feature_names)
print(df.T)  # Transpose for better viewing

print("\n" + "="*60)
print("Statistics")
print("="*60)
print(f"Mean of all features:\n{features.mean(axis=0)[:10]}...")  # First 10
print(f"\nStd of all features:\n{features.std(axis=0)[:10]}...")   # First 10
