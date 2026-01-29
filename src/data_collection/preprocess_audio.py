"""
Audio preprocessing script
Normalize, trim, and standardize audio files
"""
import os
from pathlib import Path
from typing import Optional
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.audio_utils import preprocess_audio, save_audio


def preprocess_dataset(
    input_dir: str = "data/raw",
    output_dir: str = "data/processed",
    metadata_path: str = "data/metadata.csv",
    target_sr: int = 16000,
    target_duration: float = 3.0
) -> None:
    """
    Preprocess all audio files in dataset
    
    Args:
        input_dir: Directory containing raw audio files
        output_dir: Directory to save processed audio
        metadata_path: Path to metadata CSV
        target_sr: Target sample rate
        target_duration: Target duration in seconds
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all audio files
    audio_files = list(input_path.glob("*.wav")) + \
                  list(input_path.glob("*.mp3")) + \
                  list(input_path.glob("*.flac"))
    
    print(f"Found {len(audio_files)} audio files")
    print(f"Processing to {target_sr}Hz, {target_duration}s duration...")
    
    processed_metadata = []
    failed_files = []
    
    # Process each file
    for audio_file in tqdm(audio_files, desc="Processing audio"):
        try:
            # Preprocess
            processed_audio = preprocess_audio(
                str(audio_file),
                target_sr=target_sr,
                target_duration=target_duration,
                trim=True
            )
            
            # Save to processed directory
            output_file = output_path / f"{audio_file.stem}_processed.wav"
            save_audio(processed_audio, str(output_file), sr=target_sr)
            
            # Track metadata
            processed_metadata.append({
                'original_file': str(audio_file),
                'processed_file': str(output_file),
                'sample_rate': target_sr,
                'duration': target_duration
            })
            
        except Exception as e:
            print(f"\nError processing {audio_file.name}: {e}")
            failed_files.append(str(audio_file))
    
    # Save processed metadata
    df = pd.DataFrame(processed_metadata)
    processed_metadata_path = "data/processed_metadata.csv"
    df.to_csv(processed_metadata_path, index=False)
    
    print(f"\nProcessed {len(processed_metadata)} files")
    print(f"Saved to {output_dir}")
    print(f"Metadata saved to {processed_metadata_path}")
    
    if failed_files:
        print(f"\n⚠ Failed to process {len(failed_files)} files:")
        for f in failed_files[:5]:
            print(f"  - {f}")


if __name__ == "__main__":
    preprocess_dataset()
