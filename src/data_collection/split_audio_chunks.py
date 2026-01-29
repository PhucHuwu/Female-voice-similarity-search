"""
Split long audio files into 3-second chunks
Useful for creating multiple training samples from single long recordings
"""
import librosa
import soundfile as sf
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pandas as pd


def split_audio_to_chunks(
    audio_path: str,
    output_dir: str,
    chunk_duration: float = 3.0,
    max_chunks: int = 100,
    overlap: float = 0.0,
    sr: int = 16000
) -> list:
    """
    Split audio file into fixed-duration chunks
    
    Args:
        audio_path: Path to input audio file
        output_dir: Directory to save chunks
        chunk_duration: Duration of each chunk in seconds
        max_chunks: Maximum number of chunks to create
        overlap: Overlap between chunks in seconds
        sr: Target sample rate
        
    Returns:
        List of output file paths
    """
    # Load audio
    audio, original_sr = librosa.load(audio_path, sr=sr)
    
    # Calculate chunk parameters
    chunk_samples = int(chunk_duration * sr)
    overlap_samples = int(overlap * sr)
    hop_samples = chunk_samples - overlap_samples
    
    # Calculate number of chunks
    total_samples = len(audio)
    num_chunks = min(
        max_chunks,
        (total_samples - chunk_samples) // hop_samples + 1
    )
    
    output_files = []
    audio_name = Path(audio_path).stem
    
    for i in range(num_chunks):
        start = i * hop_samples
        end = start + chunk_samples
        
        # Break if we don't have enough samples
        if end > total_samples:
            break
        
        chunk = audio[start:end]
        
        # Save chunk
        output_file = Path(output_dir) / f"{audio_name}_chunk{i:04d}.wav"
        sf.write(output_file, chunk, sr)
        output_files.append(str(output_file))
    
    return output_files


def process_all_youtube_files(
    input_dir: str = "data/raw",
    output_dir: str = "data/chunks",
    chunk_duration: float = 3.0,
    max_chunks_per_file: int = 100
):
    """
    Process all YouTube audio files and split into chunks
    
    Args:
        input_dir: Directory containing yt_*.wav files
        output_dir: Directory to save chunks
        chunk_duration: Duration of each chunk
        max_chunks_per_file: Maximum chunks per file
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all YouTube audio files
    youtube_files = sorted(input_path.glob("yt_*.wav"))
    
    if not youtube_files:
        print(f"No YouTube files found in {input_dir}")
        print("Looking for files matching: yt_*.wav")
        return
    
    print("="*60)
    print(f"Splitting {len(youtube_files)} YouTube audio files")
    print("="*60)
    print(f"Chunk duration: {chunk_duration}s")
    print(f"Max chunks per file: {max_chunks_per_file}")
    print(f"Output directory: {output_dir}\n")
    
    metadata = []
    total_chunks = 0
    
    for audio_file in tqdm(youtube_files, desc="Processing files"):
        # Get file info
        audio, sr = librosa.load(audio_file, sr=16000)
        duration = len(audio) / sr
        
        print(f"\n{audio_file.name}")
        print(f"  Duration: {duration:.1f}s")
        
        # Split into chunks
        chunks = split_audio_to_chunks(
            audio_path=str(audio_file),
            output_dir=str(output_path),
            chunk_duration=chunk_duration,
            max_chunks=max_chunks_per_file,
            overlap=0.0,
            sr=16000
        )
        
        print(f"  Created: {len(chunks)} chunks")
        total_chunks += len(chunks)
        
        # Track metadata
        for chunk_path in chunks:
            metadata.append({
                'chunk_path': chunk_path,
                'original_file': str(audio_file),
                'original_duration': duration,
                'chunk_duration': chunk_duration,
                'speaker_id': audio_file.stem
            })
    
    # Save metadata
    df = pd.DataFrame(metadata)
    metadata_path = "data/chunks_metadata.csv"
    df.to_csv(metadata_path, index=False)
    
    print("\n" + "="*60)
    print("SPLITTING COMPLETE")
    print("="*60)
    print(f"Total YouTube files: {len(youtube_files)}")
    print(f"Total chunks created: {total_chunks}")
    print(f"Average chunks per file: {total_chunks/len(youtube_files):.1f}")
    print(f"Metadata saved to: {metadata_path}")
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Process chunks:")
    print("   python src/data_collection/preprocess_audio.py")
    print("\n2. Build database:")
    print("   python scripts/build_database.py")


if __name__ == "__main__":
    process_all_youtube_files(
        input_dir="data/raw",
        output_dir="data/chunks",
        chunk_duration=3.0,
        max_chunks_per_file=100
    )
