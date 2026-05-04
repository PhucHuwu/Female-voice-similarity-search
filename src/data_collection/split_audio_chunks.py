"""
Split long audio files into 3-second chunks
Useful for creating multiple training samples from single long recordings
"""
import librosa
import soundfile as sf
from pathlib import Path
import numpy as np
import random
from tqdm import tqdm
import pandas as pd


def split_audio_to_chunks(
    audio_path: str,
    output_dir: str,
    chunk_duration: float = 5.0,
    max_chunks: int = 20,
    overlap: float = 0.0,
    sr: int = 16000,
    trim_start_end_sec: float = 30.0,
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
    
    # Trim first/last part to reduce intro/outro noise
    trim_samples = int(trim_start_end_sec * sr)
    if len(audio) > 2 * trim_samples:
        audio = audio[trim_samples:-trim_samples]

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
    query_output_dir: str = "data/query_short",
    query_long_output_dir: str = "data/query_long",
    chunk_duration: float = 5.0,
    max_chunks_per_file: int = 20,
    query_chunks_per_file: int = 1,
    trim_start_end_sec: float = 30.0,
    long_min_duration_sec: float = 10.0,
    long_max_duration_sec: float = 20.0,
    seed: int = 42,
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
    query_output_path = Path(query_output_dir)
    query_long_output_path = Path(query_long_output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    query_output_path.mkdir(parents=True, exist_ok=True)
    query_long_output_path.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(seed)
    
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
    print(f"Trim head/tail: {trim_start_end_sec}s each")
    print(f"Base chunks directory: {output_dir}")
    print(f"Query short directory: {query_output_dir}")
    print(f"Query long directory: {query_long_output_dir}")
    print(f"Query chunks per file: {query_chunks_per_file}\n")
    
    metadata = []
    total_chunks = 0
    total_query_chunks = 0
    total_query_long = 0
    
    for audio_file in tqdm(youtube_files, desc="Processing files"):
        # Get file info
        audio, sr = librosa.load(audio_file, sr=16000)
        duration = len(audio) / sr
        
        print(f"\n{audio_file.name}")
        print(f"  Duration: {duration:.1f}s")
        
        # Split into fixed 5s chunks
        chunks = split_audio_to_chunks(
            audio_path=str(audio_file),
            output_dir=str(output_path),
            chunk_duration=chunk_duration,
            max_chunks=max_chunks_per_file,
            overlap=0.0,
            sr=16000,
            trim_start_end_sec=trim_start_end_sec,
        )
        
        query_count = min(query_chunks_per_file, max(0, len(chunks) - 1)) if len(chunks) > 1 else 0
        query_set = set(chunks[-query_count:]) if query_count > 0 else set()

        moved_query_chunks = 0
        for chunk_path in chunks:
            if chunk_path in query_set:
                src = Path(chunk_path)
                dst = query_output_path / src.name
                src.replace(dst)
                moved_query_chunks += 1

        base_count = len(chunks) - moved_query_chunks
        print(f"  Created: {len(chunks)} short chunks (base={base_count}, query_short={moved_query_chunks})")
        total_chunks += base_count
        total_query_chunks += moved_query_chunks

        # Create one random long query chunk (10-20s) from trimmed audio
        audio_trimmed, _ = librosa.load(str(audio_file), sr=16000)
        trim_samples = int(trim_start_end_sec * 16000)
        if len(audio_trimmed) > 2 * trim_samples:
            audio_trimmed = audio_trimmed[trim_samples:-trim_samples]

        total_sec_trimmed = len(audio_trimmed) / 16000
        long_created = False
        if total_sec_trimmed >= long_min_duration_sec:
            long_dur = rnd.uniform(long_min_duration_sec, min(long_max_duration_sec, total_sec_trimmed))
            long_samples = int(long_dur * 16000)
            if len(audio_trimmed) > long_samples:
                start = rnd.randint(0, len(audio_trimmed) - long_samples)
            else:
                start = 0
            end = start + long_samples
            long_chunk = audio_trimmed[start:end]

            dur_str = f"{long_dur:.2f}".replace(".", "p")
            long_name = f"{audio_file.stem}_longq_d{dur_str}s.wav"
            long_path = query_long_output_path / long_name
            sf.write(str(long_path), long_chunk, 16000)
            long_created = True
            total_query_long += 1

            metadata.append({
                'chunk_path': str(long_path),
                'set_type': 'query_long',
                'original_file': str(audio_file),
                'original_duration': duration,
                'chunk_duration': float(long_dur),
                'speaker_id': audio_file.stem
            })

        if not long_created:
            print("  Long query: skipped (trimmed audio too short)")
        
        # Track metadata
        for chunk_path in chunks:
            is_query = chunk_path in query_set
            final_chunk_path = str((query_output_path / Path(chunk_path).name) if is_query else Path(chunk_path))
            metadata.append({
                'chunk_path': final_chunk_path,
                'set_type': 'query_short' if is_query else 'base',
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
    print(f"Total base chunks: {total_chunks}")
    print(f"Total query short chunks: {total_query_chunks}")
    print(f"Total query long chunks: {total_query_long}")
    print(f"Average base chunks per file: {total_chunks/len(youtube_files):.1f}")
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
        query_output_dir="data/query_short",
        query_long_output_dir="data/query_long",
        chunk_duration=5.0,
        max_chunks_per_file=20,
        query_chunks_per_file=1,
        trim_start_end_sec=30.0,
        long_min_duration_sec=10.0,
        long_max_duration_sec=20.0,
        seed=42,
    )
