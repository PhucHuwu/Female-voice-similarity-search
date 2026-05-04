"""Create random long query chunks (10-20s) for robustness testing."""
from pathlib import Path
import random

import librosa
import soundfile as sf
from tqdm import tqdm


def create_long_query_chunks(
    input_dir: str = "data/raw",
    output_dir: str = "data/query_long",
    min_duration_sec: float = 10.0,
    max_duration_sec: float = 20.0,
    chunks_per_file: int = 1,
    trim_start_end_sec: float = 30.0,
    sr: int = 16000,
    seed: int = 42,
) -> None:
    """Generate random long chunks from each raw file.

    Output filename includes duration, e.g.
    yt_voice_videoid_longq_d12.34s_r00.wav
    """
    rnd = random.Random(seed)

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(input_path.glob("yt_*.wav"))
    if not raw_files:
        print(f"No raw files found in {input_dir}")
        return

    print("=" * 60)
    print("Creating random long query chunks")
    print("=" * 60)
    print(f"Input files: {len(raw_files)}")
    print(f"Duration range: {min_duration_sec}-{max_duration_sec}s")
    print(f"Chunks per file: {chunks_per_file}")
    print(f"Output: {output_dir}\n")

    created = 0
    skipped = 0

    for audio_file in tqdm(raw_files, desc="Generating long queries"):
        audio, _ = librosa.load(str(audio_file), sr=sr)

        trim_samples = int(trim_start_end_sec * sr)
        if len(audio) > 2 * trim_samples:
            audio = audio[trim_samples:-trim_samples]

        total_sec = len(audio) / sr
        if total_sec < min_duration_sec:
            skipped += 1
            continue

        for idx in range(chunks_per_file):
            dur = rnd.uniform(min_duration_sec, min(max_duration_sec, total_sec))
            seg_samples = int(dur * sr)

            if len(audio) == seg_samples:
                start = 0
            else:
                start = rnd.randint(0, len(audio) - seg_samples)
            end = start + seg_samples
            chunk = audio[start:end]

            dur_str = f"{dur:.2f}".replace(".", "p")
            out_name = f"{audio_file.stem}_longq_d{dur_str}s_r{idx:02d}.wav"
            out_path = output_path / out_name
            sf.write(str(out_path), chunk, sr)
            created += 1

    print("\n" + "=" * 60)
    print("LONG QUERY GENERATION COMPLETE")
    print("=" * 60)
    print(f"Created files: {created}")
    print(f"Skipped raw files: {skipped}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    create_long_query_chunks(
        input_dir="data/raw",
        output_dir="data/query_long",
        min_duration_sec=10.0,
        max_duration_sec=20.0,
        chunks_per_file=1,
        trim_start_end_sec=30.0,
        sr=16000,
        seed=42,
    )
