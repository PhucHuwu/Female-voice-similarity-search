"""Build SQLite database with metadata and feature vectors."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import librosa
from tqdm import tqdm

from src.feature_extraction.extractor import AudioFeatureExtractor
from src.vector_database.metadata_db import MetadataDB, parse_processed_filename, load_video_catalog


def build_database(
    processed_audio_dir: str = "data/processed",
    features_output: str = "database/features.npy",
    metadata_db_path: str = "database/metadata.db",
    video_catalog_csv: str = "data/list_video.csv",
    min_required_files: int = 500,
):
    """
    Build complete database: extract features and save to SQLite
    
    Args:
        processed_audio_dir: Directory with processed audio files
        features_output: Path to save features array
        metadata_db_path: Path to SQLite metadata database
    """
    print("="*60)
    print("Building Voice Similarity Search Database")
    print("="*60)
    
    # Get all processed audio files
    audio_dir = Path(processed_audio_dir)
    audio_files = sorted(list(audio_dir.glob("*.wav")))
    
    if len(audio_files) == 0:
        print(f"No audio files found in {processed_audio_dir}")
        print("Please run data collection and preprocessing first:")
        print("  1. python src/data_collection/download_audio.py")
        print("  2. python src/data_collection/preprocess_audio.py")
        return

    if len(audio_files) < min_required_files:
        print(
            f"Dataset requirement not met: found {len(audio_files)} files, "
            f"need at least {min_required_files} files."
        )
        return
    
    print(f"\nFound {len(audio_files)} audio files")
    
    # Initialize feature extractor
    extractor = AudioFeatureExtractor()
    feature_dim = extractor.get_feature_dimension()
    print(f"Feature dimension: {feature_dim}")
    
    # Extract features from all files
    print("\nExtracting features...")
    features_list = []
    failed_files = []
    metadata_records = []
    video_catalog = load_video_catalog(video_catalog_csv)
    
    for idx, audio_file in enumerate(tqdm(audio_files, desc="Processing")):
        try:
            # Extract features
            features = extractor.extract_from_file(str(audio_file))
            
            # Validate features
            if np.isnan(features).any() or np.isinf(features).any():
                print(f"\nInvalid features for {audio_file.name}, skipping")
                failed_files.append(str(audio_file))
                continue
            
            features_list.append(features)
            vector_idx = len(features_list) - 1
            file_path = str(audio_file)

            parsed = parse_processed_filename(file_path)
            sr = librosa.get_samplerate(file_path)
            duration = librosa.get_duration(path=file_path)

            source_video_id = parsed.get("video_id")
            source_info = video_catalog.get(source_video_id, {}) if source_video_id else {}
            voice_name = parsed.get("voice") or source_info.get("voice")

            metadata_records.append({
                "vector_idx": vector_idx,
                "file_path": file_path,
                "source_url": source_info.get("url"),
                "source_video_id": source_video_id,
                "voice_name": voice_name,
                "chunk_id": parsed.get("chunk_id"),
                "sample_rate": sr,
                "duration": duration,
                "feature_dim": feature_dim,
                "feature_vector": features,
            })
            
        except Exception as e:
            print(f"\nError processing {audio_file.name}: {e}")
            failed_files.append(str(audio_file))
    
    # Convert to numpy array
    features_array = np.array(features_list)
    print(f"\nExtracted features: {features_array.shape}")
    
    # Save features
    Path(features_output).parent.mkdir(parents=True, exist_ok=True)
    np.save(features_output, features_array)
    print(f"Features saved to {features_output}")

    # Save metadata DB
    metadata_db = MetadataDB(metadata_db_path)
    metadata_db.clear_all()
    metadata_db.upsert_records(metadata_records)
    print(f"Metadata DB saved to {metadata_db_path} ({metadata_db.count()} records)")
    
    # Statistics
    print("\n" + "="*60)
    print("Database Build Complete!")
    print("="*60)
    print(f"Total vectors: {metadata_db.count()}")
    print(f"Feature dimension: {feature_dim}")
    print("Storage type: SQLite (metadata + feature vectors)")
    
    if failed_files:
        print(f"\nFailed to process {len(failed_files)} files")
    
    print("\nDatabase is ready for similarity search!")
    print("\nNext step: Run the Streamlit app")
    print("  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    build_database()
