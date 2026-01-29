"""
Build database: Extract features and create FAISS index
Run this script after collecting and preprocessing audio files
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from tqdm import tqdm
import json

from src.feature_extraction.extractor import AudioFeatureExtractor
from src.vector_database.faiss_manager import FAISSManager


def build_database(
    processed_audio_dir: str = "data/processed",
    features_output: str = "database/features.npy",
    mapping_output: str = "database/index_mapping.json",
    faiss_index_output: str = "database/vectors/faiss_index.bin"
):
    """
    Build complete database: extract features and create FAISS index
    
    Args:
        processed_audio_dir: Directory with processed audio files
        features_output: Path to save features array
        mapping_output: Path to save index mapping
        faiss_index_output: Path to save FAISS index
    """
    print("="*60)
    print("Building Voice Similarity Search Database")
    print("="*60)
    
    # Get all processed audio files
    audio_dir = Path(processed_audio_dir)
    audio_files = sorted(list(audio_dir.glob("*.wav")))
    
    if len(audio_files) == 0:
        print(f"❌ No audio files found in {processed_audio_dir}")
        print("Please run data collection and preprocessing first:")
        print("  1. python src/data_collection/download_audio.py")
        print("  2. python src/data_collection/preprocess_audio.py")
        return
    
    print(f"\nFound {len(audio_files)} audio files")
    
    # Initialize feature extractor
    extractor = AudioFeatureExtractor()
    feature_dim = extractor.get_feature_dimension()
    print(f"Feature dimension: {feature_dim}")
    
    # Extract features from all files
    print("\n📊 Extracting features...")
    features_list = []
    mapping = {}
    failed_files = []
    
    for idx, audio_file in enumerate(tqdm(audio_files, desc="Processing")):
        try:
            # Extract features
            features = extractor.extract_from_file(str(audio_file))
            
            # Validate features
            if np.isnan(features).any() or np.isinf(features).any():
                print(f"\n⚠️ Invalid features for {audio_file.name}, skipping")
                failed_files.append(str(audio_file))
                continue
            
            features_list.append(features)
            mapping[len(features_list) - 1] = str(audio_file)
            
        except Exception as e:
            print(f"\n❌ Error processing {audio_file.name}: {e}")
            failed_files.append(str(audio_file))
    
    # Convert to numpy array
    features_array = np.array(features_list)
    print(f"\nExtracted features: {features_array.shape}")
    
    # Save features
    Path(features_output).parent.mkdir(parents=True, exist_ok=True)
    np.save(features_output, features_array)
    print(f"Features saved to {features_output}")
    
    # Save mapping
    with open(mapping_output, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    print(f"Mapping saved to {mapping_output}")
    
    # Build FAISS index
    print("\n🔧 Building FAISS index...")
    faiss_manager = FAISSManager(
        dimension=feature_dim,
        index_path=faiss_index_output,
        mapping_path=mapping_output
    )
    
    faiss_manager.create_index()
    
    # Get file paths in order
    file_paths = [mapping[i] for i in range(len(mapping))]
    faiss_manager.add_vectors(features_array, file_paths)
    
    # Save index
    faiss_manager.save_index()
    
    # Statistics
    print("\n" + "="*60)
    print("Database Build Complete!")
    print("="*60)
    stats = faiss_manager.get_stats()
    print(f"Total vectors: {stats['total_vectors']}")
    print(f"Feature dimension: {stats['dimension']}")
    print(f"Index type: {stats['index_type']}")
    
    if failed_files:
        print(f"\n⚠️ Failed to process {len(failed_files)} files")
    
    print("\n✅ Database is ready for similarity search!")
    print("\nNext step: Run the Streamlit app")
    print("  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    build_database()
