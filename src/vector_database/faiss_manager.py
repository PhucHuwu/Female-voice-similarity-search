"""
FAISS vector database manager for voice similarity search
"""
import faiss
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Optional


class FAISSManager:
    """Manage FAISS index for vector similarity search"""
    
    def __init__(
        self,
        dimension: int = 52,
        index_path: str = "database/vectors/faiss_index.bin",
        mapping_path: str = "database/index_mapping.json"
    ):
        """
        Initialize FAISS manager
        
        Args:
            dimension: Feature vector dimension
            index_path: Path to save/load FAISS index
            mapping_path: Path to save/load index-to-file mapping
        """
        self.dimension = dimension
        self.index_path = Path(index_path)
        self.mapping_path = Path(mapping_path)
        self.index: Optional[faiss.Index] = None
        self.mapping: dict = {}
        
        # Create directories if they don't exist
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
    
    def create_index(self) -> faiss.Index:
        """
        Create new FAISS index using Inner Product (Cosine Similarity)
        
        Returns:
            FAISS index
        """
        # Use IndexFlatIP for cosine similarity (after L2 normalization)
        self.index = faiss.IndexFlatIP(self.dimension)
        print(f"Created FAISS index with dimension {self.dimension} (Cosine Similarity)")
        return self.index
    
    def add_vectors(
        self,
        vectors: np.ndarray,
        file_paths: List[str]
    ) -> None:
        """
        Add vectors to index with corresponding file paths
        
        Args:
            vectors: Feature vectors array (n_samples, dimension)
            file_paths: List of audio file paths
        """
        if self.index is None:
            self.create_index()
        
        # Ensure vectors are float32
        vectors = vectors.astype('float32')
        
        # L2 normalize for cosine similarity
        faiss.normalize_L2(vectors)
        
        # Get current index count
        start_idx = self.index.ntotal
        
        # Add to index
        self.index.add(vectors)
        
        # Update mapping
        for i, file_path in enumerate(file_paths):
            self.mapping[start_idx + i] = file_path
        
        print(f"Added {len(file_paths)} vectors (L2 normalized). Total: {self.index.ntotal}")
    
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Search for k most similar vectors using cosine similarity
        
        Args:
            query_vector: Query feature vector
            k: Number of results to return
            
        Returns:
            List of (file_path, similarity_score) tuples
            similarity_score: 0-1, where 1 = identical, 0 = orthogonal
        """
        if self.index is None:
            raise ValueError("Index not loaded. Call load_index() first.")
        
        # Ensure query is 2D and float32
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        query_vector = query_vector.astype('float32')
        
        # L2 normalize query for cosine similarity
        faiss.normalize_L2(query_vector)
        
        # Search (returns inner products = cosine similarities)
        similarities, indices = self.index.search(query_vector, k)
        
        # Map indices to file paths
        results = []
        for idx, sim in zip(indices[0], similarities[0]):
            file_path = self.mapping.get(int(idx), "Unknown")
            # Clip to [0, 1] range (sometimes numerical issues cause >1)
            sim = float(np.clip(sim, 0, 1))
            results.append((file_path, sim))
        
        return results
    
    def save_index(self) -> None:
        """Save FAISS index and mapping to disk"""
        if self.index is None:
            raise ValueError("No index to save")
        
        # Save FAISS index
        faiss.write_index(self.index, str(self.index_path))
        print(f"Index saved to {self.index_path}")
        
        # Save mapping
        with open(self.mapping_path, 'w', encoding='utf-8') as f:
            # Convert int keys to string for JSON
            json_mapping = {str(k): v for k, v in self.mapping.items()}
            json.dump(json_mapping, f, indent=2, ensure_ascii=False)
        print(f"Mapping saved to {self.mapping_path}")
    
    def load_index(self) -> None:
        """Load FAISS index and mapping from disk"""
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_path}")
        
        # Load FAISS index
        self.index = faiss.read_index(str(self.index_path))
        print(f"Loaded index with {self.index.ntotal} vectors")
        
        # Load mapping
        if self.mapping_path.exists():
            with open(self.mapping_path, 'r', encoding='utf-8') as f:
                json_mapping = json.load(f)
                # Convert string keys back to int
                self.mapping = {int(k): v for k, v in json_mapping.items()}
            print(f"Loaded mapping with {len(self.mapping)} entries")
    
    def get_stats(self) -> dict:
        """Get statistics about the index"""
        if self.index is None:
            return {"status": "No index loaded"}
        
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": type(self.index).__name__,
            "mapping_size": len(self.mapping)
        }


def build_index_from_features(
    features_path: str = "database/features.npy",
    mapping_path: str = "database/index_mapping.json"
) -> FAISSManager:
    """
    Build FAISS index from pre-extracted features
    
    Args:
        features_path: Path to features numpy array
        mapping_path: Path to index mapping JSON
        
    Returns:
        FAISSManager with built index
    """
    # Load features
    features = np.load(features_path)
    print(f"Loaded features: {features.shape}")
    
    # Load mapping
    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping_dict = json.load(f)
    
    file_paths = [mapping_dict[str(i)] for i in range(len(mapping_dict))]
    
    # Create FAISS manager
    manager = FAISSManager(dimension=features.shape[1])
    manager.create_index()
    manager.add_vectors(features, file_paths)
    manager.save_index()
    
    return manager


if __name__ == "__main__":
    # Example: Build index from features
    print("Building FAISS index from features...")
    manager = build_index_from_features()
    print("\n" + "="*50)
    print("Index Statistics:")
    print(manager.get_stats())
