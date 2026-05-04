"""Feature transform utilities (StandardScaler + optional PCA)."""
from pathlib import Path
import pickle
from typing import Optional, Union

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class FeatureTransform:
    """Fit/save/load feature transforms for retrieval."""

    def __init__(
        self,
        scaler_path: str = "database/scaler.pkl",
        pca_path: str = "database/pca.pkl",
    ):
        self.scaler_path = Path(scaler_path)
        self.pca_path = Path(pca_path)
        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None

    def fit(
        self,
        features: np.ndarray,
        pca_components: Optional[Union[int, float]] = None,
    ) -> np.ndarray:
        """Fit scaler and optional PCA, return transformed features."""
        self.scaler = StandardScaler()
        x_scaled = self.scaler.fit_transform(features)

        if pca_components is not None:
            self.pca = PCA(n_components=pca_components, random_state=42)
            x_out = self.pca.fit_transform(x_scaled)
        else:
            self.pca = None
            x_out = x_scaled

        return x_out.astype(np.float32)

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Apply fitted/loaded transforms to features."""
        if self.scaler is None:
            return features.astype(np.float32)

        x = self.scaler.transform(features)
        if self.pca is not None:
            x = self.pca.transform(x)
        return x.astype(np.float32)

    def save(self) -> None:
        """Persist fitted transforms."""
        self.scaler_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

        with open(self.pca_path, "wb") as f:
            pickle.dump(self.pca, f)

    def load(self) -> None:
        """Load transforms if available."""
        if self.scaler_path.exists():
            with open(self.scaler_path, "rb") as f:
                self.scaler = pickle.load(f)

        if self.pca_path.exists():
            with open(self.pca_path, "rb") as f:
                self.pca = pickle.load(f)
