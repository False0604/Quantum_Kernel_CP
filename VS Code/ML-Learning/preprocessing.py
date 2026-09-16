"""
preprocessing.py
Leakage-free preprocessing pipeline with StandardScaler and configurable PCA.
All transformations are strictly fitted ONLY on benign training data.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from config import DEFAULT_PCA_COMPONENTS, RANDOM_STATE


class DataPreprocessor:
    """
    Leak-free preprocessor for tabular IoT network traffic.
    Fits feature selection, median imputation, standardization, and PCA
    strictly on benign training data.
    """

    def __init__(
        self,
        n_components: int = DEFAULT_PCA_COMPONENTS,
        random_state: int = RANDOM_STATE,
    ):
        self.n_components = n_components
        self.random_state = random_state

        self.fitted = False
        self.active_features: List[str] = []
        self.medians: pd.Series = pd.Series(dtype=float)
        self.scaler = StandardScaler()
        self.pca: Optional[PCA] = None

        self.explained_variance_ratio_: List[float] = []
        self.cumulative_explained_variance_: float = 0.0

    def fit(self, benign_train: pd.DataFrame) -> "DataPreprocessor":
        """
        Fits preprocessing parameters strictly on benign training data.
        """
        if benign_train.empty:
            raise ValueError("Training data cannot be empty.")

        # 1. Clean numeric features
        numeric_train = benign_train.select_dtypes(include=[np.number]).copy()
        if numeric_train.empty:
            raise ValueError("No numeric features found in training data.")

        numeric_train.replace([np.inf, -np.inf], np.nan, inplace=True)

        # 2. Compute medians on training data only
        self.medians = numeric_train.median()
        numeric_train.fillna(self.medians, inplace=True)

        # 3. Detect and remove constant features on training data
        std_devs = numeric_train.std()
        non_constant = std_devs[std_devs > 1e-9].index.tolist()
        if not non_constant:
            raise ValueError("All features in training data have zero variance.")

        self.active_features = non_constant
        cleaned_train = numeric_train[self.active_features].values

        # 4. Fit StandardScaler
        scaled_train = self.scaler.fit_transform(cleaned_train)

        # 5. Fit PCA
        max_possible_components = min(
            self.n_components,
            scaled_train.shape[1],
            scaled_train.shape[0]
        )
        self.pca = PCA(
            n_components=max_possible_components,
            random_state=self.random_state
        )
        self.pca.fit(scaled_train)

        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_.tolist()
        self.cumulative_explained_variance_ = float(np.sum(self.pca.explained_variance_ratio_))

        self.fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """
        Transforms any dataset (benign test or attack test) using the fitted parameters.
        No statistics are computed on the input dataset.
        """
        if not self.fitted:
            raise RuntimeError("DataPreprocessor must be fitted on training data before calling transform.")

        df = data.select_dtypes(include=[np.number]).copy()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Align with active features discovered in training data
        aligned_dict = {}
        for col in self.active_features:
            if col in df.columns:
                aligned_dict[col] = df[col].fillna(self.medians.get(col, 0.0))
            else:
                aligned_dict[col] = pd.Series(self.medians.get(col, 0.0), index=df.index)

        aligned = pd.DataFrame(aligned_dict, index=df.index)

        scaled = self.scaler.transform(aligned.values)
        projected = self.pca.transform(scaled)

        return projected

    def fit_transform(self, benign_train: pd.DataFrame) -> np.ndarray:
        """Convenience method to fit on benign train and return the projected features."""
        self.fit(benign_train)
        return self.transform(benign_train)

    def get_explained_variance(self) -> Dict[str, Union[int, float, List[float]]]:
        """Returns explained variance details for reporting and viva documentation."""
        return {
            "n_components": len(self.explained_variance_ratio_),
            "explained_variance_ratio": [round(v, 4) for v in self.explained_variance_ratio_],
            "cumulative_explained_variance": round(self.cumulative_explained_variance_, 4),
        }
