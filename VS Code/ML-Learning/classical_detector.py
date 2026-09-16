"""
classical_detector.py
Classical baseline anomaly detectors:
1. Classical RBF One-Class SVM (kernel='rbf', gamma='scale')
2. Isolation Forest (tree-based ensemble)
3. Local Outlier Factor (novelty detection mode)

Ensures identical preprocessing, PCA, and data splits for rigorous scientific fairness.
"""

from pathlib import Path
import time
from typing import Dict, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from config import (
    DEFAULT_PCA_COMPONENTS,
    DEFAULT_NU,
    DEFAULT_RBF_GAMMA,
    DEFAULT_IFOREST_ESTIMATORS,
    DEFAULT_IFOREST_CONTAMINATION,
    RANDOM_STATE,
)
from preprocessing import DataPreprocessor


class ClassicalRBFAnomalyDetector:
    """
    Classical One-Class SVM with Radial Basis Function (RBF) Gaussian kernel.
    Serves as the direct classical kernel-method counterpart to the Quantum Fidelity Kernel.
    """

    def __init__(
        self,
        n_components: int = DEFAULT_PCA_COMPONENTS,
        nu: float = DEFAULT_NU,
        gamma: Union[str, float] = DEFAULT_RBF_GAMMA,
        random_state: int = RANDOM_STATE,
    ):
        self.n_components = n_components
        self.nu = nu
        self.gamma = gamma
        self.random_state = random_state

        self.name = "Classical RBF OCSVM"
        self.preprocessor: Optional[DataPreprocessor] = None
        self.model: Optional[OneClassSVM] = None

        self.timings: Dict[str, float] = {
            "preprocessing_time": 0.0,
            "fit_time": 0.0,
            "predict_time": 0.0,
            "total_runtime": 0.0,
        }

    def fit(self, benign_train_df: pd.DataFrame) -> "ClassicalRBFAnomalyDetector":
        start_total = time.perf_counter()

        t0 = time.perf_counter()
        self.preprocessor = DataPreprocessor(
            n_components=self.n_components,
            random_state=self.random_state
        )
        X_pca = self.preprocessor.fit_transform(benign_train_df)
        self.timings["preprocessing_time"] = time.perf_counter() - t0

        t_fit = time.perf_counter()
        self.model = OneClassSVM(
            kernel="rbf",
            gamma=self.gamma,
            nu=self.nu
        )
        self.model.fit(X_pca)
        self.timings["fit_time"] = time.perf_counter() - t_fit
        self.timings["total_runtime"] = time.perf_counter() - start_total
        return self

    def predict(
        self, test_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("Model must be fitted before predict.")

        t0 = time.perf_counter()
        X_test_pca = self.preprocessor.transform(test_df)
        predictions = self.model.predict(X_test_pca)
        decision_raw = self.model.decision_function(X_test_pca)
        self.timings["predict_time"] = time.perf_counter() - t0

        # Anomaly score: higher value indicates more anomalous
        anomaly_scores = -decision_raw
        return predictions, anomaly_scores, decision_raw


class IsolationForestAnomalyDetector:
    """
    Isolation Forest anomaly detector.
    Isolates observations by randomly selecting a feature and split value.
    Serves as an ensemble tree-based classical benchmark.
    """

    def __init__(
        self,
        n_components: int = DEFAULT_PCA_COMPONENTS,
        n_estimators: int = DEFAULT_IFOREST_ESTIMATORS,
        contamination: Union[str, float] = DEFAULT_IFOREST_CONTAMINATION,
        random_state: int = RANDOM_STATE,
    ):
        self.n_components = n_components
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state

        self.name = "Isolation Forest"
        self.preprocessor: Optional[DataPreprocessor] = None
        self.model: Optional[IsolationForest] = None

        self.timings: Dict[str, float] = {
            "preprocessing_time": 0.0,
            "fit_time": 0.0,
            "predict_time": 0.0,
            "total_runtime": 0.0,
        }

    def fit(self, benign_train_df: pd.DataFrame) -> "IsolationForestAnomalyDetector":
        start_total = time.perf_counter()

        t0 = time.perf_counter()
        self.preprocessor = DataPreprocessor(
            n_components=self.n_components,
            random_state=self.random_state
        )
        X_pca = self.preprocessor.fit_transform(benign_train_df)
        self.timings["preprocessing_time"] = time.perf_counter() - t0

        t_fit = time.perf_counter()
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state
        )
        self.model.fit(X_pca)
        self.timings["fit_time"] = time.perf_counter() - t_fit
        self.timings["total_runtime"] = time.perf_counter() - start_total
        return self

    def predict(
        self, test_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("Model must be fitted before predict.")

        t0 = time.perf_counter()
        X_test_pca = self.preprocessor.transform(test_df)
        predictions = self.model.predict(X_test_pca)
        decision_raw = self.model.decision_function(X_test_pca)
        self.timings["predict_time"] = time.perf_counter() - t0

        # For IsolationForest, decision_function returns average path length offset:
        # positive = normal, negative = anomalous.
        # Thus: anomaly_score = -decision_raw.
        anomaly_scores = -decision_raw
        return predictions, anomaly_scores, decision_raw


class LocalOutlierFactorDetector:
    """
    Local Outlier Factor (LOF) with novelty=True for out-of-sample anomaly detection.
    """

    def __init__(
        self,
        n_components: int = DEFAULT_PCA_COMPONENTS,
        n_neighbors: int = 20,
        contamination: Union[str, float] = "auto",
        random_state: int = RANDOM_STATE,
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.random_state = random_state

        self.name = "Local Outlier Factor"
        self.preprocessor: Optional[DataPreprocessor] = None
        self.model: Optional[LocalOutlierFactor] = None

        self.timings: Dict[str, float] = {
            "preprocessing_time": 0.0,
            "fit_time": 0.0,
            "predict_time": 0.0,
            "total_runtime": 0.0,
        }

    def fit(self, benign_train_df: pd.DataFrame) -> "LocalOutlierFactorDetector":
        start_total = time.perf_counter()

        t0 = time.perf_counter()
        self.preprocessor = DataPreprocessor(
            n_components=self.n_components,
            random_state=self.random_state
        )
        X_pca = self.preprocessor.fit_transform(benign_train_df)
        self.timings["preprocessing_time"] = time.perf_counter() - t0

        t_fit = time.perf_counter()
        n_samples = len(X_pca)
        n_neighbors = min(self.n_neighbors, max(1, n_samples - 1))

        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=self.contamination,
            novelty=True
        )
        self.model.fit(X_pca)
        self.timings["fit_time"] = time.perf_counter() - t_fit
        self.timings["total_runtime"] = time.perf_counter() - start_total
        return self

    def predict(
        self, test_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("Model must be fitted before predict.")

        t0 = time.perf_counter()
        X_test_pca = self.preprocessor.transform(test_df)
        predictions = self.model.predict(X_test_pca)
        decision_raw = self.model.decision_function(X_test_pca)
        self.timings["predict_time"] = time.perf_counter() - t0

        anomaly_scores = -decision_raw
        return predictions, anomaly_scores, decision_raw
