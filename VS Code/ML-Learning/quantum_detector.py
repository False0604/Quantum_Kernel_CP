"""
quantum_detector.py
Quantum-Enabled Anomaly Detector using ZZFeatureMap, FidelityQuantumKernel,
and Precomputed One-Class SVM.
"""

from pathlib import Path
import time
from typing import Dict, Optional, Tuple, Union
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM

from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityQuantumKernel

from config import (
    DEFAULT_PCA_COMPONENTS,
    DEFAULT_QUANTUM_REPS,
    DEFAULT_NU,
    DEFAULT_TRAIN_SIZE,
    WARN_SAMPLE_THRESHOLD,
    HARD_SAMPLE_CAP,
    MODELS_DIR,
)
from preprocessing import DataPreprocessor


class QuantumKernelAnomalyDetector:
    """
    Quantum-kernel based anomaly detector for unsupervised IoT network intrusion detection.
    Maps preprocessed feature vectors to a quantum Hilbert space via ZZFeatureMap,
    computes quantum state fidelity as a kernel metric, and fits a precomputed One-Class SVM.
    """

    def __init__(
        self,
        n_components: int = DEFAULT_PCA_COMPONENTS,
        reps: int = DEFAULT_QUANTUM_REPS,
        nu: float = DEFAULT_NU,
        train_sample_limit: int = DEFAULT_TRAIN_SIZE,
    ):
        self.n_components = n_components
        self.reps = reps
        self.nu = nu
        self.train_sample_limit = train_sample_limit

        self.preprocessor: Optional[DataPreprocessor] = None
        self.feature_map = None
        self.kernel: Optional[FidelityQuantumKernel] = None
        self.model: Optional[OneClassSVM] = None

        # Reference vectors in PCA space used for kernel prediction
        self.X_train_ref: Optional[np.ndarray] = None
        self.K_train: Optional[np.ndarray] = None

        # Detailed runtime benchmarks (in seconds)
        self.timings: Dict[str, float] = {
            "preprocessing_time": 0.0,
            "train_kernel_time": 0.0,
            "fit_time": 0.0,
            "test_kernel_time": 0.0,
            "predict_time": 0.0,
            "total_runtime": 0.0,
        }

    def _build_quantum_kernel(self, dimension: int):
        """Constructs ZZFeatureMap and FidelityQuantumKernel with modern Qiskit API."""
        self.feature_map = zz_feature_map(
            feature_dimension=dimension,
            reps=self.reps,
            entanglement="linear",
        )
        self.kernel = FidelityQuantumKernel(feature_map=self.feature_map)

    def fit(self, benign_train_df: pd.DataFrame) -> "QuantumKernelAnomalyDetector":
        """
        Fits the preprocessor and trains the Quantum One-Class SVM.
        """
        start_total = time.perf_counter()

        n_samples = len(benign_train_df)
        if n_samples > HARD_SAMPLE_CAP:
            raise ValueError(
                f"Requested train sample size ({n_samples}) exceeds HARD_SAMPLE_CAP ({HARD_SAMPLE_CAP}). "
                f"Quantum kernel matrix computation scales quadratically (N^2). Use 100-500 samples."
            )
        if n_samples > WARN_SAMPLE_THRESHOLD:
            warnings.warn(
                f"Training sample size ({n_samples}) exceeds recommended threshold ({WARN_SAMPLE_THRESHOLD}). "
                f"Quantum simulation may take several minutes."
            )

        # 1. Preprocessing (StandardScaler + PCA)
        t0 = time.perf_counter()
        self.preprocessor = DataPreprocessor(n_components=self.n_components)
        X_pca = self.preprocessor.fit_transform(benign_train_df)
        self.timings["preprocessing_time"] = time.perf_counter() - t0

        # Enforce sample limit if DataFrame was not pre-sampled
        if len(X_pca) > self.train_sample_limit:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X_pca), self.train_sample_limit, replace=False)
            X_pca = X_pca[idx]

        self.X_train_ref = X_pca
        actual_dim = X_pca.shape[1]

        # 2. Build Quantum Kernel
        self._build_quantum_kernel(dimension=actual_dim)

        # 3. Compute Training Kernel Matrix: K(X_train, X_train)
        t_kern = time.perf_counter()
        self.K_train = self.kernel.evaluate(self.X_train_ref)
        self.timings["train_kernel_time"] = time.perf_counter() - t_kern

        # Sanity validation on Gram matrix
        assert self.K_train.shape == (len(self.X_train_ref), len(self.X_train_ref)), (
            f"K_train shape mismatch: expected ({len(self.X_train_ref)}, {len(self.X_train_ref)}), "
            f"got {self.K_train.shape}"
        )

        # 4. Fit One-Class SVM on precomputed kernel
        t_fit = time.perf_counter()
        self.model = OneClassSVM(kernel="precomputed", nu=self.nu)
        self.model.fit(self.K_train)
        self.timings["fit_time"] = time.perf_counter() - t_fit

        self.timings["total_runtime"] = time.perf_counter() - start_total
        return self

    def predict(
        self, test_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predicts whether test samples are normal (1) or anomaly (-1).
        Computes K_test = K(X_test, X_train_ref).
        
        Returns:
            predictions: ndarray of shape (M,) with +1 (Normal) and -1 (Anomaly)
            anomaly_scores: ndarray of shape (M,) where higher score = more anomalous
            decision_function: raw signed distance from decision boundary
        """
        if self.model is None or self.kernel is None or self.X_train_ref is None:
            raise RuntimeError("Model must be fitted before calling predict.")

        t0 = time.perf_counter()
        X_test_pca = self.preprocessor.transform(test_df)
        prep_time = time.perf_counter() - t0

        m_test = len(X_test_pca)
        n_train = len(self.X_train_ref)

        if m_test > HARD_SAMPLE_CAP:
            raise ValueError(
                f"Test sample size ({m_test}) exceeds safe cap ({HARD_SAMPLE_CAP})."
            )

        # CRITICAL: Evaluate pairwise kernel between test points and training reference points!
        # Shape must be (M_test, N_train)
        t_kern = time.perf_counter()
        K_test = self.kernel.evaluate(X_test_pca, self.X_train_ref)
        self.timings["test_kernel_time"] = time.perf_counter() - t_kern

        assert K_test.shape == (m_test, n_train), (
            f"K_test shape mismatch: expected ({m_test}, {n_train}), got {K_test.shape}"
        )

        t_pred = time.perf_counter()
        predictions = self.model.predict(K_test)
        decision_raw = self.model.decision_function(K_test)
        self.timings["predict_time"] = time.perf_counter() - t_pred

        # In sklearn OneClassSVM, decision_function > 0 indicates inlier (normal),
        # while negative values indicate outliers.
        # For standard anomaly detection evaluation (e.g. ROC-AUC), positive anomaly score
        # must correlate with being an anomaly. Thus: anomaly_score = -decision_raw.
        anomaly_scores = -decision_raw

        return predictions, anomaly_scores, decision_raw

    def save(self, filepath: Union[str, Path]):
        """Persists trained model and preprocessing pipeline to disk."""
        target = Path(filepath)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n_components": self.n_components,
            "reps": self.reps,
            "nu": self.nu,
            "train_sample_limit": self.train_sample_limit,
            "preprocessor": self.preprocessor,
            "model": self.model,
            "X_train_ref": self.X_train_ref,
            "timings": self.timings,
        }
        joblib.dump(payload, target)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "QuantumKernelAnomalyDetector":
        """Loads serialized model and reconstructs the quantum kernel."""
        payload = joblib.load(filepath)
        instance = cls(
            n_components=payload["n_components"],
            reps=payload["reps"],
            nu=payload["nu"],
            train_sample_limit=payload["train_sample_limit"],
        )
        instance.preprocessor = payload["preprocessor"]
        instance.model = payload["model"]
        instance.X_train_ref = payload["X_train_ref"]
        instance.timings = payload["timings"]

        # Reconstruct quantum kernel with same configuration
        instance._build_quantum_kernel(dimension=instance.X_train_ref.shape[1])
        return instance
