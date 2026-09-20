"""
quantum_detector.py
Quantum-Enabled Anomaly Detector using ZZFeatureMap, FidelityQuantumKernel,
and Precomputed One-Class SVM. Supports an ideal statevector simulator (default)
or an analytical depolarising-channel noise model for NISQ-style robustness studies.
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
    DEFAULT_NOISE_PROB,
    WARN_SAMPLE_THRESHOLD,
    HARD_SAMPLE_CAP,
    MODELS_DIR,
)
from preprocessing import DataPreprocessor


def apply_depolarising_noise(K_ideal: np.ndarray, noise_prob: float, n_qubits: int) -> np.ndarray:
    r"""
    Analytically applies a symmetric depolarising channel to both encoded states,
    yielding the noisy quantum kernel:

        K_noisy(x, y) = Tr( E(rho_x) E(rho_y) )
                      = (1 - p)^2 K_ideal(x, y) + 2 (1 - p) p / d + p^2 / d

    where d = 2^n_qubits. Follows directly from
        E(rho) = (1 - p) rho + (p / d) I,
        Tr(rho) = 1, Tr(I) = d.
    """
    if noise_prob <= 0.0:
        return K_ideal
    p = float(np.clip(noise_prob, 0.0, 1.0))
    d = 2 ** n_qubits
    factor_ideal = (1.0 - p) ** 2
    offset = 2.0 * (1.0 - p) * p / d + (p ** 2) / d
    return factor_ideal * K_ideal + offset


class QuantumKernelAnomalyDetector:
    """
    Quantum-kernel based anomaly detector for unsupervised IoT network intrusion detection.
    Maps preprocessed feature vectors to a quantum Hilbert space via ZZFeatureMap,
    computes quantum state fidelity as a kernel metric, and fits a precomputed One-Class SVM.

    noise_prob > 0 activates the analytical NISQ depolarising channel described in
    apply_depolarising_noise().
    """

    def __init__(
        self,
        n_components: int = DEFAULT_PCA_COMPONENTS,
        reps: int = DEFAULT_QUANTUM_REPS,
        nu: float = DEFAULT_NU,
        train_sample_limit: int = DEFAULT_TRAIN_SIZE,
        noise_prob: float = DEFAULT_NOISE_PROB,
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.reps = reps
        self.nu = nu
        self.train_sample_limit = train_sample_limit
        self.noise_prob = float(np.clip(noise_prob, 0.0, 1.0))
        self.random_state = random_state

        self.preprocessor: Optional[DataPreprocessor] = None
        self.feature_map = None
        self.kernel: Optional[FidelityQuantumKernel] = None
        self.model: Optional[OneClassSVM] = None

        self.X_train_ref: Optional[np.ndarray] = None
        self.K_train: Optional[np.ndarray] = None       # noisy kernel actually fed to OCSVM
        self.K_train_ideal: Optional[np.ndarray] = None # ideal kernel retained for diagnostics

        self.name = "Quantum Kernel OCSVM" + (
            f" (p={self.noise_prob:.3f})" if self.noise_prob > 0 else ""
        )

        self.timings: Dict[str, float] = {
            "preprocessing_time": 0.0,
            "train_kernel_time": 0.0,
            "fit_time": 0.0,
            "test_kernel_time": 0.0,
            "predict_time": 0.0,
            "total_runtime": 0.0,
        }

    def _build_quantum_kernel(self, dimension: int):
        self.feature_map = zz_feature_map(
            feature_dimension=dimension,
            reps=self.reps,
            entanglement="linear",
        )
        self.kernel = FidelityQuantumKernel(feature_map=self.feature_map)

    def _noisy(self, K: np.ndarray) -> np.ndarray:
        return apply_depolarising_noise(K, self.noise_prob, self.n_components)

    def fit(self, benign_train_df: pd.DataFrame) -> "QuantumKernelAnomalyDetector":
        start_total = time.perf_counter()

        n_samples = len(benign_train_df)
        if n_samples > HARD_SAMPLE_CAP:
            raise ValueError(
                f"Requested train sample size ({n_samples}) exceeds HARD_SAMPLE_CAP ({HARD_SAMPLE_CAP}). "
                f"Quantum kernel matrix computation scales quadratically (N^2)."
            )
        if n_samples > WARN_SAMPLE_THRESHOLD:
            warnings.warn(
                f"Training sample size ({n_samples}) exceeds recommended threshold ({WARN_SAMPLE_THRESHOLD})."
            )

        t0 = time.perf_counter()
        self.preprocessor = DataPreprocessor(n_components=self.n_components, random_state=self.random_state)
        X_pca = self.preprocessor.fit_transform(benign_train_df)
        self.timings["preprocessing_time"] = time.perf_counter() - t0

        if len(X_pca) > self.train_sample_limit:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(X_pca), self.train_sample_limit, replace=False)
            X_pca = X_pca[idx]

        self.X_train_ref = X_pca
        actual_dim = X_pca.shape[1]

        self._build_quantum_kernel(dimension=actual_dim)

        t_kern = time.perf_counter()
        self.K_train_ideal = self.kernel.evaluate(self.X_train_ref)
        self.timings["train_kernel_time"] = time.perf_counter() - t_kern

        assert self.K_train_ideal.shape == (len(self.X_train_ref), len(self.X_train_ref)), (
            f"K_train shape mismatch: expected ({len(self.X_train_ref)}, {len(self.X_train_ref)}), "
            f"got {self.K_train_ideal.shape}"
        )

        self.K_train = self._noisy(self.K_train_ideal)

        t_fit = time.perf_counter()
        self.model = OneClassSVM(kernel="precomputed", nu=self.nu)
        self.model.fit(self.K_train)
        self.timings["fit_time"] = time.perf_counter() - t_fit

        self.timings["total_runtime"] = time.perf_counter() - start_total
        return self

    def predict(
        self, test_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

        t_kern = time.perf_counter()
        K_test_ideal = self.kernel.evaluate(X_test_pca, self.X_train_ref)
        self.timings["test_kernel_time"] = time.perf_counter() - t_kern

        assert K_test_ideal.shape == (m_test, n_train), (
            f"K_test shape mismatch: expected ({m_test}, {n_train}), got {K_test_ideal.shape}"
        )

        K_test = self._noisy(K_test_ideal)

        t_pred = time.perf_counter()
        predictions = self.model.predict(K_test)
        decision_raw = self.model.decision_function(K_test)
        self.timings["predict_time"] = time.perf_counter() - t_pred

        anomaly_scores = -decision_raw
        return predictions, anomaly_scores, decision_raw

    def save(self, filepath: Union[str, Path]):
        target = Path(filepath)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n_components": self.n_components,
            "reps": self.reps,
            "nu": self.nu,
            "train_sample_limit": self.train_sample_limit,
            "noise_prob": self.noise_prob,
            "preprocessor": self.preprocessor,
            "model": self.model,
            "X_train_ref": self.X_train_ref,
            "timings": self.timings,
        }
        joblib.dump(payload, target)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "QuantumKernelAnomalyDetector":
        payload = joblib.load(filepath)
        instance = cls(
            n_components=payload["n_components"],
            reps=payload["reps"],
            nu=payload["nu"],
            train_sample_limit=payload["train_sample_limit"],
            noise_prob=payload.get("noise_prob", 0.0),
        )
        instance.preprocessor = payload["preprocessor"]
        instance.model = payload["model"]
        instance.X_train_ref = payload["X_train_ref"]
        instance.timings = payload["timings"]
        instance._build_quantum_kernel(dimension=instance.X_train_ref.shape[1])
        return instance
