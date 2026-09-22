"""
quantum_detector_v2.py
Configurable quantum kernel anomaly detector.

Compared with the baseline quantum_detector.py this version exposes:
  - feature_map: one of {"zz", "pauli_z", "z_only", "trainable"}
  - angle_range: optional (lo, hi) tuple that bounds the classical input angle
    to prevent rotation-gate wrap-around
  - reps: circuit repetition count
  - nu: OCSVM outlier bound
  - trainable_iters: gradient-ascent iterations for the variational feature map

For "trainable" the encoder is a small parameterised circuit whose angles are
optimised by the parameter-shift rule to maximise the mean fidelity between
benign training points (kernel-target alignment for the one-class case).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM

from qiskit.circuit.library import zz_feature_map, pauli_feature_map
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit_machine_learning.kernels import FidelityQuantumKernel

from config import (
    DEFAULT_PCA_COMPONENTS,
    DEFAULT_QUANTUM_REPS,
    DEFAULT_NU,
    DEFAULT_TRAIN_SIZE,
    DEFAULT_NOISE_PROB,
    HARD_SAMPLE_CAP,
)
from preprocessing import DataPreprocessor
from quantum_detector import apply_depolarising_noise
from advanced_quantum_engine import VariationalQuantumMetricLearner


SUPPORTED_FEATURE_MAPS = {"zz", "pauli_z", "z_only", "trainable"}


class QuantumKernelDetectorV2:
    """
    Flexible quantum-kernel one-class SVM. All fits are strictly on benign data.
    """

    def __init__(
        self,
        n_components: int = DEFAULT_PCA_COMPONENTS,
        feature_map: str = "zz",
        reps: int = DEFAULT_QUANTUM_REPS,
        nu: float = DEFAULT_NU,
        train_sample_limit: int = DEFAULT_TRAIN_SIZE,
        noise_prob: float = DEFAULT_NOISE_PROB,
        angle_range: Optional[Tuple[float, float]] = (0.0, np.pi),
        trainable_iters: int = 15,
        trainable_lr: float = 0.10,
        trainable_layers: int = 1,
        random_state: int = 42,
    ):
        if feature_map not in SUPPORTED_FEATURE_MAPS:
            raise ValueError(f"feature_map must be one of {SUPPORTED_FEATURE_MAPS}, got '{feature_map}'.")

        self.n_components = n_components
        self.feature_map_kind = feature_map
        self.reps = reps
        self.nu = nu
        self.train_sample_limit = train_sample_limit
        self.noise_prob = float(np.clip(noise_prob, 0.0, 1.0))
        self.angle_range = angle_range
        self.trainable_iters = trainable_iters
        self.trainable_lr = trainable_lr
        self.trainable_layers = trainable_layers
        self.random_state = random_state

        self.preprocessor: Optional[DataPreprocessor] = None
        self.feature_map = None            # qiskit circuit for the static maps
        self.kernel: Optional[FidelityQuantumKernel] = None
        self.variational: Optional[VariationalQuantumMetricLearner] = None
        self.model: Optional[OneClassSVM] = None

        self.X_train_ref: Optional[np.ndarray] = None
        self.K_train: Optional[np.ndarray] = None

        self.name = f"Quantum-{feature_map}({n_components}q, reps={reps}, nu={nu}, angle={angle_range})"

        self.timings: Dict[str, float] = {
            "preprocessing_time": 0.0,
            "kernel_time": 0.0,
            "fit_time": 0.0,
            "total_runtime": 0.0,
        }

    # ------------- Feature map construction -------------

    def _build_feature_map(self, dim: int) -> None:
        if self.feature_map_kind == "zz":
            self.feature_map = zz_feature_map(feature_dimension=dim, reps=self.reps, entanglement="linear")
            self.kernel = FidelityQuantumKernel(feature_map=self.feature_map)
        elif self.feature_map_kind == "pauli_z":
            # PauliFeatureMap with single-qubit Z rotations, no ZZ entanglement:
            self.feature_map = pauli_feature_map(
                feature_dimension=dim, reps=self.reps, paulis=["Z", "ZZ"], entanglement="linear"
            )
            self.kernel = FidelityQuantumKernel(feature_map=self.feature_map)
        elif self.feature_map_kind == "z_only":
            # Weakest-entanglement variant: Z rotations only (no cross-qubit gates)
            self.feature_map = pauli_feature_map(
                feature_dimension=dim, reps=self.reps, paulis=["Z"], entanglement="linear"
            )
            self.kernel = FidelityQuantumKernel(feature_map=self.feature_map)
        elif self.feature_map_kind == "trainable":
            self.variational = VariationalQuantumMetricLearner(
                n_qubits=dim,
                n_layers=self.trainable_layers,
                learning_rate=self.trainable_lr,
                max_iterations=self.trainable_iters,
                random_state=self.random_state,
            )
        else:  # pragma: no cover
            raise ValueError(self.feature_map_kind)

    # ------------- Kernel evaluation dispatch -------------

    def _kernel_matrix(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        if self.feature_map_kind == "trainable":
            assert self.variational is not None
            return self.variational.evaluate_kernel_matrix(X1, X2)
        assert self.kernel is not None
        return self.kernel.evaluate(X1, X2) if X2 is not None else self.kernel.evaluate(X1)

    # ------------- Fit / predict -------------

    def fit(self, benign_train_df: pd.DataFrame) -> "QuantumKernelDetectorV2":
        start_total = time.perf_counter()

        n_samples = len(benign_train_df)
        if n_samples > HARD_SAMPLE_CAP:
            raise ValueError(f"Training samples ({n_samples}) exceed HARD_SAMPLE_CAP ({HARD_SAMPLE_CAP}).")

        t0 = time.perf_counter()
        self.preprocessor = DataPreprocessor(
            n_components=self.n_components,
            random_state=self.random_state,
            angle_range=self.angle_range,
        )
        X = self.preprocessor.fit_transform(benign_train_df)
        self.timings["preprocessing_time"] = time.perf_counter() - t0

        if len(X) > self.train_sample_limit:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(X), self.train_sample_limit, replace=False)
            X = X[idx]
        self.X_train_ref = X

        self._build_feature_map(X.shape[1])

        # For the trainable path, first optimise theta on benign points
        if self.feature_map_kind == "trainable":
            assert self.variational is not None
            self.variational.fit_alignment(X[: min(20, len(X))])

        t_kern = time.perf_counter()
        K_ideal = self._kernel_matrix(X)
        self.timings["kernel_time"] = time.perf_counter() - t_kern

        self.K_train = apply_depolarising_noise(K_ideal, self.noise_prob, self.n_components)

        t_fit = time.perf_counter()
        self.model = OneClassSVM(kernel="precomputed", nu=self.nu)
        self.model.fit(self.K_train)
        self.timings["fit_time"] = time.perf_counter() - t_fit

        self.timings["total_runtime"] = time.perf_counter() - start_total
        return self

    def predict(
        self, test_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("Model must be fitted before predict.")

        X_test = self.preprocessor.transform(test_df)
        K_test_ideal = self._kernel_matrix(X_test, self.X_train_ref)
        K_test = apply_depolarising_noise(K_test_ideal, self.noise_prob, self.n_components)

        predictions = self.model.predict(K_test)
        decision_raw = self.model.decision_function(K_test)
        anomaly_scores = -decision_raw
        return predictions, anomaly_scores, decision_raw
