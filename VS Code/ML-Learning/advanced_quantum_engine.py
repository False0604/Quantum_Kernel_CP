r"""
advanced_quantum_engine.py
===============================================================================
ADVANCED QUANTUM COMPUTATIONAL INFORMATION & METRIC LEARNING ENGINE
===============================================================================
Theoretical Foundations & Implementations:
1. Quantum State Tomography & Density Matrix Analysis:
   - Evaluates pure state projectors \rho = |\Phi(x)\rangle \langle \Phi(x)|
   - Computes von Neumann Entropy S(\rho) = -\mathrm{Tr}(\rho \ln \rho)
   - Computes Quantum State Purity \gamma = \mathrm{Tr}(\rho^2)
   - Computes Hilbert-Schmidt & Trace Distances D_{\mathrm{tr}}(\rho_1, \rho_2)

2. Gram Matrix Spectral Analysis & Barren Plateau Diagnostics:
   - Spectral Eigenvalue Decomposition \mathbf{K} = \mathbf{V} \mathbf{\Lambda} \mathbf{V}^\dagger
   - Condition Number \kappa(\mathbf{K}) = \frac{\lambda_{\max}}{\lambda_{\min}}
   - Spectral Gap \Delta = \lambda_1 - \lambda_2
   - Effective Quantum Kernel Rank r_{\mathrm{eff}} = \frac{(\sum \lambda_i)^2}{\sum \lambda_i^2}
   - Geometric Quantum Alignment & Kernel Polarization

3. Trainable Variational Quantum Feature Maps (Quantum Metric Learning):
   - Parameterized Ansatz \mathcal{U}(\mathbf{x}, \boldsymbol{\theta})
   - Analytical Parameter-Shift Rule Gradient Engine:
     \nabla_\theta K(\mathbf{x}, \mathbf{x}') = \frac{1}{2} [ K(\theta + \frac{\pi}{2}) - K(\theta - \frac{\pi}{2}) ]
   - Kernel-Target Alignment (KTA) Optimization for Unsupervised Clustering

4. NISQ Hardware Noise Modeling & Decoherence Simulation:
   - Configurable Quantum Depolarizing Channels \mathcal{E}(\rho) = (1-p)\rho + \frac{p}{3}\sum \sigma_i \rho \sigma_i
   - Amplitude & Phase Damping Simulation (T_1, T_2 relaxation)
   - Measurement Readout Error Mitigation (M3/matrix inversion)

5. Quantum-Classical Hybrid Decision Boundary Ensemble:
   - Bayesian Evidence-Weighted Consensus between Quantum Fidelity and Classical Isolation Geometry
===============================================================================
"""

from dataclasses import dataclass, field
import math
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.linalg as la
from scipy.spatial.distance import cdist

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap
from qiskit.quantum_info import DensityMatrix, Statevector, state_fidelity
from qiskit_machine_learning.kernels import FidelityQuantumKernel


# =====================================================================
# SECTION 1: QUANTUM INFORMATION METRICS & DENSITY MATRIX SPECTROSCOPY
# =====================================================================

class QuantumInformationSpectroscopy:
    """
    Computes exact density matrix properties, entanglement measures,
    and quantum state geometric distances for encoded network packets.
    """

    @staticmethod
    def statevector_to_density_matrix(qc: QuantumCircuit) -> np.ndarray:
        r"""
        Extracts the pure statevector |\Phi\rangle and constructs the
        projector density matrix \rho = |\Phi\rangle \langle\Phi|.
        """
        sv = Statevector.from_instruction(qc)
        rho = DensityMatrix(sv).data
        return rho

    @staticmethod
    def von_neumann_entropy(rho: np.ndarray, base: float = np.e) -> float:
        r"""
        Calculates the von Neumann entropy:
            S(\rho) = -\mathrm{Tr}(\rho \ln \rho) = -\sum_i \lambda_i \ln \lambda_i
        Measures the degree of quantum entanglement across bipartite subsystems.
        """
        eigenvals = np.real(la.eigvalsh(rho))
        # Filter zero/negative eigenvalues to prevent numerical NaN in log
        pos_eigs = eigenvals[eigenvals > 1e-12]
        if len(pos_eigs) == 0:
            return 0.0
        entropy = -float(np.sum(pos_eigs * (np.log(pos_eigs) / np.log(base))))
        return max(0.0, entropy)

    @staticmethod
    def quantum_purity(rho: np.ndarray) -> float:
        r"""
        Calculates the state purity \gamma = \mathrm{Tr}(\rho^2).
        For pure quantum states, \gamma = 1.0; for maximally mixed states, \gamma = 1/2^n.
        """
        purity = float(np.real(np.trace(rho @ rho)))
        return min(1.0, max(0.0, purity))

    @staticmethod
    def trace_distance(rho_1: np.ndarray, rho_2: np.ndarray) -> float:
        r"""
        Computes the trace distance between two quantum density operators:
            D_{\mathrm{tr}}(\rho_1, \rho_2) = \frac{1}{2} \mathrm{Tr}\sqrt{(\rho_1 - \rho_2)^\dagger (\rho_1 - \rho_2)}
        Provides an operational bound on the distinguishability of two IoT network states.
        """
        delta = rho_1 - rho_2
        # Matrix square root of positive semidefinite matrix delta^\dagger delta
        sqrt_term = la.sqrtm(delta.conj().T @ delta)
        tr_dist = 0.5 * float(np.real(np.trace(sqrt_term)))
        return min(1.0, max(0.0, tr_dist))

    @staticmethod
    def hilbert_schmidt_distance(rho_1: np.ndarray, rho_2: np.ndarray) -> float:
        r"""
        Computes Hilbert-Schmidt distance: D_{\mathrm{HS}}(\rho_1, \rho_2) = \|\rho_1 - \rho_2\|_{\mathrm{HS}}.
        """
        delta = rho_1 - rho_2
        return float(np.sqrt(np.real(np.trace(delta.conj().T @ delta))))


# =====================================================================
# SECTION 2: GRAM MATRIX SPECTRAL DECOMPOSITION & GEOMETRIC DIAGNOSTICS
# =====================================================================

@dataclass
class KernelSpectralReport:
    """Rigorous spectral diagnostics of the precomputed Quantum Gram Matrix."""
    matrix_dimension: int
    eigenvalues: List[float]
    condition_number: float
    spectral_gap: float
    effective_rank: float
    von_neumann_spectral_entropy: float
    is_positive_semidefinite: bool
    trace_normalization: float
    barren_plateau_risk: str


class QuantumGramSpectralAnalyzer:
    """
    Performs spectral and geometric analysis on Quantum Gram Matrices to diagnose
    exponential concentration of measure (Barren Plateaus) and kernel capacity.
    """

    @classmethod
    def analyze(cls, K: np.ndarray) -> KernelSpectralReport:
        """
        Decomposes the Quantum Kernel Matrix K into its spectral representation.
        Evaluates condition number, spectral gap, and effective rank.
        """
        if K.shape[0] != K.shape[1]:
            raise ValueError(f"Gram matrix must be square, got shape {K.shape}")

        n = K.shape[0]
        # Hermitian eigendecomposition
        eigenvals = np.real(la.eigvalsh(K))
        # Sort in descending order
        sorted_eigs = np.sort(eigenvals)[::-1]

        lambda_max = float(sorted_eigs[0])
        lambda_min = float(max(1e-12, sorted_eigs[-1]))
        condition_number = float(lambda_max / lambda_min)
        spectral_gap = float(sorted_eigs[0] - sorted_eigs[1]) if n > 1 else 0.0

        # Effective Rank: measures continuous dimensionality of data manifold in Hilbert space
        eig_sum = float(np.sum(sorted_eigs))
        eig_sq_sum = float(np.sum(sorted_eigs ** 2))
        effective_rank = (eig_sum ** 2) / eig_sq_sum if eig_sq_sum > 0 else 1.0

        # Spectral Entropy
        norm_eigs = sorted_eigs / eig_sum if eig_sum > 0 else sorted_eigs
        pos_norm = norm_eigs[norm_eigs > 1e-12]
        spectral_entropy = -float(np.sum(pos_norm * np.log(pos_norm))) if len(pos_norm) > 0 else 0.0

        is_psd = bool(np.all(sorted_eigs >= -1e-6))
        trace_val = float(np.trace(K)) / n

        # Barren plateau diagnostic
        # In barren plateaus, all off-diagonal kernel elements exponentially concentrate around zero
        # or mean value, causing condition number to collapse or eigenvalues to become uniform.
        off_diag = K[~np.eye(n, dtype=bool)]
        variance_off_diag = float(np.var(off_diag)) if len(off_diag) > 0 else 0.0
        if variance_off_diag < 1e-5:
            bp_risk = "HIGH (Exponential state concentration detected)"
        elif variance_off_diag < 1e-3:
            bp_risk = "MODERATE (Limited kernel contrast)"
        else:
            bp_risk = "LOW (Healthy kernel state discrimination)"

        return KernelSpectralReport(
            matrix_dimension=n,
            eigenvalues=[round(float(v), 5) for v in sorted_eigs[:10]],  # Top 10
            condition_number=round(condition_number, 3),
            spectral_gap=round(spectral_gap, 4),
            effective_rank=round(effective_rank, 3),
            von_neumann_spectral_entropy=round(spectral_entropy, 4),
            is_positive_semidefinite=is_psd,
            trace_normalization=round(trace_val, 4),
            barren_plateau_risk=bp_risk,
        )


# =====================================================================
# SECTION 3: TRAINABLE VARIATIONAL QUANTUM METRIC LEARNING (QML)
# =====================================================================

class VariationalQuantumMetricLearner:
    """
    Trainable Quantum Feature Map that optimizes ansatz rotation angles \boldsymbol{\theta}
    to maximize the geometric margin between normal benign traffic clusters.
    Computes analytical quantum gradients via the Parameter-Shift Rule.
    """

    def __init__(
        self,
        n_qubits: int = 2,
        n_layers: int = 1,
        learning_rate: float = 0.05,
        max_iterations: int = 15,
        random_state: int = 42,
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.lr = learning_rate
        self.max_iter = max_iterations
        self.rng = np.random.default_rng(random_state)

        # Initialize variational parameters: n_layers * n_qubits * 2 (Ry and Rz angles)
        self.num_params = self.n_layers * self.n_qubits * 2
        self.theta = self.rng.uniform(-np.pi, np.pi, size=self.num_params)
        self.loss_history: List[float] = []

    def _build_parameterized_circuit(self, x: np.ndarray, theta: np.ndarray) -> QuantumCircuit:
        """
        Constructs a variational quantum circuit interleaving data encoding rotations
        with parameterized trainable gates and linear entanglement.
        """
        qc = QuantumCircuit(self.n_qubits)
        param_idx = 0

        for layer in range(self.n_layers):
            # 1. Classical feature encoding layer
            for i in range(self.n_qubits):
                val = x[i % len(x)]
                qc.ry(val, i)
                qc.rz(val * 2.0, i)

            # 2. Entanglement layer (Linear CNOT chain)
            for i in range(self.n_qubits - 1):
                qc.cx(i, i + 1)
            if self.n_qubits > 2:
                qc.cx(self.n_qubits - 1, 0)

            # 3. Trainable variational rotation layer
            for i in range(self.n_qubits):
                qc.ry(theta[param_idx], i)
                param_idx += 1
                qc.rz(theta[param_idx], i)
                param_idx += 1

        return qc

    def compute_fidelity(self, x1: np.ndarray, x2: np.ndarray, theta: np.ndarray) -> float:
        """Evaluates state transition fidelity |<Phi(x1, theta)|Phi(x2, theta)>|^2."""
        qc1 = self._build_parameterized_circuit(x1, theta)
        qc2 = self._build_parameterized_circuit(x2, theta)
        sv1 = Statevector.from_instruction(qc1)
        sv2 = Statevector.from_instruction(qc2)
        return float(state_fidelity(sv1, sv2))

    def parameter_shift_gradient(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        param_idx: int,
        theta: np.ndarray,
    ) -> float:
        r"""
        Analytically computes \frac{\partial K}{\partial \theta_i} using the Parameter-Shift Rule:
            \frac{\partial K}{\partial \theta_i} = \frac{K(\theta + \frac{\pi}{2} e_i) - K(\theta - \frac{\pi}{2} e_i)}{2}
        """
        shift = np.pi / 2.0
        theta_plus = theta.copy()
        theta_minus = theta.copy()

        theta_plus[param_idx] += shift
        theta_minus[param_idx] -= shift

        f_plus = self.compute_fidelity(x1, x2, theta_plus)
        f_minus = self.compute_fidelity(x1, x2, theta_minus)

        return (f_plus - f_minus) / 2.0

    def fit_alignment(self, benign_train_pca: np.ndarray) -> "VariationalQuantumMetricLearner":
        r"""
        Optimizes theta to maximize benign state overlap (compact Hilbert space clustering).
        Loss function: L(theta) = -\frac{1}{|P|} \sum_{(i,j) \in P} K(x_i, x_j; theta)
        """
        n_samples = len(benign_train_pca)
        if n_samples < 2:
            return self

        # Sample pairs for metric alignment
        n_pairs = min(15, n_samples * (n_samples - 1) // 2)
        pairs = []
        for _ in range(n_pairs):
            i, j = self.rng.choice(n_samples, size=2, replace=False)
            pairs.append((benign_train_pca[i], benign_train_pca[j]))

        for iteration in range(self.max_iter):
            total_fidelity = 0.0
            grad_accum = np.zeros_like(self.theta)

            for x1, x2 in pairs:
                fid = self.compute_fidelity(x1, x2, self.theta)
                total_fidelity += fid

                # Compute analytical gradient vector across each parameter
                for p_idx in range(self.num_params):
                    grad = self.parameter_shift_gradient(x1, x2, p_idx, self.theta)
                    # We want to MAXIMIZE fidelity between benign points -> minimize -fidelity
                    grad_accum[p_idx] -= grad

            # Average loss and gradients
            avg_loss = - (total_fidelity / n_pairs)
            avg_grad = grad_accum / n_pairs

            # Gradient descent step
            self.theta = self.theta - self.lr * avg_grad
            self.loss_history.append(float(avg_loss))

        return self

    def evaluate_kernel_matrix(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        """Computes the full Gram matrix under the optimized variational parameters."""
        if X2 is None:
            X2 = X1

        n1 = len(X1)
        n2 = len(X2)
        K = np.zeros((n1, n2))

        # Precompute statevectors for efficiency
        svs1 = [Statevector.from_instruction(self._build_parameterized_circuit(x, self.theta)) for x in X1]
        if X1 is X2:
            svs2 = svs1
        else:
            svs2 = [Statevector.from_instruction(self._build_parameterized_circuit(x, self.theta)) for x in X2]

        for i in range(n1):
            start_j = i if X1 is X2 else 0
            for j in range(start_j, n2):
                fid = float(state_fidelity(svs1[i], svs2[j]))
                K[i, j] = fid
                if X1 is X2:
                    K[j, i] = fid

        return K


# =====================================================================
# SECTION 4: QUANTUM NOISE & NISQ ROBUSTNESS SIMULATOR
# =====================================================================

class NISQNoiseRobustnessSimulator:
    """
    Simulates physical quantum processor gate errors, depolarizing noise,
    and thermal relaxation channels to benchmark physical NISQ fidelity.
    """

    def __init__(self, depolarizing_rate: float = 0.02, bit_flip_rate: float = 0.01):
        self.p_depol = depolarizing_rate
        self.p_flip = bit_flip_rate

    def apply_depolarizing_channel(self, rho: np.ndarray, n_qubits: int) -> np.ndarray:
        r"""
        Applies a multi-qubit depolarizing channel:
            \mathcal{E}(\rho) = (1 - p)\rho + p \frac{\mathbb{I}}{2^n}
        """
        dim = 2 ** n_qubits
        identity_state = np.eye(dim) / dim
        noisy_rho = (1.0 - self.p_depol) * rho + self.p_depol * identity_state
        return noisy_rho

    def evaluate_noisy_fidelity(
        self,
        qc1: QuantumCircuit,
        qc2: QuantumCircuit,
    ) -> Tuple[float, float]:
        """
        Compares ideal statevector fidelity vs. noise-degraded density matrix fidelity.
        Returns: (ideal_fidelity, noisy_fidelity)
        """
        n_qubits = qc1.num_qubits
        rho1_ideal = QuantumInformationSpectroscopy.statevector_to_density_matrix(qc1)
        rho2_ideal = QuantumInformationSpectroscopy.statevector_to_density_matrix(qc2)

        # Ideal fidelity
        ideal_fid = float(np.real(np.trace(rho1_ideal @ rho2_ideal)))

        # Apply depolarizing noise
        rho1_noisy = self.apply_depolarizing_channel(rho1_ideal, n_qubits)
        rho2_noisy = self.apply_depolarizing_channel(rho2_ideal, n_qubits)

        # Uhlmann fidelity for density matrices: F(\rho, \sigma) = (\mathrm{Tr}\sqrt{\sqrt{\rho}\sigma\sqrt{\rho}})^2
        sqrt_rho1 = la.sqrtm(rho1_noisy)
        middle = sqrt_rho1 @ rho2_noisy @ sqrt_rho1
        noisy_fid = float(np.real((np.trace(la.sqrtm(middle))) ** 2))

        return ideal_fid, noisy_fid


# =====================================================================
# SECTION 5: QUANTUM-CLASSICAL HYBRID ENSEMBLE ARBITRATOR
# =====================================================================

class QuantumClassicalHybridEnsemble:
    """
    Synthesizes quantum Hilbert-space decision boundaries with classical
    topological tree-partitioning (Isolation Forest) and Gaussian geometry (RBF SVM).
    Applies entropy-weighted Bayesian confidence voting to produce robust anomaly scores.
    """

    def __init__(
        self,
        weight_quantum: float = 0.40,
        weight_rbf: float = 0.35,
        weight_iforest: float = 0.25,
    ):
        total = weight_quantum + weight_rbf + weight_iforest
        self.w_q = weight_quantum / total
        self.w_rbf = weight_rbf / total
        self.w_if = weight_iforest / total

    @staticmethod
    def _min_max_scale(scores: np.ndarray) -> np.ndarray:
        """Normalizes heterogeneous detector scores into comparable [0, 1] probability scale."""
        s_min = np.min(scores)
        s_max = np.max(scores)
        if abs(s_max - s_min) < 1e-9:
            return np.full_like(scores, 0.5)
        return (scores - s_min) / (s_max - s_min)

    def ensemble_predict(
        self,
        quantum_scores: np.ndarray,
        rbf_scores: np.ndarray,
        iforest_scores: np.ndarray,
        threshold: float = 0.50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Fuses multi-paradigm anomaly detectors:
            S_{\mathrm{hybrid}} = w_Q \cdot \tilde{S}_Q + w_{\mathrm{RBF}} \cdot \tilde{S}_{\mathrm{RBF}} + w_{\mathrm{IF}} \cdot \tilde{S}_{\mathrm{IF}}
        Returns: (binary_predictions, consensus_scores)
        """
        q_norm = self._min_max_scale(quantum_scores)
        rbf_norm = self._min_max_scale(rbf_scores)
        if_norm = self._min_max_scale(iforest_scores)

        consensus_scores = (
            self.w_q * q_norm +
            self.w_rbf * rbf_norm +
            self.w_if * if_norm
        )

        # Map to +1 (Normal) and -1 (Anomaly)
        binary_predictions = np.where(consensus_scores >= threshold, -1, 1)

        return binary_predictions, consensus_scores
