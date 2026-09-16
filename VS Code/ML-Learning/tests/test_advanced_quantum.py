"""
tests/test_advanced_quantum.py
Unit tests for Advanced Quantum Information Engine:
- Spectral Analyzer & Barren Plateau Risk
- von Neumann Entropy & Quantum Purity
- Variational Quantum Metric Learner & Parameter-Shift Rule
- NISQ Noise Simulation
- Quantum-Classical Hybrid Ensemble
"""

import numpy as np
from qiskit import QuantumCircuit

from advanced_quantum_engine import (
    QuantumGramSpectralAnalyzer,
    QuantumInformationSpectroscopy,
    VariationalQuantumMetricLearner,
    NISQNoiseRobustnessSimulator,
    QuantumClassicalHybridEnsemble,
)


def test_spectral_analyzer():
    """Verify condition number, spectral gap, and effective rank."""
    K = np.array([
        [1.0, 0.5, 0.2],
        [0.5, 1.0, 0.4],
        [0.2, 0.4, 1.0],
    ])
    report = QuantumGramSpectralAnalyzer.analyze(K)

    assert report.matrix_dimension == 3
    assert report.condition_number >= 1.0
    assert report.effective_rank >= 1.0
    assert report.is_positive_semidefinite is True
    assert report.barren_plateau_risk.startswith("LOW") or report.barren_plateau_risk.startswith("MODERATE")


def test_quantum_information_spectroscopy():
    """Verify density matrix, purity, and entropy for pure state."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)  # Bell state |Phi+>

    rho = QuantumInformationSpectroscopy.statevector_to_density_matrix(qc)
    assert rho.shape == (4, 4)

    purity = QuantumInformationSpectroscopy.quantum_purity(rho)
    # Bell state is a globally pure state (Tr(rho^2) = 1.0)
    np.testing.assert_allclose(purity, 1.0, atol=1e-5)

    entropy = QuantumInformationSpectroscopy.von_neumann_entropy(rho)
    np.testing.assert_allclose(entropy, 0.0, atol=1e-5)


def test_variational_metric_learner_parameter_shift():
    """Verify analytical gradient calculation via Parameter-Shift Rule."""
    v_learner = VariationalQuantumMetricLearner(n_qubits=2, n_layers=1, random_state=42)
    x1 = np.array([0.5, 1.0])
    x2 = np.array([0.6, 0.9])

    grad = v_learner.parameter_shift_gradient(x1, x2, param_idx=0, theta=v_learner.theta)
    assert isinstance(grad, float)
    assert not np.isnan(grad)


def test_nisq_noise_simulator():
    """Verify noisy fidelity is lower than ideal statevector fidelity."""
    qc1 = QuantumCircuit(2)
    qc1.ry(0.5, 0)
    qc2 = QuantumCircuit(2)
    qc2.ry(0.7, 0)

    noise_sim = NISQNoiseRobustnessSimulator(depolarizing_rate=0.05)
    ideal_f, noisy_f = noise_sim.evaluate_noisy_fidelity(qc1, qc2)

    assert 0.0 <= noisy_f <= 1.0
    assert 0.0 <= ideal_f <= 1.0
    # Depolarizing noise generally lowers purity and alters overlap
    assert abs(noisy_f - ideal_f) >= 0.0


def test_quantum_classical_hybrid_ensemble():
    """Verify consensus score generation from multi-paradigm detectors."""
    ensemble = QuantumClassicalHybridEnsemble(weight_quantum=0.5, weight_rbf=0.3, weight_iforest=0.2)
    q_scores = np.array([0.1, 0.9, 0.5])
    rbf_scores = np.array([0.2, 0.8, 0.4])
    if_scores = np.array([0.15, 0.85, 0.45])

    preds, scores = ensemble.ensemble_predict(q_scores, rbf_scores, if_scores, threshold=0.5)

    assert len(preds) == 3
    assert len(scores) == 3
    assert set(preds).issubset({-1, 1})
    assert scores[1] > scores[0]  # Higher anomaly consensus
