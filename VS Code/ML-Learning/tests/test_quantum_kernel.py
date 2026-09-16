"""
tests/test_quantum_kernel.py
Unit tests for Qiskit ZZFeatureMap and FidelityQuantumKernel properties.
"""

import numpy as np
from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityQuantumKernel


def test_quantum_kernel_matrix_properties():
    """
    Test fundamental mathematical properties of FidelityQuantumKernel:
    1. Gram matrix shape: (N, N)
    2. Diagonal elements: K[i, i] approx 1.0 (state fidelity with itself is 1)
    3. Symmetry: K[i, j] approx K[j, i]
    4. Off-diagonal bounds: 0.0 <= K[i, j] <= 1.0
    """
    dimension = 2
    reps = 1
    fm = zz_feature_map(feature_dimension=dimension, reps=reps)
    kernel = FidelityQuantumKernel(feature_map=fm)

    # 4 sample points
    X = np.array([
        [0.2, 0.4],
        [0.8, 1.2],
        [-0.5, 0.1],
        [0.0, 0.0],
    ])

    K_train = kernel.evaluate(X)

    # 1. Shape
    assert K_train.shape == (4, 4), f"Expected (4, 4), got {K_train.shape}"

    # 2. Diagonal is approximately 1.0
    diag = np.diag(K_train)
    np.testing.assert_allclose(diag, np.ones(4), atol=1e-5)

    # 3. Symmetry
    np.testing.assert_allclose(K_train, K_train.T, atol=1e-5)

    # 4. Range [0, 1]
    assert np.all(K_train >= -1e-5)
    assert np.all(K_train <= 1.0 + 1e-5)


def test_quantum_kernel_rectangular_evaluation():
    """
    Test rectangular prediction kernel K_test = K(X_test, X_train):
    Expected shape: (M, N)
    """
    dimension = 2
    fm = zz_feature_map(feature_dimension=dimension, reps=1)
    kernel = FidelityQuantumKernel(feature_map=fm)

    X_train = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])  # N=3
    X_test = np.array([[0.15, 0.25], [0.9, 0.8]])             # M=2

    K_test = kernel.evaluate(X_test, X_train)

    assert K_test.shape == (2, 3), f"Expected (2, 3), got {K_test.shape}"
    assert np.all(K_test >= -1e-5)
    assert np.all(K_test <= 1.0 + 1e-5)
