"""
tests/test_models.py
Unit tests for Quantum and Classical anomaly detector model interfaces.
"""

import numpy as np
import pandas as pd
import pytest

from quantum_detector import QuantumKernelAnomalyDetector
from classical_detector import ClassicalRBFAnomalyDetector, IsolationForestAnomalyDetector


def test_quantum_detector_fit_and_predict():
    """Verify QuantumKernelAnomalyDetector fit, predict, score count and bounds."""
    np.random.seed(42)
    train_df = pd.DataFrame(np.random.normal(0, 1, (12, 4)))
    test_df = pd.DataFrame(np.random.normal(0, 1, (6, 4)))

    detector = QuantumKernelAnomalyDetector(n_components=2, reps=1, nu=0.15, train_sample_limit=10)
    detector.fit(train_df)

    preds, scores, raw_dec = detector.predict(test_df)

    assert len(preds) == 6
    assert len(scores) == 6
    assert len(raw_dec) == 6
    assert set(preds).issubset({-1, 1})
    # Verify anomaly score inverse relationship with decision function
    np.testing.assert_allclose(scores, -raw_dec)


def test_classical_rbf_detector():
    """Verify ClassicalRBFAnomalyDetector."""
    train_df = pd.DataFrame(np.random.normal(0, 1, (15, 5)))
    test_df = pd.DataFrame(np.random.normal(0, 1, (8, 5)))

    detector = ClassicalRBFAnomalyDetector(n_components=2, nu=0.1)
    detector.fit(train_df)
    preds, scores, _ = detector.predict(test_df)

    assert len(preds) == 8
    assert len(scores) == 8
    assert set(preds).issubset({-1, 1})


def test_isolation_forest_detector():
    """Verify IsolationForestAnomalyDetector."""
    train_df = pd.DataFrame(np.random.normal(0, 1, (15, 5)))
    test_df = pd.DataFrame(np.random.normal(0, 1, (8, 5)))

    detector = IsolationForestAnomalyDetector(n_components=2)
    detector.fit(train_df)
    preds, scores, _ = detector.predict(test_df)

    assert len(preds) == 8
    assert len(scores) == 8
    assert set(preds).issubset({-1, 1})
