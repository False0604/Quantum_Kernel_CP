"""
tests/test_pipeline_e2e.py
End-to-end integration test running raw data -> preprocessing -> quantum kernel -> OCSVM -> evaluation.
"""

import numpy as np
import pandas as pd

from data_loader import TabularDataLoader, SyntheticAnomalyGenerator
from preprocessing import DataPreprocessor
from quantum_detector import QuantumKernelAnomalyDetector
from classical_detector import ClassicalRBFAnomalyDetector
from evaluation import AnomalyEvaluation


def test_full_pipeline_end_to_end():
    """Validates complete workflow on a synthetic IoT stream."""
    np.random.seed(42)
    # 1. Generate synthetic benign stream (15 samples, 6 features)
    benign_raw = pd.DataFrame(
        np.random.normal(loc=25.0, scale=3.0, size=(25, 6)),
        columns=[f"sensor_{i}" for i in range(6)]
    )

    # 2. Partition
    train_df, test_benign_df = TabularDataLoader.partition_benign(benign_raw, train_size=15, test_size=10)

    # 3. Create synthetic anomalies
    test_anomaly_df = SyntheticAnomalyGenerator.generate(test_benign_df, n_samples=10, perturbation_factor=4.0)

    # 4. Combine test
    test_combined = pd.concat([test_benign_df, test_anomaly_df], ignore_index=True)
    y_true = np.array([0] * 10 + [1] * 10)

    # 5. Fit & Predict Quantum
    q_det = QuantumKernelAnomalyDetector(n_components=2, reps=1, nu=0.1, train_sample_limit=15)
    q_det.fit(train_df)
    q_preds, q_scores, _ = q_det.predict(test_combined)

    assert len(q_preds) == 20
    assert len(q_scores) == 20

    # 6. Evaluate
    res = AnomalyEvaluation.evaluate(
        model_name="Quantum E2E Test",
        predictions=q_preds,
        anomaly_scores=q_scores,
        runtime_seconds=q_det.timings["total_runtime"],
        y_true=y_true,
    )

    assert res["has_ground_truth"] is True
    assert res["roc_auc"] is not None
    assert 0.0 <= res["roc_auc"] <= 1.0
