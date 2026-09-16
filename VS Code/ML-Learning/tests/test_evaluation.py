"""
tests/test_evaluation.py
Unit tests for evaluation metrics and reporting.
"""

import numpy as np
from evaluation import AnomalyEvaluation


def test_anomaly_evaluation_perfect_separation():
    """Test metrics under perfect prediction."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # Model: 1 is normal, -1 is anomaly
    preds = np.array([1, 1, 1, 1, -1, -1, -1, -1])
    scores = np.array([-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0])

    res = AnomalyEvaluation.evaluate(
        model_name="PerfectDetector",
        predictions=preds,
        anomaly_scores=scores,
        runtime_seconds=0.12,
        y_true=y_true,
    )

    assert res["roc_auc"] == 1.0
    assert res["pr_auc"] == 1.0
    assert res["f1"] == 1.0
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["tp"] == 4
    assert res["fp"] == 0
    assert res["tn"] == 4
    assert res["fn"] == 0


def test_anomaly_evaluation_no_labels():
    """Test evaluation when ground truth is None."""
    preds = np.array([1, -1, 1, -1])
    scores = np.array([-0.5, 0.5, -0.2, 0.8])

    res = AnomalyEvaluation.evaluate(
        model_name="UnsupervisedModel",
        predictions=preds,
        anomaly_scores=scores,
        runtime_seconds=0.05,
        y_true=None,
    )

    assert res["has_ground_truth"] is False
    assert res["roc_auc"] is None
    assert res["anomaly_count"] == 2
    assert res["anomaly_rate"] == 0.5
