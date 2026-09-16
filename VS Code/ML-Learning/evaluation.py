"""
evaluation.py
Standardized, leakage-free evaluation metrics for IoT anomaly detection.
Computes ROC-AUC, PR-AUC, F1, Precision, Recall, and Confusion Matrix.
Strictly distinguishes real attack evaluations from synthetic anomaly validations.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)


class AnomalyEvaluation:
    """
    Evaluates anomaly detection performance against binary ground-truth labels.
    Binary convention:
        0 = Normal / Benign
        1 = Anomaly / Attack
    Model raw prediction convention:
        +1 = Normal
        -1 = Anomaly
    """

    @staticmethod
    def evaluate(
        model_name: str,
        predictions: np.ndarray,
        anomaly_scores: np.ndarray,
        runtime_seconds: float,
        y_true: Optional[np.ndarray] = None,
        evaluation_type: str = "Real Attack Evaluation",
        dataset_name: str = "N-BaIoT",
        train_samples: int = 0,
        test_samples: int = 0,
        features_before_pca: int = 0,
        quantum_features: int = 0,
    ) -> Dict[str, Any]:
        """
        Computes evaluation metrics and confusion matrix.
        Handles single-class or missing ground-truth safely.
        """
        n_total = len(predictions)
        # Convert model predictions (+1: normal, -1: anomaly) to binary (0: normal, 1: anomaly)
        y_pred_binary = np.where(predictions == -1, 1, 0)
        anomaly_count = int(np.sum(y_pred_binary == 1))
        anomaly_rate = float(anomaly_count / n_total) if n_total > 0 else 0.0

        result: Dict[str, Any] = {
            "model": model_name,
            "dataset": dataset_name,
            "evaluation_type": evaluation_type,
            "train_samples": train_samples,
            "test_samples": test_samples,
            "features_before_pca": features_before_pca,
            "quantum_features": quantum_features,
            "anomaly_count": anomaly_count,
            "anomaly_rate": round(anomaly_rate, 4),
            "runtime_seconds": round(runtime_seconds, 4),
            "has_ground_truth": False,
            "roc_auc": None,
            "pr_auc": None,
            "f1": None,
            "precision": None,
            "recall": None,
            "tp": None,
            "fp": None,
            "tn": None,
            "fn": None,
            "roc_curve": None,
            "pr_curve": None,
            "notes": "",
        }

        if y_true is None:
            result["notes"] = "Metrics unavailable: no ground-truth labels supplied."
            return result

        y_true = np.asarray(y_true, dtype=int)
        if len(y_true) != n_total:
            raise ValueError(
                f"Length mismatch: len(y_true)={len(y_true)} vs len(predictions)={n_total}"
            )

        unique_labels = np.unique(y_true)
        result["has_ground_truth"] = True

        # Calculate discrete classification metrics
        result["f1"] = round(float(f1_score(y_true, y_pred_binary, zero_division=0)), 4)
        result["precision"] = round(float(precision_score(y_true, y_pred_binary, zero_division=0)), 4)
        result["recall"] = round(float(recall_score(y_true, y_pred_binary, zero_division=0)), 4)

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred_binary, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        result["tn"] = int(tn)
        result["fp"] = int(fp)
        result["fn"] = int(fn)
        result["tp"] = int(tp)
        result["confusion_matrix"] = cm.tolist()

        # Continuous threshold metrics (require both normal and anomaly classes)
        if len(unique_labels) < 2:
            result["notes"] = "ROC-AUC and PR-AUC require both normal and anomaly test samples."
            return result

        try:
            auc = float(roc_auc_score(y_true, anomaly_scores))
            pr_auc = float(average_precision_score(y_true, anomaly_scores))
            result["roc_auc"] = round(auc, 4)
            result["pr_auc"] = round(pr_auc, 4)

            # Store curve coordinates for plotting
            fpr, tpr, _ = roc_curve(y_true, anomaly_scores)
            precisions, recalls, _ = precision_recall_curve(y_true, anomaly_scores)
            result["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
            result["pr_curve"] = {"precision": precisions.tolist(), "recall": recalls.tolist()}
        except Exception as e:
            result["notes"] = f"Curve calculation error: {str(e)}"

        return result

    @staticmethod
    def to_dataframe(results_list: List[Dict[str, Any]]) -> pd.DataFrame:
        """Converts a list of evaluation result dicts into a structured pandas DataFrame."""
        summary_cols = [
            "model",
            "dataset",
            "evaluation_type",
            "train_samples",
            "test_samples",
            "features_before_pca",
            "quantum_features",
            "roc_auc",
            "pr_auc",
            "f1",
            "precision",
            "recall",
            "runtime_seconds",
            "anomaly_count",
            "anomaly_rate",
            "tp",
            "fp",
            "tn",
            "fn",
        ]
        rows = []
        for r in results_list:
            row = {col: r.get(col) for col in summary_cols}
            rows.append(row)
        return pd.DataFrame(rows)
