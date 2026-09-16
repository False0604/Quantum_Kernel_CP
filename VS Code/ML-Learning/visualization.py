"""
visualization.py
Publication-grade visualizer for Quantum vs Classical anomaly detection experiments.
Generates ROC curves, PR curves, confusion matrices, score distributions, and benchmark charts.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")  # Headless rendering safe for CLI and Streamlit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import FIGURES_DIR


class Visualizer:
    """Creates clear, publication-ready figures comparing quantum and classical models."""

    @staticmethod
    def plot_roc_curves(
        results_list: List[Dict[str, Any]],
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Plots overlaid ROC curves for all evaluated models."""
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        plotted_any = False

        for idx, res in enumerate(results_list):
            curve = res.get("roc_curve")
            auc = res.get("roc_auc")
            model_name = res.get("model", f"Model {idx}")

            if curve is not None and auc is not None:
                plotted_any = True
                color = colors[idx % len(colors)]
                ax.plot(
                    curve["fpr"],
                    curve["tpr"],
                    label=f"{model_name} (AUC = {auc:.3f})",
                    color=color,
                    lw=2.2,
                )

        ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1.5, label="Random Guess (AUC = 0.500)")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate (FPR)", fontsize=12, fontweight="bold")
        ax.set_ylabel("True Positive Rate (TPR)", fontsize=12, fontweight="bold")
        ax.set_title("Receiver Operating Characteristic (ROC) Comparison", fontsize=14, fontweight="bold", pad=12)
        ax.legend(loc="lower right", frameon=True, fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        fig.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight")

        return fig

    @staticmethod
    def plot_pr_curves(
        results_list: List[Dict[str, Any]],
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Plots overlaid Precision-Recall curves."""
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

        for idx, res in enumerate(results_list):
            curve = res.get("pr_curve")
            pr_auc = res.get("pr_auc")
            model_name = res.get("model", f"Model {idx}")

            if curve is not None and pr_auc is not None:
                color = colors[idx % len(colors)]
                ax.plot(
                    curve["recall"],
                    curve["precision"],
                    label=f"{model_name} (PR-AUC = {pr_auc:.3f})",
                    color=color,
                    lw=2.2,
                )

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall (Detection Rate)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Precision", fontsize=12, fontweight="bold")
        ax.set_title("Precision-Recall (PR) Curve Comparison", fontsize=14, fontweight="bold", pad=12)
        ax.legend(loc="lower left", frameon=True, fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        fig.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight")

        return fig

    @staticmethod
    def plot_confusion_matrices(
        results_list: List[Dict[str, Any]],
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Renders side-by-side confusion matrices for all evaluated models."""
        valid_results = [r for r in results_list if r.get("confusion_matrix") is not None]
        n_models = len(valid_results)

        if n_models == 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "No ground-truth labels available for confusion matrix.", ha="center", va="center")
            return fig

        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5), dpi=150, squeeze=False)

        for idx, res in enumerate(valid_results):
            ax = axes[0, idx]
            cm = np.array(res["confusion_matrix"])
            model_name = res.get("model", f"Model {idx}")

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                ax=ax,
                xticklabels=["Normal", "Anomaly"],
                yticklabels=["Normal", "Anomaly"],
                annot_kws={"size": 13, "weight": "bold"},
            )
            ax.set_title(f"{model_name}\nF1: {res.get('f1', 'N/A')}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Predicted Label", fontsize=11)
            if idx == 0:
                ax.set_ylabel("True Label", fontsize=11)
            else:
                ax.set_ylabel("")

        fig.suptitle("Confusion Matrix Comparison", fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight")

        return fig

    @staticmethod
    def plot_metric_comparison(
        results_list: List[Dict[str, Any]],
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Bar chart comparing ROC-AUC, PR-AUC, and F1 across models."""
        models = []
        roc_aucs = []
        pr_aucs = []
        f1_scores = []

        for r in results_list:
            if r.get("has_ground_truth"):
                models.append(r.get("model"))
                roc_aucs.append(r.get("roc_auc") or 0.0)
                pr_aucs.append(r.get("pr_auc") or 0.0)
                f1_scores.append(r.get("f1") or 0.0)

        if not models:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "Supervised metrics unavailable.", ha="center", va="center")
            return fig

        x = np.arange(len(models))
        width = 0.25

        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
        r1 = ax.bar(x - width, roc_aucs, width, label="ROC-AUC", color="#2b5c8f")
        r2 = ax.bar(x, pr_aucs, width, label="PR-AUC", color="#4ba3e3")
        r3 = ax.bar(x + width, f1_scores, width, label="F1-Score", color="#80c271")

        # Label bars
        for rects in (r1, r2, r3):
            for rect in rects:
                h = rect.get_height()
                if h > 0.01:
                    ax.annotate(
                        f"{h:.3f}",
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8.5,
                        fontweight="bold",
                    )

        ax.set_ylabel("Score (0.0 to 1.0)", fontsize=12, fontweight="bold")
        ax.set_title("Model Detection Performance Comparison", fontsize=14, fontweight="bold", pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.legend(loc="upper right", frameon=True)
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)
        fig.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight")

        return fig

    @staticmethod
    def plot_runtime_comparison(
        results_list: List[Dict[str, Any]],
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Bar chart comparing runtime in seconds across models."""
        models = [r.get("model") for r in results_list]
        runtimes = [r.get("runtime_seconds", 0.0) for r in results_list]

        fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
        bars = ax.bar(models, runtimes, color=["#6baed6", "#74c476", "#fd8d3c"], width=0.45)

        for bar in bars:
            yval = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                yval,
                f"{yval:.2f} s",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )

        ax.set_ylabel("Execution Time (seconds)", fontsize=12, fontweight="bold")
        ax.set_title("Runtime & Computational Cost Comparison", fontsize=14, fontweight="bold", pad=12)
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)
        fig.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight")

        return fig

    @staticmethod
    def plot_score_distributions(
        scores_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plots histograms/KDE of anomaly scores for benign vs attack test samples.
        scores_dict: { model_name: (benign_scores, attack_scores) }
        """
        n_models = len(scores_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 4.2), dpi=150, squeeze=False)

        for idx, (name, (b_scores, a_scores)) in enumerate(scores_dict.items()):
            ax = axes[0, idx]
            ax.hist(b_scores, bins=25, alpha=0.6, color="green", label="Benign Test", density=True)
            if a_scores is not None and len(a_scores) > 0:
                ax.hist(a_scores, bins=25, alpha=0.6, color="red", label="Attack Test", density=True)

            ax.set_title(f"{name}\nScore Distribution", fontsize=11, fontweight="bold")
            ax.set_xlabel("Anomaly Score (Higher = More Anomalous)", fontsize=10)
            if idx == 0:
                ax.set_ylabel("Probability Density", fontsize=10)
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(True, linestyle=":", alpha=0.4)

        fig.suptitle("Anomaly Score Distribution (Benign vs Attack)", fontsize=13, fontweight="bold", y=1.02)
        fig.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight")

        return fig

    @staticmethod
    def plot_kernel_matrix_heatmap(
        K_matrix: np.ndarray,
        max_display: int = 50,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Renders a high-resolution heatmap of the Quantum Fidelity Gram Matrix K_train.
        Visually proves symmetry, unit diagonal (K_ii = 1.0), and quantum state overlap.
        """
        display_k = K_matrix[:max_display, :max_display] if len(K_matrix) > max_display else K_matrix
        fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=150)
        cax = ax.imshow(display_k, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
        cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("State Transition Fidelity |<Φ(x)|Φ(x')>|²", fontsize=10, fontweight="bold")

        ax.set_title(f"Quantum Fidelity Kernel Gram Matrix (First {len(display_k)}x{len(display_k)})", fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Sample Index j", fontsize=10)
        ax.set_ylabel("Sample Index i", fontsize=10)
        fig.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight")

        return fig

    @staticmethod
    def plot_pca_variance(
        explained_variance_ratio: List[float],
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Plots PCA Scree Plot showing variance explained by each principal component."""
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
        n = len(explained_variance_ratio)
        x = np.arange(1, n + 1)
        cumulative = np.cumsum(explained_variance_ratio)

        bars = ax.bar(x, explained_variance_ratio, color="#4ba3e3", alpha=0.8, label="Individual Variance")
        line = ax.plot(x, cumulative, color="#e6550d", marker="o", lw=2, label="Cumulative Variance")

        for i, (bar, cum) in enumerate(zip(bars, cumulative)):
            h = bar.get_height()
            ax.annotate(f"{h*100:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 2),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
            ax.annotate(f"{cum*100:.1f}%", xy=(x[i], cum), xytext=(0, 5),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color="#e6550d", fontweight="bold")

        ax.set_xlabel("Principal Component Index (Qubit Dimension)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Explained Variance Ratio", fontsize=10, fontweight="bold")
        ax.set_title("PCA Feature Space Variance Explained (Scree Plot)", fontsize=12, fontweight="bold", pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f"PC {i}" for i in x])
        ax.set_ylim(0, 1.12)
        ax.legend(loc="center right", frameon=True)
        ax.grid(True, linestyle=":", alpha=0.5)
        fig.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight")

        return fig

    @staticmethod
    def plot_ablation_sweep(
        sweep_df: pd.DataFrame,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Plots ROC-AUC and Runtime progression across quantum dimension sweep."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)

        # Plot 1: ROC-AUC vs Qubits
        for model in sweep_df["model"].unique():
            sub = sweep_df[sweep_df["model"] == model].sort_values("sweep_qubits")
            ax1.plot(sub["sweep_qubits"], sub["roc_auc"], marker="o", lw=2.2, label=model)

        ax1.set_xlabel("Quantum Register Dimension (Qubits)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("ROC-AUC Score", fontsize=11, fontweight="bold")
        ax1.set_title("Detection Accuracy vs. Quantum Dimensions", fontsize=12, fontweight="bold")
        ax1.set_ylim(0.4, 1.05)
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="lower right")

        # Plot 2: Runtime vs Qubits
        for model in sweep_df["model"].unique():
            sub = sweep_df[sweep_df["model"] == model].sort_values("sweep_qubits")
            ax2.plot(sub["sweep_qubits"], sub["runtime_seconds"], marker="s", lw=2.2, label=model)

        ax2.set_xlabel("Quantum Register Dimension (Qubits)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Runtime (seconds, log scale)", fontsize=11, fontweight="bold")
        ax2.set_yscale("log")
        ax2.set_title("Computational Cost vs. Quantum Dimensions", fontsize=12, fontweight="bold")
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="upper left")

        fig.tight_layout()
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight")

        return fig
