"""
run_experiment.py
Command-line experiment runner and reproducible benchmark suite.
Executes Quantum Kernel OCSVM, Classical RBF OCSVM, and Isolation Forest
on identical data partitions, saving results, configurations, and figures.
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config import (
    DEFAULT_PCA_COMPONENTS,
    DEFAULT_QUANTUM_REPS,
    DEFAULT_TRAIN_SIZE,
    DEFAULT_TEST_SIZE,
    DEFAULT_NU,
    RANDOM_STATE,
    RESULTS_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    DEVICE_NAMES,
)
from data_loader import DatasetDiscovery, TabularDataLoader, SyntheticAnomalyGenerator
from quantum_detector import QuantumKernelAnomalyDetector
from classical_detector import ClassicalRBFAnomalyDetector, IsolationForestAnomalyDetector
from evaluation import AnomalyEvaluation
from visualization import Visualizer


def run_experiment(
    device_id: int = 1,
    attack_name: Optional[str] = "mirai.ack",
    use_synthetic: bool = False,
    n_components: int = DEFAULT_PCA_COMPONENTS,
    reps: int = DEFAULT_QUANTUM_REPS,
    train_size: int = DEFAULT_TRAIN_SIZE,
    test_size: int = DEFAULT_TEST_SIZE,
    nu: float = DEFAULT_NU,
    save_models: bool = False,
) -> pd.DataFrame:
    """
    Executes an end-to-end benchmark comparison across all three detectors.
    """
    print("\n=======================================================")
    print("QUANTUM vs CLASSICAL IoT ANOMALY DETECTION BENCHMARK")
    print("=======================================================\n")

    # 1. Discover and load datasets
    devices = DatasetDiscovery.discover_files()
    if device_id not in devices or devices[device_id]["benign"] is None:
        raise FileNotFoundError(f"Device {device_id} benign dataset not found.")

    benign_path = devices[device_id]["benign"]
    device_label = DEVICE_NAMES.get(device_id, f"Device {device_id}")
    print(f"Target Device: {device_id} ({device_label})")
    print(f"Loading benign traffic from: {benign_path.name}...")

    # Load only necessary rows to keep load time fast
    total_needed_benign = (train_size + test_size) * 4
    benign_df, benign_meta = TabularDataLoader.load_csv(benign_path, nrows=total_needed_benign)
    print(f"Raw Benign Features: {benign_df.shape[1]} | Available Samples: {len(benign_df)}")

    # Strictly partition benign into non-overlapping train and test splits
    benign_train, benign_test = TabularDataLoader.partition_benign(
        benign_df,
        train_size=train_size,
        test_size=test_size,
        random_state=RANDOM_STATE,
    )
    print(f"Benign Train Partition: {len(benign_train)} samples")
    print(f"Benign Test Partition:  {len(benign_test)} samples (zero leakage)")

    # 2. Acquire Attack Samples
    evaluation_type = "Synthetic anomaly validation"
    attack_test: Optional[pd.DataFrame] = None

    if not use_synthetic and attack_name and attack_name in devices[device_id]["attacks"]:
        attack_path = devices[device_id]["attacks"][attack_name]
        print(f"Loading real attack traffic from: {attack_path.name}...")
        raw_attack, _ = TabularDataLoader.load_csv(attack_path, nrows=test_size * 4)
        attack_test = TabularDataLoader.sample_attack(raw_attack, sample_size=test_size, random_state=RANDOM_STATE)
        evaluation_type = f"Real Attack ({attack_name})"
        print(f"Real Attack Test Samples: {len(attack_test)}")
    else:
        print("Real attack dataset not selected or unavailable. Using controlled synthetic anomaly validation...")
        attack_test = SyntheticAnomalyGenerator.generate(
            benign_test,
            n_samples=test_size,
            perturbation_factor=3.5,
            random_state=RANDOM_STATE,
        )
        evaluation_type = "Synthetic anomaly validation"
        print(f"Synthetic Anomaly Test Samples: {len(attack_test)}")

    # Combine test partitions and establish ground truth labels
    # Convention: 0 = Benign/Normal, 1 = Attack/Anomaly
    test_df = pd.concat([benign_test, attack_test], ignore_index=True)
    y_true = np.array([0] * len(benign_test) + [1] * len(attack_test), dtype=int)
    print(f"Total Test Samples: {len(test_df)} (Benign: {len(benign_test)}, Attack: {len(attack_test)})\n")

    # 3. Model Training & Inference (Identical Splits & Features)
    results_list: List[Dict[str, Any]] = []
    scores_dict: Dict[str, Any] = {}

    # Model 1: Quantum Kernel One-Class SVM
    print("-------------------------------------------------------")
    print(f"1. Training Quantum Kernel One-Class SVM (Qubits: {n_components}, Reps: {reps}, nu: {nu})...")
    q_detector = QuantumKernelAnomalyDetector(
        n_components=n_components,
        reps=reps,
        nu=nu,
        train_sample_limit=train_size,
    )
    t0 = time.perf_counter()
    q_detector.fit(benign_train)
    q_preds, q_scores, _ = q_detector.predict(test_df)
    q_total_time = time.perf_counter() - t0
    print(f"   Completed in {q_total_time:.2f}s (Kernel Eval: {q_detector.timings['train_kernel_time']:.2f}s)")

    q_eval = AnomalyEvaluation.evaluate(
        model_name="Quantum Kernel OCSVM",
        predictions=q_preds,
        anomaly_scores=q_scores,
        runtime_seconds=q_total_time,
        y_true=y_true,
        evaluation_type=evaluation_type,
        dataset_name=f"N-BaIoT Device {device_id}",
        train_samples=len(benign_train),
        test_samples=len(test_df),
        features_before_pca=benign_df.shape[1],
        quantum_features=n_components,
    )
    results_list.append(q_eval)
    scores_dict["Quantum Kernel OCSVM"] = (
        q_scores[:len(benign_test)],
        q_scores[len(benign_test):]
    )

    if save_models:
        q_path = MODELS_DIR / f"quantum_detector_dev{device_id}.joblib"
        q_detector.save(q_path)
        print(f"   Saved quantum detector artifact to {q_path.name}")

    # Model 2: Classical RBF One-Class SVM
    print("-------------------------------------------------------")
    print(f"2. Training Classical RBF One-Class SVM (PCA: {n_components}, nu: {nu})...")
    c_rbf = ClassicalRBFAnomalyDetector(
        n_components=n_components,
        nu=nu,
        random_state=RANDOM_STATE,
    )
    t0 = time.perf_counter()
    c_rbf.fit(benign_train)
    rbf_preds, rbf_scores, _ = c_rbf.predict(test_df)
    rbf_total_time = time.perf_counter() - t0
    print(f"   Completed in {rbf_total_time:.2f}s")

    rbf_eval = AnomalyEvaluation.evaluate(
        model_name="Classical RBF OCSVM",
        predictions=rbf_preds,
        anomaly_scores=rbf_scores,
        runtime_seconds=rbf_total_time,
        y_true=y_true,
        evaluation_type=evaluation_type,
        dataset_name=f"N-BaIoT Device {device_id}",
        train_samples=len(benign_train),
        test_samples=len(test_df),
        features_before_pca=benign_df.shape[1],
        quantum_features=n_components,
    )
    results_list.append(rbf_eval)
    scores_dict["Classical RBF OCSVM"] = (
        rbf_scores[:len(benign_test)],
        rbf_scores[len(benign_test):]
    )

    # Model 3: Isolation Forest
    print("-------------------------------------------------------")
    print(f"3. Training Isolation Forest (PCA: {n_components})...")
    c_iforest = IsolationForestAnomalyDetector(
        n_components=n_components,
        random_state=RANDOM_STATE,
    )
    t0 = time.perf_counter()
    c_iforest.fit(benign_train)
    if_preds, if_scores, _ = c_iforest.predict(test_df)
    if_total_time = time.perf_counter() - t0
    print(f"   Completed in {if_total_time:.2f}s")

    if_eval = AnomalyEvaluation.evaluate(
        model_name="Isolation Forest",
        predictions=if_preds,
        anomaly_scores=if_scores,
        runtime_seconds=if_total_time,
        y_true=y_true,
        evaluation_type=evaluation_type,
        dataset_name=f"N-BaIoT Device {device_id}",
        train_samples=len(benign_train),
        test_samples=len(test_df),
        features_before_pca=benign_df.shape[1],
        quantum_features=n_components,
    )
    results_list.append(if_eval)
    scores_dict["Isolation Forest"] = (
        if_scores[:len(benign_test)],
        if_scores[len(benign_test):]
    )

    # 4. Generate Visualizations
    print("\n-------------------------------------------------------")
    print("Generating and exporting publication figures...")
    Visualizer.plot_roc_curves(results_list, save_path=FIGURES_DIR / "roc_curves.png")
    Visualizer.plot_pr_curves(results_list, save_path=FIGURES_DIR / "pr_curves.png")
    Visualizer.plot_confusion_matrices(results_list, save_path=FIGURES_DIR / "confusion_matrices.png")
    Visualizer.plot_metric_comparison(results_list, save_path=FIGURES_DIR / "metric_comparison.png")
    Visualizer.plot_runtime_comparison(results_list, save_path=FIGURES_DIR / "runtime_comparison.png")
    Visualizer.plot_score_distributions(scores_dict, save_path=FIGURES_DIR / "score_distributions.png")
    print(f"All figures successfully saved to: {FIGURES_DIR}")

    # 5. Export Results Table & Metadata
    comparison_df = AnomalyEvaluation.to_dataframe(results_list)
    csv_out = RESULTS_DIR / "model_comparison.csv"
    comparison_df.to_csv(csv_out, index=False)
    print(f"Model comparison table saved to: {csv_out}")

    exp_config = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device_id": device_id,
        "device_name": device_label,
        "attack_name": attack_name,
        "evaluation_type": evaluation_type,
        "train_size": train_size,
        "test_size": test_size,
        "n_components": n_components,
        "reps": reps,
        "nu": nu,
        "random_state": RANDOM_STATE,
        "quantum_timings": q_detector.timings,
    }
    config_out = RESULTS_DIR / "experiment_config.json"
    with open(config_out, "w") as f:
        json.dump(exp_config, f, indent=2)
    print(f"Experiment configuration saved to: {config_out}")

    # 6. Display Clean Summary Table
    print("\n=======================================================")
    print("FINAL BENCHMARK RESULTS")
    print("=======================================================")
    disp_cols = ["model", "roc_auc", "pr_auc", "f1", "precision", "recall", "runtime_seconds"]
    print(comparison_df[disp_cols].to_string(index=False))
    print("=======================================================\n")

    # Scientific interpretation
    q_auc = q_eval.get("roc_auc")
    rbf_auc = rbf_eval.get("roc_auc")
    if q_auc is not None and rbf_auc is not None:
        diff = q_auc - rbf_auc
        if abs(diff) < 0.01:
            comparison_phrase = "demonstrated comparable discrimination performance"
        elif diff > 0:
            comparison_phrase = f"demonstrated a +{diff:.3f} ROC-AUC advantage"
        else:
            comparison_phrase = f"trailed the classical baseline by {abs(diff):.3f} ROC-AUC"

        print("SCIENTIFIC OBSERVATION:")
        print(
            f"The Quantum Kernel One-Class SVM achieved a ROC-AUC of {q_auc:.3f}, "
            f"compared with {rbf_auc:.3f} for the Classical RBF One-Class SVM and "
            f"{if_eval.get('roc_auc'):.3f} for Isolation Forest. Under this configuration, the quantum kernel "
            f"{comparison_phrase}. Computationally, the classical baseline ran in "
            f"{rbf_total_time:.2f}s versus {q_total_time:.2f}s for the simulated quantum kernel.\n"
        )

    return comparison_df


def run_ablation_sweep(
    device_id: int = 1,
    attack_name: str = "mirai.ack",
    qubit_list: List[int] = [2, 4, 6],
    train_size: int = 200,
    test_size: int = 100,
) -> pd.DataFrame:
    """
    Executes a controlled ablation sweep over quantum qubit dimensions.
    """
    print("\n=======================================================")
    print(f"RUNNING QUANTUM DIMENSION SWEEP (Qubits: {qubit_list})")
    print("=======================================================\n")

    sweep_results = []
    for k in qubit_list:
        print(f"\n--- Testing Dimension: {k} Qubits ---")
        df = run_experiment(
            device_id=device_id,
            attack_name=attack_name,
            n_components=k,
            train_size=train_size,
            test_size=test_size,
        )
        df["sweep_qubits"] = k
        sweep_results.append(df)

    combined = pd.concat(sweep_results, ignore_index=True)
    sweep_out = RESULTS_DIR / "ablation_sweep_results.csv"
    combined.to_csv(sweep_out, index=False)
    print(f"\nAblation sweep results saved to: {sweep_out}")
    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Quantum vs Classical IoT Anomaly Detection Experiment Runner"
    )
    parser.add_argument("--device", type=int, default=1, help="N-BaIoT Device ID (1-9)")
    parser.add_argument("--attack", type=str, default="mirai.ack", help="Attack type name or 'synthetic'")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic anomaly validation")
    parser.add_argument("-k", "--components", type=int, default=DEFAULT_PCA_COMPONENTS, help="PCA/Quantum dimensions (2,4,6,8)")
    parser.add_argument("--reps", type=int, default=DEFAULT_QUANTUM_REPS, help="Feature map circuit repetitions")
    parser.add_argument("-n", "--train-size", type=int, default=DEFAULT_TRAIN_SIZE, help="Number of benign training samples")
    parser.add_argument("-m", "--test-size", type=int, default=DEFAULT_TEST_SIZE, help="Number of test samples per class")
    parser.add_argument("--nu", type=float, default=DEFAULT_NU, help="One-Class SVM nu parameter")
    parser.add_argument("--save-models", action="store_true", help="Persist trained models to models/")
    parser.add_argument("--sweep", action="store_true", help="Run ablation sweep over quantum dimensions")

    args = parser.parse_args()

    if args.sweep:
        run_ablation_sweep(
            device_id=args.device,
            attack_name=args.attack,
            qubit_list=[2, 4, 6],
            train_size=args.train_size,
            test_size=args.test_size,
        )
    else:
        run_experiment(
            device_id=args.device,
            attack_name=args.attack,
            use_synthetic=args.synthetic,
            n_components=args.components,
            reps=args.reps,
            train_size=args.train_size,
            test_size=args.test_size,
            nu=args.nu,
            save_models=args.save_models,
        )


if __name__ == "__main__":
    main()
