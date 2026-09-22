"""
sweep_quantum_variants.py

Runs an exhaustive comparison of quantum-kernel variants against classical
baselines on two synthetic data regimes:

  Regime A: Gaussian benign + heavy-shift anomalies (favours classical Euclidean)
  Regime B: Ring-manifold benign + off-manifold anomalies (small displacement)

The result is a single markdown-ready table with mean +/- std ROC-AUC over
three seeds for every configuration. No numbers are fabricated: this script is
the source of truth.
"""

from __future__ import annotations
import time
from typing import Callable, Dict

import numpy as np
import pandas as pd

from data_loader import SyntheticBenignGenerator, SyntheticAnomalyGenerator, TabularDataLoader
from classical_detector import (
    ClassicalRBFAnomalyDetector, IsolationForestAnomalyDetector, LocalOutlierFactorDetector,
)
from autoencoder_detector import AutoencoderAnomalyDetector
from quantum_detector_v2 import QuantumKernelDetectorV2
from benchmark_harness import run_multi_seed_benchmark, aggregated_to_dataframe


# ---------- data regimes ----------

def gaussian_regime(seed: int):
    """Gaussian benign + 3.5-sigma shifted anomalies."""
    benign = SyntheticBenignGenerator.generate(n_samples=350, n_features=15, random_state=seed)
    return benign, None, True  # attack_df, use_synthetic


def ring_manifold_regime(seed: int):
    """Benign points on a low-dim ring manifold in R^10; attacks slightly off-manifold."""
    rng = np.random.default_rng(seed)
    n_benign = 350
    theta = rng.uniform(0, 2 * np.pi, size=n_benign)
    # 2-D circle embedded in 10-D via random rotation
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    R = rng.normal(0.0, 1.0, size=(2, 10))
    benign = circle @ R + rng.normal(0.0, 0.05, size=(n_benign, 10))

    n_attack = 200
    theta_atk = rng.uniform(0, 2 * np.pi, size=n_attack)
    off_manifold = np.stack([1.6 * np.cos(theta_atk), 1.6 * np.sin(theta_atk)], axis=1)
    attack = off_manifold @ R + rng.normal(0.0, 0.05, size=(n_attack, 10))

    cols = [f"feat_{i:02d}" for i in range(10)]
    return pd.DataFrame(benign, columns=cols), pd.DataFrame(attack, columns=cols), False


# ---------- detector factory zoo ----------

def _classical_factories(qubits: int) -> Dict[str, Callable]:
    return {
        "Classical RBF OCSVM": lambda s: ClassicalRBFAnomalyDetector(n_components=qubits, nu=0.1, random_state=s),
        "Isolation Forest":    lambda s: IsolationForestAnomalyDetector(n_components=qubits, random_state=s),
        "Local Outlier Factor":lambda s: LocalOutlierFactorDetector(n_components=qubits, random_state=s),
        "Shallow Autoencoder": lambda s: AutoencoderAnomalyDetector(n_components=qubits, epochs=20, patience=5, random_state=s),
    }


def _quantum_factories(qubits: int) -> Dict[str, Callable]:
    """All quantum variants under test."""
    def mk(fm: str, angle_range, reps: int = 1, nu: float = 0.1, note: str = ""):
        label = f"Q-{fm} (angle={'yes' if angle_range else 'no'}, reps={reps}, nu={nu}){note}"
        return label, lambda s: QuantumKernelDetectorV2(
            n_components=qubits, feature_map=fm, reps=reps, nu=nu,
            train_sample_limit=60, noise_prob=0.0,
            angle_range=angle_range,
            random_state=s,
        )

    facts = dict([
        mk("zz",       None,           reps=1, nu=0.10),     # A) baseline (no angle scaling)
        mk("zz",       (0.0, np.pi),   reps=1, nu=0.10),     # B) angle scaling
        mk("zz",       (0.0, np.pi),   reps=2, nu=0.10),     # C) deeper ZZ + angle
        mk("pauli_z",  (0.0, np.pi),   reps=1, nu=0.10),     # D) PauliFeatureMap Z+ZZ
        mk("z_only",   (0.0, np.pi),   reps=1, nu=0.10),     # E) Z-only (no entanglement)
        mk("zz",       (0.0, np.pi),   reps=1, nu=0.30),     # F) higher nu
        mk("trainable",(0.0, np.pi),   reps=1, nu=0.10),     # G) trainable variational
    ])
    return facts


def _run_regime(regime_name: str, regime_fn, qubits: int, seeds=(0, 1, 2)):
    # Use the first seed's data as the common substrate. Regime factories are
    # deterministic in the seed, so the benchmark harness re-partitions per seed.
    benign, attack, use_synth = regime_fn(seeds[0])

    factories = {}
    factories.update(_classical_factories(qubits))
    factories.update(_quantum_factories(qubits))

    print(f"\n===== Regime: {regime_name}  ({len(factories)} detectors, {len(seeds)} seeds) =====")
    t0 = time.perf_counter()
    agg, _ = run_multi_seed_benchmark(
        factories, benign, attack,
        dataset_label=regime_name,
        evaluation_type="synthetic" if use_synth else "labelled",
        train_size=80, test_size=50, seeds=list(seeds),
        use_synthetic=use_synth,
        latency_batch=30, progress=lambda m: None,
    )
    print(f"wall_time_seconds={round(time.perf_counter() - t0, 1)}")

    df = aggregated_to_dataframe(agg)
    df = df.sort_values("roc_auc_mean", ascending=False)
    for _, row in df.iterrows():
        print(f"  {row['detector']:60s}  ROC-AUC {row['roc_auc']}   F1 {row['f1']}   runtime {row['runtime_s']}s")
    return df


def main():
    regime_a = _run_regime("Gaussian benign + shift anomaly", gaussian_regime, qubits=2)
    regime_b = _run_regime("Ring-manifold benign + off-manifold anomaly", ring_manifold_regime, qubits=2)

    print("\n=====  4-qubit sweep on ring manifold  =====")
    regime_b_4q = _run_regime("Ring manifold (4 qubits)", ring_manifold_regime, qubits=4)

    print("\nDone.")


if __name__ == "__main__":
    main()
