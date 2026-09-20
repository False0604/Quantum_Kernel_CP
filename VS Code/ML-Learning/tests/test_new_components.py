"""
tests/test_new_components.py
Tests for the new components added to the quantum anomaly detection stack:
  1. TON-IoT loader (TabularDataLoader.load_toniot_csv)
  2. Shallow autoencoder detector
  3. Quantum kernel noise-model transform
  4. Multi-seed benchmark harness with latency and persistence
"""

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

from data_loader import (
    TabularDataLoader,
    SyntheticBenignGenerator,
    SyntheticAnomalyGenerator,
)
from autoencoder_detector import AutoencoderAnomalyDetector
from quantum_detector import QuantumKernelAnomalyDetector, apply_depolarising_noise
from classical_detector import IsolationForestAnomalyDetector
from benchmark_harness import (
    run_multi_seed_benchmark,
    aggregated_to_dataframe,
    latency_to_dataframe,
    persist_run,
)


# ---------- 1. TON-IoT loader ----------

def test_toniot_loader_separates_benign_and_attack(tmp_path: Path):
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({
        "temperature": rng.normal(20, 2, n),
        "humidity": rng.normal(50, 5, n),
        "vendor": rng.choice(["A", "B", "C"], size=n),
        "label": [0] * 40 + [1] * 20,
        "type": ["normal"] * 40 + ["ddos"] * 20,
    })
    csv_path = tmp_path / "fridge_sensor.csv"
    df.to_csv(csv_path, index=False)

    benign, attack, meta = TabularDataLoader.load_toniot_csv(csv_path)
    assert meta["benign_rows"] == 40
    assert meta["attack_rows"] == 20
    assert benign.shape[0] == 40
    assert attack.shape[0] == 20
    # Categorical vendor column must have been numerically encoded
    assert benign.select_dtypes(include=[np.number]).shape[1] == benign.shape[1]


# ---------- 2. Shallow autoencoder detector ----------

def test_autoencoder_fit_predict_shapes():
    train_df = SyntheticBenignGenerator.generate(n_samples=120, n_features=10, random_state=1)
    test_df = SyntheticBenignGenerator.generate(n_samples=30, n_features=10, random_state=2)

    ae = AutoencoderAnomalyDetector(n_components=4, epochs=10, patience=5, random_state=0)
    ae.fit(train_df)
    preds, scores, raw = ae.predict(test_df)

    assert len(preds) == 30 and len(scores) == 30 and len(raw) == 30
    assert set(np.unique(preds)).issubset({-1, 1})
    assert (scores >= 0).all(), "Reconstruction MSE cannot be negative"


def test_autoencoder_flags_perturbed_samples_more_often():
    train_df = SyntheticBenignGenerator.generate(n_samples=300, n_features=8, random_state=42)
    benign_test = SyntheticBenignGenerator.generate(n_samples=60, n_features=8, random_state=123)
    attack_test = SyntheticAnomalyGenerator.generate(
        benign_test, n_samples=60, perturbation_factor=6.0, random_state=7
    )

    ae = AutoencoderAnomalyDetector(n_components=4, epochs=25, patience=5, random_state=0)
    ae.fit(train_df)
    _, benign_scores, _ = ae.predict(benign_test)
    _, attack_scores, _ = ae.predict(attack_test)
    # Attack reconstruction error should sit strictly higher on average
    assert attack_scores.mean() > benign_scores.mean()


# ---------- 3. Quantum noise-model transform ----------

def test_apply_depolarising_noise_bounds():
    K_ideal = np.array([[1.0, 0.4], [0.4, 1.0]])
    # p = 0 is a no-op
    np.testing.assert_allclose(apply_depolarising_noise(K_ideal, 0.0, 2), K_ideal)

    # p = 1 collapses to 1/d off- and on-diagonal
    K_full = apply_depolarising_noise(K_ideal, 1.0, 2)
    assert np.allclose(K_full, 1.0 / 4)

    # Intermediate p keeps result bounded within [0, 1]
    K_mid = apply_depolarising_noise(K_ideal, 0.3, 2)
    assert (K_mid >= 0).all() and (K_mid <= 1.0 + 1e-9).all()


def test_quantum_detector_ideal_vs_noisy_gram_differ():
    train_df = pd.DataFrame(np.random.RandomState(1).normal(0, 1, (14, 4)))
    ideal = QuantumKernelAnomalyDetector(
        n_components=2, reps=1, train_sample_limit=10, noise_prob=0.0
    ).fit(train_df)
    noisy = QuantumKernelAnomalyDetector(
        n_components=2, reps=1, train_sample_limit=10, noise_prob=0.05
    ).fit(train_df)
    # Off-diagonal magnitudes shrink under depolarising noise
    off_ideal = ideal.K_train_ideal - np.diag(np.diag(ideal.K_train_ideal))
    off_noisy = noisy.K_train - np.diag(np.diag(noisy.K_train))
    assert np.linalg.norm(off_noisy) <= np.linalg.norm(off_ideal) + 1e-6


# ---------- 4. Benchmark harness ----------

def test_multi_seed_benchmark_and_persistence(tmp_path: Path):
    benign_df = SyntheticBenignGenerator.generate(n_samples=200, n_features=8, random_state=42)

    def _if_factory(seed: int):
        return IsolationForestAnomalyDetector(n_components=4, random_state=seed)

    def _ae_factory(seed: int):
        return AutoencoderAnomalyDetector(n_components=4, epochs=8, patience=4, random_state=seed)

    aggregated, metadata = run_multi_seed_benchmark(
        detector_factories={
            "Isolation Forest": _if_factory,
            "Shallow Autoencoder": _ae_factory,
        },
        benign_df=benign_df,
        attack_df=None,
        dataset_label="test dataset",
        evaluation_type="synthetic",
        train_size=60,
        test_size=30,
        seeds=[0, 1],
        use_synthetic=True,
        latency_batch=40,
    )

    assert set(aggregated.keys()) == {"Isolation Forest", "Shallow Autoencoder"}
    for r in aggregated.values():
        assert len(r.per_seed) == 2
        assert not np.isnan(r.roc_auc_mean)
        assert r.latency is not None
        assert r.latency.us_per_sample_median > 0

    df = aggregated_to_dataframe(aggregated)
    assert "roc_auc" in df.columns

    lat = latency_to_dataframe(aggregated)
    assert len(lat) == 2

    run_dir = persist_run(aggregated, metadata, config={"note": "test"}, output_dir=tmp_path)
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "metrics_aggregated.csv").exists()
    assert (run_dir / "metrics_latency.csv").exists()
    assert (run_dir / "metrics_per_seed.csv").exists()
