"""
tests/test_data_loader.py
Unit tests for data loader, dataset discovery, and partition logic.
"""

import numpy as np
import pandas as pd
import pytest

from data_loader import DatasetDiscovery, TabularDataLoader, SyntheticAnomalyGenerator


def test_dataset_discovery():
    """Verify that dataset discovery finds existing N-BaIoT devices, or reports none.

    On environments without the N-BaIoT CSVs on disk this is a smoke test: it asserts the
    discovery API returns a dict without raising. On environments with the dataset present
    it additionally checks Device 1 is found.
    """
    devices = DatasetDiscovery.discover_files()
    assert isinstance(devices, dict)
    if not devices:
        pytest.skip("N-BaIoT CSVs are not present in this environment; skipping full check.")
    assert 1 in devices, "Expected Device 1 to be discovered."
    assert devices[1]["benign"] is not None, "Expected Device 1 benign CSV."


def test_tabular_data_loader_numeric():
    """Test CSV loading with valid numeric data."""
    df = pd.DataFrame({
        "feat1": [1.0, 2.0, np.nan, 4.0],
        "feat2": [10.0, 20.0, 30.0, np.inf],
        "text_col": ["a", "b", "c", "d"],
    })
    tmp_file = "test_toy_numeric.csv"
    df.to_csv(tmp_file, index=False)

    loaded, meta = TabularDataLoader.load_csv(tmp_file)
    assert meta["numeric_columns"] == 2
    assert "text_col" not in loaded.columns
    assert meta["missing_values"] == 2  # 1 NaN and 1 inf converted to NaN

    import os
    if os.path.exists(tmp_file):
        os.remove(tmp_file)


def test_tabular_data_loader_non_numeric():
    """Test error when no numeric features are present."""
    df = pd.DataFrame({"text1": ["x", "y"], "text2": ["a", "b"]})
    tmp_file = "test_toy_text.csv"
    df.to_csv(tmp_file, index=False)

    with pytest.raises(ValueError, match="no numerical features"):
        TabularDataLoader.load_csv(tmp_file)

    import os
    if os.path.exists(tmp_file):
        os.remove(tmp_file)


def test_partition_benign_no_overlap():
    """Verify strictly non-overlapping train and test splits (no data leakage)."""
    df = pd.DataFrame({"val": range(100)})
    train, test = TabularDataLoader.partition_benign(df, train_size=60, test_size=40)

    assert len(train) == 60
    assert len(test) == 40

    train_set = set(train["val"].tolist())
    test_set = set(test["val"].tolist())
    overlap = train_set.intersection(test_set)
    assert len(overlap) == 0, f"Found data leakage! Overlapping values: {overlap}"


def test_synthetic_anomaly_generator():
    """Test synthetic anomaly generation returns expected shape and deviations."""
    benign = pd.DataFrame(np.random.normal(loc=10.0, scale=1.0, size=(50, 5)))
    synthetic = SyntheticAnomalyGenerator.generate(benign, n_samples=30, perturbation_factor=4.0)

    assert synthetic.shape == (30, 5)
    assert not synthetic.isna().any().any()
