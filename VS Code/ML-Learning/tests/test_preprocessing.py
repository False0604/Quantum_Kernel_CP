"""
tests/test_preprocessing.py
Unit tests for leak-free preprocessor (StandardScaler + PCA).
"""

import numpy as np
import pandas as pd
import pytest

from preprocessing import DataPreprocessor


def test_preprocessor_fit_and_transform():
    """Test standard fit and transform with constant feature removal."""
    train_df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": [10.0, 20.0, 30.0, 40.0, 50.0],
        "const": [7.0, 7.0, 7.0, 7.0, 7.0],  # Constant
    })
    test_df = pd.DataFrame({
        "a": [1.5, 3.5],
        "b": [15.0, 35.0],
        "const": [7.0, 7.0],
    })

    prep = DataPreprocessor(n_components=2)
    X_train_pca = prep.fit_transform(train_df)

    assert prep.fitted is True
    assert "const" not in prep.active_features
    assert X_train_pca.shape == (5, 2)

    X_test_pca = prep.transform(test_df)
    assert X_test_pca.shape == (2, 2)


def test_preprocessor_leakage_protection():
    """Verify that transforming test data does NOT alter fitted scaler or PCA."""
    train_df = pd.DataFrame(np.random.normal(0, 1, size=(50, 4)), columns=list("ABCD"))
    test_df = pd.DataFrame(np.random.normal(100, 50, size=(20, 4)), columns=list("ABCD"))

    prep = DataPreprocessor(n_components=2)
    prep.fit(train_df)

    scaler_mean_before = prep.scaler.mean_.copy()
    scaler_var_before = prep.scaler.var_.copy()
    pca_comp_before = prep.pca.components_.copy()

    _ = prep.transform(test_df)

    np.testing.assert_array_equal(prep.scaler.mean_, scaler_mean_before)
    np.testing.assert_array_equal(prep.scaler.var_, scaler_var_before)
    np.testing.assert_array_equal(prep.pca.components_, pca_comp_before)


def test_preprocessor_explained_variance():
    """Check explained variance structure."""
    df = pd.DataFrame(np.random.rand(30, 6))
    prep = DataPreprocessor(n_components=3)
    prep.fit(df)
    ev = prep.get_explained_variance()

    assert ev["n_components"] == 3
    assert len(ev["explained_variance_ratio"]) == 3
    assert 0.0 <= ev["cumulative_explained_variance"] <= 1.01
