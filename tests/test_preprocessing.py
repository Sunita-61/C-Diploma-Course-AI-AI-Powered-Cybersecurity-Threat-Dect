"""Automated tests for data cleaning, leakage prevention, and preprocessors."""

import numpy as np
import pandas as pd
import pytest

from src.threat_detection.preprocessing import (
    build_preprocessor,
    clean_dataframe,
    encode_binary_target,
    separate_features_and_target,
)


def test_clean_dataframe_dedup_and_inf():
    """Verifies that clean_dataframe removes exact duplicates and converts inf to NaN."""
    df = pd.DataFrame({
        "event_id": ["E1", "E1", "E2"],
        "metric": [10.0, 10.0, np.inf],
    })
    cleaned = clean_dataframe(df, drop_duplicates=True, replace_inf=True)

    assert len(cleaned) == 2
    assert np.isnan(cleaned.iloc[1]["metric"])


def test_target_encoding():
    """Verifies string labels are normalized into standard 0/1 binary values."""
    raw_labels = pd.Series(["normal", "Benign", "DDoS", "PortScan", "0", "1"])
    encoded = encode_binary_target(raw_labels)

    expected = pd.Series([0, 0, 1, 1, 0, 1])
    pd.testing.assert_series_equal(encoded, expected, check_names=False)


def test_leakage_prevention_feature_separation():
    """Verifies target labels and non-predictive identifiers are excluded from features."""
    df = pd.DataFrame({
        "event_id": ["E1", "E2", "E3"],
        "timestamp": ["2026-01-01T00:00:00Z"] * 3,
        "source_ip": ["192.0.2.1"] * 3,
        "destination_ip": ["203.0.113.1"] * 3,
        "connection_count": [5, 10, 15],
        "label": [0, 1, 0],
    })

    X, y = separate_features_and_target(df, target_column="label", drop_identifiers=True)

    assert "label" not in X.columns
    assert "event_id" not in X.columns
    assert "source_ip" not in X.columns
    assert "destination_ip" not in X.columns
    assert "connection_count" in X.columns
    assert y is not None
    assert len(y) == 3


def test_preprocessor_pipeline_transformation():
    """Verifies ColumnTransformer correctly imputes and encodes numerical and categorical data."""
    X_train = pd.DataFrame({
        "duration": [1.0, 2.0, np.nan, 4.0],
        "protocol": ["TCP", "UDP", "TCP", "ICMP"],
    })

    preprocessor = build_preprocessor(
        numeric_features=["duration"],
        categorical_features=["protocol"],
    )

    transformed = preprocessor.fit_transform(X_train)
    # Check that output is numeric 2D array without NaN
    assert isinstance(transformed, np.ndarray)
    assert not np.isnan(transformed).any()
    assert transformed.shape[0] == 4
