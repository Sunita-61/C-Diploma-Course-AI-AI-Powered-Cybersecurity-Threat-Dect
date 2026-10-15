"""Automated tests for feature engineering and ratio calculations."""

import pandas as pd
import pytest

from src.threat_detection.features import (
    engineer_domain_features,
    extract_feature_column_types,
)


def test_engineer_domain_features_division_by_zero():
    """Verifies that ratio calculations handle zero connection counts without ZeroDivisionError."""
    df = pd.DataFrame({
        "connection_count": [0, 10],
        "failed_connections": [0, 5],
        "bytes_transferred": [0, 1000],
        "duration_seconds": [0.0, 2.0],
        "destination_port": [22, 80],
    })

    engineered = engineer_domain_features(df)

    assert "failure_ratio" in engineered.columns
    assert "bytes_per_connection" in engineered.columns
    assert "bytes_per_second" in engineered.columns
    assert "is_watchlist_port" in engineered.columns

    # Port 22 is in watchlist; Port 80 is standard web
    assert engineered.iloc[0]["is_watchlist_port"] == 1
    assert engineered.iloc[1]["is_watchlist_port"] == 0

    # Ensure no infinite or NaN values in computed ratios
    assert not engineered["failure_ratio"].isin([float("inf"), float("-inf")]).any()
    assert not engineered["bytes_per_second"].isin([float("inf"), float("-inf")]).any()


def test_extract_feature_column_types():
    """Verifies correct separation of numeric and non-numeric columns."""
    df = pd.DataFrame({
        "num_1": [1, 2, 3],
        "num_2": [1.5, 2.5, 3.5],
        "cat_1": ["A", "B", "C"],
    })
    num_cols, cat_cols = extract_feature_column_types(df)

    assert "num_1" in num_cols
    assert "num_2" in num_cols
    assert "cat_1" in cat_cols
