"""Feature engineering and feature extraction module for cybersecurity telemetry.

Safely calculates behavioral domain ratios (failure rate, transfer rate) without
division-by-zero errors or target leakage.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd

from src.threat_detection.config import WATCHLIST_PORTS
from src.threat_detection.utils import (
    DataValidationError,
    logger,
    validate_dataframe_not_empty,
)


def engineer_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes defensive behavioral features if supporting raw columns are present.

    Features engineered only if base columns exist:
    - failure_ratio = failed_connections / max(connection_count, 1)
    - bytes_per_connection = bytes_transferred / max(connection_count, 1)
    - bytes_per_second = bytes_transferred / max(duration_seconds, 0.001)
    - is_watchlist_port = 1 if destination_port in WATCHLIST_PORTS else 0

    Args:
        df: Input DataFrame containing raw network fields.

    Returns:
        New DataFrame with engineered features appended.
    """
    validate_dataframe_not_empty(df, context_name="Input to feature engineering")
    out = df.copy()

    # Safe failure ratio calculation
    if "failed_connections" in out.columns and "connection_count" in out.columns:
        denom = out["connection_count"].replace(0, 1)
        out["failure_ratio"] = (out["failed_connections"] / denom).clip(0.0, 1.0)

    # Safe bytes per connection
    if "bytes_transferred" in out.columns and "connection_count" in out.columns:
        denom = out["connection_count"].replace(0, 1)
        out["bytes_per_connection"] = (out["bytes_transferred"] / denom).clip(lower=0.0)

    # Safe bytes per second (transfer rate)
    if "bytes_transferred" in out.columns and "duration_seconds" in out.columns:
        denom = out["duration_seconds"].apply(lambda d: max(float(d), 0.001) if pd.notnull(d) else 1.0)
        out["bytes_per_second"] = (out["bytes_transferred"] / denom).clip(lower=0.0)

    # Watchlist port indicator
    if "destination_port" in out.columns:
        watchlist_set = set(WATCHLIST_PORTS.keys())
        out["is_watchlist_port"] = (
            out["destination_port"]
            .apply(lambda p: 1 if pd.notnull(p) and int(p) in watchlist_set else 0)
            .astype(int)
        )

    return out


def extract_feature_column_types(
    df: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    """Identifies and separates numerical and categorical feature columns for training.

    Excludes identifiers, raw labels, and metadata columns.

    Args:
        df: DataFrame of candidate features.

    Returns:
        Tuple of (numeric_column_names, categorical_column_names).
    """
    validate_dataframe_not_empty(df, context_name="DataFrame for feature type extraction")

    numeric_cols: List[str] = []
    categorical_cols: List[str] = []

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    logger.debug(
        f"Extracted {len(numeric_cols)} numeric and {len(categorical_cols)} categorical feature columns."
    )
    return numeric_cols, categorical_cols
