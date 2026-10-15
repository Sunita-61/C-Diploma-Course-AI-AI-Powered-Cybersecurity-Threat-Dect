"""Automated tests for synthetic simulation, prediction engine, and pipeline orchestration."""

from pathlib import Path
import pandas as pd
import pytest

from src.threat_detection.model import load_model_pipeline
from src.threat_detection.prediction import run_unified_detection
from src.threat_detection.simulation import generate_synthetic_events
from src.threat_detection.utils import DataValidationError


def test_synthetic_event_reproducibility():
    """Verifies that the same seed produces identical synthetic datasets."""
    df1 = generate_synthetic_events(count=50, seed=123)
    df2 = generate_synthetic_events(count=50, seed=123)

    pd.testing.assert_frame_equal(df1, df2)


def test_synthetic_failed_connections_constraint():
    """Verifies that failed_connections never exceeds connection_count in generated telemetry."""
    df = generate_synthetic_events(count=200, seed=42)

    assert (df["failed_connections"] <= df["connection_count"]).all()
    assert (df["simulated"] == True).all()


def test_missing_model_file_handling():
    """Verifies that attempting to load a non-existent model raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_model_pipeline("non_existent_model_binary.joblib")


def test_unified_detection_runs_without_ml_models():
    """Verifies that unified detection succeeds in rule-only mode when no models are loaded."""
    df = generate_synthetic_events(count=25, seed=99)

    enriched_df, alerts = run_unified_detection(
        df=df,
        supervised_pipeline=None,
        supervised_metadata=None,
        anomaly_pipeline=None,
        anomaly_metadata=None,
        enable_rules=True,
    )

    assert len(enriched_df) == 25
    assert "rule_findings_count" in enriched_df.columns
    assert isinstance(alerts, list)
