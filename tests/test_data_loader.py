"""Automated tests for data loading and schema inspection."""

from pathlib import Path
import pandas as pd
import pytest

from src.threat_detection.data_loader import get_dataset_summary, load_dataset
from src.threat_detection.utils import DataValidationError


def test_load_valid_csv(tmp_path: Path):
    """Verifies that a well-formed CSV loads cleanly into a DataFrame."""
    sample_file = tmp_path / "valid_traffic.csv"
    sample_data = pd.DataFrame({
        "event_id": ["E1", "E2"],
        "connection_count": [10, 20],
        "protocol": ["TCP", "UDP"],
    })
    sample_data.to_csv(sample_file, index=False)

    df = load_dataset(sample_file)
    assert len(df) == 2
    assert list(df.columns) == ["event_id", "connection_count", "protocol"]


def test_load_missing_file():
    """Verifies FileNotFoundError when attempting to load a non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_dataset("non_existent_file_path_12345.csv")


def test_load_empty_file(tmp_path: Path):
    """Verifies DataValidationError when loading a 0-byte file."""
    empty_file = tmp_path / "empty.csv"
    empty_file.touch()

    with pytest.raises(DataValidationError):
        load_dataset(empty_file)


def test_get_dataset_summary():
    """Verifies calculation of dimensions, nulls, duplicates, and data types."""
    df = pd.DataFrame({
        "feature_a": [1.0, 2.0, None, 1.0],
        "feature_b": ["A", "B", "C", "A"],
    })
    summary = get_dataset_summary(df)

    assert summary["total_rows"] == 4
    assert summary["total_columns"] == 2
    assert summary["duplicate_rows"] == 1
    assert summary["missing_counts"]["feature_a"] == 1
    assert "feature_a" in summary["numeric_columns"]
    assert "feature_b" in summary["categorical_columns"]
