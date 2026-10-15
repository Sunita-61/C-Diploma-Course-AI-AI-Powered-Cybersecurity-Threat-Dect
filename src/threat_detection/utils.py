"""Utility functions for logging, directory management, and input validation.

Provides shared helpers across threat detection modules, CLI scripts, and the web app.
"""

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd


class ThreatDetectionError(Exception):
    """Base exception class for threat detection pipeline errors."""
    pass


class DataValidationError(ThreatDetectionError):
    """Raised when dataset schema, content, or types fail validation."""
    pass


class ModelPipelineError(ThreatDetectionError):
    """Raised when machine learning training or prediction operations fail."""
    pass


def setup_logger(name: str = "threat_detection", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a standardized logger instance.

    Args:
        name: Name of the logger.
        level: Logging level (e.g. logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger instance with formatted output.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = setup_logger()


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensures a directory exists, creating all parents if necessary.

    Args:
        path: Directory path as string or Path.

    Returns:
        Path object pointing to the ensured directory.
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_current_utc_timestamp() -> str:
    """Returns current UTC timestamp formatted as an ISO 8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_dataframe_not_empty(df: pd.DataFrame, context_name: str = "Dataset") -> None:
    """Validates that a DataFrame is not None and contains rows.

    Args:
        df: Pandas DataFrame to check.
        context_name: Descriptive name for clear error messaging.

    Raises:
        DataValidationError: If df is None or empty.
    """
    if df is None:
        raise DataValidationError(f"{context_name} is None. Valid DataFrame expected.")
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(f"{context_name} must be a pandas DataFrame, got {type(df)}.")
    if df.empty:
        raise DataValidationError(f"{context_name} is empty (0 rows).")


def validate_required_columns(
    df: pd.DataFrame, required_columns: List[str], context_name: str = "Dataset"
) -> None:
    """Validates that required columns exist in the DataFrame.

    Args:
        df: Pandas DataFrame to inspect.
        required_columns: List of column names that must be present.
        context_name: Descriptive context for error reporting.

    Raises:
        DataValidationError: If any required column is missing.
    """
    validate_dataframe_not_empty(df, context_name)
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise DataValidationError(
            f"{context_name} is missing required columns: {missing}. Present columns: {list(df.columns)}"
        )
