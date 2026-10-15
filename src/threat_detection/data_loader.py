"""Data loading and inspection module for network telemetry files.

Safely loads CSV files, validates schemas, and provides structured summary statistics.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from src.threat_detection.utils import (
    DataValidationError,
    logger,
    validate_dataframe_not_empty,
)


def load_dataset(
    filepath: Union[str, Path],
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Loads a CSV dataset from disk with robust validation.

    Args:
        filepath: Path to the CSV file.
        nrows: Optional maximum number of rows to read.

    Returns:
        pd.DataFrame containing the loaded dataset.

    Raises:
        FileNotFoundError: If the file does not exist on disk.
        DataValidationError: If the file is empty, malformed, or unreadable.
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path}")

    if not path.is_file():
        raise DataValidationError(f"Expected a file at '{path}', but found a directory.")

    if path.stat().st_size == 0:
        raise DataValidationError(f"File at '{path}' is completely empty (0 bytes).")

    try:
        # Load CSV using pandas
        df = pd.read_csv(path, nrows=nrows)
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError(f"File '{path}' contains no data or empty columns.") from exc
    except pd.errors.ParserError as exc:
        raise DataValidationError(f"Failed to parse CSV at '{path}': {exc}") from exc
    except Exception as exc:
        raise DataValidationError(f"Unexpected error reading file '{path}': {exc}") from exc

    validate_dataframe_not_empty(df, context_name=f"Dataset loaded from '{path.name}'")
    logger.info(f"Loaded dataset '{path.name}' with {len(df)} rows and {len(df.columns)} columns.")
    return df


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Generates an inspection summary of row count, column types, and missing values.

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary with structured metadata about the dataset.
    """
    validate_dataframe_not_empty(df, context_name="Input for dataset summary")

    total_rows = int(len(df))
    total_cols = int(len(df.columns))
    duplicate_rows = int(df.duplicated().sum())

    missing_counts = df.isnull().sum().to_dict()
    missing_pct = {col: round(float(cnt / total_rows * 100), 2) for col, cnt in missing_counts.items()}

    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    categorical_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]

    return {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "duplicate_rows": duplicate_rows,
        "columns": list(df.columns),
        "data_types": dtypes,
        "missing_counts": missing_counts,
        "missing_percentages": missing_pct,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
    }
