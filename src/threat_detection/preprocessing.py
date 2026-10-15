"""Data cleaning and preprocessing module for network telemetry.

Provides robust handling for missing values, infinite values, deduplication,
target leakage prevention, and builds leak-free Scikit-learn preprocessing pipelines.
Supports both binary ('label') and multiclass ('attack_cat') targets for UNSW-NB15 and synthetic telemetry.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.threat_detection.config import POTENTIAL_TARGET_COLUMNS
from src.threat_detection.utils import (
    DataValidationError,
    logger,
    validate_dataframe_not_empty,
)

# Columns that represent identifiers or metadata and should NOT be used as features
IDENTIFIER_COLUMNS: List[str] = [
    "id",
    "ID",
    "event_id",
    "timestamp",
    "source_ip",
    "destination_ip",
    "srcip",
    "dstip",
    "sport",
    "dsport",
    "scenario",
    "simulated",
]

# Set of columns that represent targets. When one is selected, others must be excluded
# to prevent severe target leakage (e.g. attack_cat leaking label, or vice versa).
TARGET_LEAKAGE_COLUMNS: List[str] = [
    "label",
    "Label",
    "attack_cat",
    "Attack_cat",
    "class",
    "Class",
    "target",
    "is_attack",
]


def clean_dataframe(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    replace_inf: bool = True,
) -> pd.DataFrame:
    """Performs preliminary data hygiene on an incoming DataFrame.

    - Cleans whitespace from column names.
    - Optionally identifies and removes duplicate records.
    - Replaces infinite values (+/- inf) with NaN so imputers can handle them safely.

    Args:
        df: Input DataFrame.
        drop_duplicates: Whether to drop exact duplicate rows.
        replace_inf: Whether to replace infinite floating point values with NaN.

    Returns:
        Cleaned copy of the DataFrame.
    """
    validate_dataframe_not_empty(df, context_name="DataFrame to clean")
    cleaned_df = df.copy()

    # Normalize column names (strip whitespace)
    cleaned_df.columns = [str(col).strip() for col in cleaned_df.columns]

    if replace_inf:
        num_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            cleaned_df[num_cols] = cleaned_df[num_cols].replace([np.inf, -np.inf], np.nan)

    if drop_duplicates:
        initial_count = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        dropped = initial_count - len(cleaned_df)
        if dropped > 0:
            logger.info(f"Dropped {dropped} duplicate rows.")

    return cleaned_df


def identify_target_column(
    df: pd.DataFrame, target_candidate: Optional[str] = None
) -> Optional[str]:
    """Detects target column name based on user input or common standard names.

    Args:
        df: Input DataFrame.
        target_candidate: User-specified target column name.

    Returns:
        Target column name string if identified, or None if not found.
    """
    if target_candidate and target_candidate in df.columns:
        return target_candidate

    for col in POTENTIAL_TARGET_COLUMNS:
        if col in df.columns:
            return col
    return None


def encode_binary_target(series: pd.Series) -> pd.Series:
    """Normalizes various common network dataset labels to binary 0 (Normal) and 1 (Suspicious).

    Handles strings such as 'normal', 'benign', 'attack', 'malicious', 'anomaly', etc.

    Args:
        series: Target column Series.

    Returns:
        Binary pd.Series containing integers 0 and 1.
    """
    if pd.api.types.is_numeric_dtype(series):
        unique_vals = set(series.dropna().unique())
        if unique_vals.issubset({0, 1}):
            return series.astype(int)

    lowered = series.astype(str).str.strip().str.lower()
    normal_values = {"0", "0.0", "normal", "benign", "legitimate", "good", "normal_web_browsing"}

    binary_series = lowered.apply(lambda val: 0 if val in normal_values else 1)
    return binary_series.astype(int)


def encode_target_series(
    series: pd.Series,
    target_name: str,
    df_context: Optional[pd.DataFrame] = None,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Encodes target series appropriately for binary or multiclass classification tasks.

    - If target is 'attack_cat', investigates dataset semantics:
      * Rows where attack_cat is missing and binary label == 0 are mapped to 'Normal' (justified by dataset design).
      * Contradictory rows where binary label == 1 but attack_cat is missing are flagged as 'Ambiguous_Attack'
        instead of silently mislabeling them as normal traffic.
      * Strips whitespace and standardizes category names (e.g. 'Backdoor' -> 'Backdoors').
    - If target is 'label' or binary, encodes as 0 (Normal) and 1 (Attack).

    Args:
        series: Target column Series.
        target_name: Name of target column.
        df_context: Optional surrounding DataFrame to cross-reference related target fields.

    Returns:
        Tuple of (processed_target_series, target_metadata_dict).
    """
    target_name_lower = target_name.lower().strip()

    if target_name_lower in ["attack_cat"]:
        clean_series = series.copy()
        
        # Check if missing or empty
        is_missing = clean_series.isna() | clean_series.astype(str).str.strip().isin(["", "-", "nan", "None"])

        # Cross-reference with binary label if available to prevent contradictory mislabeling
        has_binary_label = False
        binary_label_col = None
        if df_context is not None:
            for col in ["label", "Label"]:
                if col in df_context.columns:
                    has_binary_label = True
                    binary_label_col = col
                    break

        if has_binary_label and binary_label_col is not None:
            bin_labels = encode_binary_target(df_context[binary_label_col])
            contradictory_mask = is_missing & (bin_labels == 1)
            normal_missing_mask = is_missing & (bin_labels == 0)

            if contradictory_mask.sum() > 0:
                logger.warning(
                    f"Detected {int(contradictory_mask.sum())} contradictory records where binary label indicates attack (1) "
                    "but attack_cat was missing. Flagged as 'Ambiguous_Attack' rather than silently mislabeling as Normal."
                )
                clean_series = clean_series.mask(contradictory_mask, "Ambiguous_Attack")

            clean_series = clean_series.mask(normal_missing_mask, "Normal")
        else:
            # When binary label is absent, unattacked flows leave attack_cat blank
            clean_series = clean_series.mask(is_missing, "Normal")

        # Standardize strings and plurals
        clean_series = clean_series.astype(str).str.strip()
        clean_series = clean_series.replace({
            "Backdoor": "Backdoors",
            "": "Normal",
            "-": "Normal",
        })

        classes = sorted(list(clean_series.unique()))
        target_meta = {
            "target_type": "multiclass",
            "target_column": target_name,
            "classes": classes,
            "n_classes": len(classes),
        }
        return clean_series, target_meta
    else:
        # Binary target
        bin_series = encode_binary_target(series)
        target_meta = {
            "target_type": "binary",
            "target_column": target_name,
            "classes": [0, 1],
            "n_classes": 2,
        }
        return bin_series, target_meta


def separate_features_and_target(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    drop_identifiers: bool = True,
    return_metadata: bool = False,
) -> Union[Tuple[pd.DataFrame, Optional[pd.Series]], Tuple[pd.DataFrame, Optional[pd.Series], Dict[str, Any]]]:
    """Separates features from target label and removes non-predictive identifiers and leaking targets.

    Prevents target leakage by ensuring:
    1. The selected target column is removed from features.
    2. Non-selected target columns (e.g. 'attack_cat' when target is 'label') are stripped.
    3. Row IDs, IP addresses, and event identifiers are removed.

    Args:
        df: Cleaned input DataFrame.
        target_column: Column name holding class labels.
        drop_identifiers: Whether to drop identifier/meta columns.
        return_metadata: If True, returns (X, y, target_meta). If False, returns (X, y) for backward compatibility.

    Returns:
        (X, y) or (X, y, target_meta) depending on return_metadata.
    """
    validate_dataframe_not_empty(df, context_name="Input to feature separation")

    y: Optional[pd.Series] = None
    target_meta: Dict[str, Any] = {"target_type": "none", "target_column": None}
    cols_to_drop = []

    detected_target = identify_target_column(df, target_column)
    if detected_target is not None:
        y, target_meta = encode_target_series(df[detected_target], detected_target, df_context=df)
        cols_to_drop.append(detected_target)


        # Drop ALL other target leakage columns that exist in the dataframe
        for leak_col in TARGET_LEAKAGE_COLUMNS:
            if leak_col in df.columns and leak_col != detected_target and leak_col not in cols_to_drop:
                cols_to_drop.append(leak_col)
                logger.info(
                    f"Prevented target leakage: dropped '{leak_col}' while predicting target '{detected_target}'."
                )

    if drop_identifiers:
        for ident in IDENTIFIER_COLUMNS:
            if ident in df.columns and ident not in cols_to_drop:
                cols_to_drop.append(ident)
                logger.info(f"Excluded non-predictive identifier column '{ident}' from feature set.")

    X = df.drop(columns=cols_to_drop, errors="ignore")
    if return_metadata:
        return X, y, target_meta
    return X, y



def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """Constructs a Scikit-learn ColumnTransformer to prevent data leakage.

    Numeric pipeline: SimpleImputer(median) -> StandardScaler.
    Categorical pipeline: SimpleImputer(most_frequent) -> OneHotEncoder(handle_unknown='ignore').

    Args:
        numeric_features: List of numerical column names.
        categorical_features: List of categorical column names.

    Returns:
        Fitted or unfitted ColumnTransformer.
    """
    transformers = []

    if numeric_features:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipeline, numeric_features))

    if categorical_features:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipeline, categorical_features))

    if not transformers:
        raise DataValidationError("At least one numerical or categorical feature must be provided.")

    return ColumnTransformer(transformers=transformers, remainder="drop")
