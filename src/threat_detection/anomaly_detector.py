"""Unsupervised anomaly detection module using Isolation Forest.

Identifies multivariate statistical outliers in network telemetry.
Anomalous instances represent uncommon parameter combinations, not confirmed attacks.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.threat_detection.config import (
    DEFAULT_ANOMALY_CONTAMINATION,
    DEFAULT_RANDOM_SEED,
    MODELS_DIR,
)
from src.threat_detection.utils import (
    DataValidationError,
    ModelPipelineError,
    ensure_directory,
    get_current_utc_timestamp,
    logger,
    validate_dataframe_not_empty,
)


def fit_anomaly_detector(
    X: pd.DataFrame,
    contamination: float = DEFAULT_ANOMALY_CONTAMINATION,
    random_state: int = DEFAULT_RANDOM_SEED,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Fits an Isolation Forest pipeline on numeric baseline features.

    Pipeline structure:
    1. SimpleImputer(median)
    2. StandardScaler()
    3. IsolationForest(contamination=contamination, random_state=random_state)

    Args:
        X: Feature DataFrame (numeric columns will be extracted).
        contamination: Expected proportion of outliers in the data (0.01 to 0.50).
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (fitted_pipeline, metadata_dictionary).

    Raises:
        DataValidationError: If X contains no numeric columns or insufficient rows.
    """
    validate_dataframe_not_empty(X, context_name="Anomaly detector training data")

    numeric_cols = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
    if not numeric_cols:
        raise DataValidationError("Isolation Forest requires at least one numeric feature column.")

    if len(X) < 10:
        raise DataValidationError(f"Insufficient samples for anomaly detection: {len(X)} provided.")

    if not 0.0 < contamination <= 0.5:
        raise DataValidationError(f"Contamination must be in range (0.0, 0.5], got {contamination}.")

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("detector", IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )),
    ])

    X_numeric = X[numeric_cols]
    logger.info(f"Fitting Isolation Forest on {len(X_numeric)} samples using features: {numeric_cols}")

    try:
        pipeline.fit(X_numeric)
    except Exception as exc:
        raise ModelPipelineError(f"Failed to fit Isolation Forest: {exc}") from exc

    metadata = {
        "model_type": "isolation_forest",
        "contamination": contamination,
        "random_state": random_state,
        "feature_columns": numeric_cols,
        "train_samples": int(len(X_numeric)),
        "created_at": get_current_utc_timestamp(),
    }

    return pipeline, metadata


def score_anomalies(
    pipeline: Pipeline,
    X: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Scores records using a fitted Isolation Forest pipeline.

    Scikit-learn IsolationForest outputs:
    - raw predict: +1 (normal), -1 (anomaly).
    We map this to binary indicator: 0 (Normal), 1 (Anomaly).
    - decision_function: average anomaly score. Lower scores indicate more abnormal patterns.
    We return inverted score where higher value = higher abnormality.

    Args:
        pipeline: Fitted Pipeline containing an IsolationForest estimator.
        X: DataFrame of candidate records.
        feature_columns: Expected numeric column names (if None, extracted from X).

    Returns:
        Tuple of (binary_anomaly_indicator: 0 or 1, continuous_abnormality_score).
    """
    validate_dataframe_not_empty(X, context_name="Input to anomaly scoring")

    if feature_columns is not None:
        missing = [c for c in feature_columns if c not in X.columns]
        if missing:
            raise DataValidationError(f"Input data missing expected anomaly features: {missing}")
        X_eval = X[feature_columns]
    else:
        num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        X_eval = X[num_cols]

    # Preprocess and score
    detector = pipeline.named_steps["detector"]
    preprocessor = Pipeline([
        ("imputer", pipeline.named_steps["imputer"]),
        ("scaler", pipeline.named_steps["scaler"]),
    ])
    X_scaled = preprocessor.transform(X_eval)

    # Scikit-learn: -1 for outlier, +1 for inlier
    raw_preds = detector.predict(X_scaled)
    anomaly_binary = np.where(raw_preds == -1, 1, 0)

    # Decision function: negative values indicate outliers; lower means more abnormal
    raw_scores = detector.decision_function(X_scaled)
    # Invert so higher score represents higher anomaly intensity
    anomaly_scores = -1.0 * raw_scores

    return anomaly_binary, anomaly_scores


def save_anomaly_pipeline(
    pipeline: Pipeline,
    metadata: Dict[str, Any],
    filepath: Optional[Union[str, Path]] = None,
) -> Path:
    """Saves fitted anomaly pipeline and metadata to disk."""
    if filepath is None:
        ensure_directory(MODELS_DIR)
        filepath = MODELS_DIR / "isolation_forest.joblib"
    else:
        filepath = Path(filepath)
        ensure_directory(filepath.parent)

    payload = {"pipeline": pipeline, "metadata": metadata}
    joblib.dump(payload, filepath)
    logger.info(f"Anomaly pipeline saved to: {filepath}")
    return filepath


def load_anomaly_pipeline(filepath: Union[str, Path]) -> Tuple[Pipeline, Dict[str, Any]]:
    """Loads fitted anomaly pipeline and metadata from disk."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Anomaly model not found at: {path}")

    try:
        payload = joblib.load(path)
        return payload["pipeline"], payload.get("metadata", {})
    except Exception as exc:
        raise ModelPipelineError(f"Failed loading anomaly pipeline from {path}: {exc}") from exc
