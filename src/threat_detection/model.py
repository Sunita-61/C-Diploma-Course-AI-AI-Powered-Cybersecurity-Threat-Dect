"""Supervised machine learning training, persistence, and pipeline management.

Builds leak-free Scikit-learn pipelines with Random Forest and Logistic Regression.
Supports both single-dataset splits and predefined benchmark partitions (e.g. UNSW-NB15).
Saves and loads pipelines and feature schemas with Joblib.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.threat_detection.config import (
    DEFAULT_MAX_ITER,
    DEFAULT_RANDOM_SEED,
    DEFAULT_RF_ESTIMATORS,
    DEFAULT_TEST_SIZE,
    MODELS_DIR,
)
from src.threat_detection.features import extract_feature_column_types
from src.threat_detection.preprocessing import build_preprocessor
from src.threat_detection.utils import (
    DataValidationError,
    ModelPipelineError,
    ensure_directory,
    get_current_utc_timestamp,
    logger,
    validate_dataframe_not_empty,
)

ModelArchitecture = Literal["random_forest", "logistic_regression"]


def train_supervised_model(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None,
    model_type: ModelArchitecture = "random_forest",
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_SEED,
    target_metadata: Optional[Dict[str, Any]] = None,
    n_estimators: int = DEFAULT_RF_ESTIMATORS,
    max_iter: int = DEFAULT_MAX_ITER,
) -> Tuple[Pipeline, Dict[str, Any], Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
    """Trains a supervised machine learning pipeline with strict leakage prevention.

    Supports two partitioning workflows:
    1. Predefined Partitions (e.g. UNSW-NB15 official train & test files):
       Passes X as X_train and explicitly provided X_test, y_test. Preprocessing is fitted ONLY on X_train.
    2. Single Dataset Workflow (e.g. synthetic telemetry):
       Performs a stratified holdout train_test_split.

    Args:
        X: Training features (or full dataset features if X_test is None).
        y: Training target labels (or full dataset labels if y_test is None).
        X_test: Optional predefined test feature set.
        y_test: Optional predefined test target set.
        model_type: 'random_forest' or 'logistic_regression'.
        test_size: Holdout fraction used if X_test is not provided.
        random_state: Random seed for reproducibility.
        target_metadata: Optional dictionary with target type ('binary' or 'multiclass') and classes.
        n_estimators: Number of trees for Random Forest.
        max_iter: Maximum iterations for Logistic Regression.

    Returns:
        Tuple containing:
        - Fitted sklearn Pipeline
        - Metadata dictionary detailing feature schema, hyperparameters, and split sizes
        - (X_train, X_test, y_train, y_test) data partitions

    Raises:
        DataValidationError: If dataset is too small or contains fewer than 2 distinct classes.
        ModelPipelineError: If training fails.
    """
    validate_dataframe_not_empty(X, context_name="Feature matrix X for model training")
    if y is None or len(y) != len(X):
        raise DataValidationError(f"Target y length ({len(y) if y is not None else 0}) must match X length ({len(X)}).")

    unique_classes = set(y.dropna().unique())
    if len(unique_classes) < 2:
        raise DataValidationError(
            f"Supervised classification requires at least 2 distinct classes, found: {unique_classes}"
        )

    if len(X) < 10:
        raise DataValidationError(f"Insufficient training samples: {len(X)} records provided. At least 10 required.")

    # Determine partition strategy
    if X_test is not None and y_test is not None:
        validate_dataframe_not_empty(X_test, context_name="Predefined test feature matrix X_test")
        if len(y_test) != len(X_test):
            raise DataValidationError(
                f"y_test length ({len(y_test)}) must match X_test length ({len(X_test)})."
            )
        # Align features: keep only features present in both train and test
        common_features = [col for col in X.columns if col in X_test.columns]
        if not common_features:
            raise DataValidationError("X_train and X_test share no common feature columns.")
        X_train_clean = X[common_features].copy()
        X_test_clean = X_test[common_features].copy()
        y_train_clean = y.copy()
        y_test_clean = y_test.copy()
        split_strategy = "predefined_official_files"
    else:
        # Single dataset: stratified train/test split
        stratify_target = y if y.value_counts().min() >= 2 else None
        X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_target
        )
        split_strategy = f"stratified_holdout_{int(test_size * 100)}pct"

    # Identify numeric and categorical columns strictly on training data
    numeric_cols, categorical_cols = extract_feature_column_types(X_train_clean)
    feature_schema = {
        "all_features": list(X_train_clean.columns),
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
    }

    # Build leak-free preprocessing transformer fitted ONLY on training set
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    # Initialize classifier
    if model_type == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        )
    elif model_type == "logistic_regression":
        classifier = LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
            class_weight="balanced",
        )
    else:
        raise ModelPipelineError(f"Unsupported model architecture: {model_type}")

    # Assemble complete pipeline: Preprocessing -> Estimator
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    logger.info(
        f"Training {model_type.replace('_', ' ').title()} ({split_strategy}) "
        f"on {len(X_train_clean)} train samples, evaluating on {len(X_test_clean)} test samples..."
    )

    try:
        pipeline.fit(X_train_clean, y_train_clean)
    except Exception as exc:
        raise ModelPipelineError(f"Failed during model fitting: {exc}") from exc

    # Determine classes actually learned
    model_classes = [str(c) for c in pipeline.classes_]

    metadata = {
        "model_type": model_type,
        "created_at": get_current_utc_timestamp(),
        "random_state": random_state,
        "split_strategy": split_strategy,
        "train_samples": int(len(X_train_clean)),
        "test_samples": int(len(X_test_clean)),
        "class_distribution_train": {str(k): int(v) for k, v in y_train_clean.value_counts().items()},
        "class_distribution_test": {str(k): int(v) for k, v in y_test_clean.value_counts().items()},
        "classes": model_classes,
        "target_metadata": target_metadata or {},
        "feature_schema": feature_schema,
    }

    logger.info(f"Model {model_type} trained successfully. Classes: {model_classes}")
    return pipeline, metadata, (X_train_clean, X_test_clean, y_train_clean, y_test_clean)


def save_model_pipeline(
    pipeline: Pipeline,
    metadata: Dict[str, Any],
    filepath: Optional[Union[str, Path]] = None,
) -> Path:
    """Serializes a fitted Scikit-learn Pipeline and its metadata dictionary using Joblib."""
    if filepath is None:
        ensure_directory(MODELS_DIR)
        model_name = metadata.get("model_type", "supervised_model")
        filepath = MODELS_DIR / f"{model_name}.joblib"
    else:
        filepath = Path(filepath)
        ensure_directory(filepath.parent)

    payload = {
        "pipeline": pipeline,
        "metadata": metadata,
    }
    joblib.dump(payload, filepath)
    logger.info(f"Model and metadata saved to: {filepath}")
    return filepath


def load_model_pipeline(filepath: Union[str, Path]) -> Tuple[Pipeline, Dict[str, Any]]:
    """Loads a previously saved model pipeline and metadata from disk."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at: {path}")

    try:
        payload = joblib.load(path)
        if not isinstance(payload, dict) or "pipeline" not in payload:
            raise ModelPipelineError(f"File at {path} does not match expected model payload format.")
        return payload["pipeline"], payload.get("metadata", {})
    except Exception as exc:
        raise ModelPipelineError(f"Failed to load model from {path}: {exc}") from exc
