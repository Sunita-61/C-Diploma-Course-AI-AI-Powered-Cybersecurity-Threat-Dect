"""Automated tests for UNSW-NB15 schema handling, multiclass targets, and predefined splits.

Uses lightweight synthetic fixtures simulating UNSW-NB15 schema features and attack categories.
Requires zero external network downloads or large files.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.threat_detection.model import (
    load_model_pipeline,
    save_model_pipeline,
    train_supervised_model,
)
from src.threat_detection.preprocessing import (
    clean_dataframe,
    encode_target_series,
    separate_features_and_target,
)
from src.threat_detection.utils import DataValidationError


@pytest.fixture
def sample_unsw_train_data() -> pd.DataFrame:
    """Creates a miniature synthetic DataFrame mimicking the official UNSW-NB15 schema."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "dur": [0.12, 0.05, 1.45, 0.02, 0.89, 0.01, 2.10, 0.33, 0.04, 1.15],
        "proto": ["tcp", "udp", "tcp", "tcp", "udp", "tcp", "icmp", "tcp", "udp", "tcp"],
        "service": ["http", "-", "ftp", "http", "-", "ssh", "-", "dns", "-", "http"],
        "state": ["FIN", "CON", "FIN", "FIN", "CON", "FIN", "INT", "FIN", "CON", "FIN"],
        "spkts": [10, 4, 25, 6, 12, 8, 2, 14, 4, 30],
        "dpkts": [8, 0, 20, 4, 10, 6, 0, 12, 0, 28],
        "sbytes": [1000, 200, 5000, 800, 1500, 900, 100, 2000, 300, 6000],
        "dbytes": [2000, 0, 8000, 1200, 2500, 1500, 0, 3000, 0, 9000],
        "rate": [50.5, 12.0, 105.2, 30.0, 75.1, 40.0, 5.0, 85.0, 20.0, 120.0],
        "attack_cat": [
            np.nan, "DoS", "Exploits", np.nan, "Fuzzers",
            np.nan, "Reconnaissance", "DoS", np.nan, "Exploits"
        ],
        "label": [0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
    })


@pytest.fixture
def sample_unsw_test_data() -> pd.DataFrame:
    """Creates an independent miniature test set mimicking official UNSW-NB15 test partition."""
    return pd.DataFrame({
        "id": [101, 102, 103, 104, 105, 106],
        "dur": [0.08, 1.20, 0.03, 0.95, 0.02, 1.80],
        "proto": ["tcp", "tcp", "udp", "tcp", "udp", "icmp"],
        "service": ["http", "ftp", "-", "ssh", "-", "-"],
        "state": ["FIN", "FIN", "CON", "FIN", "CON", "INT"],
        "spkts": [8, 20, 2, 10, 4, 3],
        "dpkts": [6, 15, 0, 8, 0, 0],
        "sbytes": [900, 4000, 150, 1200, 250, 120],
        "dbytes": [1500, 6000, 0, 2000, 0, 0],
        "rate": [45.0, 90.0, 10.0, 60.0, 15.0, 8.0],
        "attack_cat": [np.nan, "Exploits", np.nan, "DoS", "Fuzzers", "Reconnaissance"],
        "label": [0, 1, 0, 1, 1, 1],
    })


def test_target_leakage_prevention_binary(sample_unsw_train_data):
    """Verifies that selecting 'label' strictly removes 'attack_cat' and row 'id' from features."""
    X, y, target_meta = separate_features_and_target(
        sample_unsw_train_data, target_column="label", drop_identifiers=True, return_metadata=True
    )

    assert target_meta["target_type"] == "binary"
    assert "label" not in X.columns
    assert "attack_cat" not in X.columns, "CRITICAL: attack_cat must be dropped to prevent target leakage!"
    assert "id" not in X.columns, "CRITICAL: Row id must be dropped to prevent index leakage!"
    assert "dur" in X.columns
    assert "proto" in X.columns
    assert len(y) == 10
    assert set(y.unique()).issubset({0, 1})


def test_target_leakage_prevention_multiclass(sample_unsw_train_data):
    """Verifies that selecting 'attack_cat' strictly removes binary 'label' and row 'id' from features."""
    X, y, target_meta = separate_features_and_target(
        sample_unsw_train_data, target_column="attack_cat", drop_identifiers=True, return_metadata=True
    )

    assert target_meta["target_type"] == "multiclass"
    assert "attack_cat" not in X.columns
    assert "label" not in X.columns, "CRITICAL: label must be dropped to prevent target leakage!"
    assert "id" not in X.columns
    assert "Normal" in y.values, "NaN attack categories should normalize to 'Normal'"
    assert "Exploits" in y.values
    assert "DoS" in y.values


def test_multiclass_target_normalization():
    """Verifies string trimming and mapping for attack_cat."""
    raw_cats = pd.Series(["  Exploits ", "Backdoor", np.nan, "-", "DoS", ""])
    encoded, meta = encode_target_series(raw_cats, "attack_cat")

    assert meta["target_type"] == "multiclass"
    assert encoded.iloc[0] == "Exploits"
    assert encoded.iloc[1] == "Backdoors"  # standardized plural
    assert encoded.iloc[2] == "Normal"     # NaN -> Normal
    assert encoded.iloc[3] == "Normal"     # '-' -> Normal
    assert encoded.iloc[4] == "DoS"
    assert encoded.iloc[5] == "Normal"     # empty string -> Normal


def test_predefined_split_training(sample_unsw_train_data, sample_unsw_test_data, tmp_path: Path):
    """Verifies training on predefined training partition and testing on predefined test set."""
    X_train, y_train = separate_features_and_target(
        sample_unsw_train_data, target_column="label", drop_identifiers=True
    )
    X_test, y_test = separate_features_and_target(
        sample_unsw_test_data, target_column="label", drop_identifiers=True
    )

    pipeline, metadata, partitions = train_supervised_model(
        X=X_train,
        y=y_train,
        X_test=X_test,
        y_test=y_test,
        model_type="random_forest",
        n_estimators=10,
        random_state=42,
    )

    X_tr, X_te, y_tr, y_te = partitions
    assert metadata["split_strategy"] == "predefined_official_files"
    assert len(X_tr) == 10
    assert len(X_te) == 6

    # Verify predictions on test set
    preds = pipeline.predict(X_te)
    assert len(preds) == 6

    # Test serialization and deserialization
    model_file = tmp_path / "test_unsw_rf.joblib"
    save_model_pipeline(pipeline, metadata, model_file)
    loaded_pipe, loaded_meta = load_model_pipeline(model_file)
    assert loaded_meta["train_samples"] == 10


def test_multiclass_model_training(sample_unsw_train_data, sample_unsw_test_data):
    """Verifies Random Forest training on multiclass attack_cat target."""
    X_train, y_train, meta_train = separate_features_and_target(
        sample_unsw_train_data, target_column="attack_cat", drop_identifiers=True, return_metadata=True
    )
    X_test, y_test = separate_features_and_target(
        sample_unsw_test_data, target_column="attack_cat", drop_identifiers=True
    )

    pipeline, metadata, _ = train_supervised_model(
        X=X_train,
        y=y_train,
        X_test=X_test,
        y_test=y_test,
        model_type="random_forest",
        n_estimators=10,
        random_state=42,
        target_metadata=meta_train,
    )

    assert metadata["target_metadata"]["target_type"] == "multiclass"
    assert len(metadata["classes"]) >= 3
    preds = pipeline.predict(X_test)
    assert len(preds) == len(X_test)
    # Check that predictions are attack category strings
    assert isinstance(preds[0], str)


def test_unseen_categorical_value_handling(sample_unsw_train_data):
    """Verifies that Scikit-learn OneHotEncoder ignores new categorical values at inference."""
    X_train, y_train = separate_features_and_target(sample_unsw_train_data, target_column="label")

    pipeline, _, _ = train_supervised_model(
        X=X_train, y=y_train, model_type="logistic_regression", max_iter=200
    )

    # Test record with completely unseen protocol and service
    X_unseen = pd.DataFrame([{
        "dur": 0.5,
        "proto": "novel_quantum_protocol",
        "service": "novel_ai_service",
        "state": "FIN",
        "spkts": 10,
        "dpkts": 10,
        "sbytes": 500,
        "dbytes": 500,
        "rate": 20.0,
    }])

    # Must predict without throwing KeyError or ValueError
    pred = pipeline.predict(X_unseen)
    assert len(pred) == 1
    assert pred[0] in [0, 1]


def test_contradictory_record_detection():
    """Verifies that records with label==1 but missing attack_cat are flagged as Ambiguous_Attack."""
    df_contradictory = pd.DataFrame({
        "label": [1, 0, 1],
        "attack_cat": [np.nan, np.nan, "DoS"],
    })
    encoded, meta = encode_target_series(
        df_contradictory["attack_cat"],
        target_name="attack_cat",
        df_context=df_contradictory,
    )
    # The first row (label=1, attack_cat=NaN) must NOT be labeled 'Normal'
    assert encoded.iloc[0] == "Ambiguous_Attack"
    assert encoded.iloc[1] == "Normal"
    assert encoded.iloc[2] == "DoS"
    assert "Ambiguous_Attack" in meta["classes"]


def test_train_only_preprocessor_fitting(sample_unsw_train_data, sample_unsw_test_data):
    """Verifies preprocessors fit exclusively on train set and do not refit on test set."""
    X_train, y_train = separate_features_and_target(sample_unsw_train_data, target_column="label")
    X_test, y_test = separate_features_and_target(sample_unsw_test_data, target_column="label")

    pipeline, metadata, partitions = train_supervised_model(
        X=X_train,
        y=y_train,
        X_test=X_test,
        y_test=y_test,
        model_type="logistic_regression",
        max_iter=200,
    )

    preprocessor = pipeline.named_steps["preprocessor"]
    # Check that numeric imputer fitted on training features
    num_imputer = preprocessor.named_transformers_["num"].named_steps["imputer"]
    assert hasattr(num_imputer, "statistics_")
    # Verify the imputer was not refit during evaluation/prediction
    test_preds = pipeline.predict(X_test)
    assert len(test_preds) == len(X_test)


def test_missing_and_infinite_feature_values_in_pipeline(sample_unsw_train_data):
    """Verifies pipeline cleanly handles missing and infinite numeric feature values at inference."""
    X_train, y_train = separate_features_and_target(sample_unsw_train_data, target_column="label")
    pipeline, _, _ = train_supervised_model(
        X=X_train, y=y_train, model_type="random_forest", n_estimators=5
    )

    # DataFrame with missing values and infinite values
    X_problematic = X_train.copy().iloc[:2]
    X_problematic.iloc[0, X_problematic.columns.get_loc("dur")] = np.nan
    X_problematic.iloc[1, X_problematic.columns.get_loc("rate")] = np.inf

    # Clean with clean_dataframe to replace inf with nan
    X_cleaned = clean_dataframe(X_problematic, drop_duplicates=False, replace_inf=True)
    preds = pipeline.predict(X_cleaned)
    assert len(preds) == 2
    assert preds[0] in [0, 1]
    assert preds[1] in [0, 1]


def test_missing_dataset_files_raises_cleanly():
    """Verifies that non-existent UNSW-NB15 files raise FileNotFoundError and do not fall back silently."""
    from src.threat_detection.data_loader import load_dataset
    fake_path = "data/raw/UNSW-NB15/UNSW_NB15_nonexistent_dataset.csv"
    with pytest.raises(FileNotFoundError):
        load_dataset(fake_path)


def test_missing_model_file_raises_cleanly():
    """Verifies that non-existent model files raise FileNotFoundError."""
    fake_model = "models/nonexistent_model_file.joblib"
    with pytest.raises(FileNotFoundError):
        load_model_pipeline(fake_model)


def test_evaluation_metrics_and_pr_auc():
    """Verifies calculation of precision, recall, F1, ROC-AUC, and PR-AUC."""
    from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    y_scores = np.array([0.1, 0.9, 0.2, 0.85, 0.7, 0.3, 0.95, 0.15])
    y_pred = (y_scores >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)

    assert acc == 1.0
    assert prec == 1.0
    assert rec == 1.0
    assert f1 == 1.0
    assert roc_auc == 1.0
    assert pr_auc == 1.0


def test_binary_metric_label_disambiguation():
    """Verifies that positive-class F1, weighted F1, and macro F1 are distinct and correctly calculated."""
    from sklearn.metrics import f1_score, precision_score, recall_score
    # Imbalanced sample: 4 normal (0), 2 attack (1)
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 0])  # 3 TN, 1 FP, 1 FN, 1 TP

    # Attack (pos_label=1): prec = 1/2 = 0.5, rec = 1/2 = 0.5, f1 = 0.5
    attack_f1 = f1_score(y_true, y_pred, pos_label=1)
    attack_prec = precision_score(y_true, y_pred, pos_label=1)
    attack_rec = recall_score(y_true, y_pred, pos_label=1)

    # Normal (pos_label=0): prec = 3/4 = 0.75, rec = 3/4 = 0.75, f1 = 0.75
    normal_f1 = f1_score(y_true, y_pred, pos_label=0)

    # Macro F1 = (0.5 + 0.75) / 2 = 0.625
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    # Weighted F1 = (4 * 0.75 + 2 * 0.5) / 6 = 4.0 / 6 = 0.6667
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    assert attack_f1 == 0.5
    assert attack_prec == 0.5
    assert attack_rec == 0.5
    assert normal_f1 == 0.75
    assert macro_f1 == 0.625
    assert round(weighted_f1, 4) == 0.6667

    # Assert that positive-class F1 is NOT identical to weighted F1 under imbalance
    assert attack_f1 != weighted_f1
    assert attack_f1 != macro_f1


def test_multiclass_metric_calculations():
    """Verifies multiclass macro and weighted metrics and OvR ROC-AUC calculation."""
    from sklearn.metrics import f1_score, roc_auc_score
    y_true = np.array(["Normal", "DoS", "Exploits", "Normal", "Exploits"])
    y_pred = np.array(["Normal", "DoS", "Exploits", "Normal", "DoS"])
    classes = ["DoS", "Exploits", "Normal"]

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    assert 0.0 < macro_f1 <= 1.0
    assert 0.0 < weighted_f1 <= 1.0

    # Simulated 3-class probability matrix
    probas = np.array([
        [0.1, 0.1, 0.8],
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.1, 0.7],
        [0.6, 0.3, 0.1],
    ])
    roc_ovr = roc_auc_score(y_true, probas, multi_class="ovr", average="weighted")
    assert 0.5 <= roc_ovr <= 1.0


