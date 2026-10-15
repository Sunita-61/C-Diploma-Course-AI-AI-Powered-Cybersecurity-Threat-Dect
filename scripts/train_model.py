"""CLI utility to train supervised classifiers and anomaly detectors.

Supports both single-file datasets (synthetic) and official predefined train/test partitions (UNSW-NB15).
Supports binary targets ('label') and multiclass targets ('attack_cat').
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.threat_detection.anomaly_detector import fit_anomaly_detector, save_anomaly_pipeline
from src.threat_detection.config import MODELS_DIR
from src.threat_detection.data_loader import load_dataset
from src.threat_detection.features import engineer_domain_features
from src.threat_detection.model import save_model_pipeline, train_supervised_model
from src.threat_detection.preprocessing import clean_dataframe, separate_features_and_target
from src.threat_detection.utils import ensure_directory, logger


def train_models_cli(
    data_path: str = None,
    train_path: str = None,
    test_path: str = None,
    target_col: str = None,
    model_type: str = "random_forest",
    train_anomaly: bool = True,
    output_dir: str = None,
    model_name: str = None,
    n_estimators: int = 100,
    max_iter: int = 1000,
) -> None:
    """Orchestrates model training, evaluation split, and model persistence."""
    print(f"\n{'='*75}")
    print(f" MODEL TRAINING PIPELINE: {model_type.upper()}")
    print(f"{'='*75}\n")

    out_path = Path(output_dir) if output_dir else MODELS_DIR
    ensure_directory(out_path)

    # 1. Determine dataset input paths
    if train_path:
        primary_train_file = Path(train_path)
    elif data_path:
        primary_train_file = Path(data_path)
    else:
        print("[ERROR] Please specify either --data-path or --train-path.")
        sys.exit(1)

    if not primary_train_file.exists():
        print(f"[ERROR] Training file not found at: {primary_train_file}")
        print("        Ensure the dataset has been downloaded and placed in the expected directory.")
        sys.exit(1)

    # 2. Load and prepare training data
    print(f"[*] Loading training dataset from: {primary_train_file}")
    df_raw_train = load_dataset(primary_train_file)
    df_clean_train = clean_dataframe(df_raw_train, drop_duplicates=True, replace_inf=True)
    df_train_eng = engineer_domain_features(df_clean_train)

    X_train, y_train, target_meta = separate_features_and_target(
        df_train_eng, target_column=target_col, drop_identifiers=True, return_metadata=True
    )


    if y_train is None:
        print("[ERROR] No valid target column found for supervised training.")
        print("        Provide a dataset with a target label or specify --target-col.")
        sys.exit(1)

    print(f"    Train records: {len(X_train):,}")
    print(f"    Target column: '{target_meta.get('target_column')}' (Type: {target_meta.get('target_type')})")
    print(f"    Feature columns ({len(X_train.columns)}): {list(X_train.columns)}")
    print(f"    Target class distribution:\n{y_train.value_counts()}")

    # 3. Check for predefined test partition
    X_test, y_test = None, None
    if test_path:
        test_file = Path(test_path)
        if not test_file.exists():
            print(f"[ERROR] Predefined test file not found at: {test_file}")
            sys.exit(1)
        print(f"\n[*] Loading official test partition from: {test_file}")
        df_raw_test = load_dataset(test_file)
        df_clean_test = clean_dataframe(df_raw_test, drop_duplicates=False, replace_inf=True)
        df_test_eng = engineer_domain_features(df_clean_test)

        X_test, y_test = separate_features_and_target(
            df_test_eng, target_column=target_col, drop_identifiers=True
        )
        print(f"    Test records: {len(X_test):,}")

    # 4. Supervised Model Training
    print(f"\n[*] Training supervised {model_type} pipeline...")
    pipeline, metadata, partitions = train_supervised_model(
        X=X_train,
        y=y_train,
        X_test=X_test,
        y_test=y_test,
        model_type=model_type,
        test_size=0.25,
        target_metadata=target_meta,
        n_estimators=n_estimators,
        max_iter=max_iter,
    )
    X_tr_final, X_te_final, y_tr_final, y_te_final = partitions

    save_name = model_name if model_name else f"{model_type}.joblib"
    saved_supervised_path = out_path / save_name
    save_model_pipeline(pipeline, metadata, saved_supervised_path)
    print(f"[+] Supervised model saved to:\n    {saved_supervised_path}")
    print(f"    - Train partition: {len(X_tr_final):,} samples")
    print(f"    - Test partition:  {len(X_te_final):,} samples")

    # 5. Optional Anomaly Detector Training on normal traffic baseline
    if train_anomaly:
        print("\n[*] Training unsupervised Isolation Forest detector on normal baseline...")
        normal_mask = (y_tr_final == 0) | (y_tr_final == "Normal")
        X_baseline = X_tr_final[normal_mask] if normal_mask.sum() >= 10 else X_tr_final

        try:
            iso_pipe, iso_meta = fit_anomaly_detector(X_baseline, contamination=0.05)
            saved_iso_path = out_path / "isolation_forest.joblib"
            save_anomaly_pipeline(iso_pipe, iso_meta, saved_iso_path)
            print(f"[+] Anomaly detector saved to:\n    {saved_iso_path}")
        except Exception as exc:
            print(f"[!] Warning: Anomaly detector fitting skipped: {exc}")

    print(f"\n{'='*75}")
    print(" TRAINING COMPLETE")
    print(f"{'='*75}\n")


def main():
    parser = argparse.ArgumentParser(description="Train threat detection machine learning models.")
    parser.add_argument("--data-path", type=str, default=None, help="Path to single CSV dataset.")
    parser.add_argument("--train-path", type=str, default=None, help="Path to official training CSV partition.")
    parser.add_argument("--test-path", type=str, default=None, help="Path to official testing CSV partition.")
    parser.add_argument("--target-col", type=str, default=None, help="Name of target label column (e.g. 'label' or 'attack_cat').")
    parser.add_argument(
        "--model-type",
        type=str,
        default="random_forest",
        choices=["random_forest", "logistic_regression"],
        help="Supervised algorithm to train.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Optional custom filename for saved model (e.g. 'unsw_nb15_rf.joblib').",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees for Random Forest (default: 100).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum iterations for Logistic Regression (default: 1000).",
    )
    parser.add_argument(
        "--skip-anomaly",
        action="store_true",
        help="Skip training Isolation Forest anomaly detector.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom directory for saved model binaries.",
    )
    args = parser.parse_args()
    train_models_cli(
        data_path=args.data_path,
        train_path=args.train_path,
        test_path=args.test_path,
        target_col=args.target_col,
        model_type=args.model_type,
        train_anomaly=not args.skip_anomaly,
        output_dir=args.output_dir,
        model_name=args.model_name,
        n_estimators=args.n_estimators,
        max_iter=args.max_iter,
    )


if __name__ == "__main__":
    main()
