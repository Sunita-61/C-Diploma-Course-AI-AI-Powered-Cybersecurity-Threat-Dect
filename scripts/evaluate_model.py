"""Model evaluation script computing cybersecurity classification metrics.

Calculates precision, recall, F1, confusion matrices, and ROC-AUC.
Supports both binary ('label') and multiclass ('attack_cat') targets.
Supports independent test partitions (e.g. UNSW_NB15_testing-set.csv).
Exports structured evaluation reports and plots confusion matrix figures.
"""

import argparse
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.threat_detection.config import FIGURES_DIR, MODELS_DIR, REPORTS_DIR
from src.threat_detection.data_loader import load_dataset
from src.threat_detection.features import engineer_domain_features
from src.threat_detection.model import load_model_pipeline
from src.threat_detection.preprocessing import clean_dataframe, separate_features_and_target
from src.threat_detection.utils import ensure_directory, get_current_utc_timestamp, logger


def evaluate_model_cli(
    model_path: str,
    data_path: str,
    target_col: str = None,
    is_test_set: bool = False,
    report_output: str = None,
    figure_output: str = None,
) -> None:
    """Evaluates a saved model pipeline against an evaluation dataset and generates reports."""
    print(f"\n{'='*75}")
    print(" MODEL EVALUATION & CYBERSECURITY IMPACT ANALYSIS")
    print(f"{'='*75}\n")

    ensure_directory(REPORTS_DIR)
    ensure_directory(FIGURES_DIR)

    # 1. Load Model Pipeline & Metadata
    print(f"[*] Loading model pipeline from: {model_path}")
    pipeline, metadata = load_model_pipeline(model_path)
    model_type = metadata.get("model_type", "model")
    feature_schema = metadata.get("feature_schema", {}).get("all_features", [])
    model_classes = metadata.get("classes", [str(c) for c in pipeline.classes_])
    target_meta = metadata.get("target_metadata", {})
    target_type = target_meta.get("target_type", "binary" if len(model_classes) == 2 else "multiclass")

    # 2. Load Evaluation Dataset
    print(f"[*] Loading evaluation dataset from: {data_path}")
    df_raw = load_dataset(data_path)
    df_clean = clean_dataframe(df_raw, drop_duplicates=False, replace_inf=True)
    df_engineered = engineer_domain_features(df_clean)

    target_to_use = target_col or target_meta.get("target_column")
    X, y, eval_target_meta = separate_features_and_target(
        df_engineered, target_column=target_to_use, drop_identifiers=True, return_metadata=True
    )
    if y is None:
        print("[ERROR] Evaluation dataset must contain a target label column.")
        sys.exit(1)

    # 3. Partition Handling: Predefined Test File vs Holdout Split
    if is_test_set or "test" in Path(data_path).name.lower():
        print(f"[*] Evaluating directly on full independent test file: {len(X):,} records.")
        X_test = X
        y_test = y
        split_description = "Independent predefined test partition"
    else:
        random_state = metadata.get("random_state", 42)
        stratify_target = y if y.value_counts().min() >= 2 else None
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.25, random_state=random_state, stratify=stratify_target
        )
        print(f"[*] Evaluating on {len(X_test):,} holdout test samples (25% stratified split).")
        split_description = "25% stratified holdout split"

    # Ensure required features exist in test set
    if feature_schema:
        missing = [f for f in feature_schema if f not in X_test.columns]
        if missing:
            print(f"[ERROR] Test dataset is missing required features: {missing}")
            sys.exit(1)
        X_eval = X_test[feature_schema]
    else:
        X_eval = X_test

    # 4. Generate Predictions & Probabilities
    y_pred = pipeline.predict(X_eval)

    # Convert to string representations for consistent multiclass alignment
    y_test_str = [str(val) for val in y_test]
    y_pred_str = [str(val) for val in y_pred]
    unique_test_classes = sorted(list(set(y_test_str).union(set(y_pred_str))))

    # Multiclass vs Binary classification metrics
    is_binary = len(unique_test_classes) <= 2 and ("0" in unique_test_classes or 0 in set(y_test))

    acc = float(accuracy_score(y_test_str, y_pred_str))
    prec_weighted = float(precision_score(y_test_str, y_pred_str, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_test_str, y_pred_str, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_test_str, y_pred_str, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_test_str, y_pred_str, average="macro", zero_division=0))

    if is_binary:
        # Standard binary positive label is 1 / "1" (Attack)
        pos_label = "1" if "1" in unique_test_classes else unique_test_classes[-1]
        attack_prec = float(precision_score(y_test_str, y_pred_str, pos_label=pos_label, zero_division=0))
        attack_rec = float(recall_score(y_test_str, y_pred_str, pos_label=pos_label, zero_division=0))
        attack_f1 = float(f1_score(y_test_str, y_pred_str, pos_label=pos_label, zero_division=0))
    else:
        attack_prec = None
        attack_rec = None
        attack_f1 = None

    # ROC-AUC & PR-AUC computation
    roc_auc = None
    pr_auc = None
    if hasattr(pipeline, "predict_proba"):
        try:
            probas = pipeline.predict_proba(X_eval)
            if is_binary and probas.shape[1] >= 2:
                # Ensure binary numeric target for AUC calculation
                y_test_num = np.array([1 if str(v) == "1" else 0 for v in y_test])
                roc_auc = float(roc_auc_score(y_test_num, probas[:, 1]))
                pr_auc = float(average_precision_score(y_test_num, probas[:, 1]))
            elif not is_binary and probas.shape[1] == len(pipeline.classes_):
                # Multiclass OvR ROC-AUC
                roc_auc = float(roc_auc_score(y_test, probas, multi_class="ovr", average="weighted"))
        except Exception:
            roc_auc = None
            pr_auc = None

    cm = confusion_matrix(y_test_str, y_pred_str, labels=unique_test_classes)
    cls_report_str = classification_report(y_test_str, y_pred_str, labels=unique_test_classes, zero_division=0)

    # 5. Display Console Metrics
    print("\n--- Model Classification Performance ---")
    print(f"  Target Type:                {target_type.upper()} ({len(unique_test_classes)} classes)")
    print(f"  Accuracy:                   {acc:.4f}")
    if is_binary:
        print(f"  Attack Precision (pos=1):   {attack_prec:.4f}")
        print(f"  Attack Recall / Det. Rate:  {attack_rec:.4f}")
        print(f"  Attack F1-Score (pos=1):    {attack_f1:.4f}")
    print(f"  Weighted Precision:         {prec_weighted:.4f}")
    print(f"  Weighted Recall:            {rec_weighted:.4f}")
    print(f"  Weighted F1-Score:          {f1_weighted:.4f}")
    print(f"  Macro F1-Score:             {f1_macro:.4f}")
    if roc_auc is not None:
        print(f"  ROC-AUC:                    {roc_auc:.4f}")
    if pr_auc is not None:
        print(f"  PR-AUC (Avg Precision):     {pr_auc:.4f}")

    print("\n--- Detailed Classification Report ---\n")
    print(cls_report_str)

    # 6. Cybersecurity Operational Impact Explanation
    if is_binary and cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        cyber_context = (
            "CYBERSECURITY OPERATIONAL IMPACT:\n"
            f"  - True Negatives ({tn}): Normal network sessions correctly identified as benign.\n"
            f"  - False Positives ({fp}): Benign traffic incorrectly flagged as an intrusion. "
            "High FP rates cause analyst alert fatigue in the SOC and inflate incident response costs.\n"
            f"  - False Negatives ({fn}): Real attacks missed by the model. "
            "In production, a false negative can lead to undetected persistence, lateral movement, or data exfiltration.\n"
            f"  - True Positives ({tp}): Real attacks correctly flagged for triage.\n"
        )
    else:
        cyber_context = (
            "MULTICLASS ATTACK CATEGORIZATION IMPACT:\n"
            "  - Accurately distinguishing specific attack families (e.g., DoS vs Exploits vs Reconnaissance) "
            "enables SOC playbooks to trigger targeted containment strategies (e.g. rate limiting vs endpoint isolation).\n"
            "  - Underrepresented attack classes with lower recall highlight areas where supplementary heuristic rules "
            "or anomaly detection must act as a secondary defensive layer.\n"
        )
    print(cyber_context)

    # 7. Save Figure (Confusion Matrix)
    model_stem = Path(model_path).stem
    fig_path = Path(figure_output) if figure_output else FIGURES_DIR / f"{model_stem}_confusion_matrix.png"

    plt.figure(figsize=(8, 6) if not is_binary else (6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=unique_test_classes,
        yticklabels=unique_test_classes,
    )
    plt.title(f"Confusion Matrix: {model_stem.replace('_', ' ').title()}\n({split_description})")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual True Class")
    if not is_binary:
        plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[+] Confusion matrix plot saved to:\n    {fig_path}")

    # 8. Save Structured Text & JSON Reports
    report_file_txt = Path(report_output) if report_output else REPORTS_DIR / f"{model_stem}_evaluation_report.txt"
    report_file_json = report_file_txt.with_suffix(".json")

    timestamp_now = get_current_utc_timestamp()
    report_content = f"""======================================================================
CYBERSECURITY THREAT DETECTION MODEL EVALUATION REPORT
Model: {model_stem}
Dataset: {Path(data_path).name}
Evaluation Date: {timestamp_now}
Split Strategy: {split_description}
======================================================================

1. DATASET & EVALUATION CONTEXT
----------------------------------------------------------------------
Evaluation dataset path: {data_path}
Target field evaluated: {target_to_use} ({target_type})
Test samples evaluated: {len(X_test):,}
Model architecture: {model_type}

2. CLASSIFICATION PERFORMANCE METRICS
----------------------------------------------------------------------
Accuracy:                  {acc:.4f}
Attack Precision (pos=1):  {f"{attack_prec:.4f}" if is_binary else 'N/A'}
Attack Recall (pos=1):     {f"{attack_rec:.4f}" if is_binary else 'N/A'}
Attack F1-Score (pos=1):   {f"{attack_f1:.4f}" if is_binary else 'N/A'}
Weighted Precision:        {prec_weighted:.4f}
Weighted Recall:           {rec_weighted:.4f}
Weighted F1-Score:         {f1_weighted:.4f}
Macro F1-Score:            {f1_macro:.4f}
ROC-AUC:                   {'N/A' if roc_auc is None else f"{roc_auc:.4f}"}
PR-AUC (Average Precision):{'N/A' if pr_auc is None else f"{pr_auc:.4f}"}

3. CONFUSION MATRIX (Classes: {unique_test_classes})
----------------------------------------------------------------------
{cm}

4. DETAILED CLASSIFICATION REPORT
----------------------------------------------------------------------
{cls_report_str}

5. DEFENSIVE CYBERSECURITY OPERATIONAL IMPACT
----------------------------------------------------------------------
{cyber_context}
"""
    with open(report_file_txt, "w", encoding="utf-8") as f:
        f.write(report_content)

    structured_data = {
        "model_file": model_stem,
        "dataset_name": Path(data_path).name,
        "evaluation_timestamp": timestamp_now,
        "target_field": target_to_use,
        "target_type": target_type,
        "split_strategy": split_description,
        "test_samples": int(len(X_test)),
        "classes": unique_test_classes,
        "metrics": {
            "accuracy": acc,
            "attack_precision": attack_prec,
            "attack_recall": attack_rec,
            "attack_f1": attack_f1,
            "weighted_precision": prec_weighted,
            "weighted_recall": rec_weighted,
            "weighted_f1": f1_weighted,
            "macro_f1": f1_macro,
            # Backward-compatibility aliases
            "precision": attack_prec if is_binary else prec_weighted,
            "recall": attack_rec if is_binary else rec_weighted,
            "f1_score": attack_f1 if is_binary else f1_weighted,
            "f1_macro": f1_macro,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
        },
        "confusion_matrix": cm.tolist(),
    }
    with open(report_file_json, "w", encoding="utf-8") as f:
        json.dump(structured_data, f, indent=2)

    print(f"[+] Text report saved to:\n    {report_file_txt}")
    print(f"[+] JSON report saved to:\n    {report_file_json}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate threat detection model performance.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to saved .joblib model.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to evaluation dataset.")
    parser.add_argument("--target-col", type=str, default=None, help="Target label column (e.g. 'label' or 'attack_cat').")
    parser.add_argument(
        "--is-test-set",
        action="store_true",
        help="Evaluate on the complete passed file without holdout re-splitting.",
    )
    parser.add_argument("--report-output", type=str, default=None, help="Custom path for text report.")
    parser.add_argument("--figure-output", type=str, default=None, help="Custom path for matrix plot.")
    args = parser.parse_args()
    evaluate_model_cli(
        model_path=args.model_path,
        data_path=args.data_path,
        target_col=args.target_col,
        is_test_set=args.is_test_set,
        report_output=args.report_output,
        figure_output=args.figure_output,
    )


if __name__ == "__main__":
    main()
