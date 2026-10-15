"""Unified detection and prediction orchestrator.

Integrates rule-based heuristics, supervised machine learning classification,
and unsupervised anomaly detection into a coherent security analysis workflow.
Preserves clear architectural boundaries between distinct detection methodologies.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.threat_detection.alerts import (
    SecurityAlert,
    compile_alerts,
    create_alert_from_anomaly,
    create_alert_from_ml_prediction,
    create_alert_from_rule_finding,
)
from src.threat_detection.anomaly_detector import score_anomalies
from src.threat_detection.features import engineer_domain_features
from src.threat_detection.preprocessing import (
    clean_dataframe,
    separate_features_and_target,
)
from src.threat_detection.rules import evaluate_single_event_rules
from src.threat_detection.utils import (
    DataValidationError,
    ModelPipelineError,
    logger,
    validate_dataframe_not_empty,
)


def run_unified_detection(
    df: pd.DataFrame,
    supervised_pipeline: Optional[Pipeline] = None,
    supervised_metadata: Optional[Dict[str, Any]] = None,
    anomaly_pipeline: Optional[Pipeline] = None,
    anomaly_metadata: Optional[Dict[str, Any]] = None,
    enable_rules: bool = True,
    failed_threshold: int = 5,
    volume_threshold: int = 100,
) -> Tuple[pd.DataFrame, List[SecurityAlert]]:
    """Runs a complete defensive evaluation across incoming network telemetry.

    Orchestration sequence:
    1. Clean incoming telemetry and engineer defensive domain features.
    2. Evaluate heuristic detection rules for explainable triage.
    3. If a supervised pipeline is provided, evaluate feature matrix and obtain class predictions.
    4. If an anomaly pipeline is provided, evaluate numeric feature outliers.
    5. Compile, deduplicate, and return structured SecurityAlert instances.

    Args:
        df: Input DataFrame containing network telemetry.
        supervised_pipeline: Fitted Scikit-learn Pipeline (optional).
        supervised_metadata: Metadata with feature schema (optional).
        anomaly_pipeline: Fitted Isolation Forest Pipeline (optional).
        anomaly_metadata: Metadata with anomaly feature names (optional).
        enable_rules: Whether to run rule-based heuristic checks.
        failed_threshold: Configurable threshold for connection failures.
        volume_threshold: Configurable threshold for volume spikes.

    Returns:
        Tuple of:
        - Enriched DataFrame with columns detailing rule, ML, and anomaly findings.
        - Deduplicated list of SecurityAlert objects.
    """
    validate_dataframe_not_empty(df, context_name="Input telemetry to unified detection")

    # Step 1: Data hygiene and feature engineering
    cleaned = clean_dataframe(df, drop_duplicates=False, replace_inf=True)
    engineered = engineer_domain_features(cleaned)

    # Initialize enrichment columns
    result_df = engineered.copy()
    result_df["rule_findings_count"] = 0
    result_df["ml_prediction"] = pd.Series([None] * len(result_df), dtype="object")
    result_df["ml_probability"] = pd.Series([None] * len(result_df), dtype="object")
    result_df["anomaly_flag"] = pd.Series([None] * len(result_df), dtype="object")
    result_df["anomaly_score"] = pd.Series([None] * len(result_df), dtype="object")

    alerts_list: List[SecurityAlert] = []

    # Step 2: Rule-based Heuristic Detection
    if enable_rules:
        for idx, row in result_df.iterrows():
            event_dict = row.to_dict()
            findings = evaluate_single_event_rules(
                event=event_dict,
                failed_threshold=failed_threshold,
                volume_threshold=volume_threshold,
            )
            result_df.at[idx, "rule_findings_count"] = len(findings)
            for f in findings:
                finding_dict = {
                    "event_id": event_dict.get("event_id", f"EVT-{idx:05d}"),
                    "timestamp": event_dict.get("timestamp"),
                    "source_ip": event_dict.get("source_ip", "UNKNOWN"),
                    "destination_ip": event_dict.get("destination_ip", "UNKNOWN"),
                    "simulated": bool(event_dict.get("simulated", False)),
                    **f.to_dict(),
                }
                alerts_list.append(create_alert_from_rule_finding(finding_dict))

    # Step 3: Supervised ML Predictions (if pipeline supplied)
    if supervised_pipeline is not None and supervised_metadata is not None:
        try:
            expected_features = supervised_metadata.get("feature_schema", {}).get("all_features", [])
            # Check if features are available
            available_features = [f for f in expected_features if f in result_df.columns]
            if len(available_features) == len(expected_features) and len(expected_features) > 0:
                X_infer = result_df[expected_features]
                preds = supervised_pipeline.predict(X_infer)
                result_df["ml_prediction"] = preds

                probs = None
                if hasattr(supervised_pipeline, "predict_proba"):
                    proba_matrix = supervised_pipeline.predict_proba(X_infer)
                    if proba_matrix.shape[1] >= 2:
                        probs = proba_matrix[:, 1]
                        result_df["ml_probability"] = [round(float(p), 4) for p in probs]

                model_name = supervised_metadata.get("model_type", "supervised_ml").upper()

                # Generate ML alerts for suspicious predictions (binary 1 or attack categories)
                for idx, pred in enumerate(preds):
                    event_dict = result_df.iloc[idx].to_dict()
                    prob_val = float(probs[idx]) if probs is not None else None
                    alert = create_alert_from_ml_prediction(
                        event_dict=event_dict,
                        prediction_label=pred,
                        probability=prob_val,
                        model_name=model_name,
                    )
                    if alert:
                        alerts_list.append(alert)
            else:
                logger.warning(
                    f"Supervised model skipped: missing features {set(expected_features) - set(result_df.columns)}"
                )
        except Exception as exc:
            logger.error(f"Error during supervised ML inference: {exc}")

    # Step 4: Unsupervised Anomaly Detection (if pipeline supplied)
    if anomaly_pipeline is not None and anomaly_metadata is not None:
        try:
            num_cols = anomaly_metadata.get("feature_columns", [])
            available_num_cols = [c for c in num_cols if c in result_df.columns]
            if len(available_num_cols) == len(num_cols) and len(num_cols) > 0:
                flags, scores = score_anomalies(
                    pipeline=anomaly_pipeline,
                    X=result_df,
                    feature_columns=num_cols,
                )
                result_df["anomaly_flag"] = flags
                result_df["anomaly_score"] = [round(float(s), 4) for s in scores]

                for idx, flag in enumerate(flags):
                    if flag == 1:
                        event_dict = result_df.iloc[idx].to_dict()
                        alert = create_alert_from_anomaly(
                            event_dict=event_dict,
                            anomaly_score=float(scores[idx]),
                        )
                        alerts_list.append(alert)
            else:
                logger.warning(
                    f"Anomaly scoring skipped: missing numeric columns {set(num_cols) - set(result_df.columns)}"
                )
        except Exception as exc:
            logger.error(f"Error during anomaly scoring: {exc}")

    # Step 5: Deduplicate and sort alerts
    deduped_alerts = compile_alerts(alerts_list)

    return result_df, deduped_alerts
