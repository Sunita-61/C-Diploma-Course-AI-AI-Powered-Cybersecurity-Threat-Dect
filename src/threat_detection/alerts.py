"""Structured security alert generation and export module.

Produces standardized, traceable security alerts from rule findings,
supervised ML predictions, and anomaly detection outputs.
"""

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import pandas as pd

from src.threat_detection.config import ALERTS_DIR
from src.threat_detection.utils import (
    DataValidationError,
    ensure_directory,
    get_current_utc_timestamp,
    logger,
)


@dataclass
class SecurityAlert:
    """Standardized representation of an actionable cybersecurity alert."""
    alert_id: str
    event_id: str
    timestamp: str
    source_ip: str
    destination_ip: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    category: str
    detection_method: str  # RULE_BASED, SUPERVISED_RF, SUPERVISED_LR, ANOMALY_ISOLATION_FOREST
    rule_id: Optional[str]
    description: str
    status: str  # NEW, REVIEWED, DISMISSED
    simulated: bool

    def to_dict(self) -> Dict[str, Any]:
        """Converts alert to a serializable dictionary."""
        return asdict(self)


def generate_alert_id(event_id: str, method: str, finding_key: str) -> str:
    """Generates a deterministic short ID for deduplication and tracking.

    Args:
        event_id: Originating event identifier.
        method: Detection mechanism name.
        finding_key: Unique rule ID or anomaly tag.

    Returns:
        Formatted alert identifier string like 'ALT-A1B2C3D4'.
    """
    digest = hashlib.sha256(f"{event_id}:{method}:{finding_key}".encode("utf-8")).hexdigest()[:8].upper()
    return f"ALT-{digest}"


def create_alert_from_rule_finding(
    finding_dict: Dict[str, Any],
    status: str = "NEW",
) -> SecurityAlert:
    """Constructs a SecurityAlert from a rule finding dictionary.

    Args:
        finding_dict: Dictionary produced by rule engine.
        status: Initial alert workflow status (default: 'NEW').

    Returns:
        SecurityAlert instance.
    """
    event_id = str(finding_dict.get("event_id", "UNKNOWN_EVT"))
    rule_id = str(finding_dict.get("rule_id", "RULE_GENERIC"))
    alert_id = generate_alert_id(event_id, "RULE_BASED", rule_id)

    timestamp = str(finding_dict.get("timestamp") or get_current_utc_timestamp())
    source_ip = str(finding_dict.get("source_ip", "UNKNOWN"))
    destination_ip = str(finding_dict.get("destination_ip", "UNKNOWN"))
    severity = str(finding_dict.get("severity", "MEDIUM"))
    category = str(finding_dict.get("category", "HEURISTIC_RULE_MATCH"))
    message = str(finding_dict.get("message", "Rule condition matched."))
    simulated = bool(finding_dict.get("simulated", False))

    return SecurityAlert(
        alert_id=alert_id,
        event_id=event_id,
        timestamp=timestamp,
        source_ip=source_ip,
        destination_ip=destination_ip,
        severity=severity,
        category=category,
        detection_method="RULE_BASED",
        rule_id=rule_id,
        description=message,
        status=status,
        simulated=simulated,
    )


def create_alert_from_ml_prediction(
    event_dict: Dict[str, Any],
    prediction_label: Any,
    probability: Optional[float] = None,
    model_name: str = "SUPERVISED_RF",
    status: str = "NEW",
) -> Optional[SecurityAlert]:
    """Generates an alert when a supervised ML classifier flags suspicious activity.

    Supports both binary classifications (0=Normal, 1=Suspicious) and multiclass
    attack families (e.g. 'Exploits', 'DoS', 'Reconnaissance').

    Args:
        event_dict: Original event data dictionary.
        prediction_label: Prediction output (int or string).
        probability: Optional calibrated probability (0.0 to 1.0).
        model_name: Name of model architecture.
        status: Initial alert workflow status.

    Returns:
        SecurityAlert if prediction is suspicious, else None.
    """
    normal_labels = {0, "0", "0.0", "normal", "Normal", "benign", "Benign"}
    if prediction_label in normal_labels:
        return None

    event_id = str(event_dict.get("event_id", event_dict.get("id", "UNKNOWN_EVT")))
    alert_id = generate_alert_id(event_id, model_name, str(prediction_label))
    timestamp = str(event_dict.get("timestamp") or get_current_utc_timestamp())

    # Map probability to explainable severity
    if probability is not None:
        if probability >= 0.85:
            severity = "HIGH"
        elif probability >= 0.65:
            severity = "MEDIUM"
        else:
            severity = "LOW"
        prob_desc = f" (Confidence: {probability * 100:.1f}%)"
    else:
        severity = "MEDIUM"
        prob_desc = ""

    label_str = str(prediction_label)
    category = "STATISTICAL_ML_DETECTION" if label_str in ["1", "1.0"] else f"ATTACK_{label_str.upper()}"
    description = (
        f"Supervised model '{model_name}' classified activity pattern as '{label_str}'{prob_desc}. "
        "Requires analyst verification."
    )

    return SecurityAlert(
        alert_id=alert_id,
        event_id=event_id,
        timestamp=timestamp,
        source_ip=str(event_dict.get("source_ip", event_dict.get("srcip", "UNKNOWN"))),
        destination_ip=str(event_dict.get("destination_ip", event_dict.get("dstip", "UNKNOWN"))),
        severity=severity,
        category=category,
        detection_method=model_name,
        rule_id=None,
        description=description,
        status=status,
        simulated=bool(event_dict.get("simulated", False)),
    )



def create_alert_from_anomaly(
    event_dict: Dict[str, Any],
    anomaly_score: float,
    status: str = "NEW",
) -> SecurityAlert:
    """Generates an alert when Isolation Forest identifies an outlier record.

    Args:
        event_dict: Original event data dictionary.
        anomaly_score: Raw or normalized anomaly score.
        status: Initial alert workflow status.

    Returns:
        SecurityAlert instance.
    """
    event_id = str(event_dict.get("event_id", "UNKNOWN_EVT"))
    alert_id = generate_alert_id(event_id, "ANOMALY_ISOLATION_FOREST", "OUTLIER")
    timestamp = str(event_dict.get("timestamp") or get_current_utc_timestamp())

    description = (
        f"Isolation Forest flagged record as multivariate statistical outlier "
        f"(Anomaly score: {anomaly_score:.3f}). Note: Anomalies indicate uncommon features, not confirmed threats."
    )

    return SecurityAlert(
        alert_id=alert_id,
        event_id=event_id,
        timestamp=timestamp,
        source_ip=str(event_dict.get("source_ip", "UNKNOWN")),
        destination_ip=str(event_dict.get("destination_ip", "UNKNOWN")),
        severity="LOW",
        category="UNSUPERVISED_ANOMALY",
        detection_method="ANOMALY_ISOLATION_FOREST",
        rule_id=None,
        description=description,
        status=status,
        simulated=bool(event_dict.get("simulated", False)),
    )


def compile_alerts(alerts: List[SecurityAlert]) -> List[SecurityAlert]:
    """Deduplicates a list of alerts based on unique alert_id.

    Args:
        alerts: List of SecurityAlert instances.

    Returns:
        Deduplicated list of SecurityAlerts preserving original order.
    """
    seen: Set[str] = set()
    deduped: List[SecurityAlert] = []

    for alert in alerts:
        if alert.alert_id not in seen:
            seen.add(alert.alert_id)
            deduped.append(alert)

    return deduped


def export_alerts_to_csv(
    alerts: List[SecurityAlert],
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Exports structured alerts list to a CSV file.

    Args:
        alerts: List of SecurityAlert objects.
        output_path: Destination file path. Defaults to outputs/alerts/alerts_<timestamp>.csv.

    Returns:
        Path to the saved CSV file.
    """
    if output_path is None:
        ensure_directory(ALERTS_DIR)
        timestamp_slug = get_current_utc_timestamp().replace(":", "-")
        output_path = ALERTS_DIR / f"security_alerts_{timestamp_slug}.csv"
    else:
        output_path = Path(output_path)
        ensure_directory(output_path.parent)

    records = [alert.to_dict() for alert in alerts]
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    logger.info(f"Exported {len(records)} security alerts to: {output_path}")
    return output_path
