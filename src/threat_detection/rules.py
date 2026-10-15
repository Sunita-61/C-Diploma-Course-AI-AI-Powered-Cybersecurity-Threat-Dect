"""Transparent defensive rule-based detection engine.

Implements explainable heuristic rules for security event analysis.
All findings generated are transparent heuristics for demonstration/triage
and are explicitly distinct from verified security breaches.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
import pandas as pd

from src.threat_detection.config import (
    FAILED_CONNECTIONS_THRESHOLD,
    HIGH_VOLUME_THRESHOLD,
    WATCHLIST_PORTS,
)
from src.threat_detection.utils import logger


@dataclass
class RuleFinding:
    """Structured result returned when an event matches a detection rule."""
    rule_id: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    category: str
    message: str

    def to_dict(self) -> Dict[str, Any]:
        """Converts finding to a standard dictionary."""
        return asdict(self)


def evaluate_single_event_rules(
    event: Dict[str, Any],
    failed_threshold: int = FAILED_CONNECTIONS_THRESHOLD,
    volume_threshold: int = HIGH_VOLUME_THRESHOLD,
    port_watchlist: Optional[Dict[int, str]] = None,
) -> List[RuleFinding]:
    """Evaluates defensive heuristic rules against a single network telemetry record.

    Rules evaluated:
    - RULE_DATA_01: Integrity check (negative values or failed > total).
    - RULE_FAIL_01: Repeated failed connections (potential brute force / auth attack).
    - RULE_VOL_01: Abnormal connection volume (potential flood / denial of service).
    - RULE_PORT_01: Access to administrative or commonly targeted service port.

    Args:
        event: Dictionary representing one network event.
        failed_threshold: Threshold for repeated connection failures.
        volume_threshold: Threshold for high connection count.
        port_watchlist: Optional custom mapping of monitored ports to service descriptions.

    Returns:
        List of RuleFinding objects for each matched heuristic.
    """
    if port_watchlist is None:
        port_watchlist = WATCHLIST_PORTS

    findings: List[RuleFinding] = []

    # Safe extraction of numeric fields with fallback
    conn_count = event.get("connection_count")
    failed_conn = event.get("failed_connections")
    dst_port = event.get("destination_port")

    # Convert to numeric if possible
    try:
        conn_count_num = float(conn_count) if conn_count is not None and pd.notnull(conn_count) else 0.0
    except (ValueError, TypeError):
        conn_count_num = 0.0

    try:
        failed_conn_num = float(failed_conn) if failed_conn is not None and pd.notnull(failed_conn) else 0.0
    except (ValueError, TypeError):
        failed_conn_num = 0.0

    # Rule DATA_01: Data consistency & sanity check
    if conn_count_num < 0 or failed_conn_num < 0:
        findings.append(
            RuleFinding(
                rule_id="RULE_DATA_01",
                severity="LOW",
                category="DATA_INTEGRITY_VIOLATION",
                message="Negative connection count or failed counter detected; telemetry anomaly.",
            )
        )
    elif failed_conn_num > conn_count_num:
        findings.append(
            RuleFinding(
                rule_id="RULE_DATA_01",
                severity="MEDIUM",
                category="DATA_INTEGRITY_VIOLATION",
                message=f"Failed connection count ({int(failed_conn_num)}) exceeds total connections ({int(conn_count_num)}).",
            )
        )

    # Rule FAIL_01: Repeated connection failures
    if failed_conn_num >= failed_threshold:
        severity = "HIGH" if failed_conn_num >= (failed_threshold * 3) else "MEDIUM"
        findings.append(
            RuleFinding(
                rule_id="RULE_FAIL_01",
                severity=severity,
                category="AUTHENTICATION_ANOMALY",
                message=(
                    f"Observed {int(failed_conn_num)} failed connections "
                    f"(configured threshold: {failed_threshold}). Possible credential or probe activity."
                ),
            )
        )

    # Rule VOL_01: High connection volume spike
    if conn_count_num >= volume_threshold:
        severity = "HIGH" if conn_count_num >= (volume_threshold * 2) else "MEDIUM"
        findings.append(
            RuleFinding(
                rule_id="RULE_VOL_01",
                severity=severity,
                category="TRAFFIC_VOLUME_SPIKE",
                message=(
                    f"Connection volume reached {int(conn_count_num)} connections "
                    f"(configured threshold: {volume_threshold}). Possible DoS/burst traffic."
                ),
            )
        )

    # Rule PORT_01: Sensitive or high-risk destination port
    if dst_port is not None and pd.notnull(dst_port):
        try:
            port_int = int(dst_port)
            if port_int in port_watchlist:
                service_desc = port_watchlist[port_int]
                findings.append(
                    RuleFinding(
                        rule_id="RULE_PORT_01",
                        severity="LOW",
                        category="SUSPICIOUS_PORT_ACCESS",
                        message=(
                            f"Connection to watched port {port_int} ({service_desc}). "
                            "Heuristic flag: review whether this service is expected."
                        ),
                    )
                )
        except (ValueError, TypeError):
            pass

    return findings


def evaluate_dataframe_rules(
    df: pd.DataFrame,
    failed_threshold: int = FAILED_CONNECTIONS_THRESHOLD,
    volume_threshold: int = HIGH_VOLUME_THRESHOLD,
    port_watchlist: Optional[Dict[int, str]] = None,
) -> List[Dict[str, Any]]:
    """Evaluates rules across all rows in a DataFrame.

    Args:
        df: Input DataFrame of network records.
        failed_threshold: Threshold for repeated connection failures.
        volume_threshold: Threshold for high volume.
        port_watchlist: Optional custom watched ports dictionary.

    Returns:
        List of event finding records, each combining event metadata with finding details.
    """
    all_findings = []

    for idx, row in df.iterrows():
        event_dict = row.to_dict()
        findings = evaluate_single_event_rules(
            event=event_dict,
            failed_threshold=failed_threshold,
            volume_threshold=volume_threshold,
            port_watchlist=port_watchlist,
        )
        for finding in findings:
            record = {
                "event_index": idx,
                "event_id": event_dict.get("event_id", f"EVT-{idx:05d}"),
                "timestamp": event_dict.get("timestamp", ""),
                "source_ip": event_dict.get("source_ip", "UNKNOWN"),
                "destination_ip": event_dict.get("destination_ip", "UNKNOWN"),
                "protocol": event_dict.get("protocol", "UNKNOWN"),
                "destination_port": event_dict.get("destination_port", None),
                "simulated": bool(event_dict.get("simulated", False)),
                **finding.to_dict(),
            }
            all_findings.append(record)

    logger.debug(f"Rule evaluation completed: generated {len(all_findings)} findings across {len(df)} events.")
    return all_findings
