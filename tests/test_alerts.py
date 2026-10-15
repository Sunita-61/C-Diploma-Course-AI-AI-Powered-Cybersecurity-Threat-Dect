"""Automated tests for security alert structure, deduplication, and CSV export."""

from pathlib import Path
import pandas as pd
import pytest

from src.threat_detection.alerts import (
    SecurityAlert,
    compile_alerts,
    create_alert_from_anomaly,
    create_alert_from_ml_prediction,
    create_alert_from_rule_finding,
    export_alerts_to_csv,
)


def test_structured_alert_creation_and_schema():
    """Verifies that rule findings produce complete structured SecurityAlert objects."""
    finding = {
        "event_id": "EVT-100",
        "timestamp": "2026-01-01T12:00:00Z",
        "source_ip": "192.0.2.10",
        "destination_ip": "203.0.113.5",
        "severity": "HIGH",
        "category": "AUTHENTICATION_ANOMALY",
        "rule_id": "RULE_FAIL_01",
        "message": "Repeated failures detected",
        "simulated": True,
    }

    alert = create_alert_from_rule_finding(finding)

    assert isinstance(alert, SecurityAlert)
    assert alert.event_id == "EVT-100"
    assert alert.severity == "HIGH"
    assert alert.rule_id == "RULE_FAIL_01"
    assert alert.detection_method == "RULE_BASED"
    assert alert.status == "NEW"
    assert alert.simulated is True
    assert alert.alert_id.startswith("ALT-")


def test_alert_deduplication():
    """Verifies that identical findings from repeated processing are deduplicated."""
    finding = {
        "event_id": "EVT-200",
        "rule_id": "RULE_VOL_01",
        "severity": "MEDIUM",
        "category": "TRAFFIC_VOLUME_SPIKE",
        "message": "High volume",
    }

    alert1 = create_alert_from_rule_finding(finding)
    alert2 = create_alert_from_rule_finding(finding)

    deduped = compile_alerts([alert1, alert2])
    assert len(deduped) == 1


def test_export_alerts_to_csv(tmp_path: Path):
    """Verifies that compile and export writes alerts cleanly to a CSV file."""
    alert = SecurityAlert(
        alert_id="ALT-001",
        event_id="EVT-01",
        timestamp="2026-01-01T00:00:00Z",
        source_ip="192.0.2.1",
        destination_ip="203.0.113.1",
        severity="LOW",
        category="PORT_PROBE",
        detection_method="RULE_BASED",
        rule_id="RULE_PORT_01",
        description="Port test",
        status="NEW",
        simulated=True,
    )

    out_file = tmp_path / "test_alerts.csv"
    saved_path = export_alerts_to_csv([alert], output_path=out_file)

    assert saved_path.exists()
    df = pd.read_csv(saved_path)
    assert len(df) == 1
    assert df.iloc[0]["alert_id"] == "ALT-001"
    assert df.iloc[0]["simulated"] == True
