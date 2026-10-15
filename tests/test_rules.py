"""Automated tests for defensive rule-based heuristics."""

import pandas as pd
import pytest

from src.threat_detection.rules import (
    evaluate_dataframe_rules,
    evaluate_single_event_rules,
)


def test_rule_repeated_failed_connections():
    """Verifies that failed connections exceeding threshold triggers RULE_FAIL_01."""
    event = {
        "event_id": "EVT-01",
        "connection_count": 20,
        "failed_connections": 15,
        "destination_port": 80,
    }
    findings = evaluate_single_event_rules(event, failed_threshold=5)

    rule_ids = [f.rule_id for f in findings]
    assert "RULE_FAIL_01" in rule_ids


def test_rule_high_connection_volume():
    """Verifies that total connection count exceeding threshold triggers RULE_VOL_01."""
    event = {
        "event_id": "EVT-02",
        "connection_count": 350,
        "failed_connections": 2,
        "destination_port": 443,
    }
    findings = evaluate_single_event_rules(event, volume_threshold=100)

    rule_ids = [f.rule_id for f in findings]
    assert "RULE_VOL_01" in rule_ids


def test_rule_invalid_event_counters():
    """Verifies that counter inconsistency triggers RULE_DATA_01."""
    # Failed exceeds total
    event_inconsistent = {
        "event_id": "EVT-03",
        "connection_count": 10,
        "failed_connections": 15,
    }
    findings = evaluate_single_event_rules(event_inconsistent)
    rule_ids = [f.rule_id for f in findings]
    assert "RULE_DATA_01" in rule_ids

    # Negative counter
    event_negative = {
        "event_id": "EVT-04",
        "connection_count": -5,
        "failed_connections": 0,
    }
    findings_neg = evaluate_single_event_rules(event_negative)
    assert "RULE_DATA_01" in [f.rule_id for f in findings_neg]


def test_multiple_findings_for_single_event():
    """Verifies that an event can trigger multiple distinct rules simultaneously."""
    event_multi = {
        "event_id": "EVT-05",
        "connection_count": 500,  # triggers high volume
        "failed_connections": 40,  # triggers repeated failures
        "destination_port": 3389,  # triggers watchlist port
    }
    findings = evaluate_single_event_rules(
        event_multi, failed_threshold=5, volume_threshold=100
    )

    rule_ids = {f.rule_id for f in findings}
    assert "RULE_VOL_01" in rule_ids
    assert "RULE_FAIL_01" in rule_ids
    assert "RULE_PORT_01" in rule_ids
    assert len(findings) == 3
