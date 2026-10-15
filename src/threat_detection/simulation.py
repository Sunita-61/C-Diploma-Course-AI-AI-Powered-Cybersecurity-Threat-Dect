"""Synthetic network event generator for reproducible testing and dashboard demos.

Generates realistic network activity logs adhering to RFC 5737 documentation IP ranges.
Simulates normal activity as well as suspicious patterns (brute force, high volume,
and unusual port probes) without performing any real network operations.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Union
import numpy as np
import pandas as pd

from src.threat_detection.config import (
    DEFAULT_RANDOM_SEED,
    DEFAULT_SIMULATION_COUNT,
    SIMULATED_DATA_DIR,
    WATCHLIST_PORTS,
)
from src.threat_detection.utils import (
    DataValidationError,
    ensure_directory,
    get_current_utc_timestamp,
    logger,
)

# RFC 5737 Reserved IPv4 blocks for documentation and examples
DOC_SRC_PREFIXES = ["192.0.2.", "198.51.100."]
DOC_DST_PREFIXES = ["203.0.113.", "198.51.100."]

SCENARIOS = [
    "normal_web_browsing",
    "repeated_connection_failures",
    "high_connection_volume",
    "unusual_destination_port",
]


def generate_synthetic_events(
    count: int = DEFAULT_SIMULATION_COUNT,
    seed: int = DEFAULT_RANDOM_SEED,
    anomaly_ratio: float = 0.20,
) -> pd.DataFrame:
    """Generates synthetic network telemetry records for educational demonstration.

    Args:
        count: Number of events to generate (must be >= 1).
        seed: Random seed for exact reproducibility.
        anomaly_ratio: Proportion of suspicious/anomalous scenario events (0.0 to 1.0).

    Returns:
        pd.DataFrame containing simulated network records with standard schema.

    Raises:
        DataValidationError: If count < 1 or anomaly_ratio is outside [0, 1].
    """
    if count < 1:
        raise DataValidationError(f"Simulation event count must be at least 1, got {count}.")
    if not 0.0 <= anomaly_ratio <= 1.0:
        raise DataValidationError(
            f"Anomaly ratio must be between 0.0 and 1.0, got {anomaly_ratio}."
        )

    rng = np.random.default_rng(seed)
    base_time = datetime.now(timezone.utc) - timedelta(hours=2)

    records = []
    suspicious_scenarios = [
        "repeated_connection_failures",
        "high_connection_volume",
        "unusual_destination_port",
    ]

    watchlist_port_list = list(WATCHLIST_PORTS.keys())

    for idx in range(count):
        event_id = f"SIM-{idx + 1:05d}"
        event_time = base_time + timedelta(seconds=int(idx * 25 + rng.integers(-5, 5)))
        timestamp_str = event_time.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Determine scenario based on anomaly ratio
        is_suspicious = rng.random() < anomaly_ratio
        if not is_suspicious:
            scenario = "normal_web_browsing"
            protocol = rng.choice(["TCP", "UDP"], p=[0.85, 0.15])
            dst_port = int(rng.choice([80, 443, 53, 8080], p=[0.25, 0.60, 0.12, 0.03]))
            connection_count = int(rng.integers(1, 45))
            # Normal failures are 0 or rarely 1-2
            failed_connections = int(rng.choice([0, 1, 2], p=[0.85, 0.12, 0.03]))
            failed_connections = min(failed_connections, connection_count)
            bytes_transferred = int(rng.integers(200, 50000))
            duration_seconds = round(float(rng.uniform(0.1, 15.0)), 2)
            src_ip = f"{rng.choice(DOC_SRC_PREFIXES)}{rng.integers(10, 50)}"
            dst_ip = f"{rng.choice(DOC_DST_PREFIXES)}{rng.integers(1, 20)}"

        else:
            scenario = str(rng.choice(suspicious_scenarios))
            if scenario == "repeated_connection_failures":
                # High failure rate (e.g., brute force / credential guessing)
                protocol = "TCP"
                dst_port = int(rng.choice([21, 22, 23, 3389]))
                connection_count = int(rng.integers(15, 80))
                # High ratio of failures (80% to 100%)
                fail_rate = float(rng.uniform(0.70, 0.95))
                failed_connections = max(1, int(connection_count * fail_rate))
                failed_connections = min(failed_connections, connection_count)
                bytes_transferred = int(rng.integers(500, 12000))
                duration_seconds = round(float(rng.uniform(1.0, 30.0)), 2)
                src_ip = f"{rng.choice(DOC_SRC_PREFIXES)}{rng.integers(100, 200)}"
                dst_ip = f"{rng.choice(DOC_DST_PREFIXES)}{rng.integers(1, 10)}"

            elif scenario == "high_connection_volume":
                # Spike in volume (e.g. DoS / SYN flood simulation)
                protocol = rng.choice(["TCP", "UDP"], p=[0.7, 0.3])
                dst_port = int(rng.choice([80, 443, 8080]))
                connection_count = int(rng.integers(120, 500))
                failed_connections = int(rng.integers(10, 60))
                failed_connections = min(failed_connections, connection_count)
                bytes_transferred = int(rng.integers(100000, 2000000))
                duration_seconds = round(float(rng.uniform(0.05, 5.0)), 2)
                src_ip = f"{rng.choice(DOC_SRC_PREFIXES)}{rng.integers(201, 250)}"
                dst_ip = f"{rng.choice(DOC_DST_PREFIXES)}{rng.integers(1, 5)}"

            else:  # unusual_destination_port
                protocol = "TCP"
                dst_port = int(rng.choice(watchlist_port_list))
                connection_count = int(rng.integers(5, 40))
                failed_connections = int(rng.integers(0, 5))
                failed_connections = min(failed_connections, connection_count)
                bytes_transferred = int(rng.integers(1000, 30000))
                duration_seconds = round(float(rng.uniform(0.5, 12.0)), 2)
                src_ip = f"{rng.choice(DOC_SRC_PREFIXES)}{rng.integers(50, 99)}"
                dst_ip = f"{rng.choice(DOC_DST_PREFIXES)}{rng.integers(21, 30)}"

        # Ground truth label for educational validation (0 = Normal, 1 = Suspicious)
        label = 0 if scenario == "normal_web_browsing" else 1

        records.append({
            "event_id": event_id,
            "timestamp": timestamp_str,
            "source_ip": src_ip,
            "destination_ip": dst_ip,
            "protocol": protocol,
            "destination_port": dst_port,
            "connection_count": connection_count,
            "failed_connections": failed_connections,
            "bytes_transferred": bytes_transferred,
            "duration_seconds": duration_seconds,
            "scenario": scenario,
            "label": label,
            "simulated": True,
        })

    df = pd.DataFrame(records)
    logger.info(
        f"Generated {len(df)} synthetic events (Seed: {seed}, Suspicious: {int(df['label'].sum())})"
    )
    return df


def save_simulated_events(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Saves generated simulation DataFrame to CSV.

    Args:
        df: Synthetic DataFrame.
        output_path: Optional destination Path. Defaults to data/simulated/simulated_traffic.csv.

    Returns:
        Path to the saved CSV file.
    """
    if output_path is None:
        ensure_directory(SIMULATED_DATA_DIR)
        output_path = SIMULATED_DATA_DIR / "simulated_traffic.csv"
    else:
        output_path = Path(output_path)
        ensure_directory(output_path.parent)

    df.to_csv(output_path, index=False)
    logger.info(f"Simulated events saved to: {output_path}")
    return output_path
