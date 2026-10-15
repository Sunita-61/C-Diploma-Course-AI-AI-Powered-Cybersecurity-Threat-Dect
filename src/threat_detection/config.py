"""Centralized project configuration and directory management.

Uses pathlib for platform-independent paths and provides defaults for detection
rules, machine learning pipelines, and simulation parameters.
"""

from pathlib import Path
from typing import Dict, List, Set

# Base directory paths (relative to repository root)
PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

# Data directory paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SIMULATED_DATA_DIR = DATA_DIR / "simulated"
UNSW_NB15_DIR = RAW_DATA_DIR / "UNSW-NB15"
UNSW_NB15_TRAIN_FILE = UNSW_NB15_DIR / "UNSW_NB15_training-set.csv"
UNSW_NB15_TEST_FILE = UNSW_NB15_DIR / "UNSW_NB15_testing-set.csv"


# Models and Artifacts
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
FIGURES_DIR = OUTPUTS_DIR / "figures"
ALERTS_DIR = OUTPUTS_DIR / "alerts"
DOCS_DIR = PROJECT_ROOT / "docs"

# Simulation and reproducibility
DEFAULT_RANDOM_SEED: int = 42
DEFAULT_SIMULATION_COUNT: int = 250

# Rule-Based Detection Thresholds (Configurable Heuristics)
FAILED_CONNECTIONS_THRESHOLD: int = 5
HIGH_VOLUME_THRESHOLD: int = 100
SUSPICIOUS_RATIO_THRESHOLD: float = 0.50

# Configurable demonstration port watchlist
# Note: In real networks, these ports host common services; they are flagged here
# strictly as an educational heuristic when unexpected volumes/failures occur.
WATCHLIST_PORTS: Dict[int, str] = {
    21: "FTP (Unencrypted File Transfer)",
    22: "SSH (Remote Administration)",
    23: "Telnet (Insecure Remote Shell)",
    25: "SMTP (Mail Relay)",
    3389: "RDP (Windows Remote Desktop)",
    4444: "Default Listener / Meterpreter Port",
    445: "SMB (Server Message Block)",
    139: "NetBIOS Session Service",
    8080: "HTTP Alternate / Proxy",
}

# Machine Learning Defaults
DEFAULT_TEST_SIZE: float = 0.25
DEFAULT_ANOMALY_CONTAMINATION: float = 0.05
DEFAULT_RF_ESTIMATORS: int = 100
DEFAULT_MAX_ITER: int = 1000

# Supported protocols and common target columns
SUPPORTED_PROTOCOLS: List[str] = ["TCP", "UDP", "ICMP"]
POTENTIAL_TARGET_COLUMNS: List[str] = [
    "label",
    "Label",
    "attack_cat",
    "class",
    "Class",
    "target",
    "is_attack",
]

UNSW_NB15_ATTACK_CATEGORIES: List[str] = [
    "Normal",
    "Fuzzers",
    "Analysis",
    "Backdoors",
    "DoS",
    "Exploits",
    "Generic",
    "Reconnaissance",
    "Shellcode",
    "Worms",
]


# Supported file formats
SUPPORTED_FILE_EXTENSIONS: Set[str] = {".csv"}
