# System Architecture & Technical Design

## 1. Overview
The **AI-Powered Cybersecurity Threat Detection System** is designed as a modular defensive analysis pipeline. It decouples data ingestion, preprocessing, heuristic analysis, machine learning inference, and presentation.

```
                          +-------------------------------+
                          |   Network Telemetry Input     |
                          | (Simulated or Public CSV File)|
                          +---------------+---------------+
                                          |
                                          v
                          +---------------+---------------+
                          |    Data Loader & Validator    |
                          | (Schema checks & dimensions)  |
                          +---------------+---------------+
                                          |
                                          v
                          +---------------+---------------+
                          |   Preprocessing & Cleaning    |
                          | (Deduplication, inf handling) |
                          +---------------+---------------+
                                          |
                                          v
                          +---------------+---------------+
                          |     Feature Engineering       |
                          | (Domain ratios & port flags)  |
                          +---------------+---------------+
                                          |
                +-------------------------+-------------------------+
                |                         |                         |
                v                         v                         v
     +--------------------+    +--------------------+    +--------------------+
     | Rule-Based Engine  |    |  Supervised Model  |    |  Anomaly Detector  |
     | (Heuristic checks) |    | (Random Forest/LR) |    | (Isolation Forest) |
     +----------+---------+    +----------+---------+    +----------+---------+
                |                         |                         |
                +-------------------------+-------------------------+
                                          |
                                          v
                          +---------------+---------------+
                          | Unified Prediction & Triage   |
                          |   (Correlation & Synthesis)   |
                          +---------------+---------------+
                                          |
                                          v
                          +---------------+---------------+
                          | Structured Alert Compilation  |
                          | (Deduplication & Severity)    |
                          +---------------+---------------+
                                          |
                +-------------------------+-------------------------+
                |                                                   |
                v                                                   v
     +--------------------+                              +--------------------+
     | Streamlit Web App  |                              | CSV Alert Export & |
     | (Visual Analytics) |                              | Evaluation Reports |
     +--------------------+                              +--------------------+
```

---

## 2. Component Breakdown

### A. Configuration & Utilities (`config.py`, `utils.py`)
- Centralizes all path definitions using Python's standard `pathlib.Path` to guarantee platform independence across Windows, macOS, and Linux.
- Sets global random seeds (default: `42`) for full experimental reproducibility.
- Defines heuristic thresholds (`FAILED_CONNECTIONS_THRESHOLD = 5`, `HIGH_VOLUME_THRESHOLD = 100`).
- Provides centralized logging, ISO-8601 UTC timestamp formatting, and custom validation exceptions (`DataValidationError`, `ModelPipelineError`).

### B. Ingestion & Validation (`data_loader.py`)
- Safe reading of CSV streams with size checks and parsing validation.
- Validates non-empty datasets, summarizes missing value distributions, row and column counts, and column types.

### C. Data Cleaning & Leakage Prevention (`preprocessing.py`)
- Replaces infinite values (`+/- inf`) with `NaN` so Scikit-learn imputers can process them.
- Deduplicates identical records.
- Standardizes binary targets (`0 = Normal`, `1 = Suspicious`).
- **Data Leakage Safeguard**: Drops non-predictive identifiers (`event_id`, `timestamp`, `source_ip`, `destination_ip`) and target labels before feeding features into training pipelines.
- Implements `ColumnTransformer` wrapping `SimpleImputer` and `StandardScaler` (for numbers) and `OneHotEncoder(handle_unknown='ignore')` (for categories).

### D. Feature Engineering (`features.py`)
- Calculates defensive ratios with division-by-zero protection:
  - `failure_ratio = failed_connections / max(connection_count, 1)`
  - `bytes_per_connection = bytes_transferred / max(connection_count, 1)`
  - `bytes_per_second = bytes_transferred / max(duration_seconds, 0.001)`
  - `is_watchlist_port = 1 if destination_port in WATCHLIST_PORTS else 0`

### E. Defensive Heuristic Rules (`rules.py`)
- Transparent, deterministic detection logic:
  - `RULE_FAIL_01`: Flags repeated connection failures (potential brute-force/credential stuffing).
  - `RULE_VOL_01`: Flags traffic spikes exceeding volume thresholds (potential DoS/burst flood).
  - `RULE_PORT_01`: Flags access to monitored administrative ports (SSH, Telnet, RDP, Metasploit default).
  - `RULE_DATA_01`: Flags data inconsistencies (e.g. `failed_connections > connection_count`).

### F. Machine Learning Engine (`model.py`, `anomaly_detector.py`)
- **Supervised Learning**: Evaluates Random Forest and Logistic Regression using stratified train/test splits. Model artifacts are serialized with Joblib.
- **Unsupervised Anomaly Detection**: Isolation Forest detects multivariate outliers in continuous feature space, fitted on baseline normal traffic.

### G. Prediction & Correlation Orchestrator (`prediction.py`, `alerts.py`)
- Correlates findings from all three engines.
- Maintains strict distinction between rule findings, statistical probabilities, and unsupervised anomalies.
- Deduplicates alerts and assigns explainable severity scores (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`).
