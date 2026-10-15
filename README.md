# AI-Powered Cybersecurity Threat Detection System

[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.14-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: Pytest](https://img.shields.io/badge/Tests-35%20Passed-brightgreen.svg)](tests/)
[![UI: Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)


A modular, explainable defensive cybersecurity application that analyzes network activity records, correlates heuristic rules with supervised machine learning and unsupervised anomaly detection, and presents actionable security alerts through an interactive Streamlit dashboard.

---

## 1. Problem Statement
Security Operations Centers (SOCs) face overwhelming volumes of network logs daily. Traditional rule-based firewalls produce high rates of false positives or miss subtle multi-stage attacks, while black-box machine learning models lack the transparency needed for swift human triage. 

This project demonstrates a **defense-in-depth architecture** that bridges deterministic heuristics, supervised classification, and unsupervised anomaly detection, providing human analysts with traceable, prioritized alerts.

---

## 2. Project Objectives
- **Data Ingestion & Hygiene**: Validate schemas, deduplicate telemetry, and handle missing and infinite values safely.
- **Leakage-Free Preprocessing**: Implement Scikit-learn pipelines where transformations are fitted strictly on training data, stripping non-predictive identifiers.
- **Multi-Layered Detection Engine**:
  - *Heuristic Rules*: Fast, explainable checks for repeated authentication failures, volume spikes, and watched port probes.
  - *Supervised Learning*: Random Forest and Logistic Regression classifiers trained to detect known attack patterns.
  - *Unsupervised Learning*: Isolation Forest detecting multivariate continuous outliers without ground-truth labels.
- **Alert Triage & Correlation**: Synthesize findings into structured alerts with deterministic IDs, severity scoring, and CSV export.
- **Interactive Security Portal**: A beginner-friendly Streamlit web application running offline with synthetic demonstrations or custom CSV uploads.

---

## 3. Technology Stack
- **Language**: Python 3.11 / 3.14
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (Random Forest, Logistic Regression, Isolation Forest, ColumnTransformer, Pipeline)
- **Model Persistence**: Joblib
- **Visualization**: Matplotlib, Seaborn
- **Web Application**: Streamlit
- **Testing**: Pytest, Pytest-Cov
- **Version Control**: Git

---

## 4. System Architecture

```
                  +-----------------------------------+
                  | Network Activity (Simulated/CSV)  |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------+-----------------+
                  | Ingestion, Cleaning & Feature Eng |
                  +-----------------+-----------------+
                                    |
            +-----------------------+-----------------------+
            |                       |                       |
            v                       v                       v
 +--------------------+   +--------------------+  +--------------------+
 | Heuristic Rules    |   | Supervised ML      |  | Anomaly Detector   |
 | (Failures, Volume, |   | (Random Forest,    |  | (Isolation Forest) |
 |  Port Watchlist)   |   |  Logistic Reg.)    |  |                    |
 +----------+---------+   +---------+----------+  +---------+----------+
            |                       |                       |
            +-----------------------+-----------------------+
                                    |
                                    v
                  +-----------------+-----------------+
                  |   Unified Threat Orchestrator     |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------+-----------------+
                  |   Structured Security Alerts      |
                  +-----------------+-----------------+
                                    |
                     +--------------+--------------+
                     |                             |
                     v                             v
           +--------------------+        +--------------------+
           | Streamlit Web App  |        | CSV Export & Eval  |
           | (Analyst Console)  |        | (Metrics & Plots)  |
           +--------------------+        +--------------------+
```

---

## 5. Folder Structure

```
AI-Cybersecurity-Threat-Detection/
│
├── app/
│   ├── __init__.py
│   └── dashboard.py               # Streamlit interactive security analyst dashboard
│
├── data/
│   ├── raw/                       # Raw external datasets (ignored by git)
│   ├── processed/                 # Cleaned and feature-engineered datasets
│   ├── simulated/                 # Synthetic demonstration network telemetry
│   └── README.md
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_experiments.ipynb
│   └── README.md
│
├── src/
│   └── threat_detection/
│       ├── __init__.py
│       ├── config.py              # Centralized paths and thresholds
│       ├── utils.py               # Logging, paths, and DataFrame validation
│       ├── simulation.py          # Synthetic telemetry generator (RFC 5737 IPs)
│       ├── data_loader.py         # CSV loading and schema summary inspection
│       ├── preprocessing.py       # Data cleaning, target encoding, pipeline builder
│       ├── features.py            # Defensive domain ratios and port flags
│       ├── rules.py               # Transparent heuristic rule checks
│       ├── model.py               # Supervised Random Forest and Logistic Regression
│       ├── anomaly_detector.py    # Isolation Forest outlier detection
│       ├── prediction.py          # Unified threat correlation orchestrator
│       └── alerts.py              # Structured security alert generation and export
│
├── scripts/
│   ├── inspect_data.py            # CLI: Inspect dataset dimensions, types, and labels
│   ├── prepare_data.py            # CLI: Clean and feature-engineer raw datasets
│   ├── train_model.py             # CLI: Train Random Forest / Logistic Regression / IsoForest
│   └── evaluate_model.py          # CLI: Evaluate models, generate reports and plots
│
├── models/                        # Saved model artifacts (.joblib)
├── outputs/
│   ├── reports/                   # Model evaluation text & JSON reports
│   ├── figures/                   # Confusion matrix plots (.png)
│   └── alerts/                    # Exported security alert CSVs
│
├── images/
│   ├── .gitkeep
│   └── README.md                  # Screenshot capture guide
│
├── docs/
│   ├── architecture.md            # Detailed component design
│   ├── dataset_documentation.md   # External benchmark ingestion guide
│   ├── model_card.md              # Model governance and performance details
│   ├── threat_model.md            # Defensive scope and threat boundaries
│   └── limitations.md             # Transparent analysis of false positives/negatives
│
├── tests/
│   ├── test_alerts.py
│   ├── test_data_loader.py
│   ├── test_features.py
│   ├── test_prediction.py
│   ├── test_preprocessing.py
│   ├── test_rules.py
│   └── test_unsw_integration.py
│
├── main.py                        # Centralized CLI entry point
├── requirements.txt               # Core runtime dependencies
├── requirements-dev.txt           # Development and testing dependencies
├── .gitignore                     # Git ignore rules
├── .gitattributes                 # Line endings configuration
├── LICENSE                        # MIT License with dataset attribution disclaimer
├── CONTRIBUTING.md                # Contribution guidelines
└── README.md                      # Comprehensive project documentation
```

---

## 6. Installation & Environment Setup

Run the following commands in **Windows PowerShell**:

```powershell
# 1. Verify Python is installed (Python 3.11+ recommended)
python --version

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# 4. Upgrade pip and install runtime & testing dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

---

## 7. Running the Automated Test Suite

Execute the 35 automated unit and integration tests across all core modules:

```powershell
python -m pytest tests -v
```

All 35 tests execute in ~1.8 seconds without external network calls or large dataset downloads.

---

## 8. Quickstart: One-Command Offline Demonstration

To run the complete pipeline offline (data generation, model training, evaluation, threat correlation, and alert export):

```powershell
python main.py run-demo
```

---

## 9. Launching the Interactive Web Dashboard

Launch the Streamlit analyst interface:

```powershell
streamlit run app/dashboard.py
```

The browser will open to `http://localhost:8501`. 
- **Synthetic Demonstration Mode**: Allows instant exploration with configurable event counts, random seeds, and anomaly ratios.
- **Upload Network CSV**: Allows uploading custom network flow datasets for instant automated triage.
- **Alert Triage & Export**: Filter by severity or detection engine and export correlated findings to CSV.

---

## 10. Command-Line Workflow Guide

### A. Inspect Datasets
```powershell
# 1. Inspect synthetic telemetry
python scripts/inspect_data.py --data-path data/simulated/simulated_traffic.csv --target-col label

# 2. Inspect UNSW-NB15 dataset (binary target)
python scripts/inspect_data.py --data-path data/raw/UNSW-NB15/UNSW_NB15_training-set.csv --target-col label

# 3. Inspect UNSW-NB15 attack categories (multiclass target)
python scripts/inspect_data.py --data-path data/raw/UNSW-NB15/UNSW_NB15_training-set.csv --target-col attack_cat
```

### B. Clean & Engineer Features
```powershell
# Prepares cleaned dataset and calculates defensive behavioral ratios
python scripts/prepare_data.py --input-path data/simulated/simulated_traffic.csv
```

### C. Train Machine Learning Models

#### 1. Synthetic Telemetry (Default Demo):
```powershell
# Train Random Forest
python scripts/train_model.py --data-path data/simulated/simulated_traffic.csv --model-type random_forest

# Train Logistic Regression
python scripts/train_model.py --data-path data/simulated/simulated_traffic.csv --model-type logistic_regression
```

#### 2. UNSW-NB15 Predefined Partitions (Official Benchmark):
```powershell
# Train Binary Random Forest (Normal vs Attack)
python scripts/train_model.py --train-path data/raw/UNSW-NB15/UNSW_NB15_training-set.csv --test-path data/raw/UNSW-NB15/UNSW_NB15_testing-set.csv --target-col label --model-type random_forest --model-name unsw_nb15_rf_binary.joblib

# Train Multiclass Random Forest (9 Attack Categories)
python scripts/train_model.py --train-path data/raw/UNSW-NB15/UNSW_NB15_training-set.csv --test-path data/raw/UNSW-NB15/UNSW_NB15_testing-set.csv --target-col attack_cat --model-type random_forest --model-name unsw_nb15_rf_multiclass.joblib

# Train Binary Logistic Regression
python scripts/train_model.py --train-path data/raw/UNSW-NB15/UNSW_NB15_training-set.csv --test-path data/raw/UNSW-NB15/UNSW_NB15_testing-set.csv --target-col label --model-type logistic_regression --model-name unsw_nb15_lr_binary.joblib
```

### D. Evaluate Saved Models

#### 1. Evaluate Synthetic Models:
```powershell
python scripts/evaluate_model.py --model-path models/random_forest.joblib --data-path data/simulated/simulated_traffic.csv --target-col label
```

#### 2. Evaluate UNSW-NB15 Models on Independent Official Test Partition:
```powershell
# Evaluate Binary Model on Testing Partition
python scripts/evaluate_model.py --model-path models/unsw_nb15_rf_binary.joblib --data-path data/raw/UNSW-NB15/UNSW_NB15_testing-set.csv --target-col label --is-test-set

# Evaluate Multiclass Model on Testing Partition
python scripts/evaluate_model.py --model-path models/unsw_nb15_rf_multiclass.joblib --data-path data/raw/UNSW-NB15/UNSW_NB15_testing-set.csv --target-col attack_cat --is-test-set

# Evaluate Binary Logistic Regression Baseline on Testing Partition
python scripts/evaluate_model.py --model-path models/unsw_nb15_lr_binary.joblib --data-path data/raw/UNSW-NB15/UNSW_NB15_testing-set.csv --target-col label --is-test-set
```

Evaluation generates:
- Text report: `outputs/reports/<model_name>_evaluation_report.txt`
- JSON metrics: `outputs/reports/<model_name>_evaluation_report.json`
- Confusion matrix figure: `outputs/figures/<model_name>_confusion_matrix.png`

---

## 11. Official UNSW-NB15 Benchmark Evaluation Results

All models were evaluated strictly on the official independent test set (`UNSW_NB15_testing-set.csv`, 82,332 records) after training exclusively on the official training set (`UNSW_NB15_training-set.csv`, 175,341 records). Non-predictive IDs and alternate targets were dropped to guarantee zero target leakage.

### A. Binary Classification Comparison (`label` = 0 Normal, 1 Attack)

| Metric | Random Forest (`unsw_nb15_rf_binary`) | Logistic Regression Baseline (`unsw_nb15_lr_binary`) | Operational Impact in SOC |
| :--- | :---: | :---: | :--- |
| **Accuracy** | **89.21%** | 83.59% | Overall session classification correctness. |
| **Attack Detection Rate (Recall)** | **97.46%** | 93.09% | **Primary Metric**: RF detects 97.46% of all true network attacks. |
| **Attack Precision (pos=1)** | **85.10%** | 80.26% | Proportion of attack-flagged sessions that are genuine threats. |
| **Attack F1-Score (pos=1)** | **90.86%** | 86.20% | Positive-class harmonic mean for malicious traffic. |
| **Weighted Precision** | **90.10%** | 84.40% | Multi-class weighted precision accounting for normal traffic. |
| **Weighted Recall** | **89.21%** | 83.59% | Multi-class weighted recall across all sessions. |
| **Weighted F1-Score** | **89.05%** | 83.31% | Class-prevalence weighted F1 across benign and malicious flows. |
| **Macro F1-Score** | **88.84%** | 82.98% | Unweighted average F1 across normal and attack classes. |
| **ROC-AUC** | **0.9804** | 0.9559 | Excellent separation capability across probability thresholds. |
| **PR-AUC (Avg Precision)** | **0.9838** | 0.9666 | Superior precision-recall trade-off under attack class imbalance. |
| **False Negatives (Missed Attacks)** | **1,151** | 3,133 | **RF delivers a 63.2% reduction in missed attacks** compared to LR. |
| **False Positives (False Alarms)** | **7,734** | 10,377 | **SOC Burden**: 7,734 benign flows flagged as threats require analyst triage. |

### B. Multiclass Attack Categorization (`unsw_nb15_rf_multiclass`)

- **Overall Accuracy**: 69.21% | **Weighted-F1**: 74.20% | **Macro-F1**: 50.10% | **ROC-AUC (OvR Weighted)**: 0.9416
- **Per-Class Breakdown**:
  - `Generic`: **96%** recall (Precision: 1.00, F1: 0.98, Support: 18,871)
  - `Reconnaissance`: **84%** recall (Precision: 0.85, F1: 0.84, Support: 3,496)
  - `Shellcode`: **80%** recall (Precision: 0.28, F1: 0.42, Support: 378)
  - `Fuzzers`: **72%** recall (Precision: 0.26, F1: 0.38, Support: 6,062)
  - `Exploits`: **63%** recall (Precision: 0.78, F1: 0.70, Support: 11,132)
  - `Normal`: **62%** recall (Precision: 0.99, F1: 0.77, Support: 37,000)
  - `Backdoors`: **58%** recall (Precision: 0.05, F1: 0.09, Support: 583)
  - `Worms`: **52%** recall (Precision: 0.66, F1: 0.58, Support: 44)
  - `DoS`: **17%** recall (Precision: 0.38, F1: 0.24, Support: 4,089) — *Severe statistical overlap with Fuzzers and high-volume Exploits*.
  - `Analysis`: **2%** recall (Precision: 0.01, F1: 0.01, Support: 677) — *Stealthy, low-byte port scans blend into benign web traffic*.

> **Defensive Engineering Disclosure**:
> - **False-Positive Burden**: Although binary Random Forest detects 97.46% of attacks, its 7,734 false positives represent 20.9% of all benign testing flows being flagged. In an enterprise SOC, this requires secondary thresholding and rule correlation to mitigate alert fatigue.
> - **Weak DoS & Analysis Recall**: Multiclass ML alone cannot be trusted to categorize DoS (17% recall) or Analysis (2% recall). The system relies on its deterministic **Heuristic Rules** (`rule_high_connection_volume`, `rule_repeated_failed_connections`) and **Unsupervised Isolation Forest** to catch these specific threat categories.

---

## 12. Screenshots Section

When the dashboard is running, capture interface views and store them in `images/`:

| Overview & KPI Metrics | Alert Triage & Visual Analytics |
|:---:|:---:|
| *(Add `images/dashboard_overview.png`)* | *(Add `images/detection_insights.png`)* |

See [`images/README.md`](images/README.md) for instructions on capturing clean screenshots.

---

## 13. Ethical Use & Operational Limitations

- **Defensive Education Only**: This software is designed exclusively for authorized network defensive analytics and educational portfolio demonstrations. It contains no exploit payloads, network disruption mechanisms, or port scanners.
- **Privacy Assurance**: All synthetic telemetry utilizes RFC 5737 reserved documentation IP blocks (`192.0.2.0/24`, `198.51.100.0/24`, `203.0.113.0/24`).
- **No False Certainty**: Detections are labeled as **heuristic findings** or **statistical anomalies**, never as confirmed security breaches. Human investigation is required before taking incident response actions.
- Read [`docs/limitations.md`](docs/limitations.md) and [`docs/threat_model.md`](docs/threat_model.md) for deeper operational context.

---

## 14. Future Enhancements
- [ ] Add PCAP file ingestion support using Scapy.
- [ ] Implement sliding time-window aggregations for streaming event logs.
- [ ] Add SHAP (SHapley Additive exPlanations) for per-alert feature contribution explanations.
- [ ] Provide MITRE ATT&CK technique mapping for generated alerts.

---

## 15. License
This project is licensed under the [MIT License](LICENSE). Third-party datasets (such as CIC-IDS2017, UNSW-NB15, or KDD Cup 1999) are subject to their respective institutional licenses.
