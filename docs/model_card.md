# Model Card: Threat Detection Classifiers & Anomaly Detector

## 1. Model Details
- **Architectures**:
  - Supervised 1: Random Forest Classifier (`sklearn.ensemble.RandomForestClassifier`, 100 estimators, balanced class weights).
  - Supervised 2: Logistic Regression (`sklearn.linear_model.LogisticRegression`, balanced class weights, max_iter=1000).
  - Unsupervised: Isolation Forest (`sklearn.ensemble.IsolationForest`, contamination=0.05).
- **Library Versions**: Scikit-learn >= 1.4, Joblib >= 1.3, Python 3.11 / 3.14.
- **Date**: 2026.
- **Task Types**:
  1. **Binary Classification**: Normal (`0`) vs Suspicious/Attack (`1`).
  2. **Multiclass Classification**: 9 attack categories (`Fuzzers`, `Analysis`, `Backdoors`, `DoS`, `Exploits`, `Generic`, `Reconnaissance`, `Shellcode`, `Worms`) vs `Normal`.
  3. **Unsupervised Anomaly Detection**: Continuous abnormality scoring for multivariate outliers.

---

## 2. Intended Use
- **Primary Use**: Educational portfolio demonstration, defensive telemetry analysis, and preliminary Security Operations Center (SOC) triage.
- **Out of Scope**: Autonomous firewall execution, automated packet drop without analyst review, or offensive counter-attacks.

---

## 3. Training & Evaluation Datasets
- **Supported Inputs**:
  - **Synthetic Telemetry**: Built-in 250-event generator (`seed=42`) using RFC 5737 documentation IPs.
  - **UNSW-NB15 Public Benchmark**: 175,341 training records and 82,332 testing records created by the Australian Centre for Cyber Security (ACCS).
- **Leakage Prevention Safeguards**:
  - Excludes non-predictive identifiers (`id`, `event_id`, `timestamp`, `source_ip`, `destination_ip`).
  - Automatically drops `attack_cat` when predicting `label`, and drops `label` when predicting `attack_cat`.
  - Fits all imputers, scalers, and encoders strictly on the training partition.
  - Handles unseen categorical values at test time via `OneHotEncoder(handle_unknown='ignore')`.

---

## 4. Evaluation Strategy & Reference Metrics

### A. Synthetic Reference Baseline (63 holdout samples)
- **Accuracy**: 0.9841
- **Precision (Weighted)**: 1.0000
- **Recall (Detection Rate)**: 0.9091
- **F1-Score (Weighted)**: 0.9524
- **ROC-AUC**: 1.0000
- **Confusion Matrix**: True Negatives = 52, False Positives = 0, False Negatives = 1, True Positives = 10.

### B. UNSW-NB15 Benchmark Evaluation (Official Test Set: 82,332 samples)
Evaluated strictly on `UNSW_NB15_testing-set.csv` (175,341 train, 82,332 test) without data leakage or test refitting:

#### 1. Binary Random Forest (`models/unsw_nb15_rf_binary.joblib`):
- **Accuracy**: 0.8921
- **Attack Precision (pos=1)**: 0.8510
- **Attack Recall / Detection Rate (pos=1)**: 0.9746 (97.46% of all true intrusions detected)
- **Attack F1-Score (pos=1)**: 0.9086
- **Weighted Precision**: 0.9010
- **Weighted Recall**: 0.8921
- **Weighted F1-Score**: 0.8905
- **Macro F1-Score**: 0.8884
- **ROC-AUC**: 0.9804
- **PR-AUC (Average Precision)**: 0.9838
- **Confusion Matrix**:
  - True Negatives: 29,266
  - False Positives: 7,734 (20.9% false alarm rate on benign traffic)
  - False Negatives: 1,151 (2.54% attack miss rate)
  - True Positives: 44,181

#### 2. Binary Logistic Regression Baseline (`models/unsw_nb15_lr_binary.joblib`):
- **Accuracy**: 0.8359
- **Attack Precision (pos=1)**: 0.8026
- **Attack Recall / Detection Rate (pos=1)**: 0.9309
- **Attack F1-Score (pos=1)**: 0.8620
- **Weighted Precision**: 0.8440
- **Weighted Recall**: 0.8359
- **Weighted F1-Score**: 0.8331
- **Macro F1-Score**: 0.8298
- **ROC-AUC**: 0.9559
- **PR-AUC (Average Precision)**: 0.9666
- **Comparison**: Random Forest achieves higher detection (+4.37% recall) with 63.2% fewer false negatives (1,151 vs 3,133).

#### 3. Multiclass Attack Categorization (`models/unsw_nb15_rf_multiclass.joblib`):
- **Accuracy**: 0.6921
- **Weighted Precision**: 0.8554
- **Weighted Recall**: 0.6921
- **Weighted F1-Score**: 0.7420
- **Macro F1-Score**: 0.5010
- **ROC-AUC (OvR Weighted)**: 0.9416
- **Per-Class Breakdown**:
  - Generic: Recall 0.96, Precision 1.00, F1 0.98, Support 18,871
  - Reconnaissance: Recall 0.84, Precision 0.85, F1 0.84, Support 3,496
  - Shellcode: Recall 0.80, Precision 0.28, F1 0.42, Support 378
  - Fuzzers: Recall 0.72, Precision 0.26, F1 0.38, Support 6,062
  - Exploits: Recall 0.63, Precision 0.78, F1 0.70, Support 11,132
  - Normal: Recall 0.62, Precision 0.99, F1 0.77, Support 37,000
  - Backdoors: Recall 0.58, Precision 0.05, F1 0.09, Support 583
  - Worms: Recall 0.52, Precision 0.66, F1 0.58, Support 44
  - DoS: Recall 0.17, Precision 0.38, F1 0.24, Support 4,089 (*Severe overlap with Fuzzers and high-rate Exploits*)
  - Analysis: Recall 0.02, Precision 0.01, F1 0.01, Support 677 (*Low-byte scans mimic benign browsing*)

Full evaluation reports and confusion matrix plots are saved to `outputs/reports/` and `outputs/figures/`.

---

## 5. Ethical Governance & Operational Limitations
- **No Guaranteed Protection**: Benchmark effectiveness does not imply 100% defense against live zero-day exploits or stealthy slow scans.
- **Distribution Shift**: Traffic distributions vary widely between enterprise networks and cyber range testbeds. Models require continuous retraining and tuning against local baselines.
- **SOC Fatigue vs Stealth**: The decision threshold should be calibrated to balance analyst alert volume (False Positives) against breach detection sensitivity (False Negatives).
