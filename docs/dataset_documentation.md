# Dataset Documentation & Ingestion Guide

This document describes how to obtain, organize, preprocess, and train on network security datasets with this system.

---

## 1. Supported Benchmark Datasets

### A. UNSW-NB15 Network Intrusion Dataset
- **Official Source**: [UNSW Canberra Cyber Research](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- **Creators**: Nour Moustafa and Jill Slay, Australian Centre for Cyber Security (ACCS).
- **Collection Setup**: Generated in the Cyber Range Lab of UNSW Canberra using the IXIA PerfectStorm tool to combine authentic modern normal network traffic with contemporary synthetic attack behaviors.
- **Collection Volume**: Approximately 2.54 million raw network flow records across four primary CSV captures.
- **Standard Predefined Benchmark Files**:
  - `UNSW_NB15_training-set.csv` (175,341 records)
  - `UNSW_NB15_testing-set.csv` (82,332 records)
- **Directory Location**:
  Place downloaded files into:
  ```
  data/raw/UNSW-NB15/
  ├── UNSW_NB15_training-set.csv
  ├── UNSW_NB15_testing-set.csv
  └── README.md
  ```

#### Features and Attributes (49 Documented Fields):
1. **Flow Identifiers & Basic Features**: `id`, `dur` (duration), `proto` (protocol), `service` (http, ftp, ssh, dns, etc.), `state`.
2. **Packet & Byte Counts**: `spkts`, `dpkts`, `sbytes`, `dbytes`, `rate`, `sttl`, `dttl`, `sload`, `dload`, `sloss`, `dloss`.
3. **Time & Jitter Features**: `sinpkt`, `dinpkt`, `sjit`, `djit`, `swin`, `stcpb`, `dtcpb`, `dwin`, `tcprtt`, `synack`, `ackdat`.
4. **Content & Connection Metrics**: `smean`, `dmean`, `trans_depth`, `response_body_len`, `ct_srv_src`, `ct_state_ttl`, `ct_dst_ltm`, `ct_src_dport_ltm`, `ct_dst_sport_ltm`, `ct_dst_src_ltm`, `is_ftp_login`, `ct_ftp_cmd`, `ct_flw_http_mthd`, `ct_src_ltm`, `ct_srv_dst`, `is_sm_ips_ports`.
5. **Target Columns**:
   - Binary Target: `label` (`0 = Normal`, `1 = Attack`).
   - Multiclass Target: `attack_cat` (Nine attack families: `Fuzzers`, `Analysis`, `Backdoors`, `DoS`, `Exploits`, `Generic`, `Reconnaissance`, `Shellcode`, `Worms`, plus `Normal`).

#### Target Leakage Safeguards for UNSW-NB15:
- **Exclusion of Row ID**: The `id` column must always be dropped; otherwise, tree-based models learn partition indices rather than behavioral traffic features.
- **Mutual Target Stripping**:
  - When training on binary `label`, `attack_cat` is automatically stripped from the feature matrix `X`. If retained, `attack_cat != 'Normal'` trivially reveals the label.
  - When training on multiclass `attack_cat`, binary `label` is automatically stripped from `X`.

#### Predefined Partitioning (Avoiding Train-Test Contamination):
The authors of UNSW-NB15 created predefined training and testing files. Merging these files and performing a naive random split causes **temporal session leakage**, where packets from the exact same TCP flow appear in both partitions. The system natively supports training on `UNSW_NB15_training-set.csv` and evaluating on `UNSW_NB15_testing-set.csv`.

---

### B. Synthetic Demonstration Telemetry (Included Out-of-the-Box)
- **Path**: `data/simulated/simulated_traffic.csv`
- **Purpose**: Zero-setup offline demonstration, automated test suite, and rapid triage experimentation.
- **Scenarios Included**:
  - `normal_web_browsing`: Low connection volume, standard web ports (80, 443, 53), near-zero failure rates.
  - `repeated_connection_failures`: High failure ratios on administrative ports (SSH 22, FTP 21, Telnet 23, RDP 3389).
  - `high_connection_volume`: Volumetric bursts with high byte counts simulating DoS.
  - `unusual_destination_port`: Connection attempts toward watched administrative services.
- **Privacy Guarantee**: All IP addresses use RFC 5737 documentation blocks (`192.0.2.0/24`, `198.51.100.0/24`, `203.0.113.0/24`). Every event is flagged `simulated: True`.

---

## 2. Ingestion & Execution Commands for UNSW-NB15

### A. Inspect Dataset Schema
```powershell
# Inspect training partition
python scripts/inspect_data.py --data-path data/raw/UNSW-NB15/UNSW_NB15_training-set.csv --target-col label

# Inspect attack categories
python scripts/inspect_data.py --data-path data/raw/UNSW-NB15/UNSW_NB15_training-set.csv --target-col attack_cat
```

### B. Train Supervised Models on Predefined Partitions
```powershell
# 1. Binary Classification (Random Forest)
python scripts/train_model.py --train-path data/raw/UNSW-NB15/UNSW_NB15_training-set.csv --test-path data/raw/UNSW-NB15/UNSW_NB15_testing-set.csv --target-col label --model-type random_forest --model-name unsw_nb15_rf_binary.joblib

# 2. Binary Classification (Logistic Regression)
python scripts/train_model.py --train-path data/raw/UNSW-NB15/UNSW_NB15_training-set.csv --test-path data/raw/UNSW-NB15/UNSW_NB15_testing-set.csv --target-col label --model-type logistic_regression --model-name unsw_nb15_lr_binary.joblib

# 3. Multiclass Attack Categorization (Random Forest)
python scripts/train_model.py --train-path data/raw/UNSW-NB15/UNSW_NB15_training-set.csv --test-path data/raw/UNSW-NB15/UNSW_NB15_testing-set.csv --target-col attack_cat --model-type random_forest --model-name unsw_nb15_rf_multiclass.joblib
```

### C. Evaluate on Independent Official Test Partition
```powershell
# Evaluate Binary Random Forest
python scripts/evaluate_model.py --model-path models/unsw_nb15_rf_binary.joblib --data-path data/raw/UNSW-NB15/UNSW_NB15_testing-set.csv --target-col label --is-test-set

# Evaluate Multiclass Random Forest
python scripts/evaluate_model.py --model-path models/unsw_nb15_rf_multiclass.joblib --data-path data/raw/UNSW-NB15/UNSW_NB15_testing-set.csv --target-col attack_cat --is-test-set
```

---

## 3. Academic Citation
When using UNSW-NB15 in academic projects, portfolios, or publications, cite the primary reference:
```bibtex
@inproceedings{moustafa2015unsw,
  title={UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set)},
  author={Moustafa, Nour and Slay, Jill},
  booktitle={2015 military communications and information systems conference (MilCIS)},
  pages={1--6},
  year={2015},
  organization={IEEE}
}
```

---

## 4. Operational Boundaries & Ethical Disclaimer
- High benchmark scores on UNSW-NB15 do not guarantee production effectiveness against zero-day exploits in live corporate infrastructure.
- IXIA PerfectStorm generates synthetic attack traffic patterns that differ from real-world advanced persistent threat (APT) campaigns.
- Benchmarks demonstrate algorithmic capability; human analyst review remains essential in operational triage.
