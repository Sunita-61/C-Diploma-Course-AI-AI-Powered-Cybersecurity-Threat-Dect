# Operational Limitations & Technical Assumptions

Honest engineering requires understanding where defensive machine learning systems encounter failure modes. This document transparently outlines the limitations of this project.

---

## 1. False Positives and SOC Fatigue
- **Heuristic Sensitivity**: Setting a low connection failure threshold (e.g., 3 failures) will inevitably flag benign user errors (e.g., forgotten passwords, typoed credentials, or momentary Wi-Fi disconnects).
- **Impact**: In a Security Operations Center (SOC), a high false positive rate creates alert fatigue, causing analysts to overlook critical genuine incidents.

## 2. False Negatives & Advanced Persistent Threats (APTs)
- **Low-and-Slow Attacks**: Skilled attackers rarely trigger volumetric rules. A stealthy scan with 1 probe every 30 minutes avoids volume and burst heuristics.
- **Novel Zero-Day Payloads**: Supervised models only classify behavior resembling their training distribution. Novel attack methods with benign-looking flow characteristics will not be caught by supervised models alone.

## 3. Anomaly Detection != Malicious Activity
- **Unsupervised Outliers**: Isolation Forest identifies multivariate statistical deviations in continuous parameter space.
- **Benign Anomalies**: High-bandwidth video conferences, automated system backups, software updates, or scheduled batch jobs are unusual compared to ordinary web browsing, but they are completely benign.
- **Key Principle**: The system labels outliers as **"ANOMALY"**, never as **"CONFIRMED BREACH"**.

## 4. Synthetic Telemetry Boundaries
- The included synthetic generator is designed for educational triage and testing without requiring live packet capture.
- Real-world enterprise traffic features packet jitter, retransmissions, NAT translation artifacts, and protocol encapsulation that synthetic generators approximate but do not fully duplicate.

## 5. Dataset Shift & Time-Based Generalization
- Random train/test splits can produce overly optimistic accuracy scores when flow records from the same connection burst exist in both sets.
- In production, time-based splits (training on week 1, evaluating on week 2) provide a more rigorous test of temporal generalization.
