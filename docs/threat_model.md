# Threat Model & Defensive Scope

## 1. Application Purpose & Trust Boundaries
This system is an **offline educational analytics platform** intended to assist security analysts in triaging high-volume telemetry.

```
+-----------------------------------------------------------+
| UNTRUSTED EXTERNAL ENVIRONMENT                           |
| - External attackers / botnets / automated scanners       |
| - Malicious payload traffic                               |
+-----------------------------+-----------------------------+
                              | Network Sensors / PCAP Logs
                              v
+-----------------------------------------------------------+
| TRUST BOUNDARY: Local Analytics System                    |
| +-------------------------------------------------------+ |
| | Input CSV Telemetry                                   | |
| +---------------------------+---------------------------+ |
|                             v                             |
| | Preprocessing & Sanitization (inf, NaN, Type Checks)  | |
| +---------------------------+---------------------------+ |
|                             v                             |
| | Threat Detection Engine (Rules, Supervised, Anomaly)  | |
| +---------------------------+---------------------------+ |
|                             v                             |
| | Read-Only Alert Presentation & CSV Export             | |
| +-------------------------------------------------------+ |
+-----------------------------------------------------------+
```

---

## 2. Potential Attack Vectors Against the Detection System

1. **Adversarial Perturbation / Evasion**:
   - Attackers can modify packet inter-arrival times or pad packet sizes to mimic legitimate web browsing.
   - Mitigation: Multi-layered detection combining deterministic rules with unsupervised outlier detection.
2. **Denial of Service via Telemetry Exhaustion (Algorithmic Complexity)**:
   - Flooding logs with millions of malformed entries to crash the analytics dashboard.
   - Mitigation: Strict size checks on uploaded CSV files and chunked pandas ingestion.
3. **Data Poisoning (if retraining pipelines on uncurated logs)**:
   - Attackers injecting malicious flows labeled as normal to distort the model decision boundary.
   - Mitigation: Preprocessing fits on pristine baselines; alerts clearly distinguish unverified anomalies from confirmed attacks.

---

## 3. Defensive Guarantees
- The system is completely **passive**: it does not send packets, execute firewall rules, or perform active countermeasures.
- It operates **strictly offline**: no telemetry or dataset contents are transmitted to external cloud services or third parties.
