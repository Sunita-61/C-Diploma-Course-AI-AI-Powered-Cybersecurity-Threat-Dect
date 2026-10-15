# Data Directory Guide

This folder holds datasets for the AI-Powered Cybersecurity Threat Detection System.

## Directory Layout:
- `raw/`: Stores pristine, original external datasets (e.g. raw CSV dumps of CIC-IDS2017, UNSW-NB15, KDD Cup 1999). Never modify files here directly. Ignored by git.
- `processed/`: Stores cleaned, validated, and preprocessed datasets produced by `scripts/prepare_data.py`. Ignored by git.
- `simulated/`: Stores generated synthetic telemetry produced by `src/threat_detection/simulation.py` for testing and offline demos.

## External Datasets
Refer to [`docs/dataset_documentation.md`](../docs/dataset_documentation.md) for details on downloading, placing, and preparing external benchmark datasets.
