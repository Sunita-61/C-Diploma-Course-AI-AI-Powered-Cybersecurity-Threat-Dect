"""CLI utility to inspect network datasets and print exploratory statistics."""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add project root to path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.threat_detection.data_loader import get_dataset_summary, load_dataset
from src.threat_detection.preprocessing import identify_target_column
from src.threat_detection.utils import logger


def inspect_dataset_cli(data_path: str, target_col: str = None, nrows: int = None) -> None:
    """Inspects a dataset file and displays comprehensive terminal statistics."""
    print(f"\n{'='*70}")
    print(f" DATASET INSPECTION REPORT: {Path(data_path).name}")
    print(f"{'='*70}\n")

    try:
        df = load_dataset(data_path, nrows=nrows)

    except Exception as exc:
        print(f"[ERROR] Could not load dataset: {exc}")
        sys.exit(1)

    summary = get_dataset_summary(df)

    print(f"[*] Basic Dimensions:")
    print(f"    - Total Rows:       {summary['total_rows']:,}")
    print(f"    - Total Columns:    {summary['total_columns']}")
    print(f"    - Duplicate Rows:   {summary['duplicate_rows']:,}")

    # Check infinite values
    num_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = {col: int(np.isinf(df[col]).sum()) for col in num_cols if np.isinf(df[col]).sum() > 0}

    print(f"\n[*] Column Types & Missing Values:")
    print(f"    {'Column Name':<30} {'Type':<12} {'Missing':<10} {'% Missing':<10}")
    print(f"    {'-'*30} {'-'*12} {'-'*10} {'-'*10}")
    for col in summary["columns"]:
        col_type = summary["data_types"][col]
        missing = summary["missing_counts"][col]
        pct = summary["missing_percentages"][col]
        print(f"    {col:<30} {col_type:<12} {missing:<10} {pct:<10.2f}%")

    if inf_counts:
        print(f"\n[!] Infinite Values Detected ({len(inf_counts)} columns):")
        for col, cnt in inf_counts.items():
            print(f"    - {col}: {cnt:,} infinite values")
    else:
        print("\n[*] Infinite Values Check: None detected (0 infinite floats).")


    print(f"\n[*] Feature Breakdown:")
    print(f"    - Numerical Features ({len(summary['numeric_columns'])}): {summary['numeric_columns']}")
    print(f"    - Categorical Features ({len(summary['categorical_columns'])}): {summary['categorical_columns']}")

    target_name = identify_target_column(df, target_col)
    if target_name:
        print(f"\n[*] Target Column Identified: '{target_name}'")
        dist = df[target_name].value_counts(dropna=False)
        print("    Class Distribution:")
        for val, count in dist.items():
            pct = (count / len(df)) * 100
            print(f"      - {val}: {count:,} ({pct:.2f}%)")
    else:
        print("\n[!] No standard target column found. (Pass --target-col to specify manually).")

    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Inspect CSV network telemetry dataset.")
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the CSV dataset file to inspect.",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default=None,
        help="Optional name of the target label column.",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Optional limit on maximum rows to inspect (useful for large datasets).",
    )
    args = parser.parse_args()
    inspect_dataset_cli(args.data_path, args.target_col, nrows=args.nrows)



if __name__ == "__main__":
    main()
