"""CLI utility to clean, engineer features, and prepare datasets for modeling."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.threat_detection.config import PROCESSED_DATA_DIR
from src.threat_detection.data_loader import load_dataset
from src.threat_detection.features import engineer_domain_features
from src.threat_detection.preprocessing import clean_dataframe
from src.threat_detection.utils import ensure_directory, logger


def prepare_dataset_cli(
    input_path: str,
    output_path: str = None,
    drop_duplicates: bool = True,
) -> None:
    """Cleans raw CSV and saves engineered version to processed directory."""
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"[ERROR] Input file does not exist: {input_file}")
        sys.exit(1)

    if output_path is None:
        ensure_directory(PROCESSED_DATA_DIR)
        output_file = PROCESSED_DATA_DIR / f"prepared_{input_file.name}"
    else:
        output_file = Path(output_path)
        ensure_directory(output_file.parent)

    print(f"\n[*] Loading raw dataset from: {input_file}")
    df_raw = load_dataset(input_file)
    print(f"    Raw records: {len(df_raw):,}")

    print("[*] Performing data cleaning (deduplication, infinite value handling)...")
    df_clean = clean_dataframe(df_raw, drop_duplicates=drop_duplicates, replace_inf=True)
    print(f"    Cleaned records: {len(df_clean):,}")

    print("[*] Engineering domain cybersecurity features (ratios, port flags)...")
    df_prepared = engineer_domain_features(df_clean)
    print(f"    Prepared records: {len(df_prepared):,}, Columns: {len(df_prepared.columns)}")

    df_prepared.to_csv(output_file, index=False)
    print(f"[+] Successfully wrote prepared dataset to:\n    {output_file}\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare and clean network dataset.")
    parser.add_argument("--input-path", type=str, required=True, help="Path to input raw CSV.")
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional destination path for prepared CSV.",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable deduplication if duplicate records are intended.",
    )
    args = parser.parse_args()
    prepare_dataset_cli(
        input_path=args.input_path,
        output_path=args.output_path,
        drop_duplicates=not args.no_dedup,
    )


if __name__ == "__main__":
    main()
