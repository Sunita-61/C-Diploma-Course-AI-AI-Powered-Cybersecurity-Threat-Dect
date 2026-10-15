"""Main command-line entry point for AI-Powered Cybersecurity Threat Detection System.

Supports offline pipeline execution, synthetic telemetry generation, model training,
evaluation, alert generation, and dashboard launching.
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.threat_detection.alerts import export_alerts_to_csv
from src.threat_detection.config import (
    ALERTS_DIR,
    DEFAULT_RANDOM_SEED,
    DEFAULT_SIMULATION_COUNT,
    FIGURES_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    SIMULATED_DATA_DIR,
)
from src.threat_detection.data_loader import load_dataset
from src.threat_detection.prediction import run_unified_detection
from src.threat_detection.simulation import generate_synthetic_events, save_simulated_events
from src.threat_detection.utils import ensure_directory, logger


def execute_full_demonstration_pipeline(
    count: int = DEFAULT_SIMULATION_COUNT,
    seed: int = DEFAULT_RANDOM_SEED,
) -> None:
    """Runs the complete offline demonstration workflow:
    1. Generate synthetic network activity.
    2. Save telemetry.
    3. Train Random Forest & Isolation Forest pipelines.
    4. Run unified detection (Rules + ML + Anomaly).
    5. Compile and export structured Security Alerts to CSV.
    """
    print("\n" + "=" * 75)
    print(" AI-POWERED CYBERSECURITY THREAT DETECTION: OFFLINE DEMO PIPELINE")
    print("=" * 75 + "\n")

    # Step 1: Generate synthetic events
    print(f"[*] Step 1/5: Generating {count} reproducible synthetic network events (Seed={seed})...")
    df_sim = generate_synthetic_events(count=count, seed=seed)
    sim_path = save_simulated_events(df_sim)
    print(f"    [+] Saved synthetic telemetry to: {sim_path}")

    # Step 2: Train Supervised & Anomaly Models
    print("\n[*] Step 2/5: Training Random Forest classifier and Isolation Forest detector...")
    from scripts.train_model import train_models_cli
    train_models_cli(
        data_path=str(sim_path),
        target_col="label",
        model_type="random_forest",
        train_anomaly=True,
    )

    # Step 3: Evaluate Model Performance
    print("\n[*] Step 3/5: Evaluating Random Forest model on holdout test partition...")
    from scripts.evaluate_model import evaluate_model_cli
    rf_model_path = MODELS_DIR / "random_forest.joblib"
    evaluate_model_cli(
        model_path=str(rf_model_path),
        data_path=str(sim_path),
        target_col="label",
    )

    # Step 4: Run Unified Threat Detection
    print("\n[*] Step 4/5: Running Unified Detection Engine (Rules + ML + Anomaly)...")
    from src.threat_detection.anomaly_detector import load_anomaly_pipeline
    from src.threat_detection.model import load_model_pipeline

    rf_pipe, rf_meta = load_model_pipeline(rf_model_path)
    iso_pipe, iso_meta = load_anomaly_pipeline(MODELS_DIR / "isolation_forest.joblib")

    enriched_df, alerts = run_unified_detection(
        df=df_sim,
        supervised_pipeline=rf_pipe,
        supervised_metadata=rf_meta,
        anomaly_pipeline=iso_pipe,
        anomaly_metadata=iso_meta,
        enable_rules=True,
    )
    print(f"    [+] Analyzed {len(enriched_df)} events.")
    print(f"    [+] Generated {len(alerts)} deduplicated Security Alerts.")

    # Step 5: Export Alerts to CSV
    print("\n[*] Step 5/5: Exporting structured security alerts to CSV...")
    exported_csv = export_alerts_to_csv(alerts)
    print(f"    [+] Alerts exported to: {exported_csv}")

    print("\n" + "=" * 75)
    print(" DEMO PIPELINE COMPLETED SUCCESSFULLY")
    print(" Next Step: Launch the interactive web dashboard with:")
    print("   streamlit run app/dashboard.py")
    print("=" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="AI-Powered Cybersecurity Threat Detection System CLI",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: run-demo
    subparsers.add_parser(
        "run-demo",
        help="Run the complete end-to-end demonstration workflow offline.",
    )

    # Command: simulate
    sim_parser = subparsers.add_parser("simulate", help="Generate synthetic network events.")
    sim_parser.add_argument("--count", type=int, default=DEFAULT_SIMULATION_COUNT, help="Number of events.")
    sim_parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED, help="Random seed.")

    # Command: dashboard
    subparsers.add_parser("dashboard", help="Launch the Streamlit interactive dashboard.")

    args = parser.parse_args()

    if args.command == "run-demo" or args.command is None:
        if args.command is None:
            print("[INFO] No subcommand provided. Running default offline demonstration pipeline...\n")
        execute_full_demonstration_pipeline()

    elif args.command == "simulate":
        print(f"[*] Generating {args.count} synthetic events...")
        df_sim = generate_synthetic_events(count=args.count, seed=args.seed)
        out = save_simulated_events(df_sim)
        print(f"[+] Saved to {out}")

    elif args.command == "dashboard":
        print("[*] Launching Streamlit dashboard...")
        subprocess.run(["streamlit", "run", "app/dashboard.py"])


if __name__ == "__main__":
    main()
