"""Streamlit Web Dashboard for AI-Powered Cybersecurity Threat Detection System.

Provides an interactive analyst portal for inspecting network telemetry,
configuring heuristic thresholds, viewing ML classifications and anomaly indicators,
evaluating the UNSW-NB15 benchmark or synthetic demonstration, and exporting structured security alerts.
"""

import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="AI-Powered Threat Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.threat_detection.alerts import SecurityAlert, compile_alerts, export_alerts_to_csv
from src.threat_detection.anomaly_detector import load_anomaly_pipeline
from src.threat_detection.config import (
    DEFAULT_RANDOM_SEED,
    DEFAULT_SIMULATION_COUNT,
    FAILED_CONNECTIONS_THRESHOLD,
    FIGURES_DIR,
    HIGH_VOLUME_THRESHOLD,
    MODELS_DIR,
    REPORTS_DIR,
    UNSW_NB15_DIR,
    UNSW_NB15_TEST_FILE,
    UNSW_NB15_TRAIN_FILE,
    WATCHLIST_PORTS,
)
from src.threat_detection.data_loader import get_dataset_summary, load_dataset
from src.threat_detection.model import load_model_pipeline
from src.threat_detection.prediction import run_unified_detection
from src.threat_detection.simulation import generate_synthetic_events
from src.threat_detection.utils import DataValidationError, logger


# Custom Styling for Security Analyst Aesthetic
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .demo-banner {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        color: #92400E;
        padding: 0.8rem 1.2rem;
        border-radius: 4px;
        margin-bottom: 1.2rem;
        font-weight: 500;
    }
    .benchmark-banner {
        background-color: #EFF6FF;
        border-left: 5px solid #3B82F6;
        color: #1E40AF;
        padding: 0.8rem 1.2rem;
        border-radius: 4px;
        margin-bottom: 1.2rem;
        font-weight: 500;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_available_models() -> List[str]:
    """Finds all trained .joblib models available in the models/ directory."""
    if not MODELS_DIR.exists():
        return []
    models = [p.name for p in MODELS_DIR.glob("*.joblib") if "isolation_forest" not in p.name]
    return sorted(models)


def load_cached_models(selected_model_file: Optional[str] = None) -> Tuple[Optional[Any], Optional[Dict[str, Any]], Optional[Any], Optional[Dict[str, Any]]]:
    """Attempts to load pre-trained models from the models directory."""
    rf_pipeline, rf_metadata = None, None
    iso_pipeline, iso_metadata = None, None

    if selected_model_file:
        model_path = MODELS_DIR / selected_model_file
        if model_path.exists():
            try:
                rf_pipeline, rf_metadata = load_model_pipeline(model_path)
            except Exception as exc:
                logger.error(f"Failed to load model {model_path}: {exc}")

    iso_path = MODELS_DIR / "isolation_forest.joblib"
    if iso_path.exists():
        try:
            iso_pipeline, iso_metadata = load_anomaly_pipeline(iso_path)
        except Exception:
            iso_pipeline, iso_metadata = None, None

    return rf_pipeline, rf_metadata, iso_pipeline, iso_metadata


def render_header(mode: str):
    """Renders the top application header and disclaimer banner."""
    st.markdown('<div class="main-title">🛡️ AI-Powered Cybersecurity Threat Detection System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Defensive network activity analysis, behavioral heuristics, and suspicious-event detection.</div>',
        unsafe_allow_html=True,
    )

    if mode == "Synthetic Demonstration Mode":
        st.markdown(
            '<div class="demo-banner">⚠️ DEMONSTRATION MODE: The displayed telemetry events are synthetic '
            'and do not represent live network traffic. All detections are educational heuristics and statistical flags, '
            'not confirmed security breaches.</div>',
            unsafe_allow_html=True,
        )
    elif mode == "UNSW-NB15 Benchmark Mode":
        st.markdown(
            '<div class="benchmark-banner">🔬 UNSW-NB15 BENCHMARK MODE: Evaluating real captured network flow records '
            'from the Australian Centre for Cyber Security (ACCS) public research dataset.</div>',
            unsafe_allow_html=True,
        )


def main():
    available_model_files = get_available_models()

    # --- SIDEBAR CONTROLS ---
    st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.sidebar.title("Analyst Controls")

    data_source_mode = st.sidebar.radio(
        "Data Source Mode",
        ["Synthetic Demonstration Mode", "UNSW-NB15 Benchmark Mode", "Upload Network CSV"],
        help="Select whether to use reproducible simulated telemetry, official UNSW-NB15 data, or upload your own CSV records.",
    )

    # Model selector
    st.sidebar.subheader("Supervised Model Selection")
    if available_model_files:
        chosen_model_name = st.sidebar.selectbox(
            "Active Classifier",
            available_model_files,
            index=0,
            help="Select which saved model pipeline to use for threat inference.",
        )
    else:
        chosen_model_name = None
        st.sidebar.caption("No trained supervised models found in `models/`.")

    rf_pipe, rf_meta, iso_pipe, iso_meta = load_cached_models(chosen_model_name)

    df_active: Optional[pd.DataFrame] = None
    dataset_label: str = "Synthetic Traffic"

    if data_source_mode == "Synthetic Demonstration Mode":
        dataset_label = "Synthetic Demonstration Telemetry"
        st.sidebar.subheader("Simulation Parameters")
        sim_count = st.sidebar.slider("Number of Events", min_value=50, max_value=1000, value=250, step=50)
        sim_seed = st.sidebar.number_input("Random Seed (Reproducibility)", min_value=1, max_value=999999, value=DEFAULT_RANDOM_SEED)
        anomaly_ratio = st.sidebar.slider("Suspicious Scenario Ratio", min_value=0.05, max_value=0.50, value=0.20, step=0.05)

        with st.spinner("Generating reproducible synthetic telemetry..."):
            df_active = generate_synthetic_events(count=sim_count, seed=int(sim_seed), anomaly_ratio=anomaly_ratio)

    elif data_source_mode == "UNSW-NB15 Benchmark Mode":
        dataset_label = "UNSW-NB15 Research Dataset"
        st.sidebar.subheader("UNSW-NB15 Partition")

        train_exists = UNSW_NB15_TRAIN_FILE.exists()
        test_exists = UNSW_NB15_TEST_FILE.exists()

        if not (train_exists or test_exists):
            render_header(data_source_mode)
            st.warning(
                "### 📁 UNSW-NB15 Dataset Files Not Detected\n\n"
                f"The official dataset partition files were not found in:\n`{UNSW_NB15_DIR}`\n\n"
                "**How to configure:**\n"
                "1. Download the official CSV files from [UNSW Canberra Cyber](https://research.unsw.edu.au/projects/unsw-nb15-dataset).\n"
                "2. Place `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv` inside `data/raw/UNSW-NB15/`.\n"
                "3. In the meantime, switch to **Synthetic Demonstration Mode** or **Upload Network CSV** above to test the application."
            )
            return

        file_options = []
        if test_exists:
            file_options.append("Official Testing Partition (UNSW_NB15_testing-set.csv)")
        if train_exists:
            file_options.append("Official Training Partition (UNSW_NB15_training-set.csv)")

        selected_partition = st.sidebar.selectbox("Select Partition", file_options)
        target_file = UNSW_NB15_TEST_FILE if "testing" in selected_partition else UNSW_NB15_TRAIN_FILE

        sample_limit = st.sidebar.slider(
            "Rows to Load for Interactive Analysis",
            min_value=500,
            max_value=10000,
            value=2000,
            step=500,
            help="Limits rows for responsive web visualization. Full datasets are evaluated via CLI scripts.",
        )

        with st.spinner(f"Loading {sample_limit:,} records from {target_file.name}..."):
            try:
                df_active = load_dataset(target_file, nrows=sample_limit)
                st.sidebar.success(f"Loaded {len(df_active):,} records.")
            except Exception as exc:
                st.error(f"Error loading UNSW-NB15 file: {exc}")
                return

    else:  # Upload Network CSV
        dataset_label = "Uploaded CSV Telemetry"
        st.sidebar.subheader("Dataset Upload")
        uploaded_file = st.sidebar.file_uploader(
            "Upload Network Telemetry (CSV)",
            type=["csv"],
            help="Upload a CSV dataset containing network telemetry columns.",
        )

        if uploaded_file is not None:
            try:
                df_active = pd.read_csv(uploaded_file)
                if df_active.empty:
                    st.error("Uploaded CSV file is empty. Please upload a dataset with data rows.")
                    df_active = None
                else:
                    st.sidebar.success(f"Loaded {len(df_active):,} records.")
            except Exception as exc:
                st.error(f"Failed to parse uploaded CSV: {exc}")
                df_active = None
        else:
            st.info("👈 Please upload a network CSV dataset from the sidebar, or switch to 'Synthetic Demonstration Mode' or 'UNSW-NB15 Benchmark Mode'.")
            render_header(data_source_mode)
            return

    # Detection Threshold Configuration
    st.sidebar.subheader("Heuristic Thresholds")
    failed_threshold = st.sidebar.number_input(
        "Failed Connections Threshold",
        min_value=1,
        max_value=50,
        value=FAILED_CONNECTIONS_THRESHOLD,
        help="Flag repeated authentication or connection failures.",
    )
    volume_threshold = st.sidebar.number_input(
        "High Connection Volume Threshold",
        min_value=10,
        max_value=2000,
        value=HIGH_VOLUME_THRESHOLD,
        help="Flag sudden spikes in total connection counts.",
    )

    # Filter Options
    st.sidebar.subheader("Filter Findings")
    selected_severity = st.sidebar.selectbox("Filter by Severity", ["ALL", "LOW", "MEDIUM", "HIGH", "CRITICAL"])
    selected_method = st.sidebar.selectbox(
        "Filter by Detection Method",
        ["ALL", "RULE_BASED", "SUPERVISED_RF", "SUPERVISED_LR", "ANOMALY_ISOLATION_FOREST"],
    )

    render_header(data_source_mode)

    if df_active is None:
        return

    # Execute Unified Detection Pipeline
    with st.spinner("Executing defense analysis and threat correlation..."):
        enriched_df, alerts = run_unified_detection(
            df=df_active,
            supervised_pipeline=rf_pipe,
            supervised_metadata=rf_meta,
            anomaly_pipeline=iso_pipe,
            anomaly_metadata=iso_meta,
            enable_rules=True,
            failed_threshold=int(failed_threshold),
            volume_threshold=int(volume_threshold),
        )

    # Convert alerts to DataFrame for filtering and display
    alerts_df = pd.DataFrame([a.to_dict() for a in alerts]) if alerts else pd.DataFrame(columns=[
        "alert_id", "event_id", "timestamp", "source_ip", "destination_ip", "severity",
        "category", "detection_method", "rule_id", "description", "status", "simulated"
    ])

    # Filter alerts according to sidebar controls
    filtered_alerts_df = alerts_df.copy()
    if selected_severity != "ALL" and not filtered_alerts_df.empty:
        filtered_alerts_df = filtered_alerts_df[filtered_alerts_df["severity"] == selected_severity]
    if selected_method != "ALL" and not filtered_alerts_df.empty:
        filtered_alerts_df = filtered_alerts_df[filtered_alerts_df["detection_method"] == selected_method]

    # --- TOP KPI METRICS ---
    total_events = len(enriched_df)
    events_with_rules = int((enriched_df["rule_findings_count"] > 0).sum())
    high_sev_count = int((alerts_df["severity"].isin(["HIGH", "CRITICAL"])).sum()) if not alerts_df.empty else 0
    anom_count = int((enriched_df["anomaly_flag"] == 1).sum()) if "anomaly_flag" in enriched_df.columns else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Events Analyzed", f"{total_events:,}")
    with col2:
        st.metric("Events with Rule Findings", f"{events_with_rules:,}")
    with col3:
        st.metric("High/Critical Alerts", f"{high_sev_count:,}")
    with col4:
        st.metric("Multivariate Anomalies", f"{anom_count:,}")

    st.caption(f"📊 Active Dataset: **{dataset_label}** | Total Generated Alerts: **{len(alerts):,}** (Heuristic & statistical flags, NOT confirmed security breaches).")
    st.markdown("---")

    # --- VISUALIZATION TABS ---
    tab_analytics, tab_alerts, tab_events, tab_model_info = st.tabs(
        ["📊 Visual Analytics", "🚨 Security Alerts", "📋 Event Telemetry", "🤖 Model Card & Evaluation"]
    )

    with tab_analytics:
        st.subheader("Telemetry & Detection Insights")
        vcol1, vcol2 = st.columns(2)

        with vcol1:
            st.markdown("**Alert Severity Breakdown**")
            if not alerts_df.empty and "severity" in alerts_df.columns:
                sev_counts = alerts_df["severity"].value_counts()
                fig, ax = plt.subplots(figsize=(6, 4))
                colors = {"LOW": "#3B82F6", "MEDIUM": "#F59E0B", "HIGH": "#EF4444", "CRITICAL": "#7F1D1D"}
                palette = [colors.get(s, "#6B7280") for s in sev_counts.index]
                sns.barplot(x=sev_counts.index, y=sev_counts.values, hue=sev_counts.index, palette=palette, legend=False, ax=ax)
                ax.set_ylabel("Number of Alerts")
                ax.set_xlabel("Severity Level")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.info("No alert records available to plot severity distribution.")

        with vcol2:
            st.markdown("**Detection Method Breakdown**")
            if not alerts_df.empty and "detection_method" in alerts_df.columns:
                meth_counts = alerts_df["detection_method"].value_counts()
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(x=meth_counts.index, y=meth_counts.values, hue=meth_counts.index, palette="crest", legend=False, ax=ax)
                ax.set_ylabel("Alert Count")
                ax.set_xlabel("Detection Engine")
                plt.xticks(rotation=15)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.info("No alert records available to plot method distribution.")

        st.markdown("**Detection Category Distribution**")
        if not alerts_df.empty and "category" in alerts_df.columns:
            cat_counts = alerts_df["category"].value_counts()
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.barplot(y=cat_counts.index, x=cat_counts.values, hue=cat_counts.index, palette="Blues_r", legend=False, ax=ax)
            ax.set_xlabel("Occurrences")
            ax.set_ylabel("Category")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with tab_alerts:
        st.subheader("Correlated Security Alerts")
        st.write(f"Showing **{len(filtered_alerts_df)}** alerts matching current filters.")

        if not filtered_alerts_df.empty:
            st.dataframe(
                filtered_alerts_df[[
                    "alert_id", "event_id", "timestamp", "severity", "category",
                    "detection_method", "source_ip", "destination_ip", "description", "status"
                ]],
                use_container_width=True,
            )

            # CSV Download Button
            csv_buffer = io.StringIO()
            filtered_alerts_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Export Filtered Alerts (CSV)",
                data=csv_buffer.getvalue(),
                file_name="security_alerts_export.csv",
                mime="text/csv",
            )
        else:
            st.success("No alerts found matching the selected filters.")

    with tab_events:
        st.subheader("Raw & Engineered Telemetry Events")
        st.dataframe(enriched_df.head(500), use_container_width=True)

    with tab_model_info:
        st.subheader("Machine Learning Model Governance & Actual Evaluation")
        mcol1, mcol2 = st.columns(2)

        with mcol1:
            st.markdown("### Supervised Model")
            if rf_pipe is not None and rf_meta is not None:
                st.success(f"Active Model: **{chosen_model_name}**")
                st.json({
                    "model_type": rf_meta.get("model_type"),
                    "trained_at": rf_meta.get("created_at"),
                    "split_strategy": rf_meta.get("split_strategy", "stratified_holdout"),
                    "train_samples": rf_meta.get("train_samples"),
                    "test_samples": rf_meta.get("test_samples"),
                    "classes": rf_meta.get("classes"),
                    "features_count": len(rf_meta.get("feature_schema", {}).get("all_features", [])),
                })
            else:
                st.warning(
                    "No trained supervised model loaded.\n\n"
                    "Train a model using `python scripts/train_model.py`."
                )

        with mcol2:
            st.markdown("### Anomaly Detector (Isolation Forest)")
            if iso_pipe is not None and iso_meta is not None:
                st.success("Active Detector: **Isolation Forest Pipeline**")
                st.json({
                    "trained_at": iso_meta.get("created_at"),
                    "contamination": iso_meta.get("contamination"),
                    "baseline_samples": iso_meta.get("train_samples"),
                    "features": iso_meta.get("feature_columns"),
                })
            else:
                st.warning("No Isolation Forest model found in `models/`.")

        # Show actual saved evaluation report and confusion matrix figure if present
        if chosen_model_name:
            stem = Path(chosen_model_name).stem
            eval_report_file = REPORTS_DIR / f"{stem}_evaluation_report.txt"
            eval_report_json = eval_report_file.with_suffix(".json")
            eval_fig_file = FIGURES_DIR / f"{stem}_confusion_matrix.png"

            if eval_report_json.exists():
                try:
                    with open(eval_report_json, "r", encoding="utf-8") as jf:
                        report_data = json.load(jf)
                    metrics = report_data.get("metrics", {})
                    is_bin = report_data.get("target_type") == "binary"

                    st.markdown(f"### Offline Benchmark Metrics: `{chosen_model_name}`")
                    if is_bin:
                        kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5, kpi_c6 = st.columns(6)
                        kpi_c1.metric("Attack Recall (Det. Rate)", f"{metrics.get('attack_recall', metrics.get('recall', 0)):.2%}")
                        kpi_c2.metric("Attack Precision (pos=1)", f"{metrics.get('attack_precision', metrics.get('precision', 0)):.2%}")
                        kpi_c3.metric("Attack F1 (pos=1)", f"{metrics.get('attack_f1', metrics.get('f1_score', 0)):.4f}")
                        kpi_c4.metric("Weighted F1", f"{metrics.get('weighted_f1', metrics.get('f1_score', 0)):.4f}")
                        kpi_c5.metric("Macro F1", f"{metrics.get('macro_f1', 0):.4f}")
                        kpi_c6.metric("PR-AUC (Avg Precision)", f"{metrics.get('pr_auc', 0):.4f}" if metrics.get('pr_auc') else "N/A")
                    else:
                        kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
                        kpi_c1.metric("Overall Accuracy", f"{metrics.get('accuracy', 0):.2%}")
                        kpi_c2.metric("Weighted F1-Score", f"{metrics.get('weighted_f1', metrics.get('f1_score', 0)):.4f}")
                        kpi_c3.metric("Macro F1-Score", f"{metrics.get('macro_f1', 0):.4f}")
                        kpi_c4.metric("ROC-AUC (OvR Weighted)", f"{metrics.get('roc_auc', 0):.4f}" if metrics.get('roc_auc') else "N/A")
                except Exception as exc:
                    logger.warning(f"Could not parse structured evaluation json: {exc}")

            if "unsw_nb15" in chosen_model_name.lower():
                st.warning(
                    "**🛡️ Defensive Operational Disclosures (UNSW-NB15 Benchmark):**\n\n"
                    "- **False-Positive Burden**: In binary Random Forest, 7,734 benign test flows were flagged as threats (20.9% false alarm rate on benign traffic). In production SOCs, secondary risk thresholding is required to mitigate alert fatigue.\n"
                    "- **Multiclass Class Imbalance**: Supervised trees exhibit low recall on `DoS` (17.05%) and `Analysis` (1.92%) flows due to statistical packet overlap with benign web browsing and fuzzers. These threat families require multi-layered heuristic rule detection."
                )

            if eval_fig_file.exists():
                st.markdown(f"### Saved Confusion Matrix: `{chosen_model_name}`")
                st.image(str(eval_fig_file), caption=f"Confusion Matrix ({stem})", use_container_width=True)

            if eval_report_file.exists():
                st.markdown(f"### Saved Evaluation Report for `{chosen_model_name}`")
                with open(eval_report_file, "r", encoding="utf-8") as f:
                    st.text(f.read())
            else:
                st.info(f"No saved evaluation report found at `outputs/reports/{stem}_evaluation_report.txt`.")


if __name__ == "__main__":
    main()
