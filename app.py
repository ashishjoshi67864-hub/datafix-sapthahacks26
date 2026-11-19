"""Thin-client Streamlit workbench consuming FastAPI over HTTP with zero duplicated cleaning logic."""

import io
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import httpx
import pandas as pd
import streamlit as st

# ============================================================================
# Page Configuration & Styling
# ============================================================================

st.set_page_config(
    page_title="DataFix — Autonomous Data Quality Workbench",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# Session State Initialization
# ============================================================================

if "raw_csv_bytes" not in st.session_state:
    st.session_state.raw_csv_bytes = None
if "raw_filename" not in st.session_state:
    st.session_state.raw_filename = "sample_dirty.csv"
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None

# ============================================================================
# Constants & Built-in Benchmark Datasets
# ============================================================================

BUILTIN_DATASETS = {
    "sample_dirty.csv": {
        "label": "sample_dirty.csv (24 rows — HTML, Dups, Outliers, Typos, Quarantine)",
        "path": Path("sample_dirty.csv"),
    },
    "sample_data/dirty_ecommerce_orders.csv": {
        "label": "dirty_ecommerce_orders.csv (100 rows — E-Commerce Orders Domain)",
        "path": Path("sample_data/dirty_ecommerce_orders.csv"),
    },
    "sample_data/dirty_hospital_records.csv": {
        "label": "dirty_hospital_records.csv (100 rows — Healthcare Patient Domain)",
        "path": Path("sample_data/dirty_hospital_records.csv"),
    },
    "sample_data/dirty_hr_payroll.csv": {
        "label": "dirty_hr_payroll.csv (100 rows — HR & Payroll Domain)",
        "path": Path("sample_data/dirty_hr_payroll.csv"),
    },
}


# ============================================================================
# Backend Communication Helpers
# ============================================================================

def fetch_health(base_url: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Checks the health of the FastAPI backend and Ollama detective."""
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/health", timeout=3.0)
        if resp.status_code == 200:
            return resp.json(), None
        return None, f"HTTP {resp.status_code}: {resp.text}"
    except Exception as exc:
        return None, str(exc)


# ============================================================================
# Sidebar Configuration & Health Panel
# ============================================================================

st.sidebar.title("🛠️ DataFix Control Panel")

api_url = st.sidebar.text_input(
    "FastAPI Backend URL",
    value=os.getenv("DATAFIX_API_URL", "http://127.0.0.1:8000"),
    help="Address where the main.py FastAPI backend server is running.",
)

st.sidebar.markdown("### 🔌 System & AI Health")
if st.sidebar.button("🔄 Refresh Status", use_container_width=True):
    st.session_state.pop("cached_health", None)

health_info, health_err = fetch_health(api_url)

if health_info:
    st.sidebar.success("✅ **FastAPI Backend:** Online")
    ollama_info = health_info.get("ollama", {})
    if ollama_info.get("reachable"):
        model_name = ollama_info.get("model", "qwen2.5:3b")
        if ollama_info.get("model_available"):
            st.sidebar.success(f"🤖 **Ollama Detective:** Ready (`{model_name}`)")
        else:
            st.sidebar.warning(f"⚠️ **Ollama Active:** Model `{model_name}` not found")
    else:
        st.sidebar.info("💤 **Ollama Detective:** Offline (Deterministic Fallback)")
else:
    st.sidebar.error("❌ **FastAPI Backend:** Offline")
    st.sidebar.caption("Start the backend server in your terminal:")
    st.sidebar.code(
        r".\.venv\Scripts\uvicorn.exe main:app --host 127.0.0.1 --port 8000",
        language="powershell",
    )

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Pipeline Execution Settings")

llm_mode = st.sidebar.selectbox(
    "Execution Mode",
    options=["auto", "deterministic", "ollama"],
    index=0,
    help=(
        "• auto: Uses local Ollama if available; falls back cleanly.\n"
        "• deterministic: Instant local heuristics (<1s, zero LLM).\n"
        "• ollama: Live local 3B LLM detective on CPU (~30-60s)."
    ),
)
if llm_mode == "ollama":
    st.sidebar.caption("⏱️ _Note: Local CPU LLM inference takes ~30–60s on CPU._")

skip_stage2 = st.sidebar.checkbox(
    "Stage 1 Only (Skip Stage 2 AI)",
    value=False,
    help="Skip all Stage 2 semantic AI standardization, FD imputation, and root-cause diagnosis.",
)

st.sidebar.markdown("---")
st.sidebar.caption("100% Local • Neuro-Symbolic • Zero Raw-Row PII Leakage")


# ============================================================================
# Main Header & Title Banner
# ============================================================================

st.title("🛠️ DataFix — Autonomous Data Quality Workbench")
st.markdown(
    "**100% Local Neuro-Symbolic Tabular Data Quality, Diagnosis & Repair Engine**  \n"
    "Combines deterministic profiling, identifier protection, and KS-test drift rollbacks with metadata-only local AI detective repair."
)

# ============================================================================
# Input Panel: Demo Dataset Selector or Upload
# ============================================================================

input_source = st.radio(
    "Select Input Source:",
    ["Load Built-in Demo Dataset", "Upload CSV File"],
    horizontal=True,
)

if input_source == "Load Built-in Demo Dataset":
    selected_key = st.selectbox(
        "Choose a benchmark dataset:",
        options=list(BUILTIN_DATASETS.keys()),
        format_func=lambda k: BUILTIN_DATASETS[k]["label"],
    )
    dataset_info = BUILTIN_DATASETS[selected_key]
    if dataset_info["path"].is_file():
        file_bytes = dataset_info["path"].read_bytes()
        file_name = dataset_info["path"].name
        if st.session_state.raw_filename != file_name or st.session_state.raw_csv_bytes != file_bytes:
            st.session_state.raw_csv_bytes = file_bytes
            st.session_state.raw_filename = file_name
            st.session_state.pipeline_result = None
    else:
        st.warning(f"File '{dataset_info['path']}' not found on disk.")
else:
    uploaded_file = st.file_uploader("Upload a dirty CSV file to repair", type=["csv"])
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name
        if st.session_state.raw_csv_bytes != file_bytes or st.session_state.raw_filename != file_name:
            st.session_state.raw_csv_bytes = file_bytes
            st.session_state.raw_filename = file_name
            st.session_state.pipeline_result = None

# ============================================================================
# Raw Dataset Preview & Execution CTA
# ============================================================================

if st.session_state.raw_csv_bytes:
    try:
        raw_df = pd.read_csv(io.BytesIO(st.session_state.raw_csv_bytes), dtype=str)
        st.markdown(
            f"**Raw Dataset Preview:** `{st.session_state.raw_filename}` "
            f"({len(raw_df)} rows, {len(raw_df.columns)} columns)"
        )
        st.dataframe(raw_df.head(10), use_container_width=True)
    except Exception as exc:
        st.error(f"Error previewing CSV: {exc}")

    col_btn, _ = st.columns([2, 5])
    with col_btn:
        run_btn = st.button("🚀 Run DataFix Pipeline", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("Executing DataFix pipeline (profiling, cleaning, KS-testing, semantic AI)..."):
            try:
                files = {"file": (st.session_state.raw_filename, st.session_state.raw_csv_bytes, "text/csv")}
                params = {
                    "llm_mode": llm_mode,
                    "skip_stage2": "true" if skip_stage2 else "false",
                }
                resp = httpx.post(
                    f"{api_url.rstrip('/')}/clean",
                    files=files,
                    params=params,
                    timeout=180.0,
                )
                if resp.status_code == 200:
                    st.session_state.pipeline_result = resp.json()
                    st.rerun()
                elif resp.status_code in [400, 413]:
                    st.error(f"Validation Error ({resp.status_code}): {resp.json().get('detail', resp.text)}")
                else:
                    st.error(f"Backend Server Error ({resp.status_code}): {resp.json().get('detail', resp.text)}")
            except httpx.ConnectError:
                st.error("Cannot connect to the FastAPI backend service. Ensure it is running:")
                st.code(
                    r".\.venv\Scripts\uvicorn.exe main:app --host 127.0.0.1 --port 8000",
                    language="powershell",
                )
            except httpx.TimeoutException:
                st.error("Pipeline request timed out after 180 seconds. Try switching to 'deterministic' mode for instant results.")
            except Exception as exc:
                st.error(f"Unexpected error communicating with API: {exc}")


# ============================================================================
# Results Dashboard
# ============================================================================

if st.session_state.pipeline_result:
    result = st.session_state.pipeline_result
    summary = result.get("summary", {})
    quality_report = result.get("quality_report", {})
    repairs = result.get("repairs", [])
    unresolved = result.get("unresolved_issues", [])
    root_causes = result.get("root_cause_diagnoses", [])
    warnings = result.get("warnings", [])
    pipeline_status = result.get("status", "completed")

    st.markdown("---")
    st.header("📊 DataFix Quality & Audit Dashboard")

    # Status & Warnings Banner
    stat_col, warn_col = st.columns([1, 4])
    with stat_col:
        if pipeline_status == "completed":
            st.success(f"**Pipeline:** `{pipeline_status}`")
        else:
            st.warning(f"**Pipeline:** `{pipeline_status}`")

    with warn_col:
        if warnings:
            for w in warnings:
                st.warning(f"⚠️ {w}")
        else:
            st.info("Execution completed smoothly with zero warnings.")

    # Executive KPI Metric Row (4 Columns)
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    before_score = quality_report.get("before_health_score", 0.0)
    after_score = quality_report.get("after_health_score", 0.0)
    delta_score = after_score - before_score

    kpi1.metric(
        "Data Health Score",
        f"{after_score:.2f} / 100",
        f"{delta_score:+.2f} pts (from {before_score:.2f})",
    )
    kpi2.metric(
        "Rows Cleaned",
        f"{summary.get('cleaned_rows', 0)}",
        f"-{summary.get('duplicates_removed', 0)} duplicate rows",
    )
    kpi3.metric(
        "Total Repairs",
        f"{summary.get('total_repairs', 0)}",
        f"S1: {summary.get('stage1_repairs', 0)} | S2: {summary.get('stage2_repairs', 0)}",
    )
    kpi4.metric(
        "Quarantined (Untouched)",
        f"{summary.get('unresolved_count', 0)}",
        "Protected from Hallucination",
    )

    # Guardrails & Cryptographic Provenance Expander
    with st.expander("🛡️ Statistical Drift Guard & Cryptographic Provenance", expanded=False):
        st.markdown("**Protected Identifier Columns (Locked from Modification & Imputation):**")
        prot_cols = summary.get("protected_identifier_columns", [])
        if prot_cols:
            st.write(" ".join([f"`{c}`" for c in prot_cols]))
        else:
            st.write("_None detected_")

        st.markdown("**Kolmogorov-Smirnov (KS) Drift Verification:**")
        ks_checks = quality_report.get("ks_drift_checks", [])
        if ks_checks:
            st.dataframe(pd.DataFrame(ks_checks), use_container_width=True, hide_index=True)
        else:
            st.caption("No numeric KNN imputations required KS testing.")

        st.markdown("**Cryptographic Dataset Provenance (SHA-256):**")
        st.code(
            f"Input  SHA-256: {quality_report.get('input_sha256', 'N/A')}\n"
            f"Output SHA-256: {quality_report.get('output_sha256', 'N/A')}",
            language="text",
        )

    # 4 Interactive Workbench Tabs
    tab_diag, tab_repairs, tab_quarantine, tab_export = st.tabs([
        "🔍 Root-Cause Diagnosis",
        "🛠️ Repair Audit Trail",
        "🛡️ Quarantine / Unresolved",
        "📊 Before vs. After & Exports",
    ])

    # Tab 1: Root-Cause Diagnosis
    with tab_diag:
        st.markdown("### 🔍 Systemic Root-Cause Diagnosis")
        st.caption("AI-synthesized error clusters identifying upstream ETL, scraping, or ingestion flaws.")
        if root_causes:
            for rc in root_causes:
                with st.container(border=True):
                    c_id = rc.get("cluster_id", "RC")
                    cause = rc.get("likely_cause", "Data quality anomaly cluster")
                    st.subheader(f"🏷️ [{c_id}] {cause}")
                    cols_str = ", ".join([f"`{c}`" for c in rc.get("affected_columns", [])])
                    st.markdown(f"**Affected Columns:** {cols_str} | **Affected Rows:** `{rc.get('affected_rows', 0)}`")
                    st.markdown(f"**Recommended Preventive Action:** {rc.get('recommended_action', 'N/A')}")
        else:
            st.info("No systemic root-cause anomalies detected.")

    # Tab 2: Repair Audit Trail
    with tab_repairs:
        st.markdown("### 🛠️ Auditable Cell Repair Log")
        st.caption("Every repaired cell is tracked with original 0-based CSV row index, before/after values, and deterministic confidence.")
        if repairs:
            rep_df = pd.DataFrame(repairs)
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                stages = sorted(rep_df["stage"].dropna().unique().tolist())
                sel_stages = st.multiselect("Filter by Stage", options=stages, default=stages)
            with f_col2:
                tiers = sorted(rep_df["confidence_tier"].dropna().unique().tolist())
                sel_tiers = st.multiselect("Filter by Confidence Tier", options=tiers, default=tiers)
            with f_col3:
                cols = sorted(rep_df["column"].dropna().unique().tolist())
                sel_cols = st.multiselect("Filter by Column", options=cols, default=cols)

            filtered_df = rep_df[
                rep_df["stage"].isin(sel_stages) &
                rep_df["confidence_tier"].isin(sel_tiers) &
                (rep_df["column"].isin(sel_cols) | rep_df["column"].isna())
            ]
            st.dataframe(filtered_df, use_container_width=True, hide_index=True)
            st.caption(f"Showing {len(filtered_df)} of {len(rep_df)} total repairs.")
        else:
            st.info("No repairs were needed; dataset was already completely clean.")

    # Tab 3: Quarantine / Unresolved
    with tab_quarantine:
        st.markdown("### 🛡️ Quarantined Cells & Unresolved Issues")
        st.info(
            "💡 **Blast-Radius Principle:** DataFix refuses to hallucinate uncertain repairs. "
            "Cells with deterministic confidence < 0.60, unsupported functional dependencies, "
            "or domain boundary traps are quarantined and preserved untouched for human verification."
        )
        if unresolved:
            unres_df = pd.DataFrame(unresolved)
            st.dataframe(unres_df, use_container_width=True, hide_index=True)
        else:
            st.success("🎉 Zero unresolved issues! All detected anomalies were repaired with high confidence.")

    # Tab 4: Before vs. After & Exports
    with tab_export:
        st.markdown("### 📊 Side-by-Side Dataset Preview")
        raw_preview_df = pd.read_csv(io.BytesIO(st.session_state.raw_csv_bytes), dtype=str)
        cleaned_csv_str = result.get("cleaned_csv", "")
        cleaned_df = pd.read_csv(io.StringIO(cleaned_csv_str), dtype=str)

        col_raw, col_cleaned = st.columns(2)
        with col_raw:
            st.markdown(f"**Original Dirty CSV (`{st.session_state.raw_filename}`):**")
            st.dataframe(raw_preview_df, use_container_width=True)
        with col_cleaned:
            st.markdown(f"**Repaired Clean CSV (`cleaned_{st.session_state.raw_filename}`):**")
            st.dataframe(cleaned_df, use_container_width=True)

        st.markdown("---")
        st.markdown("### 💾 Export Cleaned Dataset & Audit Trail")
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            st.download_button(
                label="⬇️ Download Cleaned CSV",
                data=cleaned_csv_str,
                file_name=f"cleaned_{st.session_state.raw_filename}",
                mime="text/csv",
                type="primary",
                use_container_width=True,
            )
        with exp_col2:
            audit_payload = {k: v for k, v in result.items() if k != "cleaned_csv"}
            audit_json = json.dumps(audit_payload, indent=2)
            stem = Path(st.session_state.raw_filename).stem
            st.download_button(
                label="⬇️ Download Full Audit Report (JSON)",
                data=audit_json,
                file_name=f"{stem}_audit_report.json",
                mime="application/json",
                use_container_width=True,
            )
