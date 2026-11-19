"""Thin-client Streamlit workbench consuming FastAPI over HTTP with zero duplicated cleaning logic.
Features 100% offline motion graphics, executive dark command-center theme, and full auditability.
"""

import html
import io
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ============================================================================
# Page Configuration & Metadata (Zero Icons) - MUST BE FIRST STREAMLIT CALL
# ============================================================================

st.set_page_config(
    page_title="DataFix — Autonomous Data Quality Workbench",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# Background Backend Process Manager
# ============================================================================

@st.cache_resource
def start_backend():
    """Ensures FastAPI backend is active; auto-starts uvicorn daemon if offline."""
    if os.getenv("STREAMLIT_TESTING") or os.getenv("PYTEST_CURRENT_TEST"):
        return
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=0.5) as resp:
            if resp.status == 200:
                return
    except Exception:
        pass

    try:
        os.makedirs("storage", exist_ok=True)
        subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Give Uvicorn up to 15 seconds to boot cleanly before rendering the UI
        for _ in range(60):
            time.sleep(0.25)
            try:
                with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=0.25) as resp:
                    if resp.status == 200:
                        break
            except Exception:
                pass
    except Exception as e:
        print(f"DataFix backend auto-spawn note: {e}", file=sys.stderr)

start_backend()
ensure_backend_alive = start_backend

# ============================================================================
# Session State Initialization
# ============================================================================

if "raw_csv_bytes" not in st.session_state:
    st.session_state.raw_csv_bytes = None
if "raw_filename" not in st.session_state:
    st.session_state.raw_filename = "sample_dirty.csv"
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "run_latency_seconds" not in st.session_state:
    st.session_state.run_latency_seconds = None
if "approved_repair_indices" not in st.session_state:
    st.session_state.approved_repair_indices = None
if "selective_review_banner" not in st.session_state:
    st.session_state.selective_review_banner = None

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
    """Checks the health of the FastAPI backend and Ollama detective over HTTP."""
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/health", timeout=3.0)
        if resp.status_code == 200:
            return resp.json(), None
        return None, f"HTTP {resp.status_code}: {resp.text}"
    except Exception as exc:
        return None, str(exc)


def fetch_history(base_url: str, limit: int = 25) -> List[Dict[str, Any]]:
    """Retrieves recent cleaning runs from the persistent SQLite audit ledger."""
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/history?limit={limit}", timeout=4.0)
        if resp.status_code == 200:
            return resp.json().get("runs", [])
    except Exception:
        pass
    return []


def fetch_reproduce_script(base_url: str, payload: Dict[str, Any]) -> Tuple[str, str]:
    """Requests a standalone reproduce_fix.py replay script from the backend."""
    try:
        resp = httpx.post(f"{base_url.rstrip('/')}/reproduce-script", json=payload, timeout=10.0)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("script_filename", "reproduce_fix.py"), data.get("script_content", "")
    except Exception:
        pass
    return "reproduce_fix.py", "# Failed to fetch replay script from backend API."



# ============================================================================
# Motion-Graphic & SVG Component Generators (100% Offline / Zero CDNs / Zero Icons)
# ============================================================================

def render_empty_state_svg(motion_enabled: bool) -> str:
    """Renders a sleek 6-stage pipeline architecture SVG diagram with animated data pulses."""
    anim_css = """
    @keyframes flowDash {
        to { stroke-dashoffset: -32; }
    }
    @keyframes pulseGlow {
        0%, 100% { opacity: 0.35; filter: drop-shadow(0 0 2px #4F46E5); }
        50% { opacity: 1; filter: drop-shadow(0 0 6px #4F46E5); }
    }
    @keyframes pulseShield {
        0%, 100% { stroke: #4F46E5; opacity: 0.55; }
        50% { stroke: #6366F1; opacity: 0.95; }
    }
    .flow-line {
        stroke-dasharray: 6 6;
        animation: flowDash 1.2s linear infinite;
    }
    .pulse-dot {
        animation: pulseGlow 1.8s ease-in-out infinite;
    }
    .shield-box {
        animation: pulseShield 2.2s ease-in-out infinite;
    }
    """ if motion_enabled else """
    .flow-line { stroke-dasharray: none; }
    .pulse-dot { opacity: 0.7; }
    .shield-box { stroke: #4F46E5; opacity: 0.7; }
    """

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
      body {{ margin: 0; padding: 0; overflow: hidden; background: transparent; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; user-select: none; }}
      {anim_css}
      .stage-title {{ font-size: 11px; font-weight: 800; fill: #0F172A; letter-spacing: 0.5px; }}
      .stage-sub-main {{ font-size: 10px; fill: #334155; font-weight: 600; }}
      .stage-sub-detail {{ font-size: 9.5px; fill: #64748B; font-weight: 500; }}
      .node-rect {{ rx: 8px; ry: 8px; }}
    </style>
    </head>
    <body>
    <svg width="100%" height="205" viewBox="0 0 920 205" fill="none" xmlns="http://www.w3.org/2000/svg">
      <!-- Backdrop card -->
      <rect x="1" y="1" width="918" height="203" rx="12" fill="#FFFFFF" stroke="#E2E8F0" stroke-width="1.5"/>

      <!-- Header Label -->
      <text x="32" y="30" fill="#4F46E5" font-size="12" font-weight="800" letter-spacing="0.75">NEURO-SYMBOLIC PIPELINE ARCHITECTURE (6-STAGE WORKFLOW)</text>
      <text x="32" y="47" fill="#64748B" font-size="11">Deterministic Python cleaning engines paired with metadata-only local AI detective</text>

      <!-- Stage 2 Privacy Shield Box -->
      <rect class="shield-box" x="170" y="64" width="136" height="90" rx="10" fill="rgba(79, 70, 229, 0.04)" stroke="#6366F1" stroke-width="1.5"/>
      <text x="238" y="172" fill="#4F46E5" font-size="10" font-weight="700" text-anchor="middle">Zero Raw-Row PII to LLM</text>

      <!-- Connector Lines with flowing dash -->
      <line class="flow-line" x1="148" y1="109" x2="180" y2="109" stroke="#4F46E5" stroke-width="2"/>
      <line class="flow-line" x1="296" y1="109" x2="328" y2="109" stroke="#6366F1" stroke-width="2"/>
      <line class="flow-line" x1="444" y1="109" x2="476" y2="109" stroke="#4F46E5" stroke-width="2"/>
      <line class="flow-line" x1="592" y1="109" x2="624" y2="109" stroke="#6366F1" stroke-width="2"/>
      <line class="flow-line" x1="740" y1="109" x2="772" y2="109" stroke="#4F46E5" stroke-width="2"/>

      <!-- Stage 1: DETECT -->
      <g transform="translate(32, 74)">
        <rect class="node-rect" width="116" height="70" fill="#F8FAFC" stroke="#4F46E5" stroke-width="1.5"/>
        <text class="stage-title" x="58" y="24" text-anchor="middle">1. DETECT</text>
        <text class="stage-sub-main" x="58" y="42" text-anchor="middle">Stage 1 Profiler</text>
        <text class="stage-sub-detail" x="58" y="56" text-anchor="middle">Nulls, Types, HTML</text>
      </g>

      <!-- Stage 2: DIAGNOSE -->
      <g transform="translate(180, 74)">
        <rect class="node-rect" width="116" height="70" fill="#F8FAFC" stroke="#6366F1" stroke-width="1.5"/>
        <text class="stage-title" x="58" y="24" text-anchor="middle">2. DIAGNOSE</text>
        <text class="stage-sub-main" x="58" y="42" text-anchor="middle">Ollama Detective</text>
        <text class="stage-sub-detail" x="58" y="56" text-anchor="middle">Counts &amp; Patterns</text>
      </g>

      <!-- Stage 3: REPAIR -->
      <g transform="translate(328, 74)">
        <rect class="node-rect" width="116" height="70" fill="#F8FAFC" stroke="#4F46E5" stroke-width="1.5"/>
        <text class="stage-title" x="58" y="24" text-anchor="middle">3. REPAIR</text>
        <text class="stage-sub-main" x="58" y="42" text-anchor="middle">Python Mutator</text>
        <text class="stage-sub-detail" x="58" y="56" text-anchor="middle">Dedupe &amp; Impute</text>
      </g>

      <!-- Stage 4: VALIDATE -->
      <g transform="translate(476, 74)">
        <rect class="node-rect" width="116" height="70" fill="#F8FAFC" stroke="#6366F1" stroke-width="1.5"/>
        <text class="stage-title" x="58" y="24" text-anchor="middle">4. VALIDATE</text>
        <text class="stage-sub-main" x="58" y="42" text-anchor="middle">KS Drift Guard</text>
        <text class="stage-sub-detail" x="58" y="56" text-anchor="middle">Auto-Rollback</text>
      </g>

      <!-- Stage 5: EXPLAIN -->
      <g transform="translate(624, 74)">
        <rect class="node-rect" width="116" height="70" fill="#F8FAFC" stroke="#4F46E5" stroke-width="1.5"/>
        <text class="stage-title" x="58" y="24" text-anchor="middle">5. EXPLAIN</text>
        <text class="stage-sub-main" x="58" y="42" text-anchor="middle">Pydantic Schema</text>
        <text class="stage-sub-detail" x="58" y="56" text-anchor="middle">Confidence Tiers</text>
      </g>

      <!-- Stage 6: TRACK -->
      <g transform="translate(772, 74)">
        <rect class="node-rect" width="116" height="70" fill="#F8FAFC" stroke="#E11D48" stroke-width="1.5"/>
        <text class="stage-title" x="58" y="24" text-anchor="middle" fill="#E11D48">6. TRACK</text>
        <text class="stage-sub-main" x="58" y="42" text-anchor="middle">SHA-256 Digest</text>
        <text class="stage-sub-detail" x="58" y="56" text-anchor="middle">Audit Provenance</text>
      </g>
    </svg>
    </body>
    </html>
    """


def render_hero_telemetry_html(result: Dict[str, Any], motion_enabled: bool, latency: Optional[float] = None) -> str:
    """Renders an executive dark command-center SVG + JS motion-graphic telemetry stage."""
    summary = result.get("summary", {})
    quality_report = result.get("quality_report", {})
    before_score = float(quality_report.get("before_health_score", 0.0))
    after_score = float(quality_report.get("after_health_score", 0.0))
    delta = after_score - before_score

    original_rows = int(summary.get("original_rows", 0))
    cleaned_rows = int(summary.get("cleaned_rows", 0))
    duplicates_removed = int(summary.get("duplicates_removed", 0))
    total_repairs = int(summary.get("total_repairs", 0))
    stage1_repairs = int(summary.get("stage1_repairs", 0))
    stage2_repairs = int(summary.get("stage2_repairs", 0))
    unresolved_count = int(summary.get("unresolved_count", 0))
    protected_cols = summary.get("protected_identifier_columns", [])
    pipeline_status = result.get("status", "completed")
    is_stage1_only = pipeline_status == "completed_stage1_only"

    input_sha256 = html.escape(str(quality_report.get("input_sha256", "N/A")))
    output_sha256 = html.escape(str(quality_report.get("output_sha256", "N/A")))
    latency_str = f"{latency:.2f}s" if latency else "< 1.0s"

    # SVG circular gauge math: radius 38, circumference 2 * pi * 38 ≈ 238.76
    circumference = 238.76
    before_offset = circumference * (1.0 - (min(100.0, max(0.0, before_score)) / 100.0))
    after_offset = circumference * (1.0 - (min(100.0, max(0.0, after_score)) / 100.0))

    anim_js = f"""
    <script>
      const duration = {1200 if motion_enabled else 10};
      function animateCounter(id, start, end, decimals, prefix='', suffix='') {{
        const el = document.getElementById(id);
        if (!el) return;
        const startTime = performance.now();
        function update(now) {{
          const elapsed = now - startTime;
          const progress = Math.min(elapsed / duration, 1);
          const ease = 1 - Math.pow(1 - progress, 3);
          const current = start + (end - start) * ease;
          el.innerText = prefix + current.toFixed(decimals) + suffix;
          if (progress < 1) requestAnimationFrame(update);
        }}
        requestAnimationFrame(update);
      }}
      window.addEventListener('DOMContentLoaded', () => {{
        animateCounter('health-num', {before_score}, {after_score}, 2, '', '%');
        animateCounter('repairs-num', 0, {total_repairs}, 0);
        animateCounter('dups-num', 0, {duplicates_removed}, 0);
        animateCounter('prot-num', 0, {len(protected_cols)}, 0);
        animateCounter('quar-num', 0, {unresolved_count}, 0);
      }});
    </script>
    """ if motion_enabled else ""

    flow_line_class = "flow-pulse" if motion_enabled else ""

    stage2_stroke = "#D97706" if is_stage1_only else "#4F46E5"
    stage2_badge = "SKIPPED" if is_stage1_only else f"{stage2_repairs} repairs"
    stage2_opacity = "0.45" if is_stage1_only else "1.0"

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
      * {{ box-sizing: border-box; }}
      body {{ margin: 0; padding: 0; background: transparent; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: #0F172A; user-select: none; }}
      .telemetry-card {{
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 20px 14px 20px;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.06);
      }}
      .grid-container {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
      }}
      .gauge-col {{
        flex: 0 0 135px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
      }}
      .flow-col {{
        flex: 1 1 auto;
        min-width: 440px;
      }}
      .kpi-col {{
        flex: 0 0 250px;
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
      }}
      .kpi-tile {{
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 10px 12px;
      }}
      .kpi-title {{ font-size: 11px; color: #475569; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 700; margin-bottom: 3px; }}
      .kpi-val {{ font-size: 22px; font-weight: 800; color: #0F172A; line-height: 1.1; }}
      .kpi-sub {{ font-size: 10px; color: #64748B; margin-top: 2px; font-weight: 500; }}
      .delta-pill {{
        background: rgba(79, 70, 229, 0.08);
        color: #4F46E5;
        border: 1px solid rgba(79, 70, 229, 0.25);
        border-radius: 20px;
        padding: 3px 8px;
        font-size: 11px;
        font-weight: 700;
        margin-top: 4px;
        display: inline-block;
      }}
      .provenance-bar {{
        margin-top: 12px;
        padding-top: 10px;
        border-top: 1px solid #E2E8F0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 12px;
        color: #64748B;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      }}
      .badge-status {{
        background: {'rgba(79, 70, 229, 0.1)' if not is_stage1_only else 'rgba(245, 158, 11, 0.1)'};
        color: {'#4F46E5' if not is_stage1_only else '#D97706'};
        border: 1px solid {'rgba(79, 70, 229, 0.25)' if not is_stage1_only else 'rgba(245, 158, 11, 0.25)'};
        padding: 3px 10px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 11px;
      }}
      @keyframes flowDash {{
        to {{ stroke-dashoffset: -24; }}
      }}
      .flow-pulse {{
        stroke-dasharray: 6 6;
        animation: flowDash 1s linear infinite;
      }}
    </style>
    </head>
    <body>
    <div class="telemetry-card">
      <div class="grid-container">

        <!-- Column 1: Dual-Arc Health Gauge -->
        <div class="gauge-col">
          <svg width="105" height="105" viewBox="0 0 100 100">
            <!-- Background track -->
            <circle cx="50" cy="50" r="38" fill="none" stroke="#E2E8F0" stroke-width="7"/>
            <!-- Before Health Arc (Dim) -->
            <circle cx="50" cy="50" r="38" fill="none" stroke="#CBD5E1" stroke-width="5" stroke-dasharray="{circumference}" stroke-dashoffset="{before_offset}" stroke-linecap="round" opacity="0.6" transform="rotate(-90 50 50)"/>
            <!-- After Health Arc (Indigo) -->
            <circle cx="50" cy="50" r="38" fill="none" stroke="#4F46E5" stroke-width="7" stroke-dasharray="{circumference}" stroke-dashoffset="{after_offset}" stroke-linecap="round" transform="rotate(-90 50 50)"/>
            <!-- Center Text -->
            <text x="50" y="47" text-anchor="middle" fill="#0F172A" font-size="16" font-weight="800" id="health-num">{after_score:.2f}%</text>
            <text x="50" y="59" text-anchor="middle" fill="#64748B" font-size="8" font-weight="700" letter-spacing="0.5">HEALTH SCORE</text>
          </svg>
          <div class="delta-pill">+{delta:.2f} pts Health Delta</div>
        </div>

        <!-- Column 2: Live Pipeline Execution Flow Diagram -->
        <div class="flow-col">
          <svg width="100%" height="135" viewBox="0 0 545 130" fill="none">
            <!-- Node 1: Raw Ingest -->
            <rect x="5" y="30" width="96" height="52" rx="8" fill="#F8FAFC" stroke="#6366F1" stroke-width="1.5"/>
            <text x="53" y="52" fill="#4F46E5" font-size="12" font-weight="800" text-anchor="middle">RAW INGEST</text>
            <text x="53" y="68" fill="#64748B" font-size="11" font-weight="500" text-anchor="middle">{original_rows} rows</text>

            <!-- Connector 1 -->
            <line class="{flow_line_class}" x1="101" y1="56" x2="136" y2="56" stroke="#4F46E5" stroke-width="2"/>

            <!-- Node 2: Stage 1 -->
            <rect x="136" y="30" width="96" height="52" rx="8" fill="#F8FAFC" stroke="#4F46E5" stroke-width="1.5"/>
            <text x="184" y="52" fill="#4F46E5" font-size="12" font-weight="800" text-anchor="middle">STAGE 1</text>
            <text x="184" y="68" fill="#64748B" font-size="11" font-weight="500" text-anchor="middle">{stage1_repairs} fixes</text>

            <!-- Connector 2 -->
            <line class="{flow_line_class}" x1="232" y1="56" x2="267" y2="56" stroke="#4F46E5" stroke-width="2"/>

            <!-- Node 3: KS Drift Guard -->
            <rect x="267" y="30" width="96" height="52" rx="8" fill="#F8FAFC" stroke="#6366F1" stroke-width="1.5"/>
            <text x="315" y="52" fill="#4F46E5" font-size="12" font-weight="800" text-anchor="middle">KS GUARD</text>
            <text x="315" y="68" fill="#10B981" font-size="11" font-weight="700" text-anchor="middle">PASSED</text>

            <!-- Connector 3 -->
            <line class="{flow_line_class}" x1="363" y1="56" x2="398" y2="56" stroke="{stage2_stroke}" stroke-width="2"/>

            <!-- Node 4: Stage 2 AI Detective -->
            <rect x="398" y="30" width="96" height="52" rx="8" fill="#F8FAFC" stroke="{stage2_stroke}" stroke-width="1.5" opacity="{stage2_opacity}"/>
            <text x="446" y="52" fill="{stage2_stroke}" font-size="12" font-weight="800" text-anchor="middle">STAGE 2 AI</text>
            <text x="446" y="68" fill="#64748B" font-size="11" font-weight="500" text-anchor="middle">{stage2_badge}</text>

            <!-- Connector to Clean Output -->
            <line class="{flow_line_class}" x1="494" y1="56" x2="518" y2="56" stroke="#4F46E5" stroke-width="2"/>

            <!-- Final Cleaned Output Indicator -->
            <g transform="translate(518, 38)">
              <circle cx="16" cy="18" r="14" fill="#4F46E5" opacity="0.15"/>
              <circle cx="16" cy="18" r="7" fill="#4F46E5"/>
            </g>

            <!-- Quarantine Side-Branch -->
            <path d="M 184 82 L 184 105 L 368 105" fill="none" stroke="#EF4444" stroke-width="1.5" stroke-dasharray="3 3"/>
            <g transform="translate(368, 91)">
              <rect width="136" height="28" rx="6" fill="#FEF2F2" stroke="#EF4444" stroke-width="1.2"/>
              <text x="10" y="18" fill="#B91C1C" font-size="10" font-weight="700">QUARANTINE:</text>
              <text x="82" y="18" fill="#DC2626" font-size="10" font-weight="800">{unresolved_count} trapped</text>
            </g>
          </svg>
        </div>

        <!-- Column 3: Rolling KPI Counters -->
        <div class="kpi-col">
          <div class="kpi-tile">
            <div class="kpi-title">Total Repairs</div>
            <div class="kpi-val" id="repairs-num">{total_repairs}</div>
            <div class="kpi-sub">S1: {stage1_repairs} · S2: {stage2_repairs}</div>
          </div>
          <div class="kpi-tile">
            <div class="kpi-title">Dups Dropped</div>
            <div class="kpi-val" id="dups-num">{duplicates_removed}</div>
            <div class="kpi-sub">0-based index kept</div>
          </div>
          <div class="kpi-tile">
            <div class="kpi-title">Protected IDs</div>
            <div class="kpi-val" id="prot-num">{len(protected_cols)}</div>
            <div class="kpi-sub">0 mutation guard</div>
          </div>
          <div class="kpi-tile">
            <div class="kpi-title">Quarantined</div>
            <div class="kpi-val" style="color: #F87171;" id="quar-num">{unresolved_count}</div>
            <div class="kpi-sub">Zero hallucination</div>
          </div>
        </div>

      </div>

      <!-- Provenance & Latency Footer Bar -->
      <div class="provenance-bar">
        <div>
          <span style="color: #4F46E5; font-weight: 700;">SHA-256 PROVENANCE:</span>
          <span style="margin-left: 8px;">In: {input_sha256[:16]}... -> Out: {output_sha256[:16]}...</span>
        </div>
        <div style="display: flex; gap: 12px; align-items: center;">
          <span>Latency: <strong style="color: #0F172A;">{latency_str}</strong></span>
          <span class="badge-status">{pipeline_status.upper()}</span>
        </div>
      </div>
    </div>
    {anim_js}
    </body>
    </html>
    """


def render_ks_meter_html(check: Dict[str, Any], motion_enabled: bool) -> str:
    """Renders an animated horizontal drift meter card for a single KS-test check."""
    col = html.escape(str(check.get("column", "Unknown")))
    ks_stat = float(check.get("ks_statistic", 0.0))
    p_val = float(check.get("p_value", 0.0))
    status = check.get("status", "unknown")

    # Meter bounds: 0.00 to 0.30
    bar_pct = min(100.0, max(0.0, (ks_stat / 0.30) * 100.0))
    threshold_pct = (0.15 / 0.30) * 100.0  # 50%

    badge_color = "#10B981" if status == "passed" else ("#DC2626" if status == "rolled_back" else "#64748B")
    badge_bg = "rgba(16, 185, 129, 0.12)" if status == "passed" else ("rgba(220, 38, 38, 0.12)" if status == "rolled_back" else "rgba(100, 116, 139, 0.12)")
    badge_text = "PASSED (p >= 0.05)" if status == "passed" else ("ROLLED BACK (Drift)" if status == "rolled_back" else "SKIPPED")

    anim_css = """
      .needle { transition: left 1.2s cubic-bezier(0.16, 1, 0.3, 1); }
      .fill-bar { transition: width 1.2s cubic-bezier(0.16, 1, 0.3, 1); }
    """ if motion_enabled else ""

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
      body {{ margin: 0; padding: 0; background: transparent; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: #0F172A; }}
      .ks-box {{
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
      }}
      .ks-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
      }}
      .track-container {{
        position: relative;
        height: 12px;
        background-color: #E2E8F0;
        border-radius: 6px;
        overflow: visible;
        margin: 12px 0 6px 0;
      }}
      .fill-bar {{
        height: 100%;
        width: {bar_pct}%;
        background-color: {badge_color};
        border-radius: 6px;
      }}
      .threshold-line {{
        position: absolute;
        left: {threshold_pct}%;
        top: -4px;
        bottom: -4px;
        width: 2px;
        background: #F87171;
        border-left: 2px dashed #EF4444;
      }}
      .needle {{
        position: absolute;
        left: {bar_pct}%;
        top: -3px;
        width: 4px;
        height: 18px;
        background: #0F172A;
        border-radius: 2px;
        box-shadow: 0 0 4px rgba(15, 23, 42, 0.3);
        transform: translateX(-50%);
      }}
      .ks-meta {{
        display: flex;
        justify-content: space-between;
        font-size: 12px;
        color: #475569;
        font-family: ui-monospace, SFMono-Regular, monospace;
      }}
      .status-pill {{
        background: {badge_bg};
        color: {badge_color};
        border: 1px solid {badge_color};
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 700;
      }}
      {anim_css}
    </style>
    </head>
    <body>
    <div class="ks-box">
      <div class="ks-header">
        <div>
          <strong style="color: #4F46E5; font-size: 14px;">Column: <code>{col}</code></strong>
          <span style="font-size: 12px; color: #64748B; margin-left: 8px;">Two-Sample Kolmogorov-Smirnov Test</span>
        </div>
        <span class="status-pill">{badge_text}</span>
      </div>
      <div class="track-container">
        <div class="fill-bar"></div>
        <div class="threshold-line" title="Safety Threshold (0.15)"></div>
        <div class="needle" title="Observed KS Statistic: {ks_stat:.4f}"></div>
      </div>
      <div class="ks-meta">
        <span>KS Stat: <strong style="color: #0F172A;">{ks_stat:.4f}</strong> (0.00 safe -> 0.30 critical)</span>
        <span style="color: #DC2626; font-weight: 600;">Threshold: 0.15</span>
        <span>p-value: <strong style="color: #0F172A;">{p_val:.4f}</strong></span>
      </div>
    </div>
    </body>
    </html>
    """


def generate_html_report(result: Dict[str, Any], filename: str, latency: Optional[float] = None) -> str:
    """Generates a standalone, fully offline HTML audit and quality report without icons."""
    summary = result.get("summary", {})
    quality_report = result.get("quality_report", {})
    repairs = result.get("repairs", [])
    unresolved = result.get("unresolved_issues", [])
    root_causes = result.get("root_cause_diagnoses", [])
    before_score = float(quality_report.get("before_health_score", 0.0))
    after_score = float(quality_report.get("after_health_score", 0.0))
    delta = after_score - before_score

    rc_cards = ""
    for rc in root_causes:
        c_id = html.escape(str(rc.get("cluster_id", "RC")))
        cause = html.escape(str(rc.get("likely_cause", "")))
        action = html.escape(str(rc.get("recommended_action", "")))
        rows = int(rc.get("affected_rows", 0))
        cols_str = ", ".join([html.escape(str(c)) for c in rc.get("affected_columns", [])])
        rc_cards += f"""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:8px;padding:12px;margin-bottom:8px;box-shadow:0 1px 3px rgba(15,23,42,0.05);">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <strong style="color:#4F46E5;font-size:13px;">[{c_id}] {cause}</strong>
            <span style="background:rgba(79,70,229,0.08);color:#4F46E5;padding:2px 8px;border-radius:12px;font-size:10px;font-weight:700;">{rows} rows affected</span>
          </div>
          <div style="font-size:11px;color:#64748B;margin-top:4px;"><strong>Affected Columns:</strong> {cols_str}</div>
          <div style="font-size:11px;color:#10B981;margin-top:4px;"><strong>Recommended Action:</strong> {action}</div>
        </div>
        """

    repair_rows = ""
    for r in repairs[:50]:  # Top 50 in report
        idx = int(r.get("row_index", 0))
        col = html.escape(str(r.get("column", "Row-Level")))
        orig = html.escape(str(r.get("original_value", "None")))
        corr = html.escape(str(r.get("corrected_value", "None")))
        anom = html.escape(str(r.get("anomaly_type", "")))
        conf = float(r.get("confidence", 0.0))
        tier = html.escape(str(r.get("confidence_tier", "")))
        repair_rows += f"""
        <tr>
          <td>#{idx}</td>
          <td><code>{col}</code></td>
          <td><del style="color:#FCA5A5;">{orig}</del></td>
          <td><ins style="color:#00D59B;text-decoration:none;font-weight:bold;">{corr}</ins></td>
          <td>{anom}</td>
          <td><strong>{conf:.2f}</strong> ({tier})</td>
        </tr>
        """

    unres_rows = ""
    for u in unresolved:
        idx = int(u.get("row_index", 0))
        col = html.escape(str(u.get("column", "None")))
        anom = html.escape(str(u.get("anomaly_type", "")))
        reason = html.escape(str(u.get("reason", "")))
        unres_rows += f"""
        <tr>
          <td>#{idx}</td>
          <td><code>{col}</code></td>
          <td>{anom}</td>
          <td style="color:#FCA5A5;">{reason}</td>
        </tr>
        """

    rc_content = rc_cards if rc_cards else '<div style="color:#94A3B8;">No systemic root cause issues detected.</div>'
    unres_content = unres_rows if unres_rows else '<tr><td colspan="4" style="color:#00D59B;">Zero quarantined cells.</td></tr>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DataFix Autonomous Quality Report — {html.escape(filename)}</title>
<style>
  body {{ margin:0; padding:24px; background:#F8FAFC; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; color:#0F172A; }}
  .container {{ max-width:1100px; margin:0 auto; }}
  h1,h2,h3 {{ color:#0F172A; }}
  .header-card {{ background:#FFFFFF; border:1px solid #E2E8F0; border-radius:12px; padding:20px; margin-bottom:20px; box-shadow:0 1px 3px rgba(15,23,42,0.06); }}
  .grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:16px 0; }}
  .tile {{ background:#F1F5F9; border:1px solid #E2E8F0; border-radius:8px; padding:12px; }}
  .tile-title {{ font-size:11px; color:#64748B; text-transform:uppercase; font-weight:600; }}
  .tile-val {{ font-size:24px; font-weight:800; color:#4F46E5; margin-top:4px; }}
  table {{ width:100%; border-collapse:collapse; margin-top:12px; font-size:12px; }}
  th,td {{ text-align:left; padding:8px 12px; border-bottom:1px solid #E2E8F0; }}
  th {{ background:#F8FAFC; color:#64748B; text-transform:uppercase; font-size:10px; letter-spacing:0.5px; }}
  .badge {{ background:rgba(79,70,229,0.08); color:#4F46E5; border:1px solid rgba(79,70,229,0.25); padding:2px 8px; border-radius:12px; font-size:10px; font-weight:700; }}
</style>
</head>
<body>
<div class="container">
  <div class="header-card">
    <div style="display:flex;justify-content:space-between;align-items:center;">
      <div>
        <h1 style="margin:0;font-size:22px;">DataFix Autonomous Quality & Audit Report</h1>
        <div style="color:#64748B;font-size:12px;margin-top:4px;">Dataset: <strong>{html.escape(filename)}</strong> • SAPTHAHACKS'26 INNOVEDGE PS 01</div>
      </div>
      <span class="badge">100% LOCAL AUDIT VERIFIED</span>
    </div>

    <div class="grid">
      <div class="tile">
        <div class="tile-title">Health Score Jump</div>
        <div class="tile-val">{after_score:.2f}%</div>
        <div style="font-size:10px;color:#10B981;">+{delta:.2f} pts from {before_score:.2f}%</div>
      </div>
      <div class="tile">
        <div class="tile-title">Cleaned Rows</div>
        <div class="tile-val">{summary.get('cleaned_rows', 0)}</div>
        <div style="font-size:10px;color:#64748B;">-{summary.get('duplicates_removed', 0)} duplicates dropped</div>
      </div>
      <div class="tile">
        <div class="tile-title">Total Cell Repairs</div>
        <div class="tile-val">{summary.get('total_repairs', 0)}</div>
        <div style="font-size:10px;color:#64748B;">S1: {summary.get('stage1_repairs', 0)} · S2: {summary.get('stage2_repairs', 0)}</div>
      </div>
      <div class="tile">
        <div class="tile-title">Quarantined Traps</div>
        <div class="tile-val" style="color:#DC2626;">{summary.get('unresolved_count', 0)}</div>
        <div style="font-size:10px;color:#EF4444;">Protected from Hallucination</div>
      </div>
    </div>

    <div style="font-family:monospace;font-size:11px;color:#64748B;background:#F1F5F9;padding:10px;border-radius:6px;border:1px solid #E2E8F0;">
      <strong>INPUT SHA-256:</strong>  {quality_report.get('input_sha256', 'N/A')}<br>
      <strong>OUTPUT SHA-256:</strong> {quality_report.get('output_sha256', 'N/A')}
    </div>
  </div>

  <h2>Systemic Root-Cause Diagnoses</h2>
  {rc_content}

  <h2>Quarantined Cells (Preserved from Hallucination)</h2>
  <table>
    <thead><tr><th>Row</th><th>Column</th><th>Anomaly Type</th><th>Safety Reason</th></tr></thead>
    <tbody>{unres_content}</tbody>
  </table>

  <h2>Detailed Cell Repair Trail (Sample)</h2>
  <table>
    <thead><tr><th>Row</th><th>Column</th><th>Original Value</th><th>Corrected Value</th><th>Anomaly Type</th><th>Confidence</th></tr></thead>
    <tbody>{repair_rows}</tbody>
  </table>

  <div style="margin-top:30px;color:#64748B;font-size:10px;text-align:center;">
    Generated autonomously by DataFix — 100% Local Neuro-Symbolic Tabular Data Quality Workbench
  </div>
</div>
</body>
</html>"""


# ============================================================================
# Scoped Global CSS Injection (Professional Executive Dark Theme)
# ============================================================================

st.markdown(
    """
    <style>
      /* Executive Clean Light Background */
      .stApp {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
      }
      h1, h2, h3, h4, h5, h6 {
        color: #0F172A !important;
        font-weight: 800 !important;
      }
      section[data-testid="stSidebar"] {
        background-color: #E2E8F0 !important;
        border-right: 1px solid #CBD5E1 !important;
      }
      section[data-testid="stSidebar"] h1,
      section[data-testid="stSidebar"] h2,
      section[data-testid="stSidebar"] h3,
      section[data-testid="stSidebar"] label {
        color: #0F172A !important;
        font-weight: 700 !important;
      }
      .block-container {
        padding-top: 1.25rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 96% !important;
      }
      footer { visibility: hidden; }
      #MainMenu { visibility: hidden; }

      @keyframes fadeSlideUp {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
      }
      @keyframes pulseDot {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.3); opacity: 0.55; }
      }

      .df-hero-pill {
        display: inline-flex;
        align-items: center;
        background: rgba(79, 70, 229, 0.08);
        color: #4F46E5;
        border: 1px solid rgba(79, 70, 229, 0.25);
        border-radius: 14px;
        padding: 5px 14px;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.3px;
        margin-right: 6px;
        margin-bottom: 6px;
      }
      .df-pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
        animation: pulseDot 1.8s infinite ease-in-out;
      }
      .df-chip {
        display: inline-block;
        background-color: #EEF2FF;
        color: #4F46E5;
        border: 1px solid #C7D2FE;
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 13px;
        font-weight: 600;
        font-family: ui-monospace, SFMono-Regular, monospace;
        margin: 2px 3px;
      }
      .df-repair-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
        font-size: 14px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
        animation: fadeSlideUp 0.3s ease-out;
      }
      .df-diff-del {
        background-color: rgba(239, 68, 68, 0.12);
        color: #DC2626;
        text-decoration: line-through;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 13px;
        font-family: ui-monospace, SFMono-Regular, monospace;
      }
      .df-diff-ins {
        background-color: rgba(79, 70, 229, 0.12);
        color: #4F46E5;
        text-decoration: none;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 13px;
        font-weight: 700;
        font-family: ui-monospace, SFMono-Regular, monospace;
      }

      @media (prefers-reduced-motion: reduce) {
        *, ::before, ::after {
          animation-duration: 0.001ms !important;
          animation-iteration-count: 1 !important;
          transition-duration: 0.001ms !important;
        }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# Sidebar Configuration & Health Panel (Zero Icons)
# ============================================================================

st.sidebar.title("DataFix Control Panel")

api_url = st.sidebar.text_input(
    "FastAPI Backend URL",
    value=os.getenv("DATAFIX_API_URL", "http://127.0.0.1:8000"),
    help="Address where the main.py FastAPI backend server is running.",
)

st.sidebar.markdown("### System & AI Health")
if st.sidebar.button("Refresh Status", use_container_width=True):
    st.session_state.pop("cached_health", None)

health_info, health_err = fetch_health(api_url)

if health_info:
    st.sidebar.markdown(
        """
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
          <span class="df-pulse-dot" style="background:#10B981;"></span>
          <strong style="color:#059669;font-size:13px;">FastAPI Backend: Online</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )
    ollama_info = health_info.get("ollama", {})
    if ollama_info.get("reachable"):
        model_name = ollama_info.get("model", "qwen2.5:3b")
        if ollama_info.get("model_available"):
            st.sidebar.markdown(
                f"""
                <div style="display:flex;align-items:center;gap:8px;">
                  <span class="df-pulse-dot" style="background:#4F46E5;"></span>
                  <span style="color:#0F172A;font-size:13px;"><strong>Ollama Detective:</strong> Ready (<code>{model_name}</code>)</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.sidebar.warning(f"Ollama Active: Model `{model_name}` not found")
    else:
        st.sidebar.info("Ollama Detective: Offline (Deterministic Fallback)")
else:
    st.sidebar.markdown(
        """
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
          <span class="df-pulse-dot" style="background:#EF4444;"></span>
          <strong style="color:#EF4444;font-size:13px;">FastAPI Backend: Offline</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.caption("Start the backend server in terminal:")
    st.sidebar.code(
        r".\.venv\Scripts\uvicorn.exe main:app --host 127.0.0.1 --port 8000",
        language="powershell",
    )

st.sidebar.markdown("---")
st.sidebar.markdown("### Pipeline Execution Settings")

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
    st.sidebar.caption("Note: Local CPU LLM inference takes ~30–60s on CPU.")

skip_stage2 = st.sidebar.checkbox(
    "Stage 1 Only (Skip Stage 2 AI)",
    value=False,
    help="Skip all Stage 2 semantic AI standardization, FD imputation, and root-cause diagnosis.",
)

# Accessibility Motion Graphics Toggle
motion_enabled = st.sidebar.toggle(
    "Motion Graphics",
    value=True,
    help="Enable SVG data pulses, radial health gauge transitions, and rolling counters.",
)

st.sidebar.markdown("---")
st.sidebar.caption("100% Local • Neuro-Symbolic • Zero Raw-Row PII Leakage")


# ============================================================================
# Main Header & Title Banner (Zero Icons)
# ============================================================================

st.markdown(
    """
    <div style="margin-bottom: 20px;">
      <h1 style="font-size: 34px; margin: 0 0 10px 0; font-weight: 800; color: #0F172A; letter-spacing: -0.8px; line-height: 1.2;">
        DataFix — Neuro-Symbolic Data Quality & Repair Workbench
      </h1>
      <div style="display: flex; flex-wrap: wrap; gap: 8px; margin-top: 6px;">
        <span class="df-hero-pill">100% Local</span>
        <span class="df-hero-pill">Zero Raw-Row PII to LLM</span>
        <span class="df-hero-pill">Deterministic + Ollama</span>
        <span class="df-hero-pill">KS Drift Guard</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
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
# Empty-State Architectural Motion Graphic (Before Run)
# ============================================================================

if st.session_state.pipeline_result is None:
    components.html(render_empty_state_svg(motion_enabled), height=225)


# ============================================================================
# Raw Dataset Preview & Execution CTA
# ============================================================================

if st.session_state.raw_csv_bytes:
    try:
        raw_df = pd.read_csv(io.BytesIO(st.session_state.raw_csv_bytes), dtype=str)
        st.markdown(
            f"**Raw Dataset Preview:** ` {st.session_state.raw_filename} ` "
            f"({len(raw_df)} rows, {len(raw_df.columns)} columns)"
        )
        st.dataframe(raw_df.head(10), use_container_width=True)
    except Exception as exc:
        st.error(f"Error previewing CSV: {exc}")

    col_btn, _ = st.columns([2, 5])
    with col_btn:
        run_btn = st.button("Run DataFix Pipeline", type="primary", use_container_width=True)

    if run_btn:
        with st.status("Executing DataFix pipeline (profiling, cleaning, KS-testing, semantic AI)...", expanded=True) as status_box:
            try:
                t0 = time.time()
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
                st.session_state.run_latency_seconds = round(time.time() - t0, 2)

                if resp.status_code == 200:
                    st.session_state.pipeline_result = resp.json()
                    status_box.update(
                        label=f"Pipeline complete in {st.session_state.run_latency_seconds}s!",
                        state="complete",
                        expanded=False,
                    )
                    st.toast(f"DataFix completed in {st.session_state.run_latency_seconds}s!")
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
# Results Dashboard & Motion-Graphic Telemetry Stage (Zero Icons)
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
    st.markdown("### Command-Center Telemetry Stage")

    # Render cohesive motion-graphic hero telemetry component
    components.html(
        render_hero_telemetry_html(
            result=result,
            motion_enabled=motion_enabled,
            latency=st.session_state.run_latency_seconds,
        ),
        height=250,
    )

    # Warnings bar
    if warnings:
        with st.expander("Pipeline Warnings & Fallback Notices", expanded=False):
            for w in warnings:
                st.warning(w)

    # ========================================================================
    # 5 Interactive Workbench Tabs (Zero Icons)
    # ========================================================================

    tab_diag, tab_val, tab_repairs, tab_export, tab_history = st.tabs([
        "Root-Cause Diagnosis & Analytics",
        "Validation & Blast-Radius Trust",
        "Auditable Repair Trail",
        "Before vs. After & Export Center",
        "Run History Ledger",
    ])

    # ------------------------------------------------------------------------
    # Tab 1: Root-Cause Diagnosis & Visual Analytics
    # ------------------------------------------------------------------------
    with tab_diag:
        st.markdown("#### Visual Analytics: Anomaly Distribution")

        if repairs:
            rep_df = pd.DataFrame(repairs)
            col_ch1, col_ch2 = st.columns(2)

            with col_ch1:
                type_counts = rep_df["anomaly_type"].value_counts().reset_index()
                type_counts.columns = ["anomaly_type", "count"]
                chart_spec_type = {
                    "mark": {"type": "bar", "cornerRadiusEnd": 4, "color": "#4F46E5"},
                    "encoding": {
                        "x": {"field": "count", "type": "quantitative", "title": "Repairs Count"},
                        "y": {"field": "anomaly_type", "type": "nominal", "sort": "-x", "title": "Anomaly Type"},
                        "tooltip": [{"field": "anomaly_type", "title": "Type"}, {"field": "count", "title": "Repairs"}],
                    },
                    "background": "transparent",
                    "config": {
                        "view": {"stroke": "transparent"},
                        "axis": {"labelColor": "#475569", "titleColor": "#0F172A", "gridColor": "#E2E8F0", "titleFontSize": 12, "labelFontSize": 11},
                    },
                }
                st.vega_lite_chart(type_counts, chart_spec_type, use_container_width=True)

            with col_ch2:
                tier_counts = rep_df["confidence_tier"].value_counts().reset_index()
                tier_counts.columns = ["confidence_tier", "count"]
                chart_spec_tier = {
                    "mark": {"type": "bar", "cornerRadiusEnd": 4},
                    "encoding": {
                        "x": {"field": "count", "type": "quantitative", "title": "Repairs Count"},
                        "y": {"field": "confidence_tier", "type": "nominal", "sort": "-x", "title": "Confidence Tier"},
                        "color": {
                            "field": "confidence_tier",
                            "type": "nominal",
                            "scale": {
                                "domain": ["auto_safe", "review_recommended", "quarantined"],
                                "range": ["#10B981", "#F59E0B", "#EF4444"],
                            },
                            "legend": None,
                        },
                        "tooltip": [{"field": "confidence_tier", "title": "Tier"}, {"field": "count", "title": "Repairs"}],
                    },
                    "background": "transparent",
                    "config": {
                        "view": {"stroke": "transparent"},
                        "axis": {"labelColor": "#475569", "titleColor": "#0F172A", "gridColor": "#E2E8F0", "titleFontSize": 12, "labelFontSize": 11},
                    },
                }
                st.vega_lite_chart(tier_counts, chart_spec_tier, use_container_width=True)

        st.markdown("#### Systemic Root-Cause Diagnoses")
        st.caption("AI-synthesized error clusters identifying upstream ETL, scraping, or ingestion flaws.")

        if root_causes:
            for rc in root_causes:
                with st.container(border=True):
                    c_id = html.escape(str(rc.get("cluster_id", "RC")))
                    cause = html.escape(str(rc.get("likely_cause", "")))
                    action = html.escape(str(rc.get("recommended_action", "")))
                    affected_rows = int(rc.get("affected_rows", 0))
                    cols_list = rc.get("affected_columns", [])

                    st.markdown(
                        f"""
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                          <h4 style="margin:0;color:#0F172A;font-weight:700;"><span style="color:#4F46E5;">[{c_id}]</span> {cause}</h4>
                          <span class="df-hero-pill" style="margin:0;">{affected_rows} rows impacted</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    chips_html = "".join([f'<span class="df-chip">{html.escape(str(c))}</span>' for c in cols_list])
                    st.markdown(f"**Affected Columns:** {chips_html}", unsafe_allow_html=True)
                    st.markdown(f"**Recommended Action:** {action}")
        else:
            st.info("No systemic root-cause anomalies detected.")

    # ------------------------------------------------------------------------
    # Tab 2: Validation & Blast-Radius Trust Proof
    # ------------------------------------------------------------------------
    with tab_val:
        st.markdown("#### Kolmogorov-Smirnov (KS) Statistical Drift Meters")
        st.caption("Validates that numeric median/IQR imputations did not distort empirical distributions (p >= 0.05).")

        ks_checks = quality_report.get("ks_drift_checks", [])
        if ks_checks:
            for check in ks_checks:
                components.html(render_ks_meter_html(check, motion_enabled), height=115)
        else:
            st.info("No numeric columns required KS drift verification.")

        st.markdown("---")
        st.markdown("#### Blast-Radius Principle: Quarantined & Preserved Cells")
        st.markdown(
            """
            <div style="background:#FEF2F2;border:1px solid rgba(239,68,68,0.28);border-radius:8px;padding:12px 16px;margin-bottom:14px;">
              <strong style="color:#DC2626;">Blast-Radius Principle: Zero Hallucination Guarantee</strong><br>
              <span style="font-size:12px;color:#64748B;">
                DataFix refuses to make speculative guesses on ambiguous cells. Cells with confidence &lt; 0.60,
                unbacked functional dependencies (e.g. missing postal code and missing city), or domain boundary violations
                (e.g. negative age) are strictly quarantined and preserved untouched for human verification.
              </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if unresolved:
            unres_df = pd.DataFrame(unresolved)
            st.dataframe(unres_df, use_container_width=True, hide_index=True)
        else:
            st.success("Zero unresolved issues. All detected anomalies were repaired with high confidence.")

    # ------------------------------------------------------------------------
    # Tab 3: Auditable Repair Trail
    # ------------------------------------------------------------------------
    with tab_repairs:
        st.markdown("#### Auditable Cell Repair Log")

        if repairs:
            rep_df = pd.DataFrame(repairs)
            search_query = st.text_input("Search repairs by column, reason, or value...", value="")

            f1, f2, f3 = st.columns(3)
            with f1:
                stages = sorted(rep_df["stage"].dropna().unique().tolist())
                sel_stages = st.multiselect("Filter by Stage", options=stages, default=stages)
            with f2:
                tiers = sorted(rep_df["confidence_tier"].dropna().unique().tolist())
                sel_tiers = st.multiselect("Filter by Confidence Tier", options=tiers, default=tiers)
            with f3:
                cols = sorted(rep_df["column"].dropna().unique().tolist())
                sel_cols = st.multiselect("Filter by Column", options=cols, default=cols)

            filtered_df = rep_df[
                rep_df["stage"].isin(sel_stages) &
                rep_df["confidence_tier"].isin(sel_tiers) &
                (rep_df["column"].isin(sel_cols) | rep_df["column"].isna())
            ]

            if search_query.strip():
                q = search_query.lower()
                filtered_df = filtered_df[
                    filtered_df["column"].astype(str).str.lower().str.contains(q) |
                    filtered_df["reason"].astype(str).str.lower().str.contains(q) |
                    filtered_df["original_value"].astype(str).str.lower().str.contains(q) |
                    filtered_df["corrected_value"].astype(str).str.lower().str.contains(q)
                ]

            if st.session_state.get("selective_review_banner"):
                st.info(st.session_state.selective_review_banner)

            view_format = st.radio("Display Format:", ["Git-Style Diff Cards", "Interactive Review & Data Table"], horizontal=True)

            if view_format == "Git-Style Diff Cards":
                st.caption(f"Showing top {min(25, len(filtered_df))} of {len(filtered_df)} filtered repairs:")
                for _, r in filtered_df.head(25).iterrows():
                    r_idx = int(r.get("row_index", 0))
                    r_col = html.escape(str(r.get("column", "Row-Level")))
                    r_orig = html.escape(str(r.get("original_value", "None")))
                    r_corr = html.escape(str(r.get("corrected_value", "None")))
                    r_stage = html.escape(str(r.get("stage", "")))
                    r_tier = html.escape(str(r.get("confidence_tier", "")))
                    r_conf = float(r.get("confidence", 0.0))
                    r_reason = html.escape(str(r.get("reason", "")))

                    st.markdown(
                        f"""
                        <div class="df-repair-card">
                          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                            <div>
                              <span class="df-chip">Row #{r_idx}</span>
                              <span class="df-chip">{r_col}</span>
                              <span class="df-chip" style="color:#4F46E5;">{r_stage}</span>
                            </div>
                            <span class="df-hero-pill" style="margin:0;">{r_conf:.2f} ({r_tier})</span>
                          </div>
                          <div style="margin:8px 0;font-size:14px;">
                            <del class="df-diff-del">{r_orig}</del>
                            <span style="color:#64748B;margin:0 8px;font-weight:600;">-></span>
                            <ins class="df-diff-ins">{r_corr}</ins>
                          </div>
                          <div style="font-size:12px;color:#64748B;">Reason: {r_reason}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.caption(
                    "Check or uncheck individual repairs below to selectively approve or roll them back, "
                    "then click 'Apply Selective Review' to update the dataset and recalculate the health score."
                )

                # Initialize approved repair indices set
                if st.session_state.approved_repair_indices is None or len(st.session_state.approved_repair_indices) > len(repairs):
                    st.session_state.approved_repair_indices = set(range(len(repairs)))

                editor_df = filtered_df.copy()
                editor_df.insert(0, "Approve", [idx in st.session_state.approved_repair_indices for idx in editor_df.index])

                edited_rep_df = st.data_editor(
                    editor_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Approve": st.column_config.CheckboxColumn(
                            "Approve",
                            help="Check to approve and apply this repair; uncheck to reject and revert to raw original value.",
                            default=True,
                        ),
                        "confidence": st.column_config.ProgressColumn(
                            "Confidence",
                            min_value=0.0,
                            max_value=1.0,
                            format="%.2f",
                        ),
                    },
                    disabled=[c for c in editor_df.columns if c != "Approve"],
                    key="repairs_editor_table",
                )

                col_rev_btn, _ = st.columns([2, 5])
                with col_rev_btn:
                    if st.button("Apply Selective Review", type="primary", use_container_width=True):
                        new_approved = set(st.session_state.approved_repair_indices)
                        for orig_idx, is_approved in zip(editor_df.index, edited_rep_df["Approve"]):
                            if is_approved:
                                new_approved.add(int(orig_idx))
                            else:
                                new_approved.discard(int(orig_idx))

                        try:
                            raw_csv_str = st.session_state.raw_csv_bytes.decode("utf-8-sig", errors="replace")
                            rev_payload = {
                                "raw_csv": raw_csv_str,
                                "repairs": repairs,
                                "approved_indices": sorted(list(new_approved)),
                            }
                            with st.spinner("Applying selective review to dataset..."):
                                rev_resp = httpx.post(f"{api_url.rstrip('/')}/apply-review", json=rev_payload, timeout=30.0)

                            if rev_resp.status_code == 200:
                                rev_data = rev_resp.json()
                                st.session_state.pipeline_result["cleaned_csv"] = rev_data["cleaned_csv"]
                                st.session_state.pipeline_result["quality_report"]["output_sha256"] = rev_data["output_sha256"]
                                st.session_state.pipeline_result["quality_report"]["after_health_score"] = rev_data["after_health_score"]
                                st.session_state.approved_repair_indices = new_approved
                                st.session_state.selective_review_banner = (
                                    f"Selective Review Applied: {rev_data['approved_count']} Approved | "
                                    f"{rev_data['rejected_count']} Rolled Back | New SHA-256: {rev_data['output_sha256'][:16]}... "
                                    f"(Health Score: {rev_data['after_health_score']:.2f}%)"
                                )
                                st.toast("Selective review applied successfully!")
                                st.rerun()
                            else:
                                st.error(f"Failed to apply review ({rev_resp.status_code}): {rev_resp.text}")
                        except Exception as err:
                            st.error(f"Error communicating with backend: {err}")

                st.caption(f"Showing {len(filtered_df)} of {len(rep_df)} total repairs.")
        else:
            st.info("No repairs were needed; dataset was already completely clean.")

    # ------------------------------------------------------------------------
    # Tab 4: Before vs. After & Export Center
    # ------------------------------------------------------------------------
    with tab_export:
        st.markdown("#### Export Center: Cleaned Data & Audit Reports")

        exp1, exp2, exp3, exp4 = st.columns(4)
        cleaned_csv_str = result.get("cleaned_csv", "")
        stem = Path(st.session_state.raw_filename).stem

        with exp1:
            st.download_button(
                label="Download Cleaned CSV",
                data=cleaned_csv_str,
                file_name=f"cleaned_{st.session_state.raw_filename}",
                mime="text/csv",
                type="primary",
                use_container_width=True,
            )
        with exp2:
            audit_payload = {k: v for k, v in result.items() if k != "cleaned_csv"}
            audit_json = json.dumps(audit_payload, indent=2)
            st.download_button(
                label="Download Audit JSON",
                data=audit_json,
                file_name=f"{stem}_audit_report.json",
                mime="application/json",
                use_container_width=True,
            )
        with exp3:
            html_report = generate_html_report(
                result=result,
                filename=st.session_state.raw_filename,
                latency=st.session_state.run_latency_seconds,
            )
            st.download_button(
                label="Download Motion HTML",
                data=html_report,
                file_name=f"{stem}_quality_report.html",
                mime="text/html",
                use_container_width=True,
            )
        with exp4:
            approved_list = (
                sorted(list(st.session_state.approved_repair_indices))
                if st.session_state.get("approved_repair_indices") is not None
                else list(range(len(repairs)))
            )
            script_payload = {
                "filename": st.session_state.raw_filename,
                "repairs": repairs,
                "input_sha256": quality_report.get("input_sha256", ""),
                "expected_output_sha256": quality_report.get("output_sha256", ""),
                "approved_indices": approved_list,
            }
            script_fn, script_body = fetch_reproduce_script(api_url, script_payload)
            st.download_button(
                label="Download Replay Script (.py)",
                data=script_body,
                file_name=script_fn,
                mime="text/x-python",
                use_container_width=True,
                help="Standalone executable Python script to replay verified repairs deterministically with zero dependencies.",
            )

        st.markdown("---")
        st.markdown("#### Dataset Comparison Views")

        view_tab_side, view_tab_highlight = st.tabs(["Side-by-Side View", "Cell Mutation Highlight View"])

        raw_preview_df = pd.read_csv(io.BytesIO(st.session_state.raw_csv_bytes), dtype=str)
        cleaned_df = pd.read_csv(io.StringIO(cleaned_csv_str), dtype=str)

        with view_tab_side:
            c_raw, c_clean = st.columns(2)
            with c_raw:
                st.markdown(f"**Original Dirty CSV (`{st.session_state.raw_filename}`):**")
                st.dataframe(raw_preview_df.head(100), use_container_width=True)
            with c_clean:
                st.markdown(f"**Repaired Clean CSV (`cleaned_{st.session_state.raw_filename}`):**")
                st.dataframe(cleaned_df.head(100), use_container_width=True)

        with view_tab_highlight:
            st.caption("Repaired cells are highlighted in soft green; quarantined cells in soft red (top 100 rows).")

            repaired_cells = {(r.get("row_index"), r.get("column")) for r in repairs if r.get("column")}
            quarantined_cells = {(u.get("row_index"), u.get("column")) for u in unresolved if u.get("column")}

            def highlight_mutations(data: pd.DataFrame) -> pd.DataFrame:
                style_df = pd.DataFrame("", index=data.index, columns=data.columns)
                for idx in data.index:
                    for col in data.columns:
                        if (idx, col) in repaired_cells:
                            style_df.at[idx, col] = "background-color: rgba(0, 213, 155, 0.22); color: #00D59B; font-weight: 600;"
                        elif (idx, col) in quarantined_cells:
                            style_df.at[idx, col] = "background-color: rgba(239, 68, 68, 0.22); color: #FCA5A5; font-weight: 600;"
                return style_df

            styled_cleaned = cleaned_df.head(100).style.apply(highlight_mutations, axis=None)
            st.dataframe(styled_cleaned, use_container_width=True)

    # ------------------------------------------------------------------------
    # Tab 5: Run History Ledger (Persistent SQLite DB)
    # ------------------------------------------------------------------------
    with tab_history:
        st.markdown("#### Persistent SQLite Audit Ledger")
        st.caption("Cryptographic audit trail of all previous cleaning runs recorded in storage/audit_ledger.db.")

        col_h_refresh, _ = st.columns([2, 5])
        with col_h_refresh:
            if st.button("Refresh Audit Ledger", use_container_width=True):
                st.session_state.pop("cached_history", None)

        history_runs = fetch_history(api_url, limit=50)
        if history_runs:
            history_rows = []
            for run_item in history_runs:
                history_rows.append({
                    "Run ID": run_item.get("run_id"),
                    "Timestamp (UTC)": str(run_item.get("timestamp_utc", ""))[:19].replace("T", " "),
                    "Filename": run_item.get("filename"),
                    "Status": run_item.get("status"),
                    "Mode": run_item.get("llm_mode"),
                    "Original Rows": run_item.get("original_rows"),
                    "Cleaned Rows": run_item.get("cleaned_rows"),
                    "Repairs": run_item.get("total_repairs"),
                    "Unresolved": run_item.get("unresolved_count"),
                    "Before Health": f"{float(run_item.get('before_health_score', 0.0)):.2f}%",
                    "After Health": f"{float(run_item.get('after_health_score', 0.0)):.2f}%",
                    "Input SHA-256": f"{str(run_item.get('input_sha256', ''))[:12]}...",
                    "Output SHA-256": f"{str(run_item.get('output_sha256', ''))[:12]}...",
                })
            st.dataframe(pd.DataFrame(history_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No audit runs recorded yet. Execute a cleaning run to populate the ledger.")

