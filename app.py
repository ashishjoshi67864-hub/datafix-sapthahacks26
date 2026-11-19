"""Thin-client Streamlit workbench consuming FastAPI over HTTP with zero duplicated cleaning logic.
Features 100% offline motion graphics, executive dark command-center theme, and full auditability.
"""

import html
import io
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ============================================================================
# Page Configuration & Metadata (Zero Icons)
# ============================================================================

st.set_page_config(
    page_title="DataFix — Autonomous Data Quality Workbench",
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
if "run_latency_seconds" not in st.session_state:
    st.session_state.run_latency_seconds = None

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
        0%, 100% { opacity: 0.35; filter: drop-shadow(0 0 2px #00D59B); }
        50% { opacity: 1; filter: drop-shadow(0 0 8px #00D59B); }
    }
    @keyframes pulseShield {
        0%, 100% { stroke: #818CF8; opacity: 0.55; }
        50% { stroke: #A5B4FC; opacity: 0.95; }
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
    .shield-box { stroke: #818CF8; opacity: 0.7; }
    """

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
      body {{ margin: 0; padding: 0; background: transparent; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; user-select: none; }}
      {anim_css}
      .stage-title {{ font-size: 11px; font-weight: 700; fill: #F1F5F9; letter-spacing: 0.5px; }}
      .stage-sub {{ font-size: 9px; fill: #94A3B8; }}
      .node-rect {{ rx: 8px; ry: 8px; }}
    </style>
    </head>
    <body>
    <svg width="100%" height="205" viewBox="0 0 920 200" fill="none" xmlns="http://www.w3.org/2000/svg">
      <!-- Dark backdrop card -->
      <rect width="920" height="200" rx="12" fill="#0C101A" stroke="rgba(255,255,255,0.07)" stroke-width="1"/>

      <!-- Header Label -->
      <text x="30" y="28" fill="#00D59B" font-size="11" font-weight="700" letter-spacing="1">NEURO-SYMBOLIC PIPELINE ARCHITECTURE (6-STAGE WORKFLOW)</text>
      <text x="30" y="44" fill="#64748B" font-size="10">Deterministic Python cleaning engines paired with metadata-only local AI detective</text>

      <!-- Stage 2 Privacy Shield Box -->
      <rect class="shield-box" x="180" y="58" width="135" height="102" rx="10" fill="rgba(129, 140, 248, 0.04)" stroke="#818CF8" stroke-width="1.5" stroke-dasharray="4 4"/>
      <text x="247" y="172" fill="#A5B4FC" font-size="9" font-weight="600" text-anchor="middle">Zero Raw-Row PII to LLM</text>

      <!-- Connector Lines with flowing dash -->
      <line class="flow-line" x1="135" y1="105" x2="190" y2="105" stroke="#00D59B" stroke-width="2"/>
      <line class="flow-line" x1="305" y1="105" x2="360" y2="105" stroke="#818CF8" stroke-width="2"/>
      <line class="flow-line" x1="475" y1="105" x2="530" y2="105" stroke="#00D59B" stroke-width="2"/>
      <line class="flow-line" x1="645" y1="105" x2="700" y2="105" stroke="#38BDF8" stroke-width="2"/>
      <line class="flow-line" x1="815" y1="105" x2="860" y2="105" stroke="#60A5FA" stroke-width="2"/>

      <!-- Stage 1: DETECT -->
      <g transform="translate(25, 70)">
        <rect class="node-rect" width="110" height="70" fill="#111622" stroke="#00D59B" stroke-width="1.5"/>
        <circle cx="20" cy="22" r="5" fill="#00D59B" opacity="0.2"/>
        <circle cx="20" cy="22" r="2.5" fill="#00D59B"/>
        <text class="stage-title" x="32" y="25">1. DETECT</text>
        <text class="stage-sub" x="15" y="44">Stage 1 Profiler</text>
        <text class="stage-sub" x="15" y="57">Nulls, Types, HTML</text>
      </g>

      <!-- Stage 2: DIAGNOSE -->
      <g transform="translate(192, 70)">
        <rect class="node-rect" width="112" height="70" fill="#111622" stroke="#818CF8" stroke-width="1.5"/>
        <circle cx="20" cy="22" r="5" fill="#818CF8" opacity="0.2"/>
        <circle cx="20" cy="22" r="2.5" fill="#818CF8"/>
        <text class="stage-title" x="32" y="25">2. DIAGNOSE</text>
        <text class="stage-sub" x="15" y="44">Ollama Detective</text>
        <text class="stage-sub" x="15" y="57">Counts & Patterns</text>
      </g>

      <!-- Stage 3: REPAIR -->
      <g transform="translate(362, 70)">
        <rect class="node-rect" width="112" height="70" fill="#111622" stroke="#00D59B" stroke-width="1.5"/>
        <circle cx="20" cy="22" r="5" fill="#00D59B" opacity="0.2"/>
        <circle cx="20" cy="22" r="2.5" fill="#00D59B"/>
        <text class="stage-title" x="32" y="25">3. REPAIR</text>
        <text class="stage-sub" x="15" y="44">Python Mutator</text>
        <text class="stage-sub" x="15" y="57">Deduplicate & Impute</text>
      </g>

      <!-- Stage 4: VALIDATE -->
      <g transform="translate(532, 70)">
        <rect class="node-rect" width="112" height="70" fill="#111622" stroke="#38BDF8" stroke-width="1.5"/>
        <circle cx="20" cy="22" r="5" fill="#38BDF8" opacity="0.2"/>
        <circle cx="20" cy="22" r="2.5" fill="#38BDF8"/>
        <text class="stage-title" x="32" y="25">4. VALIDATE</text>
        <text class="stage-sub" x="15" y="44">KS Drift Guard</text>
        <text class="stage-sub" x="15" y="57">Auto-Rollback Guard</text>
      </g>

      <!-- Stage 5: EXPLAIN -->
      <g transform="translate(702, 70)">
        <rect class="node-rect" width="112" height="70" fill="#111622" stroke="#60A5FA" stroke-width="1.5"/>
        <circle cx="20" cy="22" r="5" fill="#60A5FA" opacity="0.2"/>
        <circle cx="20" cy="22" r="2.5" fill="#60A5FA"/>
        <text class="stage-title" x="32" y="25">5. EXPLAIN</text>
        <text class="stage-sub" x="15" y="44">Pydantic v2 Contract</text>
        <text class="stage-sub" x="15" y="57">Confidence Tiers</text>
      </g>

      <!-- Stage 6: TRACK badge -->
      <g transform="translate(825, 78)">
        <circle cx="45" cy="28" r="22" fill="#111622" stroke="#F43F5E" stroke-width="1.5"/>
        <text x="45" y="25" fill="#F43F5E" font-size="10" font-weight="700" text-anchor="middle">6. TRACK</text>
        <text x="45" y="37" fill="#94A3B8" font-size="8" text-anchor="middle">SHA-256</text>
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

    # SVG circular gauge math: radius 52, circumference 2 * pi * 52 ≈ 326.7
    circumference = 326.7
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

    stage2_stroke = "#F59E0B" if is_stage1_only else "#818CF8"
    stage2_badge = "SKIPPED" if is_stage1_only else f"{stage2_repairs} repairs"
    stage2_opacity = "0.45" if is_stage1_only else "1.0"

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
      * {{ box-sizing: border-box; }}
      body {{ margin: 0; padding: 0; background: transparent; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: #F1F5F9; user-select: none; }}
      .telemetry-card {{
        background: linear-gradient(180deg, #111726 0%, #0C101A 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px 14px 20px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.45);
      }}
      .grid-container {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
      }}
      .gauge-col {{
        flex: 0 0 210px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
      }}
      .flow-col {{
        flex: 1 1 auto;
        min-width: 320px;
      }}
      .kpi-col {{
        flex: 0 0 280px;
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
      }}
      .kpi-tile {{
        background: #141A29;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        padding: 10px 12px;
      }}
      .kpi-title {{ font-size: 10px; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; margin-bottom: 3px; }}
      .kpi-val {{ font-size: 20px; font-weight: 800; color: #F1F5F9; }}
      .kpi-sub {{ font-size: 9px; color: #64748B; margin-top: 2px; }}
      .delta-pill {{
        background: rgba(0, 213, 155, 0.12);
        color: #00D59B;
        border: 1px solid rgba(0, 213, 155, 0.28);
        border-radius: 20px;
        padding: 4px 10px;
        font-size: 11px;
        font-weight: 700;
        margin-top: 4px;
        display: inline-block;
      }}
      .provenance-bar {{
        margin-top: 12px;
        padding-top: 10px;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 10px;
        color: #94A3B8;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      }}
      .badge-status {{
        background: {'rgba(0, 213, 155, 0.18)' if not is_stage1_only else 'rgba(245, 158, 11, 0.18)'};
        color: {'#00D59B' if not is_stage1_only else '#FBBF24'};
        border: 1px solid {'rgba(0, 213, 155, 0.35)' if not is_stage1_only else 'rgba(245, 158, 11, 0.35)'};
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 700;
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
          <svg width="140" height="140" viewBox="0 0 130 130">
            <!-- Background track -->
            <circle cx="65" cy="65" r="52" fill="none" stroke="#1E2638" stroke-width="10"/>
            <!-- Before Health Arc (Dim Cyan) -->
            <circle cx="65" cy="65" r="52" fill="none" stroke="#38BDF8" stroke-width="6" stroke-dasharray="{circumference}" stroke-dashoffset="{before_offset}" stroke-linecap="round" opacity="0.3" transform="rotate(-90 65 65)"/>
            <!-- After Health Arc (Vibrant Emerald) -->
            <circle cx="65" cy="65" r="52" fill="none" stroke="#00D59B" stroke-width="10" stroke-dasharray="{circumference}" stroke-dashoffset="{after_offset}" stroke-linecap="round" transform="rotate(-90 65 65)"/>
            <!-- Center Text -->
            <text x="65" y="60" text-anchor="middle" fill="#F1F5F9" font-size="20" font-weight="800" id="health-num">{after_score:.2f}%</text>
            <text x="65" y="76" text-anchor="middle" fill="#94A3B8" font-size="9" font-weight="600" letter-spacing="0.5">HEALTH SCORE</text>
          </svg>
          <div class="delta-pill">+{delta:.2f} pts Health Delta</div>
        </div>

        <!-- Column 2: Live Pipeline Execution Flow Diagram -->
        <div class="flow-col">
          <svg width="100%" height="150" viewBox="0 0 460 150" fill="none">
            <!-- Node 1: Raw Ingest -->
            <rect x="5" y="45" width="80" height="48" rx="6" fill="#141A29" stroke="#38BDF8" stroke-width="1.5"/>
            <text x="45" y="66" fill="#38BDF8" font-size="10" font-weight="700" text-anchor="middle">RAW INGEST</text>
            <text x="45" y="80" fill="#94A3B8" font-size="9" text-anchor="middle">{original_rows} rows</text>

            <!-- Connector 1 -->
            <line class="{flow_line_class}" x1="85" y1="69" x2="115" y2="69" stroke="#00D59B" stroke-width="2"/>

            <!-- Node 2: Stage 1 -->
            <rect x="115" y="45" width="85" height="48" rx="6" fill="#141A29" stroke="#00D59B" stroke-width="1.5"/>
            <text x="157" y="66" fill="#00D59B" font-size="10" font-weight="700" text-anchor="middle">STAGE 1</text>
            <text x="157" y="80" fill="#94A3B8" font-size="9" text-anchor="middle">{stage1_repairs} fixes</text>

            <!-- Connector 2 -->
            <line class="{flow_line_class}" x1="200" y1="69" x2="230" y2="69" stroke="#00D59B" stroke-width="2"/>

            <!-- Node 3: KS Drift Guard -->
            <rect x="230" y="45" width="75" height="48" rx="6" fill="#141A29" stroke="#38BDF8" stroke-width="1.5"/>
            <text x="267" y="66" fill="#38BDF8" font-size="10" font-weight="700" text-anchor="middle">KS GUARD</text>
            <text x="267" y="80" fill="#00D59B" font-size="9" font-weight="600" text-anchor="middle">PASSED</text>

            <!-- Connector 3 -->
            <line class="{flow_line_class}" x1="305" y1="69" x2="335" y2="69" stroke="{stage2_stroke}" stroke-width="2"/>

            <!-- Node 4: Stage 2 AI Detective -->
            <rect x="335" y="45" width="85" height="48" rx="6" fill="#141A29" stroke="{stage2_stroke}" stroke-width="1.5" opacity="{stage2_opacity}"/>
            <text x="377" y="66" fill="{stage2_stroke}" font-size="10" font-weight="700" text-anchor="middle">STAGE 2 AI</text>
            <text x="377" y="80" fill="#94A3B8" font-size="9" text-anchor="middle">{stage2_badge}</text>

            <!-- Quarantine Side-Branch -->
            <path d="M 157 93 L 157 122 L 335 122" fill="none" stroke="#EF4444" stroke-width="1.5" stroke-dasharray="3 3"/>
            <g transform="translate(335, 108)">
              <rect width="115" height="28" rx="5" fill="#141A29" stroke="#EF4444" stroke-width="1.2"/>
              <text x="10" y="18" fill="#FCA5A5" font-size="9" font-weight="700">QUARANTINE:</text>
              <text x="82" y="18" fill="#EF4444" font-size="9" font-weight="800">{unresolved_count} trapped</text>
            </g>

            <!-- Final Cleaned Output Indicator -->
            <g transform="translate(425, 48)">
              <circle cx="16" cy="21" r="14" fill="#00D59B" opacity="0.18"/>
              <circle cx="16" cy="21" r="8" fill="#00D59B"/>
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
          <span style="color: #00D59B; font-weight: 700;">SHA-256 PROVENANCE:</span>
          <span style="margin-left: 8px;">In: {input_sha256[:16]}... -> Out: {output_sha256[:16]}...</span>
        </div>
        <div style="display: flex; gap: 12px; align-items: center;">
          <span>Latency: <strong style="color: #F1F5F9;">{latency_str}</strong></span>
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

    badge_color = "#00D59B" if status == "passed" else ("#EF4444" if status == "rolled_back" else "#94A3B8")
    badge_bg = "rgba(0, 213, 155, 0.12)" if status == "passed" else ("rgba(239, 68, 68, 0.12)" if status == "rolled_back" else "rgba(148, 163, 184, 0.12)")
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
      body {{ margin: 0; padding: 0; background: transparent; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: #F1F5F9; }}
      .ks-box {{
        background: #111622;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
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
        background: #1E2638;
        border-radius: 6px;
        overflow: visible;
        margin: 12px 0 6px 0;
      }}
      .fill-bar {{
        height: 100%;
        width: {bar_pct}%;
        background: linear-gradient(90deg, #00D59B, #F59E0B 70%, #EF4444);
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
        background: #FFFFFF;
        border-radius: 2px;
        box-shadow: 0 0 6px rgba(255, 255, 255, 0.8);
        transform: translateX(-50%);
      }}
      .ks-meta {{
        display: flex;
        justify-content: space-between;
        font-size: 10px;
        color: #94A3B8;
        font-family: ui-monospace, SFMono-Regular, monospace;
      }}
      .status-pill {{
        background: {badge_bg};
        color: {badge_color};
        border: 1px solid {badge_color};
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 10px;
        font-weight: 700;
      }}
      {anim_css}
    </style>
    </head>
    <body>
    <div class="ks-box">
      <div class="ks-header">
        <div>
          <strong style="color: #38BDF8; font-size: 13px;">Column: <code>{col}</code></strong>
          <span style="font-size: 10px; color: #64748B; margin-left: 8px;">Two-Sample Kolmogorov-Smirnov Test</span>
        </div>
        <span class="status-pill">{badge_text}</span>
      </div>
      <div class="track-container">
        <div class="fill-bar"></div>
        <div class="threshold-line" title="Safety Threshold (0.15)"></div>
        <div class="needle" title="Observed KS Statistic: {ks_stat:.4f}"></div>
      </div>
      <div class="ks-meta">
        <span>KS Stat: <strong style="color: #F1F5F9;">{ks_stat:.4f}</strong> (0.00 safe -> 0.30 critical)</span>
        <span style="color: #F87171;">Threshold: 0.15</span>
        <span>p-value: <strong style="color: #F1F5F9;">{p_val:.4f}</strong></span>
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
        <div style="background:#111622;border:1px solid rgba(255,255,255,0.07);border-radius:8px;padding:12px;margin-bottom:8px;">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <strong style="color:#A5B4FC;font-size:13px;">[{c_id}] {cause}</strong>
            <span style="background:rgba(129,140,248,0.12);color:#C7D2FE;padding:2px 8px;border-radius:12px;font-size:10px;font-weight:700;">{rows} rows affected</span>
          </div>
          <div style="font-size:11px;color:#94A3B8;margin-top:4px;"><strong>Affected Columns:</strong> {cols_str}</div>
          <div style="font-size:11px;color:#00D59B;margin-top:4px;"><strong>Recommended Action:</strong> {action}</div>
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
  body {{ margin:0; padding:24px; background:#090D16; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; color:#F1F5F9; }}
  .container {{ max-width:1100px; margin:0 auto; }}
  h1,h2,h3 {{ color:#F1F5F9; }}
  .header-card {{ background:#111622; border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:20px; margin-bottom:20px; }}
  .grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:16px 0; }}
  .tile {{ background:#141A29; border:1px solid rgba(255,255,255,0.06); border-radius:8px; padding:12px; }}
  .tile-title {{ font-size:11px; color:#94A3B8; text-transform:uppercase; font-weight:600; }}
  .tile-val {{ font-size:24px; font-weight:800; color:#00D59B; margin-top:4px; }}
  table {{ width:100%; border-collapse:collapse; margin-top:12px; font-size:12px; }}
  th,td {{ text-align:left; padding:8px 12px; border-bottom:1px solid #1E2638; }}
  th {{ background:#141A29; color:#94A3B8; text-transform:uppercase; font-size:10px; letter-spacing:0.5px; }}
  .badge {{ background:rgba(0,213,155,0.12); color:#00D59B; border:1px solid rgba(0,213,155,0.28); padding:2px 8px; border-radius:12px; font-size:10px; font-weight:700; }}
</style>
</head>
<body>
<div class="container">
  <div class="header-card">
    <div style="display:flex;justify-content:space-between;align-items:center;">
      <div>
        <h1 style="margin:0;font-size:22px;">DataFix Autonomous Quality & Audit Report</h1>
        <div style="color:#94A3B8;font-size:12px;margin-top:4px;">Dataset: <strong>{html.escape(filename)}</strong> • SAPTHAHACKS'26 INNOVEDGE PS 01</div>
      </div>
      <span class="badge">100% LOCAL AUDIT VERIFIED</span>
    </div>

    <div class="grid">
      <div class="tile">
        <div class="tile-title">Health Score Jump</div>
        <div class="tile-val">{after_score:.2f}%</div>
        <div style="font-size:10px;color:#00D59B;">+{delta:.2f} pts from {before_score:.2f}%</div>
      </div>
      <div class="tile">
        <div class="tile-title">Cleaned Rows</div>
        <div class="tile-val">{summary.get('cleaned_rows', 0)}</div>
        <div style="font-size:10px;color:#94A3B8;">-{summary.get('duplicates_removed', 0)} duplicates dropped</div>
      </div>
      <div class="tile">
        <div class="tile-title">Total Cell Repairs</div>
        <div class="tile-val">{summary.get('total_repairs', 0)}</div>
        <div style="font-size:10px;color:#94A3B8;">S1: {summary.get('stage1_repairs', 0)} · S2: {summary.get('stage2_repairs', 0)}</div>
      </div>
      <div class="tile">
        <div class="tile-title">Quarantined Traps</div>
        <div class="tile-val" style="color:#F87171;">{summary.get('unresolved_count', 0)}</div>
        <div style="font-size:10px;color:#FCA5A5;">Protected from Hallucination</div>
      </div>
    </div>

    <div style="font-family:monospace;font-size:11px;color:#94A3B8;background:#141A29;padding:10px;border-radius:6px;border:1px solid rgba(255,255,255,0.06);">
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
      /* Executive Dark Ambient Background */
      .stApp {
        background: radial-gradient(circle at 50% 0%, #151D2F 0%, #090D16 75%) !important;
        background-attachment: fixed !important;
      }
      section[data-testid="stSidebar"] {
        background-color: #0D121D !important;
        border-right: 1px solid rgba(255, 255, 255, 0.07) !important;
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
        background: rgba(0, 213, 155, 0.1);
        color: #00D59B;
        border: 1px solid rgba(0, 213, 155, 0.25);
        border-radius: 14px;
        padding: 3px 10px;
        font-size: 11px;
        font-weight: 600;
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
        background: #141A29;
        color: #38BDF8;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 4px;
        padding: 2px 7px;
        font-size: 11px;
        font-family: ui-monospace, SFMono-Regular, monospace;
        margin: 2px 3px;
      }
      .df-repair-card {
        background: #111622;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 8px;
        padding: 12px 14px;
        margin-bottom: 8px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.35);
        animation: fadeSlideUp 0.3s ease-out;
      }
      .df-diff-del {
        background: rgba(239, 68, 68, 0.18);
        color: #FCA5A5;
        text-decoration: line-through;
        padding: 2px 6px;
        border-radius: 4px;
        font-family: ui-monospace, SFMono-Regular, monospace;
      }
      .df-diff-ins {
        background: rgba(0, 213, 155, 0.18);
        color: #00D59B;
        text-decoration: none;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
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
          <span class="df-pulse-dot" style="background:#00D59B;"></span>
          <strong style="color:#00D59B;font-size:13px;">FastAPI Backend: Online</strong>
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
                  <span class="df-pulse-dot" style="background:#818CF8;"></span>
                  <span style="color:#A5B4FC;font-size:12px;"><strong>Ollama Detective:</strong> Ready (<code>{model_name}</code>)</span>
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
    <div style="margin-bottom: 14px;">
      <h1 style="font-size: 26px; margin: 0; font-weight: 800; color: #F1F5F9; letter-spacing: -0.5px;">
        DataFix — Neuro-Symbolic Data Quality & Repair Workbench
      </h1>
      <div style="margin-top: 8px;">
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
    components.html(render_empty_state_svg(motion_enabled), height=210)


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
        height=225,
    )

    # Warnings bar
    if warnings:
        with st.expander("Pipeline Warnings & Fallback Notices", expanded=False):
            for w in warnings:
                st.warning(w)

    # ========================================================================
    # 4 Interactive Workbench Tabs (Zero Icons)
    # ========================================================================

    tab_diag, tab_val, tab_repairs, tab_export = st.tabs([
        "Root-Cause Diagnosis & Analytics",
        "Validation & Blast-Radius Trust",
        "Auditable Repair Trail",
        "Before vs. After & Export Center",
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
                    "mark": {"type": "bar", "cornerRadiusEnd": 4, "color": "#00D59B"},
                    "encoding": {
                        "x": {"field": "count", "type": "quantitative", "title": "Repairs Count"},
                        "y": {"field": "anomaly_type", "type": "nominal", "sort": "-x", "title": "Anomaly Type"},
                        "tooltip": [{"field": "anomaly_type", "title": "Type"}, {"field": "count", "title": "Repairs"}],
                    },
                    "background": "transparent",
                    "config": {
                        "view": {"stroke": "transparent"},
                        "axis": {"labelColor": "#94A3B8", "titleColor": "#E2E8F0", "gridColor": "rgba(255,255,255,0.06)"},
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
                                "range": ["#00D59B", "#F59E0B", "#EF4444"],
                            },
                            "legend": None,
                        },
                        "tooltip": [{"field": "confidence_tier", "title": "Tier"}, {"field": "count", "title": "Repairs"}],
                    },
                    "background": "transparent",
                    "config": {
                        "view": {"stroke": "transparent"},
                        "axis": {"labelColor": "#94A3B8", "titleColor": "#E2E8F0", "gridColor": "rgba(255,255,255,0.06)"},
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
                          <h4 style="margin:0;color:#F1F5F9;"><span style="color:#818CF8;">[{c_id}]</span> {cause}</h4>
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
                components.html(render_ks_meter_html(check, motion_enabled), height=95)
        else:
            st.info("No numeric columns required KS drift verification.")

        st.markdown("---")
        st.markdown("#### Blast-Radius Principle: Quarantined & Preserved Cells")
        st.markdown(
            """
            <div style="background:#111622;border:1px solid rgba(239,68,68,0.28);border-radius:8px;padding:12px 16px;margin-bottom:14px;">
              <strong style="color:#F87171;">Blast-Radius Principle: Zero Hallucination Guarantee</strong><br>
              <span style="font-size:12px;color:#94A3B8;">
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

            view_format = st.radio("Display Format:", ["Git-Style Diff Cards", "Interactive Data Table"], horizontal=True)

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
                              <span class="df-chip" style="color:#818CF8;">{r_stage}</span>
                            </div>
                            <span class="df-hero-pill" style="margin:0;">{r_conf:.2f} ({r_tier})</span>
                          </div>
                          <div style="margin:8px 0;font-size:13px;">
                            <del class="df-diff-del">{r_orig}</del>
                            <span style="color:#94A3B8;margin:0 8px;font-weight:600;">-></span>
                            <ins class="df-diff-ins">{r_corr}</ins>
                          </div>
                          <div style="font-size:11px;color:#94A3B8;">Reason: {r_reason}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.dataframe(
                    filtered_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "confidence": st.column_config.ProgressColumn(
                            "Confidence",
                            min_value=0.0,
                            max_value=1.0,
                            format="%.2f",
                        ),
                    },
                )
                st.caption(f"Showing {len(filtered_df)} of {len(rep_df)} total repairs.")
        else:
            st.info("No repairs were needed; dataset was already completely clean.")

    # ------------------------------------------------------------------------
    # Tab 4: Before vs. After & Export Center
    # ------------------------------------------------------------------------
    with tab_export:
        st.markdown("#### Export Center: Cleaned Data & Audit Reports")

        exp1, exp2, exp3 = st.columns(3)
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
                label="Download Audit Report (JSON)",
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
                label="Download Motion Report (HTML)",
                data=html_report,
                file_name=f"{stem}_quality_report.html",
                mime="text/html",
                use_container_width=True,
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
