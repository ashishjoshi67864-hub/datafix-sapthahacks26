# 🛠️ DataFix — 100% Local Neuro-Symbolic Tabular Data Quality Workbench

> **SAPTHAHACKS'26 — INNOVEDGE PS 01**  
> **Team:** Utsava V, Ashish Joshi, Utsav Panduranga, Vanja Y  
> **Status:** 100% Complete • 44/44 Pytest Verification Suite Passing • Judge Ready

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2.6.1-E92063.svg?logo=pydantic&logoColor=white)](https://docs.pydantic.dev)
[![Ollama](https://img.shields.io/badge/Ollama-qwen2.5:3b-black.svg?logo=ollama&logoColor=white)](https://ollama.ai)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Pytest-44%20Passed-brightgreen.svg)](https://pytest.org)

---

## 📌 Executive Summary & Problem Statement

Modern enterprise data pipelines are plagued by dirty tabular data: malformed HTML scraps, unstandardized categorical entries, accidental duplicates, missing values, and impossible domain bounds. Traditional rule-based cleaners are rigid and brittle, while modern "LLM data cleaners" introduce catastrophic failure modes:

1. **Severe Privacy Breaches:** Uploading raw customer rows, phone numbers, and salaries to external LLM APIs violates GDPR, HIPAA, and enterprise data policies.
2. **Hallucination & Silent Corruption:** Unconstrained LLMs invent non-existent categories (e.g., standardizing `"BLR"` to `"Atlantis"`), mutate protected numerical identifiers (e.g., converting phone numbers to scientific notation), or overwrite domain anomalies with fabricated averages.
3. **Statistical Distortion:** Arbitrary LLM or mean imputations skew underlying probability distributions, invalidating downstream machine learning models.

**DataFix solves this with a 100% local, neuro-symbolic architecture:**
- **Symbolic Muscle (Deterministic Python):** Profiles datasets, enforces strict protected identifier boundaries, executes HTML/whitespace stripping, handles exact deduplication while preserving original 0-based indices, computes safe median/IQR numeric imputations, and statistically validates changes with a **Kolmogorov-Smirnov (KS) drift test** that triggers automatic rollback upon distribution divergence.
- **Neural Advisory (Local LLM Detective):** Runs locally via Ollama (`qwen2.5:3b`). The LLM **never** sees raw dataset rows—it receives only aggregated value frequencies and error counts. Every proposal is gated by strict Pydantic `extra="forbid"` contracts and verified against observed domain sets before any mutation occurs. If Ollama is offline or uninstalled, DataFix falls back seamlessly to deterministic heuristics with zero downtime.

---

## 🏛️ Neuro-Symbolic Two-Stage Architecture

```
                                  [ Uploaded CSV File ]
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │       Cryptographic Ingest & Hashing          │
                    │  • Compute 64-char Hex Input SHA-256 Digest   │
                    │  • Stable 0-Based Row Index Tracking          │
                    └───────────────────────┬───────────────────────┘
                                            │
                                            ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Deterministic Engine & Statistical Drift Guard (`cleaning_engine.py`)         │
├───────────────────────────────────────────────────────────────────────────────────────┤
│  1. Identifier Protection     : Auto-detects & freezes customer_id, phone, postal_code │
│  2. Deterministic Cleaning    : Regex HTML tag stripping & whitespace normalization   │
│  3. Exact Deduplication       : Drops identical rows while maintaining row indices    │
│  4. Safe Numeric Imputation   : Median/IQR imputation on non-protected numeric fields  │
│  5. KS-Test Drift Rollback    : Reverts numeric imputations if p-value < 0.05         │
└───────────────────────────────────────────┬───────────────────────────────────────────┘
                                            │
                                  Post-Stage 1 Clean DF
                                            │
                                            ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Local AI Detective & Metadata-Only Repairs (`ai_engine.py`)                  │
├───────────────────────────────────────────────────────────────────────────────────────┤
│  [ Privacy Boundary ] Send ONLY column frequencies & error counts (ZERO raw rows)     │
│                                           │                                           │
│         ┌─────────────────────────────────┴─────────────────────────────────┐         │
│         ▼                                                                   ▼         │
│  [ Ollama Online ]                                                [ Offline Fallback ]│
│  • qwen2.5:3b (local CPU/GPU)                                     • Deterministic     │
│  • Pydantic extra="forbid" schema                                   heuristics &      │
│  • Title-case & abbreviation map                                    lookup tables     │
│         │                                                                   │         │
│         └─────────────────────────────────┬─────────────────────────────────┘         │
│                                           │                                           │
│  [ Python Verification Gate ] Reject hallucinations (e.g. "Atlantis" -> BLR rejected) │
│  [ Functional Dependency Engine ] Cross-column imputation (e.g. 560001 -> Bengaluru)  │
│  [ Blast-Radius Quarantine ] Untouched cells with confidence < 0.60 flagged for review│
│  [ Root-Cause Synthesis ] 2-4 systemic diagnosis clusters with remediation actions    │
└───────────────────────────────────────────┬───────────────────────────────────────────┘
                                            │
                                            ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│ ORCHESTRATION & EXPORT (`pipeline.py`, `main.py`, `app.py`)                           │
├───────────────────────────────────────────────────────────────────────────────────────┤
│  • Quality Health Score Delta : Before Score -> After Score (0 - 100%)                │
│  • Cryptographic Output Hash  : Compute 64-char Hex Output SHA-256 Digest             │
│  • Stateless FastAPI Backend  : GET /health, POST /clean (in-memory, upload guards)   │
│  • Thin-Client Streamlit UI   : Zero duplicate logic; pure HTTP client to FastAPI     │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Verified Multi-Domain Benchmark Table

DataFix was evaluated across 4 multi-domain benchmark datasets representing enterprise HR, e-commerce, and healthcare domains. All repairs and scores are verified by automated tests:

| Dataset | Domain | Raw Rows | Cleaned Rows | Duplicates Dropped | Stage 1 Repairs | Stage 2 Repairs | Total Repairs | Quarantined Issues | Initial Health | Cleaned Health | Health Score Delta |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`sample_dirty.csv`** | General HR & Admin | 24 | 22 | 2 | 15 | 10 | 25 | 3 | 95.44% | **99.01%** | <span style="color:green">**+3.57%**</span> |
| **`dirty_ecommerce_orders.csv`** | Retail & Orders | 100 | 96 | 4 | 113 | 69 | 182 | 27 | 96.93% | **99.36%** | <span style="color:green">**+2.43%**</span> |
| **`dirty_hospital_records.csv`** | Healthcare & Patients | 100 | 96 | 4 | 113 | 74 | 187 | 20 | 97.02% | **99.57%** | <span style="color:green">**+2.55%**</span> |
| **`dirty_hr_payroll.csv`** | Enterprise Payroll | 100 | 96 | 4 | 122 | 59 | 181 | 23 | 96.52% | **99.41%** | <span style="color:green">**+2.89%**</span> |

> **Audit Guarantee:** In all 4 benchmarks, zero protected identifiers were modified, zero non-duplicate valid rows were dropped, and all original 0-based row indices were preserved exactly.

---

## ⚡ Quickstart Guide

### Prerequisites
- **Python:** 3.11.x (recommended)
- **Local LLM Daemon (Optional but Recommended):** [Ollama](https://ollama.ai)
  ```bash
  ollama pull qwen2.5:3b
  ```
  *(Note: If Ollama is not installed or running, DataFix automatically runs in its deterministic fallback mode with zero configuration).*

---

### Step 1: Clone & Setup Virtual Environment

#### Windows PowerShell:
```powershell
# Clone repository
git clone https://github.com/ashishjoshi67864-hub/datafix-sapthahacks26.git
cd datafix-sapthahacks26

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install pinned dependencies
pip install -r requirements.txt
```

#### macOS / Linux:
```bash
# Clone repository
git clone https://github.com/ashishjoshi67864-hub/datafix-sapthahacks26.git
cd datafix-sapthahacks26

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

---

### Step 2: Run Two-Terminal Launch

DataFix uses a clean client-server architecture: FastAPI handles all processing statelessly, while Streamlit acts as an interactive thin client.

#### Terminal 1 — Start FastAPI Backend:
```bash
# Windows
.\.venv\Scripts\uvicorn.exe main:app --host 127.0.0.1 --port 8000

# macOS / Linux
uvicorn main:app --host 127.0.0.1 --port 8000
```
*Backend runs at: `http://127.0.0.1:8000` (Interactive API docs at `http://127.0.0.1:8000/docs`)*

#### Terminal 2 — Start Streamlit Workbench:
```bash
# Windows
.\.venv\Scripts\streamlit.exe run app.py

# macOS / Linux
streamlit run app.py
```
*UI automatically opens in your browser at: `http://localhost:8501`*

---

### Step 3: Run the Automated Verification Suite

Run all 44 unit and integration tests across schemas, Stage 1 engine, KS statistical drift, Stage 2 AI guardrails, FastAPI endpoints, and Streamlit AST safety:

```bash
# Windows
.\.venv\Scripts\python.exe -m pytest -v

# macOS / Linux
python -m pytest -v
```

**Expected output:**
```text
======================= 44 passed, 5 warnings in 7.43s =======================
```

---

## 🔌 API Reference

The FastAPI service exposes lightweight, validated endpoints consuming and returning strict Pydantic v2 schemas:

### 1. `GET /`
Returns service metadata and documentation links.
```json
{
  "service": "DataFix API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health"
}
```

### 2. `GET /health`
Probes system readiness and local Ollama daemon reachability.
```json
{
  "status": "ok",
  "service": "datafix",
  "ollama": {
    "reachable": true,
    "model": "qwen2.5:3b",
    "model_available": true,
    "detail": "Ollama is reachable. Model 'qwen2.5:3b' available."
  },
  "supported_llm_modes": ["auto", "deterministic", "ollama"]
}
```

### 3. `POST /clean`
Stateless multipart file upload executing the complete neuro-symbolic repair pipeline.

- **Query Parameters:**
  - `llm_mode` *(optional, string)*: `"auto"` (default), `"deterministic"`, or `"ollama"`.
  - `model` *(optional, string)*: Custom Ollama model tag (defaults to `"qwen2.5:3b"`).
  - `skip_stage2` *(optional, boolean)*: Set `true` to run Stage 1 deterministic cleaning only.
- **Upload Guards:**
  - `400 Bad Request`: Non-`.csv` file extensions, empty byte payloads, or missing columns.
  - `413 Request Entity Too Large`: File uploads exceeding 20 MB.
- **Response Schema (`DataFixResponse`):**
  - `status`: `"completed"` or `"completed_stage1_only"`.
  - `quality_report`: Initial and cleaned Health Scores, 64-char input/output SHA-256 hashes, KS drift verification results.
  - `summary`: Original vs. cleaned row counts, duplicate counts, repairs categorized by stage.
  - `repairs`: Detailed list of every mutation (row index, column, original value, corrected value, anomaly type, repair method, confidence score, confidence tier, human-readable rationale).
  - `unresolved_issues`: Quarantined cells with confidence `< 0.60` left unmutated.
  - `root_cause_diagnoses`: Aggregated root-cause clusters explaining upstream ingestion failures.
  - `cleaned_csv`: Fully sanitized, UTF-8 CSV string ready for download.

---

## 🎬 2-Minute Live Judge Demo Script

Follow this script to demonstrate all key innovations of DataFix to judges in under 2 minutes:

| Time | Step | Action | What to Highlight to Judges |
|---|---|---|---|
| **0:00 - 0:20** | **1. Launch & Health** | Open `http://localhost:8501`. Note the sidebar panel. | **100% Local & Zero Cloud:** Point to "System & AI Health" confirming the FastAPI backend and Ollama detective are running entirely offline on your laptop. |
| **0:20 - 0:45** | **2. Load Dirty Dataset** | Under "Select Input Source", choose `sample_dirty.csv`. Select `deterministic` mode. Click **"🚀 Run DataFix Pipeline"**. | **Instant Execution (<1s):** Show execution speed and deterministic safety. Point out the **Health Score jump from 95.44% $\to$ 99.01%**, 25 total repairs, and 2 duplicates eliminated. |
| **0:45 - 1:10** | **3. Provenance & KS Drift** | Scroll to **"Dataset Cryptographic Provenance"** and **"Statistical KS-Drift Verification"**. | **Tamper-Evident Audit:** Point to the 64-character input vs. output SHA-256 digests. Show that numeric age/salary distributions passed the Kolmogorov-Smirnov test ($p > 0.05$), proving statistical integrity was maintained. |
| **1:10 - 1:35** | **4. Root-Cause & Quarantine** | Scroll to **"Systemic Root-Cause Diagnoses"** and **"Quarantined Issues"**. | **Explainability & Blast-Radius Safety:** Show the generated root-cause clusters (e.g. `RC-CATEGORICAL-DRIFT`, `RC-INGEST-RETRY`). Point to Row 15 (`age=-35`) and Row 22 (missing city/dept): explain that DataFix **refused to hallucinate**, preserving raw values and quarantining them at `confidence < 0.60`. |
| **1:35 - 2:00** | **5. Download & Audit Log** | Expand the **"Detailed Repair Audit Log"** and click **"⬇️ Download Cleaned CSV"**. | **Neuro-Symbolic Traceability:** Show how every cell repair records exact row index, anomaly type, before/after values, and confidence tier (`AUTO_SAFE` vs `REVIEW_RECOMMENDED`). |

---

## 🔒 Security, Privacy & Blast-Radius Guarantees

1. **Zero Raw-Row PII Transmission:**
   - Ollama prompts receive only aggregated `{value: frequency}` dictionaries for low-cardinality categorical columns. Names, emails, phone numbers, and raw record combinations are never sent to any LLM.
2. **Strict Schema Gatekeeping:**
   - All LLM responses are validated using Pydantic v2 models with `model_config = ConfigDict(extra="forbid")`. Unsolicited fields are immediately rejected.
3. **Python Verification of Proposed Repairs:**
   - Standardized values proposed by the LLM must match an already dominant observed value in the dataset. Hallucinations like `"Atlantis"` are detected and discarded.
4. **Blast-Radius Quarantine:**
   - When missing categorical cells lack cross-column evidence (e.g., Row 22 has no postal code to infer city), the cell is **not** guessed. It is preserved as null and quarantined with `confidence < 0.60` (`ConfidenceTier.QUARANTINED`).
5. **Deterministic Fallback Safety Net:**
   - If Ollama is offline, times out, or emits malformed text, DataFix falls back to rule-based heuristics without crashing, ensuring 100% pipeline reliability.

---

## 👥 Hackathon Team & Acknowledgments

Built with pride for **SAPTHAHACKS'26** (INNOVEDGE PS 01):
- **Utsava V**
- **Ashish Joshi**
- **Utsav Panduranga**
- **Vanja Y**

*DataFix — Autonomous, Neuro-Symbolic, 100% Local Data Quality Engineering.*
