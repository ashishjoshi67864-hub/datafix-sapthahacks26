# DataFix — 100% Local Neuro-Symbolic Tabular Data Quality Workbench

> Built for **SAPTHAHACKS'26** (INNOVEDGE PS 01) by **Utsava V, Ashish Joshi, Utsav Panduranga, and Vanja Y**

DataFix is a fully local, neuro-symbolic workbench for detecting, diagnosing, repairing, validating, explaining, and tracking quality issues in tabular (CSV) datasets. Every cell mutation is executed and validated deterministically by Python; the local LLM only advises — it never writes data.

---

## Pinned Tech Stack

| Layer | Technology | Version |
|---|---|---|
| API backend | FastAPI | 0.110.0 |
| ASGI server | Uvicorn | 0.28.0 |
| Data contracts | Pydantic v2 | 2.6.1 |
| Data engine | Pandas | 2.2.0 |
| Numerics | NumPy | 1.26.4 |
| ML / stats | Scikit-Learn | 1.4.1.post1 |
| HTML cleaning | BeautifulSoup4 | 4.12.3 |
| Local LLM client | Ollama Python SDK | 0.3.3 |
| UI workbench | Streamlit | 1.32.0 |
| HTTP client | Requests / HTTPX | latest / 0.27.2 |
| Testing | Pytest | latest |
| Python runtime | CPython | 3.11 (target) |

---

## 6-Stage Capability Mapping

| Stage | Capability | Primary Module |
|---|---|---|
| **DETECT** | Deterministic profiling: nulls, type violations, duplicates, outliers, HTML artifacts, encoding issues | `cleaning_engine.py` |
| **DIAGNOSE** | Local-LLM root-cause analysis on summarized metadata (never raw rows) with rule-based fallback | `ai_engine.py` |
| **REPAIR** | Confidence-tiered safe mutations executed entirely by deterministic Python | `cleaning_engine.py` |
| **VALIDATE** | KS-test drift rollback, post-repair re-profiling, identifier protection | `cleaning_engine.py` |
| **EXPLAIN** | Pydantic v2 response contracts: per-issue confidence, rationale, applied fix, unresolved issues | `schemas.py` |
| **TRACK** | SHA-256 dataset hashing, stable 0-based row indexing, full pipeline orchestration | `pipeline.py` |

> `main.py` exposes `/health` and `/clean` over HTTP; `app.py` renders the Streamlit UI by calling `main.py`.

---

## Strict AI Boundary

- **100% local** — no data leaves the machine. No cloud APIs. No telemetry.
- **Ollama model** — default `qwen2.5:7b`; fallback `qwen2.5:3b` (selectable via the `OLLAMA_MODEL` environment variable, configured in Step 6).
- **What the LLM receives** — only summarized error metadata and unique categorical value counts. **Raw dataset rows are never passed to the model.**
- **What the LLM never does** — it never writes, mutates, or validates data. All cell mutations are executed by deterministic Pandas/NumPy/SciPy code.
- **Deterministic fallback** — a rule-based fallback in `ai_engine.py` guarantees zero crashes and zero silent failures if Ollama is offline or returns a malformed response.

---

## Thin-Client UI Architecture

`app.py` (Streamlit) ──HTTP──▶ `main.py` (FastAPI / Uvicorn)

The UI contains **zero** duplicated cleaning logic. All data processing happens exclusively inside the FastAPI backend, keeping the workbench stateless and testable independently.

---

## Folder Structure

```
datafix/                    ← repository root
├── schemas.py              ← Pydantic v2 contracts  (EXPLAIN + TRACK)
├── cleaning_engine.py      ← Stage 1 deterministic engine (DETECT + REPAIR + VALIDATE)
├── ai_engine.py            ← Stage 2 Ollama detective  (DIAGNOSE + EXPLAIN)
├── pipeline.py             ← Orchestrator            (DETECT → TRACK)
├── main.py                 ← FastAPI backend          (/health + /clean)
├── app.py                  ← Streamlit thin client
├── tests/
│   └── __init__.py         ← Test suite placeholder
├── requirements.txt        ← Pinned dependency manifest
├── .gitignore
├── README.md
└── plan.md                 ← Detailed 11-step development plan
```

---

## Virtual Environment Activation

**PowerShell**
```powershell
.\.venv\Scripts\Activate.ps1
```

**CMD**
```cmd
.venv\Scripts\activate.bat
```

**Run the backend** (after activation)
```powershell
uvicorn main:app --reload --port 8000
```

**Run the UI** (separate terminal, after activation)
```powershell
streamlit run app.py
```

---

## Development Roadmap

| Step | Description | Status |
|---|---|---|
| **Step 1** | Project initialization: workspace, venv, pinned deps, docs, initial commit | [ ] |
| **Step 2** | Pydantic v2 schemas: `DataFixRequest`, quality reports, diagnoses, `DataFixResponse` | [ ] |
| **Step 3** | Stage 1 cleaning engine: null imputation, type coercion, dedup, outlier detection, HTML stripping | [ ] |
| **Step 4** | KS-test drift detection and rollback guard | [ ] |
| **Step 5** | Stage 2 AI engine: Ollama summarized-metadata prompting and rule-based fallback | [ ] |
| **Step 6** | Pipeline orchestrator: Stage 1 → Stage 2 → validation, SHA-256 hashing, row-index stability | [ ] |
| **Step 7** | FastAPI backend: `/health` and `/clean` endpoints with Pydantic v2 contracts | [ ] |
| **Step 8** | Streamlit thin-client UI: upload, results display, download repaired CSV | [ ] |
| **Step 9** | Pytest suite: schemas, Stage 1 engine, KS rollback, pipeline fallback, FastAPI routes | [ ] |
| **Step 10** | End-to-end integration test with a real dirty CSV sample | [ ] |
| **Step 11** | Hardening: logging, error boundaries, `OLLAMA_MODEL` env var, production README | [ ] |

> **Honest status note**: As of Step 1, none of the application logic listed above has been implemented. All `.py` files contain only module-level docstrings.
