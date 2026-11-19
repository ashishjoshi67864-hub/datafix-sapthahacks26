"""Fast, deterministic, offline-safe integration test suite for Stage 2 (ai_engine.py),
end-to-end orchestration (pipeline.py), FastAPI endpoints (main.py), and Streamlit
thin-client frontend safety (app.py).
"""

import ast
import io
import json
import re
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from ai_engine import Stage2Result, check_ollama_health, run_stage2_semantic_cleaning
from cleaning_engine import load_csv, run_stage1_cleaning
from main import app as fastapi_app
from pipeline import compute_sha256, run_datafix_pipeline
from schemas import (
    AnomalyType,
    CleaningStage,
    ConfidenceTier,
    DataFixResponse,
    PipelineStatus,
    RepairItem,
    RepairMethod,
    RootCauseDiagnosis,
    UnresolvedIssue,
)

SAMPLE_DIRTY_PATH = Path("sample_dirty.csv")


# ============================================================================
# a) Stage 2 Semantic Cleaning, Privacy & Guardrails (ai_engine.py)
# ============================================================================

def test_stage2_standardization_imputation_and_quarantine():
    """Verifies Stage 2 deterministic standardization, cross-column FD imputation,

    and strict quarantine of unbacked missing cells on sample_dirty.csv.
    """
    df = load_csv(SAMPLE_DIRTY_PATH)
    stage1_res = run_stage1_cleaning(df)
    stage2_res = run_stage2_semantic_cleaning(stage1_res, llm_mode="deterministic")

    cleaned_df = stage2_res.cleaned_df

    # 1. City variants (BLR, Banglore, Mumbay, BOM, New Dehli) repaired to Bengaluru, Mumbai, Delhi
    assert cleaned_df.loc[2, "city"] == "Bengaluru"
    assert cleaned_df.loc[5, "city"] == "Bengaluru"
    assert cleaned_df.loc[9, "city"] == "Mumbai"
    assert cleaned_df.loc[13, "city"] == "Mumbai"
    assert cleaned_df.loc[19, "city"] == "Delhi"

    # Verify repair item tracking for cities
    city_repairs = {
        r.row_index: r for r in stage2_res.repairs
        if r.column == "city" and r.anomaly_type in (AnomalyType.CATEGORICAL_INCONSISTENCY, AnomalyType.SPELLING_ERROR)
    }
    assert 2 in city_repairs and city_repairs[2].corrected_value == "Bengaluru"
    assert 5 in city_repairs and city_repairs[5].corrected_value == "Bengaluru"
    assert 9 in city_repairs and city_repairs[9].corrected_value == "Mumbai"
    assert 13 in city_repairs and city_repairs[13].corrected_value == "Mumbai"
    assert 19 in city_repairs and city_repairs[19].corrected_value == "Delhi"

    # 2. Department typos (Enginering, Markting, Finanse) repaired to Engineering, Marketing, Finance
    assert cleaned_df.loc[4, "department"] == "Engineering"
    assert cleaned_df.loc[13, "department"] == "Marketing"
    assert cleaned_df.loc[21, "department"] == "Finance"

    dept_repairs = {
        r.row_index: r for r in stage2_res.repairs
        if r.column == "department" and r.anomaly_type == AnomalyType.SPELLING_ERROR
    }
    assert 4 in dept_repairs and dept_repairs[4].corrected_value == "Engineering"
    assert 13 in dept_repairs and dept_repairs[13].corrected_value == "Marketing"
    assert 21 in dept_repairs and dept_repairs[21].corrected_value == "Finance"

    # 3. Missing city imputation on Row 12 (560001 -> Bengaluru) and Row 21 (400001 -> Mumbai)
    assert cleaned_df.loc[12, "city"] == "Bengaluru"
    assert cleaned_df.loc[21, "city"] == "Mumbai"

    impute_repairs = {
        r.row_index: r for r in stage2_res.repairs
        if r.anomaly_type == AnomalyType.MISSING_CATEGORICAL
    }
    assert 12 in impute_repairs
    assert impute_repairs[12].column == "city"
    assert impute_repairs[12].corrected_value == "Bengaluru"
    assert impute_repairs[12].repair_method == RepairMethod.AI_CATEGORICAL_IMPUTE
    assert impute_repairs[12].confidence >= 0.90
    assert impute_repairs[12].confidence_tier == ConfidenceTier.AUTO_SAFE

    assert 21 in impute_repairs
    assert impute_repairs[21].column == "city"
    assert impute_repairs[21].corrected_value == "Mumbai"
    assert impute_repairs[21].repair_method == RepairMethod.AI_CATEGORICAL_IMPUTE
    assert impute_repairs[21].confidence >= 0.90
    assert impute_repairs[21].confidence_tier == ConfidenceTier.AUTO_SAFE

    # 4. Row 22 (city & department missing without evidence) stay NaN untouched and quarantined
    assert pd.isna(cleaned_df.loc[22, "city"])
    assert pd.isna(cleaned_df.loc[22, "department"])

    row22_unres = [u for u in stage2_res.unresolved_issues if u.row_index == 22]
    assert len(row22_unres) == 2
    row22_cols = {u.column for u in row22_unres}
    assert row22_cols == {"city", "department"}
    for u in row22_unres:
        assert u.confidence < 0.60
        assert u.confidence_tier == ConfidenceTier.QUARANTINED
        assert u.anomaly_type == AnomalyType.MISSING_CATEGORICAL

    # 5. Row 15 age=-35 stays untouched, protected columns untouched, stable 0-based indices
    assert float(cleaned_df.loc[15, "age"]) == -35.0
    for col in stage1_res.protected_identifier_columns:
        assert list(cleaned_df[col]) == list(stage1_res.cleaned_df[col])

    expected_indices = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23]
    assert list(cleaned_df.index) == expected_indices

    # 6. len(root_cause_diagnoses) >= 2
    assert len(stage2_res.root_cause_diagnoses) >= 2
    for diag in stage2_res.root_cause_diagnoses:
        assert isinstance(diag, RootCauseDiagnosis)
        assert diag.cluster_id.startswith("RC-")
        assert len(diag.likely_cause) > 0
        assert len(diag.recommended_action) > 0


def test_stage2_ollama_prompt_privacy_zero_pii_leak():
    """Verifies that outgoing Ollama prompt payloads contain strictly aggregated

    frequencies and zero raw row PII from sample_dirty.csv.
    """
    df = load_csv(SAMPLE_DIRTY_PATH)
    stage1_res = run_stage1_cleaning(df)

    captured_prompts = []

    def mock_chat(*args, **kwargs) -> Dict[str, Any]:
        prompt = kwargs.get("messages", [{}])[0].get("content", "")
        captured_prompts.append(prompt)
        if "standardization" in prompt.lower():
            return {
                "message": {
                    "content": json.dumps(
                        {
                            "column": "city",
                            "mappings": [
                                {
                                    "original_value": "BLR",
                                    "standardized_value": "Bengaluru",
                                    "reason": "airport code",
                                }
                            ],
                        }
                    )
                }
            }
        elif "imputation" in prompt.lower():
            return {"message": {"content": json.dumps({"proposals": []})}}
        else:
            return {
                "message": {
                    "content": json.dumps(
                        {
                            "diagnoses": [
                                {
                                    "cluster_id": "RC-TEST",
                                    "affected_columns": ["city"],
                                    "likely_cause": "manual input",
                                    "affected_rows": 5,
                                    "recommended_action": "use enum dropdown",
                                }
                            ]
                        }
                    )
                }
            }

    mock_client = MagicMock()
    mock_client.list.return_value = {"models": [{"name": "qwen2.5:3b"}]}
    mock_client.chat.side_effect = mock_chat

    with patch("ollama.Client", return_value=mock_client):
        stage2_res = run_stage2_semantic_cleaning(stage1_res, llm_mode="ollama")

    assert len(captured_prompts) >= 1
    assert stage2_res.ollama_used is True

    # Assert ZERO raw PII appears anywhere in any captured prompt
    raw_pii_list = [
        "Aarav Sharma",
        "Diya Patel",
        "Rohan Mehta",
        "Kavita Krishnan",
        "Arjun Singh",
        "Vikram Malhotra",
        "Rahul Verma",
        "Sneha Rao",
        "Neha Gupta",
        "Pooja Hegde",
        "Amitabh Sen",
        "Priya Nair",
        "Rajesh Pillai",
        "Sunita Pillai",
        "Nikhil Reddy",
        "Deepak Joshi",
        "Meera Iyer",
        "Sanjay Saxena",
        "Kiran Kulkarni",
        "Ritu Bhatia",
        "Ananya Deshmukh",
        "Suresh Menon",
        "9876543210",
        "9876543211",
        "9876543212",
        "9876543213",
        "9876543214",
        "9876543215",
    ]

    all_prompts_text = "\n".join(captured_prompts)
    for pii in raw_pii_list:
        assert pii not in all_prompts_text, f"Raw PII '{pii}' leaked into Ollama prompt!"


def test_stage2_rejects_hallucinated_llm_category_and_invalid_json():
    """Verifies Python verification gate rejects hallucinated categories and

    malformed non-JSON outputs without crashing Stage 2.
    """
    df = load_csv(SAMPLE_DIRTY_PATH)
    stage1_res = run_stage1_cleaning(df)

    # 1. Hostile Response: Hallucinated category ("BLR" -> "Atlantis")
    mock_client_hallucinate = MagicMock()
    mock_client_hallucinate.list.return_value = {"models": [{"name": "qwen2.5:3b"}]}
    mock_client_hallucinate.chat.return_value = {
        "message": {
            "content": json.dumps(
                {
                    "column": "city",
                    "mappings": [
                        {
                            "original_value": "BLR",
                            "standardized_value": "Atlantis",
                            "reason": "mythical city",
                        }
                    ],
                }
            )
        }
    }

    with patch("ollama.Client", return_value=mock_client_hallucinate):
        res1 = run_stage2_semantic_cleaning(stage1_res, llm_mode="ollama")

    # Verification gate must reject "Atlantis"; fallback maps "BLR" -> "Bengaluru"
    assert "Atlantis" not in res1.cleaned_df["city"].values
    assert res1.cleaned_df.loc[2, "city"] == "Bengaluru"

    # 2. Hostile Response: Malformed / non-JSON output ("not valid json")
    mock_client_bad_json = MagicMock()
    mock_client_bad_json.list.return_value = {"models": [{"name": "qwen2.5:3b"}]}
    mock_client_bad_json.chat.return_value = {
        "message": {"content": "not valid json"}
    }

    with patch.dict("os.environ", {"DATAFIX_ENABLE_LLM_CACHE": "0"}):
        with patch("ollama.Client", return_value=mock_client_bad_json):
            res2 = run_stage2_semantic_cleaning(stage1_res, llm_mode="ollama")

    # Must not crash, records warning, and fallback successfully repairs data
    assert isinstance(res2, Stage2Result)
    assert len(res2.repairs) > 0
    assert any("Ollama standardization failed" in w for w in res2.warnings)
    assert res2.cleaned_df.loc[2, "city"] == "Bengaluru"


def test_stage2_offline_fallback_when_ollama_unreachable():
    """Verifies that an unreachable Ollama daemon falls back cleanly to deterministic

    heuristics with zero unhandled exceptions and logs a warning.
    """
    df = load_csv(SAMPLE_DIRTY_PATH)
    stage1_res = run_stage1_cleaning(df)

    with patch("ollama.Client", side_effect=ConnectionError("Ollama daemon down")):
        stage2_res = run_stage2_semantic_cleaning(stage1_res, llm_mode="auto")

    assert stage2_res.ollama_used is False
    assert stage2_res.mode_used == "deterministic_fallback"
    assert any("Ollama inactive" in w or "Ollama connection error" in w for w in stage2_res.warnings)
    assert len(stage2_res.repairs) == 10
    assert len(stage2_res.unresolved_issues) == 2


# ============================================================================
# b) End-to-End Pipeline & Cryptographic Provenance (pipeline.py)
# ============================================================================

def test_pipeline_end_to_end_and_sha256_provenance():
    """Verifies end-to-end pipeline execution on sample_dirty.csv: cryptographic

    SHA-256 provenance, health delta, repair counts, exact JSON round-trip, and clean CSV.
    """
    resp = run_datafix_pipeline(SAMPLE_DIRTY_PATH, llm_mode="deterministic")

    # Status & SHA-256 digests
    assert resp.status == PipelineStatus.COMPLETED
    assert len(resp.quality_report.input_sha256) == 64
    assert len(resp.quality_report.output_sha256) == 64
    assert re.fullmatch(r"[0-9a-f]{64}", resp.quality_report.input_sha256)
    assert re.fullmatch(r"[0-9a-f]{64}", resp.quality_report.output_sha256)
    assert resp.quality_report.input_sha256 != resp.quality_report.output_sha256

    # Health score improvement
    assert resp.quality_report.after_health_score > resp.quality_report.before_health_score

    # Summary metrics
    assert resp.summary.original_rows == 24
    assert resp.summary.cleaned_rows == 22
    assert resp.summary.duplicates_removed == 2
    assert resp.summary.stage1_repairs == 15
    assert resp.summary.stage2_repairs == 10
    assert resp.summary.total_repairs == resp.summary.stage1_repairs + resp.summary.stage2_repairs
    assert resp.summary.unresolved_count == len(resp.unresolved_issues)

    # Exact JSON serialization round-trip
    json_str = resp.model_dump_json()
    round_trip_resp = DataFixResponse.model_validate_json(json_str)
    assert round_trip_resp == resp

    # Cleaned CSV reload verification
    reloaded_df = pd.read_csv(io.StringIO(resp.cleaned_csv))
    assert len(reloaded_df) == 22
    assert not any("Unnamed" in c for c in reloaded_df.columns)
    assert list(reloaded_df.columns) == list(resp.summary.columns)


def test_pipeline_skip_stage2_and_fault_isolation():
    """Verifies skip_stage2 returns COMPLETED_STAGE1_ONLY, and simulated Stage 2

    crashes are fault-isolated without impacting Stage 1 repairs.
    """
    # 1. Explicit skip_stage2=True
    resp1 = run_datafix_pipeline(SAMPLE_DIRTY_PATH, skip_stage2=True)
    assert resp1.status == PipelineStatus.COMPLETED_STAGE1_ONLY
    assert resp1.summary.stage2_repairs == 0
    assert resp1.summary.stage1_repairs == 15
    assert any("Stage 2 semantic AI skipped by request" in w for w in resp1.warnings)

    # 2. Simulated Stage 2 crash fault isolation
    with patch("pipeline.run_stage2_semantic_cleaning", side_effect=RuntimeError("Simulated Stage 2 crash")):
        resp2 = run_datafix_pipeline(SAMPLE_DIRTY_PATH)

    assert resp2.status == PipelineStatus.COMPLETED_STAGE1_ONLY
    assert resp2.summary.stage1_repairs == 15
    assert resp2.summary.stage2_repairs == 0
    assert any("Simulated Stage 2 crash" in w for w in resp2.warnings)


def test_pipeline_empty_csv_raises_value_error():
    """Verifies that attempting to run pipeline on an empty byte buffer raises ValueError."""
    with pytest.raises(ValueError, match="empty"):
        run_datafix_pipeline(b"")


@pytest.mark.parametrize(
    "csv_path",
    [
        "sample_data/dirty_ecommerce_orders.csv",
        "sample_data/dirty_hospital_records.csv",
        "sample_data/dirty_hr_payroll.csv",
    ],
)
def test_pipeline_multidomain_benchmarks(csv_path: str):
    """Verifies that all three multi-domain benchmark dirty datasets execute successfully

    through the pipeline with health score improvements and deduplication.
    """
    resp = run_datafix_pipeline(csv_path, llm_mode="deterministic")
    assert resp.status == PipelineStatus.COMPLETED
    assert resp.quality_report.after_health_score > resp.quality_report.before_health_score
    assert resp.summary.duplicates_removed > 0
    assert isinstance(resp, DataFixResponse)


# ============================================================================
# c) FastAPI Service (main.py via TestClient)
# ============================================================================

def test_api_root_and_health():
    """Verifies GET / metadata and GET /health with mocked Ollama check."""
    client = TestClient(fastapi_app)

    # 1. GET /
    root_resp = client.get("/")
    assert root_resp.status_code == 200
    assert "service" in root_resp.json()
    assert root_resp.json()["service"] == "DataFix API"

    # 2. GET /health with mocked check_ollama_health
    mock_health = {
        "reachable": True,
        "model": "qwen2.5:3b",
        "model_available": True,
        "detail": "Ollama is reachable.",
    }
    with patch("main.check_ollama_health", return_value=mock_health):
        health_resp = client.get("/health")

    assert health_resp.status_code == 200
    health_json = health_resp.json()
    assert health_json["status"] == "ok"
    assert health_json["ollama"] == mock_health


def test_api_clean_valid_and_skip_stage2():
    """Verifies POST /clean endpoint with valid CSV upload and skip_stage2 query param."""
    client = TestClient(fastapi_app)
    csv_bytes = SAMPLE_DIRTY_PATH.read_bytes()

    # 1. Full clean with deterministic LLM mode
    resp = client.post(
        "/clean?llm_mode=deterministic",
        files={"file": ("sample_dirty.csv", csv_bytes, "text/csv")},
    )
    assert resp.status_code == 200
    data = DataFixResponse.model_validate(resp.json())
    assert data.status == PipelineStatus.COMPLETED
    assert data.summary.cleaned_rows == 22
    assert data.summary.total_repairs == 25

    # 2. Skip Stage 2 query parameter
    resp_skip = client.post(
        "/clean?skip_stage2=true",
        files={"file": ("sample_dirty.csv", csv_bytes, "text/csv")},
    )
    assert resp_skip.status_code == 200
    data_skip = DataFixResponse.model_validate(resp_skip.json())
    assert data_skip.status == PipelineStatus.COMPLETED_STAGE1_ONLY
    assert data_skip.summary.stage2_repairs == 0


def test_api_clean_upload_guards():
    """Verifies HTTP upload guards: non-CSV rejection (400), empty file (400),

    and oversized upload exceeding 20 MB (413).
    """
    client = TestClient(fastapi_app)

    # 1. Non-CSV extension -> 400
    r_txt = client.post(
        "/clean",
        files={"file": ("notes.txt", b"id,name\n1,alice", "text/plain")},
    )
    assert r_txt.status_code == 400
    assert "Only .csv files are supported" in r_txt.json()["detail"]

    # 2. Empty CSV -> 400
    r_empty = client.post(
        "/clean",
        files={"file": ("empty.csv", b"", "text/csv")},
    )
    assert r_empty.status_code == 400
    assert "empty" in r_empty.json()["detail"].lower()

    # 3. Oversized file (> 20 MB) -> 413
    oversized_data = b"x" * (20 * 1024 * 1024 + 1024)
    r_large = client.post(
        "/clean",
        files={"file": ("huge.csv", oversized_data, "text/csv")},
    )
    assert r_large.status_code == 413
    assert "exceeds" in r_large.json()["detail"].lower()


# ============================================================================
# d) Streamlit Frontend Thin-Client Safety (app.py)
# ============================================================================

def test_frontend_thin_client_ast_and_render():
    """Verifies that app.py is a pure thin client without business logic imports

    and renders cleanly with zero uncaught exceptions via AppTest.
    """
    # 1. AST Inspection: Zero disallowed imports
    app_path = Path("app.py")
    tree = ast.parse(app_path.read_text(encoding="utf-8"), filename="app.py")
    disallowed = {"cleaning_engine", "ai_engine", "pipeline", "ollama"}
    imported_modules = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_modules.add(node.module.split(".")[0])

    forbidden = imported_modules.intersection(disallowed)
    assert not forbidden, f"app.py violates thin-client architecture by importing: {forbidden}"

    # 2. AppTest render check with mocked backend health
    mock_health_resp = {
        "status": "ok",
        "service": "datafix",
        "ollama": {"reachable": False, "model": "qwen2.5:3b", "model_available": False},
        "supported_llm_modes": ["auto", "deterministic", "ollama"],
    }
    with patch("httpx.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_health_resp
        at = AppTest.from_file("app.py").run()

    assert len(at.exception) == 0, f"AppTest rendered with exceptions: {at.exception}"



# ============================================================================
# e) Reproducibility, Idempotency, Clean-CSV, and UTF-8 BOM Tests
# ============================================================================

def test_pipeline_exact_reproducibility_sha256():
    """Running run_datafix_pipeline twice on sample_dirty.csv (deterministic) produces
    identical input_sha256, output_sha256, cleaned_csv, and total repair count.
    """
    csv_bytes = SAMPLE_DIRTY_PATH.read_bytes()

    result1 = run_datafix_pipeline(csv_bytes, filename="sample_dirty.csv", llm_mode="deterministic")
    result2 = run_datafix_pipeline(csv_bytes, filename="sample_dirty.csv", llm_mode="deterministic")

    assert result1.quality_report.input_sha256 == result2.quality_report.input_sha256, \
        "Input SHA-256 must be identical across two runs of the same input"
    assert result1.quality_report.output_sha256 == result2.quality_report.output_sha256, \
        "Output SHA-256 must be identical (deterministic output)"
    assert result1.cleaned_csv == result2.cleaned_csv, \
        "Cleaned CSV content must be byte-for-byte identical across two deterministic runs"
    assert result1.summary.total_repairs == result2.summary.total_repairs, \
        "Total repair count must be identical across two deterministic runs"


def test_pipeline_already_clean_csv_zero_corruption():
    """Passing a perfectly clean CSV produces 0 repairs, 0 unresolved, 0 duplicates,
    before_health_score == after_health_score == 100.0, and no cell value is altered.
    """
    clean_csv = "customer_id,name,city,age\n101,Alice,Mumbai,30\n102,Bob,Delhi,25\n103,Carol,Chennai,35\n"
    csv_bytes = clean_csv.encode("utf-8")

    result = run_datafix_pipeline(csv_bytes, filename="clean.csv", llm_mode="deterministic")

    assert result.summary.total_repairs == 0, \
        f"Expected 0 repairs on a clean CSV, got {result.summary.total_repairs}"
    assert result.summary.unresolved_count == 0, \
        f"Expected 0 unresolved issues on a clean CSV, got {result.summary.unresolved_count}"
    assert result.summary.duplicates_removed == 0, \
        f"Expected 0 duplicates on a clean CSV, got {result.summary.duplicates_removed}"
    assert result.quality_report.before_health_score == 100.0, \
        f"Expected before_health_score=100.0, got {result.quality_report.before_health_score}"
    assert result.quality_report.after_health_score == 100.0, \
        f"Expected after_health_score=100.0, got {result.quality_report.after_health_score}"

    # Verify no cell values were altered
    import io
    cleaned_df = pd.read_csv(io.StringIO(result.cleaned_csv), dtype=str)
    original_df = pd.read_csv(io.StringIO(clean_csv), dtype=str)
    assert cleaned_df.shape == original_df.shape, "Shape must not change on a clean CSV"
    for col in original_df.columns:
        for idx in original_df.index:
            assert str(original_df.at[idx, col]) == str(cleaned_df.at[idx, col]), \
                f"Cell [{idx}, {col}] was unexpectedly mutated on a clean CSV"


def test_pipeline_idempotency_on_reclean():
    """Re-running run_datafix_pipeline on the cleaned_csv output of sample_dirty.csv
    introduces 0 new repairs, proving DataFix never over-mutates already-repaired data.
    """
    import io as _io
    csv_bytes = SAMPLE_DIRTY_PATH.read_bytes()

    # First pass
    result1 = run_datafix_pipeline(csv_bytes, filename="sample_dirty.csv", llm_mode="deterministic")
    cleaned_bytes = result1.cleaned_csv.encode("utf-8")

    # Second pass on the already-cleaned output
    result2 = run_datafix_pipeline(cleaned_bytes, filename="cleaned.csv", llm_mode="deterministic")

    assert result2.summary.total_repairs == 0, (
        f"Idempotency violation: re-cleaning an already-cleaned CSV introduced "
        f"{result2.summary.total_repairs} new repairs: "
        f"{[(r.column, r.original_value, r.corrected_value) for r in result2.repairs[:5]]}"
    )
    assert result2.summary.duplicates_removed == 0, \
        f"Re-cleaning introduced unexpected duplicate removal: {result2.summary.duplicates_removed}"


def test_pipeline_utf8_bom_excel_csv():
    """Passing a UTF-8 BOM (EF BB BF) Excel-exported CSV strips the BOM cleanly
    so that identifier detection (customer_id) and repairs work identically to
    the BOM-free equivalent.
    """
    # Build identical CSV content with and without BOM
    base_csv = "customer_id,name,city,age\n101,Alice,Mumbai,30\n102,Bob,  Delhi  ,25\n103,Carol,Mumbai,35\n"
    bom_csv_bytes = b"\xef\xbb\xbf" + base_csv.encode("utf-8")
    plain_csv_bytes = base_csv.encode("utf-8")

    result_bom = run_datafix_pipeline(bom_csv_bytes, filename="excel_export.csv", llm_mode="deterministic")
    result_plain = run_datafix_pipeline(plain_csv_bytes, filename="plain.csv", llm_mode="deterministic")

    # BOM must be stripped: first column must be "customer_id", not "\ufeffcustomer_id"
    assert "customer_id" in result_bom.summary.columns, (
        f"BOM not stripped: columns are {result_bom.summary.columns}"
    )
    bom_col = next((c for c in result_bom.summary.columns if "\ufeff" in c), None)
    assert bom_col is None, f"BOM character still present in column name: {bom_col!r}"

    # Protected identifier detection must work identically regardless of BOM
    assert "customer_id" in result_bom.summary.protected_identifier_columns, \
        "customer_id must be detected as protected identifier even from BOM CSV"

    # Repair counts must be identical (BOM should not affect cleaning logic)
    assert result_bom.summary.total_repairs == result_plain.summary.total_repairs, (
        f"Repair count differs between BOM ({result_bom.summary.total_repairs}) "
        f"and plain ({result_plain.summary.total_repairs}) CSV"
    )


# ============================================================================
# f) High-Value Rubric: Calibrated Health Score, IsolationForest, Near-Dups, & 10k Scale
# ============================================================================

def test_health_score_honest_calibration_on_dirty_benchmarks():
    """Verifies that the calibrated health score reflects row-level defect density:
    dirty benchmarks score honestly (< 80.0%), show dramatic improvements (> +15 pts),
    and clean data achieves a perfect 100.0%.
    """
    # 1. sample_dirty.csv
    res_sample = run_datafix_pipeline("sample_dirty.csv", llm_mode="deterministic")
    assert res_sample.quality_report.before_health_score < 80.0, (
        f"Expected sample_dirty before score < 80%, got {res_sample.quality_report.before_health_score}"
    )
    assert res_sample.quality_report.after_health_score > res_sample.quality_report.before_health_score + 15.0, (
        f"Expected > 15 pt improvement, got {res_sample.quality_report.health_delta}"
    )

    # 2. dirty_ecommerce_orders.csv
    res_ecom = run_datafix_pipeline("sample_data/dirty_ecommerce_orders.csv", llm_mode="deterministic")
    assert res_ecom.quality_report.before_health_score < 80.0, (
        f"Expected ecommerce before score < 80%, got {res_ecom.quality_report.before_health_score}"
    )
    assert res_ecom.quality_report.after_health_score > res_ecom.quality_report.before_health_score + 15.0, (
        f"Expected > 15 pt improvement, got {res_ecom.quality_report.health_delta}"
    )

    # 3. Clean CSV achieves 100.0%
    clean_csv = "id,name,city,age\n1,Alice,Mumbai,25\n2,Bob,Delhi,30\n3,Charlie,Pune,35\n"
    res_clean = run_datafix_pipeline(clean_csv.encode("utf-8"), filename="clean.csv", llm_mode="deterministic")
    assert res_clean.quality_report.before_health_score == 100.0
    assert res_clean.quality_report.after_health_score == 100.0


def test_multivariate_isolation_forest_catches_unseen_schema_outlier():
    """Unseen schema: 40-row sensor dataset (sensor_id, temp_c, pressure_kpa, flow_rate)
    with a severe multivariate outlier at row 25 (temp=950, pressure=12000, flow=85000).
    IsolationForest must flag and quarantine it without mutating the underlying cells.
    """
    rows = ["sensor_id,temp_c,pressure_kpa,flow_rate"]
    for i in range(1, 41):
        if i == 25:
            # Extreme multivariate anomaly
            rows.append(f"S{i:03d},950.0,12000.0,85000.0")
        else:
            # Normal distribution around temp=25, pressure=101.3, flow=50
            t = 24.0 + (i % 5) * 0.4
            p = 100.0 + (i % 4) * 0.5
            f = 48.0 + (i % 6) * 0.7
            rows.append(f"S{i:03d},{t:.1f},{p:.1f},{f:.1f}")

    csv_data = "\n".join(rows) + "\n"
    resp = run_datafix_pipeline(csv_data.encode("utf-8"), filename="sensors.csv", llm_mode="deterministic")

    # Find the multivariate anomaly in unresolved_issues
    outlier_issues = [
        iss for iss in resp.unresolved_issues
        if iss.anomaly_type == AnomalyType.MULTIVARIATE_OUTLIER
    ]
    assert len(outlier_issues) >= 1, "IsolationForest must detect the multivariate sensor outlier"
    target_issue = next((iss for iss in outlier_issues if iss.row_index == 24), None)
    assert target_issue is not None, f"Row 24 (S025) must be flagged as multivariate outlier, found {outlier_issues}"
    assert target_issue.confidence_tier == ConfidenceTier.QUARANTINED
    assert target_issue.confidence < 0.60

    # Ensure zero destructive mutation on the flagged row
    df_clean = pd.read_csv(io.StringIO(resp.cleaned_csv))
    row_25_cleaned = df_clean[df_clean["sensor_id"] == "S025"]
    assert not row_25_cleaned.empty, "Row S025 must not be dropped"
    assert float(row_25_cleaned["temp_c"].iloc[0]) == 950.0, "Cell value must remain untouched"


def test_near_duplicate_row_quarantined_without_silent_drop():
    """Verifies that fuzzy near-duplicate records (e.g. identical phone/key with minor name typo)
    are quarantined in unresolved_issues without being silently dropped from the cleaned CSV.
    """
    csv_data = (
        "customer_id,name,phone,city\n"
        "101,Alice Smith,9876543210,Mumbai\n"
        "102,Alice Smyth,9876543210,Mumbai\n"
    )
    resp = run_datafix_pipeline(csv_data.encode("utf-8"), filename="near_dup.csv", llm_mode="deterministic")

    # Check unresolved issues contains NEAR_DUPLICATE
    near_dup_issues = [
        iss for iss in resp.unresolved_issues
        if iss.anomaly_type == AnomalyType.NEAR_DUPLICATE
    ]
    assert len(near_dup_issues) >= 1, "Near duplicate record must be logged in unresolved_issues"
    assert near_dup_issues[0].confidence_tier == ConfidenceTier.QUARANTINED
    assert near_dup_issues[0].confidence < 0.60
    assert "similarity" in near_dup_issues[0].reason

    # Check that both rows are preserved (never silently dropped)
    df_clean = pd.read_csv(io.StringIO(resp.cleaned_csv))
    assert len(df_clean) == 2, f"Both rows must be preserved, got {len(df_clean)} rows"
    names = set(df_clean["name"])
    assert "Alice Smith" in names and "Alice Smyth" in names, "Neither row should be deleted"


def test_scalability_10000_rows_throughput():
    """Generates an in-memory 10,050-row synthetic dataset with dirty artifacts
    (whitespace, HTML tags, duplicate rows, missing numerical values) and verifies
    that run_datafix_pipeline finishes in < 10.0 seconds with status COMPLETED.
    """
    import time

    base_cities = ["Mumbai", "Bengaluru", "Delhi", "Chennai", "Hyderabad"]
    lines = ["id,customer_name,city,salary"]
    for i in range(10000):
        c = base_cities[i % len(base_cities)]
        if i % 100 == 0:
            lines.append(f"{i},<b>User_{i}</b>,  {c}  ,")
        else:
            lines.append(f"{i},User_{i},{c},{30000 + (i % 50000)}")
    # Add 50 duplicate rows matching rows 1 to 50
    for i in range(1, 51):
        lines.append(f"{i},User_{i},{base_cities[i % len(base_cities)]},{30000 + (i % 50000)}")

    csv_bytes = "\n".join(lines).encode("utf-8")

    start_time = time.perf_counter()
    resp = run_datafix_pipeline(csv_bytes, filename="scale_10k.csv", llm_mode="deterministic")
    duration = time.perf_counter() - start_time

    assert resp.status == PipelineStatus.COMPLETED
    assert resp.summary.original_rows == 10050
    assert resp.summary.duplicates_removed == 50
    assert resp.summary.cleaned_rows == 10000
    assert resp.summary.total_repairs >= 100  # HTML tags and whitespace repaired
    assert resp.quality_report.after_health_score >= resp.quality_report.before_health_score
    assert duration < 10.0, f"Pipeline exceeded 10.0s SLA on 10k rows (took {duration:.2f}s)"


# ============================================================================
# g) Phase B Tests: SQLite Audit Ledger, Selective Review, Replay Script, & LLM Cache
# ============================================================================

def test_sqlite_audit_ledger_and_history_endpoints(tmp_path, monkeypatch):
    """Verifies persistent SQLite audit ledger integration: POST /clean automatically
    records run metadata, GET /history lists the run, and GET /history/{run_id} returns
    detailed run data (and 404 for unknown run_id).
    """
    test_db = tmp_path / "test_audit_ledger.db"
    monkeypatch.setenv("DATAFIX_LEDGER_DB", str(test_db))

    client = TestClient(fastapi_app)
    csv_bytes = SAMPLE_DIRTY_PATH.read_bytes()

    # 1. Clean dataset via POST /clean
    clean_resp = client.post(
        "/clean?llm_mode=deterministic",
        files={"file": ("sample_dirty.csv", csv_bytes, "text/csv")},
    )
    assert clean_resp.status_code == 200
    resp_data = clean_resp.json()
    in_sha = resp_data["quality_report"]["input_sha256"]
    out_sha = resp_data["quality_report"]["output_sha256"]

    # 2. Query GET /history
    hist_resp = client.get("/history?limit=10")
    assert hist_resp.status_code == 200
    runs = hist_resp.json().get("runs", [])
    assert len(runs) >= 1
    target_run = runs[0]
    assert target_run["filename"] == "sample_dirty.csv"
    assert target_run["input_sha256"] == in_sha
    assert target_run["output_sha256"] == out_sha
    assert target_run["total_repairs"] == resp_data["summary"]["total_repairs"]
    run_id = target_run["run_id"]

    # 3. Query GET /history/{run_id}
    detail_resp = client.get(f"/history/{run_id}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["run_id"] == run_id
    assert "response_summary" in detail
    assert detail["response_summary"]["summary"]["total_repairs"] == resp_data["summary"]["total_repairs"]

    # 4. Unknown run_id returns 404
    err_resp = client.get("/history/non_existent_run_id_123")
    assert err_resp.status_code == 404


def test_selective_review_rollback_reverts_rejected_repair():
    """Verifies interactive Git-style selective repair rollback: POST /apply-review
    allows rejecting individual repairs, reverting only the rejected cells back to
    their raw original values while applying all approved repairs.
    """
    client = TestClient(fastapi_app)
    csv_bytes = SAMPLE_DIRTY_PATH.read_bytes()

    # 1. First run /clean to get repairs list
    clean_resp = client.post(
        "/clean?llm_mode=deterministic",
        files={"file": ("sample_dirty.csv", csv_bytes, "text/csv")},
    )
    assert clean_resp.status_code == 200
    resp_data = clean_resp.json()
    repairs = resp_data["repairs"]
    assert len(repairs) > 0

    # Find first cell-level repair to selectively reject
    cell_idx = next(i for i, r in enumerate(repairs) if r["column"] is not None)
    rejected_repair = repairs[cell_idx]
    target_row = rejected_repair["row_index"]
    target_col = rejected_repair["column"]
    target_raw_val = rejected_repair["original_value"]

    # Approve all repairs EXCEPT the chosen one
    approved_indices = [i for i in range(len(repairs)) if i != cell_idx]

    review_payload = {
        "raw_csv": csv_bytes.decode("utf-8-sig"),
        "repairs": repairs,
        "approved_indices": approved_indices,
    }
    rev_resp = client.post("/apply-review", json=review_payload)
    assert rev_resp.status_code == 200
    rev_data = rev_resp.json()

    assert rev_data["rejected_count"] == 1
    assert rev_data["approved_count"] == len(repairs) - 1
    assert rev_data["output_sha256"] != resp_data["quality_report"]["output_sha256"]

    # Verify that the rejected cell reverted to its raw original value in the returned CSV
    df_reviewed = pd.read_csv(io.StringIO(rev_data["cleaned_csv"]), dtype=str)
    assert df_reviewed.at[target_row, target_col] == str(target_raw_val)


def test_reproduce_fix_script_executes_and_matches_sha256(tmp_path):
    """Verifies that the standalone replay script generated by POST /reproduce-script
    is fully executable with standard Python, executes without errors, and reproduces
    the exact output SHA-256 byte-for-byte.
    """
    import subprocess
    import sys

    client = TestClient(fastapi_app)
    csv_bytes = SAMPLE_DIRTY_PATH.read_bytes()

    clean_resp = client.post(
        "/clean?llm_mode=deterministic",
        files={"file": ("sample_dirty.csv", csv_bytes, "text/csv")},
    )
    assert clean_resp.status_code == 200
    resp_data = clean_resp.json()
    expected_out_sha = resp_data["quality_report"]["output_sha256"]

    script_payload = {
        "filename": "sample_dirty.csv",
        "repairs": resp_data["repairs"],
        "input_sha256": resp_data["quality_report"]["input_sha256"],
        "expected_output_sha256": expected_out_sha,
        "approved_indices": None,
    }
    script_resp = client.post("/reproduce-script", json=script_payload)
    assert script_resp.status_code == 200
    script_json = script_resp.json()
    script_content = script_json["script_content"]

    # Write script to temporary folder
    script_file = tmp_path / script_json["script_filename"]
    script_file.write_text(script_content, encoding="utf-8")
    out_csv = tmp_path / "replay_output.csv"

    # Actually execute the generated script using the current Python interpreter
    run_res = subprocess.run(
        [sys.executable, str(script_file), str(SAMPLE_DIRTY_PATH.resolve()), str(out_csv)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert run_res.returncode == 0
    assert "VERIFICATION PASSED" in run_res.stdout

    # Verify byte-for-byte SHA-256 match
    import hashlib
    actual_out_sha = hashlib.sha256(out_csv.read_bytes()).hexdigest()
    assert actual_out_sha == expected_out_sha, (
        f"Replay script output SHA-256 ({actual_out_sha}) must match "
        f"DataFixResponse output SHA-256 ({expected_out_sha}) byte-for-byte"
    )


def test_ollama_sha256_metadata_cache_hit(tmp_path, monkeypatch):
    """Verifies that the SHA-256 metadata-only Ollama prompt cache eliminates redundant
    LLM inference: second run on identical Stage1Result loads from cache with zero new Ollama calls.
    """
    cache_dir = tmp_path / "llm_cache"
    monkeypatch.setenv("DATAFIX_LLM_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("DATAFIX_ENABLE_LLM_CACHE", "1")

    df = load_csv(SAMPLE_DIRTY_PATH)
    stage1_res = run_stage1_cleaning(df)

    # Mock Ollama client
    mock_ollama_module = MagicMock()
    mock_client = MagicMock()
    mock_ollama_module.Client.return_value = mock_client

    # Return valid mock JSON tailored to the requested schema
    def mock_chat_side_effect(*args, **kwargs):
        messages = kwargs.get("messages", [])
        prompt_text = messages[0]["content"] if messages else ""
        if "Missing categorical cells" in prompt_text:
            return {"message": {"content": json.dumps({"proposals": []})}}
        elif "master tabular data quality detective" in prompt_text:
            return {"message": {"content": json.dumps({"diagnoses": []})}}
        else:
            return {
                "message": {
                    "content": json.dumps({
                        "column": "city",
                        "mappings": [
                            {
                                "original_value": "BLR",
                                "standardized_value": "Bengaluru",
                                "reason": "Standardized abbreviation",
                            }
                        ],
                    })
                }
            }

    mock_client.chat.side_effect = mock_chat_side_effect

    # First pass: cache miss, calls Ollama
    with patch.dict("sys.modules", {"ollama": mock_ollama_module}):
        with patch("ai_engine.check_ollama_health", return_value={"reachable": True, "model": "qwen2.5:3b", "model_available": True}):
            res1 = run_stage2_semantic_cleaning(stage1_res, llm_mode="ollama")

    initial_call_count = mock_client.chat.call_count
    assert initial_call_count >= 1, "First run must query Ollama on cache miss"

    # Second pass: cache hit, zero new calls to mock_client.chat
    with patch.dict("sys.modules", {"ollama": mock_ollama_module}):
        with patch("ai_engine.check_ollama_health", return_value={"reachable": True, "model": "qwen2.5:3b", "model_available": True}):
            res2 = run_stage2_semantic_cleaning(stage1_res, llm_mode="ollama")

    second_call_count = mock_client.chat.call_count
    assert second_call_count == initial_call_count, (
        f"Second run should have hit cache with 0 new Ollama calls, "
        f"but call count increased from {initial_call_count} to {second_call_count}"
    )

    # Ensure output results match
    assert len(res1.repairs) == len(res2.repairs)
    assert res1.mode_used == res2.mode_used == "ollama"
    assert res1.ollama_used is True and res2.ollama_used is True


