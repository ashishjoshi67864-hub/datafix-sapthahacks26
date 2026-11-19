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
