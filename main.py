"""FastAPI backend exposing /health and /clean endpoints with strict Pydantic v2 response contracts."""

import re
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from ai_engine import check_ollama_health
from ledger import get_audit_run, list_audit_runs, record_audit_run
from pipeline import (
    apply_selective_review,
    generate_reproduce_script,
    run_datafix_pipeline,
)
from schemas import (
    ApplyReviewRequest,
    ApplyReviewResponse,
    DataFixResponse,
    ReproduceScriptRequest,
    ReproduceScriptResponse,
)

# ============================================================================
# Application Setup & CORS
# ============================================================================

app = FastAPI(
    title="DataFix API",
    description="100% local neuro-symbolic tabular data quality, diagnosis, and repair engine",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/", tags=["General"])
def root() -> Dict[str, str]:
    """Lightweight root metadata endpoint."""
    return {
        "service": "DataFix API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["General"])
def health_check(
    model: Optional[str] = Query(None, description="Optional target Ollama model to check"),
) -> Dict[str, Any]:
    """Health check endpoint reporting service status and Ollama availability."""
    ollama_info = check_ollama_health(model=model, timeout=2.0)
    return {
        "status": "ok",
        "service": "datafix",
        "ollama": ollama_info,
        "supported_llm_modes": ["auto", "deterministic", "ollama"],
    }


@app.post(
    "/clean",
    response_model=DataFixResponse,
    summary="Clean and diagnose CSV dataset",
    description="Stateless neuro-symbolic tabular data quality, diagnosis, and repair pipeline.",
    tags=["Cleaning"],
)
async def clean_csv(
    file: UploadFile = File(..., description="Uploaded CSV file to clean"),
    llm_mode: Optional[str] = Query(None, description="LLM mode: 'auto', 'deterministic', or 'ollama'"),
    model: Optional[str] = Query(None, description="Ollama model tag to use for Stage 2 semantic AI"),
    skip_stage2: bool = Query(False, description="Whether to skip Stage 2 semantic AI repairs"),
) -> DataFixResponse:
    """Uploads and repairs a tabular CSV dataset."""
    filename = file.filename or "uploaded.csv"

    # Enforce safe filename guard (must end with .csv, case-insensitive)
    if not filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .csv files are supported.",
        )

    # Ingest in-memory bytes
    raw_bytes = await file.read()

    # Enforce max upload size guard (20 MB)
    if len(raw_bytes) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds the 20 MB limit.",
        )

    # Enforce empty content guard
    if len(raw_bytes) == 0 or not raw_bytes.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded CSV is empty or has zero bytes.",
        )

    # Run blocking pipeline in worker threadpool off the event loop
    try:
        response = await run_in_threadpool(
            run_datafix_pipeline,
            csv_source=raw_bytes,
            filename=filename,
            llm_mode=llm_mode,
            model=model,
            skip_stage2=skip_stage2,
        )
        # Non-fatal persistent SQLite audit ledger entry
        try:
            record_audit_run(response, llm_mode=llm_mode or "auto")
        except Exception:
            pass
        return response
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline execution failed: {exc}",
        )


@app.get("/history", tags=["Audit Ledger"])
def get_history(limit: int = Query(25, ge=1, le=100)) -> Dict[str, Any]:
    """Returns recent cleaning run entries from the persistent SQLite audit ledger."""
    runs = list_audit_runs(limit=limit)
    return {"runs": runs}


@app.get("/history/{run_id}", tags=["Audit Ledger"])
def get_history_run(run_id: str) -> Dict[str, Any]:
    """Returns detailed audit metadata for a specific run ID, or 404 if not found."""
    run = get_audit_run(run_id)
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Audit run '{run_id}' not found.",
        )
    return run


@app.post("/apply-review", response_model=ApplyReviewResponse, tags=["Review"])
def apply_review_endpoint(payload: ApplyReviewRequest) -> ApplyReviewResponse:
    """Selectively applies user-approved repairs to the raw CSV, reverting

    rejected repairs and restoring rejected duplicate rows.
    """
    try:
        raw_bytes = payload.raw_csv.encode("utf-8")
        result = apply_selective_review(
            raw_csv_bytes=raw_bytes,
            repairs=payload.repairs,
            approved_indices=payload.approved_indices,
        )
        return ApplyReviewResponse(**result)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Selective review application failed: {exc}",
        )


@app.post("/reproduce-script", response_model=ReproduceScriptResponse, tags=["Replay"])
def reproduce_script_endpoint(payload: ReproduceScriptRequest) -> ReproduceScriptResponse:
    """Generates a standalone, executable Python 3 replay script (reproduce_fix.py)

    matching the exact SHA-256 provenance hash.
    """
    try:
        stem = Path(payload.filename).stem or "dataset"
        clean_stem = re.sub(r"[^\w\-]", "_", stem)
        script_filename = f"reproduce_fix_{clean_stem}.py"
        script_content = generate_reproduce_script(
            filename=payload.filename,
            repairs=payload.repairs,
            input_sha256=payload.input_sha256,
            expected_output_sha256=payload.expected_output_sha256,
            approved_indices=payload.approved_indices,
        )
        return ReproduceScriptResponse(
            script_filename=script_filename,
            script_content=script_content,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Replay script generation failed: {exc}",
        )



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
