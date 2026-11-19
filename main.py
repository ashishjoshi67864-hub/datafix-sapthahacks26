"""FastAPI backend exposing /health and /clean endpoints with strict Pydantic v2 response contracts."""

from typing import Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from ai_engine import check_ollama_health
from pipeline import run_datafix_pipeline
from schemas import DataFixResponse

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
