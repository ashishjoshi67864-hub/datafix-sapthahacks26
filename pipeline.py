"""Orchestrator (DETECT -> TRACK): coordinates Stage 1, fault-isolated Stage 2, post-repair validation, stable 0-based row indexing, and SHA-256 dataset hashing."""

import hashlib
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from ai_engine import run_stage2_semantic_cleaning
from cleaning_engine import compute_health_score, load_csv, run_stage1_cleaning
from schemas import (
    DataFixResponse,
    DatasetSummary,
    PipelineStatus,
    QualityReport,
    RepairItem,
    RootCauseDiagnosis,
    UnresolvedIssue,
)


# ============================================================================
# Helpers
# ============================================================================

def compute_sha256(data: bytes) -> str:
    """Computes a hexadecimal SHA-256 digest of raw byte data."""
    return hashlib.sha256(data).hexdigest()


def serialize_dataframe_to_csv(df: pd.DataFrame) -> str:
    """Serializes a DataFrame to a clean UTF-8 CSV string with \\n line endings

    and no unnamed index column.
    """
    return df.to_csv(index=False, lineterminator="\n")


# ============================================================================
# Primary Pipeline Entry Point
# ============================================================================

def run_datafix_pipeline(
    csv_source: Union[bytes, str, Path],
    filename: str = "uploaded.csv",
    llm_mode: Optional[str] = None,
    model: Optional[str] = None,
    skip_stage2: bool = False,
) -> DataFixResponse:
    """End-to-end DataFix tabular repair pipeline.

    1. Ingests raw CSV bytes, file paths, or strings statelessly in-memory.
    2. Computes cryptographic input SHA-256.
    3. Executes Stage 1 deterministic profiling, cleaning, identifier protection, and KS-drift checks.
    4. Executes fault-isolated Stage 2 semantic AI standardizations, functional dependency imputations, and root-cause diagnoses.
    5. Computes post-repair quality health score, serializes cleaned CSV, and computes output SHA-256.
    6. Returns a fully validated DataFixResponse Pydantic contract.
    """
    target_filename = filename

    # 1. Normalize csv_source to raw bytes
    if isinstance(csv_source, Path):
        raw_bytes = csv_source.read_bytes()
        if filename == "uploaded.csv":
            target_filename = csv_source.name
    elif isinstance(csv_source, str):
        if "\n" in csv_source or "\r" in csv_source:
            raw_bytes = csv_source.encode("utf-8")
        else:
            p = Path(csv_source)
            try:
                if p.is_file():
                    raw_bytes = p.read_bytes()
                    if filename == "uploaded.csv":
                        target_filename = p.name
                else:
                    raw_bytes = csv_source.encode("utf-8")
            except OSError:
                raw_bytes = csv_source.encode("utf-8")
    elif isinstance(csv_source, bytes):
        raw_bytes = csv_source
    else:
        raise ValueError(f"Unsupported csv_source type: {type(csv_source)}")

    # 2. Compute input SHA-256
    input_sha256 = compute_sha256(raw_bytes)

    # 3. Validate non-empty bytes and load DataFrame
    if not raw_bytes or not raw_bytes.strip():
        raise ValueError("Uploaded CSV is empty or has no columns.")

    raw_df = load_csv(raw_bytes)
    if raw_df.empty and len(raw_df.columns) == 0:
        raise ValueError("Uploaded CSV is empty or has no columns.")
    if len(raw_df.columns) == 0:
        raise ValueError("Uploaded CSV is empty or has no columns.")

    # 4. Run Stage 1 Deterministic Profiling & Cleaning
    stage1 = run_stage1_cleaning(raw_df)

    # 5. Fault-Isolated Stage 2 Semantic AI Engine
    if skip_stage2:
        final_df = stage1.cleaned_df
        stage2_repairs: List[RepairItem] = []
        stage2_unresolved: List[UnresolvedIssue] = []
        root_causes: List[RootCauseDiagnosis] = []
        status = PipelineStatus.COMPLETED_STAGE1_ONLY
        combined_warnings = list(stage1.warnings) + [
            "Stage 2 semantic AI skipped by request; returned Stage 1 deterministic output only."
        ]
        after_health_score = stage1.stage1_health_score
    else:
        try:
            stage2 = run_stage2_semantic_cleaning(stage1, llm_mode=llm_mode, model=model)
            final_df = stage2.cleaned_df
            stage2_repairs = stage2.repairs
            stage2_unresolved = stage2.unresolved_issues
            root_causes = stage2.root_cause_diagnoses
            status = PipelineStatus.COMPLETED
            combined_warnings = list(stage1.warnings)
            for w in stage2.warnings:
                if w not in combined_warnings:
                    combined_warnings.append(w)
            after_health_score = compute_health_score(final_df, stage1.protected_identifier_columns)
        except Exception as exc:
            final_df = stage1.cleaned_df
            stage2_repairs = []
            stage2_unresolved = []
            root_causes = []
            status = PipelineStatus.COMPLETED_STAGE1_ONLY
            combined_warnings = list(stage1.warnings) + [
                f"Stage 2 semantic cleaning encountered an unexpected error ({type(exc).__name__}: {exc}); falling back to Stage 1 output."
            ]
            after_health_score = stage1.stage1_health_score

    # 6. Serialize cleaned CSV & Compute Output SHA-256
    cleaned_csv = serialize_dataframe_to_csv(final_df)
    output_sha256 = compute_sha256(cleaned_csv.encode("utf-8"))

    # 7. Assemble Structured Contract
    all_repairs = stage1.repairs + stage2_repairs
    all_unresolved = stage1.unresolved_issues + stage2_unresolved

    summary = DatasetSummary(
        original_rows=stage1.original_rows,
        cleaned_rows=len(final_df),
        columns=list(final_df.columns),
        total_repairs=len(all_repairs),
        stage1_repairs=len(stage1.repairs),
        stage2_repairs=len(stage2_repairs),
        duplicates_removed=stage1.duplicates_removed,
        unresolved_count=len(all_unresolved),
        protected_identifier_columns=stage1.protected_identifier_columns,
    )

    quality_report = QualityReport(
        before_health_score=stage1.before_health_score,
        after_health_score=after_health_score,
        input_sha256=input_sha256,
        output_sha256=output_sha256,
        ks_drift_checks=stage1.ks_drift_checks,
    )

    return DataFixResponse(
        status=status,
        filename=target_filename,
        summary=summary,
        quality_report=quality_report,
        root_cause_diagnoses=root_causes,
        repairs=all_repairs,
        unresolved_issues=all_unresolved,
        warnings=combined_warnings,
        cleaned_csv=cleaned_csv,
    )
