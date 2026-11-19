"""Orchestrator (DETECT -> TRACK): coordinates Stage 1, fault-isolated Stage 2, post-repair validation, stable 0-based row indexing, and SHA-256 dataset hashing."""

import hashlib
import json
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from ai_engine import run_stage2_semantic_cleaning
from cleaning_engine import (
    compute_health_score,
    detect_identifier_columns,
    load_csv,
    run_stage1_cleaning,
)
from schemas import (
    AnomalyType,
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


# ============================================================================
# Selective Review & Deterministic Replay Script Generator
# ============================================================================

def apply_selective_review(
    raw_csv_bytes: bytes,
    repairs: List[RepairItem],
    approved_indices: List[int],
) -> dict:
    """Selectively applies only user-approved repairs to the raw CSV, reverting

    any unchecked/rejected repairs back to their raw original values, and
    retaining any rejected duplicate rows.
    """
    df = load_csv(raw_csv_bytes)
    approved_set = set(approved_indices)
    approved_repairs = [r for i, r in enumerate(repairs) if i in approved_set]

    # 1. Separate approved repairs into duplicate drops and cell edits
    drop_indices = [
        r.row_index
        for r in approved_repairs
        if r.anomaly_type == AnomalyType.EXACT_DUPLICATE or r.column is None
    ]
    cell_repairs = [
        r
        for r in approved_repairs
        if r.column is not None and r.anomaly_type != AnomalyType.EXACT_DUPLICATE
    ]

    # 2. Apply approved cell mutations onto raw DataFrame
    for r in cell_repairs:
        if r.row_index in df.index and r.column in df.columns:
            df.at[r.row_index, r.column] = "" if r.corrected_value is None else str(r.corrected_value)

    # 3. Drop approved duplicate rows
    if drop_indices:
        df = df.drop(index=[idx for idx in drop_indices if idx in df.index], errors="ignore")

    cleaned_csv = serialize_dataframe_to_csv(df)
    output_sha256 = compute_sha256(cleaned_csv.encode("utf-8"))
    protected_cols = detect_identifier_columns(df)
    after_health_score = compute_health_score(df, protected_cols)

    return {
        "cleaned_csv": cleaned_csv,
        "output_sha256": output_sha256,
        "after_health_score": after_health_score,
        "approved_count": len(approved_repairs),
        "rejected_count": len(repairs) - len(approved_repairs),
        "cleaned_rows": len(df),
    }


def generate_reproduce_script(
    filename: str,
    repairs: List[RepairItem],
    input_sha256: str,
    expected_output_sha256: str,
    approved_indices: Optional[List[int]] = None,
) -> str:
    """Generates a standalone, executable, deterministic Python 3 replay script

    (reproduce_fix.py) that reproduces the exact cleaned CSV and cryptographic
    output SHA-256 digest from the raw input CSV using Python stdlib.
    """
    if approved_indices is not None:
        approved_set = set(approved_indices)
        effective_repairs = [r for i, r in enumerate(repairs) if i in approved_set]
    else:
        effective_repairs = list(repairs)

    drop_indices = [
        int(r.row_index)
        for r in effective_repairs
        if r.anomaly_type == AnomalyType.EXACT_DUPLICATE or r.column is None
    ]
    cell_edits = [
        (int(r.row_index), str(r.column), None if r.corrected_value is None else str(r.corrected_value))
        for r in effective_repairs
        if r.column is not None and r.anomaly_type != AnomalyType.EXACT_DUPLICATE
    ]

    drop_repr = json.dumps(drop_indices)
    cell_edits_repr = json.dumps(cell_edits, indent=4)

    script = f'''"""DataFix Standalone Deterministic Replay Script
Generated by DataFix PS 01 — Neuro-Symbolic Tabular Quality Workbench

Replays verified repairs deterministically onto raw input CSV to reproduce cleaned dataset.
Verified byte-for-byte against cryptographic SHA-256 provenance hash.
"""

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

EXPECTED_INPUT_SHA256 = "{input_sha256}"
EXPECTED_OUTPUT_SHA256 = "{expected_output_sha256}"
ORIGINAL_FILENAME = "{filename}"

# Approved duplicate row indices to drop
DROP_ROW_INDICES = {drop_repr}

# Approved cell-level repairs: (row_index, column_name, corrected_value)
CELL_EDITS = {cell_edits_repr}


def compute_sha256(filepath_or_bytes):
    if isinstance(filepath_or_bytes, (str, Path)):
        data = Path(filepath_or_bytes).read_bytes()
    else:
        data = filepath_or_bytes
    return hashlib.sha256(data).hexdigest()


def replay_fix(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        print(f"Error: Input file '{{input_path}}' not found.", file=sys.stderr)
        sys.exit(1)

    raw_bytes = input_path.read_bytes()
    in_sha = compute_sha256(raw_bytes)
    print(f"[*] Input File: {{input_path.name}} (SHA-256: {{in_sha[:16]}}...)")
    if in_sha != EXPECTED_INPUT_SHA256:
        print(f"[!] Warning: Input SHA-256 differs from recorded audit hash (expected: {{EXPECTED_INPUT_SHA256[:16]}}...)")

    try:
        import pandas as pd
        import io

        try:
            content = raw_bytes.decode("utf-8-sig")
        except UnicodeDecodeError:
            content = raw_bytes.decode("latin-1")

        df = pd.read_csv(io.StringIO(content), dtype=str, keep_default_na=False)

        # 1. Apply approved cell mutations
        for row_idx, col, corrected_val in CELL_EDITS:
            if row_idx in df.index and col in df.columns:
                df.at[row_idx, col] = "" if corrected_val is None else str(corrected_val)

        # 2. Drop approved duplicate rows
        df = df.drop(index=[r for r in DROP_ROW_INDICES if r in df.index], errors="ignore")

        # 3. Serialize cleanly matching DataFix format
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(str(output_path), index=False, lineterminator="\\n")

    except ImportError:
        import io

        try:
            content = raw_bytes.decode("utf-8-sig")
        except UnicodeDecodeError:
            content = raw_bytes.decode("latin-1")

        reader = csv.reader(io.StringIO(content))
        rows = list(reader)
        if not rows:
            print("Error: Empty CSV input.", file=sys.stderr)
            sys.exit(1)

        headers = rows[0]
        col_to_idx = {{c: i for i, c in enumerate(headers)}}
        data_rows = rows[1:]

        for row_idx, col, corrected_val in CELL_EDITS:
            if 0 <= row_idx < len(data_rows) and col in col_to_idx:
                c_idx = col_to_idx[col]
                data_rows[row_idx][c_idx] = "" if corrected_val is None else str(corrected_val)

        drop_set = set(DROP_ROW_INDICES)
        kept_rows = [r for i, r in enumerate(data_rows) if i not in drop_set]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, lineterminator="\\n")
            writer.writerow(headers)
            writer.writerows(kept_rows)

    out_bytes = output_path.read_bytes()
    out_sha = compute_sha256(out_bytes)
    print(f"[*] Output Saved: {{output_path}} (SHA-256: {{out_sha[:16]}}...)")

    match = (out_sha == EXPECTED_OUTPUT_SHA256)
    if match:
        print("[+] VERIFICATION PASSED: Output SHA-256 matches recorded audit hash byte-for-byte!")
    else:
        print(f"[-] VERIFICATION NOTE: Output SHA-256 differs (expected: {{EXPECTED_OUTPUT_SHA256[:16]}}...)")

    return match


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: python {{Path(sys.argv[0]).name}} <input.csv> <output.csv>")
        sys.exit(1)

    in_file = sys.argv[1]
    out_file = sys.argv[2]
    success = replay_fix(in_file, out_file)
    sys.exit(0 if success else 2)
'''
    return script

