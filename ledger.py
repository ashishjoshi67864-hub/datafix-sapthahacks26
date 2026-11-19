"""Persistent SQLite audit ledger for tracking DataFix cleaning runs."""

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from schemas import DataFixResponse

DEFAULT_LEDGER_PATH = "storage/audit_ledger.db"


def get_db_path(db_path: Optional[Union[Path, str]] = None) -> Path:
    """Resolves target SQLite database file path from parameter or environment."""
    if db_path:
        return Path(db_path)
    env_val = os.environ.get("DATAFIX_LEDGER_DB")
    if env_val:
        return Path(env_val)
    return Path(DEFAULT_LEDGER_PATH)


def init_ledger_db(db_path: Optional[Union[Path, str]] = None) -> Path:
    """Initializes the SQLite audit ledger table schema if not already present."""
    path = get_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(path)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_runs (
                run_id TEXT PRIMARY KEY,
                timestamp_utc TEXT NOT NULL,
                filename TEXT NOT NULL,
                status TEXT NOT NULL,
                llm_mode TEXT NOT NULL,
                input_sha256 TEXT NOT NULL,
                output_sha256 TEXT NOT NULL,
                original_rows INTEGER NOT NULL,
                cleaned_rows INTEGER NOT NULL,
                total_repairs INTEGER NOT NULL,
                unresolved_count INTEGER NOT NULL,
                before_health_score REAL NOT NULL,
                after_health_score REAL NOT NULL,
                response_summary_json TEXT NOT NULL
            );
            """
        )
        conn.commit()
    return path


def record_audit_run(
    response: DataFixResponse,
    llm_mode: str = "deterministic",
    db_path: Optional[Union[Path, str]] = None,
) -> str:
    """Records an executed pipeline run into the SQLite audit ledger."""
    path = init_ledger_db(db_path)
    run_id = uuid.uuid4().hex[:12]
    timestamp_utc = datetime.now(timezone.utc).isoformat()

    summary_payload = {
        "summary": response.summary.model_dump(),
        "quality_report": response.quality_report.model_dump(),
        "root_cause_diagnoses": [d.model_dump() for d in response.root_cause_diagnoses],
        "warnings": response.warnings,
    }

    with sqlite3.connect(str(path)) as conn:
        conn.execute(
            """
            INSERT INTO audit_runs (
                run_id, timestamp_utc, filename, status, llm_mode,
                input_sha256, output_sha256, original_rows, cleaned_rows,
                total_repairs, unresolved_count, before_health_score,
                after_health_score, response_summary_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                timestamp_utc,
                response.filename,
                response.status.value,
                llm_mode,
                response.quality_report.input_sha256,
                response.quality_report.output_sha256,
                response.summary.original_rows,
                response.summary.cleaned_rows,
                response.summary.total_repairs,
                response.summary.unresolved_count,
                response.quality_report.before_health_score,
                response.quality_report.after_health_score,
                json.dumps(summary_payload),
            ),
        )
        conn.commit()

    return run_id


def list_audit_runs(
    limit: int = 25,
    db_path: Optional[Union[Path, str]] = None,
) -> List[Dict[str, Any]]:
    """Retrieves recent audit runs from SQLite, ordered by descending timestamp."""
    path = init_ledger_db(db_path)
    with sqlite3.connect(str(path)) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT run_id, timestamp_utc, filename, status, llm_mode,
                   input_sha256, output_sha256, original_rows, cleaned_rows,
                   total_repairs, unresolved_count, before_health_score,
                   after_health_score, response_summary_json
            FROM audit_runs
            ORDER BY timestamp_utc DESC
            LIMIT ?
            """,
            (max(1, limit),),
        )
        rows = cur.fetchall()

    result = []
    for r in rows:
        item = dict(r)
        try:
            item["response_summary"] = json.loads(item.pop("response_summary_json"))
        except Exception:
            item["response_summary"] = {}
        result.append(item)
    return result


def get_audit_run(
    run_id: str,
    db_path: Optional[Union[Path, str]] = None,
) -> Optional[Dict[str, Any]]:
    """Retrieves a single audit run by run_id, or None if not found."""
    path = init_ledger_db(db_path)
    with sqlite3.connect(str(path)) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT run_id, timestamp_utc, filename, status, llm_mode,
                   input_sha256, output_sha256, original_rows, cleaned_rows,
                   total_repairs, unresolved_count, before_health_score,
                   after_health_score, response_summary_json
            FROM audit_runs
            WHERE run_id = ?
            """,
            (run_id,),
        )
        row = cur.fetchone()

    if not row:
        return None

    item = dict(row)
    try:
        item["response_summary"] = json.loads(item.pop("response_summary_json"))
    except Exception:
        item["response_summary"] = {}
    return item
