"""Stage 1 (DETECT + REPAIR + VALIDATE): deterministic Pandas/NumPy/SciPy/BS4/Scikit-Learn profiling, identifier protection, safe repairs, and KS-test drift rollback. Never calls an LLM."""

import difflib
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Set, TextIO, Tuple, Union

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from dateutil import parser as date_parser
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from schemas import (
    AnomalyType,
    CleaningStage,
    KSDriftResult,
    RepairItem,
    RepairMethod,
    UnresolvedIssue,
)


# ============================================================================
# Dataclasses
# ============================================================================

@dataclass
class Stage1Result:
    cleaned_df: pd.DataFrame
    repairs: List[RepairItem]
    unresolved_issues: List[UnresolvedIssue]
    ks_drift_checks: List[KSDriftResult]
    warnings: List[str]
    protected_identifier_columns: List[str]
    original_rows: int
    duplicates_removed: int
    before_health_score: float
    stage1_health_score: float
    metadata_summary: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Constants & Identifier Tokens
# ============================================================================

ID_TOKENS = {
    "id",
    "phone",
    "mobile",
    "tel",
    "contact",
    "postal",
    "zip",
    "zipcode",
    "pincode",
    "pin",
    "email",
    "ssn",
    "account",
    "aadhaar",
    "pan",
}

NULL_PLACEHOLDERS = {
    "",
    "null",
    "none",
    "na",
    "n/a",
    "nan",
    "-",
    "?",
}

NUMERIC_KEYWORDS = {
    "metric",
    "score",
    "amount",
    "bill",
    "salary",
    "price",
    "qty",
    "quantity",
    "total",
    "num",
    "count",
    "age",
    "rate",
    "ratio",
    "val",
    "measure",
}

HTML_PATTERN = re.compile(r"<[a-zA-Z/][^>]*>")
EMAIL_REGEX = re.compile(r"^[\w\.\+\-]+@[a-zA-Z0-9\-]+\.[a-zA-Z0-9\-\.]+$")
ISO_DATE_REGEX = re.compile(r"^\d{4}-\d{2}-\d{2}$")

KNOWN_CITY_ABBREVIATIONS = {
    "BLR": "Bengaluru",
    "BOM": "Mumbai",
    "DEL": "Delhi",
    "HYD": "Hyderabad",
    "MAA": "Chennai",
    "CCU": "Kolkata",
    "PNQ": "Pune",
    "AMD": "Ahmedabad",
    "BLRU": "Bengaluru",
    "ND": "Delhi",
}

KNOWN_CITY_TYPOS = {
    "BANGLORE": "Bengaluru",
    "BANGALORE": "Bengaluru",
    "MUMBAY": "Mumbai",
    "BOMBAY": "Mumbai",
    "DEHLI": "Delhi",
    "NEW DEHLI": "Delhi",
    "CALCUTTA": "Kolkata",
    "MADRAS": "Chennai",
}

KNOWN_DEPT_ABBREVIATIONS = {
    "ENG": "Engineering",
    "MKT": "Marketing",
    "FIN": "Finance",
    "OPS": "Operations",
    "HR": "Human Resources",
    "IT": "Information Technology",
    "QA": "Quality Assurance",
}


# ============================================================================
# Core Helper Functions
# ============================================================================

def load_csv(source: Union[str, Path, bytes, BinaryIO, TextIO]) -> pd.DataFrame:
    """Loads a CSV into a DataFrame with string dtypes and 0-based RangeIndex.

    Never turns postal codes, phones, or pincodes into floats or silently strips
    padded strings.
    """
    if isinstance(source, (str, Path)):
        try:
            df = pd.read_csv(source, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(source, dtype=str, keep_default_na=False, encoding="latin-1")
    elif isinstance(source, bytes):
        try:
            df = pd.read_csv(io.BytesIO(source), dtype=str, keep_default_na=False, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(source), dtype=str, keep_default_na=False, encoding="latin-1")
    else:
        df = pd.read_csv(source, dtype=str, keep_default_na=False)

    df.index = pd.RangeIndex(len(df))
    return df


def detect_identifier_columns(df: pd.DataFrame) -> List[str]:
    """Detects identifier columns using token/boundary matching.

    Avoids substring false positives (e.g. matching 'id' in 'valid').
    """
    protected: List[str] = []
    for col in df.columns:
        col_clean = str(col).strip()
        tokens = [t.lower() for t in re.split(r"[\s_\-]+", col_clean) if t]
        if any(tok in ID_TOKENS for tok in tokens):
            protected.append(col)
        elif col_clean.lower().endswith("id") or col_clean.lower().endswith("pincode"):
            protected.append(col)
    return protected


def compute_health_score(
    df: pd.DataFrame,
    protected_cols: Optional[List[str]] = None,
) -> float:
    """Computes a deterministic, honestly calibrated 0.0-100.0 quality score.

    Weighted as:
    - Completeness (25%): 50% cell completeness, 50% row completeness
    - Uniqueness (20%): 1.0 - (dup_rows / total_rows)
    - Structural, Format & Domain Validity (30%): 1.0 - (defect_rows / total_rows)
    - Consistency & Relational Integrity (25%): 1.0 - (defect_rows / total_rows)
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    total_cells = total_rows * total_cols

    if total_rows == 0 or total_cols == 0:
        return 100.0

    # 1. Completeness (25%): 50% cell completeness, 50% row completeness
    null_cells = 0
    rows_with_null = set()
    for col in df.columns:
        for idx, val in df[col].items():
            if pd.isna(val) or str(val).strip().lower() in NULL_PLACEHOLDERS:
                null_cells += 1
                rows_with_null.add(idx)
    cell_comp = 1.0 - (null_cells / total_cells)
    row_comp = 1.0 - (len(rows_with_null) / total_rows)
    completeness = max(0.0, min(1.0, 0.5 * cell_comp + 0.5 * row_comp))

    # 2. Uniqueness (20%)
    dup_rows = df.duplicated().sum()
    uniqueness = max(0.0, min(1.0, 1.0 - (dup_rows / total_rows)))

    # 3. Structural, Format & Domain Validity (30%)
    validity_defect_rows = set()
    for col in df.columns:
        col_lower = str(col).lower()
        is_age = "age" in col_lower
        is_fin_or_qty = any(k in col_lower for k in ["amount", "bill", "salary", "price", "qty", "quantity", "total"])
        is_email = "email" in col_lower
        tokens = [t.lower() for t in re.split(r"[\s_\-]+", col_lower) if t]
        is_date = any(t in ["date", "dob", "joining", "admission", "discharge"] for t in tokens)

        for idx, val in df[col].items():
            if pd.isna(val) or str(val).strip().lower() in NULL_PLACEHOLDERS:
                continue
            str_val = str(val)
            if HTML_PATTERN.search(str_val) or str_val != str_val.strip() or "  " in str_val:
                validity_defect_rows.add(idx)
                continue
            if is_email and not EMAIL_REGEX.match(str_val.strip()):
                validity_defect_rows.add(idx)
                continue
            if is_date and not ISO_DATE_REGEX.match(str_val.strip()):
                validity_defect_rows.add(idx)
                continue
            if is_age or is_fin_or_qty:
                clean_str = re.sub(r"[₹\$€£\s,]|Rs\.?", "", str_val)
                try:
                    num = float(clean_str)
                    if is_age and (num < 0 or num > 120):
                        validity_defect_rows.add(idx)
                    elif is_fin_or_qty and num < 0:
                        validity_defect_rows.add(idx)
                except ValueError:
                    validity_defect_rows.add(idx)
    validity = max(0.0, min(1.0, 1.0 - (len(validity_defect_rows) / total_rows)))

    # 4. Consistency & Relational Integrity (25%)
    consistency_defect_rows = set()
    col_map = {str(c).lower(): c for c in df.columns}
    qty_c = col_map.get("qty") or col_map.get("quantity")
    price_c = col_map.get("unit_price") or col_map.get("price")
    total_c = col_map.get("order_total") or col_map.get("total") or col_map.get("order_amount")
    if qty_c and price_c and total_c:
        for idx, r in df.iterrows():
            try:
                q = float(re.sub(r"[^\d\.\-]", "", str(r[qty_c])))
                p = float(re.sub(r"[^\d\.\-]", "", str(r[price_c])))
                t = float(re.sub(r"[^\d\.\-]", "", str(r[total_c])))
                if q > 0 and p > 0 and abs(t - (q * p)) > 0.01:
                    consistency_defect_rows.add(idx)
            except Exception:
                pass

    for col in df.columns:
        if col in (protected_cols or []):
            continue
        col_l = str(col).lower()
        if any(k in col_l for k in ["id", "date", "dob", "time", "email", "phone", "name", "fname", "lname", "address", "age", "salary", "price", "amount", "qty", "total", "bill"]):
            continue
        non_null_vals = [str(v).strip() for v in df[col] if not pd.isna(v) and str(v).strip().lower() not in NULL_PLACEHOLDERS]
        unique_vals = set(non_null_vals)
        if len(unique_vals) < 2 or len(unique_vals) > 30:
            continue
        val_counts = pd.Series(non_null_vals).value_counts()
        dominant = [v for v, c in val_counts.items() if c >= 2]
        if not dominant:
            dominant = list(val_counts.index[:2])
        for idx, val in df[col].items():
            if pd.isna(val) or str(val).strip().lower() in NULL_PLACEHOLDERS:
                continue
            s = str(val).strip()
            if s not in dominant:
                u = s.upper()
                if u in KNOWN_CITY_ABBREVIATIONS or u in KNOWN_CITY_TYPOS or u in KNOWN_DEPT_ABBREVIATIONS:
                    consistency_defect_rows.add(idx)
                    continue
                clean_s = re.sub(r"^(new|old|north|south|east|west)\s+", "", s.lower()).strip()
                for dom in dominant:
                    if s.lower() == dom.lower():
                        consistency_defect_rows.add(idx)
                        break
                    r1 = difflib.SequenceMatcher(None, s.lower(), dom.lower()).ratio()
                    r2 = difflib.SequenceMatcher(None, clean_s, dom.lower()).ratio()
                    if max(r1, r2) >= 0.70:
                        consistency_defect_rows.add(idx)
                        break
                    if s.isupper() and 2 <= len(s) <= 4 and (dom.upper().startswith(s) or s in dom.upper()):
                        consistency_defect_rows.add(idx)
                        break

    consistency = max(0.0, min(1.0, 1.0 - (len(consistency_defect_rows) / total_rows)))

    score = 100.0 * (0.25 * completeness + 0.20 * uniqueness + 0.30 * validity + 0.25 * consistency)
    return round(float(np.clip(score, 0.0, 100.0)), 2)


# ============================================================================
# Stage 1 Cleaning Engine
# ============================================================================

def run_stage1_cleaning(df: pd.DataFrame) -> Stage1Result:
    """Executes deterministic Stage 1 profiling, cleaning, safe repairs,

    quarantining, and KS drift-tested KNN imputation.
    """
    cleaned_df = df.copy()
    if not isinstance(cleaned_df.index, pd.RangeIndex):
        cleaned_df.index = pd.RangeIndex(len(cleaned_df))

    original_rows = len(cleaned_df)
    repairs: List[RepairItem] = []
    unresolved_issues: List[UnresolvedIssue] = []
    ks_drift_checks: List[KSDriftResult] = []
    warnings: List[str] = []
    quarantined_cells: Set[Tuple[int, str]] = set()

    # Identify protected columns
    protected_cols = detect_identifier_columns(cleaned_df)

    # Initial Health Score
    before_health_score = compute_health_score(cleaned_df, protected_cols)

    # Pre-identify date columns and candidate numeric columns
    date_cols: List[str] = []
    numeric_cols: List[str] = []
    all_null_numeric_cols: List[str] = []

    for col in cleaned_df.columns:
        col_lower = str(col).lower()
        tokens = [t.lower() for t in re.split(r"[\s_\-]+", col_lower) if t]
        is_date_col = any(t in ["date", "dob", "joining", "admission", "discharge"] for t in tokens)

        if is_date_col:
            date_cols.append(col)
        elif col not in protected_cols:
            non_nulls = [
                v for v in cleaned_df[col]
                if not pd.isna(v) and str(v).strip().lower() not in NULL_PLACEHOLDERS
            ]
            if non_nulls:
                parsed_count = 0
                for v in non_nulls:
                    clean_v = re.sub(r"[₹\$€£\s,]|Rs\.?", "", str(v))
                    try:
                        float(clean_v)
                        parsed_count += 1
                    except ValueError:
                        pass
                if parsed_count / len(non_nulls) >= 0.60:
                    numeric_cols.append(col)
            else:
                # Column is completely null - check if name implies numeric
                if any(k in col_lower for k in NUMERIC_KEYWORDS):
                    all_null_numeric_cols.append(col)

    # -------------------------------------------------------------------------
    # a. Missing Placeholder Normalization & HTML Tag Stripping (Vectorized Fast-Path)
    # -------------------------------------------------------------------------
    for col in cleaned_df.columns:
        col_series = cleaned_df[col]
        # Fast-path: check if column contains '<'
        has_angle = col_series.dropna().astype(str).str.contains("<", regex=False)
        angle_indices = set(has_angle[has_angle].index) if has_angle.any() else set()

        for idx in list(cleaned_df.index):
            val = cleaned_df.at[idx, col]
            if pd.isna(val):
                continue
            str_val = str(val)

            # Strip HTML tags if present (only check cells with '<')
            if idx in angle_indices and HTML_PATTERN.search(str_val):
                stripped = BeautifulSoup(str_val, "html.parser").get_text()
                repairs.append(
                    RepairItem(
                        row_index=int(idx),
                        column=col,
                        original_value=str_val,
                        corrected_value=stripped,
                        anomaly_type=AnomalyType.HTML_TAG,
                        repair_method=RepairMethod.DETERMINISTIC_HTML_STRIP,
                        stage=CleaningStage.STAGE_1,
                        confidence=0.98,
                        reason=f"Stripped HTML markup from column '{col}'",
                    )
                )
                cleaned_df.at[idx, col] = stripped
                str_val = stripped

            # Normalize null placeholders
            if str_val.strip().lower() in NULL_PLACEHOLDERS:
                cleaned_df.at[idx, col] = np.nan

    # -------------------------------------------------------------------------
    # b. Whitespace Normalization (Vectorized Fast-Path)
    # -------------------------------------------------------------------------
    for col in cleaned_df.columns:
        if col in numeric_cols or col in all_null_numeric_cols:
            continue
        col_series = cleaned_df[col].dropna().astype(str)
        # Fast-path: check if any cell has leading/trailing or multiple spaces
        has_ws = col_series.str.contains(r"^\s|\s$|\s{2,}", regex=True)
        if not has_ws.any():
            continue
        ws_indices = set(has_ws[has_ws].index)

        for idx in ws_indices:
            val = cleaned_df.at[idx, col]
            if pd.isna(val):
                continue
            str_val = str(val)
            cleaned_str = re.sub(r"\s+", " ", str_val).strip()

            if cleaned_str == "":
                cleaned_df.at[idx, col] = np.nan
            elif cleaned_str != str_val:
                repairs.append(
                    RepairItem(
                        row_index=int(idx),
                        column=col,
                        original_value=str_val,
                        corrected_value=cleaned_str,
                        anomaly_type=AnomalyType.WHITESPACE,
                        repair_method=RepairMethod.DETERMINISTIC_WHITESPACE,
                        stage=CleaningStage.STAGE_1,
                        confidence=0.98,
                        reason=f"Normalized leading/trailing/multiple whitespace in column '{col}'",
                    )
                )
                cleaned_df.at[idx, col] = cleaned_str

    # -------------------------------------------------------------------------
    # c. Exact Deduplication (Preserving 0-based df.index)
    # -------------------------------------------------------------------------
    dup_mask = cleaned_df.duplicated(keep="first")
    duplicates_removed = int(dup_mask.sum())

    if duplicates_removed > 0:
        for idx in cleaned_df[dup_mask].index:
            repairs.append(
                RepairItem(
                    row_index=int(idx),
                    column=None,
                    original_value="exact_duplicate_row",
                    corrected_value=None,
                    anomaly_type=AnomalyType.EXACT_DUPLICATE,
                    repair_method=RepairMethod.DETERMINISTIC_DEDUP,
                    stage=CleaningStage.STAGE_1,
                    confidence=1.0,
                    reason="Dropped exact duplicate row preserving original 0-based index",
                )
            )
        cleaned_df = cleaned_df[~dup_mask]

    # -------------------------------------------------------------------------
    # c2. Fuzzy Near-Duplicate Row Detection (Key-Blocked O(n))
    # -------------------------------------------------------------------------
    text_cols = [
        c for c in cleaned_df.columns
        if any(tok in str(c).lower() for tok in ["name", "fname", "lname", "user", "title", "person", "employee", "customer", "patient"])
    ]
    if text_cols:
        block_cols = [
            c for c in cleaned_df.columns
            if any(tok in str(c).lower() for tok in ["phone", "email", "mobile", "tel", "contact", "id", "account", "ssn"])
            and c not in text_cols
        ]
        if not block_cols:
            block_cols = [
                c for c in cleaned_df.columns
                if c not in text_cols and cleaned_df[c].nunique() >= 2
            ][:2]

        seen_near_dup_pairs: Set[Tuple[int, int]] = set()
        for b_col in block_cols:
            grouped: Dict[str, List[int]] = {}
            for idx in cleaned_df.index:
                s = str(cleaned_df.at[idx, b_col]).strip()
                if s and s.lower() not in NULL_PLACEHOLDERS:
                    grouped.setdefault(s, []).append(int(idx))
            for key_val, idx_list in grouped.items():
                if 1 < len(idx_list) <= 15:
                    for i in range(len(idx_list)):
                        for j in range(i + 1, len(idx_list)):
                            idx_a = idx_list[i]
                            idx_b = idx_list[j]
                            pair = (min(idx_a, idx_b), max(idx_a, idx_b))
                            if pair in seen_near_dup_pairs:
                                continue
                            seen_near_dup_pairs.add(pair)
                            for t_col in text_cols:
                                v_a = str(cleaned_df.at[idx_a, t_col]).strip()
                                v_b = str(cleaned_df.at[idx_b, t_col]).strip()
                                if (
                                    v_a != v_b
                                    and v_a
                                    and v_b
                                    and v_a.lower() not in NULL_PLACEHOLDERS
                                    and v_b.lower() not in NULL_PLACEHOLDERS
                                ):
                                    sim = difflib.SequenceMatcher(None, v_a.lower(), v_b.lower()).ratio()
                                    if sim >= 0.90:
                                        unresolved_issues.append(
                                            UnresolvedIssue(
                                                row_index=int(idx_b),
                                                column=t_col,
                                                original_value=v_b,
                                                anomaly_type=AnomalyType.NEAR_DUPLICATE,
                                                confidence=0.55,
                                                reason=f"Probable near-duplicate of row {idx_a} (similarity={round(sim, 2)} on '{t_col}'); quarantined for review",
                                            )
                                        )

    # -------------------------------------------------------------------------
    # d. Date Normalization & Basic Type Coercion (Vectorized Fast-Path)
    # -------------------------------------------------------------------------
    # Process Date Columns
    for col in date_cols:
        for idx in list(cleaned_df.index):
            val = cleaned_df.at[idx, col]
            if pd.isna(val) or str(val).strip() == "":
                continue
            str_val = str(val).strip()
            # Fast-path: already valid ISO format YYYY-MM-DD
            if ISO_DATE_REGEX.match(str_val):
                continue
            try:
                parsed_dt = date_parser.parse(str_val, dayfirst=True)
                iso_date = parsed_dt.strftime("%Y-%m-%d")
                if iso_date != str_val:
                    repairs.append(
                        RepairItem(
                            row_index=int(idx),
                            column=col,
                            original_value=str_val,
                            corrected_value=iso_date,
                            anomaly_type=AnomalyType.INVALID_FORMAT,
                            repair_method=RepairMethod.DETERMINISTIC_TYPE_CAST,
                            stage=CleaningStage.STAGE_1,
                            confidence=0.95,
                            reason=f"Standardized date '{str_val}' to ISO format '{iso_date}'",
                        )
                    )
                    cleaned_df.at[idx, col] = iso_date
            except Exception:
                unresolved_issues.append(
                    UnresolvedIssue(
                        row_index=int(idx),
                        column=col,
                        original_value=str_val,
                        anomaly_type=AnomalyType.INVALID_FORMAT,
                        confidence=0.35,
                        reason=f"Unparseable date value '{str_val}' in column '{col}'",
                    )
                )
                quarantined_cells.add((int(idx), col))

    # Process Numeric Columns (Casting, Whitespace & Decimal formatting)
    for col in numeric_cols:
        non_nulls = [
            re.sub(r"[₹\$€£\s,]|Rs\.?", "", str(v))
            for v in cleaned_df[col]
            if not pd.isna(v) and str(v).strip() != ""
        ]
        # Check if column is integer-dominant
        all_integers = True
        for v in non_nulls:
            try:
                flt = float(v)
                if not flt.is_integer():
                    all_integers = False
                    break
            except ValueError:
                pass

        for idx in list(cleaned_df.index):
            val = cleaned_df.at[idx, col]
            if pd.isna(val) or str(val).strip() == "":
                continue
            str_val = str(val)
            clean_str = re.sub(r"[₹\$€£\s,]|Rs\.?", "", str_val)
            try:
                flt_val = float(clean_str)
                target_val: Union[int, float] = int(round(flt_val)) if all_integers else flt_val

                # Did representation change from raw string (e.g. whitespace or decimal string)?
                if str(target_val) != str_val.strip() or str_val != str_val.strip():
                    repairs.append(
                        RepairItem(
                            row_index=int(idx),
                            column=col,
                            original_value=str_val,
                            corrected_value=target_val,
                            anomaly_type=AnomalyType.TYPE_MISMATCH,
                            repair_method=RepairMethod.DETERMINISTIC_TYPE_CAST,
                            stage=CleaningStage.STAGE_1,
                            confidence=0.96,
                            reason=f"Coerced numeric string '{str_val}' to {type(target_val).__name__} {target_val}",
                        )
                    )
                cleaned_df.at[idx, col] = target_val
            except ValueError:
                unresolved_issues.append(
                    UnresolvedIssue(
                        row_index=int(idx),
                        column=col,
                        original_value=str_val,
                        anomaly_type=AnomalyType.TYPE_MISMATCH,
                        confidence=0.30,
                        reason=f"Non-numeric value '{str_val}' in numeric column '{col}'",
                    )
                )
                quarantined_cells.add((int(idx), col))

    # -------------------------------------------------------------------------
    # e. Domain, Format & Cross-Column Rule Checks (Quarantine Untouched)
    # -------------------------------------------------------------------------
    # 1. Emails
    for col in cleaned_df.columns:
        if "email" in str(col).lower():
            for idx in list(cleaned_df.index):
                val = cleaned_df.at[idx, col]
                if pd.isna(val) or str(val).strip() == "":
                    continue
                str_val = str(val).strip()
                if not EMAIL_REGEX.match(str_val):
                    unresolved_issues.append(
                        UnresolvedIssue(
                            row_index=int(idx),
                            column=col,
                            original_value=str_val,
                            anomaly_type=AnomalyType.INVALID_FORMAT,
                            confidence=0.30,
                            reason=f"Malformed email address '{str_val}' in column '{col}'",
                        )
                    )
                    quarantined_cells.add((int(idx), col))

    # 2. Domain bounds (Age, Salary, Amount, Qty)
    for col in numeric_cols:
        col_lower = str(col).lower()
        is_age = "age" in col_lower
        is_fin_or_qty = any(k in col_lower for k in ["amount", "bill", "salary", "price", "qty", "quantity", "total"])

        for idx in list(cleaned_df.index):
            val = cleaned_df.at[idx, col]
            if pd.isna(val):
                continue
            try:
                num = float(val)
                if is_age and (num < 0 or num > 120):
                    unresolved_issues.append(
                        UnresolvedIssue(
                            row_index=int(idx),
                            column=col,
                            original_value=val,
                            anomaly_type=AnomalyType.INVALID_DOMAIN_VALUE,
                            confidence=0.30,
                            reason=f"Age value {val} outside plausible domain [0, 120]",
                        )
                    )
                    quarantined_cells.add((int(idx), col))
                elif is_fin_or_qty and num < 0:
                    unresolved_issues.append(
                        UnresolvedIssue(
                            row_index=int(idx),
                            column=col,
                            original_value=val,
                            anomaly_type=AnomalyType.INVALID_DOMAIN_VALUE,
                            confidence=0.30,
                            reason=f"Non-negative numeric column '{col}' contains negative value {val}",
                        )
                    )
                    quarantined_cells.add((int(idx), col))
            except (ValueError, TypeError):
                pass

    # 3. Cross-column math rules
    col_map = {str(c).lower(): c for c in cleaned_df.columns}
    qty_c = col_map.get("qty") or col_map.get("quantity")
    price_c = col_map.get("unit_price") or col_map.get("price")
    total_c = col_map.get("order_total") or col_map.get("total") or col_map.get("order_amount")

    if qty_c and price_c and total_c:
        for idx in list(cleaned_df.index):
            q_val = cleaned_df.at[idx, qty_c]
            p_val = cleaned_df.at[idx, price_c]
            t_val = cleaned_df.at[idx, total_c]
            if not pd.isna(q_val) and not pd.isna(p_val) and not pd.isna(t_val):
                try:
                    q = float(q_val)
                    p = float(p_val)
                    t = float(t_val)
                    if q > 0 and p > 0:
                        expected = round(q * p, 2)
                        if abs(t - expected) > 0.01:
                            target_t = int(expected) if float(expected).is_integer() else expected
                            repairs.append(
                                RepairItem(
                                    row_index=int(idx),
                                    column=total_c,
                                    original_value=t_val,
                                    corrected_value=target_t,
                                    anomaly_type=AnomalyType.RELATIONAL_MISMATCH,
                                    repair_method=RepairMethod.DETERMINISTIC_RELATIONAL_FIX,
                                    stage=CleaningStage.STAGE_1,
                                    confidence=0.98,
                                    reason=f"Recalculated {total_c} = {qty_c} * {price_c} ({q} * {p} = {target_t})",
                                )
                            )
                            cleaned_df.at[idx, total_c] = target_t
                except (ValueError, TypeError):
                    pass

    # 4. Cross-column temporal date rules
    date_pairs = [
        ("order_date", "delivery_date"),
        ("admission_date", "discharge_date"),
        ("start_date", "end_date"),
    ]
    for s_name, e_name in date_pairs:
        s_c = col_map.get(s_name)
        e_c = col_map.get(e_name)
        if s_c and e_c:
            for idx in list(cleaned_df.index):
                s_val = cleaned_df.at[idx, s_c]
                e_val = cleaned_df.at[idx, e_c]
                if not pd.isna(s_val) and not pd.isna(e_val):
                    try:
                        s_dt = date_parser.parse(str(s_val))
                        e_dt = date_parser.parse(str(e_val))
                        if e_dt < s_dt:
                            unresolved_issues.append(
                                UnresolvedIssue(
                                    row_index=int(idx),
                                    column=e_c,
                                    original_value=e_val,
                                    anomaly_type=AnomalyType.TEMPORAL_VIOLATION,
                                    confidence=0.40,
                                    reason=f"Temporal violation: {e_c} ({e_val}) precedes {s_c} ({s_val})",
                                )
                            )
                            quarantined_cells.add((int(idx), e_c))
                    except Exception:
                        pass

    # -------------------------------------------------------------------------
    # f. Safe Scaled KNNImputer + KS-Test Blast-Radius Guard
    # -------------------------------------------------------------------------
    # First, handle all-null numeric columns explicitly
    for col in all_null_numeric_cols:
        warnings.append(f"Column '{col}' is all-null numeric; skipping KNN imputation.")
        ks_drift_checks.append(
            KSDriftResult(column=col, ks_statistic=0.0, p_value=1.0, status="skipped")
        )

    eligible_knn_cols: List[str] = []
    for col in numeric_cols:
        if col in protected_cols:
            continue
        col_series = cleaned_df[col]
        # Exclude quarantined cells when measuring valid numbers
        valid_nums = [
            float(v)
            for idx, v in col_series.items()
            if not pd.isna(v) and (int(idx), col) not in quarantined_cells
        ]
        total_in_col = len(col_series)

        if len(valid_nums) == 0:
            warnings.append(f"Column '{col}' is all-null numeric; skipping KNN imputation.")
            ks_drift_checks.append(
                KSDriftResult(column=col, ks_statistic=0.0, p_value=1.0, status="skipped")
            )
        elif len(valid_nums) < 2 or (len(valid_nums) / total_in_col) < 0.50:
            warnings.append(f"Column '{col}' has insufficient valid numeric data (<50% or <2 values); skipping KNN.")
            ks_drift_checks.append(
                KSDriftResult(column=col, ks_statistic=0.0, p_value=1.0, status="skipped")
            )
        else:
            eligible_knn_cols.append(col)

    # Check if any imputation is needed
    cols_needing_impute = [
        col for col in eligible_knn_cols
        if cleaned_df[col].isna().sum() > 0
    ]

    if cols_needing_impute and len(cleaned_df) >= 2:
        # Build working matrix, temporarily masking quarantined cells
        working_sub_df = pd.DataFrame(index=cleaned_df.index)
        for col in eligible_knn_cols:
            series_copy = cleaned_df[col].copy()
            for idx in cleaned_df.index:
                if (int(idx), col) in quarantined_cells:
                    series_copy.at[idx] = np.nan
                elif not pd.isna(series_copy.at[idx]):
                    try:
                        series_copy.at[idx] = float(series_copy.at[idx])
                    except (ValueError, TypeError):
                        series_copy.at[idx] = np.nan
            working_sub_df[col] = series_copy.astype(float)

        usable_rows = int(working_sub_df.notna().any(axis=1).sum())
        if usable_rows >= 2:
            n_neighbors = min(5, max(1, usable_rows - 1))
            scaler = StandardScaler()
            imputer = KNNImputer(n_neighbors=n_neighbors)

            try:
                # Scalability guard: for large datasets (> 2000 rows), fit on deterministic stratified sample
                if len(working_sub_df) > 2000:
                    complete_mask = working_sub_df.notna().all(axis=1)
                    complete_indices = working_sub_df[complete_mask].index
                    rng = np.random.default_rng(42)
                    sample_size = min(1500, len(complete_indices))
                    if sample_size >= 10:
                        sampled_idx = rng.choice(complete_indices, size=sample_size, replace=False)
                        ref_sub = working_sub_df.loc[sampled_idx]
                    else:
                        ref_sub = working_sub_df.dropna()
                    scaled_ref = scaler.fit_transform(ref_sub)
                    imputer.fit(scaled_ref)
                    scaled_mat = scaler.transform(working_sub_df)
                    imputed_scaled = imputer.transform(scaled_mat)
                else:
                    scaled_mat = scaler.fit_transform(working_sub_df)
                    imputed_scaled = imputer.fit_transform(scaled_mat)

                imputed_mat = scaler.inverse_transform(imputed_scaled)
                imputed_df = pd.DataFrame(imputed_mat, index=working_sub_df.index, columns=eligible_knn_cols)

                # Validate each imputed column with KS test
                for col in cols_needing_impute:
                    orig_non_nulls = working_sub_df[col].dropna().values
                    imputed_series = imputed_df[col].values

                    # Run KS test comparing pre-imputation non-nulls vs post-imputation series
                    ks_stat, p_val = stats.ks_2samp(orig_non_nulls, imputed_series)
                    ks_stat = float(ks_stat)
                    p_val = float(p_val)

                    # Pass condition: p_value >= 0.05 and ks_statistic <= 0.15
                    if p_val >= 0.05 and ks_stat <= 0.15:
                        ks_drift_checks.append(
                            KSDriftResult(
                                column=col,
                                ks_statistic=round(ks_stat, 4),
                                p_value=round(p_val, 4),
                                status="passed",
                            )
                        )
                        # Commit imputed cells
                        missing_ratio = float(working_sub_df[col].isna().sum() / len(working_sub_df))
                        conf = round(float(np.clip(0.70 + 0.20 * (1.0 - missing_ratio) - 0.50 * ks_stat, 0.60, 0.89)), 2)

                        # Determine if column is integer-dominant
                        is_int = all(float(x).is_integer() for x in orig_non_nulls)

                        for idx in cleaned_df.index:
                            if pd.isna(cleaned_df.at[idx, col]) and (int(idx), col) not in quarantined_cells:
                                imp_val = imputed_df.at[idx, col]
                                final_val: Union[int, float] = int(round(imp_val)) if is_int else round(float(imp_val), 2)
                                repairs.append(
                                    RepairItem(
                                        row_index=int(idx),
                                        column=col,
                                        original_value=None,
                                        corrected_value=final_val,
                                        anomaly_type=AnomalyType.MISSING_NUMERIC,
                                        repair_method=RepairMethod.DETERMINISTIC_KNN_IMPUTE,
                                        stage=CleaningStage.STAGE_1,
                                        confidence=conf,
                                        reason=f"KNN imputed missing value (n_neighbors={n_neighbors}, KS stat={round(ks_stat, 3)}, p-value={round(p_val, 3)} >= 0.05)",
                                    )
                                )
                                cleaned_df.at[idx, col] = final_val
                    else:
                        # Auto-Rollback: Keep cells null
                        ks_drift_checks.append(
                            KSDriftResult(
                                column=col,
                                ks_statistic=round(ks_stat, 4),
                                p_value=round(p_val, 4),
                                status="rolled_back",
                            )
                        )
                        warnings.append(
                            f"KS drift guard triggered on column '{col}' (ks={round(ks_stat, 3)}, p={round(p_val, 3)}); imputation rolled back."
                        )
                        for idx in cleaned_df.index:
                            if pd.isna(cleaned_df.at[idx, col]) and (int(idx), col) not in quarantined_cells:
                                unresolved_issues.append(
                                    UnresolvedIssue(
                                        row_index=int(idx),
                                        column=col,
                                        original_value=None,
                                        anomaly_type=AnomalyType.MISSING_NUMERIC,
                                        confidence=0.45,
                                        reason=f"KNN imputation caused distribution drift (KS stat={round(ks_stat, 3)} > 0.15 or p-value={round(p_val, 3)} < 0.05); rolled back to protect distribution",
                                    )
                                )
            except Exception as e:
                warnings.append(f"KNN imputation encountered an error: {e}; skipped safely.")
        else:
            warnings.append("Insufficient usable rows for KNN imputation; skipped safely.")
    elif cols_needing_impute:
        warnings.append("Fewer than 2 rows available for KNN imputation; skipped safely.")

    # -------------------------------------------------------------------------
    # g. Generic Schema-Agnostic Functional Dependency (FD) Miner
    # -------------------------------------------------------------------------
    for col_a in cleaned_df.columns:
        if col_a in protected_cols:
            continue
        col_a_lower = str(col_a).lower()
        if any(tok in col_a_lower for tok in ["postal", "zip", "pincode", "pin"]):
            continue

        non_null_a = cleaned_df[col_a].dropna()
        n_unique_a = non_null_a.nunique()
        if not (2 <= n_unique_a <= max(50, int(0.5 * len(cleaned_df)))):
            continue

        for col_b in cleaned_df.columns:
            if col_b == col_a or col_b in protected_cols:
                continue
            col_b_lower = str(col_b).lower()
            if "city" in col_b_lower and any(tok in col_a_lower for tok in ["postal", "zip", "pincode", "pin"]):
                continue

            null_b_count = cleaned_df[col_b].isna().sum()
            if null_b_count == 0:
                continue

            non_null_b = cleaned_df[col_b].dropna()
            n_unique_b = non_null_b.nunique()
            if not (2 <= n_unique_b <= 30):
                continue

            pair_df = cleaned_df[[col_a, col_b]].dropna()
            if len(pair_df) < 10:
                continue

            # Check FD purity: A -> B
            for a_val, group in pair_df.groupby(col_a):
                b_counts = group[col_b].value_counts()
                if b_counts.empty:
                    continue
                top_b = b_counts.index[0]
                support = int(b_counts.iloc[0])
                purity = float(b_counts.iloc[0] / b_counts.sum())
                if support >= 2 and purity >= 0.95:
                    for idx in cleaned_df.index:
                        if (int(idx), col_b) in quarantined_cells:
                            continue
                        if pd.isna(cleaned_df.at[idx, col_b]) and str(cleaned_df.at[idx, col_a]).strip() == str(a_val).strip():
                            repairs.append(
                                RepairItem(
                                    row_index=int(idx),
                                    column=col_b,
                                    original_value=None,
                                    corrected_value=top_b,
                                    anomaly_type=AnomalyType.MISSING_CATEGORICAL,
                                    repair_method=RepairMethod.DETERMINISTIC_RELATIONAL_FIX,
                                    stage=CleaningStage.STAGE_1,
                                    confidence=0.92,
                                    reason=f"Data-driven functional dependency: inferred '{col_b}'='{top_b}' from '{col_a}'='{a_val}' (support={support}, purity={round(purity, 2)})",
                                )
                            )
                            cleaned_df.at[idx, col_b] = top_b

    # -------------------------------------------------------------------------
    # h. Multivariate Anomaly Detection (IsolationForest on Unseen Schemas)
    # -------------------------------------------------------------------------
    if len(cleaned_df) >= 20:
        eligible_num_cols = [
            c for c in numeric_cols
            if c not in protected_cols and cleaned_df[c].notna().sum() >= 0.8 * len(cleaned_df)
        ]
        if len(eligible_num_cols) >= 2:
            num_sub = cleaned_df[eligible_num_cols].copy()
            valid_rows = [
                idx for idx in num_sub.index
                if not any((int(idx), c) in quarantined_cells for c in eligible_num_cols)
                and num_sub.loc[idx].notna().all()
            ]
            if len(valid_rows) >= 20:
                try:
                    X_mat = num_sub.loc[valid_rows].astype(float).values
                    scaler_iso = StandardScaler()
                    X_scaled = scaler_iso.fit_transform(X_mat)
                    iso = IsolationForest(contamination=0.03, random_state=42)
                    iso_preds = iso.fit_predict(X_scaled)
                    iso_scores = iso.decision_function(X_scaled)

                    for i, r_idx in enumerate(valid_rows):
                        if iso_preds[i] == -1 and iso_scores[i] < -0.15:
                            max_z = float(np.max(np.abs(X_scaled[i])))
                            if max_z > 3.5:
                                unresolved_issues.append(
                                    UnresolvedIssue(
                                        row_index=int(r_idx),
                                        column=None,
                                        original_value=f"multivariate_vector({[round(float(v), 2) for v in X_mat[i]]})",
                                        anomaly_type=AnomalyType.MULTIVARIATE_OUTLIER,
                                        confidence=0.42,
                                        reason=f"Multivariate outlier flagged by IsolationForest (anomaly score={round(float(iso_scores[i]), 3)}, max |z-score|={round(max_z, 2)} > 3.5); quarantined untouched",
                                    )
                                )
                except Exception as exc:
                    warnings.append(f"IsolationForest check skipped: {exc}")

    # -------------------------------------------------------------------------
    # i. Post-Stage-1 Health Score & Privacy-Safe Metadata Summary
    # -------------------------------------------------------------------------
    stage1_health_score = compute_health_score(cleaned_df, protected_cols)

    # Anomaly counts by type
    anomaly_counts: Dict[str, int] = {}
    for r in repairs:
        anomaly_counts[r.anomaly_type.value] = anomaly_counts.get(r.anomaly_type.value, 0) + 1
    for u in unresolved_issues:
        anomaly_counts[u.anomaly_type.value] = anomaly_counts.get(u.anomaly_type.value, 0) + 1

    # Low cardinality distributions (categorical columns with <= 30 unique non-null values)
    low_card_dist: Dict[str, Dict[str, int]] = {}
    for col in cleaned_df.columns:
        col_lower = str(col).lower()
        tokens = [t.lower() for t in re.split(r"[\s_\-]+", col_lower) if t]
        is_name = any(t in ["name", "fname", "lname", "first_name", "last_name"] for t in tokens)
        if (
            col not in numeric_cols
            and col not in all_null_numeric_cols
            and col not in date_cols
            and col not in protected_cols
            and not is_name
        ):
            non_null_vals = [str(v) for v in cleaned_df[col] if not pd.isna(v) and str(v).strip() != ""]
            unique_vals = set(non_null_vals)
            if 0 < len(unique_vals) <= 30:
                val_counts = pd.Series(non_null_vals).value_counts().to_dict()
                low_card_dist[col] = {str(k): int(v) for k, v in val_counts.items()}

    metadata_summary: Dict[str, Any] = {
        "total_rows": int(len(cleaned_df)),
        "total_columns": int(len(cleaned_df.columns)),
        "column_dtypes": {col: str(cleaned_df[col].dtype) for col in cleaned_df.columns},
        "null_counts": {col: int(cleaned_df[col].isna().sum()) for col in cleaned_df.columns},
        "anomaly_counts_by_type": anomaly_counts,
        "protected_identifier_columns": protected_cols,
        "low_cardinality_distributions": low_card_dist,
    }

    return Stage1Result(
        cleaned_df=cleaned_df,
        repairs=repairs,
        unresolved_issues=unresolved_issues,
        ks_drift_checks=ks_drift_checks,
        warnings=warnings,
        protected_identifier_columns=protected_cols,
        original_rows=original_rows,
        duplicates_removed=duplicates_removed,
        before_health_score=before_health_score,
        stage1_health_score=stage1_health_score,
        metadata_summary=metadata_summary,
    )
