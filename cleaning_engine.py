"""Stage 1 (DETECT + REPAIR + VALIDATE): deterministic Pandas/NumPy/SciPy/BS4/Scikit-Learn profiling, identifier protection, safe repairs, and KS-test drift rollback. Never calls an LLM."""

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
    """Computes a deterministic 0.0-100.0 quality score.

    Weighted as:
    - Completeness (30%)
    - Uniqueness (20%)
    - Structural/Format/Domain Validity (25%)
    - Consistency/Relational Integrity (25%)
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    total_cells = total_rows * total_cols

    if total_rows == 0 or total_cells == 0:
        return 100.0

    # 1. Completeness
    null_cells = 0
    for col in df.columns:
        s = df[col]
        null_cells += s.apply(
            lambda v: pd.isna(v) or str(v).strip().lower() in NULL_PLACEHOLDERS
        ).sum()
    completeness = max(0.0, min(1.0, 1.0 - (null_cells / total_cells)))

    # 2. Uniqueness
    dup_rows = df.duplicated().sum()
    uniqueness = max(0.0, min(1.0, 1.0 - (dup_rows / total_rows)))

    # 3. Structural, Format & Domain Validity
    violations = 0
    for col in df.columns:
        col_lower = str(col).lower()
        is_age = "age" in col_lower
        is_fin_or_qty = any(k in col_lower for k in ["amount", "bill", "salary", "price", "qty", "quantity", "total"])
        is_email = "email" in col_lower

        for val in df[col]:
            if pd.isna(val) or str(val).strip().lower() in NULL_PLACEHOLDERS:
                continue
            str_val = str(val)
            # HTML tags
            if HTML_PATTERN.search(str_val):
                violations += 1
                continue
            # Whitespace noise
            if str_val != str_val.strip() or "  " in str_val:
                violations += 1
                continue
            # Email validation
            if is_email and not EMAIL_REGEX.match(str_val.strip()):
                violations += 1
                continue
            # Numeric domain checks
            if is_age or is_fin_or_qty:
                clean_str = re.sub(r"[₹\$€£\s,]|Rs\.?", "", str_val)
                try:
                    num = float(clean_str)
                    if is_age and (num < 0 or num > 120):
                        violations += 1
                    elif is_fin_or_qty and num < 0:
                        violations += 1
                except ValueError:
                    pass

    validity = max(0.0, min(1.0, 1.0 - (violations / total_cells)))

    # 4. Consistency & Relational Integrity
    consistency_violations = 0
    col_map = {str(c).lower(): c for c in df.columns}
    qty_col = col_map.get("qty") or col_map.get("quantity")
    price_col = col_map.get("unit_price") or col_map.get("price")
    total_col = col_map.get("order_total") or col_map.get("total") or col_map.get("order_amount")

    if qty_col and price_col and total_col:
        for _, r in df.iterrows():
            try:
                q = float(re.sub(r"[^\d\.\-]", "", str(r[qty_col])))
                p = float(re.sub(r"[^\d\.\-]", "", str(r[price_col])))
                t = float(re.sub(r"[^\d\.\-]", "", str(r[total_col])))
                if q > 0 and p > 0 and abs(t - (q * p)) > 0.01:
                    consistency_violations += 1
            except Exception:
                pass

    consistency = max(0.0, min(1.0, 1.0 - (consistency_violations / total_rows)))

    score = 100.0 * (0.30 * completeness + 0.20 * uniqueness + 0.25 * validity + 0.25 * consistency)
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
    # a. Missing Placeholder Normalization & HTML Tag Stripping
    # -------------------------------------------------------------------------
    for col in cleaned_df.columns:
        for idx in list(cleaned_df.index):
            val = cleaned_df.at[idx, col]
            if pd.isna(val):
                continue
            str_val = str(val)

            # Strip HTML tags if present
            if HTML_PATTERN.search(str_val):
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
    # b. Whitespace Normalization (Text & identifier columns only; numeric handled in d)
    # -------------------------------------------------------------------------
    for col in cleaned_df.columns:
        if col in numeric_cols or col in all_null_numeric_cols:
            continue
        for idx in list(cleaned_df.index):
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
    # d. Date Normalization & Basic Type Coercion
    # -------------------------------------------------------------------------
    # Process Date Columns
    for col in date_cols:
        for idx in list(cleaned_df.index):
            val = cleaned_df.at[idx, col]
            if pd.isna(val) or str(val).strip() == "":
                continue
            str_val = str(val).strip()
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
    # g. Post-Stage-1 Health Score & Privacy-Safe Metadata Summary
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
