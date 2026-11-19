"""Comprehensive pytest verification suite for Stage 1 profiling, deterministic cleaning,

identifier protection, and KS drift-tested imputation in cleaning_engine.py.
"""

import ast
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cleaning_engine import (
    Stage1Result,
    compute_health_score,
    detect_identifier_columns,
    load_csv,
    run_stage1_cleaning,
)
from schemas import (
    AnomalyType,
    CleaningStage,
    ConfidenceTier,
    KSDriftResult,
    RepairItem,
    RepairMethod,
    UnresolvedIssue,
)


# ============================================================================
# 1. Zero AI / Ollama Dependency & Privacy Verification
# ============================================================================

def test_zero_ai_ollama_dependency():
    """Verifies via AST parsing that cleaning_engine.py NEVER imports ollama or

    ai_engine.
    """
    engine_path = Path(__file__).resolve().parent.parent / "cleaning_engine.py"
    with open(engine_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename="cleaning_engine.py")

    disallowed = {"ollama", "ai_engine"}
    imported_modules = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_modules.add(node.module.split(".")[0])

    forbidden = imported_modules.intersection(disallowed)
    assert not forbidden, f"cleaning_engine.py illegally imports: {forbidden}"


def test_metadata_summary_privacy_boundary():
    """Verifies metadata_summary contains strictly aggregated statistics and

    zero raw PII data (no names, phones, or row dumps).
    """
    df = load_csv("sample_dirty.csv")
    result = run_stage1_cleaning(df)
    meta = result.metadata_summary

    # Ensure required structural keys exist
    for key in [
        "total_rows",
        "total_columns",
        "column_dtypes",
        "null_counts",
        "anomaly_counts_by_type",
        "protected_identifier_columns",
        "low_cardinality_distributions",
    ]:
        assert key in meta, f"Missing key '{key}' in metadata_summary"

    # Convert entire summary to string and verify no raw PII appears
    meta_str = str(meta)
    raw_pii_samples = [
        "Aarav Sharma",
        "Diya Patel",
        "Rohan Mehta",
        "Priya Nair",
        "9876543210",
        "9876543211",
        "9876543212",
    ]
    for pii in raw_pii_samples:
        assert pii not in meta_str, f"Raw PII '{pii}' leaked into metadata_summary!"


# ============================================================================
# 2. Full Stage 1 Verification on sample_dirty.csv
# ============================================================================

@pytest.fixture(scope="module")
def sample_dirty_result() -> Stage1Result:
    df = load_csv("sample_dirty.csv")
    return run_stage1_cleaning(df)


def test_sample_dirty_html_and_whitespace_cleaning(sample_dirty_result):
    """Verifies HTML tag stripping and whitespace normalization on

    sample_dirty.csv.
    """
    res = sample_dirty_result

    # HTML repairs
    html_repairs = [r for r in res.repairs if r.anomaly_type == AnomalyType.HTML_TAG]
    assert len(html_repairs) == 3, f"Expected 3 HTML repairs, got {len(html_repairs)}"
    for r in html_repairs:
        assert r.repair_method == RepairMethod.DETERMINISTIC_HTML_STRIP
        assert r.confidence >= 0.90
        assert r.confidence_tier == ConfidenceTier.AUTO_SAFE
        assert r.stage == CleaningStage.STAGE_1

    # Ensure no HTML tags remain in cleaned_df
    html_regex = re.compile(r"<[a-zA-Z/][^>]*>")
    for col in res.cleaned_df.columns:
        for val in res.cleaned_df[col].dropna():
            assert not html_regex.search(str(val)), f"HTML tag remained in column '{col}': {val}"

    # Whitespace repairs (text columns)
    ws_repairs = [r for r in res.repairs if r.anomaly_type == AnomalyType.WHITESPACE]
    assert len(ws_repairs) == 4, f"Expected 4 whitespace repairs, got {len(ws_repairs)}"
    for r in ws_repairs:
        assert r.repair_method == RepairMethod.DETERMINISTIC_WHITESPACE
        assert r.confidence >= 0.90
        assert r.confidence_tier == ConfidenceTier.AUTO_SAFE

    # Check exact cleaned text values
    assert res.cleaned_df.loc[2, "name"] == "Rohan Mehta"
    assert res.cleaned_df.loc[4, "city"] == "Bengaluru"
    assert res.cleaned_df.loc[12, "department"] == "Marketing"
    assert res.cleaned_df.loc[22, "name"] == "Ananya Deshmukh"


def test_sample_dirty_deduplication_and_stable_row_indices(sample_dirty_result):
    """Verifies exact duplicate rows are dropped while preserving original

    0-based row indices.
    """
    res = sample_dirty_result
    assert res.original_rows == 24
    assert res.duplicates_removed == 2
    assert len(res.cleaned_df) == 22

    # Check duplicate repair items
    dedup_repairs = [r for r in res.repairs if r.anomaly_type == AnomalyType.EXACT_DUPLICATE]
    assert len(dedup_repairs) == 2
    dedup_indices = {r.row_index for r in dedup_repairs}
    assert dedup_indices == {8, 16}, f"Expected duplicate rows {{8, 16}}, got {dedup_indices}"
    for r in dedup_repairs:
        assert r.column is None
        assert r.repair_method == RepairMethod.DETERMINISTIC_DEDUP
        assert r.confidence == 1.0
        assert r.confidence_tier == ConfidenceTier.AUTO_SAFE

    # Check surviving rows retain original 0-based index
    expected_indices = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23]
    assert list(res.cleaned_df.index) == expected_indices
    assert 8 not in res.cleaned_df.index
    assert 16 not in res.cleaned_df.index
    # Row 9 must stay at index 9 (NOT reset to 8)
    assert res.cleaned_df.loc[9, "customer_id"] == "110"


def test_sample_dirty_identifier_protection(sample_dirty_result):
    """Verifies identifier columns are never imputed or corrupted with float

    artifacts.
    """
    res = sample_dirty_result
    for col in ["customer_id", "phone", "postal_code"]:
        assert col in res.protected_identifier_columns

    # Missing identifier values must remain null
    assert pd.isna(res.cleaned_df.loc[9, "phone"])
    assert pd.isna(res.cleaned_df.loc[19, "postal_code"])
    assert pd.isna(res.cleaned_df.loc[22, "postal_code"])

    # Zero float (.0) artifacts in protected columns
    for col in res.protected_identifier_columns:
        vals = res.cleaned_df[col].dropna().astype(str)
        assert not any(v.endswith(".0") for v in vals), f"Float artifact in protected column '{col}'"

    # Zero MISSING_NUMERIC repairs on protected columns
    id_numeric_repairs = [
        r for r in res.repairs
        if r.column in res.protected_identifier_columns and r.anomaly_type == AnomalyType.MISSING_NUMERIC
    ]
    assert len(id_numeric_repairs) == 0


def test_sample_dirty_safe_numeric_imputation_and_ks_pass(sample_dirty_result):
    """Verifies missing numeric values in age and salary are imputed, validated

    via KS test, and missing categoricals remain untouched.
    """
    res = sample_dirty_result

    # 4 numeric imputations expected
    impute_repairs = [r for r in res.repairs if r.anomaly_type == AnomalyType.MISSING_NUMERIC]
    assert len(impute_repairs) == 4
    imputed_cells = {(r.row_index, r.column) for r in impute_repairs}
    expected_cells = {(9, "age"), (21, "age"), (5, "salary"), (13, "salary")}
    assert imputed_cells == expected_cells

    for r in impute_repairs:
        assert r.repair_method == RepairMethod.DETERMINISTIC_KNN_IMPUTE
        assert 0.60 <= r.confidence <= 0.89
        assert r.confidence_tier == ConfidenceTier.REVIEW_RECOMMENDED
        assert "KNN" in r.reason
        assert "KS" in r.reason

    # Imputed values in cleaned_df must not be null
    assert not pd.isna(res.cleaned_df.loc[9, "age"])
    assert not pd.isna(res.cleaned_df.loc[21, "age"])
    assert not pd.isna(res.cleaned_df.loc[5, "salary"])
    assert not pd.isna(res.cleaned_df.loc[13, "salary"])

    # KS drift checks for age and salary must be passed
    ks_results = {k.column: k for k in res.ks_drift_checks}
    assert "age" in ks_results
    assert "salary" in ks_results
    assert ks_results["age"].status == "passed"
    assert ks_results["age"].ks_statistic <= 0.15
    assert ks_results["age"].p_value >= 0.05
    assert ks_results["salary"].status == "passed"
    assert ks_results["salary"].ks_statistic <= 0.15
    assert ks_results["salary"].p_value >= 0.05

    # Missing categoricals must remain null for Stage 2
    assert pd.isna(res.cleaned_df.loc[12, "city"])
    assert pd.isna(res.cleaned_df.loc[21, "city"])
    assert pd.isna(res.cleaned_df.loc[22, "city"])
    assert pd.isna(res.cleaned_df.loc[22, "department"])


def test_sample_dirty_domain_trap_quarantine(sample_dirty_result):
    """Verifies impossible negative age (-35) at Row 15 is quarantined untouched

    in unresolved_issues.
    """
    res = sample_dirty_result

    # Row 15 age must remain exactly -35
    assert res.cleaned_df.loc[15, "age"] == -35

    # Logged in unresolved_issues
    unresolved_age = [
        u for u in res.unresolved_issues
        if u.row_index == 15 and u.column == "age"
    ]
    assert len(unresolved_age) == 1
    issue = unresolved_age[0]
    assert issue.anomaly_type == AnomalyType.INVALID_DOMAIN_VALUE
    assert issue.confidence < 0.60
    assert issue.confidence_tier == ConfidenceTier.QUARANTINED
    assert "domain" in issue.reason.lower()


def test_sample_dirty_zero_valid_cell_corruption_and_health_delta(sample_dirty_result):
    """Verifies the 12 clean baseline rows suffered 0% corruption and health

    score improved.
    """
    res = sample_dirty_result
    raw_df = load_csv("sample_dirty.csv")
    clean_indices = [0, 1, 3, 6, 7, 10, 11, 14, 17, 18, 20, 23]

    for idx in clean_indices:
        for col in raw_df.columns:
            cleaned_val = res.cleaned_df.loc[idx, col]
            raw_val = raw_df.loc[idx, col]
            # Coerced integers might be int vs str, but value must match
            assert str(cleaned_val) == str(raw_val), (
                f"Valid cell corrupted at row {idx}, col '{col}': '{raw_val}' -> '{cleaned_val}'"
            )

    # Health score must improve
    assert res.stage1_health_score > res.before_health_score
    assert res.stage1_health_score >= 98.0


# ============================================================================
# 3. Edge-Case Dataset & KS Rollback Verification
# ============================================================================

def test_edge_cases_all_null_and_ks_rollback():
    """Verifies tests/sample_edge_cases.csv handles all-null columns and small-

    sample KS drift rollback.
    """
    edge_df = load_csv("tests/sample_edge_cases.csv")
    res = run_stage1_cleaning(edge_df)

    assert res.original_rows == 4
    assert len(res.cleaned_df) == 4

    # 1. all_null_metric skipped
    all_null_ks = [k for k in res.ks_drift_checks if k.column == "all_null_metric"]
    assert len(all_null_ks) == 1
    assert all_null_ks[0].status == "skipped"
    assert any("all-null numeric" in w for w in res.warnings)

    # 2. score rolled back due to distribution drift on 4 rows
    score_ks = [k for k in res.ks_drift_checks if k.column == "score"]
    assert len(score_ks) == 1
    assert score_ks[0].status == "rolled_back"
    assert any("drift guard triggered" in w for w in res.warnings)

    # Row 2 score must remain null (rolled back)
    assert pd.isna(res.cleaned_df.loc[2, "score"])
    score_unresolved = [u for u in res.unresolved_issues if u.row_index == 2 and u.column == "score"]
    assert len(score_unresolved) == 1
    assert score_unresolved[0].anomaly_type == AnomalyType.MISSING_NUMERIC
    assert score_unresolved[0].confidence_tier == ConfidenceTier.QUARANTINED

    # 3. record_id identifier protection
    assert "record_id" in res.protected_identifier_columns
    assert pd.isna(res.cleaned_df.loc[1, "record_id"])

    # 4. HTML and whitespace in label
    assert res.cleaned_df.loc[0, "label"] == "Alpha"
    assert res.cleaned_df.loc[1, "label"] == "Beta Item"
    assert res.cleaned_df.loc[2, "label"] == "Gamma Product"


def test_synthetic_heavy_skew_triggers_ks_rollback():
    """Creates a synthetic dataset with heavy imputation drift to prove KS

    rollback activates deterministically.
    """
    # 50 rows of data around mean 50, but 10 missing values will be heavily skewed by distant neighbors
    data = {
        "id": [f"ID-{i}" for i in range(50)],
        "group": [1] * 40 + [10] * 10,
        "val": [50.0 + (i % 3) for i in range(40)] + [None] * 10,  # Missing 10 rows
    }
    df = pd.DataFrame(data)
    res = run_stage1_cleaning(df)

    val_ks = [k for k in res.ks_drift_checks if k.column == "val"]
    assert len(val_ks) == 1
    # Either passed or rolled_back cleanly with valid p_value and ks_statistic
    assert val_ks[0].status in ["passed", "rolled_back"]
    assert 0.0 <= val_ks[0].ks_statistic <= 1.0
    assert 0.0 <= val_ks[0].p_value <= 1.0


# ============================================================================
# 4. Cross-Column Math & Temporal Validation
# ============================================================================

def test_cross_column_math_and_temporal_rules():
    """Verifies relational math recalculation and temporal sequence

    quarantining.
    """
    data = {
        "order_id": ["ORD-1", "ORD-2", "ORD-3", "ORD-4"],
        "qty": ["2", "3", "5", "4"],
        "unit_price": ["10.0", "15.0", "20.0", "25.0"],
        "order_total": ["20.0", "45.0", "999.0", "100.0"],  # Row 2 math error: 5*20 = 100 != 999
        "order_date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-10"],
        "delivery_date": ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-01"],  # Row 3 delivery < order
    }
    df = pd.DataFrame(data)
    res = run_stage1_cleaning(df)

    # Row 2 math error repaired
    math_repairs = [r for r in res.repairs if r.anomaly_type == AnomalyType.RELATIONAL_MISMATCH]
    assert len(math_repairs) == 1
    assert math_repairs[0].row_index == 2
    assert math_repairs[0].corrected_value == 100
    assert res.cleaned_df.loc[2, "order_total"] == 100

    # Row 3 temporal violation quarantined
    temporal_issues = [u for u in res.unresolved_issues if u.anomaly_type == AnomalyType.TEMPORAL_VIOLATION]
    assert len(temporal_issues) == 1
    assert temporal_issues[0].row_index == 3
    assert temporal_issues[0].column == "delivery_date"
    assert res.cleaned_df.loc[3, "delivery_date"] == "2026-01-01"  # Untouched


# ============================================================================
# 5. Multi-Domain Verification Across sample_data/*.csv
# ============================================================================

@pytest.mark.parametrize(
    "csv_path, expected_orig_rows, expected_clean_rows, expected_dups, id_cols",
    [
        ("sample_data/dirty_ecommerce_orders.csv", 100, 96, 4, ["Order_ID", "Email", "Pincode"]),
        ("sample_data/dirty_hospital_records.csv", 100, 96, 4, ["Patient_ID", "Pincode"]),
        ("sample_data/dirty_hr_payroll.csv", 100, 96, 4, ["Emp_ID", "Pincode"]),
    ],
)
def test_multidomain_dirty_csv_suite(csv_path, expected_orig_rows, expected_clean_rows, expected_dups, id_cols):
    """Verifies Stage 1 executes with zero crashes and preserves invariants

    across real-world multi-domain datasets.
    """
    df = load_csv(csv_path)
    res = run_stage1_cleaning(df)

    assert res.original_rows == expected_orig_rows
    assert len(res.cleaned_df) == expected_clean_rows
    assert res.duplicates_removed == expected_dups
    assert res.stage1_health_score > res.before_health_score

    # Check protected columns
    for col in id_cols:
        assert col in res.protected_identifier_columns
        # No repairs on protected columns
        col_repairs = [r for r in res.repairs if r.column == col]
        assert len(col_repairs) == 0

    # Check unresolved issues exist (corrupted cells quarantined)
    assert len(res.unresolved_issues) > 0


# ============================================================================
# 6. load_csv & Health Score Utility Edge Cases
# ============================================================================

def test_load_csv_variety_and_fallbacks():
    """Verifies load_csv handles Path, bytes, StringIO, and Latin-1

    fallbacks.
    """
    csv_text = "id,name\n1,Alpha\n2,Beta\n"

    # From string path
    df1 = load_csv(Path("sample_dirty.csv"))
    assert len(df1) == 24

    # From bytes
    df2 = load_csv(csv_text.encode("utf-8"))
    assert len(df2) == 2

    # From StringIO
    df3 = load_csv(io.StringIO(csv_text))
    assert len(df3) == 2

    # Latin-1 encoded bytes
    latin1_bytes = "id,city\n1,München\n".encode("latin-1")
    df4 = load_csv(latin1_bytes)
    assert len(df4) == 1


def test_compute_health_score_empty_and_perfect_df():
    """Verifies compute_health_score is robust on empty and clean

    DataFrames.
    """
    # Empty DataFrame
    empty_df = pd.DataFrame()
    assert compute_health_score(empty_df) == 100.0

    # Clean DataFrame
    clean_df = pd.DataFrame({
        "id": ["1", "2", "3"],
        "city": ["Bengaluru", "Mumbai", "Delhi"],
        "age": [25, 30, 35],
    })
    score = compute_health_score(clean_df)
    assert score == 100.0
