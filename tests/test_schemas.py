"""Unit test suite for DataFix Pydantic v2 schemas and API contracts."""

import pytest
from pydantic import ValidationError

from schemas import (
    AnomalyType,
    CleaningStage,
    ColumnStandardizationOutput,
    ConfidenceTier,
    DataFixResponse,
    DatasetSummary,
    KSDriftResult,
    MissingValueInferenceOutput,
    MissingValueProposal,
    PipelineStatus,
    QualityReport,
    RepairItem,
    RepairMethod,
    RootCauseDiagnosis,
    RootCauseOutput,
    UnresolvedIssue,
    ValueMappingProposal,
    classify_confidence_tier,
)


# ============================================================================
# 1. Helper Function Tests
# ============================================================================

def test_classify_confidence_tier():
    assert classify_confidence_tier(0.95) == ConfidenceTier.AUTO_SAFE
    assert classify_confidence_tier(0.90) == ConfidenceTier.AUTO_SAFE
    assert classify_confidence_tier(0.89) == ConfidenceTier.REVIEW_RECOMMENDED
    assert classify_confidence_tier(0.60) == ConfidenceTier.REVIEW_RECOMMENDED
    assert classify_confidence_tier(0.59) == ConfidenceTier.QUARANTINED
    assert classify_confidence_tier(0.10) == ConfidenceTier.QUARANTINED
    assert classify_confidence_tier(0.0) == ConfidenceTier.QUARANTINED


# ============================================================================
# 2. Confidence Tier Auto-population & Disagreement Validator Tests
# ============================================================================

def test_repair_item_confidence_tier_auto_population():
    item_safe = RepairItem(
        row_index=0,
        column="salary",
        original_value=" 50000 ",
        corrected_value=50000,
        anomaly_type=AnomalyType.WHITESPACE,
        repair_method=RepairMethod.DETERMINISTIC_WHITESPACE,
        stage=CleaningStage.STAGE_1,
        confidence=0.96,
        reason="Stripped surrounding whitespace",
    )
    assert item_safe.confidence_tier == ConfidenceTier.AUTO_SAFE

    item_review = RepairItem(
        row_index=1,
        column="age",
        original_value=None,
        corrected_value=29.0,
        anomaly_type=AnomalyType.MISSING_NUMERIC,
        repair_method=RepairMethod.DETERMINISTIC_KNN_IMPUTE,
        stage=CleaningStage.STAGE_1,
        confidence=0.75,
        reason="KNN imputation passing KS-test",
    )
    assert item_review.confidence_tier == ConfidenceTier.REVIEW_RECOMMENDED

    item_quarantined = RepairItem(
        row_index=2,
        column="city",
        original_value="xyz",
        corrected_value="Bengaluru",
        anomaly_type=AnomalyType.CATEGORICAL_INCONSISTENCY,
        repair_method=RepairMethod.AI_CATEGORICAL_STANDARDIZE,
        stage=CleaningStage.STAGE_2,
        confidence=0.45,
        reason="Low similarity mapping",
    )
    assert item_quarantined.confidence_tier == ConfidenceTier.QUARANTINED


def test_repair_item_explicit_confidence_tier_conflict_raises_error():
    with pytest.raises(ValidationError) as exc_info:
        RepairItem(
            row_index=0,
            column="salary",
            original_value=None,
            corrected_value=1000,
            anomaly_type=AnomalyType.MISSING_NUMERIC,
            repair_method=RepairMethod.DETERMINISTIC_KNN_IMPUTE,
            stage=CleaningStage.STAGE_1,
            confidence=0.95,
            confidence_tier=ConfidenceTier.QUARANTINED,  # Conflict: 0.95 should be AUTO_SAFE
            reason="Contradictory tier test",
        )
    assert "does not match expected tier" in str(exc_info.value)


def test_unresolved_issue_confidence_tier_auto_population_and_conflict():
    issue = UnresolvedIssue(
        row_index=5,
        column="age",
        original_value=-35,
        anomaly_type=AnomalyType.INVALID_DOMAIN_VALUE,
        confidence=0.10,
        reason="Negative age outside valid domain [0, 120]",
    )
    assert issue.confidence_tier == ConfidenceTier.QUARANTINED

    with pytest.raises(ValidationError) as exc_info:
        UnresolvedIssue(
            row_index=5,
            column="age",
            original_value=-35,
            anomaly_type=AnomalyType.INVALID_DOMAIN_VALUE,
            confidence=0.10,
            confidence_tier=ConfidenceTier.AUTO_SAFE,  # Conflict
            reason="Contradictory tier test",
        )
    assert "does not match expected tier" in str(exc_info.value)


# ============================================================================
# 3. Row-Level Repair with column=None (Exact Duplicate)
# ============================================================================

def test_row_level_repair_with_none_column():
    item = RepairItem(
        row_index=14,
        column=None,
        original_value="Duplicate Row Content",
        corrected_value=None,
        anomaly_type=AnomalyType.EXACT_DUPLICATE,
        repair_method=RepairMethod.DETERMINISTIC_DEDUP,
        stage=CleaningStage.STAGE_1,
        confidence=0.99,
        reason="Dropped exact duplicate row preserving original 0-based index",
    )
    assert item.column is None
    assert item.confidence_tier == ConfidenceTier.AUTO_SAFE
    assert item.anomaly_type == AnomalyType.EXACT_DUPLICATE


# ============================================================================
# 4. JSON Round-Trip Serialization & Deserialization
# ============================================================================

def test_models_json_round_trip():
    # 1. RepairItem
    repair = RepairItem(
        row_index=3,
        column="order_total",
        original_value=250.0,
        corrected_value=25.0,
        anomaly_type=AnomalyType.RELATIONAL_MISMATCH,
        repair_method=RepairMethod.DETERMINISTIC_RELATIONAL_FIX,
        stage=CleaningStage.STAGE_1,
        confidence=0.98,
        reason="Recalculated order_total = qty * unit_price (5 * 5.0)",
    )
    repair_json = repair.model_dump_json()
    repair_back = RepairItem.model_validate_json(repair_json)
    assert repair == repair_back

    # 2. UnresolvedIssue
    issue = UnresolvedIssue(
        row_index=7,
        column="email",
        original_value="bad-email-format",
        anomaly_type=AnomalyType.INVALID_FORMAT,
        confidence=0.20,
        reason="Unparseable email format",
    )
    issue_json = issue.model_dump_json()
    issue_back = UnresolvedIssue.model_validate_json(issue_json)
    assert issue == issue_back

    # 3. KSDriftResult
    ks = KSDriftResult(
        column="salary",
        ks_statistic=0.08,
        p_value=0.45,
        status="passed",
    )
    ks_json = ks.model_dump_json()
    ks_back = KSDriftResult.model_validate_json(ks_json)
    assert ks == ks_back

    # 4. QualityReport
    report = QualityReport(
        before_health_score=68.5,
        after_health_score=94.2,
        input_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        output_sha256="ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb",
        ks_drift_checks=[ks],
    )
    report_json = report.model_dump_json()
    report_back = QualityReport.model_validate_json(report_json)
    assert report == report_back

    # 5. RootCauseDiagnosis
    diag = RootCauseDiagnosis(
        cluster_id="RC-01",
        affected_columns=["order_date", "delivery_date"],
        likely_cause="Timezone / date format swap during upstream ETL export",
        affected_rows=4,
        recommended_action="Standardize ingestion format to ISO-8601",
    )
    diag_json = diag.model_dump_json()
    diag_back = RootCauseDiagnosis.model_validate_json(diag_json)
    assert diag == diag_back

    # 6. DatasetSummary
    summary = DatasetSummary(
        original_rows=50,
        cleaned_rows=48,
        columns=["customer_id", "order_date", "order_total"],
        total_repairs=10,
        stage1_repairs=8,
        stage2_repairs=2,
        duplicates_removed=2,
        unresolved_count=1,
        protected_identifier_columns=["customer_id"],
    )
    summary_json = summary.model_dump_json()
    summary_back = DatasetSummary.model_validate_json(summary_json)
    assert summary == summary_back

    # 7. DataFixResponse
    response = DataFixResponse(
        status=PipelineStatus.COMPLETED,
        filename="sample_dirty.csv",
        summary=summary,
        quality_report=report,
        root_cause_diagnoses=[diag],
        repairs=[repair],
        unresolved_issues=[issue],
        warnings=[],
        cleaned_csv="customer_id,order_date,order_total\n101,2026-01-01,25.0\n",
    )
    response_json = response.model_dump_json()
    response_back = DataFixResponse.model_validate_json(response_json)
    assert response == response_back
    assert response_back.status == PipelineStatus.COMPLETED
    assert len(response_back.repairs) == 1
    assert response_back.repairs[0].confidence_tier == ConfidenceTier.AUTO_SAFE


# ============================================================================
# 5. Stage 2 Offline Fallback Response Contract
# ============================================================================

def test_stage2_offline_fallback_contract():
    summary = DatasetSummary(
        original_rows=30,
        cleaned_rows=30,
        columns=["id", "name", "city"],
        total_repairs=5,
        stage1_repairs=5,
        stage2_repairs=0,
        duplicates_removed=0,
        unresolved_count=2,
        protected_identifier_columns=["id"],
    )
    report = QualityReport(
        before_health_score=70.0,
        after_health_score=85.0,
        input_sha256="abc123hash",
        output_sha256="def456hash",
        ks_drift_checks=[],
    )
    fallback_diagnosis = RootCauseDiagnosis(
        cluster_id="FALLBACK-DETERMINISTIC",
        affected_columns=["city"],
        likely_cause="Categorical variance detected; Ollama was offline so Stage 2 AI inference was bypassed",
        affected_rows=2,
        recommended_action="Review unresolved categorical issues or run with active Ollama daemon",
    )
    response = DataFixResponse(
        status=PipelineStatus.COMPLETED_STAGE1_ONLY,
        filename="data.csv",
        summary=summary,
        quality_report=report,
        root_cause_diagnoses=[fallback_diagnosis],
        repairs=[],
        unresolved_issues=[],
        warnings=["Ollama daemon offline. Stage 1 completed successfully; Stage 2 bypassed with deterministic fallback."],
        cleaned_csv="id,name,city\n1,Alice,BLR\n",
    )
    assert response.status == PipelineStatus.COMPLETED_STAGE1_ONLY
    assert len(response.warnings) == 1
    assert "Ollama daemon offline" in response.warnings[0]
    assert len(response.root_cause_diagnoses) == 1
    assert response.root_cause_diagnoses[0].cluster_id == "FALLBACK-DETERMINISTIC"


# ============================================================================
# 6. Negative Tests & Validation Errors
# ============================================================================

def test_invalid_confidence_bounds():
    # confidence < 0.0
    with pytest.raises(ValidationError):
        RepairItem(
            row_index=0,
            column="col",
            original_value=None,
            corrected_value=None,
            anomaly_type=AnomalyType.WHITESPACE,
            repair_method=RepairMethod.DETERMINISTIC_WHITESPACE,
            stage=CleaningStage.STAGE_1,
            confidence=-0.1,
            reason="Invalid negative confidence",
        )

    # confidence > 1.0
    with pytest.raises(ValidationError):
        RepairItem(
            row_index=0,
            column="col",
            original_value=None,
            corrected_value=None,
            anomaly_type=AnomalyType.WHITESPACE,
            repair_method=RepairMethod.DETERMINISTIC_WHITESPACE,
            stage=CleaningStage.STAGE_1,
            confidence=1.05,
            reason="Invalid excess confidence",
        )


def test_invalid_row_index():
    with pytest.raises(ValidationError):
        RepairItem(
            row_index=-1,
            column="col",
            original_value=None,
            corrected_value=None,
            anomaly_type=AnomalyType.WHITESPACE,
            repair_method=RepairMethod.DETERMINISTIC_WHITESPACE,
            stage=CleaningStage.STAGE_1,
            confidence=0.9,
            reason="Invalid negative row index",
        )


def test_empty_reason_validation_error():
    with pytest.raises(ValidationError):
        RepairItem(
            row_index=0,
            column="col",
            original_value=None,
            corrected_value=None,
            anomaly_type=AnomalyType.WHITESPACE,
            repair_method=RepairMethod.DETERMINISTIC_WHITESPACE,
            stage=CleaningStage.STAGE_1,
            confidence=0.9,
            reason="",  # min_length=1 violated
        )


def test_invalid_health_score_bounds():
    with pytest.raises(ValidationError):
        QualityReport(
            before_health_score=105.0,  # ge=0.0, le=100.0 violated
            after_health_score=90.0,
            input_sha256="sha1",
            output_sha256="sha2",
        )


def test_invalid_ks_drift_status():
    with pytest.raises(ValidationError):
        KSDriftResult(
            column="salary",
            ks_statistic=0.1,
            p_value=0.5,
            status="unknown_status",  # Literal["passed", "rolled_back", "skipped"]
        )


# ============================================================================
# 7. LLM Output Contract Tests (extra="forbid" & No Hallucinated Confidence)
# ============================================================================

def test_llm_proposal_models_valid_instantiation():
    mapping = ValueMappingProposal(
        original_value="BLR",
        standardized_value="Bengaluru",
        reason="Common airport abbreviation for city",
    )
    col_output = ColumnStandardizationOutput(
        column="city",
        mappings=[mapping],
    )
    assert col_output.column == "city"
    assert len(col_output.mappings) == 1

    missing_prop = MissingValueProposal(
        row_index=12,
        column="department",
        proposed_value="Engineering",
        evidence="Title 'Software Engineer' strongly associates with 'Engineering'",
    )
    missing_output = MissingValueInferenceOutput(proposals=[missing_prop])
    assert len(missing_output.proposals) == 1

    root_output = RootCauseOutput(
        diagnoses=[
            RootCauseDiagnosis(
                cluster_id="RC-01",
                affected_columns=["order_date", "delivery_date"],
                likely_cause="Cross-column date discrepancy",
                affected_rows=3,
                recommended_action="Inspect source system timestamps",
            )
        ]
    )
    assert len(root_output.diagnoses) == 1


def test_llm_proposal_models_forbid_extra_confidence_field():
    # ValueMappingProposal must forbid extra="confidence"
    with pytest.raises(ValidationError) as exc:
        ValueMappingProposal(
            original_value="BLR",
            standardized_value="Bengaluru",
            reason="Abbreviation",
            confidence=0.95,  # type: ignore
        )
    assert "Extra inputs are not permitted" in str(exc.value)

    # ColumnStandardizationOutput must forbid extra="confidence"
    with pytest.raises(ValidationError) as exc:
        ColumnStandardizationOutput(
            column="city",
            mappings=[],
            confidence=0.90,  # type: ignore
        )
    assert "Extra inputs are not permitted" in str(exc.value)

    # MissingValueProposal must forbid extra="confidence"
    with pytest.raises(ValidationError) as exc:
        MissingValueProposal(
            row_index=1,
            column="dept",
            proposed_value="HR",
            evidence="Evidence text",
            confidence=0.88,  # type: ignore
        )
    assert "Extra inputs are not permitted" in str(exc.value)

    # MissingValueInferenceOutput must forbid extra="confidence"
    with pytest.raises(ValidationError) as exc:
        MissingValueInferenceOutput(
            proposals=[],
            confidence=0.85,  # type: ignore
        )
    assert "Extra inputs are not permitted" in str(exc.value)

    # RootCauseOutput must forbid extra="confidence"
    with pytest.raises(ValidationError) as exc:
        RootCauseOutput(
            diagnoses=[],
            confidence=0.99,  # type: ignore
        )
    assert "Extra inputs are not permitted" in str(exc.value)
