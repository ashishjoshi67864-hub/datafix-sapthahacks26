"""Pydantic v2 contracts (EXPLAIN + TRACK): dataset summaries, quality reports, root-cause diagnoses, confidence-tiered repairs, unresolved issues, and DataFixResponse."""

from enum import Enum
from typing import Any, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator


# ============================================================================
# Enums
# ============================================================================

class CleaningStage(str, Enum):
    STAGE_1 = "stage_1_deterministic"
    STAGE_2 = "stage_2_semantic_ai"


class PipelineStatus(str, Enum):
    COMPLETED = "completed"
    COMPLETED_STAGE1_ONLY = "completed_stage1_only"
    FAILED = "failed"


class ConfidenceTier(str, Enum):
    AUTO_SAFE = "auto_safe"                  # confidence >= 0.90
    REVIEW_RECOMMENDED = "review_recommended" # 0.60 <= confidence < 0.90
    QUARANTINED = "quarantined"              # confidence < 0.60


class AnomalyType(str, Enum):
    HTML_TAG = "html_tag"
    WHITESPACE = "whitespace"
    EXACT_DUPLICATE = "exact_duplicate"
    TYPE_MISMATCH = "type_mismatch"
    MISSING_NUMERIC = "missing_numeric"
    INVALID_FORMAT = "invalid_format"
    INVALID_DOMAIN_VALUE = "invalid_domain_value"
    RELATIONAL_MISMATCH = "relational_mismatch"
    TEMPORAL_VIOLATION = "temporal_violation"
    CATEGORICAL_INCONSISTENCY = "categorical_inconsistency"
    SPELLING_ERROR = "spelling_error"
    MISSING_CATEGORICAL = "missing_categorical"
    MULTIVARIATE_OUTLIER = "multivariate_outlier"
    NEAR_DUPLICATE = "near_duplicate"


class RepairMethod(str, Enum):
    DETERMINISTIC_HTML_STRIP = "deterministic_html_strip"
    DETERMINISTIC_WHITESPACE = "deterministic_whitespace"
    DETERMINISTIC_DEDUP = "deterministic_dedup"
    DETERMINISTIC_TYPE_CAST = "deterministic_type_cast"
    DETERMINISTIC_RELATIONAL_FIX = "deterministic_relational_fix"
    DETERMINISTIC_KNN_IMPUTE = "deterministic_knn_impute"
    AI_CATEGORICAL_STANDARDIZE = "ai_categorical_standardize"
    AI_SPELLING_FIX = "ai_spelling_fix"
    AI_CATEGORICAL_IMPUTE = "ai_categorical_impute"


# ============================================================================
# Helper Functions
# ============================================================================

def classify_confidence_tier(confidence: float) -> ConfidenceTier:
    """Classifies a numeric confidence score (0.0 - 1.0) into deterministic tiers."""
    if confidence >= 0.90:
        return ConfidenceTier.AUTO_SAFE
    elif confidence >= 0.60:
        return ConfidenceTier.REVIEW_RECOMMENDED
    else:
        return ConfidenceTier.QUARANTINED


# ============================================================================
# Core Repair, Validation, Diagnosis & API Models
# ============================================================================

class RepairItem(BaseModel):
    row_index: int = Field(..., ge=0, description="0-based row index from the original CSV")
    column: Optional[str] = Field(default=None, description="Column name; None for row-level operations such as deduplication")
    original_value: Optional[Any] = None
    corrected_value: Optional[Any] = None
    anomaly_type: AnomalyType
    repair_method: RepairMethod
    stage: CleaningStage
    confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_tier: Optional[ConfidenceTier] = None
    reason: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_or_populate_confidence_tier(self) -> "RepairItem":
        expected_tier = classify_confidence_tier(self.confidence)
        if self.confidence_tier is None:
            self.confidence_tier = expected_tier
        elif self.confidence_tier != expected_tier:
            raise ValueError(
                f"confidence_tier '{self.confidence_tier}' does not match expected tier '{expected_tier}' for confidence {self.confidence}"
            )
        return self


class UnresolvedIssue(BaseModel):
    row_index: int = Field(..., ge=0, description="0-based row index from the original CSV")
    column: Optional[str] = Field(default=None, description="Column name; None for row-level operations")
    original_value: Optional[Any] = None
    anomaly_type: AnomalyType
    confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_tier: Optional[ConfidenceTier] = None
    reason: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_or_populate_confidence_tier(self) -> "UnresolvedIssue":
        expected_tier = classify_confidence_tier(self.confidence)
        if self.confidence_tier is None:
            self.confidence_tier = expected_tier
        elif self.confidence_tier != expected_tier:
            raise ValueError(
                f"confidence_tier '{self.confidence_tier}' does not match expected tier '{expected_tier}' for confidence {self.confidence}"
            )
        return self


class KSDriftResult(BaseModel):
    column: str = Field(..., min_length=1)
    ks_statistic: float = Field(..., ge=0.0, le=1.0)
    p_value: float = Field(..., ge=0.0, le=1.0)
    status: Literal["passed", "rolled_back", "skipped"]


class QualityReport(BaseModel):
    before_health_score: float = Field(..., ge=0.0, le=100.0)
    after_health_score: float = Field(..., ge=0.0, le=100.0)
    input_sha256: str = Field(..., min_length=1)
    output_sha256: str = Field(..., min_length=1)
    ks_drift_checks: List[KSDriftResult] = Field(default_factory=list)


class RootCauseDiagnosis(BaseModel):
    cluster_id: str = Field(..., min_length=1)
    affected_columns: List[str] = Field(default_factory=list)
    likely_cause: str = Field(..., min_length=1)
    affected_rows: int = Field(..., ge=0)
    recommended_action: str = Field(..., min_length=1)


class DatasetSummary(BaseModel):
    original_rows: int = Field(..., ge=0)
    cleaned_rows: int = Field(..., ge=0)
    columns: List[str]
    total_repairs: int = Field(..., ge=0)
    stage1_repairs: int = Field(..., ge=0)
    stage2_repairs: int = Field(..., ge=0)
    duplicates_removed: int = Field(..., ge=0)
    unresolved_count: int = Field(default=0, ge=0)
    protected_identifier_columns: List[str] = Field(default_factory=list)


class DataFixResponse(BaseModel):
    status: PipelineStatus
    filename: str = Field(..., min_length=1)
    summary: DatasetSummary
    quality_report: QualityReport
    root_cause_diagnoses: List[RootCauseDiagnosis] = Field(default_factory=list)
    repairs: List[RepairItem] = Field(default_factory=list)
    unresolved_issues: List[UnresolvedIssue] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    cleaned_csv: str


# ============================================================================
# Stage 2 Local LLM Structured-Output Proposal Models (Extra Forbidden)
# ============================================================================

class ValueMappingProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    original_value: str
    standardized_value: str
    reason: str = Field(..., min_length=1)


class ColumnStandardizationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str
    mappings: List[ValueMappingProposal] = Field(default_factory=list)


class MissingValueProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row_index: int = Field(..., ge=0)
    column: str
    proposed_value: str
    evidence: str = Field(..., min_length=1)


class MissingValueInferenceOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposals: List[MissingValueProposal] = Field(default_factory=list)


class RootCauseOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    diagnoses: List[RootCauseDiagnosis] = Field(default_factory=list)
