"""Stage 2 (DIAGNOSE + EXPLAIN): local Ollama detective operating strictly on summarized error metadata and unique value counts (never raw rows) with a deterministic rule-based fallback."""

import difflib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from cleaning_engine import Stage1Result, compute_health_score
from schemas import (
    AnomalyType,
    CleaningStage,
    ColumnStandardizationOutput,
    ConfidenceTier,
    MissingValueInferenceOutput,
    RepairItem,
    RepairMethod,
    RootCauseDiagnosis,
    RootCauseOutput,
    UnresolvedIssue,
)


# ============================================================================
# Dataclasses & Configuration
# ============================================================================

@dataclass
class Stage2Result:
    cleaned_df: pd.DataFrame
    repairs: List[RepairItem]
    unresolved_issues: List[UnresolvedIssue]
    root_cause_diagnoses: List[RootCauseDiagnosis]
    warnings: List[str]
    mode_used: str  # "ollama" or "deterministic_fallback"
    ollama_used: bool
    model_name: str


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

FREE_TEXT_NAME_TOKENS = {
    "name",
    "fname",
    "lname",
    "label",
    "note",
    "comment",
    "address",
    "description",
    "email",
    "phone",
    "id",
}

NUMERIC_TOKENS = {
    "amount",
    "salary",
    "bill",
    "age",
    "price",
    "qty",
    "quantity",
    "total",
    "score",
    "metric",
    "count",
    "num",
    "cost",
}


# ============================================================================
# Health Check Helper
# ============================================================================

def check_ollama_health(model: Optional[str] = None, timeout: float = 3.0) -> Dict[str, Any]:
    """Checks whether the local Ollama daemon is reachable and whether the
    target model is available locally. Never raises an unhandled exception.
    """
    target_model = model or os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

    try:
        import ollama
        client = ollama.Client(timeout=timeout)
        models_resp = client.list()

        available_models: List[str] = []
        if hasattr(models_resp, "models"):
            for m in models_resp.models:
                name = getattr(m, "model", None) or getattr(m, "name", None)
                if name:
                    available_models.append(str(name))
        elif isinstance(models_resp, dict) and "models" in models_resp:
            for m in models_resp["models"]:
                if isinstance(m, dict) and "name" in m:
                    available_models.append(str(m["name"]))
                elif hasattr(m, "model"):
                    available_models.append(str(m.model))

        def _matches(target: str, cand: str) -> bool:
            if target == cand or cand.startswith(target) or target.startswith(cand):
                return True
            t_base = target.split(":")[0]
            c_base = cand.split(":")[0]
            return t_base == c_base

        is_available = any(_matches(target_model, m) for m in available_models)
        return {
            "reachable": True,
            "model": target_model,
            "model_available": is_available,
            "detail": f"Ollama is reachable. Model '{target_model}' {'available' if is_available else 'not found'}. Available: {available_models}",
        }
    except Exception as e:
        return {
            "reachable": False,
            "model": target_model,
            "model_available": False,
            "detail": f"Ollama connection error: {e}",
        }


# ============================================================================
# Stage 2 Semantic Cleaning Engine
# ============================================================================

def run_stage2_semantic_cleaning(
    stage1_result: Stage1Result,
    llm_mode: Optional[str] = None,
    model: Optional[str] = None,
) -> Stage2Result:
    """Executes Stage 2 AI detective and semantic repair pipeline.

    1. Identifies low-cardinality categorical columns (zero raw rows sent).
    2. Standardizes categorical variants, abbreviations, and spelling errors.
    3. Mines clean functional dependencies to impute missing categorical cells (or quarantine uncertain ones).
    4. Synthesizes systemic RootCauseDiagnosis clusters.
    """
    df = stage1_result.cleaned_df.copy()
    repairs: List[RepairItem] = []
    unresolved_issues: List[UnresolvedIssue] = []
    root_cause_diagnoses: List[RootCauseDiagnosis] = []
    warnings: List[str] = list(stage1_result.warnings)

    mode = (llm_mode or os.environ.get("LLM_MODE", "auto")).lower()
    target_model = model or os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
    ollama_timeout = float(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "30.0"))

    # Determine execution path
    ollama_ready = False
    if mode in ["auto", "ollama"]:
        health = check_ollama_health(model=target_model, timeout=3.0)
        if health["reachable"] and health["model_available"]:
            ollama_ready = True
        else:
            warnings.append(
                f"Ollama inactive or model '{target_model}' unavailable ({health['detail']}); using deterministic fallback."
            )

    mode_used = "ollama" if ollama_ready else "deterministic_fallback"
    ollama_used = ollama_ready

    # Identify eligible categorical columns (exclude protected identifiers, names, dates, and numeric columns)
    categorical_cols: List[str] = []
    for col in df.columns:
        if col in stage1_result.protected_identifier_columns:
            continue
        col_lower = str(col).lower()
        tokens = [t.lower() for t in re.split(r"[\s_\-]+", col_lower) if t]
        if any(t in FREE_TEXT_NAME_TOKENS for t in tokens):
            continue
        if any(t in ["date", "dob", "joining", "admission", "discharge"] for t in tokens):
            continue
        if any(t in NUMERIC_TOKENS for t in tokens):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            continue

        non_nulls = [str(v) for v in df[col] if not pd.isna(v) and str(v).strip() != ""]
        if not non_nulls:
            continue

        # Check if values parse as numbers
        parsed_num_count = 0
        for v in non_nulls:
            clean_v = re.sub(r"[₹\$€£\s,]|Rs\.?", "", str(v))
            try:
                float(clean_v)
                parsed_num_count += 1
            except ValueError:
                pass
        if parsed_num_count / len(non_nulls) >= 0.50:
            continue

        unique_vals = set(non_nulls)
        if 2 <= len(unique_vals) <= 30:
            categorical_cols.append(col)

    # -------------------------------------------------------------------------
    # a. Categorical & Spelling Standardization
    # -------------------------------------------------------------------------
    for col in categorical_cols:
        series = df[col]
        non_null_vals = [str(v) for v in series if not pd.isna(v) and str(v).strip() != ""]
        if not non_null_vals:
            continue

        # Establish canonical targets from dominant observed values
        val_counts = pd.Series(non_null_vals).value_counts()
        lower_groups: Dict[str, List[str]] = {}
        for v in val_counts.index:
            lower_groups.setdefault(v.lower(), []).append(v)

        canonical_targets: Dict[str, str] = {}  # lower_val -> canonical_form
        dominant_values: Set[str] = set()

        for l_key, variants in lower_groups.items():
            # Pick Title Case variant or most frequent variant
            best_variant = sorted(variants, key=lambda x: (x.istitle(), val_counts.get(x, 0)), reverse=True)[0]
            canonical_targets[l_key] = best_variant
            if sum(val_counts.get(x, 0) for x in variants) >= 2 or len(lower_groups) == 1:
                dominant_values.add(best_variant)

        if not dominant_values:
            dominant_values = set(val_counts.head(4).index)

        # Mapping dictionary: orig_str -> (target_str, AnomalyType, RepairMethod, confidence)
        mapping_dict: Dict[str, Tuple[str, AnomalyType, RepairMethod, float]] = {}

        # Determine if this column has potential variants (avoid unnecessary LLM calls)
        has_potential_variants = False
        if len(lower_groups) > 1:
            has_potential_variants = True
        else:
            for v, cnt in val_counts.items():
                if cnt <= 2 and any(c > 2 for c in val_counts.values()):
                    has_potential_variants = True
                    break
                if str(v).upper() in KNOWN_CITY_ABBREVIATIONS or str(v).upper() in KNOWN_DEPT_ABBREVIATIONS:
                    has_potential_variants = True
                    break
                if "city" in col.lower() and str(v).upper() in KNOWN_CITY_TYPOS:
                    has_potential_variants = True
                    break

        # 1. Ollama Path (Metadata-only: sends only {column, value_counts})
        if ollama_ready and has_potential_variants:
            try:
                import ollama
                prompt = (
                    f"You are a tabular data standardization expert.\n"
                    f"Column: '{col}'\n"
                    f"Observed value frequencies (JSON):\n{json.dumps(val_counts.to_dict())}\n\n"
                    f"Task: Identify any informal abbreviations (e.g. 'BLR' -> 'Bengaluru'), typographical errors (e.g. 'Banglore' -> 'Bengaluru', 'Enginering' -> 'Engineering'), or casing variations.\n"
                    f"RULE: The standardized_value MUST be one of the dominant observed values: {list(dominant_values)}.\n"
                    f"RULE: Do NOT include any 'confidence' score in your output.\n"
                    f"Return ONLY a JSON object strictly matching this schema:\n"
                    f'{{"column": "{col}", "mappings": [{{"original_value": "...", "standardized_value": "...", "reason": "..."}}]}}'
                )
                client = ollama.Client(timeout=ollama_timeout)
                response = client.chat(
                    model=target_model,
                    messages=[{"role": "user", "content": prompt}],
                    format="json",
                    keep_alive="10m",
                    options={"temperature": 0, "num_predict": 512},
                )
                content = response["message"]["content"]
                parsed = ColumnStandardizationOutput.model_validate_json(content)

                for item in parsed.mappings:
                    orig = item.original_value
                    target = item.standardized_value
                    # Verification Gate: target must be in dominant_values and not equal to orig
                    if target in dominant_values and orig != target and orig in val_counts.index:
                        is_abbrev = orig.upper() in KNOWN_CITY_ABBREVIATIONS or orig.upper() in KNOWN_DEPT_ABBREVIATIONS
                        is_case = orig.lower() == target.lower()
                        ratio = difflib.SequenceMatcher(None, orig.lower(), target.lower()).ratio()

                        if is_case:
                            mapping_dict[orig] = (target, AnomalyType.CATEGORICAL_INCONSISTENCY, RepairMethod.AI_CATEGORICAL_STANDARDIZE, 0.95)
                        elif is_abbrev:
                            mapping_dict[orig] = (target, AnomalyType.CATEGORICAL_INCONSISTENCY, RepairMethod.AI_CATEGORICAL_STANDARDIZE, 0.90)
                        else:
                            conf = round(float(np.clip(0.50 * ratio + 0.35 + 0.05, 0.72, 0.91)), 2)
                            mapping_dict[orig] = (target, AnomalyType.SPELLING_ERROR, RepairMethod.AI_SPELLING_FIX, conf)
            except Exception as e:
                warnings.append(f"Ollama standardization failed on column '{col}': {e}; using deterministic fallback.")

        # 2. Deterministic Fallback & Supplementary Path (applied if Ollama missed anything or was inactive)
        for orig in val_counts.index:
            if orig in mapping_dict:
                continue

            # a. Casing fix
            canonical_form = canonical_targets.get(orig.lower())
            if canonical_form and orig != canonical_form:
                mapping_dict[orig] = (
                    canonical_form,
                    AnomalyType.CATEGORICAL_INCONSISTENCY,
                    RepairMethod.AI_CATEGORICAL_STANDARDIZE,
                    0.94,
                )
                continue

            # b. Known domain abbreviation
            abbrev_map = KNOWN_CITY_ABBREVIATIONS if "city" in col.lower() else KNOWN_DEPT_ABBREVIATIONS
            if orig.upper() in abbrev_map:
                target_cand = abbrev_map[orig.upper()]
                matched_target = next((d for d in dominant_values if d.lower() == target_cand.lower()), None)
                if matched_target and orig != matched_target:
                    mapping_dict[orig] = (
                        matched_target,
                        AnomalyType.CATEGORICAL_INCONSISTENCY,
                        RepairMethod.AI_CATEGORICAL_STANDARDIZE,
                        0.90,
                    )
                    continue

            # c. Known city typos (e.g. Banglore -> Bengaluru, Mumbay -> Mumbai, New Dehli -> Delhi)
            if "city" in col.lower() and orig.upper() in KNOWN_CITY_TYPOS:
                target_cand = KNOWN_CITY_TYPOS[orig.upper()]
                matched_target = next((d for d in dominant_values if d.lower() == target_cand.lower()), None)
                if matched_target and orig != matched_target:
                    mapping_dict[orig] = (
                        matched_target,
                        AnomalyType.SPELLING_ERROR,
                        RepairMethod.AI_SPELLING_FIX,
                        0.88,
                    )
                    continue

            # d. Spelling typos (fuzzy matching against dominant values)
            best_match: Optional[str] = None
            best_ratio = 0.0
            for cand in dominant_values:
                ratio = difflib.SequenceMatcher(None, orig.lower(), cand.lower()).ratio()
                clean_orig = re.sub(r"^(new|old|north|south|east|west)\s+", "", orig.lower()).strip()
                clean_ratio = difflib.SequenceMatcher(None, clean_orig, cand.lower()).ratio()
                effective_ratio = max(ratio, clean_ratio)

                if effective_ratio > best_ratio:
                    best_ratio = effective_ratio
                    best_match = cand

            if best_match and best_ratio >= 0.72 and orig != best_match:
                conf = round(float(np.clip(0.50 * best_ratio + 0.35 * (val_counts.get(best_match, 0) / len(non_null_vals)) + 0.15, 0.72, 0.91)), 2)
                mapping_dict[orig] = (
                    best_match,
                    AnomalyType.SPELLING_ERROR,
                    RepairMethod.AI_SPELLING_FIX,
                    conf,
                )

        # Apply verified mappings to DataFrame
        for idx in df.index:
            cell_val = df.at[idx, col]
            if not pd.isna(cell_val) and str(cell_val) in mapping_dict:
                target, anom_type, method, conf = mapping_dict[str(cell_val)]
                repairs.append(
                    RepairItem(
                        row_index=int(idx),
                        column=col,
                        original_value=cell_val,
                        corrected_value=target,
                        anomaly_type=anom_type,
                        repair_method=method,
                        stage=CleaningStage.STAGE_2,
                        confidence=conf,
                        reason=f"Standardized '{cell_val}' -> '{target}' in column '{col}'",
                    )
                )
                df.at[idx, col] = target

    # -------------------------------------------------------------------------
    # b. Evidence-Backed Missing Categorical Imputation vs. Quarantine
    # -------------------------------------------------------------------------
    # Mine functional dependency (FD) rules from post-standardized data
    key_col_candidates = [
        c for c in df.columns
        if any(t in str(c).lower() for t in ["postal", "pincode", "zip", "code"])
    ]

    # Map: (key_col, target_col) -> dict of key_value: (target_value, support, purity)
    fd_rules: Dict[Tuple[str, str], Dict[str, Tuple[str, int, float]]] = {}

    for key_c in key_col_candidates:
        for tgt_c in categorical_cols:
            if key_c == tgt_c:
                continue
            pair_df = df[[key_c, tgt_c]].dropna()
            if len(pair_df) < 2:
                continue

            rule_lookup: Dict[str, Tuple[str, int, float]] = {}
            for k_val, group in pair_df.groupby(key_c):
                tgt_counts = group[tgt_c].value_counts()
                if not tgt_counts.empty:
                    top_tgt = tgt_counts.index[0]
                    support = int(tgt_counts.iloc[0])
                    purity = float(tgt_counts.iloc[0] / tgt_counts.sum())
                    if support >= 2 and purity >= 0.85:
                        rule_lookup[str(k_val).strip()] = (str(top_tgt), support, purity)

            if rule_lookup:
                fd_rules[(key_c, tgt_c)] = rule_lookup

    # Identify candidate missing cells in non-protected categorical columns
    candidate_imputations: List[Dict[str, Any]] = []
    for tgt_c in categorical_cols:
        for idx in df.index:
            cell_val = df.at[idx, tgt_c]
            if pd.isna(cell_val) or str(cell_val).strip() == "":
                for key_c in key_col_candidates:
                    k_val = df.at[idx, key_c]
                    if not pd.isna(k_val) and str(k_val).strip() != "":
                        candidate_imputations.append({
                            "row_index": int(idx),
                            "column": tgt_c,
                            "key_column": key_c,
                            "key_value": str(k_val).strip(),
                        })
                        break

    # 1. Ollama Path for missing categorical inference
    imputed_cells: Set[Tuple[int, str]] = set()
    if ollama_ready and candidate_imputations and fd_rules:
        try:
            import ollama
            serialized_fd = {
                f"{k_col}->{t_col}": {
                    k: {"inferred_value": v[0], "support": v[1], "purity": round(v[2], 2)}
                    for k, v in rules.items()
                }
                for (k_col, t_col), rules in fd_rules.items()
            }
            prompt = (
                f"You are a tabular data imputation expert.\n"
                f"Aggregated functional dependency statistics (JSON):\n{json.dumps(serialized_fd)}\n\n"
                f"Missing categorical cells to impute (JSON):\n{json.dumps(candidate_imputations)}\n\n"
                f"Task: Impute missing values ONLY when backed by the functional dependency statistics with high purity (>=0.85) and support (>=2).\n"
                f"RULE: Return ONLY the exact keys shown; do not add extra keys like 'inferences' or 'confidence'.\n"
                f"Return ONLY a JSON object strictly matching this schema:\n"
                f'{{"proposals": [{{"row_index": 0, "column": "...", "proposed_value": "...", "evidence": "..."}}]}}'
            )
            client = ollama.Client(timeout=ollama_timeout)
            response = client.chat(
                model=target_model,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                keep_alive="10m",
                options={"temperature": 0, "num_predict": 512},
            )
            content = response["message"]["content"]
            parsed_inf = MissingValueInferenceOutput.model_validate_json(content)

            items = getattr(parsed_inf, "proposals", None) or getattr(parsed_inf, "inferences", [])
            for inf in items:
                idx = inf.row_index
                tgt_c = inf.column
                inferred_val = getattr(inf, "proposed_value", None) or getattr(inf, "inferred_value", None)
                reason_str = getattr(inf, "evidence", None) or getattr(inf, "reason", "")
                if idx in df.index and (pd.isna(df.at[idx, tgt_c]) or str(df.at[idx, tgt_c]).strip() == ""):
                    for key_c in key_col_candidates:
                        k_val = str(df.at[idx, key_c]).strip()
                        rule = fd_rules.get((key_c, tgt_c), {}).get(k_val)
                        if rule and rule[0] == inferred_val:
                            target_val, support, purity = rule
                            conf = round(float(np.clip(0.65 + 0.25 * purity + min(0.04, 0.01 * support), 0.60, 0.94)), 2)
                            repairs.append(
                                RepairItem(
                                    row_index=int(idx),
                                    column=tgt_c,
                                    original_value=None,
                                    corrected_value=target_val,
                                    anomaly_type=AnomalyType.MISSING_CATEGORICAL,
                                    repair_method=RepairMethod.AI_CATEGORICAL_IMPUTE,
                                    stage=CleaningStage.STAGE_2,
                                    confidence=conf,
                                    reason=f"Imputed missing '{tgt_c}' as '{target_val}' based on {key_c}='{k_val}' (support={support}, purity={round(purity, 2)}): {reason_str}",
                                )
                            )
                            df.at[idx, tgt_c] = target_val
                            imputed_cells.add((idx, tgt_c))
                            break
        except Exception as e:
            warnings.append(f"Ollama missing categorical inference failed: {e}; using deterministic fallback.")

    # 2. Deterministic Fallback & Quarantine for all remaining missing cells
    for tgt_c in categorical_cols:
        for idx in df.index:
            if (idx, tgt_c) in imputed_cells:
                continue
            cell_val = df.at[idx, tgt_c]
            if pd.isna(cell_val) or str(cell_val).strip() == "":
                imputed = False
                for key_c in key_col_candidates:
                    k_val = df.at[idx, key_c]
                    if not pd.isna(k_val) and str(k_val).strip() != "":
                        k_str = str(k_val).strip()
                        rule = fd_rules.get((key_c, tgt_c), {}).get(k_str)
                        if rule:
                            target_val, support, purity = rule
                            conf = round(float(np.clip(0.65 + 0.25 * purity + min(0.04, 0.01 * support), 0.60, 0.94)), 2)
                            repairs.append(
                                RepairItem(
                                    row_index=int(idx),
                                    column=tgt_c,
                                    original_value=None,
                                    corrected_value=target_val,
                                    anomaly_type=AnomalyType.MISSING_CATEGORICAL,
                                    repair_method=RepairMethod.AI_CATEGORICAL_IMPUTE,
                                    stage=CleaningStage.STAGE_2,
                                    confidence=conf,
                                    reason=f"Imputed missing '{tgt_c}' as '{target_val}' based on {key_c}='{k_str}' (support={support}, purity={round(purity, 2)})",
                                )
                            )
                            df.at[idx, tgt_c] = target_val
                            imputed = True
                            imputed_cells.add((idx, tgt_c))
                            break

                if not imputed:
                    unresolved_issues.append(
                        UnresolvedIssue(
                            row_index=int(idx),
                            column=tgt_c,
                            original_value=None,
                            anomaly_type=AnomalyType.MISSING_CATEGORICAL,
                            confidence=0.35,
                            reason=f"Insufficient cross-column evidence to impute missing '{tgt_c}' at row {idx} safely without hallucination",
                        )
                    )

    # -------------------------------------------------------------------------
    # c. Systemic Root-Cause Diagnosis
    # -------------------------------------------------------------------------
    all_repairs = stage1_result.repairs + repairs
    all_unresolved = stage1_result.unresolved_issues + unresolved_issues

    type_counts: Dict[str, int] = {}
    type_cols: Dict[str, Set[str]] = {}
    type_rows: Dict[str, Set[int]] = {}

    for r in all_repairs:
        t_val = r.anomaly_type.value
        type_counts[t_val] = type_counts.get(t_val, 0) + 1
        if r.column:
            type_cols.setdefault(t_val, set()).add(r.column)
        type_rows.setdefault(t_val, set()).add(r.row_index)

    for u in all_unresolved:
        t_val = u.anomaly_type.value
        type_counts[t_val] = type_counts.get(t_val, 0) + 1
        if u.column:
            type_cols.setdefault(t_val, set()).add(u.column)
        type_rows.setdefault(t_val, set()).add(u.row_index)

    # 1. Ollama Path for Root Cause
    if ollama_ready:
        try:
            import ollama
            summary_payload = {
                "anomaly_counts": type_counts,
                "columns_affected": {k: sorted(list(v)) for k, v in type_cols.items()},
                "total_repairs": len(all_repairs),
                "total_unresolved": len(all_unresolved),
            }
            prompt = (
                f"You are a master tabular data quality detective.\n"
                f"Aggregated anomaly metrics (JSON):\n{json.dumps(summary_payload)}\n\n"
                f"Synthesize 2 to 4 systemic RootCauseDiagnosis clusters describing upstream systemic causes.\n"
                f"RULE: Do NOT include any extra keys. Every diagnosis must have 'cluster_id', 'affected_columns', 'likely_cause', 'affected_rows', and 'recommended_action'.\n"
                f"Return ONLY a JSON object strictly matching this schema:\n"
                f'{{"diagnoses": [{{"cluster_id": "RC-...", "affected_columns": ["col1"], "likely_cause": "...", "affected_rows": 1, "recommended_action": "..."}}]}}'
            )
            client = ollama.Client(timeout=ollama_timeout)
            response = client.chat(
                model=target_model,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                keep_alive="10m",
                options={"temperature": 0, "num_predict": 512},
            )
            content = response["message"]["content"]
            parsed_rc = RootCauseOutput.model_validate_json(content)
            if parsed_rc.diagnoses:
                root_cause_diagnoses = parsed_rc.diagnoses
        except Exception as e:
            warnings.append(f"Ollama root-cause diagnosis failed: {e}; using deterministic fallback.")

    # 2. Deterministic Fallback Synthesis
    if not root_cause_diagnoses:
        if "html_tag" in type_counts or "whitespace" in type_counts:
            cols = sorted(list(type_cols.get("html_tag", set()).union(type_cols.get("whitespace", set()))))
            rows = len(type_rows.get("html_tag", set()).union(type_rows.get("whitespace", set())))
            root_cause_diagnoses.append(
                RootCauseDiagnosis(
                    cluster_id="RC-SCRAPE-WS",
                    affected_columns=cols,
                    likely_cause="Unsanitized web scraping or rich-text form copy-paste introduced embedded HTML markup and excess whitespace.",
                    affected_rows=rows,
                    recommended_action="Deploy automated HTML tag stripping and whitespace normalization middleware at ingestion.",
                )
            )

        if "categorical_inconsistency" in type_counts or "spelling_error" in type_counts:
            cols = sorted(list(type_cols.get("categorical_inconsistency", set()).union(type_cols.get("spelling_error", set()))))
            rows = len(type_rows.get("categorical_inconsistency", set()).union(type_rows.get("spelling_error", set())))
            root_cause_diagnoses.append(
                RootCauseDiagnosis(
                    cluster_id="RC-CATEGORICAL-DRIFT",
                    affected_columns=cols,
                    likely_cause="Unconstrained free-text fields permitted informal abbreviations (e.g. airport codes), casing variance, and spelling typos.",
                    affected_rows=rows,
                    recommended_action="Replace free-text inputs with controlled enum dropdowns and client-side autocomplete validation.",
                )
            )

        if "exact_duplicate" in type_counts or "relational_mismatch" in type_counts:
            cols = sorted(list(type_cols.get("exact_duplicate", set()).union(type_cols.get("relational_mismatch", set()))))
            rows = len(type_rows.get("exact_duplicate", set()).union(type_rows.get("relational_mismatch", set())))
            root_cause_diagnoses.append(
                RootCauseDiagnosis(
                    cluster_id="RC-INGEST-RETRY",
                    affected_columns=cols if cols else ["entire_row"],
                    likely_cause="Non-idempotent ETL re-runs or duplicate webhook event delivery produced duplicate rows and out-of-sync totals.",
                    affected_rows=rows,
                    recommended_action="Enforce database uniqueness constraints and idempotent API request idempotency keys.",
                )
            )

        if "invalid_domain_value" in type_counts or "invalid_format" in type_counts or "temporal_violation" in type_counts:
            cols = sorted(list(type_cols.get("invalid_domain_value", set()).union(type_cols.get("invalid_format", set())).union(type_cols.get("temporal_violation", set()))))
            rows = len(type_rows.get("invalid_domain_value", set()).union(type_rows.get("invalid_format", set())).union(type_rows.get("temporal_violation", set())))
            root_cause_diagnoses.append(
                RootCauseDiagnosis(
                    cluster_id="RC-DOMAIN-BOUNDS",
                    affected_columns=cols,
                    likely_cause="Missing boundary and chronological checks allowed impossible domain values (e.g. negative numbers) and unparseable formats.",
                    affected_rows=rows,
                    recommended_action="Implement server-side Pydantic contract validation and SQL range check constraints.",
                )
            )

        if "missing_numeric" in type_counts or "missing_categorical" in type_counts:
            cols = sorted(list(type_cols.get("missing_numeric", set()).union(type_cols.get("missing_categorical", set()))))
            rows = len(type_rows.get("missing_numeric", set()).union(type_rows.get("missing_categorical", set())))
            root_cause_diagnoses.append(
                RootCauseDiagnosis(
                    cluster_id="RC-SPARSE-INPUT",
                    affected_columns=cols,
                    likely_cause="Optional user input fields and dropped foreign keys in upstream ETL left-joins generated sparse attribute records.",
                    affected_rows=rows,
                    recommended_action="Make critical business attributes required and audit upstream database join integrity.",
                )
            )

    return Stage2Result(
        cleaned_df=df,
        repairs=repairs,
        unresolved_issues=unresolved_issues,
        root_cause_diagnoses=root_cause_diagnoses,
        warnings=warnings,
        mode_used=mode_used,
        ollama_used=ollama_used,
        model_name=target_model,
    )
