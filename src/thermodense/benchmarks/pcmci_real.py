"""Isolated PCMCI+ runs for the assembled real five-node daily product.

This is deliberately separate from :mod:`pcmci_methods`: that module is the
frozen synthetic harness and its plan and results must not change.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import random
import re
import resource
import shutil
import sys
import time
from typing import Any, Literal, TypeAlias, cast
import warnings

from filelock import FileLock, Timeout
import numpy as np
import polars as pl

from thermodense.benchmarks import runtime
from thermodense.benchmarks import gpdctorch_gates
from thermodense.benchmarks.real_data import (
    DATE_COLUMN,
    DEFAULT_OUTPUT as DEFAULT_INPUT,
    F107_RAW_COLUMN,
    IMPUTATION_MASK_COLUMNS,
    NODE_COLUMNS,
)

SCHEMA_VERSION = "4"
RUNNER_VERSION = "pcmci-real-4"
METHODS = ("parcorr", "cmiknn", "gpdctorch")
DEFAULT_METHODS = ("parcorr",)
DEFERRED_METHODS = {"gpdc": "explicitly deferred for real-data PCMCI+ runs"}
DEFAULT_TAU_MAX = 180
DEFAULT_CMIKNN_WORKERS = 24
_GATED_ARTIFACT_VALIDATION_ERRORS = (OSError, ValueError)
MISSING_FLAG = -999999.0
SEED = 20260802
ROLLING_WINDOW = 1095
STATIONARITY_ALPHA = 0.05
MIN_STATIONARITY_SAMPLES = 2 * DEFAULT_TAU_MAX + 1

RAW_OBSERVED_DAILY = "raw_observed_daily"
CENTERED_81_DAY = "centered_81_day"
DETRENDED_ANOMALY = "detrended_anomaly"
SEASONAL_ANOMALY = "seasonal_anomaly"

F107TimingVariant: TypeAlias = Literal["raw_observed_daily", "centered_81_day"]  # noqa: UP040 -- retain Python 3.11 runtime compatibility
PreprocessingProfile: TypeAlias = Literal["detrended_anomaly", "seasonal_anomaly"]  # noqa: UP040 -- retain Python 3.11 runtime compatibility


@dataclass(frozen=True)
class SensitivityCase:
    """One preregistered cell in the PCMCI sensitivity matrix."""

    timing_variant: F107TimingVariant
    preprocessing_profile: PreprocessingProfile
    role: str


class MatrixSynthesisError(ValueError):
    """The four required ParCorr sensitivity artifacts cannot be compared."""


@dataclass(frozen=True)
class _GatedChildContext:
    """Unforgeable-by-public-API authorization created after child validation."""

    stage: gpdctorch_gates.Stage


REGISTERED_SENSITIVITY_CASES = (
    SensitivityCase(RAW_OBSERVED_DAILY, DETRENDED_ANOMALY, "primary"),
    SensitivityCase(RAW_OBSERVED_DAILY, SEASONAL_ANOMALY, "robustness"),
    SensitivityCase(CENTERED_81_DAY, DETRENDED_ANOMALY, "robustness"),
    SensitivityCase(CENTERED_81_DAY, SEASONAL_ANOMALY, "interaction_diagnostic"),
)


def sensitivity_case(
    timing_variant: F107TimingVariant, preprocessing_profile: PreprocessingProfile
) -> SensitivityCase:
    """Return a preregistered matrix cell, rejecting arbitrary transformations."""
    for case in REGISTERED_SENSITIVITY_CASES:
        if (case.timing_variant, case.preprocessing_profile) == (
            timing_variant,
            preprocessing_profile,
        ):
            return case
    raise ValueError("unregistered PCMCI sensitivity case")


def expand_sensitivity_cases() -> tuple[SensitivityCase, ...]:
    """Return the complete preregistered 2×2 PCMCI sensitivity matrix."""
    return REGISTERED_SENSITIVITY_CASES


def _case_key(case: SensitivityCase) -> str:
    return f"{case.timing_variant}/{case.preprocessing_profile}"


def _physical_node(node_name: str) -> str:
    """Normalize the two F10.7 labels wherever a physical node is compared."""
    return "f10_7" if node_name in {F107_RAW_COLUMN, NODE_COLUMNS[0]} else node_name


def _existing_paths_alias(first: Path, second: Path) -> bool:
    """Return whether two existing paths name the same inode.

    ``resolve()`` normalizes spelling and symlinks, but deliberately does not
    identify hardlinks.  A missing path is not an alias yet, and inaccessible
    paths are left to their normal read/write operation to report.
    """
    try:
        return first.samefile(second)
    except OSError:
        return False


def _expected_node_order(case: SensitivityCase) -> list[str]:
    return (
        [F107_RAW_COLUMN, *NODE_COLUMNS[1:]]
        if case.timing_variant == RAW_OBSERVED_DAILY
        else NODE_COLUMNS
    )


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _normalized_f107_identity(value: Any) -> Any:
    if isinstance(value, str):
        return _physical_node(value)
    if isinstance(value, list):
        return [_normalized_f107_identity(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalized_f107_identity(item) for key, item in value.items()}
    return value


def _signed_directed_links(
    row: dict[str, Any],
    tau_max: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Verify and read only fully directed PCMCI links from one canonical artifact."""
    artifact = row.get("artifact")
    case = row.get("sensitivity_case")
    if not isinstance(artifact, dict) or not isinstance(case, dict):
        raise MatrixSynthesisError("malformed matrix row artifact or provenance")
    path = artifact.get("path")
    expected_hash = artifact.get("sha256")
    if not isinstance(path, str) or not isinstance(expected_hash, str):
        raise MatrixSynthesisError("malformed matrix artifact reference")
    artifact_path = Path(path)
    try:
        if hashlib.sha256(artifact_path.read_bytes()).hexdigest() != expected_hash:
            raise MatrixSynthesisError(
                "matrix artifact sha256 does not match provenance"
            )
        with np.load(artifact_path, allow_pickle=False) as saved:
            required = {"graph", "p_matrix", "val_matrix", "node_names"}
            if set(saved.files) != required:
                raise MatrixSynthesisError("matrix artifact canonical keys are invalid")
            graph, p_matrix, val_matrix = (
                saved[name] for name in ("graph", "p_matrix", "val_matrix")
            )
            node_names = [str(name) for name in saved["node_names"].tolist()]
    except (OSError, ValueError) as error:
        if isinstance(error, MatrixSynthesisError):
            raise
        raise MatrixSynthesisError(f"malformed matrix artifact: {error}") from error
    if not all(matrix.ndim == 3 for matrix in (graph, p_matrix, val_matrix)):
        raise MatrixSynthesisError("matrix artifacts must be three-dimensional")
    expected_shape = (len(node_names), len(node_names), tau_max + 1)
    if any(matrix.shape != expected_shape for matrix in (graph, p_matrix, val_matrix)):
        raise MatrixSynthesisError(
            "matrix artifact shape does not match the configured lag window"
        )
    if node_names != case.get("node_order"):
        raise MatrixSynthesisError(
            "artifact node axis does not match result provenance"
        )
    links: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    diagnostics = {"partially_oriented_count": 0, "conflict_count": 0}
    for source_index, target_index, lag in np.ndindex(graph.shape):
        mark = str(graph[source_index, target_index, lag])
        if mark == "-->":
            source, target = source_index, target_index
        elif mark == "<--":
            source, target = target_index, source_index
        else:
            if mark in {"o->", "<-o"}:
                diagnostics["partially_oriented_count"] += 1
            elif mark == "x-x":
                diagnostics["conflict_count"] += 1
            continue
        value = float(val_matrix[source_index, target_index, lag])
        p_value = float(p_matrix[source_index, target_index, lag])
        if not np.isfinite(value) or not np.isfinite(p_value):
            raise MatrixSynthesisError("directed matrix evidence must be finite")
        if not 0.0 <= p_value <= 1.0:
            raise MatrixSynthesisError("directed matrix p-value must be in [0, 1]")
        sign = int(np.sign(value))
        if sign == 0:
            raise MatrixSynthesisError("directed matrix value must have a nonzero sign")
        link = {
            "source": node_names[source],
            "target": node_names[target],
            "lag": int(lag),
            "sign": sign,
            "value": value,
            "p_value": p_value,
            "graph_mark": mark,
        }
        links.setdefault((link["source"], link["target"], link["lag"], sign), link)
    return (
        sorted(
            links.values(),
            key=lambda link: (
                link["source"],
                link["target"],
                link["lag"],
                link["sign"],
            ),
        ),
        diagnostics,
    )


def _matrix_rows_by_case(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list):
        raise MatrixSynthesisError("malformed matrix rows")
    expected = {_case_key(case): case for case in REGISTERED_SENSITIVITY_CASES}
    by_case: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise MatrixSynthesisError("malformed matrix row")
        case_data = row.get("sensitivity_case", {})
        if not isinstance(case_data, dict):
            raise MatrixSynthesisError("malformed matrix row provenance")
        if row.get("schema_version") != SCHEMA_VERSION:
            raise MatrixSynthesisError("matrix row schema version mismatch")
        if row.get("runner_version") != RUNNER_VERSION:
            raise MatrixSynthesisError("matrix row runner version mismatch")
        if row.get("synthetic") is not False:
            raise MatrixSynthesisError("matrix row synthetic marker mismatch")
        key = f"{case_data.get('timing_variant')}/{case_data.get('preprocessing_profile')}"
        if (
            key not in expected
            or key in by_case
            or case_data.get("role") != expected[key].role
        ):
            raise MatrixSynthesisError(
                "incomplete or duplicate registered sensitivity case"
            )
        by_case[key] = row
    if set(by_case) != set(expected):
        raise MatrixSynthesisError("incomplete four-cell sensitivity matrix")
    return by_case


def _synthesize_matrix(
    rows: list[dict[str, Any]], *, method: str, tau_max: int
) -> dict[str, Any]:
    """Classify a complete registered matrix from its canonical artifacts."""
    by_case = _matrix_rows_by_case(rows)
    ordered = [by_case[_case_key(case)] for case in REGISTERED_SENSITIVITY_CASES]
    if any(row.get("status") != "succeeded" for row in ordered):
        raise MatrixSynthesisError("incomplete matrix: one or more cells failed")
    if any(row.get("method") != method for row in ordered):
        raise MatrixSynthesisError(f"matrix method must be {method}")
    if any(row.get("tau_max") != tau_max for row in ordered):
        raise MatrixSynthesisError(f"matrix tau_max must be {tau_max}")
    reference = ordered[0]
    for name, message in (
        ("settings", "matrix settings mismatch"),
        ("algorithm", "matrix graph/inference settings mismatch"),
        ("missing_data_policy", "matrix missing-data settings mismatch"),
    ):
        if any(row.get(name) != reference.get(name) for row in ordered[1:]):
            raise MatrixSynthesisError(message)
    reference_assumptions = _normalized_f107_identity(reference.get("link_assumptions"))
    if any(
        _normalized_f107_identity(row.get("link_assumptions")) != reference_assumptions
        for row in ordered[1:]
    ):
        raise MatrixSynthesisError("matrix link-assumption settings mismatch")
    try:
        identity = reference["sensitivity_case"]["accepted_quality_rows"]
        date_hash = identity["daily_date_sequence_sha256"]
        support_hash = identity["common_f107_support"]["sha256"]
    except (KeyError, TypeError) as error:
        raise MatrixSynthesisError(
            "malformed accepted-quality-row provenance"
        ) from error
    if any(
        row["sensitivity_case"].get("accepted_quality_rows") != identity
        for row in ordered[1:]
    ):
        raise MatrixSynthesisError("matrix accepted-quality-row identity mismatch")
    if (
        not isinstance(identity.get("row_count"), int)
        or isinstance(identity["row_count"], bool)
        or identity["row_count"] <= 0
        or not _is_sha256(date_hash)
        or not _is_sha256(support_hash)
        or not isinstance(identity["common_f107_support"].get("row_count"), int)
        or isinstance(identity["common_f107_support"]["row_count"], bool)
        or not 0
        <= identity["common_f107_support"]["row_count"]
        <= identity["row_count"]
    ):
        raise MatrixSynthesisError("malformed accepted-quality-row provenance")
    normalized_node_order: list[str] | None = None
    for case, row in zip(REGISTERED_SENSITIVITY_CASES, ordered, strict=True):
        case_data = row["sensitivity_case"]
        node_order = case_data.get("node_order")
        qualification = row.get("stationarity_qualification")
        expected_node_order = _expected_node_order(case)
        if node_order != expected_node_order or not isinstance(qualification, dict):
            raise MatrixSynthesisError("malformed stationarity provenance")
        expected_algorithm = {
            "name": "PCMCI+",
            "entry_point": "PCMCI.run_pcmciplus",
            "tau_min": 0,
            "pc_alpha": 0.05,
            "contemp_collider_rule": "majority",
            "conflict_resolution": True,
            "fdr_method": "none",
        }
        expected_missing_policy = {
            "sentinel": MISSING_FLAG,
            "remove_missing_upto_maxlag": False,
            "drivers_interpolated": False,
            "rows_dropped": False,
        }
        settings = row.get("settings")
        expected_settings = (
            {"significance": "analytic", "pc_alpha": 0.05}
            if method != "cmiknn"
            else {
                "significance": "shuffle_test",
                "pc_alpha": 0.05,
                "sig_samples": 20,
                "sig_blocklength": 4,
                "knn": 0.1,
                "shuffle_neighbors": 5,
            }
        )
        if (
            row.get("algorithm") != expected_algorithm
            or row.get("missing_data_policy") != expected_missing_policy
            or row.get("link_assumptions")
            != _link_assumption_metadata(tau_max, expected_node_order)
            or not isinstance(settings, dict)
            or any(
                settings.get(name) != value for name, value in expected_settings.items()
            )
            or not isinstance(settings.get("threads"), int)
            or isinstance(settings["threads"], bool)
            or settings["threads"] <= 0
            or (
                method == "cmiknn"
                and (
                    not isinstance(settings.get("workers"), int)
                    or isinstance(settings["workers"], bool)
                    or settings["workers"] <= 0
                )
            )
        ):
            raise MatrixSynthesisError("matrix production settings provenance mismatch")
        expected_stationarity_identity = {
            "timing_variant": case.timing_variant,
            "preprocessing_profile": case.preprocessing_profile,
            "node_order": node_order,
            "daily_date_sequence_sha256": date_hash,
            "common_f107_support_sha256": support_hash,
        }
        if qualification.get("provenance_identity") != expected_stationarity_identity:
            raise MatrixSynthesisError("stationarity provenance identity mismatch")
        qualification_eligible = qualification.get("causal_interpretation_eligible")
        qualification_evidence_only = qualification.get("sensitivity_evidence_only")
        row_eligible = row.get("causal_interpretation_eligible")
        row_evidence_only = row.get("sensitivity_evidence_only")
        if not all(
            isinstance(value, bool)
            for value in (
                qualification_eligible,
                qualification_evidence_only,
                row_eligible,
                row_evidence_only,
            )
        ):
            raise MatrixSynthesisError("malformed stationarity eligibility flags")
        if (
            row_eligible != qualification_eligible
            or row_evidence_only != qualification_evidence_only
            or qualification_evidence_only != (not qualification_eligible)
        ):
            raise MatrixSynthesisError(
                "stationarity eligibility disagrees with row provenance"
            )
        normalized = [_physical_node(node) for node in node_order]
        if normalized_node_order is None:
            normalized_node_order = normalized
        elif normalized != normalized_node_order:
            raise MatrixSynthesisError("matrix normalized node order mismatch")

    try:
        artifact_data = {
            _case_key(case): _signed_directed_links(by_case[_case_key(case)], tau_max)
            for case in REGISTERED_SENSITIVITY_CASES
        }
    except MatrixSynthesisError:
        raise
    extracted = {key: value[0] for key, value in artifact_data.items()}
    diagnostics = {key: value[1] for key, value in artifact_data.items()}
    primary_key, detrending_key, timing_key, interaction_key = (
        _case_key(case) for case in REGISTERED_SENSITIVITY_CASES
    )
    primary_links: list[dict[str, Any]] = []
    for primary in extracted[primary_key]:
        detrending_matches = [
            link
            for link in extracted[detrending_key]
            if (link["source"], link["target"], link["lag"], link["sign"])
            == (primary["source"], primary["target"], primary["lag"], primary["sign"])
        ]
        centered_matches = [
            link
            for link in extracted[timing_key]
            if (
                _physical_node(link["source"]),
                _physical_node(link["target"]),
                link["sign"],
            )
            == (
                _physical_node(primary["source"]),
                _physical_node(primary["target"]),
                primary["sign"],
            )
        ]
        failed_dimensions = [
            name
            for name, matches in (
                ("detrending", detrending_matches),
                ("timing", centered_matches),
            )
            if not matches
        ]
        failed_stationarity_cells = (
            [
                key
                for key in (primary_key, detrending_key, timing_key)
                if not by_case[key]["stationarity_qualification"][
                    "causal_interpretation_eligible"
                ]
            ]
            if not failed_dimensions
            else []
        )
        primary_links.append(
            {
                **{name: primary[name] for name in ("source", "target", "lag", "sign")},
                "classification": (
                    "factor_sensitive"
                    if failed_dimensions
                    else "stationarity_limited"
                    if failed_stationarity_cells
                    else "main_text_robust"
                ),
                "failed_dimensions": failed_dimensions,
                "failed_stationarity_cells": failed_stationarity_cells,
                "centered_matches": [
                    {
                        name: match[name]
                        for name in (
                            "source",
                            "target",
                            "lag",
                            "sign",
                            "value",
                            "p_value",
                        )
                    }
                    for match in centered_matches
                ],
                "centered_delay_equivalent": False,
                "evidence": {
                    "primary": primary,
                    "detrending": detrending_matches,
                    "timing": centered_matches,
                },
            }
        )
    primary_identities = {
        (link["source"], link["target"], link["lag"], link["sign"])
        for link in extracted[primary_key]
    }
    consumed = {detrending_key: set(), timing_key: set()}
    for primary in primary_links:
        for dimension, key in (("detrending", detrending_key), ("timing", timing_key)):
            for link in primary["evidence"][dimension]:
                consumed[key].add(
                    (link["source"], link["target"], link["lag"], link["sign"])
                )
    exploratory: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for key in (detrending_key, timing_key, interaction_key):
        for link in extracted[key]:
            identity_key = (link["source"], link["target"], link["lag"], link["sign"])
            timing_related_primary = any(
                (
                    _physical_node(link["source"]),
                    _physical_node(link["target"]),
                    link["sign"],
                )
                == (
                    _physical_node(primary["source"]),
                    _physical_node(primary["target"]),
                    primary["sign"],
                )
                for primary in extracted[primary_key]
            )
            if (
                identity_key in primary_identities
                or identity_key in consumed.get(key, set())
                or (key == interaction_key and timing_related_primary)
            ):
                continue
            item = exploratory.setdefault(
                identity_key,
                {name: link[name] for name in ("source", "target", "lag", "sign")}
                | {"source_cells": [], "evidence": []},
            )
            item["source_cells"].append(key)
            item["evidence"].append({"case": key, **link})
    result = {
        "schema_version": "1",
        "implementation_version": RUNNER_VERSION,
        "state": "complete",
        "method": method,
        "physical_lag_window_days": {"min": 0, "max": tau_max},
        "accepted_quality_rows": identity,
        "settings": reference["settings"],
        "algorithm": reference["algorithm"],
        "stationarity_eligibility": {
            key: by_case[key].get("stationarity_qualification") for key in by_case
        },
        "case_artifacts": [
            {
                "case": _case_key(case),
                **{
                    name: by_case[_case_key(case)]["artifact"][name]
                    for name in ("name", "path", "sha256")
                    if name in by_case[_case_key(case)]["artifact"]
                },
                "result_digest": by_case[_case_key(case)].get("result_digest"),
            }
            for case in REGISTERED_SENSITIVITY_CASES
        ],
        "primary_links": primary_links,
        "exploratory_links": sorted(
            exploratory.values(),
            key=lambda link: (
                link["source"],
                link["target"],
                link["lag"],
                link["sign"],
            ),
        ),
        "interaction_diagnostic_links": extracted[interaction_key],
        "orientation_diagnostics": diagnostics,
    }
    if method == "cmiknn":
        result |= {
            "method_scope": "nonlinear sensitivity; not a substitute for ParCorr",
            "untested_parcorr_lags_11_180": True,
            "untested_lag_window_days": {"min": 11, "max": DEFAULT_TAU_MAX},
        }
    return result


def synthesize_parcorr_matrix(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Preserve the public 0--180-day ParCorr synthesis contract."""
    return _synthesize_matrix(rows, method="parcorr", tau_max=DEFAULT_TAU_MAX)


def synthesize_cmiknn_matrix(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Synthesize the four bounded nonlinear 0--10-day sensitivity cells."""
    return _synthesize_matrix(
        rows, method="cmiknn", tau_max=runtime.CMIKNN_MAX_TAU_STEPS
    )


def synthesize_gpdctorch_matrix(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Synthesize the four gated GPDCtorch 0--10-day cells (not capability)."""
    return _synthesize_matrix(rows, method="gpdctorch", tau_max=10)


def compare_cmiknn_with_parcorr(
    cmiknn_agreement: dict[str, Any], parcorr_agreement: dict[str, Any] | None
) -> dict[str, Any]:
    """Qualify ParCorr with bounded nonlinear evidence, without a veto rule."""
    result: dict[str, Any] = {
        "schema_version": "1",
        "method": "cmiknn",
        "method_scope": "nonlinear sensitivity; not a substitute for ParCorr",
        "lag_window_days": {"min": 0, "max": runtime.CMIKNN_MAX_TAU_STEPS},
        "not_a_substitute_for_parcorr_days_11_180": True,
        "untested_parcorr_lags_11_180": True,
        "parcorr_primary_links_outside_cmiknn_scope": [],
        "parcorr_primary_links_outside_cmiknn_scope_count": 0,
        "cmiknn_only_method_sensitive": [],
    }
    if (
        not isinstance(cmiknn_agreement, dict)
        or cmiknn_agreement.get("state") != "complete"
        or cmiknn_agreement.get("method") != "cmiknn"
        or cmiknn_agreement.get("physical_lag_window_days")
        != {"min": 0, "max": runtime.CMIKNN_MAX_TAU_STEPS}
        or not isinstance(cmiknn_agreement.get("primary_links"), list)
    ):
        raise ValueError("CMIknn agreement is incomplete or outside its bounded scope")
    if (
        not isinstance(parcorr_agreement, dict)
        or parcorr_agreement.get("state") != "complete"
    ):
        return result | {"state": "pending_parcorr", "comparisons": []}
    cmiknn = {
        (
            _physical_node(link.get("source")),
            _physical_node(link.get("target")),
            link.get("lag"),
            link.get("sign"),
        )
        for link in cmiknn_agreement["primary_links"]
        if isinstance(link, dict)
    }
    comparisons = []
    outside_scope = []
    parcorr_in_scope = set()
    for link in parcorr_agreement.get("primary_links", []):
        if not isinstance(link, dict):
            continue
        identity = (
            _physical_node(link.get("source")),
            _physical_node(link.get("target")),
            link.get("lag"),
            link.get("sign"),
        )
        lag = identity[2]
        if not isinstance(lag, int):
            continue
        if lag > runtime.CMIKNN_MAX_TAU_STEPS:
            outside_scope.append(
                {name: link.get(name) for name in ("source", "target", "lag", "sign")}
            )
            continue
        if lag < 0:
            continue
        parcorr_in_scope.add(identity)
        comparisons.append(
            {
                "source": link.get("source"),
                "target": link.get("target"),
                "lag": lag,
                "sign": identity[3],
                "cmiknn_support": "strengthens"
                if identity in cmiknn
                else "qualifies_disagreement_or_not_detected",
                "parcorr_visibility": "retained",
                "vetoed": False,
            }
        )
    cmiknn_only = [
        {
            "source": link.get("source"),
            "target": link.get("target"),
            "lag": link.get("lag"),
            "sign": link.get("sign"),
            "classification": "cmiknn_only_method_sensitive",
            "evidence": link,
        }
        for link in cmiknn_agreement["primary_links"]
        if isinstance(link, dict)
        and (
            _physical_node(link.get("source")),
            _physical_node(link.get("target")),
            link.get("lag"),
            link.get("sign"),
        )
        not in parcorr_in_scope
    ]
    return result | {
        "state": "complete",
        "comparisons": comparisons,
        "parcorr_primary_links_outside_cmiknn_scope": outside_scope,
        "parcorr_primary_links_outside_cmiknn_scope_count": len(outside_scope),
        "cmiknn_only_method_sensitive": cmiknn_only,
    }


def compare_gpdctorch_with_parcorr(
    gpdctorch_links: list[dict[str, Any]], parcorr_agreement: dict[str, Any] | None
) -> dict[str, Any]:
    """Annotate, but never veto, available ParCorr links with 0--10 day support.

    GPDCtorch is intentionally bounded to this window and is not evidence about
    ParCorr links at days 11--180.
    """
    result: dict[str, Any] = {
        "schema_version": "1",
        "method": "gpdctorch",
        "lag_window_days": {"min": 0, "max": 10},
        "not_a_substitute_for_parcorr_days_11_180": True,
        "parcorr_primary_links_outside_gpdc_scope": [],
        "parcorr_primary_links_outside_gpdc_scope_count": 0,
        "gpdctorch_only_method_sensitive": [],
    }
    if (
        not isinstance(parcorr_agreement, dict)
        or parcorr_agreement.get("state") != "complete"
    ):
        return result | {"state": "pending_parcorr", "comparisons": []}
    gpdc = {
        (link.get("source"), link.get("target"), link.get("lag"), link.get("sign"))
        for link in gpdctorch_links
    }
    comparisons = []
    outside_scope = []
    parcorr_in_scope = set()
    for link in parcorr_agreement.get("primary_links", []):
        identity = (
            link.get("source"),
            link.get("target"),
            link.get("lag"),
            link.get("sign"),
        )
        lag = identity[2]
        if not isinstance(lag, int):
            continue
        if lag > 10:
            outside_scope.append(
                {name: link.get(name) for name in ("source", "target", "lag", "sign")}
            )
            continue
        if lag < 0:
            continue
        parcorr_in_scope.add(identity)
        comparisons.append(
            {
                "source": identity[0],
                "target": identity[1],
                "lag": lag,
                "sign": identity[3],
                "gpdctorch_support": "strengthens"
                if identity in gpdc
                else "disagrees_or_not_detected",
                "parcorr_visibility": "retained",
                "vetoed": False,
            }
        )
    gpdctorch_only = [
        {
            "source": link.get("source"),
            "target": link.get("target"),
            "lag": link.get("lag"),
            "sign": link.get("sign"),
            "classification": "gpdctorch_only_method_sensitive",
            "evidence": link,
        }
        for link in gpdctorch_links
        if (
            link.get("source"),
            link.get("target"),
            link.get("lag"),
            link.get("sign"),
        )
        not in parcorr_in_scope
    ]
    return result | {
        "state": "complete",
        "comparisons": comparisons,
        "parcorr_primary_links_outside_gpdc_scope": outside_scope,
        "parcorr_primary_links_outside_gpdc_scope_count": len(outside_scope),
        "gpdctorch_only_method_sensitive": gpdctorch_only,
    }


@dataclass(frozen=True)
class RealInput:
    dates: np.ndarray
    values: np.ndarray
    metadata: dict[str, Any]
    raw_f107: np.ndarray | None = None


def calendar_month_days(dates: np.ndarray) -> list[tuple[int, int]]:
    """Return calendar month/day keys, retaining February 29 as its own day."""
    python_dates = dates.astype("datetime64[D]").astype(object)
    return [(date.month, date.day) for date in python_dates]


def rolling_nanmean(values: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling mean which ignores, but never fills, missing values."""
    values = np.asarray(values, dtype=float)
    window = max(1, min(window, len(values)))
    finite = np.isfinite(values)
    numerator = np.convolve(np.where(finite, values, 0.0), np.ones(window), mode="same")
    denominator = np.convolve(finite.astype(float), np.ones(window), mode="same")
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(values, np.nan),
        where=denominator > 0,
    )


def rolling_nanvar(values: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling population variance that ignores missing window samples."""
    values = np.asarray(values, dtype=float)
    mean = rolling_nanmean(values, window)
    squared_mean = rolling_nanmean(np.square(values), window)
    return np.maximum(squared_mean - np.square(mean), 0.0)


def longest_contiguous_finite_span(values: np.ndarray) -> tuple[int, int] | None:
    """Return half-open bounds for the longest contiguous finite span."""
    finite = np.isfinite(np.asarray(values, dtype=float))
    best: tuple[int, int] | None = None
    start = 0
    while start < len(finite):
        if not finite[start]:
            start += 1
            continue
        end = start + 1
        while end < len(finite) and finite[end]:
            end += 1
        if best is None or end - start > best[1] - best[0]:
            best = (start, end)
        start = end
    return best


def holm_adjusted_pvalues(p_values: dict[str, float]) -> dict[str, float]:
    """Return deterministic Holm familywise adjusted p-values by node name."""
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    previous = 0.0
    family_size = len(ordered)
    for rank, (node, p_value) in enumerate(ordered):
        previous = max(previous, min(1.0, (family_size - rank) * p_value))
        adjusted[node] = previous
    return adjusted


def _adf(values: np.ndarray) -> dict[str, Any]:
    from statsmodels.tsa.stattools import adfuller

    statistic, p_value, used_lag, observations, critical_values, icbest = cast(
        tuple[float, float, int, int, dict[str, float], float],
        adfuller(values, regression="c", autolag="AIC"),
    )
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "used_lag": int(used_lag),
        "observations": int(observations),
        "critical_values": {
            key: float(value) for key, value in critical_values.items()
        },
        "information_criterion": float(icbest),
    }


def _kpss(values: np.ndarray) -> dict[str, Any]:
    from statsmodels.tsa.stattools import kpss

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        statistic, p_value, lags, critical_values = kpss(
            values, regression="c", nlags="auto"
        )
    result: dict[str, Any] = {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "used_lag": int(lags),
        "critical_values": {
            key: float(value) for key, value in critical_values.items()
        },
    }
    if captured:
        result["warnings"] = [str(warning.message) for warning in captured]
    return result


def stationarity_qualification(
    values: np.ndarray,
    dates: np.ndarray,
    node_names: list[str],
    *,
    adf: Any = _adf,
    kpss: Any = _kpss,
) -> dict[str, Any]:
    """Qualify a PCMCI preprocessing profile using ADF and KPSS with Holm control."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape != (len(dates), len(node_names)):
        raise ValueError("stationarity values, dates, and node_names must align")
    nodes: dict[str, dict[str, Any]] = {}
    raw_p_values: dict[str, dict[str, float]] = {"adf": {}, "kpss": {}}
    for index, node in enumerate(node_names):
        span = longest_contiguous_finite_span(values[:, index])
        if span is None:
            nodes[node] = {
                "sample_count": 0,
                "span": None,
                "outcome": "not_qualified_missing_span",
            }
            continue
        start, end = span
        node_result: dict[str, Any] = {
            "sample_count": end - start,
            "span": {
                "start": str(dates[start]),
                "end": str(dates[end - 1]),
                "start_index": start,
                "end_index": end - 1,
            },
        }
        if end - start < MIN_STATIONARITY_SAMPLES:
            nodes[node] = node_result | {"outcome": "not_qualified_too_short_span"}
            continue
        span_values = values[start:end, index]
        for family, test in (("adf", adf), ("kpss", kpss)):
            try:
                test_result = dict(test(span_values))
                p_value = float(test_result["p_value"])
                if not np.isfinite(p_value) or not 0.0 <= p_value <= 1.0:
                    raise ValueError("returned p_value must be finite and in [0, 1]")
            except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as error:
                node_result[family] = {
                    "outcome": "test_error",
                    "error": f"{type(error).__name__}: {error}",
                }
            else:
                node_result[family] = test_result
                raw_p_values[family][node] = p_value
        nodes[node] = node_result

    families: dict[str, dict[str, Any]] = {}
    for family, null, reject_outcome, retain_outcome in (
        ("adf", "unit_root", "reject_unit_root", "does_not_reject_unit_root"),
        (
            "kpss",
            "level_stationarity",
            "reject_level_stationarity",
            "do_not_reject_level_stationarity",
        ),
    ):
        unavailable = sorted(set(node_names) - set(raw_p_values[family]))
        adjusted = holm_adjusted_pvalues(
            raw_p_values[family] | {node: 1.0 for node in unavailable}
        )
        for node, p_value in raw_p_values[family].items():
            reject = adjusted[node] <= STATIONARITY_ALPHA
            nodes[node][family].update(
                raw_p_value=p_value,
                adjusted_p_value=adjusted[node],
                null_hypothesis=null,
                alternative_hypothesis=(
                    "stationary" if family == "adf" else "not_level_stationary"
                ),
                reject_null=reject,
                outcome=reject_outcome if reject else retain_outcome,
            )
        families[family] = {
            "family_size": len(node_names),
            "tested_nodes": sorted(raw_p_values[family]),
            "unavailable_nodes": unavailable,
            "unavailable_node_policy": "unavailable nodes occupy full-family membership with p=1; they remain unqualified",
            "adjusted_p_values": {
                node: adjusted[node] for node in sorted(raw_p_values[family])
            },
        }
    for node in node_names:
        result = nodes[node]
        if "reject_null" in result.get("adf", {}) and "reject_null" in result.get(
            "kpss", {}
        ):
            result["outcome"] = (
                "qualified"
                if result["adf"]["reject_null"] and not result["kpss"]["reject_null"]
                else "not_qualified_stationarity_test"
            )
        elif "adf" in result or "kpss" in result:
            result["outcome"] = "not_qualified_test_error"
    qualified = all(nodes[node]["outcome"] == "qualified" for node in node_names)
    return {
        "method": "ADF and KPSS stationarity qualification",
        "familywise_alpha": STATIONARITY_ALPHA,
        "multiple_testing": "Holm separately across the full graph-node family for each test family",
        "settings": {
            "adf": {
                "regression": "c",
                "autolag": "AIC",
                "null_hypothesis": "unit_root",
            },
            "kpss": {
                "regression": "c",
                "nlags": "auto",
                "null_hypothesis": "level_stationarity",
            },
            "minimum_samples": MIN_STATIONARITY_SAMPLES,
            "minimum_samples_justification": "2 * DEFAULT_TAU_MAX + 1, compatible with the production 0-180-day physical lag window and PCMCI+ requirement for more than 2*tau_max rows",
        },
        "test_families": families,
        "nodes": nodes,
        "causal_interpretation_eligible": qualified,
        "sensitivity_evidence_only": not qualified,
        "ineligibility_reason": (
            None
            if qualified
            else "one or more graph nodes did not meet PCMCI stationarity qualification"
        ),
    }


def rolling_diagnostics(values: np.ndarray) -> dict[str, np.ndarray]:
    """Return companion 365-day practical-drift diagnostics without qualification use."""
    values = np.asarray(values, dtype=float)
    return {
        "rolling_mean": np.column_stack(
            [rolling_nanmean(values[:, index], 365) for index in range(values.shape[1])]
        ),
        "rolling_variance": np.column_stack(
            [rolling_nanvar(values[:, index], 365) for index in range(values.shape[1])]
        ),
    }


def finite_standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return values.copy()
    mean = np.mean(values[finite])
    std = np.std(values[finite])
    result = values - mean if std == 0 else (values - mean) / std
    result[~finite] = np.nan
    return result


def seasonal_anomaly(values: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """Remove per-calendar-month/day means, preserving missing values.

    Calendar month/day matching aligns dates across leap and non-leap years;
    February 29 is intentionally a distinct climatology bin.
    """
    values = np.asarray(values, dtype=float)
    climatology: dict[tuple[int, int], float] = {}
    keys = calendar_month_days(dates)
    for key in set(keys):
        selected = values[[index for index, value in enumerate(keys) if value == key]]
        finite = selected[np.isfinite(selected)]
        if finite.size:
            climatology[key] = float(finite.mean())
    finite_values = values[np.isfinite(values)]
    fallback = float(finite_values.mean()) if finite_values.size else np.nan
    return values - np.array([climatology.get(key, fallback) for key in keys])


def preprocess(
    values: np.ndarray,
    dates: np.ndarray,
    profile: PreprocessingProfile = DETRENDED_ANOMALY,
) -> np.ndarray:
    """Apply one registered PCMCI preprocessing profile."""
    if profile not in {DETRENDED_ANOMALY, SEASONAL_ANOMALY}:
        raise ValueError(f"unregistered PCMCI preprocessing profile: {profile}")
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must be a two-dimensional node matrix")
    result = np.empty_like(values, dtype=float)
    for column in range(values.shape[1]):
        anomaly = seasonal_anomaly(values[:, column], dates)
        if profile == DETRENDED_ANOMALY:
            anomaly = anomaly - rolling_nanmean(anomaly, ROLLING_WINDOW)
        result[:, column] = finite_standardize(anomaly)
    return result


def validate_daily_dates(dates: np.ndarray) -> None:
    days = np.asarray(dates).astype("datetime64[D]")
    if len(days) == 0:
        raise ValueError("input CSV contains no dated rows")
    if np.isnat(days).any():
        raise ValueError("input CSV contains an invalid date")
    differences = np.diff(days.astype("int64"))
    if np.any(differences == 0):
        raise ValueError("input CSV dates must be unique")
    if np.any(differences != 1):
        raise ValueError("input CSV dates must be consecutive daily dates")


def _node_counts(values: np.ndarray, imputed: np.ndarray) -> dict[str, dict[str, int]]:
    return {
        column: {
            "observed": int((np.isfinite(values[:, index]) & ~imputed[:, index]).sum()),
            "missing": int((~np.isfinite(values[:, index])).sum()),
            "imputed": int(imputed[:, index].sum()),
        }
        for index, column in enumerate(NODE_COLUMNS)
    }


def _raw_f107_counts(values: np.ndarray) -> dict[str, int]:
    return {
        "observed": int(np.isfinite(values).sum()),
        "missing": int((~np.isfinite(values)).sum()),
        "imputed": 0,
    }


def load_input(path: Path, row_limit: int | None = None) -> RealInput:
    required = [DATE_COLUMN, *NODE_COLUMNS]
    raw = pl.read_csv(path, null_values=["", "NaN", "nan"])
    missing_columns = set(required) - set(raw.columns)
    if missing_columns:
        raise ValueError(f"input CSV is missing columns: {sorted(missing_columns)}")
    selected_columns = [
        pl.col(DATE_COLUMN)
        .cast(pl.String)
        .str.to_date("%Y-%m-%d", strict=False)
        .alias(DATE_COLUMN),
        *NODE_COLUMNS,
        *[column for column in IMPUTATION_MASK_COLUMNS if column in raw.columns],
    ]
    if F107_RAW_COLUMN in raw.columns:
        selected_columns.insert(1, F107_RAW_COLUMN)
    frame = raw.select(selected_columns)
    dates = frame[DATE_COLUMN].to_numpy().astype("datetime64[D]")
    validate_daily_dates(dates)
    if row_limit is not None:
        if row_limit <= 0:
            raise ValueError("--row-limit must be positive")
        frame = frame.head(row_limit)
        dates = dates[: len(frame)]
    values = np.column_stack(
        [frame[column].to_numpy().astype(float) for column in NODE_COLUMNS]
    )
    raw_f107 = (
        frame[F107_RAW_COLUMN].to_numpy().astype(float)
        if F107_RAW_COLUMN in frame.columns
        else None
    )
    imputed = np.column_stack(
        [
            frame[column].fill_null(False).cast(pl.Boolean).to_numpy()
            if column in frame.columns
            else np.zeros(len(frame), dtype=bool)
            for column in [f"{name}_imputed" for name in NODE_COLUMNS]
        ]
    )
    metadata = {
        "path": str(path),
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "date_range": {"start": str(dates[0]), "end": str(dates[-1])},
        "row_count": len(frame),
        "node_order": NODE_COLUMNS,
        "node_counts": _node_counts(values, imputed),
        "raw_f107": (
            {"source_column": F107_RAW_COLUMN, "counts": _raw_f107_counts(raw_f107)}
            if raw_f107 is not None
            else None
        ),
        "row_limit": row_limit,
        "row_limit_calibration_only": row_limit is not None,
        "co2_source": (
            "NOAA GML Mauna Loa daily means; source includes Maunakea "
            "substitute observations from December 2022 through 2023-07-04"
        ),
    }
    return RealInput(dates, values, metadata, raw_f107)


def _common_f107_support(input_data: RealInput) -> tuple[np.ndarray, dict[str, Any]]:
    """Freeze common F10.7 availability without compressing the daily axis."""
    raw_f107 = input_data.raw_f107
    if raw_f107 is None:
        raw_f107 = input_data.values[:, 0]
    support = np.isfinite(input_data.values[:, 0]) & np.isfinite(raw_f107)
    dates = input_data.dates.astype("datetime64[D]").astype("int64")
    return support, {
        "daily_date_sequence_sha256": hashlib.sha256(dates.tobytes()).hexdigest(),
        "row_count": len(input_data.dates),
        "common_f107_support": {
            "sha256": hashlib.sha256(support.tobytes()).hexdigest(),
            "row_count": int(support.sum()),
        },
    }


def prepare_sensitivity_input(
    input_data: RealInput, case: SensitivityCase
) -> tuple[RealInput, list[str], dict[str, Any]]:
    """Select a case's F10.7 series on the common accepted-quality rows."""
    if case.timing_variant == RAW_OBSERVED_DAILY and input_data.raw_f107 is None:
        raise ValueError("input CSV is missing raw observed daily F10.7")
    support, identity = _common_f107_support(input_data)
    raw_f107 = input_data.raw_f107
    if raw_f107 is None:
        raw_f107 = input_data.values[:, 0]
    values = input_data.values.copy()
    if case.timing_variant == RAW_OBSERVED_DAILY:
        values[:, 0] = raw_f107
        node_names = [F107_RAW_COLUMN, *NODE_COLUMNS[1:]]
        source_values = raw_f107
    else:
        node_names = NODE_COLUMNS.copy()
        source_values = input_data.values[:, 0]
    values[~support, 0] = np.nan
    metadata = input_data.metadata | {"accepted_quality_rows": identity}
    metadata["node_order"] = node_names
    node_counts = input_data.metadata.get(
        "node_counts",
        _node_counts(input_data.values, np.zeros(input_data.values.shape, dtype=bool)),
    )
    metadata["node_counts"] = {
        node_names[0]: _raw_f107_counts(values[:, 0]),
        **{name: node_counts[name] for name in NODE_COLUMNS[1:]},
    }
    metadata["f10_7"] = {
        "source_column": node_names[0],
        "source_counts": _raw_f107_counts(source_values),
        "common_support": identity["common_f107_support"],
    }
    return RealInput(input_data.dates, values, metadata), node_names, identity


def build_link_assumptions(
    tau_max: int, node_names: list[str] | None = None
) -> dict[int, dict[tuple[int, int], str]]:
    """Match the established exogenous-F10.7 assumptions in the analysis script."""
    node_names = node_names or NODE_COLUMNS
    f107_index = 0
    assumptions: dict[int, dict[tuple[int, int], str]] = {
        target: {} for target in range(len(node_names))
    }
    for target in range(len(node_names)):
        for cause in range(len(node_names)):
            for lag in range(1, tau_max + 1):
                if target == f107_index and cause != f107_index:
                    continue
                assumptions[target][(cause, -lag)] = "-?>"
    for cause in range(len(node_names)):
        for target in range(cause + 1, len(node_names)):
            if cause == f107_index:
                assumptions[target][(f107_index, 0)] = "-?>"
            elif target == f107_index:
                assumptions[cause][(f107_index, 0)] = "-?>"
            else:
                assumptions[target][(cause, 0)] = "o?o"
    return assumptions


def _link_assumption_metadata(tau_max: int, node_names: list[str]) -> dict[str, Any]:
    return {
        "source": "scripts/tigramite_causal_global_mean.py:build_link_assumptions",
        "f10_7_node": node_names[0],
        "other_nodes_cannot_cause_f10_7_at_lagged_or_contemporaneous_lags": True,
        "f10_7_self_lags_allowed": True,
        "lagged_link_mark": "-?>",
        "non_f10_7_contemporaneous_link_mark": "o?o",
        "tau_max": tau_max,
        "link_count": sum(
            len(links) for links in build_link_assumptions(tau_max, node_names).values()
        ),
    }


def _method_seed(method: str) -> int:
    return int(
        np.random.SeedSequence([SEED, METHODS.index(method)]).generate_state(1)[0]
    )


def real_method_settings(method: str, cmiknn_workers: int) -> dict[str, Any]:
    """Return only settings consumed by the real PCMCI+ execution."""
    settings = runtime.method_settings(method, cmiknn_workers)
    # The synthetic run_pcmci entry point consumes alpha_level; run_pcmciplus
    # does not expose that argument and instead uses pc_alpha.
    settings.pop("alpha_level")
    return settings


def _validate_gpdctorch_scope(
    method: str,
    tau_max: int,
    case: SensitivityCase,
    input_data: RealInput | None = None,
    gated_stage: str | None = None,
) -> None:
    if method != "gpdctorch":
        return
    expected = next(
        (stage for stage in gpdctorch_gates.stages() if stage.name == gated_stage), None
    )
    if expected is None and (
        case.timing_variant,
        case.preprocessing_profile,
        case.role,
    ) != (
        RAW_OBSERVED_DAILY,
        DETRENDED_ANOMALY,
        "primary",
    ):
        raise ValueError(
            "GPDCtorch may execute only the primary raw_observed_daily+detrended_anomaly case"
        )
    if expected is None and tau_max != 1:
        raise ValueError(
            "GPDCtorch currently supports only tau_max=1 outside the validated gate"
        )
    if expected is not None and (
        tau_max,
        case.timing_variant,
        case.preprocessing_profile,
    ) != (
        expected.tau_max,
        expected.timing_variant,
        expected.preprocessing_profile,
    ):
        raise ValueError(
            "GPDCtorch gated child does not match its validated gate stage"
        )
    if input_data is not None and (
        input_data.metadata.get("row_limit") is not None
        or input_data.metadata.get("row_limit_calibration_only") is True
    ):
        raise ValueError("GPDCtorch does not support row-limited or prefix inputs")


def run_pcmciplus(
    input_data: RealInput,
    method: str,
    tau_max: int,
    cmiknn_workers: int,
    artifact_path: Path | None = None,
    case: SensitivityCase | None = None,
) -> dict[str, Any]:
    """Run one real PCMCI+ method. Tigramite imports remain in the child process."""
    return _run_pcmciplus_impl(
        input_data, method, tau_max, cmiknn_workers, artifact_path, case, None
    )


def _run_pcmciplus_gated(
    input_data: RealInput,
    method: str,
    tau_max: int,
    cmiknn_workers: int,
    artifact_path: Path | None,
    case: SensitivityCase,
    context: _GatedChildContext,
) -> dict[str, Any]:
    return _run_pcmciplus_impl(
        input_data, method, tau_max, cmiknn_workers, artifact_path, case, context
    )


def _run_pcmciplus_impl(
    input_data: RealInput,
    method: str,
    tau_max: int,
    cmiknn_workers: int,
    artifact_path: Path | None,
    case: SensitivityCase | None,
    context: _GatedChildContext | None,
) -> dict[str, Any]:
    """Private shared implementation; only validated child context unlocks gates."""
    case = case or sensitivity_case(RAW_OBSERVED_DAILY, DETRENDED_ANOMALY)
    _validate_gpdctorch_scope(
        method, tau_max, case, input_data, context.stage.name if context else None
    )
    from tigramite import data_processing as pp
    from tigramite.independence_tests.cmiknn import CMIknn
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI

    input_data, node_names, _ = prepare_sensitivity_input(input_data, case)
    if len(input_data.dates) <= 2 * tau_max:
        raise ValueError(
            f"input has {len(input_data.dates)} rows; PCMCI+ requires more than "
            f"2*tau_max={2 * tau_max} rows"
        )
    runtime.validate_cmiknn_tau(method, tau_max)
    seed = _method_seed(method)
    random.seed(seed)
    np.random.seed(seed)
    settings = real_method_settings(method, cmiknn_workers)
    if method == "parcorr":
        test = ParCorr(significance="analytic")
    elif method == "cmiknn":
        test = CMIknn(
            significance="shuffle_test",
            sig_samples=20,
            sig_blocklength=4,
            knn=0.1,
            shuffle_neighbors=5,
            workers=cmiknn_workers,
        )
    else:
        from tigramite.independence_tests.gpdc_torch import GPDCtorch

        test = GPDCtorch(significance="analytic")
    transformed = preprocess(
        input_data.values, input_data.dates, case.preprocessing_profile
    )
    dataframe = pp.DataFrame(
        np.where(np.isfinite(transformed), transformed, MISSING_FLAG),
        datatime=np.arange(len(transformed)),
        var_names=node_names,
        missing_flag=MISSING_FLAG,
        remove_missing_upto_maxlag=False,
    )
    results = PCMCI(dataframe=dataframe, cond_ind_test=test, verbosity=0).run_pcmciplus(
        link_assumptions=build_link_assumptions(tau_max, node_names),
        tau_min=0,
        tau_max=tau_max,
        pc_alpha=0.05,
        contemp_collider_rule="majority",
        conflict_resolution=True,
        fdr_method="none",
    )
    matrices = {name: results[name] for name in ("val_matrix", "p_matrix", "graph")}
    lifecycle: dict[str, Any] | None = None
    if method == "gpdctorch":
        # The isolated child exit is the prediction-state release boundary.
        try:
            import torch

            torch.cuda.empty_cache()
            lifecycle = {
                "isolation": "one PCMCI fit per child process",
                "allocator_cache_cleanup": "torch.cuda.empty_cache after result extraction",
                "prediction_state_release_boundary": "isolated child process exit",
                "tigramite_pin": "7c7b177cfbff77e11d805ab04fc2647301da1951",
            }
        except Exception as error:
            lifecycle = {"release_error": f"{type(error).__name__}: {error}"}
    return (
        {
            "settings": settings,
            "seed": seed,
            "matrix_shapes": {
                name: list(np.asarray(value).shape) for name, value in matrices.items()
            },
            "result_digest": runtime.compact_result_digest(matrices),
            "process_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
        }
        | ({"gpdctorch_lifecycle": lifecycle} if lifecycle is not None else {})
        | (
            {
                "artifact": runtime.write_npz_artifact(
                    artifact_path, matrices, node_names=node_names
                )
            }
            if artifact_path
            else {}
        )
    )


def _child_main(args: argparse.Namespace) -> int:
    try:
        started = time.monotonic()
        case = sensitivity_case(args.timing_variant, args.preprocessing_profile)
        context = _validate_gated_child(args) if args.method == "gpdctorch" else None
        runner = _run_pcmciplus_gated if context else run_pcmciplus
        payload = (
            runner(
                load_input(args.input, args.row_limit),
                args.method,
                args.tau_max,
                args.cmiknn_workers,
                args.artifact,
                case,
                context,
            )
            if context
            else runner(
                load_input(args.input, args.row_limit),
                args.method,
                args.tau_max,
                args.cmiknn_workers,
                args.artifact,
                case,
            )
        )
        payload.update(status="succeeded", wall_seconds=time.monotonic() - started)
    except Exception as error:
        payload = {
            "status": "failed",
            "failure_reason": f"{type(error).__name__}: {error}",
        }
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0 if payload["status"] == "succeeded" else 1


def _run_isolated_case(
    args: argparse.Namespace,
    method: str,
    case: SensitivityCase,
    threads: int,
    artifact: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "thermodense.benchmarks.pcmci_real",
        "case",
        "--input",
        str(args.input),
        "--method",
        method,
        "--tau-max",
        str(args.tau_max),
        "--cmiknn-workers",
        str(args.cmiknn_workers),
        "--artifact",
        str(artifact),
        "--timing-variant",
        case.timing_variant,
        "--preprocessing-profile",
        case.preprocessing_profile,
    ]
    if args.row_limit is not None:
        command.extend(["--row-limit", str(args.row_limit)])
    return runtime.run_isolated_process(command, args.timeout, threads)


def _gate_identity(
    input_data: RealInput, threads: int = 1, output: Path | None = None
) -> dict[str, Any]:
    """Identity shared by every gated full-row fit and imported evidence."""
    _, _, accepted_rows = prepare_sensitivity_input(
        input_data, sensitivity_case(RAW_OBSERVED_DAILY, DETRENDED_ANOMALY)
    )
    return {
        "input_sha256": input_data.metadata.get("input_sha256"),
        "row_count": input_data.metadata.get("row_count"),
        "row_limit": input_data.metadata.get("row_limit"),
        "accepted_quality_rows": accepted_rows,
        "method": "gpdctorch",
        "settings": real_method_settings("gpdctorch", DEFAULT_CMIKNN_WORKERS)
        | {"threads": threads},
        "runner_version": RUNNER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "tigramite_pin": "7c7b177cfbff77e11d805ab04fc2647301da1951",
        "output_path": str(output.resolve()) if output is not None else None,
    }


def _validate_gated_stage_result(
    record: dict[str, Any],
    identity: dict[str, Any],
    hardware: dict[str, Any],
    input_data: RealInput,
) -> bool:
    """Validate one exact stage result, including legacy capability attestation."""
    result = record.get("result", {})
    stage = next(
        (item for item in gpdctorch_gates.stages() if item.name == record.get("name")),
        None,
    )
    if (
        record.get("status") != "succeeded"
        or record.get("identity") != identity
        or not isinstance(stage, gpdctorch_gates.Stage)
        or not isinstance(result, dict)
    ):
        return False
    evidence = record.get("capability_evidence")
    legacy = (
        isinstance(evidence, dict)
        and evidence.get("mode") == "legacy_environment_attestation"
    )
    if legacy:
        labels = ("host_label", "environment_label", "environment_fingerprint")
        current_settings = identity["settings"]
        legacy_settings = result.get("settings")
        expected_legacy_settings = {
            key: value for key, value in current_settings.items() if key != "threads"
        }
        if (
            stage.name != "capability"
            or evidence.get("original_labels")
            != {key: result.get(key) for key in labels}
            or evidence.get("original_git_commit") != result.get("git_commit")
            or evidence.get("tigramite_pin") != identity["tigramite_pin"]
            or evidence.get("pin_proven") is not True
            or not gpdctorch_gates._legacy_tigramite_pin(
                evidence.get("original_git_commit"), identity["tigramite_pin"]
            )
            or current_settings.get("threads") != 1
            or legacy_settings != expected_legacy_settings
            or evidence.get("legacy_settings") != legacy_settings
            or evidence.get("current_requested_settings") != current_settings
            or evidence.get("threads_provenance")
            != "legacy_absent_constrained_to_default_1"
            or evidence.get("current_hardware") != hardware
            or evidence.get("current_environment")
            != gpdctorch_gates.environment_identity()
        ):
            return False
    elif evidence is not None and evidence.get("mode") != "hardware_equality":
        return False
    case = sensitivity_case(stage.timing_variant, stage.preprocessing_profile)
    _, expected_nodes, _ = prepare_sensitivity_input(input_data, case)
    expected_case = {
        "timing_variant": stage.timing_variant,
        "preprocessing_profile": stage.preprocessing_profile,
        "role": stage.role,
        "accepted_quality_rows": identity["accepted_quality_rows"],
        "node_order": expected_nodes,
    }
    result_case = result.get("sensitivity_case")
    result_input = result.get("input")
    if (
        result.get("status") != "succeeded"
        or result.get("schema_version") != identity["schema_version"]
        or result.get("runner_version") != identity["runner_version"]
        or result.get("synthetic") is not False
        or result.get("method") != identity["method"]
        or (not legacy and result.get("gate_stage") != stage.name)
        or result.get("tau_max") != stage.tau_max
        or (not legacy and result.get("timeout_seconds") != stage.timeout_seconds)
        or (not legacy and result.get("settings") != identity["settings"])
        or (
            not legacy
            and result.get("child_settings")
            != {
                key: value
                for key, value in identity["settings"].items()
                if key != "threads"
            }
        )
        or (not legacy and result.get("hardware") != hardware)
        or result.get("package_versions")
        != gpdctorch_gates.environment_identity()["package_versions"]
        or (
            not legacy
            and result.get("git_commit")
            != gpdctorch_gates.environment_identity()["git_commit"]
        )
        or not isinstance(result_case, dict)
        or any(result_case.get(key) != value for key, value in expected_case.items())
        or not isinstance(result_input, dict)
        or any(
            result_input.get(key) != identity[key]
            for key in ("input_sha256", "row_count", "row_limit")
        )
        or result_input.get("row_limit") is not None
        or not isinstance(result.get("wall_seconds"), (int, float))
        or isinstance(result.get("wall_seconds"), bool)
        or not 0 <= result["wall_seconds"] <= stage.timeout_seconds
        or "failure_reason" not in result
        or result["failure_reason"] is not None
        or (
            not legacy
            and result.get("gpdctorch_lifecycle", {}).get("tigramite_pin")
            != identity["tigramite_pin"]
        )
    ):
        return False
    artifact = result.get("artifact")
    if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
        return False
    try:
        path = Path(artifact["path"])
        if not path.is_file() or hashlib.sha256(
            path.read_bytes()
        ).hexdigest() != artifact.get("sha256"):
            return False
        with np.load(path, allow_pickle=False) as saved:
            expected = (5, 5, stage.tau_max + 1)
            if (
                set(saved.files) != {"graph", "p_matrix", "val_matrix", "node_names"}
                or any(
                    saved[name].shape != expected
                    for name in ("graph", "p_matrix", "val_matrix")
                )
                or saved["node_names"].tolist() != result_case.get("node_order")
            ):
                return False
            matrices = {
                name: saved[name] for name in ("graph", "p_matrix", "val_matrix")
            }
        return (
            artifact.get("format") == "npz-compressed"
            and artifact.get("keys")
            == ["graph", "node_names", "p_matrix", "val_matrix"]
            and runtime.compact_result_digest(matrices) == result.get("result_digest")
        )
    except _GATED_ARTIFACT_VALIDATION_ERRORS:
        return False


def _validate_gated_child(args: argparse.Namespace) -> _GatedChildContext:
    """Validate state-file context in the isolated process; no CLI unlock exists."""
    authorization = os.environ.pop("THERMODENSE_GPDC_GATE_AUTH", None)
    if args.gate_state is None or not authorization:
        raise ValueError("GPDCtorch tau10 requires validated gate state")
    state = json.loads(args.gate_state.read_text())
    stage = next(
        (
            item
            for item in gpdctorch_gates.stages()
            if (args.tau_max, args.timing_variant, args.preprocessing_profile)
            == (item.tau_max, item.timing_variant, item.preprocessing_profile)
        ),
        None,
    )
    stage_name = stage.name if stage else None
    record = state.get("stages", {}).get(stage_name) if stage else None
    input_data = load_input(args.input)
    if (
        stage is None
        or not isinstance(record, dict)
        or record.get("status") != "running"
        or record.get("attempts", [{}])[-1].get("authorization_sha256")
        != hashlib.sha256(authorization.encode()).hexdigest()
        or record.get("identity")
        != _gate_identity(input_data, args.gate_threads, args.gate_output)
        or (args.tau_max, args.timing_variant, args.preprocessing_profile)
        != (stage.tau_max, stage.timing_variant, stage.preprocessing_profile)
    ):
        raise ValueError("GPDCtorch gated child lacks matching validated state context")
    index = gpdctorch_gates.STAGES.index(stage_name)
    hardware = gpdctorch_gates.gpu_hardware()
    identity = _gate_identity(input_data, args.gate_threads, args.gate_output)
    if any(
        not _validate_gated_stage_result(
            state["stages"].get(name, {}), identity, hardware, input_data
        )
        for name in gpdctorch_gates.STAGES[:index]
    ):
        raise ValueError("GPDCtorch gated child predecessor is not validated")
    return _GatedChildContext(stage)


def _artifact_matches(reference: Any) -> bool:
    """Return whether a recorded artifact still names its exact bytes."""
    if not isinstance(reference, dict):
        return False
    path, expected_hash = reference.get("path"), reference.get("sha256")
    if not isinstance(path, str) or not _is_sha256(expected_hash):
        return False
    try:
        return (
            Path(path).is_file()
            and hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected_hash
        )
    except OSError:
        return False


def _validate_parcorr_agreement(
    path: Path, accepted_quality_rows: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, str]]:
    """Read a complete canonical ParCorr agreement before derived publication."""
    try:
        lines = path.read_text().splitlines()
        if len(lines) != 1:
            raise ValueError("ParCorr agreement must contain exactly one JSONL object")
        agreement = json.loads(lines[0])
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"malformed ParCorr agreement: {error}") from error
    if not isinstance(agreement, dict) or (
        agreement.get("state"),
        agreement.get("method"),
        agreement.get("schema_version"),
        agreement.get("physical_lag_window_days"),
        agreement.get("accepted_quality_rows"),
    ) != (
        "complete",
        "parcorr",
        "1",
        {"min": 0, "max": DEFAULT_TAU_MAX},
        accepted_quality_rows,
    ):
        raise ValueError("ParCorr agreement is incomplete or has mismatched identity")
    expected_algorithm = {
        "name": "PCMCI+",
        "entry_point": "PCMCI.run_pcmciplus",
        "tau_min": 0,
        "pc_alpha": 0.05,
        "contemp_collider_rule": "majority",
        "conflict_resolution": True,
        "fdr_method": "none",
    }
    if (
        not isinstance(agreement.get("settings"), dict)
        or not isinstance(agreement.get("stationarity_eligibility"), dict)
        or agreement.get("algorithm") != expected_algorithm
        or set(agreement["settings"]) != {"pc_alpha", "significance", "threads"}
        or agreement["settings"].get("pc_alpha") != 0.05
        or agreement["settings"].get("significance") != "analytic"
        or not isinstance(agreement["settings"].get("threads"), int)
        or isinstance(agreement["settings"].get("threads"), bool)
        or agreement["settings"]["threads"] <= 0
    ):
        raise ValueError(
            "ParCorr agreement lacks settings, algorithm, or stationarity provenance"
        )
    artifacts = agreement.get("case_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(
        REGISTERED_SENSITIVITY_CASES
    ):
        raise ValueError("ParCorr agreement lacks complete case artifact references")
    expected_cases = {_case_key(case) for case in REGISTERED_SENSITIVITY_CASES}
    if {
        item.get("case") for item in artifacts if isinstance(item, dict)
    } != expected_cases or any(not _artifact_matches(item) for item in artifacts):
        raise ValueError("ParCorr case artifact is missing or its hash does not match")
    by_case = {item["case"]: item for item in artifacts}
    for case in REGISTERED_SENSITIVITY_CASES:
        reference = by_case[_case_key(case)]
        try:
            with np.load(Path(reference["path"]), allow_pickle=False) as saved:
                if set(saved.files) != {
                    "graph",
                    "p_matrix",
                    "val_matrix",
                    "node_names",
                }:
                    raise ValueError("ParCorr case artifact canonical keys are invalid")
                expected_nodes = _expected_node_order(case)
                if [
                    str(name) for name in saved["node_names"].tolist()
                ] != expected_nodes or any(
                    saved[name].shape != (5, 5, DEFAULT_TAU_MAX + 1)
                    for name in ("graph", "p_matrix", "val_matrix")
                ):
                    raise ValueError(
                        "ParCorr case artifact shape or node order is invalid"
                    )
        except (OSError, ValueError, KeyError) as error:
            if isinstance(error, ValueError) and str(error).startswith("ParCorr"):
                raise
            raise ValueError(f"malformed ParCorr case artifact: {error}") from error
    expected_stationarity = {
        _case_key(case): {
            "provenance_identity": {
                "timing_variant": case.timing_variant,
                "preprocessing_profile": case.preprocessing_profile,
                "node_order": _expected_node_order(case),
                "daily_date_sequence_sha256": accepted_quality_rows.get(
                    "daily_date_sequence_sha256"
                ),
                "common_f107_support_sha256": accepted_quality_rows.get(
                    "common_f107_support", {}
                ).get("sha256"),
            }
        }
        for case in REGISTERED_SENSITIVITY_CASES
    }
    stationarity = agreement["stationarity_eligibility"]
    if set(stationarity) != expected_cases or any(
        not isinstance(stationarity.get(key), dict)
        or stationarity[key].get("provenance_identity")
        != expected_stationarity[key]["provenance_identity"]
        for key in expected_cases
    ):
        raise ValueError("ParCorr agreement stationarity provenance is malformed")
    if not isinstance(agreement.get("primary_links"), list):
        raise ValueError("ParCorr agreement lacks primary links")
    known_nodes = {F107_RAW_COLUMN, *NODE_COLUMNS}
    identities = set()
    required_link_fields = {
        "source",
        "target",
        "lag",
        "sign",
        "classification",
        "failed_dimensions",
        "failed_stationarity_cells",
        "centered_matches",
        "centered_delay_equivalent",
        "evidence",
    }
    for link in agreement["primary_links"]:
        if not isinstance(link, dict):
            raise ValueError("ParCorr primary link must be an object")
        source, target, lag, sign = (
            link.get("source"),
            link.get("target"),
            link.get("lag"),
            link.get("sign"),
        )
        if not isinstance(source, str) or source not in known_nodes:
            raise ValueError("ParCorr primary link source is not a canonical node")
        if not isinstance(target, str) or target not in known_nodes:
            raise ValueError("ParCorr primary link target is not a canonical node")
        if (
            not isinstance(lag, int)
            or isinstance(lag, bool)
            or not 0 <= lag <= DEFAULT_TAU_MAX
        ):
            raise ValueError(
                "ParCorr primary link lag must be an integer from 0 to 180"
            )
        if sign not in (-1, 1) or isinstance(sign, bool):
            raise ValueError("ParCorr primary link sign must be -1 or 1")
        identity = (_physical_node(source), _physical_node(target), lag, sign)
        if identity in identities:
            raise ValueError("ParCorr agreement has duplicate primary link identities")
        identities.add(identity)
        if (
            not required_link_fields <= link.keys()
            or (
                link["classification"]
                not in {"factor_sensitive", "stationarity_limited", "main_text_robust"}
            )
            or (
                not isinstance(link["failed_dimensions"], list)
                or not all(
                    dimension in {"detrending", "timing"}
                    for dimension in link["failed_dimensions"]
                )
            )
            or (
                not isinstance(link["failed_stationarity_cells"], list)
                or not all(
                    isinstance(cell, str) for cell in link["failed_stationarity_cells"]
                )
            )
            or not isinstance(link["centered_matches"], list)
            or not isinstance(link["centered_delay_equivalent"], bool)
            or not isinstance(link["evidence"], dict)
            or set(link["evidence"])
            != {
                "primary",
                "detrending",
                "timing",
            }
        ):
            raise ValueError("ParCorr primary link has malformed canonical fields")
    return agreement, {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _derived_sources(state: dict[str, Any]) -> dict[str, Any]:
    return {
        name: {
            "identity": state["stages"][name].get("identity"),
            "artifact": state["stages"][name].get("result", {}).get("artifact"),
        }
        for name in ("primary", "raw_seasonal", "centered_detrended", "interaction")
    }


def _parcorr_source_identity(path: Path | None) -> dict[str, Any] | None:
    """Bind derived completion to the exact optional ParCorr request state."""
    if path is None:
        return None
    try:
        return {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    except OSError:
        return {"path": str(path), "sha256": None}


def _complete_gpdctorch_derived(
    state: dict[str, Any],
    state_path: Path,
    artifacts: Path,
    output: Path,
    parcorr_path: Path | None,
    identity: dict[str, Any],
) -> bool:
    """Durably finish derived artifacts; caller output is always written last."""
    derived = state.setdefault(
        "derived",
        {
            "status": "pending",
            "attempts": [],
            "source_stage_identities": {},
            "created_at": datetime.now(UTC).isoformat(),
        },
    )
    sources = _derived_sources(state)
    parcorr_request = _parcorr_source_identity(parcorr_path)
    required = ("gpdctorch_agreement", "output_agreement", "comparison")
    if (
        derived.get("status") == "complete"
        and derived.get("source_stage_identities") == sources
        and derived.get("parcorr_request") == parcorr_request
        and all(_artifact_matches(derived.get(name)) for name in required)
    ):
        return True
    now = datetime.now(UTC).isoformat()
    attempt = {
        "started_at": now,
        "status": "running",
        "source_stage_identities": sources,
        "parcorr_request": parcorr_request,
    }
    derived.update(
        status="running",
        failure_reason=None,
        source_stage_identities=sources,
        parcorr_request=parcorr_request,
        updated_at=now,
    )
    derived.setdefault("attempts", []).append(attempt)
    gpdctorch_gates.atomic_write(state_path, state)
    try:
        # Validate optional input before any derived file, especially --output, is touched.
        parcorr, parcorr_source = (
            _validate_parcorr_agreement(parcorr_path, identity["accepted_quality_rows"])
            if parcorr_path is not None
            else (None, None)
        )
        stage_rows = [state["stages"][name]["result"] for name in sources]
        agreement = synthesize_gpdctorch_matrix(stage_rows)
        agreement_artifact = runtime.write_jsonl_artifact(
            artifacts / "gpdctorch-sensitivity-agreement.jsonl", [agreement]
        )
        comparison = compare_gpdctorch_with_parcorr(agreement["primary_links"], parcorr)
        comparison["gpdctorch_agreement"] = agreement_artifact
        comparison["parcorr_agreement"] = parcorr_source
        comparison_artifact = runtime.write_jsonl_artifact(
            artifacts / "gpdctorch-parcorr-comparison.jsonl", [comparison]
        )
        # Publishing the externally selected result is the transaction's final step.
        output_artifact = runtime.write_jsonl_artifact(output, [agreement])
        attempt.update(finished_at=datetime.now(UTC).isoformat(), status="complete")
        derived.update(
            status="complete",
            gpdctorch_agreement=agreement_artifact,
            output_agreement=output_artifact,
            comparison=comparison_artifact,
            parcorr_source=parcorr_source,
            updated_at=datetime.now(UTC).isoformat(),
        )
        gpdctorch_gates.atomic_write(state_path, state)
        return True
    except (OSError, ValueError, MatrixSynthesisError, json.JSONDecodeError) as error:
        reason = f"{type(error).__name__}: {error}"
        attempt.update(
            finished_at=datetime.now(UTC).isoformat(),
            status="failed",
            failure_reason=reason,
        )
        derived.update(
            status="failed",
            failure_reason=reason,
            updated_at=datetime.now(UTC).isoformat(),
        )
        gpdctorch_gates.atomic_write(state_path, state)
        return False


def _recorded_output_matches(state: dict[str, Any], output: Path) -> bool:
    reference = state.get("derived", {}).get("output_agreement")
    return (
        isinstance(reference, dict)
        and reference.get("path") == str(output)
        and _artifact_matches(reference)
    )


def _gpdctorch_gate_lock_path(state_path: Path, artifacts: Path) -> Path:
    """Return the stable inter-process lock path for a resolved gate state."""
    lock_parent = (
        artifacts.parent if state_path.is_relative_to(artifacts) else state_path.parent
    )
    state_digest = hashlib.sha256(str(state_path).encode()).hexdigest()
    return lock_parent / f".gpdctorch-gates-{state_digest}.lock"


def run_gpdctorch_gated(args: argparse.Namespace) -> int:
    """Run the only public GPDCtorch production entry point, in gate order."""
    if args.row_limit is not None:
        raise ValueError("GPDCtorch production gates do not accept --row-limit")
    output = args.output.resolve()
    artifacts = output.parent / f"{output.stem}_artifacts"
    state_path = args.state or artifacts / "gpdctorch-gates.json"
    state_path = state_path.resolve()
    input_path = args.input.resolve()
    capability_path = (
        args.import_capability.resolve() if args.import_capability is not None else None
    )
    parcorr_path = (
        args.parcorr_agreement.resolve() if args.parcorr_agreement is not None else None
    )
    read_sources = {
        "input": input_path,
        "import capability": capability_path,
        "ParCorr agreement": parcorr_path,
    }
    for source_name, source_path in read_sources.items():
        if source_path is None:
            continue
        if (
            source_path in (output, state_path)
            or _existing_paths_alias(source_path, output)
            or _existing_paths_alias(source_path, state_path)
        ):
            raise ValueError(f"{source_name} must not overlap output or state")
        if source_path.is_relative_to(artifacts):
            raise ValueError(
                f"{source_name} must not reside inside the derived artifact directory"
            )
    if state_path == output or _existing_paths_alias(state_path, output):
        raise ValueError("state must not overlap output")
    lock_path = _gpdctorch_gate_lock_path(state_path, artifacts)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(lock_path, timeout=0):
            return _run_gpdctorch_gated_locked(
                args,
                output,
                artifacts,
                state_path,
                input_path,
                capability_path,
                parcorr_path,
            )
    except Timeout as error:
        raise gpdctorch_gates.GateError(
            f"GPDCtorch gate state is already in use: {state_path}"
        ) from error


def _run_gpdctorch_gated_locked(
    args: argparse.Namespace,
    output: Path,
    artifacts: Path,
    state_path: Path,
    input_path: Path,
    capability_path: Path | None,
    parcorr_path: Path | None,
) -> int:
    """Run a GPDCtorch gate lifecycle while its state lock is held."""
    overwrite = getattr(args, "overwrite", False)
    if overwrite:
        output.unlink(missing_ok=True)
        shutil.rmtree(artifacts, ignore_errors=True)
        state_path.unlink(missing_ok=True)
    elif not state_path.exists() and (output.exists() or artifacts.exists()):
        raise ValueError(
            "refusing existing GPDCtorch output or artifact directory; use --overwrite"
        )
    elif output.exists():
        try:
            recorded_state = json.loads(state_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"malformed gate state: {error}") from error
        if not _recorded_output_matches(recorded_state, output):
            raise ValueError(
                "refusing to replace an output not recorded by this gate state"
            )
    input_data = load_input(input_path)
    if input_data.raw_f107 is None:
        raise ValueError("input CSV is missing raw observed daily F10.7")
    threads = args.threads if args.threads is not None else 1
    identity = _gate_identity(input_data, threads, output)
    hardware = gpdctorch_gates.gpu_hardware()

    def valid(record: dict[str, Any]) -> bool:
        return _validate_gated_stage_result(record, identity, hardware, input_data)

    def execute(stage: gpdctorch_gates.Stage, authorization: str) -> dict[str, Any]:
        case = sensitivity_case(stage.timing_variant, stage.preprocessing_profile)
        artifact = artifacts / f"gpdctorch-{stage.name}.npz"
        diagnostics = prepare_case_diagnostics(input_data, case, artifacts)
        row_args = argparse.Namespace(
            host_label="unspecified",
            environment_label="unspecified",
            environment_fingerprint="unspecified",
            tau_max=stage.tau_max,
            timeout=float(stage.timeout_seconds),
            cmiknn_workers=DEFAULT_CMIKNN_WORKERS,
        )
        row = _base_row(row_args, "gpdctorch", case, threads, diagnostics)
        row["hardware"] = hardware
        row["gate_stage"] = stage.name
        row["sensitivity_case"]["role"] = stage.role
        command = [
            sys.executable,
            "-m",
            "thermodense.benchmarks.pcmci_real",
            "case",
            "--input",
            str(input_path),
            "--method",
            "gpdctorch",
            "--tau-max",
            str(stage.tau_max),
            "--cmiknn-workers",
            str(DEFAULT_CMIKNN_WORKERS),
            "--artifact",
            str(artifact),
            "--timing-variant",
            case.timing_variant,
            "--preprocessing-profile",
            case.preprocessing_profile,
            "--gate-state",
            str(state_path),
            "--gate-threads",
            str(threads),
            "--gate-output",
            str(output),
        ]
        result = runtime.run_isolated_process(
            command,
            stage.timeout_seconds,
            threads,
            {"THERMODENSE_GPDC_GATE_AUTH": authorization},
        )
        row = _merge_child_result(row, result)
        row["timeout_seconds"] = stage.timeout_seconds
        row["artifact_path"] = str(artifact)
        return row

    if args.import_capability is not None:
        gpdctorch_gates.import_capability(
            state_path,
            capability_path,
            identity,
            hardware,
            validate_success=valid,
        )
    state = gpdctorch_gates.run(
        state_path,
        identity,
        hardware,
        execute,
        retry_failed=args.retry_failed,
        validate_success=valid,
    )
    if state["state"] == "complete":
        if output.exists() and not _recorded_output_matches(state, output):
            raise ValueError(
                "refusing to replace an output not recorded by this gate state"
            )
        if not _complete_gpdctorch_derived(
            state, state_path, artifacts, output, parcorr_path, identity
        ):
            print(f"{state_path} derived-artifact-failed", file=sys.stderr)
            return 1
    print(f"{state_path} {state['state']}")
    return (
        0
        if state["state"] == "complete" and state["derived"]["status"] == "complete"
        else 1
    )


def prepare_case_diagnostics(
    input_data: RealInput,
    case: SensitivityCase,
    artifact_directory: Path,
) -> dict[str, Any]:
    """Prepare immutable, method-independent diagnostics once for one case."""
    prepared_input, node_names, accepted_rows = prepare_sensitivity_input(
        input_data, case
    )
    transformed = preprocess(
        prepared_input.values, prepared_input.dates, case.preprocessing_profile
    )
    qualification = stationarity_qualification(
        transformed, prepared_input.dates, node_names
    )
    qualification["provenance_identity"] = {
        "timing_variant": case.timing_variant,
        "preprocessing_profile": case.preprocessing_profile,
        "node_order": node_names,
        "daily_date_sequence_sha256": accepted_rows["daily_date_sequence_sha256"],
        "common_f107_support_sha256": accepted_rows["common_f107_support"]["sha256"],
    }
    rolling = rolling_diagnostics(transformed)
    rolling_artifact = runtime.write_npz_artifact(
        artifact_directory
        / f"{case.timing_variant}-{case.preprocessing_profile}-rolling.npz",
        {"dates": prepared_input.dates, **rolling},
        node_names=node_names,
    )
    rolling_artifact["diagnostic"] = (
        "365-day rolling mean and variance; does not alter qualification"
    )
    rolling_artifact["window_days"] = 365
    return {
        "input": prepared_input,
        "node_names": node_names,
        "accepted_rows": accepted_rows,
        "stationarity_qualification": qualification,
        "rolling_diagnostics": rolling_artifact,
    }


def _base_row(
    args: argparse.Namespace,
    method: str,
    case: SensitivityCase,
    threads: int,
    case_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    prepared_input = case_diagnostics["input"]
    node_names = case_diagnostics["node_names"]
    accepted_rows = case_diagnostics["accepted_rows"]
    qualification = case_diagnostics["stationarity_qualification"]
    rolling_artifact = case_diagnostics["rolling_diagnostics"]
    return {
        "schema_version": SCHEMA_VERSION,
        "runner_version": RUNNER_VERSION,
        "timestamp": datetime.now(UTC).isoformat(),
        "status": "pending",
        "synthetic": False,
        "host_label": args.host_label,
        "environment_label": args.environment_label,
        "environment_fingerprint": args.environment_fingerprint,
        "method": method,
        "tau_max": args.tau_max,
        "timeout_seconds": args.timeout,
        "deferred_methods": DEFERRED_METHODS,
        "input": prepared_input.metadata,
        "sensitivity_case": {
            "timing_variant": case.timing_variant,
            "preprocessing_profile": case.preprocessing_profile,
            "accepted_quality_rows": accepted_rows,
            "role": case.role,
            "node_order": node_names,
            "f10_7": prepared_input.metadata["f10_7"],
        },
        "preprocessing": {
            "profile": case.preprocessing_profile,
            "calendar_month_day_anomaly": True,
            "february_29_has_distinct_climatology": True,
            "seasonal_climatology": (
                "finite daily values grouped by calendar month/day across years; "
                "February 29 remains a distinct group"
            ),
            "centered_rolling_nanmean_window": (
                ROLLING_WINDOW
                if case.preprocessing_profile == DETRENDED_ANOMALY
                else None
            ),
            "finite_standardization": True,
            "missing_values_preserved": True,
        },
        "stationarity_qualification": qualification,
        "rolling_diagnostics": rolling_artifact,
        "causal_interpretation_eligible": qualification[
            "causal_interpretation_eligible"
        ],
        "sensitivity_evidence_only": qualification["sensitivity_evidence_only"],
        "algorithm": {
            "name": "PCMCI+",
            "entry_point": "PCMCI.run_pcmciplus",
            "tau_min": 0,
            "pc_alpha": 0.05,
            "contemp_collider_rule": "majority",
            "conflict_resolution": True,
            "fdr_method": "none",
        },
        "link_assumptions": _link_assumption_metadata(args.tau_max, node_names),
        "missing_data_policy": {
            "sentinel": MISSING_FLAG,
            "remove_missing_upto_maxlag": False,
            "drivers_interpolated": False,
            "rows_dropped": False,
        },
        "settings": real_method_settings(method, args.cmiknn_workers)
        | {"threads": threads},
        **(
            {
                "method_scope": "nonlinear sensitivity; not a substitute for ParCorr",
                "untested_parcorr_lags_11_180": True,
                "untested_lag_window_days": {"min": 11, "max": DEFAULT_TAU_MAX},
            }
            if method == "cmiknn"
            else {}
        ),
        "package_versions": runtime.package_versions(),
        "git_commit": runtime.git_commit(),
        "wall_seconds": None,
        "process_max_rss_bytes": None,
        "matrix_shapes": {},
        "result_digest": None,
        "artifact": None,
        "failure_reason": None,
    }


def _merge_child_result(
    parent: dict[str, Any], child: dict[str, Any]
) -> dict[str, Any]:
    """Merge an isolated child payload without letting it replace outer settings."""
    child_settings = child.get("settings")
    result = parent | {key: value for key, value in child.items() if key != "settings"}
    result["settings"] = parent["settings"]
    result["child_settings"] = child_settings
    expected_child_settings = {
        key: value for key, value in parent["settings"].items() if key != "threads"
    }
    if child.get("status") == "succeeded" and child_settings != expected_child_settings:
        result["status"] = "failed"
        result["failure_reason"] = (
            "child settings do not match parent settings without threads"
        )
    return result


def _write_incomplete_matrix_manifest(
    path: Path,
    error: MatrixSynthesisError,
    current_case: SensitivityCase | None,
    rows: list[dict[str, Any]],
) -> None:
    """Persist the controlled production-matrix failure state before raising."""
    completed = [
        {
            "case": f"{row.get('sensitivity_case', {}).get('timing_variant')}/"
            f"{row.get('sensitivity_case', {}).get('preprocessing_profile')}",
            "status": row.get("status"),
            "failure_reason": row.get("failure_reason"),
            "artifact": row.get("artifact"),
        }
        for row in rows
    ]
    if current_case is not None and _case_key(current_case) not in {
        item["case"] for item in completed
    }:
        completed.append(
            {
                "case": _case_key(current_case),
                "status": "failed",
                "failure_reason": str(error),
                "artifact": None,
            }
        )
    runtime.write_jsonl_artifact(
        path,
        [
            {
                "state": "incomplete",
                "error": str(error),
                "current_case": _case_key(current_case) if current_case else None,
                "completed_cells": completed,
            }
        ],
    )


class _TrackExplicitTauMax(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None) -> None:
        setattr(namespace, self.dest, values)
        setattr(namespace, "_tau_max_explicit", True)


def _normalize_run_args(args: argparse.Namespace) -> None:
    """Apply mode-specific defaults before validating or dispatching a run."""
    if args.cmiknn_sensitivity_matrix and not getattr(args, "_tau_max_explicit", False):
        args.tau_max = runtime.CMIKNN_MAX_TAU_STEPS


def run(args: argparse.Namespace) -> int:
    _normalize_run_args(args)
    # Resolve paths before any output-side effect or input load.
    args.input = args.input.resolve()
    args.output = args.output.resolve()
    args.parcorr_agreement = (
        args.parcorr_agreement.resolve() if args.parcorr_agreement is not None else None
    )
    artifact_directory = args.output.parent / f"{args.output.stem}_artifacts"
    if args.input == args.output or _existing_paths_alias(args.input, args.output):
        raise ValueError("input and output must be different paths")
    if args.input == artifact_directory or args.input.is_relative_to(
        artifact_directory
    ):
        raise ValueError("input must not reside inside the derived artifact directory")
    cmiknn_matrix = args.cmiknn_sensitivity_matrix
    agreement_method = "cmiknn" if cmiknn_matrix else "parcorr"
    agreement_path = (
        artifact_directory / f"{agreement_method}-sensitivity-agreement.jsonl"
    )
    production_matrix = args.production_sensitivity_matrix
    if production_matrix and cmiknn_matrix:
        raise ValueError("select only one production sensitivity matrix method")
    if production_matrix:
        if args.methods not in (None, ["parcorr"]):
            raise ValueError("production sensitivity matrix requires only ParCorr")
        if args.tau_max != DEFAULT_TAU_MAX:
            raise ValueError(
                "production sensitivity matrix requires physical lags 0-180 days"
            )
        if args.row_limit is not None:
            raise ValueError(
                "production sensitivity matrix does not accept --row-limit"
            )
    if cmiknn_matrix:
        if args.methods not in (None, ["cmiknn"]):
            raise ValueError("CMIknn sensitivity matrix requires only CMIknn")
        if args.tau_max != runtime.CMIKNN_MAX_TAU_STEPS:
            raise ValueError(
                "CMIknn sensitivity matrix requires physical lags 0-10 days"
            )
        if args.row_limit is not None:
            raise ValueError("CMIknn sensitivity matrix does not accept --row-limit")
        if args.parcorr_agreement is not None:
            parcorr_path = args.parcorr_agreement
            if (
                parcorr_path == args.output.resolve()
                or _existing_paths_alias(parcorr_path, args.output)
                or parcorr_path.is_relative_to(artifact_directory.resolve())
            ):
                raise ValueError(
                    "ParCorr agreement must not overlap CMIknn output or artifacts"
                )
    if args.methods and "gpdctorch" in args.methods:
        raise ValueError(
            "ordinary runner cannot execute GPDCtorch; use gpdctorch-gated"
        )
    if (args.output.exists() or artifact_directory.exists()) and not args.overwrite:
        raise ValueError(
            f"Refusing to overwrite existing result or artifacts: {args.output}; use --overwrite."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if production_matrix and args.overwrite:
        args.output.write_text("")
        shutil.rmtree(artifact_directory, ignore_errors=True)
    try:
        input_data = load_input(args.input, args.row_limit)
        if input_data.raw_f107 is None:
            raise ValueError("input CSV is missing raw observed daily F10.7")
    except Exception as error:
        if not (production_matrix or cmiknn_matrix):
            raise
        controlled = MatrixSynthesisError(
            f"incomplete matrix: {type(error).__name__}: {error}"
        )
        _write_incomplete_matrix_manifest(agreement_path, controlled, None, [])
        raise controlled from error
    validated_parcorr: tuple[dict[str, Any], dict[str, str]] | None = None
    if cmiknn_matrix and args.parcorr_agreement is not None:
        _, _, accepted_quality_rows = prepare_sensitivity_input(
            input_data,
            sensitivity_case(RAW_OBSERVED_DAILY, DETRENDED_ANOMALY),
        )
        validated_parcorr = _validate_parcorr_agreement(
            args.parcorr_agreement, accepted_quality_rows
        )
    if cmiknn_matrix and args.overwrite:
        args.output.write_text("")
        shutil.rmtree(artifact_directory, ignore_errors=True)
    cases = (
        expand_sensitivity_cases()
        if args.all_sensitivity_cases or production_matrix or cmiknn_matrix
        else (sensitivity_case(args.timing_variant, args.preprocessing_profile),)
    )
    methods = (
        ("parcorr",)
        if production_matrix
        else ("cmiknn",)
        if cmiknn_matrix
        else args.methods or DEFAULT_METHODS
    )
    if args.all_sensitivity_cases and "gpdctorch" in methods:
        raise ValueError("GPDCtorch sensitivity-matrix execution is not available")
    for method in methods:
        for case in cases:
            if method == "gpdctorch":
                _validate_gpdctorch_scope(method, args.tau_max, case, input_data)
    if args.overwrite and not (production_matrix or cmiknn_matrix):
        args.output.write_text("")
        shutil.rmtree(artifact_directory, ignore_errors=True)
    artifact_directory.mkdir(parents=True, exist_ok=True)
    threads = args.threads if args.threads is not None else 1
    summary: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    current_case: SensitivityCase | None = None
    try:
        for case in cases:
            current_case = case
            case_diagnostics = prepare_case_diagnostics(
                input_data, case, artifact_directory
            )
            for method in methods:
                print(
                    f"running {case.timing_variant}/{case.preprocessing_profile} {method}",
                    file=sys.stderr,
                    flush=True,
                )
                row = _base_row(args, method, case, threads, case_diagnostics)
                row = _merge_child_result(
                    row,
                    _run_isolated_case(
                        args,
                        method,
                        case,
                        threads,
                        artifact_directory
                        / f"{method}-{case.timing_variant}-{case.preprocessing_profile}.npz",
                    ),
                )
                runtime.append_jsonl(args.output, row)
                rows.append(row)
                summary[row["status"]] = summary.get(row["status"], 0) + 1
        if production_matrix:
            agreement = synthesize_parcorr_matrix(rows)
            runtime.write_jsonl_artifact(agreement_path, [agreement])
        elif cmiknn_matrix:
            agreement = synthesize_cmiknn_matrix(rows)
            cmiknn_source = runtime.write_jsonl_artifact(agreement_path, [agreement])
            if validated_parcorr is not None:
                parcorr, parcorr_source = validated_parcorr
                comparison = compare_cmiknn_with_parcorr(agreement, parcorr)
                comparison["cmiknn_agreement"] = cmiknn_source
                comparison["parcorr_agreement"] = parcorr_source
                runtime.write_jsonl_artifact(
                    artifact_directory / "cmiknn-parcorr-comparison.jsonl", [comparison]
                )
    except Exception as error:
        if not (production_matrix or cmiknn_matrix):
            raise
        controlled = (
            error
            if isinstance(error, MatrixSynthesisError)
            else MatrixSynthesisError(
                f"incomplete matrix: {type(error).__name__}: {error}"
            )
        )
        _write_incomplete_matrix_manifest(
            agreement_path, controlled, current_case, rows
        )
        if isinstance(error, MatrixSynthesisError):
            raise error
        raise controlled from error
    print(f"{args.output} {json.dumps(summary, sort_keys=True)}")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="python -m thermodense.benchmarks.pcmci_real")
    commands = result.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser(
        "run", help="run real PCMCI+ methods in isolated child processes"
    )
    run_parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--methods", choices=METHODS, nargs="+")
    run_parser.add_argument(
        "--timing-variant",
        choices=(RAW_OBSERVED_DAILY, CENTERED_81_DAY),
        default=RAW_OBSERVED_DAILY,
    )
    run_parser.add_argument(
        "--preprocessing-profile",
        choices=(DETRENDED_ANOMALY, SEASONAL_ANOMALY),
        default=DETRENDED_ANOMALY,
    )
    run_parser.add_argument("--all-sensitivity-cases", action="store_true")
    run_parser.add_argument(
        "--production-sensitivity-matrix",
        action="store_true",
        help="run exactly the four ParCorr 0-180-day sensitivity cells and synthesize agreement",
    )
    run_parser.add_argument(
        "--cmiknn-sensitivity-matrix",
        action="store_true",
        help="run exactly the four CMIknn 0-10-day sensitivity cells and synthesize agreement",
    )
    run_parser.add_argument(
        "--parcorr-agreement",
        type=Path,
        help="validated ParCorr 0-180-day agreement for an optional bounded comparison",
    )
    run_parser.set_defaults(_tau_max_explicit=False)
    run_parser.add_argument(
        "--tau-max",
        type=int,
        default=DEFAULT_TAU_MAX,
        action=_TrackExplicitTauMax,
    )
    run_parser.add_argument(
        "--row-limit", type=int, help="calibration-only prefix row limit"
    )
    run_parser.add_argument(
        "--cmiknn-workers", type=int, default=DEFAULT_CMIKNN_WORKERS
    )
    run_parser.add_argument("--threads", type=int)
    run_parser.add_argument("--timeout", type=float, default=1800.0)
    run_parser.add_argument("--host-label", default="unspecified")
    run_parser.add_argument("--environment-label", default="unspecified")
    run_parser.add_argument("--environment-fingerprint", default="unspecified")
    run_parser.add_argument("--overwrite", action="store_true")
    gated = commands.add_parser(
        "gpdctorch-gated",
        help="run the validated full-row GPDCtorch gate state machine",
    )
    gated.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    gated.add_argument("--output", type=Path, required=True)
    gated.add_argument("--state", type=Path)
    gated.add_argument("--import-capability", type=Path)
    gated.add_argument("--parcorr-agreement", type=Path)
    gated.add_argument("--row-limit", type=int, help=argparse.SUPPRESS)
    gated.add_argument("--threads", type=int)
    gated.add_argument("--retry-failed", action="store_true")
    gated.add_argument("--overwrite", action="store_true")
    case = commands.add_parser("case", help=argparse.SUPPRESS)
    case.add_argument("--input", type=Path, required=True)
    case.add_argument("--method", choices=METHODS, required=True)
    case.add_argument("--tau-max", type=int, required=True)
    case.add_argument("--row-limit", type=int)
    case.add_argument("--cmiknn-workers", type=int, default=DEFAULT_CMIKNN_WORKERS)
    case.add_argument("--artifact", type=Path, required=True)
    case.add_argument(
        "--timing-variant",
        choices=(RAW_OBSERVED_DAILY, CENTERED_81_DAY),
        required=True,
    )
    case.add_argument("--gate-state", type=Path, help=argparse.SUPPRESS)
    case.add_argument("--gate-threads", type=int, default=1, help=argparse.SUPPRESS)
    case.add_argument("--gate-output", type=Path, help=argparse.SUPPRESS)
    case.add_argument(
        "--preprocessing-profile",
        choices=(DETRENDED_ANOMALY, SEASONAL_ANOMALY),
        required=True,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "case":
        return _child_main(args)
    try:
        if args.command == "run":
            _normalize_run_args(args)
        if args.command == "gpdctorch-gated":
            if args.threads is not None and args.threads <= 0:
                raise ValueError("--threads must be positive")
            return run_gpdctorch_gated(args)
        if (
            args.timeout <= 0
            or args.tau_max < 0
            or args.cmiknn_workers <= 0
            or (args.threads is not None and args.threads <= 0)
        ):
            raise ValueError(
                "--timeout, --cmiknn-workers, and --threads must be positive; --tau-max must be non-negative."
            )
        for method in args.methods or DEFAULT_METHODS:
            if method == "cmiknn":
                runtime.validate_cmiknn_tau(method, args.tau_max)
        return run(args)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
