"""Named immutable scientific contracts for the preserved workflows.

These descriptions deliberately retain each workflow's own scope rather than
turning its settings into a shared scientific default.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Any

from thermodense.benchmarks.raw_3hour_pcmci import (
    ALGORITHM_POLICY,
    BH_SCOPE,
    CANONICAL_CALENDAR,
    LINK_ELIGIBILITY,
    MISSING_DATA_POLICY,
    REMOTE_SLOTS,
    TIGRAMITE_COMMIT,
    bh_test_count,
    node_names,
    validate_canonical_manifest_metadata,
)


DAILY_TRANSFORMATIONS = {
    "detrending": "none",
    "deseasonalizing": "none",
    "standardizing": "none",
    "interpolation": "none",
}
DAILY_ACTIVITY_CLASSIFICATION = "F10.7_OBS_CENTER81: low <100, moderate 100<=F<160, high >=160"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class WorkflowPolicy:
    """Descriptive scientific scope for one preserved workflow."""

    semantic_id: str
    cadence_hours: int
    max_lag_steps: int
    raw_f107_fit: str
    activity_classification: str
    training_only_scaling: bool
    interpretation: str
    maturity: str


GLOBAL_PARCORR_POLICY = WorkflowPolicy(
    semantic_id="global-parcorr",
    cadence_hours=24,
    max_lag_steps=180,
    raw_f107_fit="F10.7_OBS raw",
    activity_classification=DAILY_ACTIVITY_CLASSIFICATION,
    training_only_scaling=False,
    interpretation="retrospective association/sensitivity only; not causal qualification",
    maturity="completed isolated retrospective association experiment",
)
RAW_36_PARCORR_POLICY = WorkflowPolicy(
    semantic_id="raw-36-parcorr",
    cadence_hours=3,
    max_lag_steps=1440,
    raw_f107_fit="raw_daily_held",
    activity_classification="not activity-binned",
    training_only_scaling=False,
    interpretation="association/sensitivity-only; this raw workflow never permits causal wording",
    maturity="isolated raw-36 experiment; evidence not qualified",
)
GLOBAL_POLYNOMIAL_POLICY = WorkflowPolicy(
    semantic_id="global-polynomial",
    cadence_hours=24,
    max_lag_steps=180,
    raw_f107_fit="F10.7_OBS raw",
    activity_classification=DAILY_ACTIVITY_CLASSIFICATION,
    training_only_scaling=True,
    interpretation="predictive sensitivity benchmark; coefficients are not causal",
    maturity="completed bounded predictive benchmark",
)
COMPARISON_POLICY = WorkflowPolicy(
    semantic_id="comparison",
    cadence_hours=24,
    max_lag_steps=180,
    raw_f107_fit="F10.7_OBS raw",
    activity_classification=DAILY_ACTIVITY_CLASSIFICATION,
    training_only_scaling=True,
    interpretation="predictive-only derived replay; differences are not causal",
    maturity="verified derived-result replay",
)

_POLICIES = {policy.semantic_id: policy for policy in (
    GLOBAL_PARCORR_POLICY,
    RAW_36_PARCORR_POLICY,
    GLOBAL_POLYNOMIAL_POLICY,
    COMPARISON_POLICY,
)}


class PolicyMismatch(ValueError):
    """Stored metadata does not satisfy the named workflow's contract."""


def policy_for(kind: str) -> WorkflowPolicy:
    """Return the immutable policy named by ``kind``."""
    try:
        return _POLICIES[kind]
    except KeyError as error:
        raise ValueError(f"unknown workflow policy {kind!r}") from error


def _require(mapping: Mapping[str, Any], key: str, expected: Any, context: str) -> None:
    actual = mapping.get(key)
    if actual != expected:
        raise PolicyMismatch(f"{context}.{key} must be {expected!r}, got {actual!r}")


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PolicyMismatch(f"{context} must be metadata mapping")
    return value


def _daily_input(manifest: Mapping[str, Any]) -> None:
    _require(manifest, "input_contract", "global-activity-daily-v1", "input_manifest")
    _require(_mapping(manifest.get("calendar"), "input_manifest.calendar"), "cadence", "daily UTC", "input_manifest.calendar")
    _require(manifest, "transformations", DAILY_TRANSFORMATIONS, "input_manifest")
    _require(manifest, "f107_fit", "F10.7_OBS raw", "input_manifest")
    _require(manifest, "activity_classification", DAILY_ACTIVITY_CLASSIFICATION, "input_manifest")


def _raw_input(manifest: Mapping[str, Any]) -> None:
    _require(manifest, "input_contract", "raw-36-node-native-3hour-v1", "input_manifest")
    _require(manifest, "input_kind", "canonical", "input_manifest")
    _require(manifest, "calendar", CANONICAL_CALENDAR, "input_manifest")
    f107 = _mapping(manifest.get("f10_7"), "input_manifest.f10_7")
    _require(f107, "representation", "raw_daily_held", "input_manifest.f10_7")
    _require(f107, "data_type", "OBS only", "input_manifest.f10_7")
    _require(f107, "non_obs", "unavailable", "input_manifest.f10_7")
    _require(f107, "interpolation", "none", "input_manifest.f10_7")
    _require(manifest, "transformations", DAILY_TRANSFORMATIONS, "input_manifest")
    try:
        validate_canonical_manifest_metadata(manifest)
    except ValueError as error:
        raise PolicyMismatch(str(error)) from error


def validate_policy(kind: str, metadata: Mapping[str, Any], input_manifest: Mapping[str, Any] | None = None) -> None:
    """Fail closed unless completed metadata exactly retains ``kind``'s scope."""
    policy_for(kind)
    record = _mapping(metadata, "metadata")
    manifest = _mapping(input_manifest, "input_manifest")

    if kind == "global-parcorr":
        _daily_input(manifest)
        _require(record, "status", "complete", "metadata")
        _require(record, "tau_max", 180, "metadata")
        algorithm = _mapping(record.get("algorithm"), "metadata.algorithm")
        _require(algorithm, "ci_test", "analytic ParCorr", "metadata.algorithm")
        _require(algorithm, "mask_type", "y", "metadata.algorithm")
        _require(record, "interpretation", GLOBAL_PARCORR_POLICY.interpretation, "metadata")
    elif kind == "raw-36-parcorr":
        _raw_input(manifest)
        _require(record, "status", "completed", "metadata")
        _require(record, "nodes", node_names(), "metadata")
        _require(record, "algorithm", ALGORITHM_POLICY, "metadata")
        _require(record, "missing_data_policy", MISSING_DATA_POLICY, "metadata")
        _require(record, "link_eligibility", LINK_ELIGIBILITY, "metadata")
        availability = _mapping(record.get("input_availability"), "metadata.input_availability")
        if set(availability) != set(node_names()) or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value > REMOTE_SLOTS
            for value in availability.values()
        ):
            raise PolicyMismatch("metadata.input_availability must count every canonical node")
        code = _mapping(record.get("code"), "metadata.code")
        code_digest = code.get("raw_3hour_pcmci_sha256")
        if not isinstance(code_digest, str) or not _SHA256.fullmatch(code_digest):
            raise PolicyMismatch("metadata.code must identify the raw runner bytes")
        versions = _mapping(record.get("versions"), "metadata.versions")
        if any(
            not isinstance(versions.get(name), str) or not versions.get(name)
            for name in ("python", "numpy", "scipy", "statsmodels")
        ):
            raise PolicyMismatch("metadata.versions must identify the scientific runtime")
        tigramite = _mapping(versions.get("tigramite"), "metadata.versions.tigramite")
        _require(tigramite, "commit", TIGRAMITE_COMMIT, "metadata.versions.tigramite")
        if not isinstance(tigramite.get("version"), str) or not tigramite.get("version"):
            raise PolicyMismatch("metadata.versions.tigramite.version must be nonempty")
        bh_family = _mapping(record.get("bh_family"), "metadata.bh_family")
        _require(bh_family, "method", "Benjamini-Hochberg", "metadata.bh_family")
        _require(bh_family, "scope", BH_SCOPE, "metadata.bh_family")
        _require(bh_family, "test_count", bh_test_count(), "metadata.bh_family")
        stationarity = _mapping(record.get("stationarity"), "metadata.stationarity")
        _require(stationarity, "causal_interpretation_eligible", False, "metadata.stationarity")
        _require(stationarity, "causal_wording_permitted", False, "metadata.stationarity")
        _require(stationarity, "sensitivity_evidence_only", True, "metadata.stationarity")
        _require(record, "interpretation", RAW_36_PARCORR_POLICY.interpretation, "metadata")
    elif kind == "global-polynomial":
        _daily_input(manifest)
        _require(record, "status", "complete", "metadata")
        _require(_mapping(record.get("config"), "metadata.config"), "lag", 180, "metadata.config")
        _require(record, "interpretation", GLOBAL_POLYNOMIAL_POLICY.interpretation, "metadata")
        _require(record, "optimization_status", "bounded epoch budget; neither convergence nor globally optimal ridge is claimed", "metadata")
        _require(record, "evaluation_activity", "centered81 only; low <100, moderate 100<=F<160, high >=160; unknown excluded from grouped summaries", "metadata")
    else:  # comparison
        _daily_input(manifest)
        settings = _mapping(record.get("settings"), "metadata.settings")
        _require(settings, "lag_days", 180, "metadata.settings")
        interpretation = record.get("interpretation")
        if not isinstance(interpretation, str) or "not causal" not in interpretation:
            raise PolicyMismatch("metadata.interpretation must retain the predictive-only, non-causal scope")
