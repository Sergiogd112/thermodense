"""Contracts for preserved workflow policy descriptions and validation."""

from copy import deepcopy

import pytest

from thermodense.benchmarks import global_activity_input, global_activity_pcmci, raw_3hour_pcmci
from thermodense.workflow_policies import (
    DAILY_ACTIVITY_CLASSIFICATION,
    DAILY_TRANSFORMATIONS,
    GLOBAL_PARCORR_POLICY,
    GLOBAL_POLYNOMIAL_POLICY,
    RAW_36_PARCORR_POLICY,
    PolicyMismatch,
    policy_for,
    validate_policy,
)


def _daily_manifest():
    return {
        "input_contract": global_activity_input.VERSION,
        "calendar": {"cadence": "daily UTC"},
        "transformations": DAILY_TRANSFORMATIONS,
        "f107_fit": "F10.7_OBS raw",
        "activity_classification": DAILY_ACTIVITY_CLASSIFICATION,
    }


def _polynomial_metadata():
    return {
        "status": "complete",
        "config": {"lag": 180},
        "interpretation": "predictive sensitivity benchmark; coefficients are not causal",
        "optimization_status": "bounded epoch budget; neither convergence nor globally optimal ridge is claimed",
        "evaluation_activity": "centered81 only; low <100, moderate 100<=F<160, high >=160; unknown excluded from grouped summaries",
    }


def test_policy_constants_match_frozen_workflow_contracts():
    assert GLOBAL_PARCORR_POLICY.max_lag_steps == global_activity_pcmci.TAU_MAX
    assert GLOBAL_POLYNOMIAL_POLICY.raw_f107_fit == "F10.7_OBS raw"
    assert GLOBAL_PARCORR_POLICY.activity_classification == global_activity_input.ACTIVITY_CLASSIFICATION
    assert RAW_36_PARCORR_POLICY.max_lag_steps == raw_3hour_pcmci.TAU_MAX
    assert RAW_36_PARCORR_POLICY.cadence_hours == 3
    assert policy_for("global-polynomial") is GLOBAL_POLYNOMIAL_POLICY


def test_global_polynomial_policy_accepts_completed_embedded_daily_manifest():
    validate_policy("global-polynomial", _polynomial_metadata(), _daily_manifest())


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("manifest", "calendar", {"cadence": "3-hour UTC"}),
        ("metadata", "config", {"lag": 179}),
        ("manifest", "f107_fit", "F10.7_OBS_CENTER81"),
        ("manifest", "activity_classification", "low <100, high >=100"),
        ("manifest", "transformations", {**DAILY_TRANSFORMATIONS, "standardizing": "global"}),
    ],
)
def test_global_polynomial_policy_rejects_mismatched_scientific_contract(section, key, value):
    metadata, manifest = _polynomial_metadata(), _daily_manifest()
    if section == "metadata":
        metadata[key] = value
    else:
        manifest[key] = value
    with pytest.raises(PolicyMismatch):
        validate_policy("global-polynomial", metadata, manifest)


def test_global_polynomial_policy_rejects_missing_scaling_metadata():
    metadata = _polynomial_metadata()
    del metadata["optimization_status"]
    with pytest.raises(PolicyMismatch):
        validate_policy("global-polynomial", metadata, _daily_manifest())


def test_raw_policy_requires_association_only_interpretation():
    legacy_digest, provenance_digest = "0" * 64, "1" * 64
    metadata = {
        "status": "completed",
        "nodes": raw_3hour_pcmci.node_names(),
        "algorithm": raw_3hour_pcmci.ALGORITHM_POLICY,
        "missing_data_policy": raw_3hour_pcmci.MISSING_DATA_POLICY,
        "link_eligibility": raw_3hour_pcmci.LINK_ELIGIBILITY,
        "input_availability": {
            name: raw_3hour_pcmci.REMOTE_SLOTS
            for name in raw_3hour_pcmci.node_names()
        },
        "code": {"raw_3hour_pcmci_sha256": "2" * 64},
        "versions": {
            "python": "3.14",
            "numpy": "2.4",
            "scipy": "1.17",
            "statsmodels": "0.14",
            "tigramite": {
                "version": "5.2.10.1",
                "commit": raw_3hour_pcmci.TIGRAMITE_COMMIT,
            },
        },
        "bh_family": {"method": "Benjamini-Hochberg", "scope": raw_3hour_pcmci.BH_SCOPE, "test_count": raw_3hour_pcmci.bh_test_count()},
        "stationarity": {"causal_interpretation_eligible": False, "causal_wording_permitted": False, "sensitivity_evidence_only": True},
        "interpretation": RAW_36_PARCORR_POLICY.interpretation,
    }
    manifest = {
        "input_contract": raw_3hour_pcmci.VERSION,
        "input_kind": "canonical",
        "calendar": raw_3hour_pcmci.CANONICAL_CALENDAR,
        "f10_7": {"representation": "raw_daily_held", "data_type": "OBS only", "non_obs": "unavailable", "interpolation": "none"},
        "transformations": DAILY_TRANSFORMATIONS,
        "source_node_order": list(raw_3hour_pcmci.LEGACY_SOURCE_NODE_ORDER),
        "legacy_prepared_semantics": {
            "prepared_artifact_status": "verified",
            "upstream_semantics_status": "declared_by_bound_legacy_provenance",
            "schema": raw_3hour_pcmci.LEGACY_PROVENANCE_SCHEMA,
            "source_version": raw_3hour_pcmci.LEGACY_PROVENANCE_VERSION,
            "prepared_input_sha256": legacy_digest,
            "provenance_sha256": provenance_digest,
            "channel_names": list(raw_3hour_pcmci.LEGACY_SOURCE_NODE_ORDER),
            "shape": [raw_3hour_pcmci.REMOTE_SLOTS, len(raw_3hour_pcmci.LEGACY_SOURCE_NODE_ORDER)],
            "calendar": {"cadence": raw_3hour_pcmci.CALENDAR_CADENCE},
            "f107_policy": raw_3hour_pcmci.LEGACY_F107_POLICY,
            "co2_missing_policy": raw_3hour_pcmci.LEGACY_CO2_MISSING_POLICY,
            "upstream_source_sha256": {
                "analysis_bundle.npz": "5" * 64,
                "final_arrays.npz": "6" * 64,
                "final_report.json": "7" * 64,
                "co2_daily_mlo.csv": "8" * 64,
            },
        },
        "data_mobility": raw_3hour_pcmci.DATA_MOBILITY_POLICY,
        "sources": {
            "legacy_prepared_npz": {
                "sha256": legacy_digest,
                "provenance_sha256": provenance_digest,
            },
            "celestrak_sw_all_csv": {
                "sha256": "3" * 64,
                "data_type_policy": raw_3hour_pcmci.F107_SOURCE_POLICY,
            },
            "saber_50x70_npz": {
                "sha256": "4" * 64,
                "channels": list(raw_3hour_pcmci.SABER_CHANNELS),
            },
        },
    }
    validate_policy("raw-36-parcorr", metadata, manifest)
    changed = deepcopy(metadata)
    changed["stationarity"] = {"causal_interpretation_eligible": True, "causal_wording_permitted": True, "sensitivity_evidence_only": True}
    with pytest.raises(PolicyMismatch):
        validate_policy("raw-36-parcorr", changed, manifest)
    changed = deepcopy(metadata)
    changed["link_eligibility"]["forbidden"] = "none"
    with pytest.raises(PolicyMismatch):
        validate_policy("raw-36-parcorr", changed, manifest)
