"""Planning contracts for the blocked daily nonlinear CI benchmark.

This deliberately does not execute PCMCI or read the access-conditional daily
input.  The frozen :mod:`pcmci_real` runner remains the execution authority.
When the input and an RCoT/RCIT adapter are available, this module supplies the
comparison plan and result envelope that an executor must fill.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


SCHEMA_VERSION = "1"
BENCHMARK_VERSION = "daily-nonlinear-ci-1"
TAU_MAX = 10
F107_PREPROCESSING_CASES = (
    ("raw_observed_daily", "detrended_anomaly", "primary"),
    ("raw_observed_daily", "seasonal_anomaly", "robustness"),
    ("centered_81_day", "detrended_anomaly", "robustness"),
    ("centered_81_day", "seasonal_anomaly", "interaction_diagnostic"),
)
GATE_NAMES = (
    "identical_input_tau_max_10",
    "calibrated_type_i_error",
    "representative_power",
    "stability",
    "provenance",
    "runtime",
)


@dataclass(frozen=True)
class Candidate:
    """A method considered by the daily nonlinear benchmark."""

    name: str
    availability: str
    adapter: str | None
    availability_reason: str


CANDIDATES = (
    Candidate(
        "cmiknn",
        "available",
        "tigramite.independence_tests.cmiknn.CMIknn",
        "Existing daily nonlinear baseline; execution remains bounded to tau_max=10.",
    ),
    Candidate(
        "gpdctorch",
        "available_with_cuda_gate",
        "tigramite.independence_tests.gpdc_torch.GPDCtorch",
        "Existing daily baseline; the frozen runner requires its durable CUDA gates.",
    ),
    Candidate(
        "rcot_rcit",
        "adapter_absent",
        None,
        "RCoT/RCIT has no Tigramite adapter or validated project time-series/null-calibration contract.",
    ),
)


def candidate(name: str) -> Candidate:
    """Return a declared candidate without importing optional implementations."""
    for item in CANDIDATES:
        if item.name == name:
            return item
    raise ValueError(f"unknown daily nonlinear CI candidate: {name}")


def plan_document() -> dict[str, Any]:
    """Return the machine-readable, non-executing benchmark plan."""
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_version": BENCHMARK_VERSION,
        "state": "blocked_on_access_conditional_input_and_adapter",
        "execution_authority": "thermodense.benchmarks.pcmci_real",
        "input_contract": {
            "description": "same accepted-quality daily five-node input for every candidate and matrix cell",
            "required_identity_fields": [
                "input_sha256",
                "daily_date_sequence_sha256",
                "common_f107_support_sha256",
                "row_count",
            ],
            "shortened_record_forbidden": True,
        },
        "matrix": [
            {
                "timing_variant": timing,
                "preprocessing_profile": profile,
                "role": role,
                "tau_max": TAU_MAX,
            }
            for timing, profile, role in F107_PREPROCESSING_CASES
        ],
        "candidates": [asdict(item) for item in CANDIDATES],
        "promotion_gates": {
            "all_required": list(GATE_NAMES),
            "identical_input_tau_max_10": "same accepted-quality input identity and tau_max=10 across candidates",
            "calibrated_type_i_error": "null calibration and empirical type-I error are recorded against a predeclared acceptance criterion",
            "representative_power": "power is recorded on representative nonlinear alternatives against a predeclared acceptance criterion",
            "stability": "repeatability across declared seeds or resamples is recorded against a predeclared acceptance criterion",
            "provenance": "adapter version, settings, environment, input identity, and artifacts are retained",
            "runtime": "wall time and resource use are retained against a predeclared runtime criterion; speed alone cannot promote a method",
        },
        "interpretation": {
            "three_hour_parcorr": "Any disagreement is bounded daily method sensitivity, not a veto or a replacement for the 3-hour ParCorr core.",
            "tau_max_180": "outside_this_benchmark_requires_separate_decision",
            "external_method_speed_claim": "not made",
        },
    }


def result_template(candidate_name: str) -> dict[str, Any]:
    """Return a JSON-serializable result envelope for a future executor."""
    item = candidate(candidate_name)
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_version": BENCHMARK_VERSION,
        "state": "not_run",
        "candidate": asdict(item),
        "input_identity": None,
        "matrix_results": [
            {
                "timing_variant": timing,
                "preprocessing_profile": profile,
                "role": role,
                "tau_max": TAU_MAX,
                "status": "not_run",
                "artifact": None,
            }
            for timing, profile, role in F107_PREPROCESSING_CASES
        ],
        "calibration": {"null_model": None, "type_i_error": None, "acceptance_criterion": None},
        "power": {"representative_alternatives": None, "acceptance_criterion": None},
        "stability": {"replicates": None, "acceptance_criterion": None},
        "runtime": {"wall_seconds": None, "resource_use": None, "acceptance_criterion": None},
        "provenance": {"adapter_version": None, "settings": None, "environment": None},
        "promotion": {
            "decision": "pending",
            "speed_alone_promotes": False,
            "gates": {name: "not_assessed" for name in GATE_NAMES},
            "tau_max_180_decision": "separate_required",
        },
    }
