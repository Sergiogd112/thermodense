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
from pathlib import Path
import random
import resource
import shutil
import sys
import time
from typing import Any, Literal, cast
import warnings

import numpy as np
import polars as pl

from thermodense.benchmarks import runtime
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
MISSING_FLAG = -999999.0
SEED = 20260802
ROLLING_WINDOW = 1095
STATIONARITY_ALPHA = 0.05
MIN_STATIONARITY_SAMPLES = 2 * DEFAULT_TAU_MAX + 1

RAW_OBSERVED_DAILY = "raw_observed_daily"
CENTERED_81_DAY = "centered_81_day"
DETRENDED_ANOMALY = "detrended_anomaly"
SEASONAL_ANOMALY = "seasonal_anomaly"

type F107TimingVariant = Literal["raw_observed_daily", "centered_81_day"]
type PreprocessingProfile = Literal["detrended_anomaly", "seasonal_anomaly"]


@dataclass(frozen=True)
class SensitivityCase:
    """One preregistered cell in the PCMCI sensitivity matrix."""

    timing_variant: F107TimingVariant
    preprocessing_profile: PreprocessingProfile
    role: str


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
        "critical_values": {key: float(value) for key, value in critical_values.items()},
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
        "critical_values": {key: float(value) for key, value in critical_values.items()},
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
            nodes[node] = {"sample_count": 0, "span": None, "outcome": "not_qualified_missing_span"}
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
        ("kpss", "level_stationarity", "reject_level_stationarity", "do_not_reject_level_stationarity"),
    ):
        unavailable = sorted(set(node_names) - set(raw_p_values[family]))
        adjusted = holm_adjusted_pvalues(raw_p_values[family] | {node: 1.0 for node in unavailable})
        for node, p_value in raw_p_values[family].items():
            reject = adjusted[node] <= STATIONARITY_ALPHA
            nodes[node][family].update(
                raw_p_value=p_value,
                adjusted_p_value=adjusted[node],
                null_hypothesis=null,
                alternative_hypothesis=("stationary" if family == "adf" else "not_level_stationary"),
                reject_null=reject,
                outcome=reject_outcome if reject else retain_outcome,
            )
        families[family] = {
            "family_size": len(node_names),
            "tested_nodes": sorted(raw_p_values[family]),
            "unavailable_nodes": unavailable,
            "unavailable_node_policy": "unavailable nodes occupy full-family membership with p=1; they remain unqualified",
            "adjusted_p_values": {node: adjusted[node] for node in sorted(raw_p_values[family])},
        }
    for node in node_names:
        result = nodes[node]
        if "reject_null" in result.get("adf", {}) and "reject_null" in result.get("kpss", {}):
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
            "adf": {"regression": "c", "autolag": "AIC", "null_hypothesis": "unit_root"},
            "kpss": {"regression": "c", "nlags": "auto", "null_hypothesis": "level_stationarity"},
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
) -> None:
    if method != "gpdctorch":
        return
    if (case.timing_variant, case.preprocessing_profile, case.role) != (
        RAW_OBSERVED_DAILY,
        DETRENDED_ANOMALY,
        "primary",
    ):
        raise ValueError(
            "GPDCtorch may execute only the primary raw_observed_daily+detrended_anomaly case"
        )
    if tau_max != 1:
        raise ValueError("GPDCtorch currently supports only tau_max=1")
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
    case = case or sensitivity_case(RAW_OBSERVED_DAILY, DETRENDED_ANOMALY)
    _validate_gpdctorch_scope(method, tau_max, case, input_data)
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
    transformed = preprocess(input_data.values, input_data.dates, case.preprocessing_profile)
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
    return {
        "settings": settings,
        "seed": seed,
        "matrix_shapes": {
            name: list(np.asarray(value).shape) for name, value in matrices.items()
        },
        "result_digest": runtime.compact_result_digest(matrices),
        "process_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * 1024,
    } | (
        {
            "artifact": runtime.write_npz_artifact(
                artifact_path, matrices, node_names=node_names
            )
        }
        if artifact_path
        else {}
    )


def _child_main(args: argparse.Namespace) -> int:
    try:
        started = time.monotonic()
        case = sensitivity_case(args.timing_variant, args.preprocessing_profile)
        payload = run_pcmciplus(
            load_input(args.input, args.row_limit),
            args.method,
            args.tau_max,
            args.cmiknn_workers,
            args.artifact,
            case,
        )
        payload.update(status="succeeded", wall_seconds=time.monotonic() - started)
    except Exception as error:
        payload = {
            "status": "failed",
            "failure_reason": f"{type(error).__name__}: {error}",
        }
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0


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


def prepare_case_diagnostics(
    input_data: RealInput,
    case: SensitivityCase,
    artifact_directory: Path,
) -> dict[str, Any]:
    """Prepare immutable, method-independent diagnostics once for one case."""
    prepared_input, node_names, accepted_rows = prepare_sensitivity_input(input_data, case)
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
    rolling_artifact["diagnostic"] = "365-day rolling mean and variance; does not alter qualification"
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
                ROLLING_WINDOW if case.preprocessing_profile == DETRENDED_ANOMALY else None
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
        "package_versions": runtime.package_versions(),
        "git_commit": runtime.git_commit(),
        "wall_seconds": None,
        "process_max_rss_bytes": None,
        "matrix_shapes": {},
        "result_digest": None,
        "artifact": None,
        "failure_reason": None,
    }


def run(args: argparse.Namespace) -> int:
    artifact_directory = args.output.parent / f"{args.output.stem}_artifacts"
    if (args.output.exists() or artifact_directory.exists()) and not args.overwrite:
        raise ValueError(
            f"Refusing to overwrite existing result or artifacts: {args.output}; use --overwrite."
        )
    input_data = load_input(args.input, args.row_limit)
    if input_data.raw_f107 is None:
        raise ValueError("input CSV is missing raw observed daily F10.7")
    cases = (
        expand_sensitivity_cases()
        if args.all_sensitivity_cases
        else (sensitivity_case(args.timing_variant, args.preprocessing_profile),)
    )
    methods = args.methods or DEFAULT_METHODS
    if args.all_sensitivity_cases and "gpdctorch" in methods:
        raise ValueError("GPDCtorch sensitivity-matrix execution is not available")
    for method in methods:
        for case in cases:
            if method == "gpdctorch":
                _validate_gpdctorch_scope(method, args.tau_max, case, input_data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        args.output.write_text("")
        shutil.rmtree(artifact_directory, ignore_errors=True)
    artifact_directory.mkdir(parents=True, exist_ok=True)
    threads = args.threads if args.threads is not None else 1
    summary: dict[str, int] = {}
    for case in cases:
        case_diagnostics = prepare_case_diagnostics(input_data, case, artifact_directory)
        for method in methods:
            print(
                f"running {case.timing_variant}/{case.preprocessing_profile} {method}",
                file=sys.stderr,
                flush=True,
            )
            row = _base_row(args, method, case, threads, case_diagnostics)
            row.update(
                _run_isolated_case(
                    args,
                    method,
                    case,
                    threads,
                    artifact_directory
                    / f"{method}-{case.timing_variant}-{case.preprocessing_profile}.npz",
                )
            )
            runtime.append_jsonl(args.output, row)
            summary[row["status"]] = summary.get(row["status"], 0) + 1
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
    run_parser.add_argument("--tau-max", type=int, default=DEFAULT_TAU_MAX)
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
