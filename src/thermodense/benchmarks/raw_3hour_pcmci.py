"""Canonical raw 36-node native-3-hour HASDM PCMCI+/analytic-ParCorr core.

This module intentionally does not share the frozen daily ``pcmci_real``
runner or accept any of the exploratory spline-F10.7 inputs.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import csv
from dataclasses import dataclass
import hashlib
from importlib import metadata as package_metadata
import json
import os
from pathlib import Path
import platform
import re
import time
from typing import Any, cast

import numpy as np

from thermodense.artifacts import ArtifactIntegrityError, verify_files

VERSION = "raw-36-node-native-3hour-v1"
MISSING_FLAG = -999999.0
ALPHA = 0.05
TAU_MAX = 1440
ALTITUDES_KM = tuple(range(175, 826, 25))
SABER_CHANNELS = ("CO2cool", "NOcool", "O2_1delta_ver", "OH_16_ver", "OH_20_ver")
DRIVER_NAMES = ("f10_7_raw_daily_held", "ap_native_3hour", "kp_native_3hour", "mlo_co2_daily_held", *SABER_CHANNELS)
REMOTE_SLOTS = 74_657
CANONICAL_START = "2000-01-01T00:00:00"
CANONICAL_END = "2025-07-20T00:00:00"
CALENDAR_CADENCE = "3-hour UTC"
CANONICAL_CALENDAR = {
    "start": CANONICAL_START,
    "end": CANONICAL_END,
    "cadence": CALENDAR_CADENCE,
    "length": REMOTE_SLOTS,
}
TIGRAMITE_COMMIT = "7c7b177cfbff77e11d805ab04fc2647301da1951"
MISSING_DATA_POLICY = {
    "missing_flag": MISSING_FLAG,
    "remove_missing_upto_maxlag": False,
    "interpolation": "none",
}
BH_SCOPE = "every predeclared eligible link and lag"
LEGACY_SOURCE_NODE_ORDER = (
    *(f"log10rho_{altitude}km" for altitude in ALTITUDES_KM),
    "F10.7_raw",
    "Ap",
    "Kp",
    "co2_mlo_ppm",
    *SABER_CHANNELS,
)
LEGACY_PROVENANCE_SCHEMA = "raw-36-legacy-prepared-provenance-v1"
LEGACY_PROVENANCE_VERSION = "throwaway-polynomial-var-v1"
LEGACY_F107_POLICY = "hasdm_f107: documented 3-hour spline representation of raw daily observations"
LEGACY_CO2_MISSING_POLICY = "raw daily Mauna Loa ppm, negative values missing, repeated over UTC day slots without interpolation"
F107_SOURCE_POLICY = "F10.7_OBS rows with F10.7_DATA_TYPE exactly OBS only; non-OBS dates unavailable; no interpolation"
LEGACY_UPSTREAM_BASENAMES = {
    "analysis_bundle.npz",
    "final_arrays.npz",
    "final_report.json",
    "co2_daily_mlo.csv",
}
DATA_MOBILITY_POLICY = {
    "classification": "restricted provider-derived input",
    "storage": "authorized local/site storage only",
    "cloud_upload": "requires explicit approval",
}
LINK_ELIGIBILITY = {
    "lagged": "driver→density and density→density",
    "contemporaneous": "driver→density and one representative per unordered density pair",
    "forbidden": "density→driver",
}
ALGORITHM_POLICY = {
    "name": "PCMCI+",
    "ci_test": "analytic ParCorr",
    "tau_min": 0,
    "tau_max": TAU_MAX,
    "physical_lag_window_days": [0, 180],
    "pc_alpha": ALPHA,
    "fdr_method": "none",
    "contemp_collider_rule": "majority",
    "conflict_resolution": True,
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class Raw36Input:
    timestamps: np.ndarray
    values: np.ndarray
    available: np.ndarray
    node_names: list[str]
    manifest: dict[str, Any]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)  # pyright: ignore[reportArgumentType]
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def density_names() -> list[str]:
    return [f"log10rho_{altitude}km" for altitude in ALTITUDES_KM]


def node_names() -> list[str]:
    """Return the exact ordered raw graph state."""
    return [*density_names(), *DRIVER_NAMES]


def validate_legacy_upstream_hashes(value: object) -> dict[str, str]:
    """Validate the legacy producer's declared four-source identity shape."""
    if not isinstance(value, Mapping) or {
        Path(str(name)).name for name in value
    } != LEGACY_UPSTREAM_BASENAMES:
        raise ValueError("legacy provenance must declare the exact four upstream sources")
    result: dict[str, str] = {}
    for name, digest in value.items():
        if not isinstance(name, str) or not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError("legacy upstream source identities require SHA-256 digests")
        result[name] = digest
    return result


def validate_canonical_manifest_metadata(manifest: Mapping[str, Any]) -> None:
    """Validate canonical-only source semantics that fixtures may omit."""
    if manifest.get("input_kind") != "canonical":
        raise ValueError("raw result verification requires a canonical input")
    if manifest.get("calendar") != CANONICAL_CALENDAR:
        raise ValueError("canonical input calendar identity is incorrect")
    if manifest.get("source_node_order") != list(LEGACY_SOURCE_NODE_ORDER):
        raise ValueError("canonical input source-node order is unbound")
    if manifest.get("data_mobility") != DATA_MOBILITY_POLICY:
        raise ValueError("canonical input restricted-data mobility policy is absent")
    semantics = manifest.get("legacy_prepared_semantics")
    sources = manifest.get("sources")
    if not isinstance(semantics, Mapping) or not isinstance(sources, Mapping):
        raise ValueError("canonical input lacks bound legacy/source semantics")
    legacy_source = sources.get("legacy_prepared_npz")
    weather_source = sources.get("celestrak_sw_all_csv")
    saber_source = sources.get("saber_50x70_npz")
    if not all(
        isinstance(source, Mapping)
        for source in (legacy_source, weather_source, saber_source)
    ):
        raise ValueError("canonical input lacks required source identities")
    assert isinstance(legacy_source, Mapping)
    assert isinstance(weather_source, Mapping)
    assert isinstance(saber_source, Mapping)
    legacy_digest = legacy_source.get("sha256")
    provenance_digest = legacy_source.get("provenance_sha256")
    if not all(
        isinstance(value, str) and _SHA256.fullmatch(value)
        for value in (legacy_digest, provenance_digest)
    ):
        raise ValueError("canonical legacy source digests are invalid")
    for name, source in (("CelesTrak", weather_source), ("SABER", saber_source)):
        digest = source.get("sha256")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError(f"canonical {name} source digest is invalid")
    if weather_source.get("data_type_policy") != F107_SOURCE_POLICY:
        raise ValueError("canonical CelesTrak OBS-only policy is unbound")
    if saber_source.get("channels") != list(SABER_CHANNELS):
        raise ValueError("canonical SABER source channels are unbound")
    upstream_hashes = validate_legacy_upstream_hashes(
        semantics.get("upstream_source_sha256")
    )
    expected_semantics = {
        "prepared_artifact_status": "verified",
        "upstream_semantics_status": "declared_by_bound_legacy_provenance",
        "schema": LEGACY_PROVENANCE_SCHEMA,
        "source_version": LEGACY_PROVENANCE_VERSION,
        "prepared_input_sha256": legacy_digest,
        "provenance_sha256": provenance_digest,
        "channel_names": list(LEGACY_SOURCE_NODE_ORDER),
        "shape": [REMOTE_SLOTS, len(LEGACY_SOURCE_NODE_ORDER)],
        "calendar": {"cadence": CALENDAR_CADENCE},
        "f107_policy": LEGACY_F107_POLICY,
        "co2_missing_policy": LEGACY_CO2_MISSING_POLICY,
        "upstream_source_sha256": upstream_hashes,
    }
    if dict(semantics) != expected_semantics:
        raise ValueError("canonical legacy prepared semantics are inconsistent")


def _load_array(saved: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    if name not in saved.files:
        raise ValueError(f"input NPZ is missing {name}")
    value = np.asarray(saved[name])
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    return value


def _daily_held(values: np.ndarray, available: np.ndarray, name: str) -> None:
    for start in range(0, len(values), 8):
        stop = min(start + 8, len(values))
        if not np.all(available[start:stop] == available[start]):
            raise ValueError(f"{name} mask is not held across a UTC day")
        if available[start] and not np.all(values[start:stop] == values[start]):
            raise ValueError(f"{name} values are not held across a UTC day")


def calendar_declaration(timestamps: np.ndarray) -> dict[str, Any]:
    """Return the exact UTC calendar declaration for an already validated payload."""
    return {
        "start": np.datetime_as_string(timestamps[0], unit="s"),
        "end": np.datetime_as_string(timestamps[-1], unit="s"),
        "cadence": CALENDAR_CADENCE,
        "length": len(timestamps),
    }


def bh_test_count() -> int:
    """Return the one predeclared raw PCMCI+ BH-family size."""
    density_count = len(ALTITUDES_KM)
    driver_count = len(DRIVER_NAMES)
    lagged = density_count * (density_count + driver_count) * TAU_MAX
    contemporaneous_drivers = density_count * driver_count
    contemporaneous_density_pairs = density_count * (density_count - 1) // 2
    return lagged + contemporaneous_drivers + contemporaneous_density_pairs


def tigramite_identity(distribution: Any | None = None) -> dict[str, str]:
    """Require the pinned Tigramite VCS revision before an expensive fit starts."""
    installed = package_metadata.distribution("tigramite") if distribution is None else distribution
    try:
        direct_url_text = installed.read_text("direct_url.json")
        if not isinstance(direct_url_text, str):
            raise ValueError("direct_url.json is absent")
        direct_url = json.loads(direct_url_text)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("installed tigramite lacks a readable direct_url.json VCS identity") from error
    commit = direct_url.get("vcs_info", {}).get("commit_id")
    if commit != TIGRAMITE_COMMIT:
        raise ValueError(f"installed tigramite commit must be {TIGRAMITE_COMMIT}, got {commit!r}")
    return {"version": installed.version, "commit": commit}


def load_manifest(path: Path) -> Raw36Input:
    """Load the strict raw input contract; fixture manifests may use short arrays."""
    loaded_manifest = json.loads(path.read_text())
    if not isinstance(loaded_manifest, Mapping):
        raise ValueError("input manifest must be a JSON object")
    manifest = dict(loaded_manifest)
    if manifest.get("input_contract") != VERSION:
        raise ValueError("manifest is not the canonical raw-36-node input contract")
    if manifest.get("input_kind") not in {"canonical", "fixture"}:
        raise ValueError("manifest input_kind must be canonical or fixture")
    f107_policy = manifest.get("f10_7", {})
    if not isinstance(f107_policy, Mapping) or any(
        f107_policy.get(key) != expected
        for key, expected in {
            "representation": "raw_daily_held",
            "data_type": "OBS only",
            "non_obs": "unavailable",
            "interpolation": "none",
        }.items()
    ):
        raise ValueError("canonical input requires raw daily-held F10.7, never spline F10.7")
    if manifest.get("transformations") != {
        "detrending": "none", "deseasonalizing": "none", "standardizing": "none", "interpolation": "none"
    }:
        raise ValueError("canonical raw input forbids transformations and interpolation")
    core_name, core_digest = manifest.get("core_npz"), manifest.get("core_sha256")
    saber_name, saber_digest = manifest.get("saber_npz"), manifest.get("saber_sha256")
    if not all(
        isinstance(value, str)
        for value in (core_name, core_digest, saber_name, saber_digest)
    ):
        raise ValueError("input manifest must declare relative core/SABER paths and SHA-256 digests")
    declared = {
        cast(str, core_name): cast(str, core_digest),
        cast(str, saber_name): cast(str, saber_digest),
    }
    try:
        verify_files(path.parent, declared, required=declared)
    except ArtifactIntegrityError as error:
        raise ValueError("input core/SABER files do not match the manifest") from error
    core_path = (path.parent / cast(str, manifest["core_npz"])).resolve(strict=True)
    saber_path = (path.parent / cast(str, manifest["saber_npz"])).resolve(strict=True)
    if core_path == saber_path:
        raise ValueError("input manifest must declare distinct core and SABER artifacts")
    with np.load(core_path, allow_pickle=False) as core, np.load(saber_path, allow_pickle=False) as saber:
        timestamps = np.asarray(core["timestamps"]).astype("datetime64[ns]")
        count = len(timestamps)
        timestamp_ns = timestamps.astype("int64")
        day_ns = 24 * 3600 * 10**9
        if count == 0 or np.isnat(timestamps).any() or timestamp_ns[0] % day_ns or not np.all(np.diff(timestamp_ns) == 3 * 3600 * 10**9):
            raise ValueError("timestamps must be a nonempty consecutive UTC 3-hour calendar")
        if manifest.get("calendar") != calendar_declaration(timestamps):
            raise ValueError("manifest calendar does not match the core payload")
        if tuple(np.asarray(core["density_altitudes_km"]).tolist()) != ALTITUDES_KM:
            raise ValueError("density altitude nodes must be exactly 175..825 km at 25 km")
        density = _load_array(core, "density_log10_values", (count, len(ALTITUDES_KM))).astype(float)
        density_mask = _load_array(core, "density_masks", density.shape).astype(bool)
        drivers: list[np.ndarray] = []
        driver_masks: list[np.ndarray] = []
        for name in ("f107", "ap", "kp", "mlo_co2"):
            drivers.append(_load_array(core, f"{name}_values", (count,)).astype(float))
            driver_masks.append(_load_array(core, f"{name}_masks", (count,)).astype(bool))
        saber_timestamps = np.asarray(saber["timestamps"]).astype("datetime64[ns]")
        if not np.array_equal(saber_timestamps, timestamps):
            raise ValueError("SABER timestamps must exactly match the core UTC calendar")
        if tuple(str(item) for item in saber["channels"].tolist()) != SABER_CHANNELS:
            raise ValueError("SABER channels must be the five exact vertical totals")
        saber_values = _load_array(saber, "values", (len(SABER_CHANNELS), count)).astype(float)
        saber_masks = _load_array(saber, "masks", saber_values.shape).astype(bool)
    if manifest["input_kind"] == "canonical" and manifest["calendar"] != CANONICAL_CALENDAR:
        raise ValueError("canonical input must retain the exact 2000-01-01 through 2025-07-20 UTC calendar")
    if manifest["input_kind"] == "canonical":
        validate_canonical_manifest_metadata(manifest)
    f107, co2 = drivers[0], drivers[3]
    _daily_held(f107, driver_masks[0], "F10.7")
    _daily_held(co2, driver_masks[3], "MLO CO2")
    values = np.column_stack((density, *drivers, saber_values.T))
    available = np.column_stack((density_mask, *driver_masks, saber_masks.T)) & np.isfinite(values)
    values = values.copy()
    values[~available] = np.nan
    return Raw36Input(timestamps, values, available, node_names(), manifest)


def link_assumptions(node_names: list[str], tau_max: int = TAU_MAX) -> dict[int, dict[tuple[int, int], str]]:
    """Permit driver→density and density↔density, never density→driver."""
    densities = set(density_names())
    result = {target: {} for target in range(len(node_names))}
    for target_i, target in enumerate(node_names):
        if target not in densities:
            continue
        for source_i, source in enumerate(node_names):
            if source in densities or source in DRIVER_NAMES:
                for lag in range(1, tau_max + 1):
                    result[target_i][(source_i, -lag)] = "-?>"
                if source in DRIVER_NAMES:
                    result[target_i][(source_i, 0)] = "-?>"
                elif source_i < target_i:
                    result[target_i][(source_i, 0)] = "o?o"
    return result


def eligible_locations(node_names: list[str], tau_max: int = TAU_MAX) -> list[tuple[int, int, int]]:
    """Return the complete, predeclared BH family in deterministic order."""
    assumptions = link_assumptions(node_names, tau_max)
    return sorted((source, target, -lag) for target, links in assumptions.items() for (source, lag) in links)


def bh_qvalues(pvalues: np.ndarray) -> np.ndarray:
    if not np.isfinite(pvalues).all() or np.any((pvalues < 0) | (pvalues > 1)):
        raise ValueError("BH family requires finite p-values in [0, 1]")
    order = np.argsort(pvalues)
    adjusted = np.minimum.accumulate((pvalues[order] * len(pvalues) / np.arange(1, len(pvalues) + 1))[::-1])[::-1].clip(0, 1)
    result = np.empty_like(pvalues)
    result[order] = adjusted
    return result


def validate_result_matrices(result: Mapping[str, np.ndarray]) -> None:
    """Reject a fitted result that cannot represent the exact declared family."""
    expected_shape = (len(node_names()), len(node_names()), TAU_MAX + 1)
    for name in ("graph", "p_matrix", "val_matrix"):
        value = result.get(name)
        if not isinstance(value, np.ndarray) or value.shape != expected_shape:
            shape = None if not isinstance(value, np.ndarray) else value.shape
            raise ValueError(f"{name} must have shape {expected_shape}, got {shape}")
    density_count = len(ALTITUDES_KM)
    upper = np.triu_indices(density_count, k=1)
    for name in ("p_matrix", "val_matrix"):
        matrix = result[name]
        eligible = (
            matrix[:, :density_count, 1:],
            matrix[density_count:, :density_count, 0],
            matrix[:density_count, :density_count, 0][upper],
        )
        if sum(values.size for values in eligible) != bh_test_count():
            raise ValueError("fitted matrix does not cover the exact declared BH family")
        if not all(np.isfinite(values).all() for values in eligible):
            raise ValueError(f"eligible {name} values must all be finite")
        if name == "p_matrix" and any(
            np.any((values < 0) | (values > 1)) for values in eligible
        ):
            raise ValueError("eligible p_matrix values must lie in [0, 1]")


def pairwise_support(available: np.ndarray, source: int, target: int, lag: int) -> int:
    if lag == 0:
        return int((available[:, source] & available[:, target]).sum())
    return int((available[:-lag, source] & available[lag:, target]).sum())


def result_rows(result: dict[str, np.ndarray], input_data: Raw36Input) -> list[dict[str, Any]]:
    locations = eligible_locations(input_data.node_names, result["p_matrix"].shape[2] - 1)
    pvalues = np.array([result["p_matrix"][source, target, lag] for source, target, lag in locations], float)
    qvalues = bh_qvalues(pvalues)
    rows = []
    for (source, target, lag), qvalue in zip(locations, qvalues, strict=True):
        mark = str(result["graph"][source, target, lag])
        out_source, out_target = source, target
        normalized_mark = mark
        direction = "source_to_target"
        ambiguous = False
        if lag == 0 and mark == "<--":
            out_source, out_target = target, source
            normalized_mark = "-->"
            direction = "target_to_source"
        elif mark != "-->":
            direction = "ambiguous" if mark else "absent"
            ambiguous = bool(mark)
        rows.append({"graph_source": input_data.node_names[source], "graph_target": input_data.node_names[target], "source": input_data.node_names[out_source], "target": input_data.node_names[out_target], "lag_steps": lag, "lag_hours": lag * 3, "lag_days": lag / 8, "partial_r": float(result["val_matrix"][source, target, lag]), "raw_p_value": float(result["p_matrix"][source, target, lag]), "q_value": float(qvalue), "graph_mark_original": mark, "graph_mark_normalized": normalized_mark, "adjacency": bool(mark), "direction": direction, "ambiguous": ambiguous, "pairwise_support": pairwise_support(input_data.available, source, target, lag), "support_scope": "pairwise availability only; excludes conditioning variables and Tigramite cut-off", "detected": bool(qvalue <= ALPHA and mark)})
    return rows


def raw_stationarity_diagnostics(input_data: Raw36Input) -> dict[str, Any]:
    """Report raw-series ADF/KPSS diagnostics only; never transform the input."""
    from thermodense.benchmarks.pcmci_real import stationarity_qualification

    # The existing diagnostic implements the registered Holm qualification; raw
    # use is deliberately reporting-only and cannot become a preprocessing step.
    diagnostics = stationarity_qualification(input_data.values, input_data.timestamps, input_data.node_names)
    minimum_samples = 2 * TAU_MAX + 1
    too_short = []
    bounded_kpss = []
    for node, record in diagnostics["nodes"].items():
        if record["sample_count"] < minimum_samples:
            record["outcome"] = "not_qualified_too_short_span"
            too_short.append(node)
        kpss = record.get("kpss")
        if isinstance(kpss, dict):
            warnings = kpss.get("warnings", [])
            if isinstance(warnings, list) and any(
                isinstance(warning, str)
                and "actual p-value is smaller than" in warning.lower()
                for warning in warnings
            ):
                kpss["p_value_kind"] = "upper_bound"
                kpss["qualification_interpretation"] = (
                    "indeterminate; the tabulated upper bound cannot establish "
                    "Holm non-rejection"
                )
                if record.get("outcome") == "qualified":
                    record["outcome"] = "not_qualified_kpss_pvalue_bounded"
                bounded_kpss.append(node)
    if too_short:
        diagnostics["ineligibility_reason"] = "one or more graph nodes lack the 2*tau_max+1 raw observations required for this run"
    if bounded_kpss:
        diagnostics["bounded_kpss_nodes"] = bounded_kpss
        diagnostics["kpss_bound_policy"] = (
            "a returned upper-bound p-value is qualification-indeterminate and "
            "cannot support a qualified node label"
        )
    diagnostics["settings"]["minimum_samples"] = minimum_samples
    diagnostics["settings"]["minimum_samples_justification"] = "2 * 1440 + 1 for the native 3-hour 0--180-day lag window"
    diagnostics["input_representation"] = "raw; no detrending, deseasonalizing, standardizing, or interpolation"
    diagnostics["raw_test_interpretation"] = "diagnostic only; bounded KPSS p-values and non-rejection cannot establish stationarity"
    diagnostics["causal_interpretation_eligible"] = False
    diagnostics["causal_wording_permitted"] = False
    diagnostics["sensitivity_evidence_only"] = True
    diagnostics["qualification_scope"] = "raw workflow is not a registered causal preprocessing profile"
    return diagnostics


def run_pcmci(input_data: Raw36Input) -> dict[str, np.ndarray]:
    from tigramite import data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI

    values = input_data.values.copy()
    values[~input_data.available] = MISSING_FLAG
    frame = pp.DataFrame(values, datatime=np.arange(len(values)), var_names=input_data.node_names, missing_flag=MISSING_FLAG, remove_missing_upto_maxlag=False)
    return cast(
        dict[str, np.ndarray],
        PCMCI(frame, cond_ind_test=ParCorr(significance="analytic"), verbosity=0).run_pcmciplus(link_assumptions=link_assumptions(input_data.node_names), tau_min=0, tau_max=TAU_MAX, pc_alpha=ALPHA, fdr_method="none", contemp_collider_rule="majority", conflict_resolution=True),
    )


def run(manifest_path: Path, output: Path, host_label: str) -> Path:
    input_data = load_manifest(manifest_path)
    tigramite = tigramite_identity()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    provenance_path = output / "provenance.json"
    base: dict[str, Any] = {
        "version": VERSION,
        "status": "fitting",
        "host_label": host_label,
        "input_manifest": input_data.manifest,
        "nodes": input_data.node_names,
        "algorithm": ALGORITHM_POLICY,
        "missing_data_policy": MISSING_DATA_POLICY,
        "link_eligibility": LINK_ELIGIBILITY,
        "input_availability": {
            name: int(input_data.available[:, index].sum())
            for index, name in enumerate(input_data.node_names)
        },
    }
    base["interpretation"] = (
        "association/sensitivity-only; this raw workflow never permits causal wording"
    )
    base["code"] = {"raw_3hour_pcmci_sha256": _sha256(Path(__file__))}
    base["versions"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": package_metadata.version("scipy"),
        "statsmodels": package_metadata.version("statsmodels"),
        "tigramite": tigramite,
    }
    _atomic_json(provenance_path, base)
    results_path = output / "pcmci_results.npz"
    rows_path = output / "eligible_links.csv"
    fit_persisted = False
    result_validated = False
    try:
        stationarity = raw_stationarity_diagnostics(input_data)
        result = run_pcmci(input_data)
        _atomic_npz(
            results_path,
            graph=result["graph"],
            p_matrix=result["p_matrix"],
            val_matrix=result["val_matrix"],
            node_names=np.asarray(input_data.node_names),
        )
        fit_persisted = True
        validate_result_matrices(result)
        result_validated = True
        _atomic_json(
            provenance_path,
            base
            | {
                "status": "fit_complete_export_pending",
                "runtime_seconds": time.monotonic() - started,
                "stationarity": stationarity,
                "result_files": {results_path.name: _sha256(results_path)},
            },
        )
        rows = result_rows(result, input_data)
        if len(rows) != bh_test_count():
            raise ValueError("eligible-link export does not match the declared BH family")
        _atomic_csv(rows_path, rows)
        _atomic_json(
            provenance_path,
            base
            | {
                "status": "completed",
                "runtime_seconds": time.monotonic() - started,
                "bh_family": {
                    "method": "Benjamini-Hochberg",
                    "test_count": bh_test_count(),
                    "scope": BH_SCOPE,
                },
                "stationarity": stationarity,
                "result_files": {
                    results_path.name: _sha256(results_path),
                    rows_path.name: _sha256(rows_path),
                },
            },
        )
    except BaseException as error:
        if fit_persisted and not result_validated:
            status = "fit_complete_validation_failed"
        elif fit_persisted:
            status = "fit_complete_export_failed"
        else:
            status = "fit_failed_or_interrupted"
        failure = base | {
            "status": status,
            "runtime_seconds": time.monotonic() - started,
            "error": f"{type(error).__name__}: {error}",
        }
        if fit_persisted:
            failure["result_files"] = {results_path.name: _sha256(results_path)}
        _atomic_json(provenance_path, failure)
        raise
    return output


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("manifest", type=Path)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--host-label", required=True)
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(run(args.manifest, args.output, args.host_label))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
