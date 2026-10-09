"""Exploratory, resumable 3-hour Figure 5 residual-dependence pipeline.

This deliberately does not replace the daily Figure 5 prototype.  It keeps the
HASDM sampling frame intact and delegates JB execution to the locally supplied,
unmodified SET provider wrapper (which owns the documented JB driver lags).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

ALTITUDES = tuple(range(175, 826, 25))
MODELS = ("nrlmsise_00", "nrlmsis_2p0", "nrlmsis_2p1", "jb2006", "jb2008")
MISSING_FLAG = -9999.0
STEP_HOURS = 3
PHYSICAL_LAG_DAYS = 183
TAU_MAX = 1464  # 183 days * 24 / 3; intentionally not the daily prototype's lag.
ALPHA = 0.05
VERSION = "figure5-3hour-v1"
SAMPLES = Path(
    "outputs/figures/results/hasdm_msis_model_errors/data/hasdm_msis_errors_nearest_timestamp_grid_samples.parquet"
)
SW_ALL = Path("data/original/space_weather/SW-All.csv")
OUTPUT = Path("outputs/prototypes/empirical_model_figure5_3hour")
SHARDED_DIRECTORY = "sharded_cases"
SPACEHOPPER_CASE_IDS = (
    "nrlmsise_00-ap",
    "nrlmsise_00-kp",
    "nrlmsis_2p0-ap",
    "nrlmsis_2p0-kp",
    "jb2006-ap",
)
EQUIVALENCE_RTOL = 1e-11
EQUIVALENCE_ATOL = 1e-13


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def lag_steps(days: int = PHYSICAL_LAG_DAYS, step_hours: int = STEP_HOURS) -> int:
    """Convert an exactly cadence-aligned physical duration to PCMCI steps."""
    hours = days * 24
    if hours % step_hours:
        raise ValueError("physical lag must fall on a cadence boundary")
    return hours // step_hours


def complete_3hour_calendar(samples: pd.DataFrame) -> pd.DataFrame:
    """Return every 3-hour timestamp, with absent HASDM sample rows left missing."""
    timestamps = pd.to_datetime(samples["timestamp"])
    if timestamps.empty:
        raise ValueError("HASDM samples are empty")
    calendar = pd.DataFrame(
        {"timestamp": pd.date_range(timestamps.min(), timestamps.max(), freq="3h")}
    )
    if "altitude_km" not in samples:
        return calendar.merge(
            samples, on="timestamp", how="left", validate="one_to_many"
        )
    complete = pd.MultiIndex.from_product(
        [calendar.timestamp, sorted(samples.altitude_km.unique())],
        names=["timestamp", "altitude_km"],
    ).to_frame(index=False)
    return complete.merge(
        samples, on=["timestamp", "altitude_km"], how="left", validate="one_to_one"
    )


def selected_frame(samples: pd.DataFrame) -> pd.DataFrame:
    """Validate the exact nearest-grid frame: one longitude per time/altitude."""
    frame = samples.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["altitude_km"] = frame["Altitude (m)"].astype(int) // 1000
    keys = ["timestamp", "altitude_km"]
    if frame.duplicated(keys).any() or set(frame.altitude_km.unique()) != set(
        ALTITUDES
    ):
        raise ValueError(
            "HASDM frame must have one row for every timestamp/altitude key"
        )
    return frame.sort_values(keys).reset_index(drop=True)


def f107_spline(
    calendar: pd.Series, weather: pd.DataFrame
) -> tuple[np.ndarray, dict[str, object]]:
    """Cubic-spline raw daily F10.7 without edge extrapolation.

    SW-All DATE labels are treated as UTC midnight daily anchors.  Knots retain
    their supplied values exactly; values outside the knot domain are missing.
    """
    knots = weather[["DATE", "F10.7_OBS"]].dropna().copy()
    knots["DATE"] = pd.to_datetime(knots["DATE"])
    knots = knots.drop_duplicates("DATE").sort_values("DATE")
    x = (knots.DATE - knots.DATE.iloc[0]).dt.total_seconds().to_numpy() / 3600
    query_time = pd.Series(pd.to_datetime(calendar))
    query = (query_time - knots.DATE.iloc[0]).dt.total_seconds().to_numpy() / 3600
    result = np.full(len(query), np.nan)
    inside = (query >= x[0]) & (query <= x[-1])
    result[inside] = CubicSpline(x, knots["F10.7_OBS"].to_numpy(), extrapolate=False)(
        query[inside]
    )
    # Avoid numerical drift at source knots, and make the preservation contract explicit.
    for knot_x, value in zip(x, knots["F10.7_OBS"], strict=True):
        result[query == knot_x] = value
    return result, {
        "date_label_anchor": "SW-All DATE is UTC midnight",
        "source_knots": len(knots),
        "knot_start": knots.DATE.iloc[0].isoformat(),
        "knot_end": knots.DATE.iloc[-1].isoformat(),
        "extrapolation": "forbidden",
    }


def slot_drivers(calendar: pd.Series, weather: pd.DataFrame) -> pd.DataFrame:
    """Map native AP1..8/KP1..8 to UTC 00,03,...,21, never daily-average."""
    time = pd.Series(pd.to_datetime(calendar))
    date = time.dt.normalize()
    slot = time.dt.hour // 3 + 1
    indexed = weather.copy()
    indexed["DATE"] = pd.to_datetime(indexed["DATE"])
    indexed = indexed.set_index("DATE")
    output = pd.DataFrame({"timestamp": time})
    for prefix in ("AP", "KP"):
        values = []
        for day, index in zip(date, slot, strict=True):
            values.append(
                indexed.at[day, f"{prefix}{index}"] if day in indexed.index else np.nan
            )
        output[prefix.lower()] = values
    return output


def preprocess_3hour(values: np.ndarray, timestamps: pd.Series) -> np.ndarray:
    """3-hour adaptation: DOY/UTC-slot anomalies, 3-year centered mean, z-score.

    Feb 29 is a separate seasonal key; missing observations remain missing.  This
    is intentionally not claimed to be byte-identical to daily preprocessing.
    """
    series = pd.Series(np.asarray(values, dtype=float))
    time = pd.Series(pd.to_datetime(timestamps))
    # Month/day avoids shifting every post-February seasonal key in leap years.
    keys = pd.MultiIndex.from_arrays([time.dt.month, time.dt.day, time.dt.hour // 3])
    seasonal = series - series.groupby(keys).transform("mean")
    window = 3 * 365 * 8
    detrended = seasonal - seasonal.rolling(window, center=True, min_periods=1).mean()
    scale = detrended.std(ddof=0, skipna=True)
    return (
        detrended / scale if scale and np.isfinite(scale) else detrended * np.nan
    ).to_numpy()


def link_assumptions(
    columns: list[str], targets: list[str], tau_max: int
) -> dict[int, dict[tuple[int, int], str]]:
    """Bound graph scope: driver history, driver→target, and target self-history."""
    assumptions = {j: {} for j in range(len(columns))}
    target_set = set(targets)
    for target_i, target in enumerate(columns):
        for cause_i, cause in enumerate(columns):
            allowed = (
                (target not in target_set and cause not in target_set)
                or (target in target_set and cause not in target_set)
                or cause == target
            )
            if allowed:
                for lag in range(1, tau_max + 1):
                    assumptions[target_i][(cause_i, -lag)] = "-?>"
            if target in target_set and cause not in target_set:
                assumptions[target_i][(cause_i, 0)] = (
                    "-?>"  # PCMCI+ contemporaneous driver→target
                )
            elif (
                target not in target_set
                and cause not in target_set
                and cause_i < target_i
            ):
                assumptions[target_i][(cause_i, 0)] = "o?o"
    return assumptions


def cases() -> list[dict[str, str]]:
    return [
        {"id": f"{model}-{geo}", "model": model, "geomagnetic": geo}
        for model in MODELS
        for geo in ("ap", "kp")
    ]


def declared_fdr_tests(tau_max: int) -> int:
    """Predeclared driver-to-target family: two drivers × 27 targets × all lags."""
    return 2 * len(ALTITUDES) * (tau_max + 1)


def driver_target_bh_qvalues(
    raw_pvalues: np.ndarray, columns: list[str], targets: list[str], drivers: list[str]
) -> np.ndarray:
    """BH only the predeclared driver→target family, including lag zero."""
    qvalues = np.full_like(raw_pvalues, np.nan, dtype=float)
    locations = [
        (columns.index(source), columns.index(target), lag)
        for source in drivers
        for target in targets
        for lag in range(raw_pvalues.shape[2])
    ]
    pvalues = np.array([raw_pvalues[location] for location in locations], dtype=float)
    if not np.isfinite(pvalues).all():
        raise ValueError("non-finite raw p-value in predeclared BH family")
    order = np.argsort(pvalues)
    ranked = pvalues[order] * len(pvalues) / np.arange(1, len(pvalues) + 1)
    corrected = np.minimum.accumulate(ranked[::-1])[::-1].clip(0, 1)
    for location, value in zip(
        np.array(locations, dtype=object)[order], corrected, strict=True
    ):
        qvalues[tuple(location)] = value
    return qvalues


def fingerprint(case: dict[str, str], manifest: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            {"version": VERSION, "case": case, "manifest": manifest}, sort_keys=True
        ).encode()
    ).hexdigest()


def reuse_is_compatible(provenance: Path, identity: str) -> bool:
    """A present artifact is reusable only when its persisted identity matches."""
    return (
        provenance.exists()
        and json.loads(provenance.read_text()).get("fingerprint") == identity
    )


def case_by_id(case_id: str) -> dict[str, str]:
    match = next((item for item in cases() if item["id"] == case_id), None)
    if not match:
        raise ValueError(f"unknown case {case_id}; use list-cases")
    return match


def analysis_inputs(
    case: dict[str, str], output: Path, smoke: bool
) -> tuple[pd.Series, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    bundle_path = output / "analysis_bundle.npz"
    with np.load(bundle_path, allow_pickle=False) as bundle:
        timestamps = pd.Series(
            pd.to_datetime(bundle["timestamp_ns"].astype("datetime64[ns]"))
        )
        error_values = bundle[f"error_{case['model']}"]
        f107_values = bundle["f107"]
        geomagnetic_values = bundle[case["geomagnetic"]]
    manifest = {
        "analysis_bundle": sha256(bundle_path),
        "tau_steps": 2 if smoke else lag_steps(),
        "physical_lag_days": PHYSICAL_LAG_DAYS,
    }
    return timestamps, error_values, f107_values, geomagnetic_values, manifest


def pcmci_result(
    values: np.ndarray, columns: list[str], targets: list[str], tau: int
) -> dict[str, np.ndarray]:
    """Run the fixed PCMCI+ configuration used by both graph layouts."""
    from tigramite import data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI

    pcmci = PCMCI(
        pp.DataFrame(
            values,
            datatime=np.arange(len(values)),
            var_names=columns,
            missing_flag=MISSING_FLAG,
            remove_missing_upto_maxlag=False,
        ),
        cond_ind_test=ParCorr(significance="analytic"),
        verbosity=0,
    )
    return pcmci.run_pcmciplus(
        link_assumptions=link_assumptions(columns, targets, tau),
        tau_min=0,
        tau_max=tau,
        pc_alpha=ALPHA,
        fdr_method="none",
        contemp_collider_rule="majority",
        conflict_resolution=True,
    )


def driver_target_rows(
    result: dict[str, np.ndarray],
    columns: list[str],
    targets: list[str],
    drivers: list[str],
    tau: int,
) -> pd.DataFrame:
    q = driver_target_bh_qvalues(result["p_matrix"], columns, targets, drivers)
    rows = []
    for source in drivers:
        i = columns.index(source)
        for target in targets:
            j = columns.index(target)
            for lag in range(tau + 1):
                mark = str(result["graph"][i, j, lag])
                oriented = (lag == 0 and mark.endswith(">")) or (
                    lag > 0 and mark == "-->"
                )
                rows.append(
                    {
                        "source": source,
                        "target": target,
                        "lag_steps": lag,
                        "lag_hours": lag * STEP_HOURS,
                        "lag_days": lag * STEP_HOURS / 24,
                        "partial_r": float(result["val_matrix"][i, j, lag]),
                        "raw_p_value": float(result["p_matrix"][i, j, lag]),
                        "q_value": float(q[i, j, lag]),
                        "graph_mark": mark,
                        "detected": bool(oriented and q[i, j, lag] <= ALPHA),
                    }
                )
    return pd.DataFrame(rows)


def prepare(output: Path = OUTPUT) -> Path:
    samples = selected_frame(pd.read_parquet(SAMPLES))
    calendar = complete_3hour_calendar(samples)
    weather = pd.read_csv(SW_ALL)
    f107, f107_note = f107_spline(calendar.timestamp, weather)
    drivers = slot_drivers(pd.Series(calendar.timestamp.unique()), weather)
    # ``ap`` in the historical MSIS sample artifact is the pymsis model input,
    # not the native three-hour analysis driver requested here.
    calendar = calendar.drop(columns=["ap"], errors="ignore")
    calendar["f107"] = f107
    calendar = calendar.merge(
        drivers, on="timestamp", how="left", validate="many_to_one"
    )
    path = output / "prepared_msis_3hour.parquet"
    atomic_parquet(calendar, path)
    atomic_json(
        {
            "version": VERSION,
            "calendar_rows": len(calendar),
            "observed_timestamps": int(samples.timestamp.nunique()),
            "calendar_missing_timestamps": int(
                calendar["Density (kg/m^3)"].isna().sum() / len(ALTITUDES)
            ),
            "sentinel": MISSING_FLAG,
            "target_interpolation": "none",
            "f107": f107_note,
            "geomagnetic": "AP1..AP8 and KP1..KP8 map directly to UTC 00,03,...,21",
            "unsupported_cadence_mismatches": ["Mauna Loa CO2", "SABER"],
            "sources": {str(SAMPLES): sha256(SAMPLES), str(SW_ALL): sha256(SW_ALL)},
        },
        output / "prepare.provenance.json",
    )
    return path


def generate_jb(
    provider_command: list[str] | None = None,
    output: Path = OUTPUT,
    smoke: bool = False,
) -> Path:
    """Call a local provider wrapper on exact keys; it must preserve JB driver lags.

    The command receives request and output paths.  It must write positive
    ``jb2006_density`` and ``jb2008_density`` columns keyed by timestamp,
    altitude_km, and ``Longitude (deg)``.  Provider sources/indices remain
    under ignored ``data/external``; this tracked adapter never alters proxies.
    """
    provider_command = provider_command or [
        sys.executable,
        "-m",
        "scripts.run_hasdm_jb_exact",
    ]
    frame = selected_frame(pd.read_parquet(SAMPLES))
    # The source samples were produced by this existing nearest-grid selector;
    # recover its frozen nearest latitude rather than inventing a coordinate.
    from scripts.hasdm_msis_model_error_analysis import nearest_hasdm_latitude

    frame["Latitude (deg)"] = nearest_hasdm_latitude()
    if smoke:
        frame = frame[frame.timestamp.isin(sorted(frame.timestamp.unique())[:2])]
    request = output / "jb_exact_request.parquet"
    atomic_parquet(
        frame[
            [
                "timestamp",
                "altitude_km",
                "Latitude (deg)",
                "Longitude (deg)",
                "Density (kg/m^3)",
            ]
        ],
        request,
    )
    produced = output / "jb_exact_density.parquet"
    subprocess.run([*provider_command, str(request), str(produced)], check=True)
    provider_provenance = produced.with_suffix(produced.suffix + ".provider.json")
    if not provider_provenance.exists():
        raise FileNotFoundError("provider did not write required provenance")
    provider_record = json.loads(provider_provenance.read_text())
    if provider_record.get("rows") != len(frame) or not provider_record.get(
        "source_index_hashes"
    ):
        raise ValueError("provider provenance lacks exact-row or source/index identity")
    jb = pd.read_parquet(produced)
    keys = ["timestamp", "altitude_km", "Latitude (deg)", "Longitude (deg)"]
    merged = frame.merge(jb, on=keys, how="left", validate="one_to_one")
    for model in ("jb2006", "jb2008"):
        density = merged[f"{model}_density"]
        if (
            density.isna().any()
            or not np.isfinite(density).all()
            or not (density > 0).all()
        ):
            raise ValueError(
                f"{model} did not return finite positive densities for every exact key"
            )
        merged[f"error_{model}"] = np.log(density / merged["Density (kg/m^3)"])
    result = output / "jb_exact_errors.parquet"
    atomic_parquet(merged[[*keys, "error_jb2006", "error_jb2008"]], result)
    atomic_json(
        {
            "version": VERSION,
            "request_rows": len(frame),
            "matched_rows": len(merged),
            "excluded_rows": 0,
            "matching": "one-to-one timestamp/altitude/selected-latitude/selected-longitude",
            "driver_lags": {
                "jb2006": "F10/S10 1d; M10 5d; Ap 6.7h",
                "jb2008": "F10/S10 1d; M10 2d; Y10 5d; DTC at time",
            },
            "provider_command": provider_command,
            "input_hash": sha256(request),
            "output_hash": sha256(result),
            "provider_provenance": provider_record,
            "python": sys.version,
        },
        output / "jb_exact_errors.provenance.json",
    )
    return result


def build_analysis_bundle(output: Path = OUTPUT) -> Path:
    """Pack all remote analysis inputs into a portable NumPy artifact."""
    prepared_path = output / "prepared_msis_3hour.parquet"
    jb_path = output / "jb_exact_errors.parquet"
    prepared = pd.read_parquet(prepared_path)
    jb = pd.read_parquet(jb_path)
    prepared = prepared.merge(
        jb,
        on=["timestamp", "altitude_km", "Longitude (deg)"],
        how="left",
        validate="one_to_one",
    )
    timestamps = pd.DatetimeIndex(sorted(prepared.timestamp.unique()))
    errors = {
        "nrlmsise_00": "error_0",
        "nrlmsis_2p0": "error_2p0",
        "nrlmsis_2p1": "error_2p1",
        "jb2006": "error_jb2006",
        "jb2008": "error_jb2008",
    }
    arrays: dict[str, np.ndarray] = {
        "timestamp_ns": timestamps.to_numpy(dtype="datetime64[ns]").astype(np.int64),
        "altitudes_km": np.asarray(ALTITUDES, dtype=np.int16),
    }
    for model, column in errors.items():
        arrays[f"error_{model}"] = (
            prepared.pivot_table(
                index="timestamp",
                columns="altitude_km",
                values=column,
                aggfunc="first",
            )
            .reindex(index=timestamps, columns=ALTITUDES)
            .to_numpy(dtype=np.float64)
        )
    drivers = prepared.drop_duplicates("timestamp").set_index("timestamp")
    for driver in ("f107", "ap", "kp"):
        arrays[driver] = drivers[driver].reindex(timestamps).to_numpy(dtype=np.float64)
    path = output / "analysis_bundle.npz"
    atomic_npz(path, **arrays)
    atomic_json(
        {
            "version": VERSION,
            "timestamps": len(timestamps),
            "altitudes_km": list(ALTITUDES),
            "models": list(MODELS),
            "drivers": ["f107", "ap", "kp"],
            "sources": {
                str(prepared_path): sha256(prepared_path),
                str(jb_path): sha256(jb_path),
            },
            "bundle_sha256": sha256(path),
        },
        output / "analysis_bundle.provenance.json",
    )
    return path


def run_case(
    case: dict[str, str],
    output: Path = OUTPUT,
    smoke: bool = False,
    recompute: bool = False,
    results_output: Path | None = None,
) -> None:
    timestamps, error_values, f107_values, geomagnetic_values, manifest = (
        analysis_inputs(case, output, smoke)
    )
    target_names = [f"error_{alt}km" for alt in ALTITUDES]
    geo = case["geomagnetic"]
    identity = fingerprint(case, manifest)
    directory = (results_output or output) / "cases" / case["id"]
    provenance = directory / "provenance.json"
    if provenance.exists() and not recompute:
        if reuse_is_compatible(provenance, identity):
            return
        raise FileExistsError(
            f"refusing incompatible reuse of {directory}; use --recompute"
        )
    values = np.column_stack(
        [
            preprocess_3hour(error_values[:, index], timestamps)
            for index in range(len(ALTITUDES))
        ]
        + [
            preprocess_3hour(f107_values, timestamps),
            preprocess_3hour(geomagnetic_values, timestamps),
        ]
    )
    values[~np.isfinite(values)] = MISSING_FLAG
    columns = [*target_names, "f107", geo]
    tau = manifest["tau_steps"]
    result = pcmci_result(values, columns, target_names, tau)
    links = driver_target_rows(result, columns, target_names, ["f107", geo], tau)
    atomic_csv(links, directory / "driver_target_tests.csv")
    atomic_csv(links[links.detected], directory / "retained_links.csv")
    finite_counts = {
        column: int(count)
        for column, count in zip(
            columns, (values != MISSING_FLAG).sum(axis=0), strict=True
        )
    }
    atomic_json(
        {
            "fingerprint": identity,
            "case": case,
            "manifest": manifest,
            "method": "exploratory PCMCI+ / ParCorr analytic; algorithm-selected p-values are not confirmatory",
            "fdr_family": "one BH correction per graph over predeclared f107/one-geomagnetic driver to 27 targets at lag 0..tau",
            "fdr_test_count": declared_fdr_tests(tau),
            "link_scope": "driver histories; driver→target including contemporaneous; target self-history; no target peer/cross-model links",
            "finite_counts_after_preprocessing": finite_counts,
            "non_detections_explicit": int((~links.detected).sum()),
        },
        provenance,
    )


def shard_directory(output: Path, case: dict[str, str], altitude: int) -> Path:
    return output / SHARDED_DIRECTORY / case["id"] / f"{altitude}km"


def shard_settings(case: dict[str, str], altitude: int) -> dict[str, object]:
    target = f"error_{altitude}km"
    return {
        "columns": [target, "f107", case["geomagnetic"]],
        "targets": [target],
        "method": "PCMCI+ / ParCorr analytic",
        "tau_min": 0,
        "pc_alpha": ALPHA,
        "fdr_method": "none",
        "contemp_collider_rule": "majority",
        "conflict_resolution": True,
        "missing_flag": MISSING_FLAG,
        "remove_missing_upto_maxlag": False,
        "link_scope": "driver histories; driver→target including contemporaneous; target self-history",
    }


def run_shard(
    case: dict[str, str],
    altitude: int,
    output: Path = OUTPUT,
    smoke: bool = False,
    recompute: bool = False,
) -> None:
    """Run one single-threaded, one-altitude PCMCI+ graph in its own output tree."""
    if altitude not in ALTITUDES:
        raise ValueError(f"unknown altitude {altitude}; expected one of {ALTITUDES}")
    timestamps, errors, f107, geo_values, manifest = analysis_inputs(
        case, output, smoke
    )
    identity = fingerprint(case, manifest)
    shard_identity = fingerprint(case, {**manifest, "altitude_km": altitude})
    directory = shard_directory(output, case, altitude)
    provenance = directory / "provenance.json"
    if provenance.exists() and not recompute:
        if reuse_is_compatible(provenance, shard_identity):
            return
        raise FileExistsError(
            f"refusing incompatible reuse of {directory}; use --recompute"
        )
    index = ALTITUDES.index(altitude)
    target = f"error_{altitude}km"
    columns = [target, "f107", case["geomagnetic"]]
    values = np.column_stack(
        [
            preprocess_3hour(errors[:, index], timestamps),
            preprocess_3hour(f107, timestamps),
            preprocess_3hour(geo_values, timestamps),
        ]
    )
    values[~np.isfinite(values)] = MISSING_FLAG
    tau = manifest["tau_steps"]
    result = pcmci_result(values, columns, [target], tau)
    # q-values here are deliberately provisional; merge performs the one valid global BH.
    raw = driver_target_rows(
        result, columns, [target], ["f107", case["geomagnetic"]], tau
    )
    atomic_csv(raw, directory / "driver_target_raw.csv")
    finite_counts = {
        column: int(count)
        for column, count in zip(
            columns, (values != MISSING_FLAG).sum(axis=0), strict=True
        )
    }
    atomic_json(
        {
            "fingerprint": shard_identity,
            "case_fingerprint": identity,
            "case": case,
            "altitude_km": altitude,
            "manifest": manifest,
            "settings": shard_settings(case, altitude),
            "finite_counts_after_preprocessing": finite_counts,
        },
        provenance,
    )


def merge_shards(
    case: dict[str, str], output: Path = OUTPUT, smoke: bool = False
) -> Path:
    """Require all altitude shards, then apply one BH correction to their union."""
    _, _, _, _, manifest = analysis_inputs(case, output, smoke)
    identity = fingerprint(case, manifest)
    tau = manifest["tau_steps"]
    shards = []
    for altitude in ALTITUDES:
        directory = shard_directory(output, case, altitude)
        provenance_path = directory / "provenance.json"
        raw_path = directory / "driver_target_raw.csv"
        if not provenance_path.exists() or not raw_path.exists():
            raise FileNotFoundError(
                f"missing required shard for {case['id']} at {altitude}km"
            )
        provenance = json.loads(provenance_path.read_text())
        expected_shard = fingerprint(case, {**manifest, "altitude_km": altitude})
        if (
            provenance.get("fingerprint") != expected_shard
            or provenance.get("case_fingerprint") != identity
            or provenance.get("case") != case
            or provenance.get("altitude_km") != altitude
            or provenance.get("manifest") != manifest
            or provenance.get("settings") != shard_settings(case, altitude)
        ):
            raise ValueError(
                f"incompatible shard provenance for {case['id']} at {altitude}km"
            )
        shard = pd.read_csv(raw_path)
        expected = {
            (source, f"error_{altitude}km", lag)
            for source in ("f107", case["geomagnetic"])
            for lag in range(tau + 1)
        }
        observed = set(zip(shard.source, shard.target, shard.lag_steps, strict=True))
        if observed != expected or len(shard) != len(expected):
            raise ValueError(
                f"incomplete or duplicate driver-target tests in {directory}"
            )
        shards.append(shard)
    links = pd.concat(shards, ignore_index=True)
    if len(links) != declared_fdr_tests(tau):
        raise ValueError("shards do not contain the complete predeclared BH family")
    columns = [
        *[f"error_{altitude}km" for altitude in ALTITUDES],
        "f107",
        case["geomagnetic"],
    ]
    raw = np.full((len(columns), len(columns), tau + 1), np.nan)
    for row in links.itertuples(index=False):
        raw[columns.index(row.source), columns.index(row.target), row.lag_steps] = (
            row.raw_p_value
        )
    q = driver_target_bh_qvalues(raw, columns, columns[: len(ALTITUDES)], columns[-2:])
    links["q_value"] = [
        q[columns.index(row.source), columns.index(row.target), row.lag_steps]
        for row in links.itertuples(index=False)
    ]
    links["detected"] = [
        (
            (row.lag_steps == 0 and row.graph_mark.endswith(">"))
            or (row.lag_steps > 0 and row.graph_mark == "-->")
        )
        and row.q_value <= ALPHA
        for row in links.itertuples(index=False)
    ]
    directory = output / SHARDED_DIRECTORY / case["id"] / "merged"
    atomic_csv(links, directory / "driver_target_tests.csv")
    atomic_csv(links[links.detected], directory / "retained_links.csv")
    atomic_json(
        {
            "fingerprint": identity,
            "case": case,
            "manifest": manifest,
            "fdr_family": "one BH correction over 2 drivers × 27 targets × lag 0..tau",
            "fdr_test_count": declared_fdr_tests(tau),
            "provenance_label": "sharded-equivalent candidate; promotion requires verifier success",
            "shards": list(ALTITUDES),
        },
        directory / "provenance.json",
    )
    return directory / "driver_target_tests.csv"


def verify_equivalence(unsharded: Path, sharded: Path) -> dict[str, float | int]:
    """Compare all declared driver→target tests; raise rather than imply equivalence."""
    left, right = pd.read_csv(unsharded), pd.read_csv(sharded)
    keys = ["source", "target", "lag_steps"]
    for frame in (left, right):
        if frame.duplicated(keys).any():
            raise ValueError("equivalence input has duplicate source/target/lag tests")
    left, right = (
        left.sort_values(keys).reset_index(drop=True),
        right.sort_values(keys).reset_index(drop=True),
    )
    if set(map(tuple, left[keys].to_numpy())) != set(
        map(tuple, right[keys].to_numpy())
    ):
        raise ValueError("equivalence requires exact source/target/lag test sets")
    if not left[keys].equals(right[keys]):
        raise ValueError("equivalence requires identical sorted test keys")
    maximums: dict[str, float | int] = {"tests": len(left)}
    for column in ("raw_p_value", "partial_r", "q_value"):
        difference = np.abs(left[column].to_numpy() - right[column].to_numpy())
        maximums[f"max_{column}_difference"] = float(np.nanmax(difference))
        if not np.allclose(
            left[column],
            right[column],
            rtol=EQUIVALENCE_RTOL,
            atol=EQUIVALENCE_ATOL,
            equal_nan=True,
        ):
            raise ValueError(
                f"equivalence mismatch in {column}: max difference {maximums[f'max_{column}_difference']}"
            )
    for column in ("graph_mark", "detected"):
        if not left[column].equals(right[column]):
            raise ValueError(f"equivalence mismatch in {column}")
    return maximums


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list-cases")
    sub.add_parser("prepare")
    sub.add_parser("bundle")
    jb = sub.add_parser("generate-jb")
    jb.add_argument("--provider-command", nargs="+")
    jb.add_argument("--smoke", action="store_true")
    run = sub.add_parser("run")
    run.add_argument("case")
    run.add_argument("--smoke", action="store_true")
    run.add_argument("--recompute", action="store_true")
    run.add_argument("--results-output", type=Path)
    shard = sub.add_parser("run-shard")
    shard.add_argument("case")
    shard.add_argument("altitude", type=int)
    shard.add_argument("--smoke", action="store_true")
    shard.add_argument("--recompute", action="store_true")
    merge = sub.add_parser("merge-shards")
    merge.add_argument("case")
    merge.add_argument("--smoke", action="store_true")
    sub.add_parser("list-spacehopper-shards")
    verify = sub.add_parser("verify-equivalence")
    verify.add_argument("unsharded", type=Path)
    verify.add_argument("sharded", type=Path)
    sub.add_parser("summarize")
    args = parser.parse_args()
    if args.command == "list-cases":
        print(json.dumps(cases(), indent=2))
        return
    if args.command == "prepare":
        print(prepare())
        return
    if args.command == "bundle":
        print(build_analysis_bundle())
        return
    if args.command == "generate-jb":
        print(generate_jb(args.provider_command, smoke=args.smoke))
        return
    if args.command == "run":
        run_case(
            case_by_id(args.case),
            smoke=args.smoke,
            recompute=args.recompute,
            results_output=args.results_output,
        )
        return
    if args.command == "run-shard":
        run_shard(
            case_by_id(args.case),
            args.altitude,
            smoke=args.smoke,
            recompute=args.recompute,
        )
        return
    if args.command == "merge-shards":
        print(merge_shards(case_by_id(args.case), smoke=args.smoke))
        return
    if args.command == "list-spacehopper-shards":
        for case_id in SPACEHOPPER_CASE_IDS:
            for altitude in ALTITUDES:
                print(f"run-shard {case_id} {altitude}")
        return
    if args.command == "verify-equivalence":
        print(json.dumps(verify_equivalence(args.unsharded, args.sharded), indent=2))
        return
    links = [
        pd.read_csv(path)
        for path in sorted((OUTPUT / "cases").glob("*/retained_links.csv"))
    ]
    atomic_csv(
        pd.concat(links, ignore_index=True) if links else pd.DataFrame(),
        OUTPUT / "retained_links_summary.csv",
    )


if __name__ == "__main__":
    main()
