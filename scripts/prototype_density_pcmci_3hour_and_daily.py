"""Portable exploratory direct-density PCMCI+ / analytic-ParCorr runner.

This is deliberately separate from the model-error prototype.  It retains the
native cadence of each density product and makes the intentionally restricted
graph explicit: absent target-to-target links are a structural assumption, not
evidence about vertical coupling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

VERSION = "density-pcmci-3hour-daily-v2-saber"
MISSING_FLAG = -999999.0  # Match the established model-error PCMCI runner.
ALPHA = 0.05
GLOBAL_LAG_DAYS = 183
HASDM_LAG_DAYS = 61
HASDM_ALTITUDES = tuple(range(175, 826, 25))
GLOBAL_ALTITUDES = (250, 275, 325, 375, 400, 425, 475, 525, 550, 575)
HASDM_START, HASDM_END = "2000-01-01", "2025-07-20 00:00"
GLOBAL_START, GLOBAL_END = "1967-01-01", "2019-12-01"
SAMPLES = Path(
    "outputs/figures/results/hasdm_msis_model_errors/data/hasdm_msis_errors_nearest_timestamp_grid_samples.parquet"
)
GLOBAL = Path(
    "data/decoded/orbit_derived_global_mean/orbit-density-ds03-density-values.parquet"
)
WEATHER = Path("data/original/space_weather/SW-All.csv")
SABER = Path("data/decoded/saber/saber_hasdm_maunaloa_3hour.parquet")
OUTPUT = Path("outputs/prototypes/density_pcmci_3hour_and_daily")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
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


def cases() -> list[dict[str, str]]:
    return [
        *[
            {
                "id": f"hasdm_{family}-{geo}",
                "product": "hasdm",
                "family": family,
                "geomagnetic": geo,
            }
            for family in ("all", "selected")
            for geo in ("ap", "kp")
        ],
        *[
            {
                "id": f"global_mean-{geo}",
                "product": "global_mean",
                "family": "all",
                "geomagnetic": geo,
            }
            for geo in ("ap", "kp")
        ],
    ]


def case_by_id(case_id: str) -> dict[str, str]:
    case = next((item for item in cases() if item["id"] == case_id), None)
    if case is None:
        raise ValueError(f"unknown case {case_id}; use list-cases")
    return case


def lag_steps(product: str) -> int:
    return HASDM_LAG_DAYS * 8 if product == "hasdm" else GLOBAL_LAG_DAYS


def expected_fdr_tests(case: dict[str, str]) -> int:
    return driver_count(case) * len(altitudes(case)) * (lag_steps(case["product"]) + 1)


def altitudes(case: dict[str, str]) -> tuple[int, ...]:
    if case["product"] == "global_mean":
        return GLOBAL_ALTITUDES
    return HASDM_ALTITUDES if case["family"] == "all" else (175, 500, 825)


def target_indices(case: dict[str, str]) -> tuple[int, ...]:
    """Map a case's target altitude labels onto the immutable product matrix."""
    product_altitudes = (
        HASDM_ALTITUDES if case["product"] == "hasdm" else GLOBAL_ALTITUDES
    )
    return tuple(product_altitudes.index(altitude) for altitude in altitudes(case))


def saber_columns() -> list[str]:
    return [
        f"{prefix}_{altitude}km_w_m3"
        for prefix in (
            "saber_co2cool",
            "saber_nocool",
            "saber_o2_1delta_ver",
            "saber_oh_16_ver",
            "saber_oh_20_ver",
        )
        for altitude in (100, 119, 139)
    ]


def driver_count(case: dict[str, str]) -> int:
    return 17 if case["product"] == "hasdm" else 2


def complete_hasdm_calendar() -> pd.DatetimeIndex:
    return pd.date_range(HASDM_START, HASDM_END, freq="3h")


def f107_spline(times: pd.DatetimeIndex, weather: pd.DataFrame) -> np.ndarray:
    """Interpolate raw daily F10.7 only inside its UTC-midnight knot domain."""
    knots = weather[["DATE", "F10.7_OBS"]].dropna().copy()
    knots["DATE"] = pd.to_datetime(knots["DATE"])
    knots = knots.drop_duplicates("DATE").sort_values("DATE")
    origin = knots["DATE"].iloc[0]
    x = (knots["DATE"] - origin).dt.total_seconds().to_numpy() / 3600
    query = (times - origin).total_seconds().to_numpy() / 3600
    values = np.full(len(times), np.nan)
    inside = (query >= x[0]) & (query <= x[-1])
    values[inside] = CubicSpline(x, knots["F10.7_OBS"].to_numpy(), extrapolate=False)(
        query[inside]
    )
    for knot, value in zip(x, knots["F10.7_OBS"], strict=True):
        values[query == knot] = value
    return values


def native_slots(
    times: pd.DatetimeIndex, weather: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    indexed = weather.copy()
    indexed["DATE"] = pd.to_datetime(indexed["DATE"])
    indexed = indexed.drop_duplicates("DATE").set_index("DATE")
    dates, slots = times.normalize(), times.hour // 3 + 1
    ap = np.full(len(times), np.nan)
    kp = np.full(len(times), np.nan)
    for index, (day, slot) in enumerate(zip(dates, slots, strict=True)):
        if day in indexed.index:
            ap[index] = indexed.at[day, f"AP{slot}"]
            kp[index] = indexed.at[day, f"KP{slot}"]
    return ap, kp


def prepare_hasdm(
    weather: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    if not SABER.is_file() or not SABER.with_suffix(".provenance.json").is_file():
        raise FileNotFoundError(
            "SABER 3-hour parquet and provenance are required for HASDM preparation."
        )
    samples = pd.read_parquet(SAMPLES)
    samples["timestamp"] = pd.to_datetime(samples["timestamp"])
    samples["altitude_km"] = samples["Altitude (m)"].astype(int) // 1000
    if samples.duplicated(["timestamp", "altitude_km"]).any() or set(
        samples.altitude_km
    ) != set(HASDM_ALTITUDES):
        raise ValueError(
            "HASDM source must have one row per timestamp/altitude at all fixed altitudes"
        )
    calendar = complete_hasdm_calendar()
    density = samples.pivot(
        index="timestamp", columns="altitude_km", values="Density (kg/m^3)"
    ).reindex(calendar, columns=HASDM_ALTITUDES)
    values = density.to_numpy(float, copy=True)
    values[values <= 0] = np.nan
    target = np.log10(values)
    missing_timestamps = int(np.all(~np.isfinite(target), axis=1).sum())
    if missing_timestamps != 256:
        raise ValueError(
            f"expected 256 absent HASDM timestamps, found {missing_timestamps}"
        )
    ap, kp = native_slots(calendar, weather)
    saber_source = pd.read_parquet(SABER)
    saber_source["timestamp"] = pd.to_datetime(saber_source["timestamp"])
    required_saber = saber_columns()
    if (
        saber_source["timestamp"].duplicated().any()
        or not pd.DatetimeIndex(saber_source["timestamp"]).equals(calendar)
        or any(column not in saber_source for column in required_saber)
    ):
        raise ValueError(
            "SABER source is incompatible with the fixed HASDM 3-hour calendar/schema."
        )
    saber_note = json.loads(SABER.with_suffix(".provenance.json").read_text())
    if (
        saber_note.get("missing_policy")
        != "calendar slots and empty SABER bins are null; no filling or interpolation"
    ):
        raise ValueError("SABER provenance has incompatible missing-value policy.")
    arrays = {
        "hasdm_time_ns": calendar.to_numpy(dtype="datetime64[ns]").astype(np.int64),
        "hasdm_targets": target,
        "hasdm_f107": f107_spline(calendar, weather),
        "hasdm_ap": ap,
        "hasdm_kp": kp,
        "hasdm_saber": saber_source[required_saber].to_numpy(float),
    }
    return arrays, {
        "shape": list(target.shape),
        "start": str(calendar[0]),
        "end": str(calendar[-1]),
        "cadence": "3-hour",
        "finite_target_values": int(np.isfinite(target).sum()),
        "missing_target_values": int((~np.isfinite(target)).sum()),
        "driver_finite_counts": {
            "f107": int(np.isfinite(arrays["hasdm_f107"]).sum()),
            "ap": int(np.isfinite(ap).sum()),
            "kp": int(np.isfinite(kp).sum()),
            "saber": {
                column: int(np.isfinite(saber_source[column]).sum())
                for column in required_saber
            },
        },
        "missing_timestamps_all_targets": missing_timestamps,
        "target_policy": "direct log10 density; no interpolation",
        "forcing": "raw F10.7 cubic spline at UTC knots without extrapolation; AP1..8/KP1..8 native UTC slots; sparse SABER cooling and emission-proxy drivers are unfilled.",
    }


def prepare_global(
    weather: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    columns = [f"log10rho_{altitude}" for altitude in GLOBAL_ALTITUDES]
    source = pd.read_parquet(GLOBAL).loc[:, ["date", *columns]].copy()
    dates = pd.DatetimeIndex(pd.to_datetime(source["date"]))
    target = source[columns].to_numpy(float)
    expected = pd.date_range(GLOBAL_START, GLOBAL_END, freq="D")
    if (
        len(source) != 19328
        or not dates.equals(expected)
        or not np.isfinite(target).all()
    ):
        raise ValueError(
            "global-mean source must be the complete finite 19,328-row daily contract"
        )
    daily_weather = weather.copy()
    daily_weather["DATE"] = pd.to_datetime(daily_weather["DATE"])
    daily_weather = (
        daily_weather.drop_duplicates("DATE").set_index("DATE").reindex(dates)
    )
    arrays = {
        "global_time_ns": dates.to_numpy(dtype="datetime64[ns]").astype(np.int64),
        "global_targets": target,
        "global_f107": daily_weather["F10.7_OBS"].to_numpy(float),
        "global_ap": daily_weather["AP_AVG"].to_numpy(float),
        "global_kp": daily_weather["KP_SUM"].to_numpy(float),
    }
    return arrays, {
        "shape": list(target.shape),
        "start": str(dates[0].date()),
        "end": str(dates[-1].date()),
        "cadence": "daily (not upsampled to 3-hour)",
        "finite_targets": int(np.isfinite(target).sum()),
        "missing_target_values": int((~np.isfinite(target)).sum()),
        "driver_finite_counts": {
            "f107": int(np.isfinite(arrays["global_f107"]).sum()),
            "ap": int(np.isfinite(arrays["global_ap"]).sum()),
            "kp": int(np.isfinite(arrays["global_kp"]).sum()),
        },
        "target_policy": "direct source log10rho_* channels; no interpolation",
        "forcing": "raw daily F10.7 knots; AP_AVG/KP_SUM daily aggregates",
    }


def prepare(output: Path = OUTPUT) -> Path:
    weather = pd.read_csv(WEATHER)
    hasdm, hasdm_note = prepare_hasdm(weather)
    global_mean, global_note = prepare_global(weather)
    bundle = output / "analysis_bundle.npz"
    atomic_npz(bundle, **hasdm, **global_mean)
    atomic_json(
        {
            "version": VERSION,
            "bundle_sha256": sha256(bundle),
            "sources": {
                str(SAMPLES): sha256(SAMPLES),
                str(GLOBAL): sha256(GLOBAL),
                str(WEATHER): sha256(WEATHER),
                str(SABER): sha256(SABER),
                str(SABER.with_suffix(".provenance.json")): sha256(
                    SABER.with_suffix(".provenance.json")
                ),
            },
            "products": {"hasdm": hasdm_note, "global_mean": global_note},
            "cadence_difference": "HASDM remains 3-hour with 15 sparse SABER drivers; global mean remains daily without SABER and is never upsampled.",
            "missing_sentinel": MISSING_FLAG,
            "portable_runner_dependencies": ["numpy", "pandas", "scipy", "tigramite"],
        },
        output / "analysis_bundle.provenance.json",
    )
    return bundle


def preprocess(
    values: np.ndarray, times: pd.DatetimeIndex, steps_per_day: int
) -> np.ndarray:
    """Calendar month/day(/slot) anomalies, centered three-year mean, finite z-score."""
    series = pd.Series(np.asarray(values, dtype=float))
    key = (
        [times.month, times.day]
        if steps_per_day == 1
        else [times.month, times.day, times.hour // 3]
    )
    seasonal = series - series.groupby(key).transform("mean")
    detrended = (
        seasonal
        - seasonal.rolling(3 * 365 * steps_per_day, center=True, min_periods=1).mean()
    )
    scale = detrended.std(ddof=0, skipna=True)
    return (
        detrended / scale if np.isfinite(scale) and scale > 0 else detrended * np.nan
    ).to_numpy()


def link_assumptions(
    columns: list[str], targets: list[str], tau: int
) -> dict[int, dict[tuple[int, int], str]]:
    """Structural absence assumption: no target-to-target links, including vertical links."""
    result = {target_i: {} for target_i in range(len(columns))}
    target_set = set(targets)
    for target_i, target in enumerate(columns):
        for cause_i, cause in enumerate(columns):
            driver_pair = target not in target_set and cause not in target_set
            allowed_history = (
                driver_pair
                or (target in target_set and cause not in target_set)
                or cause == target
            )
            if allowed_history:
                for lag in range(1, tau + 1):
                    result[target_i][(cause_i, -lag)] = "-?>"
            if target in target_set and cause not in target_set:
                result[target_i][(cause_i, 0)] = "-?>"
            elif driver_pair and cause_i < target_i:
                result[target_i][(cause_i, 0)] = "o?o"
    return result


def bh_qvalues(pvalues: np.ndarray) -> np.ndarray:
    if not np.isfinite(pvalues).all():
        raise ValueError("non-finite raw p-value in declared BH family")
    order = np.argsort(pvalues)
    ranked = pvalues[order] * len(pvalues) / np.arange(1, len(pvalues) + 1)
    corrected = np.minimum.accumulate(ranked[::-1])[::-1].clip(0, 1)
    output = np.empty_like(pvalues)
    output[order] = corrected
    return output


def fingerprint(case: dict[str, str], bundle_hash: str) -> str:
    identity = {
        "version": VERSION,
        "case": case,
        "bundle_sha256": bundle_hash,
        "cadence": "3-hour" if case["product"] == "hasdm" else "daily",
        "tau_max": lag_steps(case["product"]),
        "settings": {
            "method": "PCMCI+ / ParCorr analytic",
            "pc_alpha": ALPHA,
            "fdr_method": "none",
            "collider_rule": "majority",
            "conflict_resolution": True,
            "missing_flag": MISSING_FLAG,
        },
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def reusable_production_artifact(
    directory: Path, case: dict[str, str], bundle_hash: str, identity: str
) -> bool:
    """Return whether a case directory is an intact production result."""
    provenance = directory / "provenance.json"
    tests, retained = (
        directory / "driver_target_tests.csv",
        directory / "retained_links.csv",
    )
    if not all(path.is_file() for path in (provenance, tests, retained)):
        return False
    try:
        record = json.loads(provenance.read_text())
        result_files = record["result_files"]
        count = sum(1 for _ in tests.open()) - 1
        hashes = {path.name: sha256(path) for path in (tests, retained)}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False
    expected = expected_fdr_tests(case)
    return bool(
        record.get("production")
        and record.get("case", {}).get("id") == case["id"]
        and record.get("bundle_sha256") == bundle_hash
        and record.get("fingerprint") == identity
        and record.get("fdr_family", {}).get("test_count") == expected
        and count == expected
        and result_files.get(tests.name) == hashes[tests.name]
        and result_files.get(retained.name) == hashes[retained.name]
    )


def load_case_bundle(
    case: dict[str, str], output: Path
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    bundle = output / "analysis_bundle.npz"
    with np.load(bundle, allow_pickle=False) as saved:
        prefix = "hasdm" if case["product"] == "hasdm" else "global"
        times = pd.DatetimeIndex(saved[f"{prefix}_time_ns"].astype("datetime64[ns]"))
        return (
            times,
            saved[f"{prefix}_targets"],
            saved[f"{prefix}_f107"],
            saved[f"{prefix}_{case['geomagnetic']}"],
            saved["hasdm_saber"]
            if case["product"] == "hasdm"
            else np.empty((len(times), 0)),
            sha256(bundle),
        )


def run_pcmci(
    values: np.ndarray, columns: list[str], targets: list[str], tau: int
) -> dict[str, np.ndarray]:
    from tigramite import data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI

    frame = pp.DataFrame(
        values,
        datatime=np.arange(len(values)),
        var_names=columns,
        missing_flag=MISSING_FLAG,
        remove_missing_upto_maxlag=False,
    )
    return PCMCI(
        frame, cond_ind_test=ParCorr(significance="analytic"), verbosity=0
    ).run_pcmciplus(
        link_assumptions=link_assumptions(columns, targets, tau),
        tau_min=0,
        tau_max=tau,
        pc_alpha=ALPHA,
        fdr_method="none",
        contemp_collider_rule="majority",
        conflict_resolution=True,
    )


def result_rows(
    result: dict[str, np.ndarray],
    columns: list[str],
    targets: list[str],
    tau: int,
    step_hours: int,
) -> pd.DataFrame:
    drivers = [column for column in columns if column not in targets]
    locations = [
        (columns.index(driver), columns.index(target), lag)
        for driver in drivers
        for target in targets
        for lag in range(tau + 1)
    ]
    qvalues = bh_qvalues(
        np.array([result["p_matrix"][item] for item in locations], float)
    )
    rows = []
    for location, qvalue in zip(locations, qvalues, strict=True):
        source_i, target_i, lag = location
        mark = str(result["graph"][location])
        oriented = mark.endswith(">") if lag == 0 else mark == "-->"
        rows.append(
            {
                "source": columns[source_i],
                "target": columns[target_i],
                "lag_steps": lag,
                "lag_hours": lag * step_hours,
                "lag_days": lag * step_hours / 24,
                "partial_r": float(result["val_matrix"][location]),
                "raw_p_value": float(result["p_matrix"][location]),
                "q_value": float(qvalue),
                "graph_mark": mark,
                "detected": bool(oriented and qvalue <= ALPHA),
            }
        )
    return pd.DataFrame(rows)


def run(
    case: dict[str, str],
    output: Path = OUTPUT,
    host_label: str | None = None,
    recompute: bool = False,
) -> Path:
    times, targets_raw, f107_raw, geo_raw, saber_raw, bundle_hash = load_case_bundle(
        case, output
    )
    identity = fingerprint(case, bundle_hash)
    directory, provenance = (
        output / "cases" / case["id"],
        output / "cases" / case["id"] / "provenance.json",
    )
    if provenance.exists() and not recompute:
        if reusable_production_artifact(directory, case, bundle_hash, identity):
            return directory
        raise FileExistsError(
            "refusing incompatible or incomplete reuse; use --recompute"
        )
    # This is deliberately not a completion marker.  A machine may disappear
    # while Tigramite is running; orchestration must restart that case from the
    # immutable bundle and only provenance.json authorizes reuse of its output.
    metadata = directory / "execution_metadata.json"
    started_epoch = time.time()
    atomic_json(
        {
            "fingerprint": identity,
            "case": case,
            "settings": {
                "tau_max": lag_steps(case["product"]),
                "pc_alpha": ALPHA,
                "method": "PCMCI+ / ParCorr analytic",
                "fdr_method": "none",
                "missing_flag": MISSING_FLAG,
            },
            "host_label": host_label,
            "started_epoch": started_epoch,
            "status": "running",
        },
        metadata,
    )
    product = case["product"]
    altitude_values = altitudes(case)
    altitude_indices = target_indices(case)
    targets = [f"log10rho_{altitude}" for altitude in altitude_values]
    steps_per_day = 8 if product == "hasdm" else 1
    values = np.column_stack(
        [
            *(
                preprocess(targets_raw[:, index], times, steps_per_day)
                for index in altitude_indices
            ),
            preprocess(f107_raw, times, steps_per_day),
            preprocess(geo_raw, times, steps_per_day),
            *(
                preprocess(saber_raw[:, index], times, steps_per_day)
                for index in range(saber_raw.shape[1])
            ),
        ]
    )
    values[~np.isfinite(values)] = MISSING_FLAG
    columns = [
        *targets,
        "f107",
        case["geomagnetic"],
        *(saber_columns() if product == "hasdm" else []),
    ]
    tau = lag_steps(product)
    started = time.monotonic()
    try:
        result = run_pcmci(values, columns, targets, tau)
    except Exception as error:
        atomic_json(
            {
                "fingerprint": identity,
                "case": case,
                "host_label": host_label,
                "started_epoch": started_epoch,
                "finished_epoch": time.time(),
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            },
            metadata,
        )
        raise
    links = result_rows(result, columns, targets, tau, 3 if product == "hasdm" else 24)
    tests_path, retained_path = (
        directory / "driver_target_tests.csv",
        directory / "retained_links.csv",
    )
    atomic_csv(links, tests_path)
    atomic_csv(links.loc[links.detected], retained_path)
    atomic_json(
        {
            "fingerprint": identity,
            "version": VERSION,
            "production": True,
            "case": case,
            "bundle_sha256": bundle_hash,
            "host_label": host_label,
            "runtime_seconds": time.monotonic() - started,
            "cadence": "3-hour" if product == "hasdm" else "daily",
            "tau_max": tau,
            "physical_max_lag_days": HASDM_LAG_DAYS
            if product == "hasdm"
            else GLOBAL_LAG_DAYS,
            "algorithm": {
                "name": "PCMCI+",
                "ci_test": "ParCorr analytic",
                "pc_alpha": ALPHA,
                "fdr_method": "none",
                "contemp_collider_rule": "majority",
                "conflict_resolution": True,
            },
            "missing_data_policy": {
                "sentinel": MISSING_FLAG,
                "remove_missing_upto_maxlag": False,
                "target_interpolation": "none",
            },
            "finite_counts_after_preprocessing": {
                name: int((values[:, index] != MISSING_FLAG).sum())
                for index, name in enumerate(columns)
            },
            "structural_assumption": "No target-to-target links are permitted, including vertical altitude links. This is exploratory structural absence, not vertical-coupling evidence.",
            "fdr_family": {
                "description": "one BH correction over every driver-to-target test at lag 0..tau",
                "test_count": expected_fdr_tests(case),
            },
            "result_files": {
                path.name: sha256(path) for path in (tests_path, retained_path)
            },
            "non_detections_explicit": int((~links.detected).sum()),
        },
        provenance,
    )
    atomic_json(
        {
            "fingerprint": identity,
            "case": case,
            "host_label": host_label,
            "started_epoch": started_epoch,
            "finished_epoch": time.time(),
            "status": "completed",
        },
        metadata,
    )
    return directory


def summarize(output: Path = OUTPUT) -> pd.DataFrame:
    rows = []
    for case in cases():
        directory = output / "cases" / case["id"]
        provenance, tests, retained = (
            directory / "provenance.json",
            directory / "driver_target_tests.csv",
            directory / "retained_links.csv",
        )
        if not all(path.exists() for path in (provenance, tests, retained)):
            raise ValueError(f"incomplete production artifact for {case['id']}")
        record = json.loads(provenance.read_text())
        if not record.get("production") or record.get("fdr_family", {}).get(
            "test_count"
        ) != expected_fdr_tests(case):
            raise ValueError(
                f"non-production or incompatible artifact for {case['id']}"
            )
        frame = pd.read_csv(tests)
        if (
            len(frame) != expected_fdr_tests(case)
            or sha256(tests) != record["result_files"][tests.name]
            or sha256(retained) != record["result_files"][retained.name]
        ):
            raise ValueError(
                f"incomplete or altered production artifact for {case['id']}"
            )
        rows.append(
            {
                "case": case["id"],
                "cadence": record["cadence"],
                "tests": len(frame),
                "detected": int(frame.detected.sum()),
                "fingerprint": record["fingerprint"],
            }
        )
    summary = pd.DataFrame(rows)
    atomic_csv(summary, output / "summary.csv")
    return summary


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--output", type=Path, default=OUTPUT)
    subcommands = command.add_subparsers(dest="command", required=True)
    subcommands.add_parser("prepare")
    subcommands.add_parser("list-cases")
    run_parser = subcommands.add_parser("run")
    run_parser.add_argument("case", choices=[case["id"] for case in cases()])
    run_parser.add_argument(
        "--host-label", required=True, help="Explicit remote/local execution label"
    )
    run_parser.add_argument("--recompute", action="store_true")
    subcommands.add_parser("summarize")
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "prepare":
        print(prepare(args.output))
    elif args.command == "list-cases":
        for case in cases():
            print(case["id"])
    elif args.command == "run":
        print(run(case_by_id(args.case), args.output, args.host_label, args.recompute))
    else:
        print(summarize(args.output).to_csv(index=False), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
