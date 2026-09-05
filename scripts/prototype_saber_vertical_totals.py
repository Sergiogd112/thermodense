"""THROWAWAY: test vertically integrated SABER products in the current HASDM cell.

Run once from the repository root with:
    uv run python scripts/prototype_saber_vertical_totals.py

This deliberately does not change the production decoder, data, bundle, or PCMCI.
"""

from __future__ import annotations

import json
import re
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import netcdf_file
from thermodense.saber import (
    CO2_FILE_RE,
    MISSING_VALUE_LIMIT,
    NO_FILE_RE,
    _coverage,
    hasdm_longitudes_by_timestamp,
)

CO2_DIR = Path("data/original/saber/co2_cooling_profiles")
NO_DIR = Path("data/original/saber/no_cooling_profiles")
L2A_DIR = Path("data/original/saber/level2a")
SAMPLES = Path(
    "outputs/figures/results/hasdm_msis_model_errors/data/"
    "hasdm_msis_errors_nearest_timestamp_grid_samples.parquet"
)
BASELINE = Path("data/decoded/saber/saber_hasdm_maunaloa_3hour.parquet")
OUTPUT_DIR = Path("outputs/prototypes/density_pcmci_3hour_and_daily")
START, END = date(2002, 1, 25), date(2025, 7, 20)
LATITUDE_CENTER = 20.0
ALTITUDE_BOUNDS_KM = (100.0, 140.0)
MIN_ALTITUDE_SPAN_KM = 38.0
CHANNELS = ("CO2cool", "NOcool", "O2_1delta_ver", "OH_16_ver", "OH_20_ver")
PROXIES = frozenset(CHANNELS[2:])
EXPECTED_L2A_FILES = 12_461
MAX_LAG = 488
RNG_SEED = 20260901
DIAGNOSTIC_DRAWS = 20_000
NODE_COUNTS = (2, 3, 4, 5, 6, 8, 10, 12, 16)


def circular_delta(longitude: np.ndarray, center: np.ndarray) -> np.ndarray:
    return np.abs((longitude - center + 180.0) % 360.0 - 180.0)


def orient(values: np.ndarray, scans: int) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError(f"expected scan/altitude matrix, got {values.shape}")
    if values.shape[0] == scans:
        return values
    if values.shape[1] == scans:
        return values.T
    raise ValueError(f"cannot identify scan dimension in {values.shape}")


def file_day(ds) -> date:
    if "date" in ds.variables:
        token = str(int(np.asarray(ds.variables["date"].data).flat[0]))
        return date(int(token[:4]), 1, 1) + pd.Timedelta(days=int(token[4:]) - 1)
    return date(int(np.asarray(ds.variables["year"].data).item()), 1, 1) + pd.Timedelta(
        days=int(np.asarray(ds.variables["day"].data).item()) - 1
    )


def slot(day: date, milliseconds: np.ndarray) -> np.ndarray:
    base = np.datetime64(day.isoformat(), "ms")
    stamps = base + milliseconds.astype("timedelta64[ms]")
    return stamps.astype("datetime64[h]").astype("datetime64[3h]")


def integrate_profiles(
    altitudes_km: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate only a wholly finite native 100--140 km profile sequence.

    This strict rule rejects a profile with any missing selected value rather than
    joining finite fragments across an internal gap.  It also leaves its observed
    endpoints unchanged: no boundary interpolation or extrapolation is performed.
    """
    totals = np.full(values.shape[0], np.nan)
    spans_km = np.full(values.shape[0], np.nan)
    accepted = np.zeros(values.shape[0], dtype=bool)
    if altitudes_km.ndim == 1:
        selected = (
            np.isfinite(altitudes_km)
            & (altitudes_km >= ALTITUDE_BOUNDS_KM[0])
            & (altitudes_km <= ALTITUDE_BOUNDS_KM[1])
        )
        if selected.sum() < 2:
            return totals, spans_km, accepted
        z = altitudes_km[selected]
        order = np.argsort(z)
        z = z[order]
        span = float(z[-1] - z[0])
        spans_km.fill(span)
        finite = np.all(
            np.isfinite(values[:, selected])
            & (values[:, selected] > MISSING_VALUE_LIMIT),
            axis=1,
        )
        if span >= MIN_ALTITUDE_SPAN_KM:
            totals[finite] = np.trapezoid(
                values[:, selected][:, order][finite], x=z * 1000.0, axis=1
            )
            accepted = finite
        return totals, spans_km, accepted
    for scan in range(values.shape[0]):
        altitude = altitudes_km if altitudes_km.ndim == 1 else altitudes_km[scan]
        selected = (
            np.isfinite(altitude)
            & (altitude >= ALTITUDE_BOUNDS_KM[0])
            & (altitude <= ALTITUDE_BOUNDS_KM[1])
        )
        if selected.sum() < 2:
            continue
        z = altitude[selected]
        y = values[scan, selected]
        if not np.all(np.isfinite(y) & (y > MISSING_VALUE_LIMIT)):
            continue
        order = np.argsort(z)
        z, y = z[order], y[order]
        span = float(z[-1] - z[0])
        spans_km[scan] = span
        if span < MIN_ALTITUDE_SPAN_KM:
            continue
        totals[scan] = float(np.trapezoid(y, x=z * 1000.0))
        accepted[scan] = True
    return totals, spans_km, accepted


def native_120_indices(altitudes_km: np.ndarray, scans: int) -> np.ndarray:
    if altitudes_km.ndim == 1:
        return np.full(scans, int(np.nanargmin(np.abs(altitudes_km - 120.0))))
    return np.nanargmin(np.abs(altitudes_km - 120.0), axis=1)


def add_profiles(
    day: date,
    values: np.ndarray,
    altitudes: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    milliseconds: np.ndarray,
    centers: np.ndarray,
    index_by_slot: dict[np.datetime64, int],
    sums: np.ndarray,
    counts: np.ndarray,
    rejection_counts: dict[str, int],
    accepted_spans: list[float],
) -> None:
    scans = values.shape[0]
    totals, spans, integrated = integrate_profiles(altitudes, values)
    level = native_120_indices(altitudes, scans)
    rows = np.arange(scans)
    lat120, lon120, time120 = (
        latitude[rows, level],
        longitude[rows, level],
        milliseconds[rows, level],
    )
    slots = slot(day, time120)
    positions = np.array([index_by_slot.get(value, -1) for value in slots])
    center = np.where(positions >= 0, centers[np.maximum(positions, 0)], np.nan)
    geo = np.isfinite(lat120) & np.isfinite(lon120) & np.isfinite(time120)
    in_cell = (
        geo
        & (positions >= 0)
        & (np.abs(lat120 - LATITUDE_CENTER) <= 5.0)
        & (circular_delta(lon120, center) <= 7.5)
    )
    rejection_counts["profiles_seen"] += scans
    rejection_counts["profiles_rejected_missing_or_insufficient_levels"] += int(
        np.sum(~np.isfinite(spans))
    )
    rejection_counts["profiles_rejected_span_below_38km"] += int(
        np.sum(np.isfinite(spans) & ~integrated)
    )
    rejection_counts["integrated_profiles_before_spatial_match"] += int(
        integrated.sum()
    )
    rejection_counts["integrated_profiles_rejected_geolocation_or_cell"] += int(
        np.sum(integrated & ~in_cell)
    )
    keep = integrated & in_cell
    np.add.at(sums, positions[keep], totals[keep])
    np.add.at(counts, positions[keep], 1)
    accepted_spans.extend(spans[keep].tolist())


def add_file(
    path: Path,
    variable: str,
    conversion: float,
    centers: np.ndarray,
    index_by_slot: dict[np.datetime64, int],
    sums: np.ndarray,
    counts: np.ndarray,
    rejection_counts: dict[str, int],
    accepted_spans: list[float],
) -> None:
    with netcdf_file(path, "r", mmap=False) as ds:
        day = file_day(ds)
        raw = np.asarray(ds.variables[variable].data, dtype=float)
        scans = raw.shape[0]
        altitude_name = "altitude" if "altitude" in ds.variables else "tpaltitude"
        altitudes = np.asarray(ds.variables[altitude_name].data, dtype=float)
        if altitudes.ndim == 2:
            altitudes = orient(altitudes, scans)
        latitude = orient(
            np.asarray(ds.variables["tplatitude"].data, dtype=float), scans
        )
        longitude = (
            orient(np.asarray(ds.variables["tplongitude"].data, dtype=float), scans)
            % 360
        )
        milliseconds = orient(np.asarray(ds.variables["time"].data, dtype=float), scans)
    add_profiles(
        day,
        orient(raw, scans) * conversion,
        altitudes,
        latitude,
        longitude,
        milliseconds,
        centers,
        index_by_slot,
        sums,
        counts,
        rejection_counts,
        accepted_spans,
    )


def add_level2a_file(
    path: Path,
    centers: np.ndarray,
    index_by_slot: dict[np.datetime64, int],
    sums: dict[str, np.ndarray],
    counts: dict[str, np.ndarray],
    rejections: dict[str, dict[str, int]],
    spans: dict[str, list[float]],
) -> None:
    """Read a selected Level2A file once, avoiding proxy-product double counting."""
    with netcdf_file(path, "r", mmap=False) as ds:
        day = file_day(ds)
        raw = np.asarray(ds.variables["O2_1delta_ver"].data, dtype=float)
        scans = raw.shape[0]
        altitude_name = "altitude" if "altitude" in ds.variables else "tpaltitude"
        altitudes = np.asarray(ds.variables[altitude_name].data, dtype=float)
        if altitudes.ndim == 2:
            altitudes = orient(altitudes, scans)
        latitude = orient(
            np.asarray(ds.variables["tplatitude"].data, dtype=float), scans
        )
        longitude = (
            orient(np.asarray(ds.variables["tplongitude"].data, dtype=float), scans)
            % 360
        )
        milliseconds = orient(np.asarray(ds.variables["time"].data, dtype=float), scans)
        values = {
            channel: orient(np.asarray(ds.variables[channel].data, dtype=float), scans)
            * 0.1
            for channel in PROXIES
        }
    for channel in PROXIES:
        add_profiles(
            day,
            values[channel],
            altitudes,
            latitude,
            longitude,
            milliseconds,
            centers,
            index_by_slot,
            sums[channel],
            counts[channel],
            rejections[channel],
            spans[channel],
        )


def coverage(
    mask: np.ndarray, counts: np.ndarray, calendar: pd.DatetimeIndex
) -> dict[str, object]:
    present = np.flatnonzero(mask)
    result: dict[str, object] = {
        "nonempty_slots": int(mask.sum()),
        "calendar_slots": int(len(mask)),
        "nonempty_fraction": float(mask.mean()),
        "observation_count": int(counts.sum()),
        "gap_definition": "empty 3-hour slots strictly between consecutive nonempty slots",
    }
    gaps = np.diff(present) - 1
    if len(gaps):
        maximum = int(np.argmax(gaps))
        result["gap_slots"] = {
            name: float(np.percentile(gaps, quantile))
            for name, quantile in (
                ("median", 50),
                ("p75", 75),
                ("p95", 95),
                ("p99", 99),
                ("max", 100),
            )
        }
        result["max_gap_bracket"] = {
            "before": calendar[present[maximum]].isoformat(),
            "after": calendar[present[maximum + 1]].isoformat(),
        }
    else:
        result["gap_slots"] = None
        result["max_gap_bracket"] = None
    return result


def shifted_masks(masks: np.ndarray) -> np.ndarray:
    positions = np.arange(masks.shape[1])
    return np.array(
        [
            np.roll(mask, lag) & (positions >= lag)
            for mask in masks
            for lag in range(MAX_LAG + 1)
        ]
    )


def intersection_counts(packed: np.ndarray, draws: np.ndarray) -> np.ndarray:
    counts = np.empty(len(draws), dtype=np.int32)
    for start in range(0, len(draws), 128):
        selected = packed[draws[start : start + 128]]
        joint = np.bitwise_and.reduce(selected, axis=1)
        counts[start : start + len(selected)] = np.bitwise_count(joint).sum(axis=1)
    return counts


def exact_minimum_pairwise_overlap(masks: np.ndarray) -> int:
    minimum: int | None = None
    for left in range(len(masks)):
        for right in range(left, len(masks)):
            correlations = signal.correlate(
                masks[left].astype(np.int16),
                masks[right].astype(np.int16),
                method="fft",
            )
            offsets = np.arange(-MAX_LAG, MAX_LAG + 1)
            values = correlations[len(masks[left]) - 1 + offsets]
            if left == right:
                values = values[offsets != 0]
            candidate = int(np.rint(values).min())
            minimum = candidate if minimum is None else min(minimum, candidate)
    return 0 if minimum is None else minimum


def sparse_lag_diagnostic(masks: np.ndarray) -> tuple[dict[str, object], np.ndarray]:
    """Return a seeded diagnostic only; it is not PCMCI execution or feasibility."""
    shifted = shifted_masks(masks)
    packed = np.packbits(shifted, axis=1)
    rng = np.random.default_rng(RNG_SEED)
    summaries: dict[str, dict[str, float | int]] = {}
    five_draws: np.ndarray | None = None
    for nodes in NODE_COUNTS:
        draws = rng.integers(0, len(shifted), size=(DIAGNOSTIC_DRAWS, nodes))
        values = intersection_counts(packed, draws)
        summaries[str(nodes)] = {
            "draws": DIAGNOSTIC_DRAWS,
            "nonzero_intersection_fraction": float((values > 0).mean()),
            "median_intersection_slots": float(np.median(values)),
            "p95_intersection_slots": float(np.percentile(values, 95)),
            "max_intersection_slots": int(values.max()),
        }
        if nodes == 5:
            five_draws = draws
    assert five_draws is not None
    return (
        {
            "label": "DIAGNOSTIC ONLY: seeded shifted-mask intersections; not PCMCI execution or a completion proof.",
            "lags_inclusive": [0, MAX_LAG],
            "seed": RNG_SEED,
            "exact_minimum_pairwise_shifted_overlap_slots": exact_minimum_pairwise_overlap(
                masks
            ),
            "seeded_shifted_node_intersections": summaries,
        },
        five_draws,
    )


def five_channel_lag_draws() -> np.ndarray:
    """Draw one distinct lag for each of the five separate physical products."""
    rng = np.random.default_rng(RNG_SEED + 1)
    draws = rng.integers(0, MAX_LAG + 1, size=(DIAGNOSTIC_DRAWS, len(CHANNELS)))
    duplicate = np.array([len(np.unique(row)) < len(row) for row in draws])
    while np.any(duplicate):
        draws[duplicate] = rng.integers(
            0, MAX_LAG + 1, size=(int(duplicate.sum()), len(CHANNELS))
        )
        duplicate = np.array([len(np.unique(row)) < len(row) for row in draws])
    return draws


def five_channel_feasibility(
    masks: np.ndarray, lag_draws: np.ndarray
) -> dict[str, float | int]:
    shifted = shifted_masks(masks).reshape(len(CHANNELS), MAX_LAG + 1, -1)
    packed = np.packbits(shifted, axis=2)
    selected = packed[np.arange(len(CHANNELS))[None, :], lag_draws]
    values = np.empty(len(lag_draws), dtype=np.int32)
    for start in range(0, len(selected), 128):
        joint = np.bitwise_and.reduce(selected[start : start + 128], axis=1)
        values[start : start + len(joint)] = np.bitwise_count(joint).sum(axis=1)
    return {
        "draws": int(len(values)),
        "zero_complete_row_fraction": float((values == 0).mean()),
        "nonzero_complete_row_fraction": float((values > 0).mean()),
        "median_complete_rows": float(np.median(values)),
        "p95_complete_rows": float(np.percentile(values, 95)),
        "max_complete_rows": int(values.max()),
    }


def l2a_files() -> list[Path]:
    pattern = re.compile(r"SABER_L2A_(\d{4})(\d{3})_\d+_02\.\d+\.nc")
    files = sorted(
        {
            path.resolve()
            for path in L2A_DIR.glob("SABER_L2A_*_*.nc")
            if pattern.fullmatch(path.name)
        }
    )
    if len(files) != EXPECTED_L2A_FILES:
        raise RuntimeError(
            f"Expected {EXPECTED_L2A_FILES} selected Level2A files, found {len(files)}."
        )
    return files


def main() -> None:
    started = time.monotonic()
    baseline = pd.read_parquet(BASELINE)
    calendar = pd.DatetimeIndex(pd.to_datetime(baseline["timestamp"]))
    centers_by_timestamp = hasdm_longitudes_by_timestamp(SAMPLES)
    centers = np.array(
        [centers_by_timestamp.get(value.to_pydatetime(), np.nan) for value in calendar]
    )
    index_by_slot = {
        value.to_datetime64(): index for index, value in enumerate(calendar)
    }
    files = {
        "CO2cool": _coverage(CO2_DIR, CO2_FILE_RE, START, END, "CO2 cooling"),
        "NOcool": _coverage(NO_DIR, NO_FILE_RE, START, END, "NO cooling"),
        "Level2A": l2a_files(),
    }
    print(
        "inputs: " + ", ".join(f"{name}={len(paths)}" for name, paths in files.items()),
        flush=True,
    )
    sums = {channel: np.zeros(len(calendar)) for channel in CHANNELS}
    counts = {channel: np.zeros(len(calendar), dtype=np.int64) for channel in CHANNELS}
    rejections = {
        channel: {
            key: 0
            for key in (
                "profiles_seen",
                "profiles_rejected_missing_or_insufficient_levels",
                "profiles_rejected_span_below_38km",
                "integrated_profiles_before_spatial_match",
                "integrated_profiles_rejected_geolocation_or_cell",
            )
        }
        for channel in CHANNELS
    }
    spans = {channel: [] for channel in CHANNELS}
    for channel in ("CO2cool", "NOcool"):
        for number, path in enumerate(files[channel], 1):
            add_file(
                path,
                channel,
                1.0,
                centers,
                index_by_slot,
                sums[channel],
                counts[channel],
                rejections[channel],
                spans[channel],
            )
            if number % 100 == 0 or number == len(files[channel]):
                print(
                    f"processed {number}/{len(files[channel])} {channel} daily files",
                    flush=True,
                )
    for number, path in enumerate(files["Level2A"], 1):
        add_level2a_file(path, centers, index_by_slot, sums, counts, rejections, spans)
        if number % 250 == 0 or number == len(files["Level2A"]):
            print(
                f"processed {number}/{len(files['Level2A'])} Level2A files", flush=True
            )
    totals = {
        channel: np.divide(
            sums[channel],
            counts[channel],
            out=np.full(len(calendar), np.nan),
            where=counts[channel] > 0,
        )
        for channel in CHANNELS
    }
    masks = np.array([np.isfinite(totals[channel]) for channel in CHANNELS])
    integrated_diagnostic, _generic_five_draws = sparse_lag_diagnostic(masks)
    baseline_columns = {
        "CO2cool": "saber_co2cool_119km_w_m3",
        "NOcool": "saber_nocool_119km_w_m3",
        "O2_1delta_ver": "saber_o2_1delta_ver_119km_w_m3",
        "OH_16_ver": "saber_oh_16_ver_119km_w_m3",
        "OH_20_ver": "saber_oh_20_ver_119km_w_m3",
    }
    baseline_masks = np.array(
        [
            np.isfinite(baseline[column].to_numpy(float))
            for column in baseline_columns.values()
        ]
    )
    lag_draws = five_channel_lag_draws()
    integrated_counts = np.array([counts[channel] for channel in CHANNELS])
    baseline_counts = np.array(
        [
            baseline[f"{column}_observations"].to_numpy(int)
            for column in baseline_columns.values()
        ]
    )
    report: dict[str, object] = {
        "prototype": "THROWAWAY native SABER 100--140 km vertical-total feasibility calculation",
        "runtime_seconds": time.monotonic() - started,
        "inputs": {
            "official_local_co2_daily_files": len(files["CO2cool"]),
            "official_local_no_daily_files": len(files["NOcool"]),
            "selected_unique_level2a_files": len(files["Level2A"]),
            "end_inclusive": END.isoformat(),
            "hasdm_cell": "20N +/-5 degrees latitude; per-slot longitude +/-7.5 degrees",
        },
        "vertical_integration": {
            "native_altitude_bounds_km_inclusive": ALTITUDE_BOUNDS_KM,
            "integration": "np.trapezoid(value_w_m3, x=native_tangent_altitude_m) after ascending altitude sort",
            "no_boundary_extrapolation_or_interpolation": True,
            "minimum_observed_altitude_span_km": MIN_ALTITUDE_SPAN_KM,
            "missing_value_rule": "reject a profile unless every selected native 100--140 km value is finite and above the SABER missing sentinel; no integration across internal missing values",
            "proxy_conversion": "O2_1delta_ver, OH_16_ver, and OH_20_ver multiplied by 0.1 from ergs/cm3/sec to W/m3 before integration",
            "profile_geolocation": "native level nearest 120 km",
        },
        "integrated_channels_w_m2": {},
        "current_119km_w_m3": {},
        "current_119km_union": coverage(
            np.any(baseline_masks, axis=0), baseline_counts.sum(axis=0), calendar
        ),
        "current_119km_intersection": coverage(
            np.all(baseline_masks, axis=0),
            np.where(np.all(baseline_masks, axis=0), baseline_counts.sum(axis=0), 0),
            calendar,
        ),
        "integrated_joint_all_five": coverage(
            np.all(masks, axis=0),
            np.where(np.all(masks, axis=0), integrated_counts.sum(axis=0), 0),
            calendar,
        ),
        "integrated_shifted_mask_diagnostic_only": integrated_diagnostic,
        "five_differently_lagged_nodes_diagnostic_only": {
            "label": "DIAGNOSTIC ONLY: one distinct lag per physical product; no PCMCI was run.",
            "lags_inclusive": [0, MAX_LAG],
            "seed": RNG_SEED + 1,
            "integrated": five_channel_feasibility(masks, lag_draws),
            "current_119km_same_lag_draws": five_channel_feasibility(
                baseline_masks, lag_draws
            ),
        },
        "limitations": [
            "Columns cover only observed native tangent-altitude endpoints; a >=38 km span still need not reach both nominal 100 and 140 km bounds.",
            "Each product is separately integrated and averaged; proxy products remain emission proxies, not cooling rates.",
            "Sparse-mask diagnostics do not establish that unlimited PCMCI conditioning is feasible.",
        ],
    }
    for index, channel in enumerate(CHANNELS):
        report["integrated_channels_w_m2"][channel] = {
            **coverage(masks[index], counts[channel], calendar),
            "accepted_profile_altitude_span_km": {
                "min": float(np.min(spans[channel])) if spans[channel] else None,
                "median": float(np.median(spans[channel])) if spans[channel] else None,
                "max": float(np.max(spans[channel])) if spans[channel] else None,
            },
            "profile_rejections": rejections[channel],
        }
        current_counts = baseline[f"{baseline_columns[channel]}_observations"].to_numpy(
            int
        )
        report["current_119km_w_m3"][channel] = coverage(
            baseline_masks[index], current_counts, calendar
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_DIR / "prototype_saber_vertical_totals.npz",
        timestamps=calendar.to_numpy(dtype="datetime64[ns]"),
        **{channel: totals[channel] for channel in CHANNELS},
        **{f"{channel}_observations": counts[channel] for channel in CHANNELS},
    )
    output = OUTPUT_DIR / "prototype_saber_vertical_totals.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output} and NPZ in {report['runtime_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
