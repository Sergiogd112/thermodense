"""THROWAWAY: coverage feasibility of native SABER vertical totals in expanded cells.

Run from the repository root:
    uv run python scripts/prototype_saber_vertical_totals_expanded_cells.py

This reads only existing local SABER files.  It neither downloads data nor changes
the production decoder, data, bundle, or PCMCI inputs.
"""

from __future__ import annotations

import json
import re
import time
from datetime import date, timedelta
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
PROXIES = CHANNELS[2:]
CELLS = {
    "baseline_10x15": (10.0, 15.0),
    "expanded_20x30": (20.0, 30.0),
    "expanded_40x60": (40.0, 60.0),
}
EXPECTED_L2A_FILES = 12_461
MAX_LAG = 488
RNG_SEED = 20260901
DIAGNOSTIC_DRAWS = 20_000
NODE_COUNTS = (2, 3, 4, 5, 6, 8, 10, 12, 16)
MATERIAL_ZERO_FRACTION = 0.05
OVERWHELMING_ZERO_FRACTION = 0.95
L2A_RE = re.compile(r"SABER_L2A_(\d{4})(\d{3})_(\d+)_02\.\d+\.nc")


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


def slots_for(day: date, milliseconds: np.ndarray) -> np.ndarray:
    base = np.datetime64(day.isoformat(), "ms")
    return (
        (base + milliseconds.astype("timedelta64[ms]"))
        .astype("datetime64[h]")
        .astype("datetime64[3h]")
    )


def integrate_profiles(
    altitudes_km: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate finite native samples in [100, 140] km without interpolation."""
    totals = np.full(values.shape[0], np.nan)
    accepted = np.zeros(values.shape[0], dtype=bool)
    for scan in range(values.shape[0]):
        altitude = altitudes_km if altitudes_km.ndim == 1 else altitudes_km[scan]
        selected = np.isfinite(altitude) & (altitude >= 100.0) & (altitude <= 140.0)
        if selected.sum() < 2:
            continue
        z, y = altitude[selected], values[scan, selected]
        if not np.all(np.isfinite(y) & (y > MISSING_VALUE_LIMIT)):
            continue
        order = np.argsort(z)
        z, y = z[order], y[order]
        if z[-1] - z[0] < MIN_ALTITUDE_SPAN_KM:
            continue
        totals[scan] = np.trapezoid(y, x=z * 1000.0)
        accepted[scan] = True
    return totals, accepted


def native_120_indices(altitudes_km: np.ndarray, scans: int) -> np.ndarray:
    distances = np.where(
        np.isfinite(altitudes_km), np.abs(altitudes_km - 120.0), np.inf
    )
    if altitudes_km.ndim == 1:
        return np.full(scans, int(np.argmin(distances)))
    return np.argmin(distances, axis=1)


def add_profiles(
    day: date,
    values: np.ndarray,
    altitudes: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    milliseconds: np.ndarray,
    orbits: np.ndarray,
    centers: np.ndarray,
    index_by_slot: dict[np.datetime64, int],
    sums: np.ndarray,
    counts: np.ndarray,
    required_pairs: list[set[tuple[date, int]]] | None,
) -> None:
    """Add one product to every cell, locating each profile at its native nearest-120 km point."""
    scans = values.shape[0]
    totals, integrated = integrate_profiles(altitudes, values)
    level = native_120_indices(altitudes, scans)
    rows = np.arange(scans)
    lat120, lon120, time120 = (
        latitude[rows, level],
        longitude[rows, level],
        milliseconds[rows, level],
    )
    positions = np.array(
        [index_by_slot.get(value, -1) for value in slots_for(day, time120)]
    )
    center = np.where(positions >= 0, centers[np.maximum(positions, 0)], np.nan)
    valid = (
        integrated
        & np.isfinite(lat120)
        & np.isfinite(lon120)
        & np.isfinite(time120)
        & (positions >= 0)
    )
    for cell_index, (latitude_width, longitude_width) in enumerate(CELLS.values()):
        keep = (
            valid
            & (np.abs(lat120 - LATITUDE_CENTER) <= latitude_width / 2)
            & (circular_delta(lon120, center) <= longitude_width / 2)
        )
        np.add.at(sums[cell_index], positions[keep], totals[keep])
        np.add.at(counts[cell_index], positions[keep], 1)
        if required_pairs is not None:
            required_pairs[cell_index].update(
                (day, int(orbit)) for orbit in orbits[keep] if orbit >= 0
            )


def add_daily_file(
    path: Path,
    channel: str,
    centers: np.ndarray,
    index_by_slot: dict[np.datetime64, int],
    sums: np.ndarray,
    counts: np.ndarray,
    required_pairs: list[set[tuple[date, int]]],
) -> None:
    with netcdf_file(path, "r", mmap=False) as ds:
        day = file_day(ds)
        raw = np.asarray(ds.variables[channel].data, dtype=float)
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
            % 360.0
        )
        milliseconds = orient(np.asarray(ds.variables["time"].data, dtype=float), scans)
        orbits = (
            np.asarray(ds.variables.get("orbit").data, dtype=int)
            if "orbit" in ds.variables
            else np.full(scans, -1)
        )
    add_profiles(
        day,
        orient(raw, scans),
        altitudes,
        latitude,
        longitude,
        milliseconds,
        orbits,
        centers,
        index_by_slot,
        sums,
        counts,
        required_pairs,
    )


def add_l2a_file(
    path: Path,
    centers: np.ndarray,
    index_by_slot: dict[np.datetime64, int],
    sums: np.ndarray,
    counts: np.ndarray,
) -> None:
    with netcdf_file(path, "r", mmap=False) as ds:
        day = file_day(ds)
        raw = np.asarray(ds.variables[PROXIES[0]].data, dtype=float)
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
            % 360.0
        )
        milliseconds = orient(np.asarray(ds.variables["time"].data, dtype=float), scans)
        orbits = (
            np.asarray(ds.variables.get("orbit").data, dtype=int)
            if "orbit" in ds.variables
            else np.full(scans, -1)
        )
        values = {
            channel: orient(np.asarray(ds.variables[channel].data, dtype=float), scans)
            * 0.1
            for channel in PROXIES
        }
    for product_index, channel in enumerate(PROXIES):
        add_profiles(
            day,
            values[channel],
            altitudes,
            latitude,
            longitude,
            milliseconds,
            orbits,
            centers,
            index_by_slot,
            sums[:, product_index],
            counts[:, product_index],
            None,
        )


def coverage(
    mask: np.ndarray, counts: np.ndarray, calendar: pd.DatetimeIndex
) -> dict[str, object]:
    present, gaps = np.flatnonzero(mask), np.diff(np.flatnonzero(mask)) - 1
    result: dict[str, object] = {
        "nonempty_slots": int(mask.sum()),
        "calendar_slots": len(mask),
        "nonempty_fraction": float(mask.mean()),
        "observation_count": int(counts.sum()),
        "gap_definition": "empty 3-hour slots strictly between consecutive nonempty slots",
    }
    if len(gaps):
        maximum = int(np.argmax(gaps))
        result["gap_slots"] = {
            name: float(np.percentile(gaps, q))
            for name, q in (
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
        result["gap_slots"] = result["max_gap_bracket"] = None
    return result


def confusion(
    predicted: np.ndarray, observed: np.ndarray, calendar: pd.DatetimeIndex
) -> dict[str, object]:
    tp, fp, fn, tn = (
        int(np.sum(predicted & observed)),
        int(np.sum(predicted & ~observed)),
        int(np.sum(~predicted & observed)),
        int(np.sum(~predicted & ~observed)),
    )
    predicted_coverage, observed_coverage = (
        coverage(predicted, predicted.astype(int), calendar),
        coverage(observed, observed.astype(int), calendar),
    )
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "jaccard": tp / (tp + fp + fn) if tp + fp + fn else None,
        "count_bias": int(predicted.sum() - observed.sum()),
        "gap_metric_difference_estimated_minus_exact_slots": None
        if predicted_coverage["gap_slots"] is None
        or observed_coverage["gap_slots"] is None
        else {
            key: predicted_coverage["gap_slots"][key]
            - observed_coverage["gap_slots"][key]
            for key in predicted_coverage["gap_slots"]
        },
        "estimated_coverage": predicted_coverage,
        "exact_local_coverage": observed_coverage,
    }


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
    result = np.empty(len(draws), dtype=np.int32)
    for start in range(0, len(draws), 128):
        result[start : start + 128] = np.bitwise_count(
            np.bitwise_and.reduce(packed[draws[start : start + 128]], axis=1)
        ).sum(axis=1)
    return result


def exact_minimum_pairwise_overlap(masks: np.ndarray) -> int:
    minimum: int | None = None
    for left in range(len(masks)):
        for right in range(left, len(masks)):
            values = signal.correlate(
                masks[left].astype(np.int32),
                masks[right].astype(np.int32),
                method="fft",
            )[len(masks[left]) - 1 + np.arange(-MAX_LAG, MAX_LAG + 1)]
            if left == right:
                values = values[np.arange(-MAX_LAG, MAX_LAG + 1) != 0]
            minimum = (
                min(minimum, int(np.rint(values).min()))
                if minimum is not None
                else int(np.rint(values).min())
            )
    return 0 if minimum is None else minimum


def diagnostic(
    masks: np.ndarray, generic_draws: dict[int, np.ndarray], physical_lags: np.ndarray
) -> dict[str, object]:
    shifted = shifted_masks(masks)
    packed = np.packbits(shifted, axis=1)
    nodes = {}
    for count, draws in generic_draws.items():
        values = intersection_counts(packed, draws)
        nodes[str(count)] = {
            "draws": len(values),
            "zero_intersection_fraction": float((values == 0).mean()),
            "nonzero_intersection_fraction": float((values > 0).mean()),
            "median_intersection_slots": float(np.median(values)),
            "p95_intersection_slots": float(np.percentile(values, 95)),
            "max_intersection_slots": int(values.max()),
        }
    packed_by_channel = np.packbits(shifted.reshape(5, MAX_LAG + 1, -1), axis=2)
    selected = packed_by_channel[np.arange(5)[None, :], physical_lags]
    values = np.empty(len(selected), dtype=np.int32)
    for start in range(0, len(selected), 128):
        values[start : start + 128] = np.bitwise_count(
            np.bitwise_and.reduce(selected[start : start + 128], axis=1)
        ).sum(axis=1)
    return {
        "label": "DIAGNOSTIC ONLY: shifted-mask intersections, not PCMCI execution.",
        "lags_inclusive": [0, MAX_LAG],
        "same_seeded_draws_for_all_cells": True,
        "exact_minimum_pairwise_shifted_overlap_slots": exact_minimum_pairwise_overlap(
            masks
        ),
        "seeded_shifted_node_intersections": nodes,
        "five_physical_products_distinct_lags": {
            "draws": len(values),
            "seed": RNG_SEED + 1,
            "zero_complete_row_fraction": float((values == 0).mean()),
            "median_complete_rows": float(np.median(values)),
            "p95_complete_rows": float(np.percentile(values, 95)),
            "max_complete_rows": int(values.max()),
        },
    }


def l2a_files() -> tuple[list[Path], dict[int, set[date]]]:
    files = sorted(
        path.resolve()
        for path in L2A_DIR.glob("SABER_L2A_*_*.nc")
        if L2A_RE.fullmatch(path.name)
    )
    if len(files) != EXPECTED_L2A_FILES:
        raise RuntimeError(
            f"Expected {EXPECTED_L2A_FILES} selected Level2A files, found {len(files)}."
        )
    available_by_orbit: dict[int, set[date]] = {}
    for path in files:
        match = L2A_RE.fullmatch(path.name)
        assert match is not None
        file_day = date(int(match.group(1)), 1, 1) + timedelta(
            days=int(match.group(2)) - 1
        )
        available_by_orbit.setdefault(int(match.group(3)), set()).add(file_day)
    return files, available_by_orbit


def locally_resolved_pairs(
    required: set[tuple[date, int]], available_by_orbit: dict[int, set[date]]
) -> set[tuple[date, int]]:
    """Apply the decoder's same-day/adjacent-day Level2A orbit lookup locally."""
    return {
        (day, orbit)
        for day, orbit in required
        if len(
            set((day - timedelta(days=1), day, day + timedelta(days=1)))
            & available_by_orbit.get(orbit, set())
        )
        == 1
    }


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
    daily_files = {
        "CO2cool": _coverage(CO2_DIR, CO2_FILE_RE, START, END, "CO2 cooling"),
        "NOcool": _coverage(NO_DIR, NO_FILE_RE, START, END, "NO cooling"),
    }
    files, available_by_orbit = l2a_files()
    print(
        f"inputs: CO2={len(daily_files['CO2cool'])}, NO={len(daily_files['NOcool'])}, Level2A={len(files)}",
        flush=True,
    )
    direct_sums = np.zeros((len(CELLS), 2, len(calendar)))
    direct_counts = np.zeros((len(CELLS), 2, len(calendar)), dtype=np.int64)
    required_pairs = [[set() for _ in CELLS] for _ in range(2)]
    for product_index, channel in enumerate(("CO2cool", "NOcool")):
        for number, path in enumerate(daily_files[channel], 1):
            add_daily_file(
                path,
                channel,
                centers,
                index_by_slot,
                direct_sums[:, product_index],
                direct_counts[:, product_index],
                required_pairs[product_index],
            )
            if number % 100 == 0 or number == len(daily_files[channel]):
                print(
                    f"processed {number}/{len(daily_files[channel])} {channel} daily files",
                    flush=True,
                )
    proxy_sums = np.zeros((len(CELLS), 3, len(calendar)))
    proxy_counts = np.zeros((len(CELLS), 3, len(calendar)), dtype=np.int64)
    for number, path in enumerate(files, 1):
        add_l2a_file(path, centers, index_by_slot, proxy_sums, proxy_counts)
        if number % 250 == 0 or number == len(files):
            print(f"processed {number}/{len(files)} Level2A files", flush=True)
    direct_masks, exact_proxy_masks = direct_counts > 0, proxy_counts > 0
    generic_rng = np.random.default_rng(RNG_SEED)
    generic_draws = {
        nodes: generic_rng.integers(
            0, 5 * (MAX_LAG + 1), size=(DIAGNOSTIC_DRAWS, nodes)
        )
        for nodes in NODE_COUNTS
    }
    lag_rng = np.random.default_rng(RNG_SEED + 1)
    physical_lags = lag_rng.integers(0, MAX_LAG + 1, size=(DIAGNOSTIC_DRAWS, 5))
    while np.any(
        duplicates := np.array([len(np.unique(row)) < 5 for row in physical_lags])
    ):
        physical_lags[duplicates] = lag_rng.integers(
            0, MAX_LAG + 1, size=(int(duplicates.sum()), 5)
        )
    report: dict[str, object] = {
        "prototype": "THROWAWAY SABER 100--140 km expanded-cell coverage feasibility",
        "runtime_seconds": None,
        "inputs": {
            "official_local_co2_daily_files": len(daily_files["CO2cool"]),
            "official_local_no_daily_files": len(daily_files["NOcool"]),
            "existing_local_level2a_files": len(files),
            "expected_complete_baseline_Level2A_files": EXPECTED_L2A_FILES,
            "end_inclusive": END.isoformat(),
            "cell_center": "20N and per-slot HASDM longitude",
        },
        "vertical_integration": {
            "products": list(CHANNELS),
            "native_altitude_bounds_km_inclusive": ALTITUDE_BOUNDS_KM,
            "integration": "np.trapezoid against sorted native altitude metres",
            "no_interpolation_or_extrapolation": True,
            "minimum_observed_altitude_span_km": MIN_ALTITUDE_SPAN_KM,
            "finite_rule": "every selected native [100,140] sample must be finite and above the SABER missing sentinel",
            "representative_geolocation_time": "native level nearest 120 km",
            "proxy_conversion": "Level2A proxies multiplied by 0.1 from ergs/cm3/sec to W/m3",
        },
        "proxy_estimation": {
            "label": "ESTIMATED: no-download full-archive Level2A availability proxy; only the local Level2A results are exact lower bounds.",
            "candidates": ["CO2cool_or_NOcool", "CO2cool_and_NOcool"],
            "selection": "one global candidate with highest mean proxy-product Jaccard on the complete exact local 10x15 validation; ties choose intersection as conservative",
        },
        "cells": {},
        "sequential_verdict": {},
        "limitations": [
            "Estimated expanded proxy availability does not demonstrate that the corresponding missing Level2A value would pass native integration.",
            "Exact local Level2A coverage is a lower bound because the local archive is incomplete.",
            "This is a coverage feasibility gate, not PCMCI execution.",
        ],
    }
    baseline_index = list(CELLS).index("baseline_10x15")
    candidates = {
        "CO2cool_or_NOcool": np.any(direct_masks[baseline_index], axis=0),
        "CO2cool_and_NOcool": np.all(direct_masks[baseline_index], axis=0),
    }
    validations = {
        name: {
            channel: confusion(
                mask, exact_proxy_masks[baseline_index, product], calendar
            )
            for product, channel in enumerate(PROXIES)
        }
        for name, mask in candidates.items()
    }
    scores = {
        name: float(
            np.mean([result["jaccard"] or 0.0 for result in by_product.values()])
        )
        for name, by_product in validations.items()
    }
    selected = max(
        scores, key=lambda name: (scores[name], name == "CO2cool_and_NOcool")
    )
    report["proxy_estimation"].update(
        {
            "baseline_10x15_candidate_metrics": validations,
            "baseline_10x15_mean_jaccard": scores,
            "selected_global_rule": selected,
        }
    )
    selected_masks: list[np.ndarray] = []
    for cell_index, (cell_name, dimensions) in enumerate(CELLS.items()):
        cell_candidates = {
            "CO2cool_or_NOcool": np.any(direct_masks[cell_index], axis=0),
            "CO2cool_and_NOcool": np.all(direct_masks[cell_index], axis=0),
        }
        estimated_proxies = np.repeat(cell_candidates[selected][None, :], 3, axis=0)
        estimated_all = np.vstack((direct_masks[cell_index], estimated_proxies))
        selected_masks.append(estimated_all)
        direct_coverages = {
            channel: coverage(
                direct_masks[cell_index, product],
                direct_counts[cell_index, product],
                calendar,
            )
            for product, channel in enumerate(CHANNELS[:2])
        }
        exact_coverages = {
            channel: coverage(
                exact_proxy_masks[cell_index, product],
                proxy_counts[cell_index, product],
                calendar,
            )
            for product, channel in enumerate(PROXIES)
        }
        required = set().union(
            *(required_pairs[product][cell_index] for product in range(2))
        )
        resolved = locally_resolved_pairs(required, available_by_orbit)
        average_size = float(np.mean([path.stat().st_size for path in files]))
        report["cells"][cell_name] = {
            "dimensions_degrees": {
                "latitude": dimensions[0],
                "longitude": dimensions[1],
            },
            "exact_CO2_NO_coverage": direct_coverages,
            **(
                {"exact_complete_baseline_Level2A_proxy_coverage": exact_coverages}
                if cell_name == "baseline_10x15"
                else {"exact_local_Level2A_lower_bound_proxy_coverage": exact_coverages}
            ),
            "estimated_full_archive_rule": selected,
            "estimated_full_archive_proxy_coverage": {
                channel: coverage(
                    estimated_proxies[product],
                    estimated_proxies[product].astype(int),
                    calendar,
                )
                for product, channel in enumerate(PROXIES)
            },
            "joint_all_five_estimated_coverage": coverage(
                np.all(estimated_all, axis=0),
                np.sum(
                    np.vstack(
                        (direct_counts[cell_index], estimated_proxies.astype(int))
                    ),
                    axis=0,
                ),
                calendar,
            ),
            "required_Level2A_day_orbit_pairs": len(required),
            "pairs_resolved_locally": len(resolved),
            "additional_files_estimated": len(required - resolved),
            "required_storage_estimate_bytes": int(round(len(required) * average_size)),
            "additional_storage_estimate_bytes": int(
                round(len(required - resolved) * average_size)
            ),
            "average_local_Level2A_file_size_bytes": average_size,
            "shifted_mask_diagnostic": diagnostic(
                estimated_all, generic_draws, physical_lags
            ),
        }

    def gate(cell_name: str) -> dict[str, object]:
        diagnostics = report["cells"][cell_name]["shifted_mask_diagnostic"]
        five_zero = diagnostics["five_physical_products_distinct_lags"][
            "zero_complete_row_fraction"
        ]
        high_node_zero = {
            str(nodes): diagnostics["seeded_shifted_node_intersections"][str(nodes)][
                "zero_intersection_fraction"
            ]
            for nodes in (12, 16)
        }
        failed = five_zero > MATERIAL_ZERO_FRACTION or all(
            value >= OVERWHELMING_ZERO_FRACTION for value in high_node_zero.values()
        )
        return {
            "verdict": "FAIL" if failed else "PASS",
            "criteria": {
                "material_five_product_zero_row_fraction_gt": MATERIAL_ZERO_FRACTION,
                "overwhelming_zero_fraction_ge": OVERWHELMING_ZERO_FRACTION,
            },
            "five_product_zero_row_fraction": five_zero,
            "12_16_node_zero_fractions": high_node_zero,
            "statement": "Coverage feasibility gate only; not PCMCI execution or proof of unlimited-conditioning feasibility.",
        }

    report["sequential_verdict"]["expanded_20x30"] = gate("expanded_20x30")
    report["sequential_verdict"]["expanded_40x60"] = gate("expanded_40x60")
    report["runtime_seconds"] = time.monotonic() - started
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_DIR / "prototype_saber_vertical_totals_expanded_cells.npz",
        timestamps=calendar.to_numpy(dtype="datetime64[ns]"),
        cell_names=np.array(list(CELLS)),
        channels=np.array(CHANNELS),
        direct_counts=direct_counts,
        exact_local_proxy_counts=proxy_counts,
        direct_masks=direct_masks,
        exact_local_proxy_masks=exact_proxy_masks,
        estimated_full_masks=np.array(selected_masks),
        physical_product_lag_draws=physical_lags,
    )
    output = OUTPUT_DIR / "prototype_saber_vertical_totals_expanded_cells.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output} and NPZ in {report['runtime_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
