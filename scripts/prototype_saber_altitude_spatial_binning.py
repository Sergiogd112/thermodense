"""THROWAWAY: test native SABER CO2 altitude strata and wider HASDM cells.

Run once from the repository root with:
    uv run python scripts/prototype_saber_altitude_spatial_binning.py
"""

from __future__ import annotations

import json
import re
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import netcdf_file
from thermodense.saber import (
    CO2_FILE_RE,
    MISSING_VALUE_LIMIT,
    _coverage,
    _file_day,
    hasdm_longitudes_by_timestamp,
)

# This is intentionally a standalone, non-production calculation.
CO2_DIR = Path("data/original/saber/co2_cooling_profiles")
L2A_DIR = Path("data/original/saber/level2a")
SAMPLES = Path(
    "outputs/figures/results/hasdm_msis_model_errors/data/"
    "hasdm_msis_errors_nearest_timestamp_grid_samples.parquet"
)
BASELINE = Path("data/decoded/saber/saber_hasdm_maunaloa_3hour.parquet")
OUTPUT = Path(
    "outputs/prototypes/density_pcmci_3hour_and_daily/"
    "prototype_saber_altitude_spatial_binning.json"
)
START, END = date(2002, 1, 25), date(2025, 7, 20)
LATITUDE_CENTER = 20.0  # Current decoded HASDM nearest latitude, not Mauna Loa.
STRATA = (
    (100.0, 113.33333333333333),
    (113.33333333333333, 126.66666666666667),
    (126.66666666666667, 140.0),
)
CELLS = {"existing_10x15_deg": (10.0, 15.0), "expanded_20x30_deg": (20.0, 30.0)}
RNG_SEED, SPARSE_SAMPLE_SIZE = 20260901, 20_000


def circular_delta(longitude: np.ndarray, center: np.ndarray) -> np.ndarray:
    return np.abs((longitude - center + 180.0) % 360.0 - 180.0)


def slot(day: date, milliseconds: np.ndarray) -> np.ndarray:
    base = np.datetime64(day.isoformat(), "ms")
    stamps = base + milliseconds.astype("timedelta64[ms]")
    return stamps.astype("datetime64[h]").astype("datetime64[3h]")


def orient(values: np.ndarray, scans: int) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError(f"expected scan/altitude matrix, got {values.shape}")
    if values.shape[0] == scans:
        return values
    if values.shape[1] == scans:
        return values.T
    raise ValueError(f"cannot identify scan dimension in {values.shape}")


def coverage(mask: np.ndarray, calendar: pd.DatetimeIndex) -> dict[str, object]:
    present = np.flatnonzero(mask)
    gaps = np.diff(present) - 1
    result: dict[str, object] = {
        "nonempty_slots": int(mask.sum()),
        "calendar_slots": int(len(mask)),
        "nonempty_fraction": float(mask.mean()),
        "gap_definition": "empty 3-hour slots strictly between consecutive nonempty slots",
    }
    if len(gaps):
        max_index = int(np.argmax(gaps))
        result.update(
            {
                "gap_slots": {
                    name: float(np.percentile(gaps, q))
                    for name, q in (
                        ("median", 50),
                        ("p75", 75),
                        ("p95", 95),
                        ("p99", 99),
                        ("max", 100),
                    )
                },
                "max_gap_bracket": {
                    "before": calendar[present[max_index]].isoformat(),
                    "after": calendar[present[max_index + 1]].isoformat(),
                },
            }
        )
    else:
        result.update({"gap_slots": None, "max_gap_bracket": None})
    return result


def slot_mean_summary(values: np.ndarray) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    return {
        "nonempty_slot_means": int(len(finite)),
        "mean_of_slot_means_w_m3": float(finite.mean()),
        "median_slot_mean_w_m3": float(np.median(finite)),
    }


def sparse_lag_diagnostic(masks: list[np.ndarray]) -> dict[str, object]:
    """A seeded sampling diagnostic only; it does not establish PCMCI feasibility."""
    rng = np.random.default_rng(RNG_SEED)
    lags = np.arange(489)
    shifted = np.array(
        [
            np.roll(mask, lag) & (np.arange(len(mask)) >= lag)
            for mask in masks
            for lag in lags
        ]
    )
    pairwise = [
        int((shifted[a] & shifted[b]).sum())
        for a in range(len(shifted))
        for b in range(a + 1, len(shifted))
    ]
    samples: dict[str, dict[str, float | int]] = {}
    for nodes in (2, 3, 4, 5):
        draws = rng.integers(0, len(shifted), size=(SPARSE_SAMPLE_SIZE, nodes))
        counts = np.fromiter(
            (np.logical_and.reduce(shifted[row]).sum() for row in draws),
            dtype=int,
            count=len(draws),
        )
        samples[str(nodes)] = {
            "draws": SPARSE_SAMPLE_SIZE,
            "nonzero_draw_fraction": float((counts > 0).mean()),
            "median_intersection_slots": float(np.median(counts)),
            "p95_intersection_slots": float(np.percentile(counts, 95)),
            "max_intersection_slots": int(counts.max()),
        }
    return {
        "label": "Sparse-lag feasibility diagnostic only; not a proof of PCMCI completion.",
        "lags_inclusive": [0, 488],
        "minimum_pairwise_shifted_bin_overlap_slots": min(pairwise),
        "seed": RNG_SEED,
        "seeded_shifted_node_intersections": samples,
    }


def locally_resolved_pairs(required: set[tuple[str, int]]) -> set[tuple[str, int]]:
    """Apply the decoder's same-day/adjacent-day Level2A orbit lookup locally."""
    available: dict[int, set[date]] = {}
    for path in L2A_DIR.glob("SABER_L2A_*_*.nc"):
        match = re.fullmatch(r"SABER_L2A_(\d{4})(\d{3})_(\d+)_02\.\d+\.nc", path.name)
        if match:
            file_day = date(int(match.group(1)), 1, 1) + timedelta(
                days=int(match.group(2)) - 1
            )
            available.setdefault(int(match.group(3)), set()).add(file_day)
    return {
        pair
        for pair in required
        if any(
            candidate in available.get(pair[1], set())
            for candidate in (
                date.fromisoformat(pair[0]) - timedelta(days=1),
                date.fromisoformat(pair[0]),
                date.fromisoformat(pair[0]) + timedelta(days=1),
            )
        )
    }


def main() -> None:
    started = time.monotonic()
    baseline = pd.read_parquet(BASELINE)
    calendar = pd.DatetimeIndex(pd.to_datetime(baseline["timestamp"]))
    longitude_by_slot = hasdm_longitudes_by_timestamp(SAMPLES)
    # The frozen HASDM sample intentionally has a small number of absent slots.
    # They remain unmatched rather than borrowing a neighbouring longitude.
    centers = np.array(
        [longitude_by_slot.get(t.to_pydatetime(), np.nan) for t in calendar]
    )
    index_by_slot = {
        timestamp.to_datetime64(): index for index, timestamp in enumerate(calendar)
    }
    files = _coverage(CO2_DIR, CO2_FILE_RE, START, END, "CO2 cooling")
    sums = {name: np.zeros((3, len(calendar))) for name in CELLS}
    counts = {name: np.zeros((3, len(calendar)), dtype=np.int64) for name in CELLS}
    required = {name: set() for name in CELLS}

    for file_index, path in enumerate(files, 1):
        with netcdf_file(path, "r", mmap=False) as ds:
            day = _file_day(ds)
            raw = np.asarray(ds.variables["CO2cool"].data, dtype=float)
            scans = raw.shape[0]
            cooling = orient(raw, scans)
            altitude = np.asarray(ds.variables["altitude"].data, dtype=float)
            latitude = orient(
                np.asarray(ds.variables["tplatitude"].data, dtype=float), scans
            )
            longitude = (
                orient(np.asarray(ds.variables["tplongitude"].data, dtype=float), scans)
                % 360.0
            )
            milliseconds = orient(
                np.asarray(ds.variables["time"].data, dtype=float), scans
            )
            orbits = np.asarray(ds.variables["orbit"].data, dtype=int)
        altitudes = (
            np.broadcast_to(altitude, cooling.shape)
            if altitude.ndim == 1
            else orient(altitude, scans)
        )
        slots = slot(day, milliseconds)
        flat_slot = np.array(
            [index_by_slot.get(value, -1) for value in slots.ravel()]
        ).reshape(slots.shape)
        center = np.where(flat_slot >= 0, centers[np.maximum(flat_slot, 0)], np.nan)
        geo_time = (
            np.isfinite(altitudes)
            & np.isfinite(latitude)
            & np.isfinite(longitude)
            & np.isfinite(milliseconds)
            & (flat_slot >= 0)
        )
        finite_value = geo_time & np.isfinite(cooling) & (cooling > MISSING_VALUE_LIMIT)
        for stratum_index, (lower, upper) in enumerate(STRATA):
            altitude_mask = (altitudes >= lower) & (
                (altitudes < upper) if stratum_index < 2 else (altitudes <= upper)
            )
            for name, (lat_width, lon_width) in CELLS.items():
                spatial = (np.abs(latitude - LATITUDE_CENTER) <= lat_width / 2) & (
                    circular_delta(longitude, center) <= lon_width / 2
                )
                footprint = geo_time & altitude_mask & spatial
                for scan_index in np.flatnonzero(footprint.any(axis=1)):
                    if orbits[scan_index] >= 0:
                        required[name].add((day.isoformat(), int(orbits[scan_index])))
                accepted = finite_value & altitude_mask & spatial
                positions = flat_slot[accepted]
                np.add.at(sums[name][stratum_index], positions, cooling[accepted])
                np.add.at(counts[name][stratum_index], positions, 1)
        if file_index % 500 == 0:
            print(f"processed {file_index}/{len(files)} CO2 daily files", flush=True)

    report: dict[str, object] = {
        "prototype": "THROWAWAY native SABER CO2 altitude-stratum spatial-binning feasibility check",
        "runtime_seconds": time.monotonic() - started,
        "inputs": {
            "official_local_co2_daily_files": len(files),
            "end_inclusive": END.isoformat(),
            "hasdm_latitude_center_deg": LATITUDE_CENTER,
            "hasdm_longitude_source": str(SAMPLES),
            "baseline": str(BASELINE),
        },
        "strata_km": [
            {"lower_inclusive": low, "upper": high, "upper_inclusive": index == 2}
            for index, (low, high) in enumerate(STRATA)
        ],
        "cells": {},
    }
    local_selected = len(list(L2A_DIR.glob("SABER_L2A_*_*.nc")))
    locally_resolved = {name: locally_resolved_pairs(required[name]) for name in CELLS}
    for name in CELLS:
        means = np.divide(
            sums[name],
            counts[name],
            out=np.full_like(sums[name], np.nan),
            where=counts[name] > 0,
        )
        bins = []
        for index, (lower, upper) in enumerate(STRATA):
            item = {
                "stratum_km": [lower, upper],
                **coverage(counts[name][index] > 0, calendar),
                "observation_count": int(counts[name][index].sum()),
                "slot_mean_summary": slot_mean_summary(means[index]),
            }
            bins.append(item)
        any_mask = np.any(counts[name] > 0, axis=0)
        report["cells"][name] = {
            "width_degrees": {"latitude": CELLS[name][0], "longitude": CELLS[name][1]},
            "altitude_bins": bins,
            "any_bin": {
                **coverage(any_mask, calendar),
                "observation_count": int(counts[name].sum()),
            },
            "required_unique_level2a_day_orbit_pairs": len(required[name]),
            "local_selected_level2a_files": local_selected,
            "required_pairs_resolved_by_current_local_level2a_files": len(
                locally_resolved[name]
            ),
        }
        if name == "expanded_20x30_deg":
            report["cells"][name]["additional_day_orbit_pairs_vs_existing_10x15"] = len(
                required[name] - required["existing_10x15_deg"]
            )
            report["cells"][name][
                "additional_level2a_files_needed_beyond_12461_local_selected_files"
            ] = len(required[name] - locally_resolved[name])
        report["cells"][name]["sparse_lag_feasibility_diagnostic"] = (
            sparse_lag_diagnostic([counts[name][i] > 0 for i in range(3)])
        )
    baseline_bins = []
    for altitude in (100, 119, 139):
        column, observation_column = (
            f"saber_co2cool_{altitude}km_w_m3",
            f"saber_co2cool_{altitude}km_w_m3_observations",
        )
        mask = np.isfinite(baseline[column].to_numpy(float))
        baseline_bins.append(
            {
                "nearest_level_km": altitude,
                **coverage(mask, calendar),
                "observation_count": int(baseline[observation_column].sum()),
            }
        )
    baseline_mask = np.any(
        [
            np.isfinite(baseline[f"saber_co2cool_{altitude}km_w_m3"].to_numpy(float))
            for altitude in (100, 119, 139)
        ],
        axis=0,
    )
    report["decoded_nearest_level_baseline"] = {
        "altitude_levels": baseline_bins,
        "any_bin": coverage(baseline_mask, calendar),
    }
    report["limitations"] = [
        "CO2cool only: this intentionally excludes NO and Level2A proxy products.",
        "The sparse-lag output is a seeded coverage diagnostic, not PCMCI execution or a completion proof.",
        "The additional Level2A-file count uses the decoder's same-day/adjacent-day local-orbit lookup; remote inventory is needed to name and acquire absent files.",
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUTPUT} in {report['runtime_seconds']:.1f}s")


if __name__ == "__main__":
    main()
