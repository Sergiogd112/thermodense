from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from scripts.generate_maunaloa_msis_full_history import (
    ALTITUDES_KM,
    HISTORY_START,
    MODEL_VERSIONS,
    REQUIRED_COLUMNS,
    aggregate_chunk,
    build_drivers,
    contiguous_coverage_end,
    daily_columns,
    generate,
    hourly_timestamps,
    validate_requested_dates,
)


def make_drivers(start: date, end: date) -> dict[date, dict[str, float]]:
    drivers = {}
    current = start
    while current <= end:
        offset = (current - start).days
        drivers[current] = {
            "F10.7_OBS": 100.0 + offset,
            "F10.7_OBS_CENTER81": 200.0 + offset,
            "AP_AVG": 300.0 + offset,
            **{f"AP{index}": 1000.0 * offset + index for index in range(1, 9)},
        }
        current += timedelta(days=1)
    return drivers


def test_build_drivers_matches_pymsis_lag_definition_across_year_boundary() -> None:
    drivers = make_drivers(date(1966, 12, 28), date(1967, 1, 2))
    timestamps = np.array(["1967-01-01T00", "1967-01-01T01"], dtype="datetime64[h]")

    f107s, f107as, aps = build_drivers(timestamps, drivers)

    assert f107s.tolist() == [drivers[date(1966, 12, 31)]["F10.7_OBS"]] * 2
    assert f107as.tolist() == [drivers[date(1967, 1, 1)]["F10.7_OBS_CENTER81"]] * 2
    expected_bins = []
    for hours_before in range(0, 58, 3):
        timestamp = datetime(1967, 1, 1) - timedelta(hours=hours_before)
        # Midnight is sufficient here; the first two timestamps share AP1.
        expected_bins.append(
            drivers[timestamp.date()][f"AP{timestamp.hour // 3 + 1}"]
        )
    assert aps[0, 0] == drivers[date(1967, 1, 1)]["AP_AVG"]
    assert aps[0, 1:5].tolist() == expected_bins[:4]
    assert aps[0, 5] == pytest.approx(np.mean(expected_bins[4:12]))
    assert aps[0, 6] == pytest.approx(np.mean(expected_bins[12:20]))
    assert aps[1].tolist() == aps[0].tolist()


def test_build_drivers_reproduces_pymsis_radio_burst_safeguard() -> None:
    drivers = make_drivers(date(1966, 12, 30), date(1967, 1, 2))
    drivers[date(1966, 12, 31)]["F10.7_OBS"] = 500.0

    f107s, _, _ = build_drivers(
        np.array(["1967-01-01T12"], dtype="datetime64[h]"), drivers
    )

    assert f107s.item() == drivers[date(1966, 12, 31)]["F10.7_OBS_CENTER81"]


def test_aggregate_chunk_has_frozen_schema_and_24_hour_statistics() -> None:
    timestamps = hourly_timestamps(date(1967, 1, 1), date(1967, 1, 1))
    f107as = np.full(24, 200.0)
    hourly_log_density = np.arange(24, dtype=float)[:, None] + np.arange(
        len(ALTITUDES_KM), dtype=float
    )
    densities = {
        model: 10.0**hourly_log_density for model in MODEL_VERSIONS
    }

    result = aggregate_chunk(timestamps, f107as, densities)

    assert ALTITUDES_KM == tuple(range(125, 826, 25))
    assert result.columns == daily_columns()
    assert result.height == 1
    assert result["nrlmsis_2p0_log10rho_daily_mean_125km"].item() == pytest.approx(11.5)
    assert result["nrlmsis_2p0_log10rho_daily_range_125km"].item() == pytest.approx(23.0)


def test_aggregate_chunk_rejects_non_positive_or_non_finite_density() -> None:
    timestamps = hourly_timestamps(date(1967, 1, 1), date(1967, 1, 1))
    densities = {
        model: np.ones((24, len(ALTITUDES_KM))) for model in MODEL_VERSIONS
    }
    densities["nrlmsise_00"][0, 0] = 0.0

    with pytest.raises(ValueError, match="non-finite or non-positive"):
        aggregate_chunk(timestamps, np.ones(24), densities)


def test_coverage_requires_a_contiguous_complete_source_span() -> None:
    drivers = make_drivers(date(1966, 12, 28), date(1967, 1, 4))
    del drivers[date(1967, 1, 3)]

    assert contiguous_coverage_end(drivers, HISTORY_START) == date(1967, 1, 2)
    with pytest.raises(ValueError, match="contiguous complete source coverage"):
        validate_requested_dates(drivers, date(1967, 1, 1), date(1967, 1, 3))


def test_generate_uses_mock_executor_and_writes_wide_daily_output(tmp_path: Path) -> None:
    drivers = make_drivers(date(1966, 12, 28), date(1967, 1, 3))
    source = tmp_path / "SW-All.csv"
    pl.DataFrame(
        [
            {"DATE": day.isoformat(), **{key: values[key] for key in REQUIRED_COLUMNS}}
            for day, values in drivers.items()
        ]
    ).write_csv(source)
    calls = []

    def executor(*args: object) -> np.ndarray:
        calls.append(args[-1])
        alts = np.asarray(args[3], dtype=float)
        return np.column_stack((1e-12 * (1.0 + alts / 1000.0), np.zeros(len(alts))))

    output = tmp_path / "daily.parquet"
    result, provenance = generate(
        date(1967, 1, 1), date(1967, 1, 2), output, source, executor
    )

    assert calls == list(MODEL_VERSIONS.values())
    assert result.height == 2
    assert result.columns == daily_columns()
    assert output.exists()
    assert output.with_suffix(".provenance.json").exists()
    assert provenance["counts"]["hourly_timestamps"] == 48
