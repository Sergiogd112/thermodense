"""Tests for the WACCM-X CO2 tracer-age analysis helpers (synthetic data only)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "co2_lag_analysis", Path(__file__).resolve().parents[1] / "scripts/waccmx/co2_lag_analysis.py"
)
lag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lag)


def monthly_axis(first="1950-01", last="2024-12", skip_year=2022):
    keys = np.arange(np.datetime64(first), np.datetime64(last) + 1)
    keys = keys[keys.astype("datetime64[Y]").astype(int) + 1970 != skip_year]
    t = keys.astype(int) // 12 + 1970 + (keys.astype(int) % 12 + 0.5) / 12
    return keys, t


@pytest.mark.parametrize("delay", [1.0, 3.0, 4.2])
def test_tracer_age_recovers_known_delay_with_season_and_2022_gap(delay):
    keys, t = monthly_axis()
    surface = lambda x: 310e-6 + 0.8e-6 * (x - 1950) + 0.012e-6 * (x - 1950) ** 2
    season = lambda x: 3e-6 * np.sin(2 * np.pi * x)
    s12 = lag.deseasonalize(keys, surface(t) + season(t))
    c12 = lag.deseasonalize(keys, surface(t - delay) + 0.5 * season(t - delay))
    good = np.isfinite(s12)
    age = lag.tracer_age(t, c12, t[good], s12[good])
    assert np.nanmean(age[t > 1965]) == pytest.approx(delay, abs=1e-3)
    gap = (keys >= np.datetime64("2021-07")) & (keys < np.datetime64("2023-07"))
    assert np.isnan(s12[gap]).all()  # the running mean never bridges 2022


def test_block_starts_never_cross_the_2022_gap_and_include_the_last_start():
    keys, t = monthly_axis("2018-01", "2025-12")
    values = np.ones(len(t))
    starts = lag.block_starts(t, values, block=24)
    for s in starts:
        assert np.all(np.abs(np.diff(t[s:s + 24]) - 1 / 12) < 1e-6)
    assert starts.max() == len(t) - 24  # last valid start is drawable
    before_gap = int(np.flatnonzero(keys == np.datetime64("2021-12"))[0])
    assert not any(s <= before_gap < s + 23 for s in starts)


def test_block_starts_skip_nan_months_and_fail_when_no_block_fits():
    _, t = monthly_axis("2000-01", "2004-12")
    values = np.ones(len(t))
    values[30] = np.nan
    assert all(not (s <= 30 < s + 24) for s in lag.block_starts(t, values, block=24))
    with pytest.raises(ValueError):
        lag.block_starts(t[:20], values[:20], block=24)


def test_supported_months_drop_a_segment_too_short_for_any_block():
    keys, t = monthly_axis("2006-01", "2024-12")
    values = np.ones(len(t))
    after_gap = keys >= np.datetime64("2023-01")
    values[after_gap & (keys < np.datetime64("2023-07"))] = np.nan  # leave 18 finite months
    keep = lag.supported_months(t, values, block=24)
    assert keep[keys < np.datetime64("2022-01")].all()
    assert not keep[after_gap].any()
