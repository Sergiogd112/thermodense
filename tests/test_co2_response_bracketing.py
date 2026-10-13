"""Tests for the CO2-response bracketing (scaled Brown et al. 2024 response)."""

from datetime import date, timedelta

import numpy as np
import polars as pl

from scripts.co2_response_bracketing import Bracket, pct
from scripts.recreate_brown_density_trend_current import F107_COL, SeriesSpec, fit_trend


def frame(seed: int = 0) -> pl.DataFrame:
    days = [date(1990, 1, 1) + timedelta(days=i) for i in range(365 * 20)]
    t = np.arange(len(days)) / 365.25
    f107 = 120 + 50 * np.sin(2 * np.pi * t / 11)
    rng = np.random.default_rng(seed)
    s = -0.0025 * t + 0.002 * np.sin(2 * np.pi * t / 11)  # CO2-only signal with a solar dependence
    y = 1.3 * s - 0.0004 * t + np.cumsum(rng.normal(0, 0.002, len(days))) * 0.05 + rng.normal(0, 0.02, len(days))
    return pl.DataFrame({"date": days, F107_COL: f107, "y": y, "s": s})


def test_any_scaling_matches_a_direct_refit():
    df = frame()
    br = Bracket("synthetic", 400.0, df)
    for k in (0.37, 1.5, 3.2):
        direct = fit_trend(SeriesSpec("synthetic", 400.0, "date", "r",
                                      df.with_columns((pl.col("y") - k * pl.col("s")).alias("r"))))
        mid, lo, hi = br.at(k)
        np.testing.assert_allclose(
            [mid, lo, hi],
            [direct["trend_percent_per_decade"], direct["trend_percent_per_decade_hac_95_ci_lower"],
             direct["trend_percent_per_decade_hac_95_ci_upper"]], rtol=1e-6, atol=1e-9)


def test_k_star_zeroes_the_trend_and_its_range_brackets_it():
    br = Bracket("synthetic", 400.0, frame(1))
    k_star, k_lo, k_hi = br.k_star()
    assert abs(br.at(k_star)[0]) < 1e-9
    assert k_lo < k_star < k_hi
    for k in (k_lo, k_hi):  # interval edges touch zero
        _, lo, hi = br.at(k)
        assert min(abs(lo), abs(hi)) < 0.01
    assert pct(0.0) == 0.0
