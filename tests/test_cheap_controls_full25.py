"""Synthetic checks of the cheap-control study code exactly as run on 2026-10-10."""

import hashlib
from pathlib import Path

import numpy as np

import scripts.cheap_controls_full25 as cc

AS_RUN_SHA256 = "fd07f4b72675c14a89f9629841781f5b8b329e75d643cb2f988296316e1dcc37"


def test_committed_file_is_byte_identical_to_the_run():
    path = Path(cc.__file__)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == AS_RUN_SHA256


def test_natural_spline_has_eleven_columns_and_is_linear_beyond_boundary_knots():
    t = np.linspace(-3.0, 25.0, 281)
    basis = cc.natural_spline(t)
    assert basis.shape == (len(t), 11)
    np.testing.assert_allclose(basis[:, 0], t)  # first column is time itself
    beyond = t > cc.SPLINE_KNOTS[-1]
    second_difference = np.diff(basis[beyond], n=2, axis=0)
    np.testing.assert_allclose(second_difference, 0.0, atol=1e-6 * np.abs(basis).max())


def test_cells_add_the_declared_time_and_co2_columns():
    n = 5
    static = np.random.default_rng(0).random((n, 28)).astype(np.float32)
    data = {"static": static, "t": np.linspace(1.0, 20.0, n)}
    record = {"placebo_coefficients": np.zeros((5, 2)).tolist()}
    width = {cell: cc.features(cell, data, record).shape[1] for cell in cc.CELLS}
    base = len(cc.BASE)
    assert width == {"N0": base, "N1": base + 2, "A": base + 1, "B": base + 3, "C": base + 11,
                     "D": base + 13, "P": base + 3, "P2": base + 2}


def test_daily_sums_match_a_direct_mape_calculation():
    obs = np.array([1.0, 2.0, 4.0, 1.0])
    data = {"jb08": np.ones(4), "obs": obs, "r": np.log(obs),
            "day": np.array(["2010-01-01", "2010-01-01", "2010-01-02", "2010-01-02"], dtype="datetime64[D]"),
            "altitude": np.array([410.0, 420.0, 430.0, 440.0]), "ap": np.array([5.0, 5.0, 20.0, 5.0])}
    r_hat = np.log(np.array([1.1, 1.8, 4.4, 1.0]))
    days, sums, counts = cc.daily_sums(data, r_hat)
    assert days == ["2010-01-01", "2010-01-02"]
    ape = np.abs(np.exp(r_hat) - obs) / obs
    np.testing.assert_allclose(sums["ape"][0], [ape[:2].sum(), ape[2:].sum()])
    assert counts[0].tolist() == [2, 2]
    quiet = cc.CATEGORIES.index("Quiet (Ap 0-15)")
    assert counts[quiet].tolist() == [2, 1]
