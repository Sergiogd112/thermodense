"""Tests for the CO2-only expectation built from Brown et al. (2024) scaling factors."""

import numpy as np
import pytest

from scripts.co2_only_expected_trends import ScalingGrid


def synthetic_grid(slope_points: int = 4) -> ScalingGrid:
    """ln R = s(z, F) * (CO2 - 368.865) with s = -0.004 at 70 sfu, -0.002 at 200 sfu, x z/400."""
    grid = object.__new__(ScalingGrid)
    grid.alt = np.array([200.0, 400.0, 600.0])
    grid.co2 = np.array([368.865, 378.8, 401.6, 431.5, 467.9])
    grid.f107 = np.array([70.0, 200.0])
    s = np.array([-0.004, -0.002])
    grid.ln_r = np.einsum("a,c,f->acf", grid.alt / 400.0, grid.co2 - grid.co2[0], s)
    x = grid.co2[:slope_points] - grid.co2[0]
    grid.slope = np.einsum("c,acf->af", x, grid.ln_r[:, :slope_points]) / (x @ x)
    grid.slope_points = slope_points
    return grid


def test_ln_ratio_recovers_linear_sensitivity_and_interpolates():
    grid = synthetic_grid()
    co2 = np.array([340.0, 368.865, 400.0])
    out = grid.ln_ratio(400.0, co2, np.array([70.0, 70.0, 200.0]))
    np.testing.assert_allclose(out, [-0.004 * (340 - 368.865), 0.0, -0.002 * (400 - 368.865)])
    mid = grid.ln_ratio(300.0, np.array([400.0]), np.array([135.0]))
    np.testing.assert_allclose(mid, 0.75 * -0.003 * (400 - 368.865))


def test_ln_ratio_clamps_f107_and_rejects_out_of_range_inputs():
    grid = synthetic_grid()
    low = grid.ln_ratio(400.0, np.array([400.0]), np.array([40.0]))
    np.testing.assert_allclose(low, -0.004 * (400 - 368.865))
    with pytest.raises(ValueError, match="altitude"):
        grid.ln_ratio(150.0, np.array([400.0]), np.array([100.0]))
    with pytest.raises(ValueError, match="historical range"):
        grid.ln_ratio(400.0, np.array([450.0]), np.array([100.0]))
