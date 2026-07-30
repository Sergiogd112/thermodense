"""Focused tests for HAC uncertainty calculations in the trend figure script."""

import numpy as np
import pytest

from scripts.recreate_brown_density_trend_current import (
    log10_slope_ci_to_percent_per_decade,
    newey_west_covariance,
)


def test_newey_west_covariance_uses_bartlett_weighted_calendar_day_scores():
    design = np.ones((3, 1))
    residuals = np.array([1.0, 2.0, 3.0])
    day_index = np.array([10, 11, 12])

    covariance = newey_west_covariance(design, residuals, day_index, max_lag=1)

    # Score meat = 1^2 + 2^2 + 3^2 + 0.5 * 2 * (2*1 + 3*2) = 22.
    np.testing.assert_allclose(covariance, [[22.0 / 9.0]])


def test_newey_west_covariance_aggregates_duplicate_dates_before_lagging():
    design = np.ones((3, 1))
    residuals = np.array([1.0, 2.0, 3.0])

    covariance = newey_west_covariance(
        design, residuals, np.array([10, 10, 11]), max_lag=1
    )

    # Daily scores are [1 + 2, 3]; meat = 3^2 + 3^2 + 0.5 * 2 * 3 * 3 = 27.
    np.testing.assert_allclose(covariance, [[27.0 / 9.0]])


def test_newey_west_covariance_keeps_missing_dates_between_calendar_lags():
    design = np.ones((2, 1))
    residuals = np.array([1.0, 2.0])

    covariance = newey_west_covariance(design, residuals, np.array([10, 12]), max_lag=1)

    # Complete daily scores are [1, 0, 2], so the one-day lag cross-product is 0.
    np.testing.assert_allclose(covariance, [[5.0 / 4.0]])


def test_log10_slope_interval_is_transformed_endpoint_by_endpoint():
    lower, upper = log10_slope_ci_to_percent_per_decade(-0.01, 0.02)

    assert lower == pytest.approx((10**-0.1 - 1.0) * 100.0)
    assert upper == pytest.approx((10**0.2 - 1.0) * 100.0)
    assert upper - 0.0 != pytest.approx(0.0 - lower)


def test_log10_slope_interval_rejects_reversed_endpoints():
    with pytest.raises(ValueError, match="lower log10 slope"):
        log10_slope_ci_to_percent_per_decade(0.01, -0.01)
