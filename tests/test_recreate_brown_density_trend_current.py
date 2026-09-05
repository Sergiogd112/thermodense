"""Focused tests for HAC uncertainty calculations in the trend figure script."""

import numpy as np
import polars as pl
import pytest

import scripts.recreate_brown_density_trend_current as brown
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


def test_collect_trends_preflight_uses_mission_parquets_not_tudelft_sentinel(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    input_paths = [tmp_path / f"input-{index}.parquet" for index in range(12)]
    for path in input_paths:
        path.touch()
    missions = tuple(
        (f"mission-{index}", path, (0.0, 1.0))
        for index, path in enumerate(input_paths[2:10])
    )
    monkeypatch.setattr(brown, "GLOBAL_MEAN_PATH", input_paths[0])
    monkeypatch.setattr(brown, "HASDM_PATH", input_paths[1])
    monkeypatch.setattr(brown, "TUDELFT_MISSIONS", missions)
    monkeypatch.setattr(brown, "MSIS_PATH", input_paths[10])
    monkeypatch.setattr(brown, "SPACE_WEATHER_PATH", input_paths[11])
    monkeypatch.setattr(brown, "TUDELFT_PATH", tmp_path / "unused-sentinel.parquet")
    monkeypatch.setattr(brown, "global_mean_specs", lambda: [object()])
    monkeypatch.setattr(brown, "long_altitude_specs", lambda *_: [])
    monkeypatch.setattr(brown, "tudelft_specs", lambda: [])
    monkeypatch.setattr(brown, "msis_specs", lambda: [])
    monkeypatch.setattr(
        brown, "fit_trend", lambda _: {"dataset": "test", "altitude_km": 1.0}
    )

    trends = brown.collect_trends(jb_path=tmp_path / "missing-jb.parquet")

    assert trends.height == 1
    assert brown.TUDELFT_PATH not in brown.required_input_paths()
    assert tuple(path for _, path, _ in missions) == brown.required_input_paths()[2:10]


def test_require_jb_preflight_requires_external_parquet(tmp_path):
    with pytest.raises(FileNotFoundError, match="maunaloa_jb"):
        brown.collect_trends(require_jb=True, jb_path=tmp_path / "maunaloa_jb.parquet")


def test_jb_specs_requires_paired_identical_altitudes_and_solar_controls(tmp_path):
    path = tmp_path / "jb.parquet"
    pl.DataFrame(
        {
            "date": ["2020-01-01"],
            brown.F107_COL: [100.0],
            "jb2006_log10rho_daily_mean_400km": [-12.0],
            "jb2008_log10rho_daily_mean_400km": [-12.1],
        }
    ).write_parquet(path)

    specs = brown.jb_specs(path)

    assert [(spec.dataset, spec.altitude_km) for spec in specs] == [
        ("JB2006 Mauna Loa baseline", 400.0),
        ("JB2008 Mauna Loa baseline", 400.0),
    ]


def test_jb_specs_requires_the_solar_control_column(tmp_path):
    path = tmp_path / "jb.parquet"
    pl.DataFrame(
        {
            "date": ["2020-01-01"],
            "jb2006_log10rho_daily_mean_400km": [-12.0],
            "jb2008_log10rho_daily_mean_400km": [-12.1],
        }
    ).write_parquet(path)

    with pytest.raises(ValueError, match=brown.F107_COL):
        brown.jb_specs(path)


@pytest.mark.parametrize(
    "columns, message",
    [
        (["jb2006_log10rho_daily_mean_400km"], "paired JB2006/JB2008"),
        (
            [
                "jb2006_log10rho_daily_mean_400km",
                "jb2008_log10rho_daily_mean_425km",
            ],
            "identical altitude sets",
        ),
    ],
)
def test_jb_specs_rejects_singleton_or_mismatched_altitudes(tmp_path, columns, message):
    path = tmp_path / "jb.parquet"
    data = {"date": ["2020-01-01"], brown.F107_COL: [100.0]}
    data.update({column: [-12.0] for column in columns})
    pl.DataFrame(data).write_parquet(path)

    with pytest.raises(ValueError, match=message):
        brown.jb_specs(path)


def test_plot_trends_styles_both_jb_baselines():
    fig, ax = brown.plt.subplots()
    trends = pl.DataFrame(
        {
            "dataset": ["JB2006 Mauna Loa baseline", "JB2008 Mauna Loa baseline"],
            "altitude_km": [400.0, 400.0],
            "trend_percent_per_decade": [-2.0, -3.0],
            "trend_percent_per_decade_hac_95_ci_lower": [-3.0, -4.0],
            "trend_percent_per_decade_hac_95_ci_upper": [-1.0, -2.0],
            "duration_bin_years": [11, 11],
        }
    )

    handles, _ = brown.plot_trends(ax, trends)

    assert [handle.get_label() for handle in handles] == ["JB2006", "JB2008"]
    assert handles[0].get_linestyle() != handles[1].get_linestyle()
    brown.plt.close(fig)


def test_parse_args_preserves_no_argument_defaults_and_accepts_jb_contract():
    assert brown.parse_args([]).require_jb is False
    args = brown.parse_args(["--require-jb", "--jb-path", "external.parquet"])
    assert args.require_jb is True
    assert args.jb_path.name == "external.parquet"


def test_plot_trends_reuses_current_renderer_on_supplied_axes():
    fig, ax = brown.plt.subplots()
    trends = pl.DataFrame(
        {
            "dataset": ["Global mean thermospheric density"],
            "altitude_km": [400.0],
            "trend_percent_per_decade": [-5.0],
            "trend_percent_per_decade_hac_95_ci_lower": [-7.0],
            "trend_percent_per_decade_hac_95_ci_upper": [-3.0],
            "duration_bin_years": [11],
        }
    )

    dataset_handles, duration_handles = brown.plot_trends(ax, trends)

    assert [handle.get_label() for handle in dataset_handles] == ["Gbl. Mean"]
    assert [handle.get_label() for handle in duration_handles] == ["11 to 22 yr"]
    assert ax.get_xlabel() == "Solar-adjusted density trend (%/dec)"
    assert ax.get_ylim()[0] == 150.0
    brown.plt.close(fig)


def test_plot_trends_accepts_shared_altitude_limits():
    fig, ax = brown.plt.subplots()
    trends = pl.DataFrame(
        {
            "dataset": ["Global mean thermospheric density"],
            "altitude_km": [400.0],
            "trend_percent_per_decade": [-5.0],
            "trend_percent_per_decade_hac_95_ci_lower": [-7.0],
            "trend_percent_per_decade_hac_95_ci_upper": [-3.0],
            "duration_bin_years": [11],
        }
    )

    brown.plot_trends(ax, trends, altitude_limits=(0.0, 850.0))

    assert ax.get_ylim() == (0.0, 850.0)
    brown.plt.close(fig)
