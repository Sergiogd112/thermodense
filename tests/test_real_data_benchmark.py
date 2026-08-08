from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from thermodense.benchmarks import real_data


def test_bounded_gap_fill_only_fills_short_interior_runs() -> None:
    values, imputed = real_data.bounded_gap_fill(
        np.array(
            [np.nan, 0.0, np.nan, np.nan, 3.0, np.nan, np.nan, np.nan, 7.0, np.nan]
        ),
        max_gap=2,
    )

    assert np.allclose(values[1:5], [0.0, 1.0, 2.0, 3.0], equal_nan=True)
    assert np.isnan(values[[0, 5, 6, 7, 9]]).all()
    assert imputed.tolist() == [
        False,
        False,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    ]


def test_build_is_hasdm_anchored_and_records_imputation_masks(tmp_path: Path) -> None:
    decoded_dir = tmp_path / "hasdm"
    decoded_dir.mkdir()
    _write_hasdm(decoded_dir / "HASDM_fixture_merged.parquet")
    space_weather = tmp_path / "space_weather.csv"
    space_weather.write_text(
        "DATE,F10.7_OBS,F10.7_OBS_CENTER81,AP_AVG\n"
        "2019-12-31,2,1,1\n2020-01-01,20,10,1\n2020-01-02,30,20,2\n"
        "2020-01-03,40,30,3\n2020-01-04,50,40,4\n"
    )
    co2 = tmp_path / "co2.csv"
    co2.write_text(
        "2020,1,1,2020.0,400\n2020,1,2,2020.0,-99\n"
        "2020,1,3,2020.0,402\n2020,1,4,2020.0,403\n"
    )

    result = real_data.build_five_node_daily(decoded_dir, space_weather, co2)

    assert result["date"].to_list() == [
        datetime(2020, 1, 1).date(),
        datetime(2020, 1, 2).date(),
        datetime(2020, 1, 3).date(),
    ]
    assert result["co2_ppm"][0] == 400.0
    assert np.isnan(result["co2_ppm"][1])
    assert result["co2_ppm"][2] == 402.0
    assert result["co2_ppm_imputed"].to_list() == [False, False, False]
    assert np.allclose(
        result["log10rho_325_daily_mean"],
        [
            np.log10(1.0e-12),
            (np.log10(1.0e-12) + np.log10(5.0e-12)) / 2,
            np.log10(5.0e-12),
        ],
    )
    assert result["log10rho_325_daily_mean_imputed"].to_list() == [False, True, False]
    assert np.allclose(
        result["log10rho_825_daily_mean"],
        [
            np.log10(2.0e-12),
            (np.log10(2.0e-12) + np.log10(4.0e-12)) / 2,
            np.log10(4.0e-12),
        ],
    )
    assert result["log10rho_825_daily_mean_imputed"].to_list() == [False, True, False]
    assert all(column in result.columns for column in real_data.IMPUTATION_MASK_COLUMNS)
    assert real_data.daily_hasdm_for_path(
        decoded_dir / "HASDM_fixture_merged.parquet", 20.0
    )["altitude_km"].unique().sort().to_list() == [325.0, 825.0]


def test_load_space_weather_daily_rejects_missing_raw_f107_source(tmp_path: Path) -> None:
    path = tmp_path / "space_weather.csv"
    path.write_text("DATE,F10.7_OBS_CENTER81,AP_AVG\n2020-01-01,10,1\n")

    with pytest.raises(ValueError, match="F10.7_OBS"):
        real_data.load_space_weather_daily(path)


def test_describe_counts_nulls_and_nans_as_missing() -> None:
    frame = pl.DataFrame(
        {
            "date": [datetime(2020, 1, 1).date(), datetime(2020, 1, 2).date()],
            "f10_7_center81": [1.0, np.nan],
            "ap_avg": [1.0, None],
            "co2_ppm": [1.0, 2.0],
            "log10rho_325_daily_mean": [1.0, 2.0],
            "log10rho_825_daily_mean": [1.0, 2.0],
        }
    )

    description = real_data.describe(frame)

    assert "rows: 2 (complete rows: 1)" in description
    assert "f10_7_center81: 1/2 present; 0 imputed" in description
    assert "ap_avg: 1/2 present; 0 imputed" in description


def test_daily_density_target_is_mean_of_intraday_log10_values(tmp_path: Path) -> None:
    path = tmp_path / "HASDM_intraday_merged.parquet"
    pl.DataFrame(
        [
            {
                "timestamp": datetime(2020, 1, 1, hour),
                real_data.HASDM_LAT_COL: 20.0,
                real_data.HASDM_LON_COL: real_data.MAUNA_LOA_LON_EAST,
                real_data.HASDM_ALT_COL: 325_000.0,
                real_data.HASDM_DENSITY_COL: density,
            }
            for hour, density in [(0, 1.0e-12), (12, 1.0e-10)]
        ]
    ).write_parquet(path)

    result = real_data.daily_hasdm_for_path(path, 20.0)

    target = result["log10rho_daily_mean"].item()
    assert target == np.mean(np.log10([1.0e-12, 1.0e-10]))
    assert target != np.log10(np.mean([1.0e-12, 1.0e-10]))


def test_invalid_nearest_longitude_does_not_substitute_farther_valid_grid(
    tmp_path: Path,
) -> None:
    path = tmp_path / "HASDM_invalid_nearest_merged.parquet"
    pl.DataFrame(
        [
            {
                "timestamp": datetime(2020, 1, 1),
                real_data.HASDM_LAT_COL: 20.0,
                real_data.HASDM_LON_COL: real_data.MAUNA_LOA_LON_EAST,
                real_data.HASDM_ALT_COL: 325_000.0,
                real_data.HASDM_DENSITY_COL: 0.0,
            },
            {
                "timestamp": datetime(2020, 1, 1),
                real_data.HASDM_LAT_COL: 20.0,
                real_data.HASDM_LON_COL: real_data.MAUNA_LOA_LON_EAST + 1.0,
                real_data.HASDM_ALT_COL: 325_000.0,
                real_data.HASDM_DENSITY_COL: 1.0e-12,
            },
        ]
    ).write_parquet(path)

    result = real_data.daily_hasdm_for_path(path, 20.0)

    assert result.is_empty()


def _write_hasdm(path: Path) -> None:
    rows = []
    for day, density_325, density_825 in [
        (1, 1.0e-12, 2.0e-12),
        (3, 5.0e-12, 4.0e-12),
    ]:
        timestamp = datetime(2020, 1, day)
        rows.extend(
            [
                {
                    "timestamp": timestamp,
                    real_data.HASDM_LAT_COL: 20.0,
                    real_data.HASDM_LON_COL: real_data.MAUNA_LOA_LON_EAST + 20.0,
                    real_data.HASDM_ALT_COL: 325_000.0,
                    real_data.HASDM_DENSITY_COL: density_325 * 10,
                },
                {
                    "timestamp": timestamp,
                    real_data.HASDM_LAT_COL: 20.0,
                    real_data.HASDM_LON_COL: real_data.MAUNA_LOA_LON_EAST,
                    real_data.HASDM_ALT_COL: 325_000.0,
                    real_data.HASDM_DENSITY_COL: density_325,
                },
                {
                    "timestamp": timestamp,
                    real_data.HASDM_LAT_COL: 20.0,
                    real_data.HASDM_LON_COL: real_data.MAUNA_LOA_LON_EAST,
                    real_data.HASDM_ALT_COL: 825_000.0,
                    real_data.HASDM_DENSITY_COL: density_825,
                },
                {
                    "timestamp": timestamp,
                    real_data.HASDM_LAT_COL: 20.0,
                    real_data.HASDM_LON_COL: real_data.MAUNA_LOA_LON_EAST,
                    real_data.HASDM_ALT_COL: 500_000.0,
                    real_data.HASDM_DENSITY_COL: 9.0e-12,
                },
            ]
        )
    pl.DataFrame(rows).write_parquet(path)
