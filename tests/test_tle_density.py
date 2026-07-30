from datetime import timezone
import numpy as np

from thermodense.tle_density import (
    TLERecord,
    density_ratios,
    filter_tles_by_epoch_gap,
    parse_satellite_list,
    parse_tle_file,
    tle_epoch,
)

LINE1_A = "1 39135U 13015F   19080.83260080  .00000626  00000-0  41128-4 0  9990"
LINE2_A = "2 39135  64.8689 325.3478 0009985 299.1213  60.8913 15.17270395326765"
LINE1_B = "1 39135U 13015F   19081.88719855  .00000706  00000-0  45079-4 0  9993"
LINE2_B = "2 39135  64.8687 321.9469 0010026 299.9445  60.0685 15.17272717326924"
LINE1_C = "1 39135U 13015F   19084.85325034  .00000586  00000-0  39127-4 0  9998"
LINE2_C = "2 39135  64.8680 312.3821 0010151 302.2741  57.7401 15.17275420327215"


def test_parse_satellite_list_reads_id_and_ballistic_coefficient(tmp_path):
    path = tmp_path / "SAT_list_ALL.txt"
    path.write_text("39136\t0.0330\n\n# ignored\n39135 0.0471\n")

    configs = parse_satellite_list(path)

    assert [config.satellite_id for config in configs] == ["39136", "39135"]
    assert configs[0].ballistic_coefficient_m2_kg == 0.0330


def test_parse_tle_file_extracts_epochs_and_mean_motion(tmp_path):
    path = tmp_path / "39135.txt"
    path.write_text(f"{LINE1_B}\n{LINE2_B}\n{LINE1_A}\n{LINE2_A}\n")

    records = parse_tle_file(path)

    assert [record.line1 for record in records] == [LINE1_A, LINE1_B]
    assert records[0].epoch.year == 2019
    assert records[0].epoch.tzinfo == timezone.utc
    assert records[0].mean_motion_rev_day == 15.17270395


def test_tle_epoch_converts_day_of_year_fraction_to_utc_datetime():
    epoch = tle_epoch(LINE1_A)

    assert epoch.isoformat().startswith("2019-03-21T19:58:56.709")


def test_filter_tles_by_epoch_gap_keeps_successive_records_at_least_three_days_apart():
    records = [
        TLERecord(LINE1_A, LINE2_A, tle_epoch(LINE1_A), 15.0),
        TLERecord(LINE1_B, LINE2_B, tle_epoch(LINE1_B), 15.1),
        TLERecord(LINE1_C, LINE2_C, tle_epoch(LINE1_C), 15.2),
    ]

    filtered = filter_tles_by_epoch_gap(records, min_days=3)

    assert [record.line1 for record in filtered] == [LINE1_A, LINE1_C]


def test_density_ratio_sign_is_model_over_observed():
    ratios = density_ratios(
        np.array([2.0, 2.0]),
        {
            "0": np.array([4.0, 1.0]),
            "2.0": np.array([2.0, np.e * 2.0]),
        },
    )

    np.testing.assert_allclose(
        ratios["ln_density_ratio_0"], [np.log(2.0), -np.log(2.0)]
    )
    np.testing.assert_allclose(ratios["ln_density_ratio_2.0"], [0.0, 1.0])
