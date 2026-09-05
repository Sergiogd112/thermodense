"""Focused contracts for the standalone direct-density PCMCI+ runner."""

import json

import numpy as np
import pandas as pd
import pytest

import scripts.prototype_density_pcmci_3hour_and_daily as runner


def test_case_expansion_and_physical_lag_conversion():
    assert [case["id"] for case in runner.cases()] == [
        "hasdm_all-ap",
        "hasdm_all-kp",
        "hasdm_selected-ap",
        "hasdm_selected-kp",
        "global_mean-ap",
        "global_mean-kp",
    ]
    assert runner.lag_steps("hasdm") == 488
    assert runner.lag_steps("global_mean") == 183
    assert runner.target_indices(runner.case_by_id("hasdm_selected-ap")) == (
        0,
        13,
        26,
    )
    assert runner.altitudes(runner.case_by_id("hasdm_all-ap")) == tuple(
        range(175, 826, 25)
    )
    assert runner.altitudes(runner.case_by_id("hasdm_selected-kp")) == (175, 500, 825)
    assert len(runner.saber_columns()) == 15
    assert runner.driver_count(runner.case_by_id("hasdm_all-ap")) == 17
    assert runner.driver_count(runner.case_by_id("global_mean-ap")) == 2


def test_complete_hasdm_calendar_is_fixed_and_has_expected_size():
    calendar = runner.complete_hasdm_calendar()
    assert calendar[0] == pd.Timestamp("2000-01-01")
    assert calendar[-1] == pd.Timestamp("2025-07-20")
    assert len(calendar) == 74657
    assert (calendar[1:] - calendar[:-1]).unique().tolist() == [pd.Timedelta(hours=3)]


def test_global_mean_daily_contract_keeps_direct_finite_channels(monkeypatch):
    dates = pd.date_range(runner.GLOBAL_START, runner.GLOBAL_END, freq="D")
    source = pd.DataFrame(
        {
            "date": dates,
            **{
                f"log10rho_{altitude}": np.full(len(dates), float(index))
                for index, altitude in enumerate(runner.GLOBAL_ALTITUDES)
            },
        }
    )
    weather = pd.DataFrame(
        {
            "DATE": dates,
            "F10.7_OBS": 70.0,
            "AP_AVG": 3.0,
            "KP_SUM": 24.0,
        }
    )
    monkeypatch.setattr(runner.pd, "read_parquet", lambda _path: source)
    arrays, note = runner.prepare_global(weather)
    assert arrays["global_targets"].shape == (19328, 10)
    assert np.array_equal(arrays["global_targets"][:, 0], np.zeros(19328))
    assert note["cadence"] == "daily (not upsampled to 3-hour)"
    assert note["missing_target_values"] == 0


def test_forcing_alignment_preserves_f107_knots_and_native_slots():
    weather = pd.DataFrame(
        {
            "DATE": pd.date_range("2020-01-02", periods=4),
            "F10.7_OBS": [70.0, 80.0, 75.0, 90.0],
            **{f"AP{i}": [i] * 4 for i in range(1, 9)},
            **{f"KP{i}": [10 + i] * 4 for i in range(1, 9)},
        }
    )
    times = pd.DatetimeIndex(
        ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-05", "2020-01-06"]
    )
    f107 = runner.f107_spline(times, weather)
    assert np.isnan(f107[0]) and np.isnan(f107[-1])
    assert f107[1] == 70.0 and f107[3] == 90.0
    slots = pd.date_range("2020-01-02", periods=8, freq="3h")
    ap, kp = runner.native_slots(slots, weather)
    assert ap.tolist() == list(range(1, 9))
    assert kp.tolist() == list(range(11, 19))


def test_preprocess_uses_month_day_slot_leap_keys_and_preserves_missingness():
    times = pd.DatetimeIndex(
        ["2020-02-28", "2020-02-29", "2020-03-01", "2021-02-28", "2021-03-01"]
    )
    result = runner.preprocess(np.array([1.0, np.nan, 3.0, 4.0, 5.0]), times, 1)
    assert np.isnan(result[1])
    assert np.isfinite(result[[0, 2, 3, 4]]).all()


def test_sparse_saber_preprocessing_preserves_missingness():
    times = pd.date_range("2020-01-01", periods=16, freq="3h")
    raw = np.arange(16, dtype=float)
    raw[[2, 9]] = np.nan
    transformed = runner.preprocess(raw, times, 8)
    assert np.isnan(transformed[[2, 9]]).all()


def test_prepare_hasdm_refuses_missing_saber_source(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "SABER", tmp_path / "missing.parquet")
    with pytest.raises(FileNotFoundError, match="SABER 3-hour"):
        runner.prepare_hasdm(pd.DataFrame())


def test_structural_assumptions_allow_only_driver_target_and_target_self_history():
    columns = ["log10rho_175", "log10rho_200", "f107", "ap"]
    assumptions = runner.link_assumptions(columns, columns[:2], 2)
    assert (2, 0) in assumptions[0] and (0, -1) in assumptions[0]
    assert (0, -1) not in assumptions[1]  # no target peer -> target link
    assert (0, -1) not in assumptions[2]  # no target -> driver link


def test_fdr_family_counts_and_bh_is_exact_declared_family():
    assert runner.expected_fdr_tests(runner.case_by_id("hasdm_all-ap")) == 17 * 27 * 489
    assert (
        runner.expected_fdr_tests(runner.case_by_id("hasdm_selected-kp"))
        == 17 * 3 * 489
    )
    assert (
        runner.expected_fdr_tests(runner.case_by_id("global_mean-ap")) == 2 * 10 * 184
    )
    assert np.allclose(
        runner.bh_qvalues(np.array([0.01, 0.02, 0.03])), [0.03, 0.03, 0.03]
    )


def test_fingerprint_reuse_rejects_changed_bundle(tmp_path, monkeypatch):
    case = runner.case_by_id("hasdm_all-ap")
    first = runner.fingerprint(case, "a")
    assert first == runner.fingerprint(case, "a")
    assert first != runner.fingerprint(case, "b")
    directory = tmp_path / "cases" / case["id"]
    directory.mkdir(parents=True)
    (directory / "provenance.json").write_text(json.dumps({"fingerprint": first}))
    monkeypatch.setattr(
        runner,
        "load_case_bundle",
        lambda *_args: (
            pd.DatetimeIndex([]),
            np.empty((0, 27)),
            np.array([]),
            np.array([]),
            np.empty((0, 15)),
            "b",
        ),
    )
    with pytest.raises(FileExistsError, match="incompatible"):
        runner.run(case, tmp_path, "test")


def test_run_reuse_requires_intact_production_artifact(tmp_path, monkeypatch):
    case, bundle_hash = runner.case_by_id("global_mean-ap"), "bundle"
    identity = runner.fingerprint(case, bundle_hash)
    directory = tmp_path / "cases" / case["id"]
    directory.mkdir(parents=True)
    tests, retained = (
        directory / "driver_target_tests.csv",
        directory / "retained_links.csv",
    )
    tests.write_text("test\n" + "row\n" * runner.expected_fdr_tests(case))
    retained.write_text("link\n")
    (directory / "provenance.json").write_text(
        json.dumps(
            {
                "production": True,
                "case": case,
                "bundle_sha256": bundle_hash,
                "fingerprint": identity,
                "fdr_family": {"test_count": runner.expected_fdr_tests(case)},
                "result_files": {
                    tests.name: runner.sha256(tests),
                    retained.name: runner.sha256(retained),
                },
            }
        )
    )
    monkeypatch.setattr(
        runner,
        "load_case_bundle",
        lambda *_args: (
            pd.DatetimeIndex([]),
            np.empty((0, 10)),
            np.array([]),
            np.array([]),
            np.empty((0, 0)),
            bundle_hash,
        ),
    )
    assert runner.run(case, tmp_path, "test") == directory
    tests.write_text(tests.read_text() + "altered\n")
    with pytest.raises(FileExistsError, match="incompatible"):
        runner.run(case, tmp_path, "test")


def test_summarize_rejects_incomplete_or_nonproduction_artifacts(tmp_path):
    with pytest.raises(ValueError, match="incomplete"):
        runner.summarize(tmp_path)
    for case in runner.cases():
        directory = tmp_path / "cases" / case["id"]
        directory.mkdir(parents=True)
        (directory / "provenance.json").write_text(json.dumps({"production": False}))
        (directory / "driver_target_tests.csv").write_text("")
        (directory / "retained_links.csv").write_text("")
    with pytest.raises(ValueError, match="non-production"):
        runner.summarize(tmp_path)
