import json

import numpy as np
import pandas as pd
import pytest

import scripts.prototype_empirical_model_figure5_3hour as figure5
from scripts.prototype_empirical_model_figure5_3hour import (
    ALTITUDES,
    PHYSICAL_LAG_DAYS,
    cases,
    complete_3hour_calendar,
    declared_fdr_tests,
    driver_target_bh_qvalues,
    f107_spline,
    fingerprint,
    lag_steps,
    link_assumptions,
    preprocess_3hour,
    reuse_is_compatible,
    slot_drivers,
    verify_equivalence,
)


def test_physical_lag_is_explicitly_1464_three_hour_steps():
    assert PHYSICAL_LAG_DAYS == 183
    assert lag_steps() == 1464


def test_complete_calendar_exposes_missing_timestamp_without_interpolation():
    sample = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 06:00"]),
            "value": [1.0, 3.0],
        }
    )
    result = complete_3hour_calendar(sample)
    assert list(result.timestamp) == list(
        pd.date_range("2020-01-01", periods=3, freq="3h")
    )
    assert np.isnan(result.loc[1, "value"])


def test_complete_calendar_expands_each_altitude_at_missing_timestamp():
    sample = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2020-01-01", "2020-01-01", "2020-01-01 06:00", "2020-01-01 06:00"],
                format="mixed",
            ),
            "altitude_km": [175, 200, 175, 200],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    result = complete_3hour_calendar(sample)
    assert len(result) == 6
    assert (
        result.loc[result.timestamp.eq(pd.Timestamp("2020-01-01 03:00")), "value"]
        .isna()
        .all()
    )


def test_f107_spline_preserves_knots_and_forbids_extrapolation():
    weather = pd.DataFrame(
        {
            "DATE": pd.date_range("2020-01-02", periods=4),
            "F10.7_OBS": [70.0, 80.0, 75.0, 90.0],
        }
    )
    times = pd.Series(
        pd.to_datetime(
            [
                "2020-01-01",
                "2020-01-02",
                "2020-01-03 12:00",
                "2020-01-05",
                "2020-01-06",
            ],
            format="mixed",
        )
    )
    result, note = f107_spline(times, weather)
    assert np.isnan(result[0]) and np.isnan(result[-1])
    assert result[1] == 70.0 and result[3] == 90.0
    assert note["date_label_anchor"] == "SW-All DATE is UTC midnight"


def test_ap_kp_native_slots_map_without_daily_averaging():
    weather = pd.DataFrame(
        {
            "DATE": ["2020-01-01"],
            **{f"AP{i}": [i] for i in range(1, 9)},
            **{f"KP{i}": [10 + i] for i in range(1, 9)},
        }
    )
    result = slot_drivers(
        pd.Series(pd.date_range("2020-01-01", periods=8, freq="3h")), weather
    )
    assert result.ap.tolist() == list(range(1, 9))
    assert result.kp.tolist() == list(range(11, 19))


def test_preprocessing_preserves_missingness_and_handles_leap_day_key():
    time = pd.Series(
        pd.to_datetime(
            ["2020-02-28", "2020-02-29", "2020-03-01", "2021-02-28", "2021-03-01"]
        )
    )
    result = preprocess_3hour(np.array([1.0, np.nan, 3.0, 4.0, 5.0]), time)
    assert np.isnan(result[1])


def test_link_scope_and_ten_model_geomagnetic_cases():
    columns = ["error_175km", "error_200km", "f107", "ap"]
    assumptions = link_assumptions(columns, columns[:2], 2)
    assert (0, -1) in assumptions[0] and (2, 0) in assumptions[0]
    assert (0, -1) not in assumptions[1] and (0, -1) not in assumptions[2]
    assert len(cases()) == 10
    assert {item["geomagnetic"] for item in cases()} == {"ap", "kp"}
    assert declared_fdr_tests(2) == 2 * 27 * 3


def test_fingerprint_is_deterministic_and_rejects_changed_identity(tmp_path):
    case = cases()[0]
    first = fingerprint(case, {"input": "a"})
    assert first == fingerprint(case, {"input": "a"})
    assert first != fingerprint(case, {"input": "b"})
    provenance = tmp_path / "provenance.json"
    provenance.write_text('{"fingerprint": "' + first + '"}')
    assert reuse_is_compatible(provenance, first)
    assert not reuse_is_compatible(provenance, fingerprint(case, {"input": "b"}))


def test_fdr_family_is_only_the_predeclared_driver_target_tests():
    columns = ["error_175km", "f107", "ap"]
    raw = np.ones((3, 3, 2))
    raw[1, 0] = [0.01, 0.02]
    raw[2, 0] = [0.03, 0.04]
    q = driver_target_bh_qvalues(raw, columns, ["error_175km"], ["f107", "ap"])
    assert np.isnan(q[0, 1]).all()
    assert q[1, 0, 0] == 0.04
    assert q[2, 0, 1] == 0.04


def _write_shards(tmp_path, monkeypatch, *, tau=1, mismatch_altitude=None):
    case = cases()[0]
    manifest = {
        "analysis_bundle": "test-bundle",
        "tau_steps": tau,
        "physical_lag_days": 183,
    }
    monkeypatch.setattr(
        figure5,
        "analysis_inputs",
        lambda case, output, smoke: (
            pd.Series(dtype="datetime64[ns]"),
            *(np.array([]) for _ in range(3)),
            manifest,
        ),
    )
    identity = fingerprint(case, manifest)
    for altitude in ALTITUDES:
        directory = figure5.shard_directory(tmp_path, case, altitude)
        directory.mkdir(parents=True, exist_ok=True)
        target = f"error_{altitude}km"
        rows = [
            {
                "source": source,
                "target": target,
                "lag_steps": lag,
                "lag_hours": lag * 3,
                "lag_days": lag / 8,
                "partial_r": 0.1,
                "raw_p_value": 0.001 if altitude == ALTITUDES[0] else 0.9,
                "q_value": 1.0,
                "graph_mark": "-->" if lag else "o->",
                "detected": False,
            }
            for source in ("f107", case["geomagnetic"])
            for lag in range(tau + 1)
        ]
        pd.DataFrame(rows).to_csv(directory / "driver_target_raw.csv", index=False)
        shard_identity = fingerprint(case, {**manifest, "altitude_km": altitude})
        if altitude == mismatch_altitude:
            shard_identity = "wrong"
        (directory / "provenance.json").write_text(
            json.dumps(
                {
                    "fingerprint": shard_identity,
                    "case_fingerprint": identity,
                    "case": case,
                    "altitude_km": altitude,
                    "manifest": manifest,
                    "settings": figure5.shard_settings(case, altitude),
                }
            )
        )
    return case


def test_spacehopper_shard_expansion_is_135_single_altitude_tasks():
    assert len(figure5.SPACEHOPPER_CASE_IDS) * len(ALTITUDES) == 135
    assert set(figure5.SPACEHOPPER_CASE_IDS) <= {case["id"] for case in cases()}


def test_merge_requires_all_27_shards_and_applies_global_bh(tmp_path, monkeypatch):
    case = _write_shards(tmp_path, monkeypatch)
    missing = (
        figure5.shard_directory(tmp_path, case, ALTITUDES[-1]) / "driver_target_raw.csv"
    )
    missing.unlink()
    with pytest.raises(FileNotFoundError, match="missing required shard"):
        figure5.merge_shards(case, tmp_path)

    case = _write_shards(tmp_path, monkeypatch)
    merged = pd.read_csv(figure5.merge_shards(case, tmp_path))
    assert len(merged) == declared_fdr_tests(1)
    # 0.001 is corrected over all 108 tests, not just the four local shard tests.
    assert merged.loc[merged.raw_p_value.eq(0.001), "q_value"].iloc[0] == pytest.approx(
        0.027
    )


def test_merge_refuses_fingerprint_mismatch(tmp_path, monkeypatch):
    case = _write_shards(tmp_path, monkeypatch, mismatch_altitude=ALTITUDES[3])
    with pytest.raises(ValueError, match="incompatible shard provenance"):
        figure5.merge_shards(case, tmp_path)


def test_equivalence_verifier_accepts_tight_match_and_refuses_difference(tmp_path):
    rows = pd.DataFrame(
        [
            {
                "source": "f107",
                "target": "error_175km",
                "lag_steps": 0,
                "raw_p_value": 0.1,
                "partial_r": 0.2,
                "q_value": 0.1,
                "graph_mark": "o->",
                "detected": True,
            }
        ]
    )
    left, right = tmp_path / "left.csv", tmp_path / "right.csv"
    rows.to_csv(left, index=False)
    close = rows.copy()
    close.loc[0, "raw_p_value"] += 1e-14
    close.to_csv(right, index=False)
    assert verify_equivalence(left, right)["tests"] == 1
    close.loc[0, "graph_mark"] = "o-o"
    close.to_csv(right, index=False)
    with pytest.raises(ValueError, match="graph_mark"):
        verify_equivalence(left, right)
