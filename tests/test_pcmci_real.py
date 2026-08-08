from __future__ import annotations

from datetime import date, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from thermodense.benchmarks import pcmci_real
from thermodense.benchmarks.real_data import DATE_COLUMN, F107_RAW_COLUMN, NODE_COLUMNS


def _frame(rows: int = 20) -> pl.DataFrame:
    dates = [date(2020, 1, 1) + timedelta(days=index) for index in range(rows)]
    data: dict[str, list[object]] = {DATE_COLUMN: dates}
    for index, column in enumerate(NODE_COLUMNS):
        data[column] = [float(index + offset) for offset in range(rows)]
        data[f"{column}_imputed"] = [False] * rows
    data[F107_RAW_COLUMN] = [float(offset) for offset in range(rows)]
    data[f"{F107_RAW_COLUMN}_imputed"] = [False] * rows
    return pl.DataFrame(data)


def test_preprocess_preserves_nan_and_standardizes_finite_values() -> None:
    dates = np.array(["2020-01-01", "2021-01-01", "2022-01-01"], dtype="datetime64[D]")
    values = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])

    result = pcmci_real.preprocess(values, dates)

    assert np.isnan(result[1, 0])
    assert np.all(np.isfinite(result[np.isfinite(values)]))
    assert np.allclose(np.nanmean(result, axis=0), 0.0)
    assert np.nanstd(result[:, 0]) == 0.0
    assert np.isclose(np.nanstd(result[:, 1]), 1.0)


def test_stationarity_qualification_passes_each_node_and_records_exact_spans() -> None:
    dates = np.datetime64("2020-01-01") + np.arange(363).astype("timedelta64[D]")
    values = np.column_stack((np.arange(363.0), np.arange(363.0)))
    values[:2, 0] = np.nan

    diagnostics = pcmci_real.stationarity_qualification(
        values,
        dates,
        ["first", "second"],
        adf=lambda _values: {"statistic": -4.0, "p_value": 0.01},
        kpss=lambda _values: {"statistic": 0.1, "p_value": 0.2},
    )

    first = diagnostics["nodes"]["first"]
    assert first["span"] == {
        "start": "2020-01-03",
        "end": "2020-12-28",
        "start_index": 2,
        "end_index": 362,
    }
    assert first["sample_count"] == 361
    assert diagnostics["causal_interpretation_eligible"] is True
    assert diagnostics["sensitivity_evidence_only"] is False
    assert first["adf"]["outcome"] == "reject_unit_root"
    assert first["kpss"]["outcome"] == "do_not_reject_level_stationarity"


@pytest.mark.parametrize(
    ("adf_p_value", "kpss_p_value", "expected_outcome"),
    [
        (0.2, 0.2, "does_not_reject_unit_root"),
        (0.01, 0.01, "reject_level_stationarity"),
    ],
)
def test_stationarity_qualification_records_test_family_failures(
    adf_p_value: float, kpss_p_value: float, expected_outcome: str
) -> None:
    values = np.column_stack((np.arange(361.0), np.arange(1.0, 362.0)))
    diagnostics = pcmci_real.stationarity_qualification(
        values,
        np.datetime64("2020-01-01") + np.arange(361).astype("timedelta64[D]"),
        ["first", "second"],
        adf=lambda _values: {"statistic": -4.0, "p_value": adf_p_value},
        kpss=lambda _values: {"statistic": 0.1, "p_value": kpss_p_value},
    )

    assert diagnostics["causal_interpretation_eligible"] is False
    assert diagnostics["sensitivity_evidence_only"] is True
    assert expected_outcome in {
        diagnostics["nodes"]["first"]["adf"]["outcome"],
        diagnostics["nodes"]["first"]["kpss"]["outcome"],
    }


def test_stationarity_qualification_handles_missing_and_too_short_spans() -> None:
    values = np.array([[np.nan, 1.0], [np.nan, np.nan], [np.nan, 2.0]])
    diagnostics = pcmci_real.stationarity_qualification(
        values,
        np.arange("2020-01-01", "2020-01-04", dtype="datetime64[D]"),
        ["missing", "short"],
    )

    assert diagnostics["causal_interpretation_eligible"] is False
    assert diagnostics["nodes"]["missing"]["outcome"] == "not_qualified_missing_span"
    assert diagnostics["nodes"]["short"]["outcome"] == "not_qualified_too_short_span"


def test_holm_adjustment_is_boundary_correct_and_node_order_invariant() -> None:
    p_values = {"z": 0.025, "a": 0.01, "m": 0.03}

    adjusted = pcmci_real.holm_adjusted_pvalues(p_values)

    assert adjusted == {"z": 0.05, "a": 0.03, "m": 0.05}
    assert adjusted == pcmci_real.holm_adjusted_pvalues(dict(reversed(list(p_values.items()))))


def test_stationarity_qualification_uses_statsmodels_and_records_raw_results() -> None:
    values = np.random.default_rng(6).normal(size=(400, 2))
    diagnostics = pcmci_real.stationarity_qualification(
        values,
        np.datetime64("2020-01-01") + np.arange(400).astype("timedelta64[D]"),
        ["first", "second"],
    )

    adf = diagnostics["nodes"]["first"]["adf"]
    kpss = diagnostics["nodes"]["first"]["kpss"]
    assert {"statistic", "raw_p_value", "adjusted_p_value", "used_lag"} <= adf.keys()
    assert {"statistic", "raw_p_value", "adjusted_p_value", "used_lag"} <= kpss.keys()


def test_stationarity_full_family_holm_includes_missing_nodes() -> None:
    values = np.column_stack((np.arange(361.0), np.full(361, np.nan)))
    diagnostics = pcmci_real.stationarity_qualification(
        values,
        np.datetime64("2020-01-01") + np.arange(361).astype("timedelta64[D]"),
        ["tested", "missing"],
        adf=lambda _values: {"statistic": -4.0, "p_value": 0.03},
        kpss=lambda _values: {"statistic": 0.1, "p_value": 0.2},
    )

    assert diagnostics["test_families"]["adf"]["family_size"] == 2
    assert diagnostics["test_families"]["adf"]["unavailable_nodes"] == ["missing"]
    assert diagnostics["nodes"]["tested"]["adf"]["adjusted_p_value"] == 0.06
    assert diagnostics["causal_interpretation_eligible"] is False


def test_stationarity_full_family_holm_includes_test_error_nodes() -> None:
    values = np.column_stack((np.arange(361.0), np.arange(1.0, 362.0)))

    def adf(series: np.ndarray) -> dict[str, float]:
        if series[0] == 0:
            return {"statistic": -4.0, "p_value": 0.03}
        raise ValueError("test failure")

    diagnostics = pcmci_real.stationarity_qualification(
        values,
        np.datetime64("2020-01-01") + np.arange(361).astype("timedelta64[D]"),
        ["tested", "error"],
        adf=adf,
        kpss=lambda _values: {"statistic": 0.1, "p_value": 0.2},
    )

    assert diagnostics["test_families"]["adf"]["unavailable_nodes"] == ["error"]
    assert diagnostics["test_families"]["kpss"]["tested_nodes"] == ["error", "tested"]
    assert diagnostics["nodes"]["tested"]["adf"]["adjusted_p_value"] == 0.06
    assert diagnostics["nodes"]["error"]["outcome"] == "not_qualified_test_error"
    assert diagnostics["nodes"]["error"]["adf"]["outcome"] == "test_error"
    assert diagnostics["nodes"]["error"]["kpss"]["raw_p_value"] == 0.2


def test_stationarity_kpss_failure_still_records_adf_and_family_membership() -> None:
    values = np.column_stack((np.arange(361.0), np.arange(1.0, 362.0)))

    def kpss(series: np.ndarray) -> dict[str, float]:
        if series[0] == 0:
            return {"statistic": 0.1, "p_value": 0.2}
        raise ValueError("test failure")

    diagnostics = pcmci_real.stationarity_qualification(
        values,
        np.datetime64("2020-01-01") + np.arange(361).astype("timedelta64[D]"),
        ["tested", "error"],
        adf=lambda _values: {"statistic": -4.0, "p_value": 0.03},
        kpss=kpss,
    )

    assert diagnostics["test_families"]["adf"]["tested_nodes"] == ["error", "tested"]
    assert diagnostics["test_families"]["kpss"]["unavailable_nodes"] == ["error"]
    assert diagnostics["nodes"]["error"]["adf"]["adjusted_p_value"] == 0.06
    assert diagnostics["nodes"]["error"]["kpss"]["outcome"] == "test_error"


@pytest.mark.parametrize("invalid_p_value", [np.nan, -0.01, 1.01])
def test_stationarity_invalid_p_values_are_unavailable(invalid_p_value: float) -> None:
    diagnostics = pcmci_real.stationarity_qualification(
        np.column_stack((np.arange(361.0), np.arange(1.0, 362.0))),
        np.datetime64("2020-01-01") + np.arange(361).astype("timedelta64[D]"),
        ["invalid", "valid"],
        adf=lambda values: {
            "statistic": -4.0,
            "p_value": invalid_p_value if values[0] == 0 else 0.01,
        },
        kpss=lambda _values: {"statistic": 0.1, "p_value": 0.2},
    )

    assert diagnostics["nodes"]["invalid"]["adf"]["outcome"] == "test_error"
    assert diagnostics["test_families"]["adf"]["unavailable_nodes"] == ["invalid"]
    assert diagnostics["nodes"]["valid"]["adf"]["adjusted_p_value"] == 0.02


def test_registered_sensitivity_cases_expand_the_full_common_row_matrix() -> None:
    cases = pcmci_real.expand_sensitivity_cases()

    assert {(case.timing_variant, case.preprocessing_profile) for case in cases} == {
        ("raw_observed_daily", "detrended_anomaly"),
        ("raw_observed_daily", "seasonal_anomaly"),
        ("centered_81_day", "detrended_anomaly"),
        ("centered_81_day", "seasonal_anomaly"),
    }
    assert {case.role for case in cases} == {
        "primary",
        "robustness",
        "interaction_diagnostic",
    }
    assert cases[0].role == "primary"
    assert cases[1].role == cases[2].role == "robustness"
    assert cases[3].role == "interaction_diagnostic"


def test_registered_sensitivity_cases_reject_unregistered_combinations() -> None:
    with pytest.raises(ValueError, match="unregistered PCMCI sensitivity case"):
        pcmci_real.sensitivity_case("raw_observed_daily", "raw_standardized")


def test_sensitivity_cases_share_exact_accepted_quality_rows() -> None:
    dates = np.arange("2020-01-01", "2020-01-06", dtype="datetime64[D]")
    values = np.ones((len(dates), len(NODE_COLUMNS)))
    values[1, 1] = np.nan
    raw_f107 = np.ones(len(dates))
    raw_f107[3] = np.nan
    input_data = pcmci_real.RealInput(dates, values, {}, raw_f107)

    prepared = [
        pcmci_real.prepare_sensitivity_input(input_data, case)
        for case in pcmci_real.expand_sensitivity_cases()
    ]

    assert {metadata["daily_date_sequence_sha256"] for _, _, metadata in prepared} == {
        prepared[0][2]["daily_date_sequence_sha256"]
    }
    assert {len(case_input.dates) for case_input, _, _ in prepared} == {5}
    assert {metadata["common_f107_support"]["sha256"] for _, _, metadata in prepared} == {
        prepared[0][2]["common_f107_support"]["sha256"]
    }
    assert prepared[0][1][0] == F107_RAW_COLUMN
    assert prepared[-1][1][0] == "f10_7_center81"


def test_sensitivity_input_preserves_daily_axis_and_common_f107_support() -> None:
    dates = np.arange("2020-01-01", "2020-01-06", dtype="datetime64[D]")
    values = np.ones((len(dates), len(NODE_COLUMNS)))
    values[2, 0] = np.nan
    values[1, 1] = np.nan
    raw_f107 = np.ones(len(dates))
    raw_f107[3] = np.nan
    input_data = pcmci_real.RealInput(dates, values, {}, raw_f107)

    raw, _, raw_identity = pcmci_real.prepare_sensitivity_input(
        input_data, pcmci_real.sensitivity_case("raw_observed_daily", "detrended_anomaly")
    )
    centered, _, centered_identity = pcmci_real.prepare_sensitivity_input(
        input_data, pcmci_real.sensitivity_case("centered_81_day", "detrended_anomaly")
    )

    assert np.array_equal(raw.dates, dates)
    assert np.array_equal(centered.dates, dates)
    assert np.diff(raw.dates.astype("int64")).tolist() == [1, 1, 1, 1]
    assert len(raw.dates) == len(centered.dates) == len(dates)
    assert np.isfinite(raw.values[:, 0]).tolist() == [True, True, False, False, True]
    assert np.isfinite(centered.values[:, 0]).tolist() == [True, True, False, False, True]
    assert np.isnan(raw.values[1, 1])
    assert raw_identity == centered_identity


def test_seasonal_anomaly_matches_calendar_month_day_and_keeps_february_29_distinct() -> (
    None
):
    dates = np.array(
        ["2020-02-28", "2021-02-28", "2020-02-29", "2020-03-01", "2021-03-01"],
        dtype="datetime64[D]",
    )

    anomalies = pcmci_real.seasonal_anomaly(
        np.array([10.0, 14.0, 100.0, 20.0, 24.0]), dates
    )

    assert np.allclose(anomalies, [-2.0, 2.0, 0.0, -2.0, 2.0])


def test_validate_daily_dates_rejects_duplicates_and_gaps() -> None:
    with np.testing.assert_raises_regex(ValueError, "unique"):
        pcmci_real.validate_daily_dates(
            np.array(["2020-01-01", "2020-01-01"], dtype="datetime64[D]")
        )
    with np.testing.assert_raises_regex(ValueError, "consecutive"):
        pcmci_real.validate_daily_dates(
            np.array(["2020-01-01", "2020-01-03"], dtype="datetime64[D]")
        )


def test_f107_link_assumptions_forbid_external_causes_but_allow_self_lags() -> None:
    assumptions = pcmci_real.build_link_assumptions(2)
    f107 = NODE_COLUMNS.index("f10_7_center81")

    assert assumptions[f107][(f107, -1)] == "-?>"
    for other in range(1, len(NODE_COLUMNS)):
        assert (other, -1) not in assumptions[f107]
        assert assumptions[other][(f107, 0)] == "-?>"


def test_load_input_records_selected_row_metadata(tmp_path: Path) -> None:
    path = tmp_path / "five_node.csv"
    frame = _frame()
    frame = frame.with_columns(
        pl.when(pl.arange(0, len(frame)) == 3)
        .then(None)
        .otherwise(pl.col(NODE_COLUMNS[1]))
        .alias(NODE_COLUMNS[1])
    )
    frame = frame.with_columns(
        pl.when(pl.arange(0, len(frame)) == 4)
        .then(True)
        .otherwise(False)
        .alias(f"{NODE_COLUMNS[2]}_imputed")
    )
    frame.write_csv(path)

    loaded = pcmci_real.load_input(path, row_limit=10)

    assert loaded.metadata["row_count"] == 10
    assert loaded.metadata["row_limit"] == 10
    assert loaded.metadata["row_limit_calibration_only"] is True
    assert loaded.metadata["node_order"] == NODE_COLUMNS
    assert loaded.metadata["node_counts"][NODE_COLUMNS[1]] == {
        "observed": 9,
        "missing": 1,
        "imputed": 0,
    }
    assert loaded.metadata["node_counts"][NODE_COLUMNS[2]] == {
        "observed": 9,
        "missing": 0,
        "imputed": 1,
    }
    assert len(loaded.metadata["input_sha256"]) == 64
    assert "Maunakea" in loaded.metadata["co2_source"]


def test_cli_rejects_invalid_real_run_options(tmp_path: Path, capsys) -> None:
    output = tmp_path / "result.jsonl"
    parsed = pcmci_real.parser().parse_args(["run", "--output", str(output)])
    assert parsed.tau_max == 180
    assert parsed.cmiknn_workers == 24
    assert parsed.timing_variant == "raw_observed_daily"
    assert parsed.preprocessing_profile == "detrended_anomaly"
    assert pcmci_real.main(["run", "--output", str(output), "--tau-max", "-1"]) == 2
    assert "--tau-max" in capsys.readouterr().err
    parsed_gpu = pcmci_real.parser().parse_args(
        ["run", "--output", str(output), "--methods", "gpdctorch"]
    )
    assert parsed_gpu.methods == ["gpdctorch"]
    with pytest.raises(SystemExit):
        pcmci_real.parser().parse_args(
            ["run", "--output", str(output), "--methods", "gpdc"]
        )


def test_real_run_enforces_cmiknn_ten_lag_step_resource_cap(
    tmp_path: Path, capsys
) -> None:
    output = tmp_path / "result.jsonl"

    assert (
        pcmci_real.main(
            [
                "run",
                "--output",
                str(output),
                "--methods",
                "cmiknn",
                "--tau-max",
                "11",
            ]
        )
        == 2
    )
    assert "resource limit of 10 lag steps" in capsys.readouterr().err

    parsed = pcmci_real.parser().parse_args(
        [
            "run",
            "--output",
            str(output),
            "--methods",
            "cmiknn",
            "--tau-max",
            "10",
        ]
    )
    assert parsed.tau_max == 10
    pcmci_real.runtime.validate_cmiknn_tau("cmiknn", parsed.tau_max)
    assert pcmci_real.real_method_settings("gpdctorch", 24) == {
        "pc_alpha": 0.05,
        "significance": "analytic",
    }


def test_tiny_real_parcorr_pcmciplus_case(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    dates = np.arange("2020-01-01", "2020-03-21", dtype="datetime64[D]")
    values = rng.normal(size=(len(dates), len(NODE_COLUMNS)))
    input_data = pcmci_real.RealInput(dates, values, {}, values[:, 0].copy())

    artifact = tmp_path / "pcmci-real-test-result.npz"
    result = pcmci_real.run_pcmciplus(
        input_data, "parcorr", tau_max=1, cmiknn_workers=1, artifact_path=artifact
    )

    assert result["matrix_shapes"]["graph"] == [len(NODE_COLUMNS), len(NODE_COLUMNS), 2]
    assert len(result["result_digest"]) == 64
    assert "alpha_level" not in result["settings"]
    with np.load(artifact, allow_pickle=False) as saved:
        assert sorted(saved.files) == ["graph", "node_names", "p_matrix", "val_matrix"]
        assert saved["node_names"].tolist() == [F107_RAW_COLUMN, *NODE_COLUMNS[1:]]
        assert saved["graph"].shape == tuple(result["matrix_shapes"]["graph"])
        assert saved["p_matrix"].shape == tuple(result["matrix_shapes"]["p_matrix"])
        assert saved["val_matrix"].shape == tuple(result["matrix_shapes"]["val_matrix"])


def test_run_records_atomic_npz_artifact_in_jsonl(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "five_node.csv"
    _frame().write_csv(input_path)
    output = tmp_path / "result.jsonl"
    args = pcmci_real.parser().parse_args(
        [
            "run",
            "--input",
            str(input_path),
            "--output",
            str(output),
            "--methods",
            "parcorr",
        ]
    )
    matrices = {
        "graph": np.array([[["-->"]]]),
        "p_matrix": np.array([[[0.25]]]),
        "val_matrix": np.array([[[0.5]]]),
    }

    calls = []

    def fake_case(args, method, case, threads, artifact):
        calls.append((method, case, artifact))
        return {
            "status": "succeeded",
            "matrix_shapes": {
                name: list(value.shape) for name, value in matrices.items()
            },
            "result_digest": pcmci_real.runtime.compact_result_digest(matrices),
            "artifact": pcmci_real.runtime.write_npz_artifact(
                artifact, matrices, node_names=NODE_COLUMNS
            ),
        }

    monkeypatch.setattr(pcmci_real, "_run_isolated_case", fake_case)
    monkeypatch.setattr(
        pcmci_real,
        "stationarity_qualification",
        lambda *_args: {
            "causal_interpretation_eligible": False,
            "sensitivity_evidence_only": True,
        },
    )

    assert pcmci_real.run(args) == 0
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == len(calls) == 1
    row = rows[0]
    assert row["sensitivity_case"]["timing_variant"] == "raw_observed_daily"
    assert row["sensitivity_case"]["preprocessing_profile"] == "detrended_anomaly"
    assert row["sensitivity_case"]["role"] == "primary"
    assert row["sensitivity_case"]["node_order"][0] == F107_RAW_COLUMN
    assert row["sensitivity_case"]["f10_7"]["source_column"] == F107_RAW_COLUMN
    assert row["sensitivity_case"]["f10_7"]["source_counts"] == {
        "observed": 20,
        "missing": 0,
        "imputed": 0,
    }
    assert "calendar month/day" in row["preprocessing"]["seasonal_climatology"]
    assert row["preprocessing"]["february_29_has_distinct_climatology"] is True
    assert row["causal_interpretation_eligible"] is False
    assert row["sensitivity_evidence_only"] is True
    assert row["stationarity_qualification"]["provenance_identity"]["node_order"] == [
        F107_RAW_COLUMN,
        *NODE_COLUMNS[1:],
    ]
    assert row["rolling_diagnostics"]["window_days"] == 365
    assert "does not alter qualification" in row["rolling_diagnostics"]["diagnostic"]
    with np.load(row["rolling_diagnostics"]["path"], allow_pickle=False) as rolling:
        assert sorted(rolling.files) == ["dates", "node_names", "rolling_mean", "rolling_variance"]
        assert rolling["node_names"].tolist() == [F107_RAW_COLUMN, *NODE_COLUMNS[1:]]
        assert rolling["rolling_mean"].shape == (20, len(NODE_COLUMNS))
        assert rolling["rolling_variance"].shape == (20, len(NODE_COLUMNS))
    assert row["artifact"]["format"] == "npz-compressed"
    assert row["artifact"]["keys"] == [
        "graph",
        "node_names",
        "p_matrix",
        "val_matrix",
    ]
    artifact = Path(row["artifact"]["path"])
    assert row["artifact"]["name"] == artifact.name
    assert (
        row["artifact"]["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    )
    with np.load(artifact, allow_pickle=False) as saved:
        assert saved["node_names"].tolist() == NODE_COLUMNS
        for name, expected in matrices.items():
            assert np.array_equal(saved[name], expected)


def test_multi_method_run_reuses_one_case_diagnostic_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    input_path = tmp_path / "five_node.csv"
    _frame().write_csv(input_path)
    output = tmp_path / "result.jsonl"
    args = pcmci_real.parser().parse_args(
        [
            "run",
            "--input",
            str(input_path),
            "--output",
            str(output),
            "--methods",
            "parcorr",
            "cmiknn",
            "--tau-max",
            "1",
        ]
    )
    qualification_calls = []
    preprocessing_calls = []
    rolling_calls = []
    artifact_calls = []

    def qualification(*_args):
        qualification_calls.append(None)
        return {"causal_interpretation_eligible": False, "sensitivity_evidence_only": True}

    def write_artifact(path, *_args, **_kwargs):
        artifact_calls.append(path)
        return {"path": str(path), "name": path.name, "sha256": "shared", "format": "npz-compressed", "keys": []}

    original_preprocess = pcmci_real.preprocess
    original_rolling = pcmci_real.rolling_diagnostics

    def preprocess(*args):
        preprocessing_calls.append(None)
        return original_preprocess(*args)

    def rolling(*args):
        rolling_calls.append(None)
        return original_rolling(*args)

    monkeypatch.setattr(pcmci_real, "stationarity_qualification", qualification)
    monkeypatch.setattr(pcmci_real, "preprocess", preprocess)
    monkeypatch.setattr(pcmci_real, "rolling_diagnostics", rolling)
    monkeypatch.setattr(pcmci_real.runtime, "write_npz_artifact", write_artifact)
    monkeypatch.setattr(pcmci_real, "_run_isolated_case", lambda *_args: {"status": "succeeded"})

    assert pcmci_real.run(args) == 0
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(preprocessing_calls) == len(qualification_calls) == len(rolling_calls) == len(artifact_calls) == 1
    assert artifact_calls[0].name == "raw_observed_daily-detrended_anomaly-rolling.npz"
    assert len(rows) == 2
    assert {row["rolling_diagnostics"]["sha256"] for row in rows} == {"shared"}
    assert {json.dumps(row["stationarity_qualification"], sort_keys=True) for row in rows} == {
        json.dumps(rows[0]["stationarity_qualification"], sort_keys=True)
    }


def test_run_executes_only_the_selected_gpdctorch_case(
    tmp_path: Path, monkeypatch
) -> None:
    input_path = tmp_path / "five_node.csv"
    _frame().write_csv(input_path)
    output = tmp_path / "result.jsonl"
    args = pcmci_real.parser().parse_args(
        [
            "run",
            "--input",
            str(input_path),
            "--output",
            str(output),
            "--methods",
            "gpdctorch",
            "--timing-variant",
            "raw_observed_daily",
            "--preprocessing-profile",
            "detrended_anomaly",
            "--tau-max",
            "1",
        ]
    )
    calls = []

    def fake_case(args, method, case, threads, artifact):
        calls.append((method, case, artifact))
        return {"status": "succeeded"}

    monkeypatch.setattr(pcmci_real, "_run_isolated_case", fake_case)

    assert pcmci_real.run(args) == 0
    assert len(calls) == 1
    method, case, artifact = calls[0]
    assert method == "gpdctorch"
    assert case == pcmci_real.sensitivity_case("raw_observed_daily", "detrended_anomaly")
    assert artifact.name == "gpdctorch-raw_observed_daily-detrended_anomaly.npz"
    row = json.loads(output.read_text())
    assert row["sensitivity_case"]["role"] == "primary"
    assert row["sensitivity_case"]["node_order"][0] == F107_RAW_COLUMN


def test_matrix_execution_emits_four_matching_case_rows(
    tmp_path: Path, monkeypatch
) -> None:
    input_path = tmp_path / "five_node.csv"
    _frame().write_csv(input_path)
    output = tmp_path / "result.jsonl"
    args = pcmci_real.parser().parse_args(
        [
            "run",
            "--input",
            str(input_path),
            "--output",
            str(output),
            "--methods",
            "parcorr",
            "--tau-max",
            "3",
            "--all-sensitivity-cases",
        ]
    )
    calls = []

    def fake_case(args, method, case, threads, artifact):
        calls.append((method, case, artifact))
        return {"status": "succeeded"}

    monkeypatch.setattr(pcmci_real, "_run_isolated_case", fake_case)

    assert pcmci_real.run(args) == 0
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == len(calls) == 4
    assert len({call[2].name for call in calls}) == 4
    assert {row["sensitivity_case"]["accepted_quality_rows"]["daily_date_sequence_sha256"] for row in rows} == {
        rows[0]["sensitivity_case"]["accepted_quality_rows"]["daily_date_sequence_sha256"]
    }
    assert {row["sensitivity_case"]["accepted_quality_rows"]["common_f107_support"]["sha256"] for row in rows} == {
        rows[0]["sensitivity_case"]["accepted_quality_rows"]["common_f107_support"]["sha256"]
    }
    assert {row["tau_max"] for row in rows} == {3}
    assert {row["missing_data_policy"]["rows_dropped"] for row in rows} == {False}
    assert {json.dumps(row["algorithm"], sort_keys=True) for row in rows} == {
        json.dumps(rows[0]["algorithm"], sort_keys=True)
    }


def test_production_matrix_runs_exact_parcorr_cells_and_writes_agreement(
    tmp_path: Path, monkeypatch
) -> None:
    input_path = tmp_path / "five_node.csv"
    _frame().write_csv(input_path)
    output = tmp_path / "result.jsonl"
    args = pcmci_real.parser().parse_args(
        [
            "run",
            "--input",
            str(input_path),
            "--output",
            str(output),
            "--production-sensitivity-matrix",
        ]
    )
    calls = []

    def fake_case(args, method, case, threads, artifact):
        calls.append((method, case))
        return {
            "status": "succeeded",
            "artifact": pcmci_real.runtime.write_npz_artifact(
                artifact,
                {
                    "graph": np.full((5, 5, 181), "", dtype="<U3"),
                    "p_matrix": np.ones((5, 5, 181)),
                    "val_matrix": np.zeros((5, 5, 181)),
                },
                node_names=(
                    [F107_RAW_COLUMN, *NODE_COLUMNS[1:]]
                    if case.timing_variant == pcmci_real.RAW_OBSERVED_DAILY
                    else NODE_COLUMNS
                ),
            ),
        }

    monkeypatch.setattr(pcmci_real, "_run_isolated_case", fake_case)

    assert pcmci_real.run(args) == 0
    assert len(calls) == 4
    assert {method for method, _case in calls} == {"parcorr"}
    agreement = json.loads(
        (tmp_path / "result_artifacts" / "parcorr-sensitivity-agreement.jsonl").read_text()
    )
    assert agreement["state"] == "complete"
    assert agreement["physical_lag_window_days"] == {"min": 0, "max": 180}


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--methods", "cmiknn"], "only ParCorr"),
        (["--tau-max", "10"], "0-180"),
        (["--row-limit", "10"], "row-limit"),
    ],
)
def test_production_matrix_rejects_nonproduction_scope(
    tmp_path: Path, argv: list[str], message: str
) -> None:
    input_path = tmp_path / "five_node.csv"
    _frame().write_csv(input_path)
    args = pcmci_real.parser().parse_args(
        [
            "run",
            "--input",
            str(input_path),
            "--output",
            str(tmp_path / "result.jsonl"),
            "--production-sensitivity-matrix",
            *argv,
        ]
    )

    with pytest.raises(ValueError, match=message):
        pcmci_real.run(args)


def test_production_matrix_emits_incomplete_agreement_when_a_cell_fails(
    tmp_path: Path, monkeypatch
) -> None:
    input_path = tmp_path / "five_node.csv"
    _frame().write_csv(input_path)
    output = tmp_path / "result.jsonl"
    args = pcmci_real.parser().parse_args(
        ["run", "--input", str(input_path), "--output", str(output), "--production-sensitivity-matrix"]
    )
    monkeypatch.setattr(
        pcmci_real, "_run_isolated_case", lambda *_args: {"status": "failed", "failure_reason": "fixture"}
    )

    with pytest.raises(pcmci_real.MatrixSynthesisError, match="incomplete"):
        pcmci_real.run(args)
    agreement = json.loads(
        (tmp_path / "result_artifacts" / "parcorr-sensitivity-agreement.jsonl").read_text()
    )
    assert agreement["state"] == "incomplete"


def test_production_matrix_preserves_synthesis_error_after_writing_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    input_path = tmp_path / "five_node.csv"
    _frame().write_csv(input_path)
    output = tmp_path / "result.jsonl"
    args = pcmci_real.parser().parse_args(
        ["run", "--input", str(input_path), "--output", str(output), "--production-sensitivity-matrix"]
    )
    synthesis_error = pcmci_real.MatrixSynthesisError("synthesis failure")
    monkeypatch.setattr(pcmci_real, "synthesize_parcorr_matrix", lambda *_args: (_ for _ in ()).throw(synthesis_error))
    monkeypatch.setattr(
        pcmci_real, "_run_isolated_case", lambda *_args: {"status": "success", "failure_reason": None}
    )

    with pytest.raises(pcmci_real.MatrixSynthesisError) as raised:
        pcmci_real.run(args)

    assert raised.value is synthesis_error
    manifest = json.loads(
        (tmp_path / "result_artifacts" / "parcorr-sensitivity-agreement.jsonl").read_text()
    )
    assert manifest["state"] == "incomplete"
    assert manifest["error"] == "synthesis failure"


def test_production_matrix_records_parent_preparation_failure(
    tmp_path: Path, monkeypatch
) -> None:
    input_path = tmp_path / "five_node.csv"
    _frame().write_csv(input_path)
    output = tmp_path / "result.jsonl"
    args = pcmci_real.parser().parse_args(
        ["run", "--input", str(input_path), "--output", str(output), "--production-sensitivity-matrix"]
    )
    monkeypatch.setattr(
        pcmci_real,
        "prepare_case_diagnostics",
        lambda *_args: (_ for _ in ()).throw(OSError("rolling artifact fixture failure")),
    )

    with pytest.raises(pcmci_real.MatrixSynthesisError, match="rolling artifact fixture failure"):
        pcmci_real.run(args)
    manifest = json.loads(
        (tmp_path / "result_artifacts" / "parcorr-sensitivity-agreement.jsonl").read_text()
    )
    assert manifest["state"] == "incomplete"
    assert manifest["current_case"] == "raw_observed_daily/detrended_anomaly"
    assert manifest["completed_cells"][0]["status"] == "failed"


@pytest.mark.parametrize("missing_raw", [False, True])
def test_production_matrix_writes_incomplete_manifest_for_input_failures(
    tmp_path: Path, missing_raw: bool
) -> None:
    input_path = tmp_path / "five_node.csv"
    if missing_raw:
        _frame().drop(F107_RAW_COLUMN).write_csv(input_path)
    args = pcmci_real.parser().parse_args(
        ["run", "--input", str(input_path), "--output", str(tmp_path / "result.jsonl"), "--production-sensitivity-matrix"]
    )

    with pytest.raises(pcmci_real.MatrixSynthesisError, match="incomplete matrix"):
        pcmci_real.run(args)
    manifest = json.loads(
        (tmp_path / "result_artifacts" / "parcorr-sensitivity-agreement.jsonl").read_text()
    )
    assert manifest["state"] == "incomplete"


def test_production_matrix_overwrite_cleans_stale_artifacts_before_input_load(tmp_path: Path) -> None:
    output = tmp_path / "result.jsonl"
    artifact_directory = tmp_path / "result_artifacts"
    artifact_directory.mkdir()
    (artifact_directory / "stale.npz").write_text("stale")
    args = pcmci_real.parser().parse_args(
        ["run", "--input", str(tmp_path / "missing.csv"), "--output", str(output),
         "--production-sensitivity-matrix", "--overwrite"]
    )

    with pytest.raises(pcmci_real.MatrixSynthesisError, match="incomplete matrix"):
        pcmci_real.run(args)

    assert not (artifact_directory / "stale.npz").exists()
    assert json.loads((artifact_directory / "parcorr-sensitivity-agreement.jsonl").read_text())["state"] == "incomplete"


@pytest.mark.parametrize("all_sensitivity_cases", [False, True])
def test_omitted_methods_defaults_to_parcorr_without_optional_method_guards(
    tmp_path: Path, monkeypatch, all_sensitivity_cases: bool
) -> None:
    input_path = tmp_path / "five_node.csv"
    _frame().write_csv(input_path)
    output = tmp_path / "result.jsonl"
    argv = ["run", "--input", str(input_path), "--output", str(output), "--tau-max", "3"]
    if all_sensitivity_cases:
        argv.append("--all-sensitivity-cases")
    calls = []

    def fail_optional_guard(*_args, **_kwargs):
        raise AssertionError("optional method guard was reached")

    def fake_case(args, method, case, threads, artifact):
        calls.append((method, case))
        return {"status": "succeeded"}

    monkeypatch.setattr(pcmci_real, "_validate_gpdctorch_scope", fail_optional_guard)
    monkeypatch.setattr(pcmci_real.runtime, "validate_cmiknn_tau", fail_optional_guard)
    monkeypatch.setattr(pcmci_real, "_run_isolated_case", fake_case)

    assert pcmci_real.main(argv) == 0
    expected_cases = 4 if all_sensitivity_cases else 1
    assert len(calls) == expected_cases
    assert {method for method, _case in calls} == {"parcorr"}


def test_gpdctorch_rejects_non_primary_and_matrix_execution(tmp_path: Path) -> None:
    input_path = tmp_path / "five_node.csv"
    _frame().write_csv(input_path)
    output = tmp_path / "result.jsonl"

    non_primary = pcmci_real.parser().parse_args(
        [
            "run",
            "--input",
            str(input_path),
            "--output",
            str(output),
            "--methods",
            "gpdctorch",
            "--timing-variant",
            "centered_81_day",
        ]
    )
    with pytest.raises(ValueError, match="only the primary"):
        pcmci_real.run(non_primary)

    matrix = pcmci_real.parser().parse_args(
        [
            "run",
            "--input",
            str(input_path),
            "--output",
            str(output),
            "--methods",
            "gpdctorch",
            "--all-sensitivity-cases",
        ]
    )
    with pytest.raises(ValueError, match="sensitivity-matrix"):
        pcmci_real.run(matrix)

    primary_tau_10 = pcmci_real.parser().parse_args(
        [
            "run",
            "--input",
            str(input_path),
            "--output",
            str(output),
            "--methods",
            "gpdctorch",
            "--tau-max",
            "10",
        ]
    )
    with pytest.raises(ValueError, match="tau_max=1"):
        pcmci_real.run(primary_tau_10)

    row_limited = pcmci_real.parser().parse_args(
        [
            "run",
            "--input",
            str(input_path),
            "--output",
            str(output),
            "--methods",
            "gpdctorch",
            "--row-limit",
            "10",
            "--tau-max",
            "1",
        ]
    )
    with pytest.raises(ValueError, match="row-limited or prefix"):
        pcmci_real.run(row_limited)


def test_gpdctorch_direct_run_accepts_only_primary_tau_one(monkeypatch) -> None:
    input_data = pcmci_real.RealInput(
        np.arange("2020-01-01", "2020-01-20", dtype="datetime64[D]"),
        np.ones((19, len(NODE_COLUMNS))),
        {},
    )
    monkeypatch.setattr(pcmci_real, "prepare_sensitivity_input", lambda data, case: (data, list(NODE_COLUMNS), {}))
    with pytest.raises(ValueError, match="tau_max=1"):
        pcmci_real.run_pcmciplus(input_data, "gpdctorch", tau_max=10, cmiknn_workers=1)

    input_data.metadata["row_limit"] = 10
    with pytest.raises(ValueError, match="row-limited or prefix"):
        pcmci_real.run_pcmciplus(input_data, "gpdctorch", tau_max=1, cmiknn_workers=1)


def test_run_refuses_to_overwrite_existing_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "result.jsonl"
    (tmp_path / "result_artifacts").mkdir()
    args = pcmci_real.parser().parse_args(["run", "--output", str(output)])

    with pytest.raises(ValueError, match="artifacts"):
        pcmci_real.run(args)


def test_pcmciplus_rejects_too_few_rows_for_lag_window() -> None:
    dates = np.arange("2020-01-01", "2020-01-10", dtype="datetime64[D]")
    values = np.ones((len(dates), len(NODE_COLUMNS)))

    with pytest.raises(ValueError, match="2\\*tau_max"):
        pcmci_real.run_pcmciplus(
            pcmci_real.RealInput(dates, values, {}, values[:, 0].copy()),
            "parcorr",
            tau_max=5,
            cmiknn_workers=1,
        )


def _matrix_row(
    tmp_path: Path,
    case: pcmci_real.SensitivityCase,
    node_names: list[str],
    links: list[tuple[int, int, int, str, float]],
) -> dict[str, object]:
    size = len(node_names)
    graph = np.full((size, size, 181), "", dtype="<U3")
    p_matrix = np.ones((size, size, 181))
    val_matrix = np.zeros((size, size, 181))
    for source, target, lag, mark, value in links:
        graph[source, target, lag] = mark
        p_matrix[source, target, lag] = 0.01
        val_matrix[source, target, lag] = value
    artifact = pcmci_real.runtime.write_npz_artifact(
        tmp_path / f"{case.timing_variant}-{case.preprocessing_profile}.npz",
        {"graph": graph, "p_matrix": p_matrix, "val_matrix": val_matrix},
        node_names=node_names,
    )
    return {
        "status": "succeeded",
        "schema_version": pcmci_real.SCHEMA_VERSION,
        "runner_version": pcmci_real.RUNNER_VERSION,
        "synthetic": False,
        "method": "parcorr",
        "tau_max": 180,
        "artifact": artifact,
        "algorithm": {
            "name": "PCMCI+",
            "entry_point": "PCMCI.run_pcmciplus",
            "tau_min": 0,
            "pc_alpha": 0.05,
            "contemp_collider_rule": "majority",
            "conflict_resolution": True,
            "fdr_method": "none",
        },
        "settings": {"significance": "analytic", "pc_alpha": 0.05, "threads": 1},
        "missing_data_policy": {
            "sentinel": pcmci_real.MISSING_FLAG,
            "remove_missing_upto_maxlag": False,
            "drivers_interpolated": False,
            "rows_dropped": False,
        },
        "link_assumptions": pcmci_real._link_assumption_metadata(180, node_names),
        "sensitivity_case": {
            "timing_variant": case.timing_variant,
            "preprocessing_profile": case.preprocessing_profile,
            "role": case.role,
            "node_order": node_names,
            "accepted_quality_rows": {
                "daily_date_sequence_sha256": "a" * 64,
                "row_count": 500,
                "common_f107_support": {"sha256": "b" * 64, "row_count": 500},
            },
        },
        "stationarity_qualification": {
            "causal_interpretation_eligible": True,
            "sensitivity_evidence_only": False,
            "provenance_identity": {
                "timing_variant": case.timing_variant,
                "preprocessing_profile": case.preprocessing_profile,
                "node_order": node_names,
                "daily_date_sequence_sha256": "a" * 64,
                "common_f107_support_sha256": "b" * 64,
            },
        },
        "causal_interpretation_eligible": True,
        "sensitivity_evidence_only": False,
    }


def test_synthesize_parcorr_matrix_classifies_agreement_and_exploration(
    tmp_path: Path,
) -> None:
    primary, raw_seasonal, centered_detrended, interaction = (
        pcmci_real.expand_sensitivity_cases()
    )
    raw_nodes = [F107_RAW_COLUMN, *NODE_COLUMNS[1:]]
    centered_nodes = NODE_COLUMNS
    rows = [
        _matrix_row(
            tmp_path,
            primary,
            raw_nodes,
            [
                (0, 1, 1, "-->", 0.4),
                (0, 2, 1, "-->", 0.3),
                (0, 3, 1, "-->", 0.2),
                (0, 4, 1, "-->", 0.1),
                (0, 0, 3, "-->", 0.1),
            ],
        ),
        _matrix_row(
            tmp_path,
            raw_seasonal,
            raw_nodes,
            [(0, 1, 1, "-->", 0.4), (0, 3, 1, "-->", 0.2), (0, 0, 3, "-->", 0.1), (0, 1, 2, "-->", 0.5)],
        ),
        _matrix_row(
            tmp_path,
            centered_detrended,
            centered_nodes,
            [(0, 1, 2, "-->", 0.4), (0, 2, 1, "-->", 0.3), (0, 0, 4, "-->", 0.1)],
        ),
        _matrix_row(tmp_path, interaction, centered_nodes, [(0, 3, 1, "-->", 0.2)]),
    ]

    agreement = pcmci_real.synthesize_parcorr_matrix(rows)

    links = {link["target"]: link for link in agreement["primary_links"]}
    assert links["ap_avg"]["classification"] == "main_text_robust"
    assert links["ap_avg"]["centered_matches"][0]["lag"] == 2
    assert links["ap_avg"]["centered_delay_equivalent"] is False
    assert links["co2_ppm"]["failed_dimensions"] == ["detrending"]
    assert links["log10rho_325_daily_mean"]["failed_dimensions"] == ["timing"]
    assert links["log10rho_825_daily_mean"]["failed_dimensions"] == ["detrending", "timing"]
    self_lag = next(link for link in agreement["primary_links"] if link["target"] == F107_RAW_COLUMN)
    assert self_lag["classification"] == "main_text_robust"
    assert self_lag["centered_matches"][0]["lag"] == 4
    assert {
        (link["source"], link["target"], link["lag"], tuple(link["source_cells"]))
        for link in agreement["exploratory_links"]
    } >= {
        (F107_RAW_COLUMN, "ap_avg", 2, ("raw_observed_daily/seasonal_anomaly",))
    }
    exploratory = next(
        link
        for link in agreement["exploratory_links"]
        if link["source"] == F107_RAW_COLUMN and link["target"] == "ap_avg" and link["lag"] == 2
    )
    assert exploratory["evidence"] == [
        {
            "case": "raw_observed_daily/seasonal_anomaly",
            "source": F107_RAW_COLUMN,
            "target": "ap_avg",
            "lag": 2,
            "sign": 1,
            "value": 0.5,
            "p_value": 0.01,
            "graph_mark": "-->",
        }
    ]
    assert agreement["interaction_diagnostic_links"][0]["target"] == "log10rho_325_daily_mean"
    assert not any(
        link["source"] == "f10_7_center81" and link["target"] == "ap_avg" and link["lag"] == 2
        for link in agreement["exploratory_links"]
    )
    assert len(agreement["case_artifacts"]) == 4


def test_synthesis_keeps_sign_mismatch_and_deduplicates_contemporaneous_orientation(
    tmp_path: Path,
) -> None:
    primary, raw_seasonal, centered_detrended, interaction = (
        pcmci_real.expand_sensitivity_cases()
    )
    raw_nodes = [F107_RAW_COLUMN, *NODE_COLUMNS[1:]]
    centered_nodes = NODE_COLUMNS
    primary_links = [(0, 1, 0, "-->", 0.4), (1, 0, 0, "<--", 0.4)]
    rows = [
        _matrix_row(tmp_path, primary, raw_nodes, primary_links),
        _matrix_row(tmp_path, raw_seasonal, raw_nodes, [(0, 1, 0, "-->", -0.4)]),
        _matrix_row(tmp_path, centered_detrended, centered_nodes, [(0, 1, 2, "-->", 0.4)]),
        _matrix_row(tmp_path, interaction, centered_nodes, [(0, 1, 0, "-->", 0.4)]),
    ]

    agreement = pcmci_real.synthesize_parcorr_matrix(rows)

    assert len(agreement["primary_links"]) == 1
    link = agreement["primary_links"][0]
    assert link["lag"] == 0
    assert link["failed_dimensions"] == ["detrending"]
    assert link["evidence"]["primary"]["p_value"] == 0.01


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda rows: rows.pop(), "incomplete"),
        (lambda rows: rows[0].update(method="cmiknn"), "method"),
        (lambda rows: rows[0].update(tau_max=10), "tau_max"),
        (lambda rows: rows[0]["settings"].update(threads=2), "settings"),
        (lambda rows: rows[0]["sensitivity_case"].update(node_order=["wrong"]), "provenance"),
        (
            lambda rows: rows[0]["sensitivity_case"]["accepted_quality_rows"].update(
                daily_date_sequence_sha256="other"
            ),
            "accepted-quality",
        ),
    ],
)
def test_synthesis_rejects_incomplete_or_mixed_matrix_identities(
    tmp_path: Path, mutate, message: str
) -> None:
    cases = pcmci_real.expand_sensitivity_cases()
    rows = [
        _matrix_row(
            tmp_path,
            case,
            [F107_RAW_COLUMN, *NODE_COLUMNS[1:]] if case.timing_variant == pcmci_real.RAW_OBSERVED_DAILY else NODE_COLUMNS,
            [],
        )
        for case in cases
    ]
    mutate(rows)

    with pytest.raises(pcmci_real.MatrixSynthesisError, match=message):
        pcmci_real.synthesize_parcorr_matrix(rows)


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("schema_version", "wrong", "schema version"),
        ("schema_version", None, "schema version"),
        ("runner_version", "wrong", "runner version"),
        ("runner_version", None, "runner version"),
        ("synthetic", True, "synthetic marker"),
        ("synthetic", None, "synthetic marker"),
    ],
)
def test_synthesis_rejects_invalid_matrix_row_identity(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    cases = pcmci_real.expand_sensitivity_cases()
    rows = [
        _matrix_row(
            tmp_path,
            case,
            [F107_RAW_COLUMN, *NODE_COLUMNS[1:]]
            if case.timing_variant == pcmci_real.RAW_OBSERVED_DAILY
            else NODE_COLUMNS,
            [],
        )
        for case in cases
    ]
    rows[0][field] = value

    with pytest.raises(pcmci_real.MatrixSynthesisError, match=message):
        pcmci_real.synthesize_parcorr_matrix(rows)


def test_synthesis_omits_partial_and_conflicting_marks_with_diagnostics(tmp_path: Path) -> None:
    primary, raw_seasonal, centered_detrended, interaction = (
        pcmci_real.expand_sensitivity_cases()
    )
    raw_nodes = [F107_RAW_COLUMN, *NODE_COLUMNS[1:]]
    centered_nodes = NODE_COLUMNS
    rows = [
        _matrix_row(tmp_path, primary, raw_nodes, [(0, 1, 1, "o->", 0.4), (0, 1, 2, "x-x", 0.4)]),
        _matrix_row(tmp_path, raw_seasonal, raw_nodes, []),
        _matrix_row(tmp_path, centered_detrended, centered_nodes, []),
        _matrix_row(tmp_path, interaction, centered_nodes, []),
    ]

    agreement = pcmci_real.synthesize_parcorr_matrix(rows)

    assert agreement["primary_links"] == []
    assert agreement["orientation_diagnostics"]["raw_observed_daily/detrended_anomaly"] == {
        "partially_oriented_count": 1,
        "conflict_count": 1,
    }


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("stale_hash", "sha256"),
        ("missing_key", "canonical keys"),
        ("truncated_lags", "shape"),
        ("wrong_axis", "node axis"),
        ("malformed_dimensions", "three-dimensional"),
        ("nonfinite_evidence", "finite"),
        ("out_of_range_p", "p-value"),
        ("zero_directed_value", "nonzero"),
        ("mixed_normalized_nodes", "stationarity provenance"),
        ("stationarity_identity", "stationarity provenance"),
    ],
)
def test_synthesis_rejects_artifact_and_provenance_integrity_failures(
    tmp_path: Path, mutation: str, message: str
) -> None:
    cases = pcmci_real.expand_sensitivity_cases()
    rows = [
        _matrix_row(
            tmp_path,
            case,
            [F107_RAW_COLUMN, *NODE_COLUMNS[1:]] if case.timing_variant == pcmci_real.RAW_OBSERVED_DAILY else NODE_COLUMNS,
            [(0, 1, 1, "-->", 0.4)],
        )
        for case in cases
    ]
    row = rows[0]
    artifact = Path(row["artifact"]["path"])
    if mutation == "stale_hash":
        artifact.write_bytes(artifact.read_bytes() + b"stale")
    elif mutation == "mixed_normalized_nodes":
        rows[2]["sensitivity_case"]["node_order"] = ["different", "density"]
        rows[2]["stationarity_qualification"]["provenance_identity"]["node_order"] = ["different", "density"]
    elif mutation == "stationarity_identity":
        row["stationarity_qualification"]["provenance_identity"]["daily_date_sequence_sha256"] = "wrong"
    else:
        with np.load(artifact, allow_pickle=False) as saved:
            contents = {name: saved[name] for name in saved.files}
        if mutation == "missing_key":
            contents.pop("p_matrix")
        elif mutation == "truncated_lags":
            for name in ("graph", "p_matrix", "val_matrix"):
                contents[name] = contents[name][..., :180]
        elif mutation == "wrong_axis":
            contents["node_names"] = np.array(["wrong", *NODE_COLUMNS[1:]])
        elif mutation == "malformed_dimensions":
            contents["graph"] = contents["graph"][0]
        elif mutation == "nonfinite_evidence":
            contents["val_matrix"][0, 1, 1] = np.nan
        elif mutation == "out_of_range_p":
            contents["p_matrix"][0, 1, 1] = 1.1
        elif mutation == "zero_directed_value":
            contents["val_matrix"][0, 1, 1] = 0.0
        np.savez_compressed(artifact, **contents)
        row["artifact"]["sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()

    with pytest.raises(pcmci_real.MatrixSynthesisError, match=message):
        pcmci_real.synthesize_parcorr_matrix(rows)


def test_synthesis_converts_malformed_rows_to_controlled_errors() -> None:
    with pytest.raises(pcmci_real.MatrixSynthesisError, match="malformed"):
        pcmci_real.synthesize_parcorr_matrix([{"sensitivity_case": None}])


def test_synthesis_rejects_stationarity_top_level_eligibility_mismatch(tmp_path: Path) -> None:
    cases = pcmci_real.expand_sensitivity_cases()
    rows = [
        _matrix_row(
            tmp_path,
            case,
            [F107_RAW_COLUMN, *NODE_COLUMNS[1:]] if case.timing_variant == pcmci_real.RAW_OBSERVED_DAILY else NODE_COLUMNS,
            [],
        )
        for case in cases
    ]
    rows[0]["causal_interpretation_eligible"] = False

    with pytest.raises(pcmci_real.MatrixSynthesisError, match="stationarity eligibility"):
        pcmci_real.synthesize_parcorr_matrix(rows)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row["stationarity_qualification"].pop("causal_interpretation_eligible"),
        lambda row: row["stationarity_qualification"].update(causal_interpretation_eligible="yes"),
        lambda row: row.update(sensitivity_evidence_only=True),
        lambda row: row["stationarity_qualification"].update(sensitivity_evidence_only=True),
    ],
)
def test_synthesis_rejects_malformed_or_noncomplementary_stationarity_flags(tmp_path: Path, mutation) -> None:
    cases = pcmci_real.expand_sensitivity_cases()
    rows = [
        _matrix_row(tmp_path, case,
                    [F107_RAW_COLUMN, *NODE_COLUMNS[1:]] if case.timing_variant == pcmci_real.RAW_OBSERVED_DAILY else NODE_COLUMNS,
                    [])
        for case in cases
    ]
    mutation(rows[0])
    with pytest.raises(pcmci_real.MatrixSynthesisError):
        pcmci_real.synthesize_parcorr_matrix(rows)


@pytest.mark.parametrize("failed_index", [0, 1, 2])
def test_agreeing_links_are_stationarity_limited_by_required_cells(
    tmp_path: Path, failed_index: int
) -> None:
    cases = pcmci_real.expand_sensitivity_cases()
    rows = [
        _matrix_row(
            tmp_path,
            case,
            [F107_RAW_COLUMN, *NODE_COLUMNS[1:]] if case.timing_variant == pcmci_real.RAW_OBSERVED_DAILY else NODE_COLUMNS,
            [(0, 1, 1 if case.timing_variant == pcmci_real.RAW_OBSERVED_DAILY else 2, "-->", 0.4)],
        )
        for case in cases
    ]
    rows[failed_index]["stationarity_qualification"]["causal_interpretation_eligible"] = False
    rows[failed_index]["stationarity_qualification"]["sensitivity_evidence_only"] = True
    rows[failed_index]["causal_interpretation_eligible"] = False
    rows[failed_index]["sensitivity_evidence_only"] = True

    agreement = pcmci_real.synthesize_parcorr_matrix(rows)

    link = agreement["primary_links"][0]
    assert link["classification"] == "stationarity_limited"
    assert link["failed_stationarity_cells"] == [
        "raw_observed_daily/detrended_anomaly",
        "raw_observed_daily/seasonal_anomaly",
        "centered_81_day/detrended_anomaly",
    ][failed_index:failed_index + 1]


@pytest.mark.parametrize(
    "mutation",
    ["node_order", "accepted_rows", "missing_policy", "algorithm", "assumptions", "settings"],
)
def test_synthesis_rejects_identically_malformed_production_schema_rows(
    tmp_path: Path, mutation: str
) -> None:
    cases = pcmci_real.expand_sensitivity_cases()
    rows = [
        _matrix_row(
            tmp_path,
            case,
            [F107_RAW_COLUMN, *NODE_COLUMNS[1:]] if case.timing_variant == pcmci_real.RAW_OBSERVED_DAILY else NODE_COLUMNS,
            [],
        )
        for case in cases
    ]
    for row in rows:
        if mutation == "node_order":
            row["sensitivity_case"]["node_order"] = list(reversed(row["sensitivity_case"]["node_order"]))
            row["stationarity_qualification"]["provenance_identity"]["node_order"] = row["sensitivity_case"]["node_order"]
        elif mutation == "accepted_rows":
            row["sensitivity_case"]["accepted_quality_rows"]["row_count"] = 0
        elif mutation == "missing_policy":
            row["missing_data_policy"]["rows_dropped"] = True
        elif mutation == "algorithm":
            row["algorithm"]["pc_alpha"] = 0.1
        elif mutation == "assumptions":
            row["link_assumptions"]["tau_max"] = 10
        else:
            row["settings"]["significance"] = "shuffle_test"

    with pytest.raises(pcmci_real.MatrixSynthesisError):
        pcmci_real.synthesize_parcorr_matrix(rows)
