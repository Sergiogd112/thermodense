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
