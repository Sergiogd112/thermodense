from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from thermodense.benchmarks import pcmci_real
from thermodense.benchmarks.real_data import DATE_COLUMN, NODE_COLUMNS


def _frame(rows: int = 20) -> pl.DataFrame:
    dates = [date(2020, 1, 1) + timedelta(days=index) for index in range(rows)]
    data: dict[str, list[object]] = {DATE_COLUMN: dates}
    for index, column in enumerate(NODE_COLUMNS):
        data[column] = [float(index + offset) for offset in range(rows)]
        data[f"{column}_imputed"] = [False] * rows
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
    assert pcmci_real.main(["run", "--output", str(output), "--tau-max", "-1"]) == 2
    assert "--tau-max" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        pcmci_real.main(["run", "--output", str(output), "--methods", "gpdc"])


def test_tiny_real_parcorr_pcmciplus_case() -> None:
    rng = np.random.default_rng(4)
    dates = np.arange("2020-01-01", "2020-03-21", dtype="datetime64[D]")
    values = rng.normal(size=(len(dates), len(NODE_COLUMNS)))
    input_data = pcmci_real.RealInput(dates, values, {})

    result = pcmci_real.run_pcmciplus(
        input_data, "parcorr", tau_max=1, cmiknn_workers=1
    )

    assert result["matrix_shapes"]["graph"] == [len(NODE_COLUMNS), len(NODE_COLUMNS), 2]
    assert len(result["result_digest"]) == 64
    assert "alpha_level" not in result["settings"]


def test_pcmciplus_rejects_too_few_rows_for_lag_window() -> None:
    dates = np.arange("2020-01-01", "2020-01-10", dtype="datetime64[D]")
    values = np.ones((len(dates), len(NODE_COLUMNS)))

    with pytest.raises(ValueError, match="2\\*tau_max"):
        pcmci_real.run_pcmciplus(
            pcmci_real.RealInput(dates, values, {}),
            "parcorr",
            tau_max=5,
            cmiknn_workers=1,
        )
