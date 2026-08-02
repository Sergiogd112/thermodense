from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from thermodense.benchmarks import pcmci_real, pcmci_real_surrogates as surrogates
from thermodense.benchmarks.real_data import DATE_COLUMN, NODE_COLUMNS


def _frame(rows: int = 20) -> pl.DataFrame:
    dates = [date(2020, 1, 1) + timedelta(days=index) for index in range(rows)]
    return pl.DataFrame(
        {
            DATE_COLUMN: dates,
            **{
                column: [float(index + offset) for offset in range(rows)]
                for index, column in enumerate(NODE_COLUMNS)
            },
        }
    )


def test_surrogates_are_deterministic_with_expected_names_and_settings() -> None:
    dates = np.arange("2020-01-01", "2022-01-01", dtype="datetime64[D]")
    first, first_settings = surrogates.generate_surrogates(dates)
    second, second_settings = surrogates.generate_surrogates(dates)

    assert list(first) == [
        *(f"surrogate_white_noise_{index}" for index in range(1, 6)),
        *(f"surrogate_sine_6mo_{index}" for index in range(1, 4)),
        *(f"surrogate_sine_11p4yr_{index}" for index in range(1, 4)),
    ]
    assert all(np.array_equal(first[name], second[name]) for name in first)
    assert first_settings == second_settings
    six_month = first_settings["surrogate_sine_6mo_1"]
    solar_cycle = first_settings["surrogate_sine_11p4yr_1"]
    assert six_month["period_days"] == 365.25 / 2
    assert solar_cycle["period_days"] == 11.4 * 365.25
    assert six_month["noise_variance"] == 0.25 * six_month["sine_variance"]
    assert solar_cycle["noise_variance"] == 0.25 * solar_cycle["sine_variance"]


def test_dynamic_assumptions_forbid_surrogate_causes_of_f107() -> None:
    node_names = [*NODE_COLUMNS, "surrogate_white_noise_1"]
    assumptions = surrogates.build_link_assumptions(node_names, tau_max=2)
    f107 = node_names.index("f10_7_center81")
    surrogate = node_names.index("surrogate_white_noise_1")

    assert assumptions[f107][(f107, -1)] == "-?>"
    assert (surrogate, -1) not in assumptions[f107]
    assert assumptions[surrogate][(f107, 0)] == "-?>"


def test_selected_link_summary_classifies_surrogate_links() -> None:
    node_names = ["physical", "surrogate_a", "surrogate_b"]
    graph = np.full((3, 3, 2), "", dtype="<U3")
    graph[1, 0, 1] = "-->"
    graph[0, 2, 1] = "-->"
    graph[1, 2, 0] = "o-o"
    p_matrix = np.full((3, 3, 2), 0.25)
    val_matrix = np.full((3, 3, 2), 0.5)

    rows = surrogates.selected_surrogate_links(
        {"graph": graph, "p_matrix": p_matrix, "val_matrix": val_matrix},
        node_names,
        {"surrogate_a", "surrogate_b"},
    )

    assert [row["relation"] for row in rows] == [
        "physical→surrogate",
        "surrogate→physical",
        "surrogate↔surrogate",
    ]
    assert all(
        set(row)
        == {"relation", "cause", "target", "lag", "graph_mark", "p_value", "val"}
        for row in rows
    )


def test_run_retains_artifact_and_surrogate_summary_in_jsonl(
    tmp_path: Path, monkeypatch
) -> None:
    input_path = tmp_path / "five_node.csv"
    _frame().write_csv(input_path)
    output = tmp_path / "result.jsonl"
    args = surrogates.parser().parse_args(
        ["run", "--input", str(input_path), "--output", str(output)]
    )
    matrices = {
        "graph": np.array([[["-->"]]]),
        "p_matrix": np.array([[[0.25]]]),
        "val_matrix": np.array([[[0.5]]]),
    }

    def fake_case(args, threads, artifact, summary):
        return {
            "status": "succeeded",
            "artifact": surrogates.runtime.write_npz_artifact(
                artifact, matrices, node_names=["physical"]
            ),
            "surrogate_link_summary": surrogates.runtime.write_jsonl_artifact(
                summary,
                [
                    {
                        "relation": "surrogate→physical",
                        "cause": "surrogate_white_noise_1",
                        "target": "physical",
                        "lag": 1,
                        "graph_mark": "-->",
                        "p_value": 0.25,
                        "val": 0.5,
                    }
                ],
            ),
        }

    monkeypatch.setattr(surrogates, "_run_isolated_case", fake_case)
    assert surrogates.run(args) == 0

    row = json.loads(output.read_text())
    assert row["surrogates"]["seed"] == 20260601
    assert row["surrogates"]["appended_before_preprocessing"] is True
    assert len(row["surrogates"]["names"]) == 11
    for retained in (row["artifact"], row["surrogate_link_summary"]):
        path = Path(retained["path"])
        assert retained["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert (
        "surrogate→physical" in Path(row["surrogate_link_summary"]["path"]).read_text()
    )


def test_tiny_parcorr_case_writes_augmented_artifact_and_summary(
    tmp_path: Path,
) -> None:
    dates = np.arange("2020-01-01", "2020-03-21", dtype="datetime64[D]")
    input_data = pcmci_real.RealInput(
        dates,
        np.random.default_rng(4).normal(size=(len(dates), len(NODE_COLUMNS))),
        {"node_order": NODE_COLUMNS},
    )
    artifact = tmp_path / "result.npz"
    summary = tmp_path / "surrogate_links.jsonl"

    result = surrogates.run_pcmciplus(
        input_data,
        tau_max=1,
        seed=20260601,
        white_count=1,
        six_month_count=1,
        solar_cycle_count=1,
        artifact_path=artifact,
        summary_path=summary,
    )

    assert result["matrix_shapes"]["graph"] == [len(NODE_COLUMNS) + 3] * 2 + [2]
    assert result["surrogate_names"] == [
        "surrogate_white_noise_1",
        "surrogate_sine_6mo_1",
        "surrogate_sine_11p4yr_1",
    ]
    assert (
        result["artifact"]["sha256"]
        == hashlib.sha256(artifact.read_bytes()).hexdigest()
    )
    with np.load(artifact, allow_pickle=False) as saved:
        assert saved["node_names"].tolist() == [
            *NODE_COLUMNS,
            "surrogate_white_noise_1",
            "surrogate_sine_6mo_1",
            "surrogate_sine_11p4yr_1",
        ]
    assert (
        result["surrogate_link_summary"]["sha256"]
        == hashlib.sha256(summary.read_bytes()).hexdigest()
    )


def test_cli_validates_surrogate_options(tmp_path: Path, capsys) -> None:
    output = tmp_path / "result.jsonl"
    parsed = surrogates.parser().parse_args(["run", "--output", str(output)])
    assert parsed.tau_max == 180
    assert parsed.surrogate_seed == 20260601
    assert (
        surrogates.main(["run", "--output", str(output), "--white-surrogates", "-1"])
        == 2
    )
    assert "surrogate counts" in capsys.readouterr().err
