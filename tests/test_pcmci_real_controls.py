from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from thermodense.benchmarks import pcmci_real, pcmci_real_controls as controls
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


def test_iaaft_is_deterministic_preserves_marginal_and_approximates_spectrum() -> None:
    values = np.sin(np.arange(512) / 11) + 0.2 * np.cos(np.arange(512) / 3)
    first, first_metadata = controls.iaaft_surrogate(values, seed=12)
    second, second_metadata = controls.iaaft_surrogate(values, seed=12)

    assert np.array_equal(first, second)
    assert first_metadata == second_metadata
    assert np.array_equal(np.sort(first), np.sort(values))
    assert first_metadata["spectral_relative_l2_error"] < 0.05


def test_iaaft_preserves_observed_marginal_and_source_missing_mask() -> None:
    values = np.sin(np.arange(512) / 11)
    values[[0, 7, 201, 511]] = np.nan

    surrogate, metadata = controls.iaaft_surrogate(values, seed=12)

    finite = np.isfinite(values)
    assert np.array_equal(np.isfinite(surrogate), finite)
    assert np.array_equal(np.sort(surrogate[finite]), np.sort(values[finite]))
    assert metadata["missing_count"] == 4
    assert metadata["missingness_matched"] is True
    assert (
        metadata["finite_marginal_reference"] == "sorted finite observed source values"
    )
    assert metadata["spectral_output_completion"] == (
        "deterministic linear interpolation with nearest finite edge fill"
    )


def test_controls_preserve_shift_values_missing_mask_and_valid_offsets() -> None:
    values = np.column_stack([np.arange(800, dtype=float), np.arange(800, dtype=float)])
    values[[2, 400], 0] = np.nan
    generated, metadata = controls.generate_controls(
        values, ["a", "b"], tau_max=180, seed=7
    )

    assert list(generated) == [
        "control_iaaft_a",
        "control_circular_shift_a",
        "control_iaaft_b",
        "control_circular_shift_b",
    ]
    shift = generated["control_circular_shift_a"]
    offset = metadata["control_circular_shift_a"]["offset"]
    assert np.array_equal(shift, np.roll(values[:, 0], offset), equal_nan=True)
    assert np.isnan(shift).sum() == np.isnan(values[:, 0]).sum()
    assert 180 < offset < len(values) - 180
    assert min(offset % 365, (-offset) % 365) > controls.ANNUAL_EXCLUSION_DAYS
    iaaft = generated["control_iaaft_a"]
    assert np.array_equal(np.isfinite(iaaft), np.isfinite(values[:, 0]))
    assert np.array_equal(
        np.sort(iaaft[np.isfinite(iaaft)]),
        np.sort(values[np.isfinite(values[:, 0]), 0]),
    )
    assert metadata["control_iaaft_a"]["source_node"] == "a"
    assert metadata["control_iaaft_a"]["missingness_matched"] is True


def test_selected_links_include_family_and_source_metadata() -> None:
    names = ["physical", "control_iaaft_physical"]
    graph = np.full((2, 2, 2), "", dtype="<U3")
    graph[1, 0, 1] = "-->"
    rows = controls.selected_control_links(
        {
            "graph": graph,
            "p_matrix": np.ones((2, 2, 2)),
            "val_matrix": np.ones((2, 2, 2)),
        },
        names,
        {"control_iaaft_physical": {"family": "iaaft", "source_node": "physical"}},
    )

    assert rows == [
        {
            "relation": "surrogate→physical",
            "cause": "control_iaaft_physical",
            "target": "physical",
            "lag": 1,
            "graph_mark": "-->",
            "p_value": 1.0,
            "val": 1.0,
            "cause_family": "iaaft",
            "cause_source": "physical",
            "target_family": "physical",
            "target_source": "physical",
        }
    ]


def test_run_records_dynamic_artifact_summary_and_control_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    input_path = tmp_path / "five-node.csv"
    _frame().write_csv(input_path)
    output = tmp_path / "result.jsonl"
    args = controls.parser().parse_args(
        ["run", "--input", str(input_path), "--output", str(output)]
    )
    matrices = {
        "graph": np.array([[["-->"]]]),
        "p_matrix": np.array([[[0.2]]]),
        "val_matrix": np.array([[[0.3]]]),
    }

    def fake_case(args, threads, artifact, summary):
        return {
            "status": "succeeded",
            "artifact": controls.runtime.write_npz_artifact(
                artifact, matrices, node_names=["physical"]
            ),
            "control_link_summary": controls.runtime.write_jsonl_artifact(summary, []),
        }

    monkeypatch.setattr(controls, "_run_isolated_case", fake_case)
    assert controls.run(args) == 0
    row = json.loads(output.read_text())
    assert row["preprocessing"]["computed_once_on_physical_nodes"] is True
    assert row["preprocessing"]["controls_appended_without_preprocessing"] is True
    assert row["controls"]["iaaft_iterations"] == 100
    assert row["missing_data_policy"]["controls_source_mask_matched"] is True
    for artifact in (row["artifact"], row["control_link_summary"]):
        path = Path(artifact["path"])
        assert artifact["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_cli_validation_and_tiny_live_pcmci(tmp_path: Path, capsys) -> None:
    output = tmp_path / "result.jsonl"
    assert (
        controls.parser().parse_args(["run", "--output", str(output)]).seed
        == controls.SEED
    )
    assert controls.main(["run", "--output", str(output), "--tau-max", "-1"]) == 2
    assert "--tau-max" in capsys.readouterr().err
    dates = np.arange("2020-01-01", "2020-03-21", dtype="datetime64[D]")
    data = pcmci_real.RealInput(
        dates, np.random.default_rng(4).normal(size=(len(dates), 5)), {}
    )
    artifact = tmp_path / "result.npz"
    summary = tmp_path / "links.jsonl"
    result = controls.run_pcmciplus(
        data, tau_max=1, seed=9, artifact_path=artifact, summary_path=summary
    )
    assert result["matrix_shapes"]["graph"] == [15, 15, 2]
    with np.load(artifact, allow_pickle=False) as saved:
        assert saved["node_names"].tolist() == [
            *NODE_COLUMNS,
            *(
                control
                for name in NODE_COLUMNS
                for control in (
                    f"control_iaaft_{name}",
                    f"control_circular_shift_{name}",
                )
            ),
        ]
