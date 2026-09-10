from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from thermodense.benchmarks import global_activity_input as inputs
from thermodense.benchmarks import global_activity_pcmci as subject


def _input() -> inputs.GlobalActivityInput:
    timestamps = np.datetime64("2020-01-01") + np.arange(4).astype("timedelta64[D]")
    values = np.arange(56, dtype=float).reshape(4, 14)
    return inputs.GlobalActivityInput(timestamps, values, np.ones_like(values, dtype=bool), np.array([90.0, 120.0, 170.0, np.nan]), inputs.NODE_NAMES.copy(), {})


def test_link_family_forbids_density_to_drivers_and_activity_selection_keeps_histories() -> None:
    assumptions = subject.link_assumptions(inputs.NODE_NAMES, 1)
    density, driver = 0, 10
    assert (driver, -1) in assumptions[density] and (driver, 0) in assumptions[density]
    assert (density, -1) not in assumptions[driver]
    # The Y-only mask is based on target dates; the preceding day remains an
    # eligible predictor history for the moderate target on day two.
    selected = inputs.group_mask(_input().activity_f107_center81, "moderate")
    mask = np.broadcast_to(~selected[:, None], (4, 14))
    assert not mask[1, 0] and mask[0, 0]


def test_rows_apply_one_group_bh_family_and_reverse_contemporaneous_edges() -> None:
    data = _input()
    result = {"graph": np.full((14, 14, 2), "-->", dtype="<U3"), "p_matrix": np.full((14, 14, 2), 0.5), "val_matrix": np.ones((14, 14, 2))}
    # A lag-zero <---mark is stored at [source, target] but means target→source.
    result["graph"][0, 1, 0] = "<--"
    rows = subject.result_rows(result, data, "low")
    reversed_row = next(row for row in rows if row["lag_days"] == 0 and row["source"] == inputs.NODE_NAMES[1] and row["target"] == inputs.NODE_NAMES[0])
    assert reversed_row["direction"] == "target_to_source"
    assert len(rows) == len(subject.eligible_locations(data.node_names, 1))
    assert {row["group"] for row in rows} == {"low"}
    assert {row["q_value"] for row in rows} == {0.5}


def test_trace_records_actual_y_only_support_and_conditioning_dimension(tmp_path: Path) -> None:
    from tigramite import data_processing as pp

    count = 20
    values = np.column_stack((np.arange(count, dtype=float), np.arange(count, dtype=float) ** 2))
    # Moderate targets are preceded by low days. They must still have their
    # preceding X history, despite that history not itself being a target date.
    selected = np.arange(count) % 2 == 1
    frame = pp.DataFrame(values, mask=np.broadcast_to(~selected[:, None], values.shape), missing_flag=subject.MISSING_FLAG, remove_missing_upto_maxlag=False)
    trace = tmp_path / "trace.jsonl"
    ci = subject._tracing_parcorr(trace)
    ci.set_dataframe(frame)
    ci.run_test([(0, -1)], [(1, 0)], [], tau_max=1)
    record = json.loads(trace.read_text().strip())
    assert record["sample_count"] == 9
    assert record["conditioning_dimension"] == 0
    assert record["test_id"] == 1 and record["tested"]
    assert "partial_r" in record and "p_value" in record


def _synthetic_manifest(tmp_path: Path, count: int = 32) -> Path:
    dates = np.datetime64("2020-01-01") + np.arange(count).astype("timedelta64[D]")
    density = tmp_path / "density.parquet"
    pl.DataFrame({"date": dates, **{name: np.sin(np.arange(count) * (index + 1)) + index for index, name in enumerate(inputs.NODE_NAMES[:10])}}).write_parquet(density)
    weather = tmp_path / "weather.csv"
    weather.write_text("DATE,F10.7_OBS,F10.7_OBS_CENTER81,AP_AVG,KP_SUM\n" + "".join(f"{date},100,120,{index % 8},{index % 17}\n" for index, date in enumerate(dates)), encoding="utf-8")
    co2 = tmp_path / "co2.csv"
    co2.write_text("".join(f"{str(date).replace('-', ',')},2020.0,{410 + index / 10}\n" for index, date in enumerate(dates)), encoding="utf-8")
    return inputs.assemble(density, weather, co2, tmp_path / "input")


def test_tiny_real_fixture_runs_full_topology_and_persists_outputs(tmp_path: Path) -> None:
    output = subject.run(_synthetic_manifest(tmp_path), tmp_path / "result", tau_max=1, fixture=True)
    with np.load(output / "pcmci_raw.npz", allow_pickle=False) as saved:
        assert saved["graph"].shape == (14, 14, 2)
    assert (output / "eligible_links.csv").is_file()
    assert (output / "ci_test_trace.jsonl").is_file()
    assert json.loads((output / "status.json").read_text())["status"] == "complete"


def test_raw_npz_is_persisted_before_atomic_csv_export_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _synthetic_manifest(tmp_path)
    graph = np.full((14, 14, 2), "", dtype="<U3")
    result = {"graph": graph, "p_matrix": np.full((14, 14, 2), 0.5), "val_matrix": np.zeros((14, 14, 2))}

    def fake_run(_data: inputs.GlobalActivityInput, _group: str, _tau: int, trace: Path) -> dict[str, np.ndarray]:
        trace.write_text('{"test_id": 1}\n')
        return result

    monkeypatch.setattr(subject, "run_pcmci", fake_run)
    monkeypatch.setattr(subject, "_atomic_csv", lambda *_args: (_ for _ in ()).throw(RuntimeError("export failure")))
    output = tmp_path / "failed-result"
    with pytest.raises(RuntimeError, match="export failure"):
        subject.run(manifest, output, tau_max=1, fixture=True)
    assert (output / "pcmci_raw.npz").is_file()
    assert json.loads((output / "status.json").read_text())["status"] == "failed"
