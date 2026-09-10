from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from thermodense.benchmarks import global_activity_input as subject


def _sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    dates = ["2020-01-01", "2020-01-03"]
    density = tmp_path / "density.parquet"
    pl.DataFrame({"date": dates, **{name: [float(index), -201.0 if index == 0 else float(index + 1)] for index, name in enumerate(subject.NODE_NAMES[:10])}}).write_parquet(density)
    weather = tmp_path / "weather.csv"
    weather.write_text("DATE,F10.7_OBS,F10.7_OBS_CENTER81,AP_AVG,KP_SUM\n2020-01-01,77,99,2,8\n2020-01-03,88,160,3,9\n", encoding="utf-8")
    co2 = tmp_path / "co2.csv"
    co2.write_text("# MLO daily\n2020,1,1,2020.0,410\n2020,1,3,2020.01,-99.99\n", encoding="utf-8")
    return density, weather, co2


def test_assembler_keeps_full_calendar_raw_fit_and_missingness(tmp_path: Path) -> None:
    manifest = subject.assemble(*_sources(tmp_path), tmp_path / "assembled")
    loaded = subject.load_manifest(manifest)

    assert loaded.node_names == subject.NODE_NAMES
    assert loaded.manifest["transformations"]["standardizing"] == "none"
    assert loaded.timestamps.tolist() == [np.datetime64("2020-01-01"), np.datetime64("2020-01-02"), np.datetime64("2020-01-03")]
    assert loaded.values[0, 10] == 77  # raw F10.7, not CENTER81 activity.
    assert loaded.activity_f107_center81[[0, 2]].tolist() == [99.0, 160.0]
    assert np.isnan(loaded.activity_f107_center81[1])
    assert not loaded.available[2, 0]  # legacy density sentinel
    assert not loaded.available[2, 13]  # negative MLO missing declaration
    assert not loaded.available[1].any()  # calendar gap retained, never interpolated


def test_group_bounds_and_input_digest_integrity(tmp_path: Path) -> None:
    assert subject.group_mask(np.array([99, 100, 159.9, 160, np.nan]), "low").tolist() == [True, False, False, False, False]
    assert subject.group_mask(np.array([99, 100, 159.9, 160, np.nan]), "moderate").tolist() == [False, True, True, False, False]
    assert subject.group_mask(np.array([99, 100, 159.9, 160, np.nan]), "high").tolist() == [False, False, False, True, False]
    assert subject.group_mask(np.array([np.nan]), "all").tolist() == [True]
    manifest = subject.assemble(*_sources(tmp_path), tmp_path / "assembled")
    core = manifest.parent / "input.npz"
    core.write_bytes(core.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="digest"):
        subject.load_manifest(manifest)


def test_assembler_rejects_duplicate_source_dates(tmp_path: Path) -> None:
    density, weather, co2 = _sources(tmp_path)
    weather.write_text("DATE,F10.7_OBS,F10.7_OBS_CENTER81,AP_AVG,KP_SUM\n2020-01-01,77,99,2,8\n2020-01-01,88,160,3,9\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate space-weather"):
        subject.assemble(density, weather, co2, tmp_path / "assembled")


def test_loader_rejects_nonboolean_mask_and_calendar_manifest_mismatch(tmp_path: Path) -> None:
    manifest = subject.assemble(*_sources(tmp_path), tmp_path / "assembled")
    with np.load(manifest.parent / "input.npz", allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    arrays["available"] = arrays["available"].astype(np.uint8)
    np.savez_compressed(manifest.parent / "input.npz", **arrays)
    payload = __import__("json").loads(manifest.read_text())
    payload["core_sha256"] = subject.sha256(manifest.parent / "input.npz")
    manifest.write_text(__import__("json").dumps(payload))
    with pytest.raises(ValueError, match="boolean dtype"):
        subject.load_manifest(manifest)
    calendar_manifest = subject.assemble(*_sources(tmp_path), tmp_path / "calendar-assembled")
    calendar_payload = __import__("json").loads(calendar_manifest.read_text())
    calendar_payload["calendar"]["length"] = 99
    calendar_manifest.write_text(__import__("json").dumps(calendar_payload))
    with pytest.raises(ValueError, match="calendar"):
        subject.load_manifest(calendar_manifest)
