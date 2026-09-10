from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from thermodense.benchmarks import raw_3hour_input as assembler
from thermodense.benchmarks import raw_3hour_pcmci as contract


def _sources(tmp_path: Path, *, legacy_names: tuple[str, ...] = assembler.LEGACY_CHANNELS, saber_timestamps: np.ndarray | None = None) -> tuple[Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    timestamps = np.datetime64("2020-01-01T00") + np.arange(16) * np.timedelta64(3, "h")
    legacy = tmp_path / "prepared_input.npz"
    values = np.column_stack((np.arange(16 * 27, dtype=float).reshape(16, 27), np.arange(16, dtype=float) + 900, np.arange(16, dtype=float), np.arange(16, dtype=float) / 3, np.repeat([410.0, 411.0], 8), np.zeros((16, 5))))
    np.savez_compressed(legacy, time_ns=timestamps.astype("datetime64[ns]").astype("int64"), values=values, channel_names=np.asarray(legacy_names))
    weather = tmp_path / "SW-All.csv"
    weather.write_text("DATE,F10.7_OBS,F10.7_DATA_TYPE\n2020-01-01,70,OBS\n2020-01-02,71,OBS\n")
    saber = tmp_path / "saber-50x70.npz"
    np.savez_compressed(saber, timestamps=timestamps if saber_timestamps is None else saber_timestamps, channels=np.asarray(contract.SABER_CHANNELS), values=np.ones((5, 16)), masks=np.ones((5, 16), dtype=bool))
    return legacy, weather, saber


def test_assembler_replaces_legacy_spline_and_emits_loader_contract(tmp_path: Path) -> None:
    legacy, weather, saber = _sources(tmp_path)
    manifest = assembler.assemble(legacy, weather, saber, tmp_path / "output", input_kind="fixture")

    loaded = contract.load_manifest(manifest)
    assert np.array_equal(loaded.values[:, 27], np.repeat([70.0, 71.0], 8))
    assert np.array_equal(loaded.values[:, 28], np.arange(16, dtype=float))
    assert np.array_equal(loaded.values[:, 30], np.repeat([410.0, 411.0], 8))
    note = json.loads((manifest.parent / "provenance.json").read_text())
    manifest_payload = json.loads(manifest.read_text())
    assert note["source_hashes"]["legacy_prepared_npz"]["discarded_channel"].startswith("F10.7_raw")
    assert (manifest.parent / "core.npz").is_file()
    assert (manifest.parent / "saber.npz").is_file()
    assert manifest_payload["saber_npz"] == "saber.npz"
    assert manifest_payload["core_sha256"] == note["outputs"]["core.npz"]
    assert manifest_payload["saber_sha256"] == note["outputs"]["saber.npz"]
    assert manifest_payload["f10_7"]["data_type"] == "OBS only"
    assert manifest_payload["legacy_prepared_semantics"]["status"] == "fixture_unverified"


def test_assembler_keeps_non_obs_f107_missing_and_requires_data_type(tmp_path: Path) -> None:
    legacy, weather, saber = _sources(tmp_path)
    weather.write_text("DATE,F10.7_OBS,F10.7_DATA_TYPE\n2020-01-01,70,PRD\n2020-01-02,71,OBS\n")
    manifest = assembler.assemble(legacy, weather, saber, tmp_path / "output", input_kind="fixture")
    loaded = contract.load_manifest(manifest)
    assert not loaded.available[:8, 27].any()
    assert loaded.available[8:, 27].all()

    legacy, weather, saber = _sources(tmp_path / "missing-type")
    weather.write_text("DATE,F10.7_OBS\n2020-01-01,70\n")
    with pytest.raises(ValueError, match="F10.7_DATA_TYPE"):
        assembler.assemble(legacy, weather, saber, tmp_path / "missing-type-output", input_kind="fixture")


def test_assembler_rejects_source_mutated_after_capture(monkeypatch, tmp_path: Path) -> None:
    legacy, weather, saber = _sources(tmp_path)
    original = assembler._saber_input

    def mutate_then_read(path: Path, timestamps: np.ndarray):
        result = original(path, timestamps)
        saber.write_bytes(b"changed after validated read")
        return result

    monkeypatch.setattr(assembler, "_saber_input", mutate_then_read)
    with pytest.raises(ValueError, match="source changed"):
        assembler.assemble(legacy, weather, saber, tmp_path / "output", input_kind="fixture")


def test_legacy_provenance_binds_the_prepared_input_semantics(tmp_path: Path) -> None:
    legacy, _weather, _saber = _sources(tmp_path)
    timestamps, *_ = assembler._legacy_input(legacy)
    digest = assembler._sha256(legacy)
    record = {
        "identity": {
            "version": assembler.LEGACY_PROVENANCE_VERSION,
            "source_sha256": {
                "analysis_bundle.npz": "0" * 64,
                "final_arrays.npz": "1" * 64,
                "final_report.json": "2" * 64,
                "co2_daily_mlo.csv": "3" * 64,
            },
            "channels": list(assembler.LEGACY_CHANNELS),
        },
        "prepared_sha256": digest,
        "shape": [len(timestamps), len(assembler.LEGACY_CHANNELS)],
        "cadence": "3-hour",
        "f107_policy": assembler.LEGACY_F107_POLICY,
        "co2_policy": assembler.LEGACY_CO2_MISSING_POLICY,
    }
    legacy.with_name(assembler.LEGACY_PROVENANCE_FILENAME).write_text(json.dumps(record))
    semantics = assembler._legacy_provenance(legacy, digest, timestamps, required=True)
    assert semantics["prepared_artifact_status"] == "verified"
    assert semantics["upstream_semantics_status"].startswith("declared_by")
    record["prepared_sha256"] = "wrong"
    legacy.with_name(assembler.LEGACY_PROVENANCE_FILENAME).write_text(json.dumps(record))
    with pytest.raises(ValueError, match="does not bind"):
        assembler._legacy_provenance(legacy, digest, timestamps, required=True)


def test_assembler_rejects_source_order_timestamp_mismatch_and_overwrite(tmp_path: Path) -> None:
    legacy, weather, saber = _sources(tmp_path, legacy_names=("wrong", *assembler.LEGACY_CHANNELS[1:]))
    with pytest.raises(ValueError, match="node order"):
        assembler.assemble(legacy, weather, saber, tmp_path / "bad-order", input_kind="fixture")

    legacy, weather, saber = _sources(tmp_path / "matching", saber_timestamps=np.datetime64("2020-01-01T03") + np.arange(16) * np.timedelta64(3, "h"))
    with pytest.raises(ValueError, match="exactly match"):
        assembler.assemble(legacy, weather, saber, tmp_path / "bad-time", input_kind="fixture")

    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        assembler.assemble(legacy, weather, saber, output, input_kind="fixture")


def test_assembler_rejects_noncanonical_length_and_nonheld_legacy_co2(tmp_path: Path) -> None:
    legacy, weather, saber = _sources(tmp_path)
    with pytest.raises(ValueError, match="exact 2000"):
        assembler.assemble(legacy, weather, saber, tmp_path / "short", input_kind="canonical")

    with np.load(legacy, allow_pickle=False) as saved:
        values = np.asarray(saved["values"]).copy()
        timestamps = np.asarray(saved["time_ns"])
        names = np.asarray(saved["channel_names"])
    values[1, 30] = 499.0
    np.savez_compressed(legacy, time_ns=timestamps, values=values, channel_names=names)
    with pytest.raises(ValueError, match="MLO CO2 values are not held"):
        assembler.assemble(legacy, weather, saber, tmp_path / "bad-co2", input_kind="fixture")
