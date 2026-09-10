from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from thermodense.benchmarks import pcmci_real
from thermodense.benchmarks import raw_3hour_pcmci as core


def _manifest(tmp_path: Path, *, kind: str = "fixture", f107: str = "raw_daily_held") -> Path:
    count = 16
    timestamps = np.datetime64("2020-01-01T00") + np.arange(count) * np.timedelta64(3, "h")
    density = np.arange(count * 27, dtype=float).reshape(count, 27)
    np.savez_compressed(tmp_path / "core.npz", timestamps=timestamps, density_altitudes_km=np.array(core.ALTITUDES_KM), density_log10_values=density, density_masks=np.ones_like(density, dtype=bool), f107_values=np.repeat([70., 71.], 8), f107_masks=np.ones(count, bool), ap_values=np.arange(count, dtype=float), ap_masks=np.ones(count, bool), kp_values=np.arange(count, dtype=float), kp_masks=np.ones(count, bool), mlo_co2_values=np.repeat([410., 411.], 8), mlo_co2_masks=np.ones(count, bool))
    np.savez_compressed(tmp_path / "saber.npz", timestamps=timestamps, channels=np.array(core.SABER_CHANNELS), values=np.ones((5, count)), masks=np.ones((5, count), bool))
    manifest = {"input_contract": core.VERSION, "input_kind": kind, "calendar": core.calendar_declaration(timestamps), "core_npz": "core.npz", "core_sha256": hashlib.sha256((tmp_path / "core.npz").read_bytes()).hexdigest(), "saber_npz": "saber.npz", "saber_sha256": hashlib.sha256((tmp_path / "saber.npz").read_bytes()).hexdigest(), "f10_7": {"representation": f107, "data_type": "OBS only", "non_obs": "unavailable", "interpolation": "none"}, "transformations": {"detrending": "none", "deseasonalizing": "none", "standardizing": "none", "interpolation": "none"}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_fixture_loader_preserves_raw_values_masks_and_rejects_spline(tmp_path: Path) -> None:
    loaded = core.load_manifest(_manifest(tmp_path))
    assert loaded.values.shape == (16, 36)
    assert loaded.node_names[:2] == ["log10rho_175km", "log10rho_200km"]
    with pytest.raises(ValueError, match="raw daily-held F10.7"):
        core.load_manifest(_manifest(tmp_path, f107="spline"))
    with pytest.raises(ValueError, match="exact 2000"):
        core.load_manifest(_manifest(tmp_path, kind="canonical"))


def test_loader_rejects_tampering_traversal_and_non_midnight_calendar(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    (tmp_path / "core.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="do not match"):
        core.load_manifest(manifest)

    traversal_root = tmp_path / "traversal"
    traversal_root.mkdir()
    traversal_manifest = _manifest(traversal_root)
    payload = json.loads(traversal_manifest.read_text())
    payload["core_npz"] = "../core.npz"
    traversal_manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="do not match"):
        core.load_manifest(traversal_manifest)

    duplicate_root = tmp_path / "duplicate"
    duplicate_root.mkdir()
    duplicate_manifest = _manifest(duplicate_root)
    payload = json.loads(duplicate_manifest.read_text())
    payload["saber_npz"] = payload["core_npz"]
    payload["saber_sha256"] = payload["core_sha256"]
    duplicate_manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="distinct core and SABER"):
        core.load_manifest(duplicate_manifest)

    alias_root = tmp_path / "alias"
    alias_root.mkdir()
    alias_manifest = _manifest(alias_root)
    payload = json.loads(alias_manifest.read_text())
    payload["saber_npz"] = "./core.npz"
    payload["saber_sha256"] = payload["core_sha256"]
    alias_manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="distinct core and SABER"):
        core.load_manifest(alias_manifest)

    midnight_root = tmp_path / "midnight"
    midnight_root.mkdir()
    midnight_manifest = _manifest(midnight_root)
    with np.load(midnight_root / "core.npz", allow_pickle=False) as saved:
        arrays = {name: np.asarray(saved[name]) for name in saved.files}
    arrays["timestamps"] = arrays["timestamps"] + np.timedelta64(3, "h")
    np.savez_compressed(midnight_root / "core.npz", **arrays)
    payload = json.loads(midnight_manifest.read_text())
    payload["core_sha256"] = hashlib.sha256(
        (midnight_root / "core.npz").read_bytes()
    ).hexdigest()
    midnight_manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="UTC 3-hour calendar"):
        core.load_manifest(midnight_manifest)


def test_link_family_has_required_directions_and_global_bh_rows() -> None:
    names = [*core.density_names(), *core.DRIVER_NAMES]
    assumptions = core.link_assumptions(names, 1)
    density, driver = 0, 27
    assert (driver, -1) in assumptions[density] and (driver, 0) in assumptions[density]
    assert (density, -1) not in assumptions[driver]
    assert (1, -1) in assumptions[density]
    assert np.allclose(core.bh_qvalues(np.array([.01, .02, .03])), [.03, .03, .03])


def test_result_rows_apply_one_bh_family_normalize_orientation_and_label_pairwise_support(tmp_path: Path) -> None:
    data = core.load_manifest(_manifest(tmp_path))
    result = {"graph": np.full((36, 36, 2), "-->", dtype="<U3"), "p_matrix": np.zeros((36, 36, 2)), "val_matrix": np.ones((36, 36, 2))}
    result["graph"][0, 1, 0] = "<--"
    data_available = data.available.copy()
    data_available[3, 0] = False
    data = core.Raw36Input(data.timestamps, data.values, data_available, data.node_names, data.manifest)
    rows = core.result_rows(result, data)
    assert len(rows) == len(core.eligible_locations(data.node_names, 1))
    density_lag = next(row for row in rows if row["source"] == "log10rho_175km" and row["target"] == "log10rho_175km" and row["lag_steps"] == 1)
    assert density_lag["pairwise_support"] == 13
    assert density_lag["support_scope"].startswith("pairwise availability only")
    reversed_zero = next(row for row in rows if row["graph_source"] == "log10rho_175km" and row["graph_target"] == "log10rho_200km" and row["lag_steps"] == 0)
    assert reversed_zero["source"] == "log10rho_200km"
    assert reversed_zero["target"] == "log10rho_175km"
    assert reversed_zero["graph_mark_normalized"] == "-->"
    assert reversed_zero["detected"]
    assert {row["q_value"] for row in rows} == {0.0}


def test_raw_stationarity_diagnostics_never_authorize_causal_wording(monkeypatch, tmp_path: Path) -> None:
    data = core.load_manifest(_manifest(tmp_path))
    diagnostic = {
        "nodes": {
            name: {"sample_count": 3_000, "outcome": "qualified"}
            for name in data.node_names
        },
        "settings": {},
        "causal_interpretation_eligible": True,
    }
    monkeypatch.setattr(pcmci_real, "stationarity_qualification", lambda *args: diagnostic)
    observed = core.raw_stationarity_diagnostics(data)
    assert not observed["causal_interpretation_eligible"]
    assert not observed["causal_wording_permitted"]
    assert observed["sensitivity_evidence_only"]
    assert observed["qualification_scope"].startswith("raw workflow")


def test_raw_stationarity_treats_smaller_than_kpss_bounds_as_indeterminate(monkeypatch, tmp_path: Path) -> None:
    data = core.load_manifest(_manifest(tmp_path))
    diagnostic = {
        "nodes": {
            name: {
                "sample_count": 3_000,
                "outcome": "qualified",
                "kpss": {
                    "p_value": 0.01,
                    "warnings": [
                        "The actual p-value is smaller than the p-value returned."
                    ],
                },
            }
            for name in data.node_names
        },
        "settings": {},
    }
    monkeypatch.setattr(pcmci_real, "stationarity_qualification", lambda *args: diagnostic)
    observed = core.raw_stationarity_diagnostics(data)
    assert all(
        record["outcome"] == "not_qualified_kpss_pvalue_bounded"
        for record in observed["nodes"].values()
    )
    assert set(observed["bounded_kpss_nodes"]) == set(data.node_names)
    assert observed["nodes"][data.node_names[0]]["kpss"]["p_value_kind"] == "upper_bound"


def test_tigramite_identity_requires_the_pinned_direct_url_commit(tmp_path: Path) -> None:
    direct_url = tmp_path / "direct_url.json"
    direct_url.write_text(json.dumps({"vcs_info": {"commit_id": core.TIGRAMITE_COMMIT}}))

    class Distribution:
        version = "5.2.10.1"

        def read_text(self, _name: str) -> str:
            return direct_url.read_text()

    assert core.tigramite_identity(Distribution()) == {"version": "5.2.10.1", "commit": core.TIGRAMITE_COMMIT}
    direct_url.write_text(json.dumps({"vcs_info": {"commit_id": "wrong"}}))
    with pytest.raises(ValueError, match="commit must be"):
        core.tigramite_identity(Distribution())


def test_bh_test_count_uses_the_complete_closed_form_family() -> None:
    density_count = len(core.ALTITUDES_KM)
    driver_count = len(core.DRIVER_NAMES)
    expected = (
        density_count * (density_count + driver_count) * core.TAU_MAX
        + density_count * driver_count
        + density_count * (density_count - 1) // 2
    )
    assert core.bh_test_count() == expected == 1_400_274


def test_result_matrix_validation_rejects_wrong_family_shape() -> None:
    malformed = {
        "graph": np.full((36, 36, 2), "", dtype="<U3"),
        "p_matrix": np.ones((36, 36, 2)),
        "val_matrix": np.zeros((36, 36, 2)),
    }
    with pytest.raises(ValueError, match="must have shape"):
        core.validate_result_matrices(malformed)


def test_result_matrix_validation_rejects_nonfinite_eligible_evidence() -> None:
    shape = (len(core.node_names()), len(core.node_names()), core.TAU_MAX + 1)
    result = {
        "graph": np.full(shape, "", dtype="<U1"),
        "p_matrix": np.ones(shape),
        "val_matrix": np.zeros(shape),
    }
    core.validate_result_matrices(result)
    result["p_matrix"][0, 0, 1] = np.nan
    with pytest.raises(ValueError, match="eligible p_matrix values must all be finite"):
        core.validate_result_matrices(result)


def test_run_reserves_output_and_preserves_fitted_matrices_when_export_fails(monkeypatch, tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    result = {
        "graph": np.full((2, 2, 1), "", dtype="<U3"),
        "p_matrix": np.ones((2, 2, 1)),
        "val_matrix": np.zeros((2, 2, 1)),
    }
    monkeypatch.setattr(core, "raw_stationarity_diagnostics", lambda data: {"causal_interpretation_eligible": False, "causal_wording_permitted": False})
    monkeypatch.setattr(core, "tigramite_identity", lambda: {"version": "test", "commit": core.TIGRAMITE_COMMIT})
    monkeypatch.setattr(core, "run_pcmci", lambda data: result)
    monkeypatch.setattr(core, "validate_result_matrices", lambda result: None)
    monkeypatch.setattr(core, "result_rows", lambda result, data: (_ for _ in ()).throw(ValueError("export failed")))
    output = tmp_path / "result"
    with pytest.raises(ValueError, match="export failed"):
        core.run(manifest, output, "test-host")
    assert (output / "pcmci_results.npz").is_file()
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["status"] == "fit_complete_export_failed"
    assert provenance["interpretation"].startswith("association/sensitivity-only")
    assert set(provenance["result_files"]) == {"pcmci_results.npz"}

    with pytest.raises(FileExistsError):
        core.run(manifest, output, "test-host")


def test_run_preserves_but_does_not_qualify_malformed_fitted_matrices(monkeypatch, tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    malformed = {
        "graph": np.full((2, 2, 1), "", dtype="<U3"),
        "p_matrix": np.ones((2, 2, 1)),
        "val_matrix": np.zeros((2, 2, 1)),
    }
    monkeypatch.setattr(core, "raw_stationarity_diagnostics", lambda data: {})
    monkeypatch.setattr(core, "tigramite_identity", lambda: {"version": "test", "commit": core.TIGRAMITE_COMMIT})
    monkeypatch.setattr(core, "run_pcmci", lambda data: malformed)
    output = tmp_path / "malformed-result"
    with pytest.raises(ValueError, match="must have shape"):
        core.run(manifest, output, "test-host")
    assert (output / "pcmci_results.npz").is_file()
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["status"] == "fit_complete_validation_failed"
    assert set(provenance["result_files"]) == {"pcmci_results.npz"}


def test_run_records_fit_failure_without_claiming_a_result(monkeypatch, tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    monkeypatch.setattr(core, "raw_stationarity_diagnostics", lambda data: {})
    monkeypatch.setattr(core, "tigramite_identity", lambda: {"version": "test", "commit": core.TIGRAMITE_COMMIT})
    monkeypatch.setattr(
        core,
        "run_pcmci",
        lambda data: (_ for _ in ()).throw(RuntimeError("fit interrupted")),
    )
    output = tmp_path / "failed-result"
    with pytest.raises(RuntimeError, match="fit interrupted"):
        core.run(manifest, output, "test-host")
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["status"] == "fit_failed_or_interrupted"
    assert "result_files" not in provenance
    assert not (output / "pcmci_results.npz").exists()
