"""No-GPU end-to-end coverage for the durable GPDCtorch gate runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import pytest
import shutil
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from thermodense.benchmarks import gpdctorch_gates, pcmci_real, runtime
from thermodense.benchmarks.real_data import DATE_COLUMN, F107_RAW_COLUMN, NODE_COLUMNS


HARDWARE = {"eligible": True, "name": "fixture", "total_memory_bytes": 1}


def _write_input(path: Path) -> None:
    rows = 400
    frame: dict[str, list[object]] = {
        DATE_COLUMN: [
            date(2020, 1, 1) + timedelta(days=index) for index in range(rows)
        ],
        F107_RAW_COLUMN: [float(index) for index in range(rows)],
    }
    for offset, node in enumerate(NODE_COLUMNS):
        frame[node] = [float(index + offset) for index in range(rows)]
        frame[f"{node}_imputed"] = [False] * rows
    pl.DataFrame(frame).write_csv(path)


def _args(
    tmp_path: Path, *, retry: bool = False, parcorr: Path | None = None
) -> argparse.Namespace:
    return argparse.Namespace(
        input=tmp_path / "input.csv",
        output=tmp_path / "agreement.jsonl",
        state=None,
        threads=None,
        row_limit=None,
        import_capability=None,
        retry_failed=retry,
        parcorr_agreement=parcorr,
    )


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


@pytest.mark.parametrize(
    "overlap",
    [
        "input_output",
        "input_state",
        "input_artifacts",
        "capability_artifacts",
        "parcorr_artifacts",
        "parcorr_output",
        "state_output",
    ],
)
def test_gated_run_rejects_path_overlap_before_overwrite(
    tmp_path: Path, overlap: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    args.overwrite = True
    artifacts = args.output.parent / f"{args.output.stem}_artifacts"
    artifacts.mkdir()
    source = tmp_path / f"{overlap}-source"
    source.write_bytes(b"source bytes")

    if overlap == "input_output":
        args.input = args.output = source
    elif overlap == "input_state":
        args.input = source
        args.state = source
    elif overlap == "input_artifacts":
        args.input = artifacts / "input.csv"
        args.input.write_bytes(b"source bytes")
    elif overlap == "capability_artifacts":
        args.import_capability = artifacts / "capability.json"
        args.import_capability.write_bytes(b"source bytes")
    elif overlap == "parcorr_artifacts":
        args.parcorr_agreement = artifacts / "parcorr.jsonl"
        args.parcorr_agreement.write_bytes(b"source bytes")
    elif overlap == "parcorr_output":
        args.parcorr_agreement = args.output = source
    elif overlap == "state_output":
        args.state = args.output
    else:
        raise AssertionError(overlap)

    monkeypatch.setattr(
        pcmci_real, "load_input", lambda _path: pytest.fail("loaded before validation")
    )
    with pytest.raises(ValueError):
        pcmci_real.run_gpdctorch_gated(args)

    assert source.read_bytes() == b"source bytes"
    if args.input.parent == artifacts and args.input.exists():
        assert args.input.read_bytes() == b"source bytes"
    if args.import_capability is not None and args.import_capability.exists():
        assert args.import_capability.read_bytes() == b"source bytes"
    if args.parcorr_agreement is not None and args.parcorr_agreement.exists():
        assert args.parcorr_agreement.read_bytes() == b"source bytes"


def _child(calls: list[tuple[str, int]], command, timeout, _threads, _environment):
    tau_max = int(_option(command, "--tau-max"))
    timing = _option(command, "--timing-variant")
    profile = _option(command, "--preprocessing-profile")
    artifact = Path(_option(command, "--artifact"))
    stage = next(
        stage
        for stage in gpdctorch_gates.stages()
        if (stage.tau_max, stage.timing_variant, stage.preprocessing_profile)
        == (tau_max, timing, profile)
    )
    calls.append((stage.name, timeout))
    nodes = pcmci_real._expected_node_order(
        pcmci_real.sensitivity_case(timing, profile)
    )
    matrices = {
        "graph": np.full((5, 5, tau_max + 1), "", dtype="<U3"),
        "p_matrix": np.ones((5, 5, tau_max + 1)),
        "val_matrix": np.zeros((5, 5, tau_max + 1)),
    }
    reference = runtime.write_npz_artifact(artifact, matrices, node_names=nodes)
    return {
        "status": "succeeded",
        "wall_seconds": 1,
        "failure_reason": None,
        "artifact": reference,
        "result_digest": runtime.compact_result_digest(matrices),
        "gpdctorch_lifecycle": {
            "tigramite_pin": "7c7b177cfbff77e11d805ab04fc2647301da1951"
        },
    }


def _parcorr_agreement(tmp_path: Path, accepted_rows: dict[str, object]) -> Path:
    rows = []
    for case in pcmci_real.expand_sensitivity_cases():
        nodes = pcmci_real._expected_node_order(case)
        matrices = {
            "graph": np.full((5, 5, 181), "", dtype="<U3"),
            "p_matrix": np.ones((5, 5, 181)),
            "val_matrix": np.zeros((5, 5, 181)),
        }
        artifact = runtime.write_npz_artifact(
            tmp_path
            / f"parcorr-{case.timing_variant}-{case.preprocessing_profile}.npz",
            matrices,
            node_names=nodes,
        )
        rows.append(
            {
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
                "settings": {
                    "pc_alpha": 0.05,
                    "significance": "analytic",
                    "threads": 1,
                },
                "missing_data_policy": {
                    "sentinel": pcmci_real.MISSING_FLAG,
                    "remove_missing_upto_maxlag": False,
                    "drivers_interpolated": False,
                    "rows_dropped": False,
                },
                "link_assumptions": pcmci_real._link_assumption_metadata(180, nodes),
                "sensitivity_case": {
                    "timing_variant": case.timing_variant,
                    "preprocessing_profile": case.preprocessing_profile,
                    "role": case.role,
                    "node_order": nodes,
                    "accepted_quality_rows": accepted_rows,
                },
                "stationarity_qualification": {
                    "causal_interpretation_eligible": True,
                    "sensitivity_evidence_only": False,
                    "provenance_identity": {
                        "timing_variant": case.timing_variant,
                        "preprocessing_profile": case.preprocessing_profile,
                        "node_order": nodes,
                        "daily_date_sequence_sha256": accepted_rows[
                            "daily_date_sequence_sha256"
                        ],
                        "common_f107_support_sha256": accepted_rows[
                            "common_f107_support"
                        ]["sha256"],
                    },
                },
                "causal_interpretation_eligible": True,
                "sensitivity_evidence_only": False,
            }
        )
    path = tmp_path / "parcorr.jsonl"
    path.write_text(json.dumps(pcmci_real.synthesize_parcorr_matrix(rows)) + "\n")
    return path


def test_gated_runner_completes_resumes_and_regenerates_only_derived(
    tmp_path: Path, monkeypatch
) -> None:
    _write_input(tmp_path / "input.csv")
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(gpdctorch_gates, "gpu_hardware", lambda: HARDWARE)
    monkeypatch.setattr(
        runtime, "run_isolated_process", lambda *args: _child(calls, *args)
    )

    args = _args(tmp_path)
    assert pcmci_real.run_gpdctorch_gated(args) == 0
    assert [name for name, _ in calls] == list(gpdctorch_gates.STAGES)
    assert [timeout for _, timeout in calls] == [18000, 86400, 86400, 86400, 86400]
    state_path = tmp_path / "agreement_artifacts" / "gpdctorch-gates.json"
    state = json.loads(state_path.read_text())
    assert state["state"] == state["derived"]["status"] == "complete"
    assert (
        json.loads(Path(state["derived"]["comparison"]["path"]).read_text())["state"]
        == "pending_parcorr"
    )

    assert pcmci_real.run_gpdctorch_gated(args) == 0
    assert len(calls) == 5
    Path(state["derived"]["comparison"]["path"]).unlink()
    assert pcmci_real.run_gpdctorch_gated(args) == 0
    assert len(calls) == 5


def test_parcorr_request_identity_rejects_invalid_and_regenerates_without_fits(
    tmp_path: Path, monkeypatch
) -> None:
    _write_input(tmp_path / "input.csv")
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(gpdctorch_gates, "gpu_hardware", lambda: HARDWARE)
    monkeypatch.setattr(
        runtime, "run_isolated_process", lambda *args: _child(calls, *args)
    )
    args = _args(tmp_path)
    assert pcmci_real.run_gpdctorch_gated(args) == 0
    output = args.output.read_text()
    invalid = tmp_path / "invalid-parcorr.jsonl"
    invalid.write_text("{}\n")
    args.parcorr_agreement = invalid
    assert pcmci_real.run_gpdctorch_gated(args) == 1
    assert len(calls) == 5
    assert args.output.read_text() == output

    state = json.loads(
        (tmp_path / "agreement_artifacts" / "gpdctorch-gates.json").read_text()
    )
    valid = _parcorr_agreement(tmp_path, state["identity"]["accepted_quality_rows"])
    args.parcorr_agreement = valid
    assert pcmci_real.run_gpdctorch_gated(args) == 0
    assert len(calls) == 5
    comparison = json.loads(
        Path(
            json.loads(
                (tmp_path / "agreement_artifacts" / "gpdctorch-gates.json").read_text()
            )["derived"]["comparison"]["path"]
        ).read_text()
    )
    assert comparison["state"] == "complete"


def test_timeouts_retry_from_the_failed_gated_stage(
    tmp_path: Path, monkeypatch
) -> None:
    _write_input(tmp_path / "input.csv")
    calls: list[tuple[str, int]] = []
    failure = {"stage": "primary"}
    monkeypatch.setattr(gpdctorch_gates, "gpu_hardware", lambda: HARDWARE)

    def child(*child_args):
        command, timeout = child_args[:2]
        stage = next(
            stage
            for stage in gpdctorch_gates.stages()
            if str(stage.tau_max) == _option(command, "--tau-max")
            and stage.timing_variant == _option(command, "--timing-variant")
            and stage.preprocessing_profile
            == _option(command, "--preprocessing-profile")
        )
        if stage.name == failure["stage"]:
            calls.append((stage.name, timeout))
            failure["stage"] = ""
            return {"status": "timeout", "failure_reason": "fixture"}
        return _child(calls, *child_args)

    monkeypatch.setattr(runtime, "run_isolated_process", child)
    args = _args(tmp_path)
    assert pcmci_real.run_gpdctorch_gated(args) == 1
    assert [name for name, _ in calls] == ["capability", "primary"]
    args.retry_failed = True
    assert pcmci_real.run_gpdctorch_gated(args) == 0
    assert [name for name, _ in calls] == [
        "capability",
        "primary",
        "primary",
        "raw_seasonal",
        "centered_detrended",
        "interaction",
    ]
    state = json.loads(
        (tmp_path / "agreement_artifacts" / "gpdctorch-gates.json").read_text()
    )
    assert len(state["stages"]["capability"]["attempts"]) == 1
    assert len(state["stages"]["primary"]["attempts"]) == 2


def test_stale_successful_primary_blocks_then_retry_reuses_later_fits(
    tmp_path: Path, monkeypatch
) -> None:
    _write_input(tmp_path / "input.csv")
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(gpdctorch_gates, "gpu_hardware", lambda: HARDWARE)
    monkeypatch.setattr(
        runtime, "run_isolated_process", lambda *args: _child(calls, *args)
    )
    args = _args(tmp_path)
    assert pcmci_real.run_gpdctorch_gated(args) == 0
    state_path = tmp_path / "agreement_artifacts" / "gpdctorch-gates.json"
    state = json.loads(state_path.read_text())
    primary_artifact = Path(state["stages"]["primary"]["result"]["artifact"]["path"])
    primary_artifact.write_bytes(b"tampered")
    before = [len(state["stages"][name]["attempts"]) for name in gpdctorch_gates.STAGES]
    assert pcmci_real.run_gpdctorch_gated(args) == 1
    blocked = json.loads(state_path.read_text())
    assert blocked["stages"]["primary"]["status"] == "stale"
    assert [
        len(blocked["stages"][name]["attempts"]) for name in gpdctorch_gates.STAGES
    ] == before
    args.retry_failed = True
    assert pcmci_real.run_gpdctorch_gated(args) == 0
    assert [name for name, _ in calls] == [*gpdctorch_gates.STAGES, "primary"]
    resumed = json.loads(state_path.read_text())
    assert len(resumed["stages"]["primary"]["attempts"]) == 2
    assert all(
        len(resumed["stages"][name]["attempts"]) == 1
        for name in gpdctorch_gates.STAGES[2:]
    )


def test_child_rejects_a_tampered_predecessor_after_authorization(
    tmp_path: Path, monkeypatch
) -> None:
    _write_input(tmp_path / "input.csv")
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(gpdctorch_gates, "gpu_hardware", lambda: HARDWARE)
    monkeypatch.setattr(
        runtime, "run_isolated_process", lambda *args: _child(calls, *args)
    )
    args = _args(tmp_path)
    assert pcmci_real.run_gpdctorch_gated(args) == 0
    state_path = tmp_path / "agreement_artifacts" / "gpdctorch-gates.json"
    state = json.loads(state_path.read_text())
    state["stages"]["primary"]["status"] = "running"
    authorization = "authorized"
    state["stages"]["primary"]["attempts"].append(
        {"authorization_sha256": hashlib.sha256(authorization.encode()).hexdigest()}
    )
    state_path.write_text(json.dumps(state))
    Path(state["stages"]["capability"]["result"]["artifact"]["path"]).write_bytes(
        b"tampered"
    )
    monkeypatch.setenv("THERMODENSE_GPDC_GATE_AUTH", authorization)

    try:
        pcmci_real._validate_gated_child(
            argparse.Namespace(
                gate_state=state_path,
                gate_threads=1,
                gate_output=args.output.resolve(),
                input=args.input,
                tau_max=10,
                timing_variant="raw_observed_daily",
                preprocessing_profile="detrended_anomaly",
            )
        )
    except ValueError as error:
        assert "predecessor" in str(error)
    else:
        raise AssertionError("tampered predecessor artifact was accepted")


def test_fresh_gated_run_refuses_existing_output_without_overwrite(
    tmp_path: Path,
) -> None:
    _write_input(tmp_path / "input.csv")
    args = _args(tmp_path)
    args.output.write_text("unrelated\n")

    try:
        pcmci_real.run_gpdctorch_gated(args)
    except ValueError as error:
        assert "--overwrite" in str(error)
    else:
        raise AssertionError("unrelated output was accepted")
    assert args.output.read_text() == "unrelated\n"


def test_overwrite_resets_explicit_state_and_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    _write_input(tmp_path / "input.csv")
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(gpdctorch_gates, "gpu_hardware", lambda: HARDWARE)
    monkeypatch.setattr(
        runtime, "run_isolated_process", lambda *args: _child(calls, *args)
    )
    args = _args(tmp_path)
    args.overwrite = True
    args.output.write_text("old output\n")
    artifacts = tmp_path / "agreement_artifacts"
    artifacts.mkdir()
    stale = artifacts / "stale"
    stale.write_text("stale")
    args.state = tmp_path / "external-state.json"
    args.state.write_text("old state")

    assert pcmci_real.run_gpdctorch_gated(args) == 0
    assert not stale.exists()
    assert json.loads(args.state.read_text())["state"] == "complete"


def test_gate_identity_rejects_a_different_output_path(
    tmp_path: Path, monkeypatch
) -> None:
    _write_input(tmp_path / "input.csv")
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(gpdctorch_gates, "gpu_hardware", lambda: HARDWARE)
    monkeypatch.setattr(
        runtime, "run_isolated_process", lambda *args: _child(calls, *args)
    )
    args = _args(tmp_path)
    assert pcmci_real.run_gpdctorch_gated(args) == 0
    second = _args(tmp_path)
    second.output = tmp_path / "other.jsonl"
    second.state = tmp_path / "agreement_artifacts" / "gpdctorch-gates.json"

    try:
        pcmci_real.run_gpdctorch_gated(second)
    except gpdctorch_gates.GateError as error:
        assert "identity" in str(error)
    else:
        raise AssertionError("state was reused for a different output path")


def test_child_without_gate_secret_returns_a_failure_payload(capsys) -> None:
    status = pcmci_real._child_main(
        argparse.Namespace(method="gpdctorch", gate_state=None)
    )

    assert status == 1
    assert json.loads(capsys.readouterr().out)["status"] == "failed"


def test_parcorr_validator_rejects_hash_valid_malformed_npz(tmp_path: Path) -> None:
    accepted = {
        "daily_date_sequence_sha256": "a" * 64,
        "row_count": 400,
        "common_f107_support": {"sha256": "b" * 64, "row_count": 400},
    }
    path = _parcorr_agreement(tmp_path, accepted)
    agreement = json.loads(path.read_text())
    forged = Path(agreement["case_artifacts"][0]["path"])
    forged.write_bytes(b"not an npz")
    agreement["case_artifacts"][0]["sha256"] = hashlib.sha256(
        forged.read_bytes()
    ).hexdigest()
    path.write_text(json.dumps(agreement) + "\n")

    try:
        pcmci_real._validate_parcorr_agreement(path, accepted)
    except ValueError as error:
        assert "artifact" in str(error)
    else:
        raise AssertionError("hash-valid malformed ParCorr NPZ was accepted")


def _legacy_evidence(tmp_path: Path, monkeypatch) -> dict[str, object]:
    _write_input(tmp_path / "input.csv")
    monkeypatch.setattr(gpdctorch_gates, "gpu_hardware", lambda: HARDWARE)
    monkeypatch.setattr(
        runtime, "run_isolated_process", lambda *args: _child([], *args)
    )
    seed_args = _args(tmp_path)
    assert pcmci_real.run_gpdctorch_gated(seed_args) == 0
    state = json.loads(
        (tmp_path / "agreement_artifacts" / "gpdctorch-gates.json").read_text()
    )
    row = state["stages"]["capability"]["result"].copy()
    legacy_artifact = tmp_path / "legacy-capability.npz"
    shutil.copy2(Path(row["artifact"]["path"]), legacy_artifact)
    row["artifact"] = row["artifact"] | {
        "path": str(legacy_artifact),
        "name": legacy_artifact.name,
    }
    for key in ("hardware", "gpdctorch_lifecycle", "gate_stage"):
        row.pop(key, None)
    row.update(
        host_label="legacy-host",
        environment_label="legacy-rtx2060",
        environment_fingerprint="legacy-driver-pin-fingerprint",
    )
    shutil.rmtree(tmp_path / "agreement_artifacts")
    (tmp_path / "agreement.jsonl").unlink()
    return row


def test_real_validator_imports_legacy_capability_and_runs_tau10_stages(
    tmp_path: Path, monkeypatch
) -> None:
    row = _legacy_evidence(tmp_path, monkeypatch)
    evidence = tmp_path / "legacy.jsonl"
    evidence.write_text(json.dumps(row) + "\n")
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(gpdctorch_gates, "_legacy_tigramite_pin", lambda *_args: True)
    monkeypatch.setattr(
        runtime, "run_isolated_process", lambda *args: _child(calls, *args)
    )
    args = _args(tmp_path)
    args.import_capability = evidence
    assert pcmci_real.run_gpdctorch_gated(args) == 0
    assert [name for name, _ in calls] == list(gpdctorch_gates.STAGES[1:])
    state = json.loads(
        (tmp_path / "agreement_artifacts" / "gpdctorch-gates.json").read_text()
    )
    capability = state["stages"]["capability"]
    assert capability["result"] == row
    assert capability["capability_evidence"]["mode"] == "legacy_environment_attestation"
    assert "hardware" not in capability["result"]
    assert "capability_evidence" not in capability["result"]


@pytest.mark.parametrize(
    "field,value", [("pin_proven", False), ("original_git_commit", "tampered")]
)
def test_legacy_import_re_attestation_blocks_tampered_resume(
    tmp_path: Path, monkeypatch, field: str, value: object
) -> None:
    row = _legacy_evidence(tmp_path, monkeypatch)
    evidence = tmp_path / "legacy.jsonl"
    evidence.write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(gpdctorch_gates, "_legacy_tigramite_pin", lambda *_args: True)
    monkeypatch.setattr(
        runtime, "run_isolated_process", lambda *args: _child([], *args)
    )
    args = _args(tmp_path)
    args.import_capability = evidence
    assert pcmci_real.run_gpdctorch_gated(args) == 0
    state_path = tmp_path / "agreement_artifacts" / "gpdctorch-gates.json"
    state = json.loads(state_path.read_text())
    state["stages"]["capability"]["capability_evidence"][field] = value
    state_path.write_text(json.dumps(state))
    args.import_capability = None
    assert pcmci_real.run_gpdctorch_gated(args) == 1
