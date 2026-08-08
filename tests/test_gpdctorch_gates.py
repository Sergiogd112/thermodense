from __future__ import annotations

from pathlib import Path
import json

import pytest

from thermodense.benchmarks import gpdctorch_gates as gates


IDENTITY = {
    "input_sha256": "a" * 64,
    "common_support": "b" * 64,
    "settings": {"pc_alpha": 0.05},
}
HARDWARE = {"eligible": True, "name": "fixture", "total_memory_bytes": 1}


def test_gates_are_ordered_resumable_and_never_use_a_prefix(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        gates,
        "environment_identity",
        lambda: {"git_commit": "pin", "package_versions": {}},
    )
    calls = []

    def execute(stage, _authorization):
        calls.append(stage)
        return {"status": "succeeded", "wall_seconds": 1, "row_limit": None}

    state = gates.run(tmp_path / "state.json", IDENTITY, HARDWARE, execute)

    assert [stage.name for stage in calls] == list(gates.STAGES)
    assert [(stage.tau_max, stage.timeout_seconds) for stage in calls[:2]] == [
        (1, 18000),
        (10, 86400),
    ]
    assert state["state"] == "complete"
    gates.run(tmp_path / "state.json", IDENTITY, HARDWARE, execute)
    assert len(calls) == 5


def test_failure_blocks_downstream_until_explicit_retry(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        gates,
        "environment_identity",
        lambda: {"git_commit": "pin", "package_versions": {}},
    )
    calls = []

    def fail_capability(stage, _authorization):
        calls.append(stage.name)
        return {"status": "timeout", "failure_reason": "fixture"}

    path = tmp_path / "state.json"
    assert gates.run(path, IDENTITY, HARDWARE, fail_capability)["state"] == "blocked"
    assert calls == ["capability"]
    assert gates.run(path, IDENTITY, HARDWARE, fail_capability)["state"] == "blocked"
    assert calls == ["capability"]
    assert (
        gates.run(
            path,
            IDENTITY,
            HARDWARE,
            lambda _stage, _authorization: {"status": "succeeded"},
            retry_failed=True,
        )["state"]
        == "complete"
    )


def test_rejects_ineligible_and_stale_identity(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        gates,
        "environment_identity",
        lambda: {"git_commit": "pin", "package_versions": {}},
    )
    with pytest.raises(gates.GateError, match="eligible CUDA"):
        gates.run(
            tmp_path / "state.json",
            IDENTITY,
            {"eligible": False, "reason": "none"},
            lambda _stage, _authorization: {},
        )
    path = tmp_path / "state.json"
    gates.run(
        path, IDENTITY, HARDWARE, lambda _stage, _authorization: {"status": "succeeded"}
    )
    with pytest.raises(gates.GateError, match="identity"):
        gates.run(
            path,
            IDENTITY | {"input_sha256": "c" * 64},
            HARDWARE,
            lambda _stage, _authorization: {},
        )


def test_import_rejects_prefix_and_stale_capability_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    environment = {"git_commit": "pin", "package_versions": {}}
    monkeypatch.setattr(gates, "environment_identity", lambda: environment)
    evidence = {
        "status": "succeeded",
        "method": "gpdctorch",
        "tau_max": 1,
        "wall_seconds": 2,
        "input": {"input_sha256": IDENTITY["input_sha256"], "row_limit": 2},
        "sensitivity_case": {
            "timing_variant": "raw_observed_daily",
            "preprocessing_profile": "detrended_anomaly",
            "role": "primary",
            "accepted_quality_rows": IDENTITY.get("accepted_quality_rows"),
        },
        "settings": IDENTITY["settings"],
        "runner_version": None,
        "git_commit": "pin",
        "package_versions": {},
        "hardware": HARDWARE,
        "artifact": {"path": "fixture", "sha256": "a" * 64},
    }
    path = tmp_path / "evidence.jsonl"
    path.write_text(json.dumps(evidence) + "\n")
    with pytest.raises(gates.GateError, match="stale"):
        gates.import_capability(
            tmp_path / "state.json",
            path,
            IDENTITY,
            HARDWARE,
            validate_success=lambda _record: False,
        )


def test_invalid_success_is_stale_and_retried_without_replaying_valid_stages(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        gates,
        "environment_identity",
        lambda: {"git_commit": "pin", "package_versions": {}},
    )
    path = tmp_path / "state.json"
    gates.run(path, IDENTITY, HARDWARE, lambda *_args: {"status": "succeeded"})
    state = json.loads(path.read_text())
    state["stages"]["primary"]["result"] = {"invalid": True}
    path.write_text(json.dumps(state))
    calls = []

    def execute(stage, _authorization):
        calls.append(stage.name)
        return {"status": "succeeded"}

    state = gates.run(
        path,
        IDENTITY,
        HARDWARE,
        execute,
        retry_failed=True,
        validate_success=lambda record: (
            record.get("result", {}).get("invalid") is not True
        ),
    )

    assert calls == ["primary"]
    assert state["state"] == "complete"
    assert len(state["stages"]["primary"]["attempts"]) == 2
    assert len(state["stages"]["capability"]["attempts"]) == 1


def test_interrupted_attempt_is_finished_and_blocks_without_retry(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        gates,
        "environment_identity",
        lambda: {"git_commit": "pin", "package_versions": {}},
    )
    state = gates.new_state(IDENTITY, HARDWARE)
    record = state["stages"]["capability"]
    record.update(status="running", identity=IDENTITY)
    record["attempts"].append({"status": "running"})
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state))

    loaded = gates.load_or_create(path, IDENTITY, HARDWARE, retry_failed=False)

    assert loaded["state"] == "blocked"
    assert record is not loaded["stages"]["capability"]
    attempt = loaded["stages"]["capability"]["attempts"][-1]
    assert attempt["status"] == "interrupted"
    assert attempt["finished_at"]
    assert attempt["failure_reason"]


def test_invalid_child_success_is_immediately_stale_and_blocked(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        gates,
        "environment_identity",
        lambda: {"git_commit": "pin", "package_versions": {}},
    )

    state = gates.run(
        tmp_path / "state.json",
        IDENTITY,
        HARDWARE,
        lambda *_args: {"status": "succeeded", "tampered": True},
        validate_success=lambda record: record["result"].get("tampered") is not True,
    )

    assert state["state"] == "blocked"
    assert state["stages"]["capability"]["status"] == "stale"
    assert state["stages"]["capability"]["attempts"][-1]["status"] == "succeeded"


def test_legacy_import_attests_environment_without_claiming_hardware_equality(
    tmp_path: Path, monkeypatch
) -> None:
    environment = {"git_commit": "current", "package_versions": {"torch": "x"}}
    identity = IDENTITY | {
        "tigramite_pin": "pin",
        "runner_version": "runner",
        "settings": {"threads": 1},
    }
    monkeypatch.setattr(gates, "environment_identity", lambda: environment)
    monkeypatch.setattr(gates, "_legacy_tigramite_pin", lambda *_args: True)
    row = {
        "host_label": "gpu-host",
        "environment_label": "gpu-env",
        "environment_fingerprint": "fingerprint",
        "git_commit": "evidence",
    }
    evidence = tmp_path / "evidence.jsonl"
    evidence.write_text(json.dumps(row) + "\n")
    validated = []

    state = gates.import_capability(
        tmp_path / "state.json",
        evidence,
        identity,
        HARDWARE,
        validate_success=lambda record: validated.append(record) is None,
    )

    assert validated[0]["result"] == row
    assert state["stages"]["capability"]["capability_evidence"] == {
        "mode": "legacy_environment_attestation",
        "original_labels": {
            "host_label": "gpu-host",
            "environment_label": "gpu-env",
            "environment_fingerprint": "fingerprint",
        },
        "original_git_commit": "evidence",
        "tigramite_pin": "pin",
        "pin_proven": True,
        "current_hardware": HARDWARE,
        "current_environment": environment,
    }
    attempt = state["stages"]["capability"]["attempts"][-1]
    assert attempt["status"] == "succeeded"
    assert attempt["started_at"] == attempt["finished_at"]
    assert [(item["from"], item["to"]) for item in state["transitions"][-2:]] == [
        ("pending", "running"),
        ("running", "pending"),
    ]
    calls = []
    gates.run(
        tmp_path / "state.json",
        identity,
        HARDWARE,
        lambda stage, _authorization: (
            calls.append(stage.name) or {"status": "succeeded"}
        ),
        validate_success=lambda _record: True,
    )
    assert calls == ["primary", "raw_seasonal", "centered_detrended", "interaction"]


@pytest.mark.parametrize(
    "mutation",
    [
        {"host_label": "unspecified"},
        {"gpdctorch_lifecycle": {}},
    ],
)
def test_legacy_import_rejects_malformed_environment_attestation(
    tmp_path: Path, monkeypatch, mutation: dict[str, object]
) -> None:
    identity = IDENTITY | {"tigramite_pin": "pin"}
    monkeypatch.setattr(
        gates,
        "environment_identity",
        lambda: {"git_commit": "current", "package_versions": {}},
    )
    monkeypatch.setattr(gates, "_legacy_tigramite_pin", lambda *_args: True)
    evidence = tmp_path / "evidence.jsonl"
    evidence.write_text(
        json.dumps(
            {
                "host_label": "gpu-host",
                "environment_label": "gpu-env",
                "environment_fingerprint": "fingerprint",
                "git_commit": "evidence",
            }
            | mutation
        )
        + "\n"
    )

    with pytest.raises(gates.GateError, match="environment-attested"):
        gates.import_capability(
            tmp_path / "state.json",
            evidence,
            identity,
            HARDWARE,
            validate_success=lambda _record: True,
        )


def test_structured_import_requires_exact_current_hardware(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        gates,
        "environment_identity",
        lambda: {"git_commit": "pin", "package_versions": {}},
    )
    evidence = tmp_path / "evidence.jsonl"
    evidence.write_text(json.dumps({"hardware": HARDWARE | {"name": "other"}}) + "\n")

    with pytest.raises(gates.GateError, match="stale"):
        gates.import_capability(
            tmp_path / "state.json",
            evidence,
            IDENTITY,
            HARDWARE,
            validate_success=lambda record: (
                record["result"].get("hardware") == HARDWARE
            ),
        )
