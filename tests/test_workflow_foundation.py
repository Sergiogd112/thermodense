from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re
import subprocess
from typing import Any

import pytest

from thermodense import checkpoints
from thermodense.cli import main
from thermodense.engine import StageSpec, WorkflowEngine, fresh_run_id
from thermodense.paths import (
    prepared_root,
    product_root,
    publication_root,
    runs_root,
    source_root,
)
from thermodense.ssh import (
    SSHExecutionAdapter,
    SSHProfileError,
    load_profile,
)
from thermodense.workflows import (
    STAGES,
    Workflow,
    WorkflowError,
    load_workflow,
    stage_range,
)


def demo_workflow(tmp_path: Path, stages: tuple[str, ...] = ("acquire",)) -> Workflow:
    config = b"workflow = 'test-demo'\n"
    config_path = tmp_path / "demo.toml"
    config_path.write_bytes(config)
    return Workflow(
        "test-demo", "test only", "migrated for tests", stages, config_path, config
    )


def test_cli_lists_and_plans_migration_state(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["workflows"]) == 0
    assert "global-mean" in capsys.readouterr().out
    assert main(["plan", "tudelft-density"]) == 0
    output = capsys.readouterr().out
    assert "acquire: unavailable" in output


def test_import_resolves_to_the_src_package() -> None:
    import thermodense

    assert thermodense.__file__ is not None
    assert Path(thermodense.__file__).parent.name == "thermodense"
    assert not Path("thermodense.py").exists()


def test_unavailable_stage_refuses_execution(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["reproduce", "global-mean", "--dry-run"]) == 2
    assert "unavailable" in capsys.readouterr().err
    assert main(["stage", "global-mean", "acquire", "--dry-run"]) == 2
    assert "unavailable" in capsys.readouterr().err


def test_stage_ranges_are_ordered() -> None:
    assert stage_range("prepare", "analyse") == ("prepare", "derive", "analyse")
    with pytest.raises(WorkflowError, match="must not come after"):
        stage_range("publish", "acquire")


def test_checkpoint_reuse_and_all_provenance_invalidations(tmp_path: Path) -> None:
    workflow = demo_workflow(tmp_path)
    source = tmp_path / "input.txt"
    source.write_text("one")
    output = tmp_path / "output.txt"
    implementation = tmp_path / "implementation.py"
    implementation.write_text("v1")
    calls: list[str] = []

    def action() -> None:
        calls.append("run")
        output.write_text(source.read_text())

    spec = {"acquire": StageSpec((source,), (output,), (implementation,), action)}
    engine = WorkflowEngine(tmp_path)
    assert engine.run(
        workflow, spec, mode="reproduce", run_id="thesis", to_stage="acquire"
    ) == ["acquire: ran"]
    assert engine.run(
        workflow, spec, mode="reproduce", run_id="thesis", to_stage="acquire"
    ) == ["acquire: cached"]
    source.write_text("two")
    assert engine.run(
        workflow, spec, mode="reproduce", run_id="thesis", to_stage="acquire"
    ) == ["acquire: ran"]
    output.write_text("tampered")
    assert engine.run(
        workflow, spec, mode="reproduce", run_id="thesis", to_stage="acquire"
    ) == ["acquire: ran"]
    implementation.write_text("v2")
    assert engine.run(
        workflow, spec, mode="reproduce", run_id="thesis", to_stage="acquire"
    ) == ["acquire: ran"]
    changed = replace(workflow, raw_config=b"changed")
    assert engine.run(
        changed, spec, mode="reproduce", run_id="thesis", to_stage="acquire"
    ) == ["acquire: ran"]
    assert len(calls) == 5


def test_failure_checkpoint_is_not_reused_and_refresh_mode_is_recorded(
    tmp_path: Path,
) -> None:
    workflow = demo_workflow(tmp_path)
    implementation = tmp_path / "implementation.py"
    implementation.write_text("v1")
    attempts = 0

    def fail() -> None:
        nonlocal attempts
        attempts += 1
        raise ValueError("broken")

    spec = {"acquire": StageSpec(implementation=(implementation,), action=fail)}
    engine = WorkflowEngine(tmp_path)
    with pytest.raises(ValueError):
        engine.run(workflow, spec, mode="refresh", run_id="new", to_stage="acquire")
    with pytest.raises(ValueError):
        engine.run(workflow, spec, mode="refresh", run_id="new", to_stage="acquire")
    checkpoint = checkpoints.load(
        runs_root(tmp_path) / "test-demo" / "new" / "acquire.json"
    )
    assert attempts == 2
    assert (
        checkpoint is not None
        and checkpoint.status == "failed"
        and checkpoint.mode == "refresh"
        and checkpoint.error == "ValueError: broken"
    )


def test_fingerprints_always_verify_content_with_streaming_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "large-artifact.bin"
    artifact.write_bytes(b"content")
    calls = 0
    original = checkpoints._hash_file

    def count_hash(path: Path, digest: Any) -> None:
        nonlocal calls
        calls += 1
        original(path, digest)

    monkeypatch.setattr(checkpoints, "_hash_file", count_hash)
    first = checkpoints.fingerprint(artifact)
    second = checkpoints.fingerprint(artifact, first)
    assert first["sha256"] == second["sha256"]
    assert calls == 2
    artifact.write_bytes(b"changed")
    checkpoints.fingerprint(artifact, second)
    assert calls == 3


def test_directory_fingerprint_uses_deterministic_tree_metadata(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    nested = artifact / "nested.bin"
    nested.write_bytes(b"content")
    first = checkpoints.fingerprint(artifact)
    assert checkpoints.fingerprint(artifact, first)["sha256"] == first["sha256"]
    nested.write_bytes(b"changed")
    changed = checkpoints.fingerprint(artifact, first)
    assert changed["tree_id"] != first["tree_id"]


def test_corrupt_checkpoint_and_malformed_workflow_are_invalid(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text("not json")
    assert checkpoints.load(checkpoint) is None
    config_dir = tmp_path / "configs" / "thesis"
    config_dir.mkdir(parents=True)
    (config_dir / "global-mean.toml").write_text("workflow = [")
    with pytest.raises(WorkflowError, match="Malformed workflow configuration"):
        load_workflow("global-mean", tmp_path)


def test_fresh_run_ids_are_sortable_and_unique() -> None:
    first, second = fresh_run_id(), fresh_run_id()
    assert first != second
    assert re.fullmatch(r"\d{8}T\d{12}Z-[0-9a-f]{8}", first)


def test_path_policy_keeps_publication_outside_ignored_runtime_layers(
    tmp_path: Path,
) -> None:
    assert source_root(tmp_path) == tmp_path / "data" / "sources"
    assert prepared_root(tmp_path) == tmp_path / "data" / "prepared"
    assert product_root(tmp_path) == tmp_path / "data" / "products"
    assert runs_root(tmp_path) == tmp_path / "runs"
    assert publication_root(tmp_path) == tmp_path / "outputs" / "figures" / "results"


def test_ssh_profile_validation_and_safe_command_construction(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles.toml"
    profiles.write_text(
        "[profiles.cluster]\nhost = 'example.org'\nuser = 'runner'\nremote_root = '/scratch/thermodense'\n"
    )
    profile = load_profile("cluster", profiles)
    commands = SSHExecutionAdapter(profile).commands(
        Path("/repo"), "global-mean", "thesis"
    )
    assert commands[0] == [
        "ssh",
        "-o",
        "BatchMode=yes",
        "runner@example.org",
        "mkdir",
        "-p",
        "/scratch/thermodense/global-mean/thesis",
    ]
    assert commands[1][0] == "rsync"
    assert "shell=True" not in repr(commands)
    with pytest.raises(SSHProfileError, match="not found"):
        load_profile("missing", profiles)
    profiles.write_text(
        "[profiles.bad]\nhost = 'example.org;bad'\nremote_root = '/scratch/thermodense'\n"
    )
    with pytest.raises(SSHProfileError, match="unsafe"):
        load_profile("bad", profiles)
    with pytest.raises(SSHProfileError, match="Run ID contains unsafe"):
        SSHExecutionAdapter(profile).commands(Path("/repo"), "global-mean", "../bad")


def test_ssh_runner_errors_are_wrapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = load_profile(
        "cluster",
        _write_profile(tmp_path / "profiles.toml"),
    )
    monkeypatch.setattr("thermodense.ssh.shutil.which", lambda _: "/usr/bin/tool")

    def fail(command: list[str], check: bool) -> None:
        raise subprocess.CalledProcessError(23, command)

    with pytest.raises(SSHProfileError, match=r"command failed \(23\): ssh"):
        SSHExecutionAdapter(profile, runner=fail).run(
            Path("/repo"), "global-mean", "thesis", ["reproduce", "global-mean"]
        )


def _write_profile(path: Path) -> Path:
    path.write_text(
        "[profiles.cluster]\nhost = 'example.org'\nuser = 'runner'\nremote_root = '/scratch/thermodense'\n"
    )
    return path


def test_all_stage_order_runs_in_canonical_sequence(tmp_path: Path) -> None:
    workflow = demo_workflow(tmp_path, STAGES)
    implementation = tmp_path / "implementation.py"
    implementation.write_text("v1")
    order: list[str] = []
    specs = {
        stage: StageSpec(
            implementation=(implementation,),
            action=lambda stage=stage: order.append(stage),
        )
        for stage in STAGES
    }
    assert WorkflowEngine(tmp_path).run(
        workflow, specs, mode="reproduce", run_id="thesis"
    ) == [f"{stage}: ran" for stage in STAGES]
    assert order == list(STAGES)
