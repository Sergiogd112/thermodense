"""Recovery contracts for the Archmini direct-density orchestrator."""

import json
from pathlib import Path

import pytest

import scripts.orchestrate_density_pcmci_from_archmini as orchestrator


def make_run_dir(tmp_path: Path) -> Path:
    for name, content in {
        "runner.py": "runner",
        "analysis_bundle.npz": "bundle",
        "run_case_sge.sh": "#!/bin/bash\n",
    }.items():
        (tmp_path / name).write_text(content)
    return tmp_path


def complete_case(directory: Path, case_id: str, bundle_hash: str, *, altered=False):
    directory.mkdir(parents=True)
    count = orchestrator.expected_tests(case_id)
    tests = directory / "driver_target_tests.csv"
    tests.write_text("x\n" + "1\n" * count)
    retained = directory / "retained_links.csv"
    retained.write_text("x\n")
    hashes = {path.name: orchestrator.sha256(path) for path in (tests, retained)}
    provenance = {
        "production": True,
        "case": {"id": case_id},
        "bundle_sha256": bundle_hash,
        "fingerprint": "fingerprint",
        "fdr_family": {"test_count": count},
        "result_files": hashes,
    }
    (directory / "provenance.json").write_text(json.dumps(provenance))
    if altered:
        tests.write_text("changed\n")


def test_initial_manifest_is_durable_prelaunch_plan(tmp_path):
    run_dir = make_run_dir(tmp_path)
    manifest = orchestrator.initial_manifest(run_dir)
    manifest_path = run_dir / "orchestration_manifest.json"
    orchestrator.save_manifest(manifest, manifest_path)
    saved = json.loads(manifest_path.read_text())
    assert saved["inputs"]["runner.py"] == orchestrator.sha256(run_dir / "runner.py")
    assert saved["cases"]["global_mean-ap"]["host"] == "spacehopper"
    assert saved["cases"]["hasdm_all-ap"]["remote_run_dir"].endswith(
        orchestrator.REMOTE_BASE
    )
    assert not list(run_dir.glob(".orchestration_manifest.json.*.tmp"))


def test_plan_and_expected_counts_cover_all_saber_cases():
    assert set(orchestrator.PLAN) == {
        "global_mean-ap",
        "global_mean-kp",
        "hasdm_all-ap",
        "hasdm_all-kp",
        "hasdm_selected-ap",
        "hasdm_selected-kp",
    }
    assert all(item["host"] == "phoenix" and item["kind"] == "sge" for key, item in orchestrator.PLAN.items() if key.startswith("hasdm_all-"))
    assert all(item["host"] == "kaggle" and item["kind"] == "kaggle" for key, item in orchestrator.PLAN.items() if key.startswith("hasdm_selected-"))
    assert orchestrator.expected_tests("global_mean-ap") == 3680
    assert orchestrator.expected_tests("hasdm_all-ap") == 224451
    assert orchestrator.expected_tests("hasdm_selected-kp") == 24939


def test_resume_does_not_relaunch_complete_or_active_cases(tmp_path, monkeypatch):
    run_dir = make_run_dir(tmp_path)
    manifest = orchestrator.initial_manifest(run_dir)
    manifest_path = run_dir / "orchestration_manifest.json"
    staged_hosts = []
    monkeypatch.setattr(
        orchestrator,
        "stage",
        lambda host, *_args: staged_hosts.append(host) or {"runner.py": "ok"},
    )
    monkeypatch.setattr(
        orchestrator,
        "remote_status",
        lambda host, _manifest: {
            case: {"status": "complete" if case == "global_mean-ap" else "active"}
            for case, data in _manifest["cases"].items()
            if data["host"] == host
        },
    )
    monkeypatch.setattr(
        orchestrator,
        "launch",
        lambda *_args: (_ for _ in ()).throw(AssertionError("launch")),
    )
    assert orchestrator.resume(run_dir, manifest, manifest_path)
    assert manifest["cases"]["global_mean-ap"]["status"] == "completed"
    assert manifest["cases"]["global_mean-kp"]["status"] == "active"
    assert all(not case["attempts"] for case in manifest["cases"].values())
    assert staged_hosts == ["phoenix", "spacehopper"]


def test_submit_with_host_only_launches_that_hosts_cases(tmp_path, monkeypatch):
    run_dir = make_run_dir(tmp_path)
    staged_hosts, launched_cases = [], []
    monkeypatch.setattr(
        orchestrator,
        "stage",
        lambda host, *_args: staged_hosts.append(host) or {"runner.py": "ok"},
    )
    monkeypatch.setattr(
        orchestrator,
        "remote_status",
        lambda host, manifest: {
            case_id: {"status": "failed"}
            for case_id, data in manifest["cases"].items()
            if data["host"] == host
        },
    )
    monkeypatch.setattr(
        orchestrator,
        "launch",
        lambda case_id, _data: launched_cases.append(case_id) or {"pid": 123},
    )

    assert (
        orchestrator.main(
            ["submit", "--run-dir", str(run_dir), "--host", "spacehopper"]
        )
        == 0
    )

    manifest = orchestrator.load_manifest(run_dir / "orchestration_manifest.json")
    assert set(manifest["cases"]) == set(orchestrator.PLAN)
    assert staged_hosts == ["spacehopper"]
    assert launched_cases == ["global_mean-ap", "global_mean-kp"]
    assert all(
        data["status"] == "pending" and not data["attempts"]
        for data in manifest["cases"].values()
        if data["host"] in {"phoenix", "kaggle"}
    )


def test_resume_cli_passes_repeatable_host_filter(tmp_path, monkeypatch):
    run_dir = make_run_dir(tmp_path)
    manifest_path = run_dir / "orchestration_manifest.json"
    orchestrator.save_manifest(orchestrator.initial_manifest(run_dir), manifest_path)
    observed_hosts = None

    def fake_resume(_run_dir, _manifest, _manifest_path, hosts=None):
        nonlocal observed_hosts
        observed_hosts = hosts
        return True

    monkeypatch.setattr(orchestrator, "resume", fake_resume)

    assert (
        orchestrator.main(
            [
                "resume",
                "--run-dir",
                str(run_dir),
                "--host",
                "spacehopper",
                "--host",
                "spacehopper",
            ]
        )
        == 0
    )
    assert observed_hosts == {"spacehopper"}


def test_resume_rejects_hosts_not_in_manifest(tmp_path):
    run_dir = make_run_dir(tmp_path)
    manifest = orchestrator.initial_manifest(run_dir)

    with pytest.raises(ValueError, match="requested host\\(s\\) not in manifest: typo"):
        orchestrator.resume(
            run_dir,
            manifest,
            run_dir / "orchestration_manifest.json",
            {"typo"},
        )


def test_stage_keeps_matching_immutable_inputs_in_place(tmp_path, monkeypatch):
    run_dir = make_run_dir(tmp_path)
    manifest = orchestrator.initial_manifest(run_dir)
    calls = []

    def fake_remote(_host, shell, *, capture=False):
        calls.append(shell)
        if "for file" in shell or "sha256sum" in shell:
            selected = {
                "runner.py": manifest["inputs"]["runner.py"],
                "analysis_bundle.npz": manifest["inputs"]["analysis_bundle.npz"],
            }
            return "\n".join(f"{value}  {name}" for name, value in selected.items())
        return ""

    monkeypatch.setattr(orchestrator, "remote", fake_remote)
    monkeypatch.setattr(
        orchestrator,
        "command",
        lambda arguments, **_kwargs: (_ for _ in ()).throw(AssertionError(arguments)),
    )
    assert orchestrator.stage("spacehopper", run_dir, False, manifest["inputs"])


def test_status_is_read_only_and_preserves_sge_interpretation(tmp_path, monkeypatch):
    run_dir = make_run_dir(tmp_path)
    manifest = orchestrator.initial_manifest(run_dir)
    manifest_path = run_dir / "orchestration_manifest.json"
    orchestrator.save_manifest(manifest, manifest_path)
    before = manifest_path.read_text()
    monkeypatch.setattr(
        orchestrator,
        "remote_status",
        lambda host, _manifest: {
            case: {"status": "exited" if case == "hasdm_all-ap" else "active"}
            for case, data in _manifest["cases"].items()
            if data["host"] == host
        },
    )
    observed, healthy = orchestrator.status(manifest)
    assert healthy and observed["hasdm_all-ap"]["status"] == "exited"
    assert manifest_path.read_text() == before


def test_status_with_host_queries_only_selected_host(tmp_path, monkeypatch):
    manifest = orchestrator.initial_manifest(make_run_dir(tmp_path))
    queried_hosts = []

    def fake_remote_status(host, _manifest):
        queried_hosts.append(host)
        return {
            case_id: {"status": "active"}
            for case_id, data in _manifest["cases"].items()
            if data["host"] == host
        }

    monkeypatch.setattr(orchestrator, "remote_status", fake_remote_status)
    observed, healthy = orchestrator.status(manifest, {"phoenix"})

    assert healthy
    assert queried_hosts == ["phoenix"]
    assert set(observed) == {
        case_id
        for case_id, data in manifest["cases"].items()
        if data["host"] == "phoenix"
    }


@pytest.mark.parametrize("operation", [orchestrator.status, orchestrator.retrieve])
def test_status_and_retrieve_reject_hosts_not_in_manifest(tmp_path, operation):
    run_dir = make_run_dir(tmp_path)
    manifest = orchestrator.initial_manifest(run_dir)

    with pytest.raises(ValueError, match="requested host\\(s\\) not in manifest: typo"):
        if operation is orchestrator.status:
            operation(manifest, {"typo"})
        else:
            operation(run_dir, manifest, run_dir / "orchestration_manifest.json", {"typo"})


def test_status_cli_passes_host_filter(tmp_path, monkeypatch):
    run_dir = make_run_dir(tmp_path)
    manifest_path = run_dir / "orchestration_manifest.json"
    orchestrator.save_manifest(orchestrator.initial_manifest(run_dir), manifest_path)
    observed_hosts = None

    def fake_status(_manifest, hosts=None):
        nonlocal observed_hosts
        observed_hosts = hosts
        return {}, True

    monkeypatch.setattr(orchestrator, "status", fake_status)
    assert orchestrator.main(["status", "--run-dir", str(run_dir), "--host", "phoenix"]) == 0
    assert observed_hosts == {"phoenix"}


def test_retrieve_cli_passes_host_filter(tmp_path, monkeypatch):
    run_dir = make_run_dir(tmp_path)
    manifest_path = run_dir / "orchestration_manifest.json"
    orchestrator.save_manifest(orchestrator.initial_manifest(run_dir), manifest_path)
    observed_hosts = None

    def fake_retrieve(_run_dir, _manifest, _manifest_path, hosts=None):
        nonlocal observed_hosts
        observed_hosts = hosts
        return True

    monkeypatch.setattr(orchestrator, "retrieve", fake_retrieve)
    assert orchestrator.main(["retrieve", "--run-dir", str(run_dir), "--host", "phoenix"]) == 0
    assert observed_hosts == {"phoenix"}


def test_remote_status_program_validates_exact_direct_command_and_sge_accounting():
    program = orchestrator.REMOTE_STATUS_PROGRAM
    assert 'b"runner.py\\x00run\\x00"+c["id"].encode()+b"\\x00"' in program
    assert '["qstat","-j",job]' in program
    assert '["qacct","-j",job]' in program
    assert '"failed"' in program and '"exited"' in program


def test_validate_case_requires_complete_unaltered_provenance(tmp_path):
    bundle_hash = "bundle"
    complete_case(tmp_path / "good", "global_mean-ap", bundle_hash)
    assert orchestrator.validate_case(tmp_path / "good", "global_mean-ap", bundle_hash)
    complete_case(tmp_path / "altered", "global_mean-ap", bundle_hash, altered=True)
    assert (
        orchestrator.validate_case(tmp_path / "altered", "global_mean-ap", bundle_hash)
        is None
    )
    assert (
        orchestrator.validate_case(tmp_path / "missing", "global_mean-ap", bundle_hash)
        is None
    )


def test_retrieve_refuses_partial_artifact(tmp_path, monkeypatch):
    run_dir = make_run_dir(tmp_path)
    manifest = orchestrator.initial_manifest(run_dir)
    manifest_path = run_dir / "orchestration_manifest.json"
    monkeypatch.setattr(
        orchestrator,
        "status",
        lambda _manifest: (
            {case: {"status": "complete"} for case in _manifest["cases"]},
            True,
        ),
    )
    monkeypatch.setattr(orchestrator, "command", lambda *_args, **_kwargs: "")
    assert not orchestrator.retrieve(run_dir, manifest, manifest_path)
    assert all(item["status"] == "refused" for item in manifest["retrieval"].values())
    assert not (run_dir / "retrieved" / "global_mean-ap").exists()


def test_retrieve_creates_parent_before_first_successful_scp(tmp_path, monkeypatch):
    run_dir = make_run_dir(tmp_path)
    manifest = orchestrator.initial_manifest(run_dir)
    manifest_path = run_dir / "orchestration_manifest.json"
    monkeypatch.setattr(
        orchestrator,
        "status",
        lambda _manifest: ({"global_mean-ap": {"status": "complete"}}, True),
    )

    def fake_command(arguments, **_kwargs):
        assert (run_dir / "retrieved").is_dir()
        complete_case(
            Path(arguments[-1]),
            "global_mean-ap",
            orchestrator.sha256(run_dir / "analysis_bundle.npz"),
        )
        return ""

    monkeypatch.setattr(orchestrator, "command", fake_command)
    assert orchestrator.retrieve(run_dir, manifest, manifest_path)
    assert (run_dir / "retrieved" / "global_mean-ap").is_dir()


def test_retrieve_with_host_only_assesses_and_retrieves_selected_cases(
    tmp_path, monkeypatch
):
    run_dir = make_run_dir(tmp_path)
    manifest = orchestrator.initial_manifest(run_dir)
    manifest_path = run_dir / "orchestration_manifest.json"
    selected_case = "hasdm_all-ap"
    observed_hosts = None

    def fake_status(_manifest, hosts=None):
        nonlocal observed_hosts
        observed_hosts = hosts
        return {selected_case: {"status": "complete"}}, True

    def fake_command(arguments, **_kwargs):
        assert selected_case in arguments[-2]
        complete_case(
            Path(arguments[-1]),
            selected_case,
            orchestrator.sha256(run_dir / "analysis_bundle.npz"),
        )
        return ""

    monkeypatch.setattr(orchestrator, "status", fake_status)
    monkeypatch.setattr(orchestrator, "command", fake_command)

    assert orchestrator.retrieve(run_dir, manifest, manifest_path, {"phoenix"})
    assert observed_hosts == {"phoenix"}
    assert manifest["retrieval"][selected_case]["status"] == "retrieved"
    assert "global_mean-ap" not in manifest["retrieval"]


def test_retrieve_removes_stale_temporary_before_scp(tmp_path, monkeypatch):
    run_dir = make_run_dir(tmp_path)
    manifest = orchestrator.initial_manifest(run_dir)
    manifest_path = run_dir / "orchestration_manifest.json"
    temporary = (
        run_dir / "retrieved" / f".global_mean-ap.{orchestrator.os.getpid()}.tmp"
    )
    temporary.mkdir(parents=True)
    (temporary / "stale").write_text("stale")
    monkeypatch.setattr(
        orchestrator,
        "status",
        lambda _manifest: ({"global_mean-ap": {"status": "complete"}}, True),
    )

    def fake_command(arguments, **_kwargs):
        assert not temporary.exists()
        complete_case(
            Path(arguments[-1]),
            "global_mean-ap",
            orchestrator.sha256(run_dir / "analysis_bundle.npz"),
        )
        return ""

    monkeypatch.setattr(orchestrator, "command", fake_command)
    assert orchestrator.retrieve(run_dir, manifest, manifest_path)


def test_retrieve_cleans_failed_temporary_without_touching_destination(
    tmp_path, monkeypatch
):
    run_dir = make_run_dir(tmp_path)
    manifest = orchestrator.initial_manifest(run_dir)
    manifest_path = run_dir / "orchestration_manifest.json"
    destination = run_dir / "retrieved" / "global_mean-ap"
    destination.mkdir(parents=True)
    (destination / "authoritative").write_text("keep")
    temporary = run_dir / "retrieved" / f".global_mean-ap.{orchestrator.os.getpid()}.tmp"
    temporary.symlink_to(tmp_path / "interrupted-retrieval")
    monkeypatch.setattr(
        orchestrator,
        "status",
        lambda _manifest: ({"global_mean-ap": {"status": "complete"}}, True),
    )

    def fake_command(arguments, **_kwargs):
        assert not temporary.exists() and not temporary.is_symlink()
        Path(arguments[-1]).mkdir()
        raise OSError("scp interrupted")

    monkeypatch.setattr(orchestrator, "command", fake_command)
    assert not orchestrator.retrieve(run_dir, manifest, manifest_path)
    assert (destination / "authoritative").read_text() == "keep"
    assert not temporary.exists() and not temporary.is_symlink()
