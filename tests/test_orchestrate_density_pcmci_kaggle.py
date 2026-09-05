"""Mocked recovery contracts for the private Kaggle PCMCI adapter."""

import json
import subprocess
import tarfile
from pathlib import Path

import pytest

import scripts.orchestrate_density_pcmci_kaggle as kaggle


@pytest.fixture(autouse=True)
def forbid_live_kaggle_subprocess(monkeypatch):
    """Any missed query seam fails instead of reaching the authenticated CLI."""
    monkeypatch.setattr(
        kaggle.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("live Kaggle subprocess invocation"),
    )


def run_dir(tmp_path: Path) -> Path:
    for name in ("runner.py", "analysis_bundle.npz", "analysis_bundle.provenance.json"):
        (tmp_path / name).write_text(name)
    return tmp_path


def test_assignment_and_kernel_metadata_are_private_cpu_offline(tmp_path):
    assert kaggle.SELECTED == ("hasdm_selected-ap", "hasdm_selected-kp")
    directory = kaggle.kernel_directory(run_dir(tmp_path), "hasdm_selected-ap")
    metadata = json.loads((directory / "kernel-metadata.json").read_text())
    assert metadata["is_private"] and not metadata["enable_gpu"] and not metadata["enable_internet"]
    assert metadata["dataset_sources"] == [kaggle.DATASET]
    program = (directory / "script.py").read_text()
    assert "--no-deps" in program
    assert "kaggle-cpu" in program
    assert 'input_root.rglob("analysis_bundle.npz")' in program
    assert 'output = work / "outputs/prototypes/density_pcmci_3hour_and_daily"' in program


def test_submit_durably_records_intent_before_dataset_and_kernel_side_effects(tmp_path, monkeypatch):
    directory = run_dir(tmp_path)
    seen = []
    wheel = directory / ".kaggle-inputs" / "tigramite-5.2.10.1-py3-none-any.whl"

    def fake_command(arguments, **_kwargs):
        seen.append((arguments, json.loads((directory / kaggle.STATE_NAME).read_text())))
        if "download" in arguments:
            wheel.parent.mkdir(exist_ok=True)
            wheel.write_text("wheel")
        return "missing" if arguments[1:3] == ["kernels", "status"] else ""

    monkeypatch.setattr(kaggle, "command", fake_command)
    dataset_exists = iter([False, True])
    monkeypatch.setattr(kaggle, "dataset_exists", lambda: next(dataset_exists))
    monkeypatch.setattr(kaggle, "dataset_status", lambda: "ready")
    assert kaggle.main(["submit", "--run-dir", str(directory)]) == 0
    dataset_call = next(state for args, state in seen if args[1:3] == ["datasets", "create"])
    push_call = next(state for args, state in seen if args[1:3] == ["kernels", "push"])
    assert dataset_call["dataset"]["status"] == "uploading"
    assert any(case["status"] == "pushing" for case in push_call["cases"].values())


def test_resume_does_not_duplicate_push_for_running_or_complete(tmp_path, monkeypatch):
    directory = run_dir(tmp_path)
    state = kaggle.initial_state(directory)
    path = directory / kaggle.STATE_NAME
    state["dataset"].update(uploaded_input_hash="same")
    monkeypatch.setattr(kaggle, "ensure_dataset", lambda *_args: None)
    pushed = []
    monkeypatch.setattr(kaggle, "command", lambda args, **_kwargs: pushed.append(args) or "running")
    assert kaggle.resume(directory, state, path)
    assert not [args for args in pushed if args[1:3] == ["kernels", "push"]]


def test_kernel_failure_is_durable_and_not_resubmitted(tmp_path, monkeypatch):
    directory = run_dir(tmp_path)
    state = kaggle.initial_state(directory)
    path = directory / kaggle.STATE_NAME
    state["dataset"]["uploaded_input_hash"] = "hash"
    state["cases"][kaggle.SELECTED[0]]["submitted_input_hash"] = "hash"
    monkeypatch.setattr(kaggle, "kernel_status", lambda _kernel: "failed")
    assert not kaggle.submit_case(directory, state, path, kaggle.SELECTED[0])
    assert state["cases"][kaggle.SELECTED[0]]["status"] == "failed"


def test_retrieve_rejects_traversal_and_reuses_identical_output(tmp_path, monkeypatch):
    directory = run_dir(tmp_path)
    state = kaggle.initial_state(directory)
    path = directory / kaggle.STATE_NAME
    monkeypatch.setattr(kaggle, "kernel_status", lambda _kernel: "complete")

    def output(arguments, **_kwargs):
        target = Path(arguments[-1]) / "hasdm_selected-ap.tar.gz"
        with tarfile.open(target, "w:gz") as archive:
            member = tarfile.TarInfo("../escape")
            member.size = 0
            archive.addfile(member)
        return ""

    monkeypatch.setattr(kaggle, "command", output)
    assert not kaggle.retrieve(directory, state, path)
    assert state["retrieval"]["hasdm_selected-ap"]["status"] == "refused"


def test_cli_status_and_missing_state(tmp_path, monkeypatch, capsys):
    directory = run_dir(tmp_path)
    path = directory / kaggle.STATE_NAME
    kaggle.save(kaggle.initial_state(directory), path)
    monkeypatch.setattr(kaggle, "kernel_status", lambda _kernel: "running")
    assert kaggle.main(["status", "--run-dir", str(directory)]) == 0
    assert "hasdm_selected-ap" in capsys.readouterr().out
    with pytest.raises(FileNotFoundError):
        kaggle.main(["resume", "--run-dir", str(tmp_path / "missing")])


def test_kernel_mine_list_is_exact_and_status_errors_remain_operational(monkeypatch):
    monkeypatch.setattr(
        kaggle,
        "query",
        lambda _args: subprocess.CompletedProcess([], 0, "Not found", ""),
    )
    assert kaggle.kernel_status("teo112/not-created") == "missing"

    monkeypatch.setattr(
        kaggle,
        "query",
        lambda _args: subprocess.CompletedProcess([], 0, '[{"ref": "teo112/not-created-other"}]', ""),
    )
    assert kaggle.kernel_status("teo112/not-created") == "missing"

    responses = iter(
        [
            subprocess.CompletedProcess([], 0, '[{"ref": "teo112/private"}]', ""),
            subprocess.CompletedProcess([], 1, "", "Cannot access kernel"),
        ]
    )
    monkeypatch.setattr(
        kaggle, "query", lambda _args: next(responses)
    )
    with pytest.raises(kaggle.OperationalError):
        kaggle.kernel_status("teo112/private")

    responses = iter(
        [
            subprocess.CompletedProcess([], 0, '[{"ref": "teo112/cancelled"}]', ""),
            subprocess.CompletedProcess([], 0, "Cancelled", ""),
        ]
    )
    monkeypatch.setattr(kaggle, "query", lambda _args: next(responses))
    assert kaggle.kernel_status("teo112/cancelled") == "failed"


def test_dataset_mine_list_requires_exact_ref_and_distinguishes_no_datasets(monkeypatch):
    calls = []
    monkeypatch.setattr(
        kaggle,
        "query",
        lambda args: calls.append(args)
        or subprocess.CompletedProcess([], 0, "No datasets found", ""),
    )
    assert not kaggle.dataset_exists()
    assert calls == [
        [
            kaggle.KAGGLE,
            "datasets",
            "list",
            "--mine",
            "-s",
            "density-pcmci-v2-saber-inputs",
            "--format",
            "json",
        ]
    ]
    monkeypatch.setattr(
        kaggle,
        "query",
        lambda _args: subprocess.CompletedProcess([], 0, '[{"ref": "teo112/density-pcmci-v2-saber-inputs-old"}]', ""),
    )
    assert not kaggle.dataset_exists()
    monkeypatch.setattr(
        kaggle,
        "query",
        lambda _args: subprocess.CompletedProcess([], 0, '[{"ref": "teo112/density-pcmci-v2-saber-inputs"}]', ""),
    )
    assert kaggle.dataset_exists()


def test_interrupted_create_probes_missing_dataset_and_retries_create(tmp_path, monkeypatch):
    directory = run_dir(tmp_path)
    state = kaggle.initial_state(directory)
    path = directory / kaggle.STATE_NAME
    wheel = directory / "wheel.whl"
    wheel.write_text("wheel")
    monkeypatch.setattr(kaggle, "bundle_directory", lambda _run: (directory, wheel))
    probes = iter([False, False, True])
    monkeypatch.setattr(kaggle, "dataset_exists", lambda: next(probes))
    monkeypatch.setattr(kaggle, "dataset_status", lambda: "ready")
    calls = []

    def interrupted(arguments, **_kwargs):
        calls.append(arguments)
        raise subprocess.CalledProcessError(1, arguments)

    monkeypatch.setattr(kaggle, "command", interrupted)
    with pytest.raises(subprocess.CalledProcessError):
        kaggle.ensure_dataset(directory, state, path)
    monkeypatch.setattr(kaggle, "command", lambda arguments, **_kwargs: calls.append(arguments) or "")
    assert kaggle.ensure_dataset(directory, state, path)
    assert [call[1:3] for call in calls] == [["datasets", "create"], ["datasets", "create"]]


def test_stale_complete_kernel_is_pushed_for_current_dataset_hash(tmp_path, monkeypatch):
    directory = run_dir(tmp_path)
    state = kaggle.initial_state(directory)
    path = directory / kaggle.STATE_NAME
    state["dataset"]["uploaded_input_hash"] = "new"
    state["cases"]["hasdm_selected-ap"]["submitted_input_hash"] = "old"
    monkeypatch.setattr(kaggle, "kernel_status", lambda _kernel: "complete")
    calls = []
    monkeypatch.setattr(kaggle, "command", lambda args, **_kwargs: calls.append(args) or "")
    assert kaggle.submit_case(directory, state, path, "hasdm_selected-ap")
    assert calls[0][1:3] == ["kernels", "push"]


def test_pending_dataset_defers_kernel_push_without_attempt(tmp_path, monkeypatch):
    directory = run_dir(tmp_path)
    state = kaggle.initial_state(directory)
    path = directory / kaggle.STATE_NAME
    monkeypatch.setattr(kaggle, "ensure_dataset", lambda *_args: False)
    monkeypatch.setattr(kaggle, "submit_case", lambda *_args: pytest.fail("must defer push"))
    assert kaggle.resume(directory, state, path)
    assert not state["cases"]["hasdm_selected-ap"]["attempts"]


def test_status_query_error_is_unhealthy_without_writing_state(tmp_path, monkeypatch):
    directory = run_dir(tmp_path)
    state = kaggle.initial_state(directory)
    before = json.dumps(state, sort_keys=True)
    monkeypatch.setattr(kaggle, "kernel_status", lambda _kernel: (_ for _ in ()).throw(kaggle.OperationalError("offline")))
    observed, healthy = kaggle.status(state)
    assert not healthy and set(observed.values()) == {"unreachable"}
    assert json.dumps(state, sort_keys=True) == before
