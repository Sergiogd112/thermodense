#!/usr/bin/env python3
"""Recoverable, private, CPU-only Kaggle orchestration for selected PCMCI cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import time
from typing import Any

try:  # Source checkout.
    from scripts.orchestrate_density_pcmci_from_archmini import PLAN, sha256, validate_case
except ModuleNotFoundError:  # Staged beside the standard adapter on Archmini.
    from orchestrate_density_pcmci_from_archmini import PLAN, sha256, validate_case

KAGGLE = str(Path.home() / ".local/bin/kaggle")
UVX = "/usr/bin/uvx"
DATASET = "teo112/density-pcmci-v2-saber-inputs"
WHEEL = "tigramite-5.2.10.1"
SELECTED = tuple(case for case, item in PLAN.items() if item["kind"] == "kaggle")
KERNELS = {case: f"teo112/density-pcmci-v2-saber-selected-{case.rsplit('-', 1)[1]}" for case in SELECTED}
STATE_NAME = "kaggle_state.json"


class OperationalError(RuntimeError):
    """A Kaggle query could not establish remote state safely."""


def now() -> float:
    return time.time()


def atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def command(arguments: list[str], *, capture: bool = False) -> str:
    result = subprocess.run(
        arguments, check=True, text=True, stdout=subprocess.PIPE if capture else None
    )
    return result.stdout.strip() if capture else ""


def query(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a read-only Kaggle CLI operation without turning absence into an error."""
    return subprocess.run(arguments, check=False, text=True, capture_output=True)


def owned_resource_exists(kind: str, reference: str) -> bool:
    """Return whether the authenticated account owns this exact resource."""
    slug = reference.rsplit("/", 1)[1]
    result = query([KAGGLE, kind, "list", "--mine", "-s", slug, "--format", "json"])
    if result.returncode:
        raise OperationalError(result.stderr.strip() or result.stdout.strip())
    text = result.stdout.strip()
    if text in {"No datasets found", "Not found", ""}:
        return False
    try:
        records = json.loads(text)
    except ValueError as error:
        raise OperationalError(f"unrecognized Kaggle {kind} list response: {text}") from error
    if not isinstance(records, list):
        raise OperationalError(f"unrecognized Kaggle {kind} list response: {text}")
    return any(record.get("ref") == reference for record in records if isinstance(record, dict))


def dataset_exists() -> bool:
    return owned_resource_exists("datasets", DATASET)


def kernel_exists(kernel: str) -> bool:
    return owned_resource_exists("kernels", kernel)


def inputs(run_dir: Path) -> dict[str, Path]:
    return {
        "analysis_bundle.npz": run_dir / "analysis_bundle.npz",
        "runner.py": run_dir / "runner.py",
        "analysis_bundle.provenance.json": run_dir / "analysis_bundle.provenance.json",
    }


def save(state: dict[str, Any], path: Path) -> None:
    state["updated_epoch"] = now()
    atomic_json(state, path)


def initial_state(run_dir: Path) -> dict[str, Any]:
    source = inputs(run_dir)
    missing = next((path for path in source.values() if not path.is_file()), None)
    if missing:
        raise FileNotFoundError(missing)
    return {
        "schema": 1,
        "dataset": {"id": DATASET, "status": "pending"},
        "inputs": {name: sha256(path) for name, path in source.items()},
        "cases": {
            case: {"kernel": KERNELS[case], "status": "pending", "attempts": []}
            for case in SELECTED
        },
        "retrieval": {},
    }


def input_hash(run_dir: Path, wheel: Path) -> str:
    digest = hashlib.sha256()
    for name, path in sorted({**inputs(run_dir), wheel.name: wheel}.items()):
        digest.update(name.encode() + b"\0" + sha256(path).encode() + b"\n")
    return digest.hexdigest()


def wheel_path(directory: Path) -> Path:
    found = sorted(directory.glob(f"{WHEEL}-*.whl"))
    if found:
        return found[0]
    command([UVX, "--from", "pip", "pip", "download", "--no-deps", "tigramite==5.2.10.1", "-d", str(directory)])
    found = sorted(directory.glob(f"{WHEEL}-*.whl"))
    if not found:
        raise RuntimeError("pip download did not produce the pinned tigramite wheel")
    return found[0]


def bundle_directory(run_dir: Path) -> tuple[Path, Path]:
    directory = run_dir / ".kaggle-inputs"
    directory.mkdir(parents=True, exist_ok=True)
    wheel = wheel_path(directory)
    for name, source in inputs(run_dir).items():
        target = directory / name
        if not target.exists() or sha256(target) != sha256(source):
            shutil.copyfile(source, target)
    metadata = {"title": "density-pcmci-v2-saber-inputs", "id": DATASET, "licenses": [{"name": "other"}]}
    atomic_json(metadata, directory / "dataset-metadata.json")
    return directory, wheel


def ensure_dataset(run_dir: Path, state: dict[str, Any], state_path: Path) -> bool:
    directory, wheel = bundle_directory(run_dir)
    digest = input_hash(run_dir, wheel)
    dataset = state["dataset"]
    exists = dataset_exists()
    presence = dataset_status() if exists else "missing"
    if dataset.get("uploaded_input_hash") == digest and presence == "ready":
        dataset.update(status="ready", epoch=now())
        save(state, state_path)
        return True
    if dataset.get("uploaded_input_hash") == digest and presence == "pending":
        dataset.update(status="pending", epoch=now())
        save(state, state_path)
        return False
    # The intent is durable before create/version, so an interrupted command is
    # safely retried and never mistaken for a confirmed upload.
    creating = presence == "missing"
    dataset.update(
        status="uploading",
        input_hash=digest,
        operation="create" if creating else "version",
        epoch=now(),
    )
    save(state, state_path)
    try:
        if not creating:
            command([KAGGLE, "datasets", "version", "-p", str(directory), "-m", f"PCMCI inputs {digest}"])
        else:
            command([KAGGLE, "datasets", "create", "-p", str(directory)])
        dataset.update(status="uploaded", uploaded_input_hash=digest, epoch=now())
    except subprocess.SubprocessError as error:
        dataset.update(status="failed", detail=str(error), epoch=now())
        raise
    finally:
        save(state, state_path)

    # Creation/versioning is asynchronous.  Probe ownership again before using
    # status so an interrupted create never turns into a permanent version loop.
    readiness = dataset_status() if dataset_exists() else "missing"
    dataset.update(status=readiness, epoch=now())
    save(state, state_path)
    return readiness == "ready"


def kernel_program(case: str) -> str:
    return f'''import gzip, json, os, shutil, subprocess, tarfile
from pathlib import Path
case = {case!r}
input_root = Path("/kaggle/input")
bundles = sorted(input_root.rglob("analysis_bundle.npz"))
if len(bundles) != 1:
    raise RuntimeError(f"expected one analysis bundle, found {{bundles}} beneath {{input_root}}")
source = bundles[0].parent
work = Path("/kaggle/working/run")
work.mkdir(parents=True, exist_ok=True)
wheels = sorted(source.glob("tigramite-5.2.10.1-*.whl"))
if len(wheels) != 1:
    raise RuntimeError(f"expected one pinned tigramite wheel, found {{wheels}} in {{source}}")
wheel = wheels[0]
subprocess.run(["python", "-m", "pip", "install", "--no-deps", str(wheel)], check=True)
shutil.copy2(source / "runner.py", work / "runner.py")
output = work / "outputs/prototypes/density_pcmci_3hour_and_daily"
output.mkdir(parents=True, exist_ok=True)
for name in ("analysis_bundle.npz", "analysis_bundle.provenance.json"):
    shutil.copy2(source / name, output / name)
subprocess.run(["python", "runner.py", "run", case, "--host-label", "kaggle-cpu"], cwd=work, check=True)
case_dir = output / "cases" / case
artifact = Path("/kaggle/working") / (case + ".tar.gz")
with artifact.open("wb") as raw:
    with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w") as archive:
            for path in sorted(case_dir.rglob("*")):
                if path.is_file():
                    info = archive.gettarinfo(str(path), arcname=str(path.relative_to(case_dir)))
                    info.mtime = 0; info.uid = info.gid = 0; info.uname = info.gname = ""
                    with path.open("rb") as handle: archive.addfile(info, handle)
(Path("/kaggle/working") / (case + ".execution.json")).write_text(json.dumps({{"case": case, "host_label": "kaggle-cpu"}}, sort_keys=True))
'''


def kernel_directory(run_dir: Path, case: str) -> Path:
    directory = run_dir / ".kaggle-kernels" / case
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "script.py").write_text(kernel_program(case))
    atomic_json({"id": KERNELS[case], "title": KERNELS[case].rsplit("/", 1)[1], "code_file": "script.py", "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_internet": False, "dataset_sources": [DATASET]}, directory / "kernel-metadata.json")
    return directory


def kernel_status(kernel: str) -> str:
    if not kernel_exists(kernel):
        return "missing"
    result = query([KAGGLE, "kernels", "status", kernel])
    if result.returncode:
        raise OperationalError(result.stderr.strip() or result.stdout.strip())
    text = f"{result.stdout}\n{result.stderr}".lower()
    if "complete" in text:
        return "complete"
    if "running" in text or "queued" in text:
        return "running"
    if "error" in text or "failed" in text or "cancel" in text:
        return "failed"
    return "missing"


def dataset_status() -> str:
    result = query([KAGGLE, "datasets", "status", DATASET])
    if result.returncode:
        raise OperationalError(result.stderr.strip() or result.stdout.strip())
    text = f"{result.stdout}\n{result.stderr}".lower()
    if "ready" in text:
        return "ready"
    if "pending" in text or "processing" in text or "running" in text:
        return "pending"
    raise OperationalError(f"unrecognized dataset status: {text.strip()}")


def submit_case(run_dir: Path, state: dict[str, Any], state_path: Path, case: str) -> bool:
    data = state["cases"][case]
    current_hash = state["dataset"].get("uploaded_input_hash")
    observed = (
        kernel_status(data["kernel"])
        if current_hash and data.get("submitted_input_hash") == current_hash
        else "missing"
    )
    if observed in {"complete", "running"}:
        data.update(status=observed, observed_epoch=now())
        save(state, state_path)
        return True
    if observed == "failed":
        data.update(status="failed", detail="Kaggle kernel failed", observed_epoch=now())
        save(state, state_path)
        return False
    if len(data["attempts"]) >= 3:
        data.update(status="failed", detail="submission missing after 3 attempts")
        save(state, state_path)
        return False
    attempt = {"epoch": now(), "status": "pushing", "input_hash": current_hash}
    data["attempts"].append(attempt)
    data["status"] = "pushing"
    save(state, state_path)
    try:
        command([KAGGLE, "kernels", "push", "-p", str(kernel_directory(run_dir, case))])
        attempt["status"] = "submitted"
        data["status"] = "submitted"
        data["submitted_input_hash"] = current_hash
        return True
    except subprocess.SubprocessError as error:
        attempt.update(status="interrupted", detail=str(error))
        data["status"] = "missing"
        return False
    finally:
        save(state, state_path)


def resume(run_dir: Path, state: dict[str, Any], state_path: Path) -> bool:
    if not ensure_dataset(run_dir, state, state_path):
        return True
    results = [submit_case(run_dir, state, state_path, case) for case in SELECTED]
    return all(results)


def status(state: dict[str, Any]) -> tuple[dict[str, str], bool]:
    result: dict[str, str] = {}
    healthy = True
    for case, data in state["cases"].items():
        try:
            result[case] = kernel_status(data["kernel"])
        except OperationalError:
            result[case] = "unreachable"
            healthy = False
    return result, healthy and all(value != "failed" for value in result.values())


def extract_artifact(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as contents:
        for member in contents.getmembers():
            target = destination / member.name
            if member.issym() or member.islnk() or not target.resolve().is_relative_to(destination.resolve()):
                raise RuntimeError("unsafe Kaggle artifact path")
        contents.extractall(destination, filter="data")


def retrieve(run_dir: Path, state: dict[str, Any], state_path: Path) -> bool:
    destination_base = run_dir / "retrieved"
    destination_base.mkdir(parents=True, exist_ok=True)
    bundle_hash = sha256(run_dir / "analysis_bundle.npz")
    healthy = True
    for case, data in state["cases"].items():
        if kernel_status(data["kernel"]) != "complete":
            state["retrieval"][case] = {"status": "not_complete", "epoch": now()}
            healthy = False
            save(state, state_path)
            continue
        temporary = Path(tempfile.mkdtemp(prefix=f".{case}.", dir=destination_base))
        try:
            command([KAGGLE, "kernels", "output", data["kernel"], "-p", str(temporary)])
            artifact = temporary / f"{case}.tar.gz"
            extracted = temporary / "case"
            extracted.mkdir()
            extract_artifact(artifact, extracted)
            hashes = validate_case(extracted, case, bundle_hash)
            if hashes is None:
                raise RuntimeError("refusing partial or altered artifact")
            destination = destination_base / case
            if destination.exists():
                if validate_case(destination, case, bundle_hash) != hashes:
                    raise RuntimeError("refusing to replace different artifact")
            else:
                extracted.replace(destination)
            state["retrieval"][case] = {"status": "retrieved", "hashes": hashes, "epoch": now()}
        except (OSError, RuntimeError, subprocess.SubprocessError, tarfile.TarError) as error:
            state["retrieval"][case] = {"status": "refused", "detail": str(error), "epoch": now()}
            healthy = False
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
            save(state, state_path)
    return healthy


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("command", choices=("submit", "status", "resume", "retrieve"), nargs="?", default="submit")
    result.add_argument("--run-dir", type=Path, default=Path.cwd())
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    run_dir = args.run_dir.resolve()
    state_path = run_dir / STATE_NAME
    if state_path.exists():
        state = json.loads(state_path.read_text())
    elif args.command == "submit":
        state = initial_state(run_dir)
        save(state, state_path)
    else:
        raise FileNotFoundError(state_path)
    if args.command == "status":
        payload, healthy = status(state)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if healthy else 1
    return 0 if (resume(run_dir, state, state_path) if args.command in {"submit", "resume"} else retrieve(run_dir, state, state_path)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
