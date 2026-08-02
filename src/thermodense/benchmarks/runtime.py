"""Public runtime primitives shared by PCMCI benchmark entry points."""

from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import tempfile
import time
from typing import Any

import numpy as np

THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OPENBLAS_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def method_settings(method: str, cmiknn_workers: int | None = None) -> dict[str, Any]:
    """Return the shared independence-test settings for a named method."""
    settings: dict[str, Any] = {"pc_alpha": 0.05, "alpha_level": 0.05}
    if method == "parcorr":
        return settings | {"significance": "analytic"}
    if method == "cmiknn":
        result = settings | {
            "significance": "shuffle_test",
            "sig_samples": 20,
            "sig_blocklength": 4,
            "knn": 0.1,
            "shuffle_neighbors": 5,
            "workers": 1,
        }
        if cmiknn_workers is not None:
            result = result | {"workers": cmiknn_workers}
        return result
    if method == "gpdc":
        return settings | {"significance": "analytic"}
    raise ValueError(f"Unknown benchmark method: {method}")


def compact_result_digest(results: dict[str, Any]) -> str:
    """Hash result matrices without serializing their contents into JSON."""
    digest_input = {
        name: {
            "shape": list(np.asarray(value).shape),
            "sha256": hashlib.sha256(
                np.ascontiguousarray(np.asarray(value)).tobytes()
            ).hexdigest(),
        }
        for name, value in sorted(results.items())
    }
    encoded = json.dumps(digest_input, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_npz_artifact(
    path: Path,
    matrices: dict[str, Any],
    *,
    node_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Atomically persist canonical result matrices in compressed NPZ."""
    contents = dict(matrices)
    if node_names is not None:
        contents["node_names"] = np.asarray(node_names)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".npz", delete=False
    ) as handle:
        temporary_path = Path(handle.name)
        try:
            np.savez_compressed(handle, **contents)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    os.replace(temporary_path, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return {
        "path": str(path),
        "name": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "format": "npz-compressed",
        "keys": sorted(contents),
    }


def write_jsonl_artifact(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Atomically persist machine-readable derived rows as durable JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".jsonl", delete=False
    ) as handle:
        temporary_path = Path(handle.name)
        try:
            for row in rows:
                handle.write(
                    (
                        json.dumps(
                            row,
                            sort_keys=True,
                            separators=(",", ":"),
                            ensure_ascii=False,
                        )
                        + "\n"
                    ).encode()
                )
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    os.replace(temporary_path, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return {
        "path": str(path),
        "name": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "format": "jsonl",
        "row_count": len(rows),
    }


def package_versions() -> dict[str, str]:
    """Report runtime package versions without requiring optional packages."""
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for package in (
        "numpy",
        "scipy",
        "tigramite",
        "scikit-learn",
        "numba",
        "dcor",
        "torch",
        "gpytorch",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


def _repo_root() -> Path | None:
    for candidate in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
        if (candidate / ".git").exists():
            return candidate
    return None


@functools.lru_cache(maxsize=1)
def git_commit() -> str | None:
    """Return the enclosing checkout HEAD, when available."""
    root = _repo_root()
    if root is None:
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except OSError, subprocess.SubprocessError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one durable JSONL record."""
    encoded = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run_isolated_process(
    command: list[str], timeout: float, threads: int
) -> dict[str, Any]:
    """Run a JSON-producing child in its own process group with a timeout."""
    environment = os.environ.copy()
    environment.update({name: str(threads) for name in THREAD_ENVIRONMENT})
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
            start_new_session=True,
        )
    except OSError as error:
        return {
            "status": "failed",
            "failure_reason": f"could not start child: {error}",
            "wall_seconds": time.monotonic() - started,
        }
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _terminate_process_tree(process)
        process.communicate()
        return {
            "status": "timeout",
            "failure_reason": f"exceeded timeout of {timeout:g} seconds",
            "wall_seconds": time.monotonic() - started,
        }
    wall_seconds = time.monotonic() - started
    if process.returncode < 0:
        return {
            "status": "killed",
            "failure_reason": f"child terminated by signal {-process.returncode}",
            "wall_seconds": wall_seconds,
        }
    try:
        payload = json.loads(stdout.strip().splitlines()[-1])
    except IndexError, json.JSONDecodeError:
        return {
            "status": "failed",
            "failure_reason": f"child produced no valid result (exit {process.returncode}): {stderr.strip()[-300:]}",
            "wall_seconds": wall_seconds,
        }
    if process.returncode != 0 and payload.get("status") == "succeeded":
        payload = {
            "status": "failed",
            "failure_reason": f"child exited {process.returncode}",
        }
    payload["wall_seconds"] = wall_seconds
    return payload


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
