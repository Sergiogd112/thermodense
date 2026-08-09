"""Atomic, provenance-aware stage checkpoints."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import tempfile
from typing import Any


CHUNK_SIZE = 1024 * 1024
_CHECKPOINT_LOAD_ERRORS = (
    FileNotFoundError,
    json.JSONDecodeError,
    TypeError,
    ValueError,
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_file(path: Path, digest: Any) -> None:
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)


def _file_metadata(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _directory_metadata(path: Path) -> tuple[str, int]:
    """Return a deterministic identity for the directory's file tree metadata."""
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        metadata = _file_metadata(item)
        digest.update(str(item.relative_to(path)).encode())
        digest.update(str(metadata["size"]).encode())
        digest.update(str(metadata["mtime_ns"]).encode())
    return digest.hexdigest(), len(files)


def fingerprint(
    path: Path, previous: dict[str, object] | None = None
) -> dict[str, object]:
    """Fingerprint content with streaming I/O.

    ``previous`` is accepted so callers can evolve checkpoint formats without an
    interface break, but content is always re-hashed. Size and modification time
    are diagnostic provenance, not substitutes for content identity.
    """
    del previous
    path = path.resolve()
    if not path.exists():
        return {"path": str(path), "state": "missing"}
    if path.is_file():
        metadata = _file_metadata(path)
        result: dict[str, object] = {"path": str(path), "state": "file", **metadata}
        digest = hashlib.sha256()
        _hash_file(path, digest)
        return {**result, "sha256": digest.hexdigest()}
    tree_id, files_count = _directory_metadata(path)
    result = {
        "path": str(path),
        "state": "directory",
        "tree_id": tree_id,
        "files": files_count,
    }
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        digest.update(str(item.relative_to(path)).encode())
        _hash_file(item, digest)
    return {**result, "sha256": digest.hexdigest()}


def fingerprints(
    paths: Iterable[Path], previous: Iterable[dict[str, object]] = ()
) -> list[dict[str, object]]:
    old = {item.get("path"): item for item in previous}
    return [fingerprint(path, old.get(str(path.resolve()))) for path in paths]


def implementation_fingerprint(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted({path.resolve() for path in paths}):
        digest.update(str(path).encode())
        if path.exists():
            _hash_file(path, digest)
        else:
            digest.update(b"MISSING")
    return digest.hexdigest()


def now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class Checkpoint:
    workflow: str
    mode: str
    stage: str
    status: str
    config_hash: str
    implementation_fingerprint: str
    inputs: list[dict[str, object]]
    outputs: list[dict[str, object]]
    started_at: str
    finished_at: str | None
    metadata: dict[str, object]
    error: str | None = None

    def matches(
        self,
        config_hash: str,
        implementation: str,
        inputs: list[dict[str, object]],
        outputs: list[dict[str, object]],
    ) -> bool:
        return self.status == "success" and (
            self.config_hash,
            self.implementation_fingerprint,
            self.inputs,
            self.outputs,
        ) == (config_hash, implementation, inputs, outputs)


def checkpoint_path(run_dir: Path, stage: str) -> Path:
    return run_dir / f"{stage}.json"


def load(path: Path) -> Checkpoint | None:
    try:
        value = json.loads(path.read_text())
        required_strings = (
            "workflow",
            "mode",
            "stage",
            "status",
            "config_hash",
            "implementation_fingerprint",
            "started_at",
        )
        if (
            not isinstance(value, dict)
            or not all(isinstance(value.get(key), str) for key in required_strings)
            or not isinstance(value.get("inputs"), list)
            or not isinstance(value.get("outputs"), list)
            or not isinstance(value.get("metadata"), dict)
            or value.get("finished_at") is not None
            and not isinstance(value.get("finished_at"), str)
            or "error" in value
            and value["error"] is not None
            and not isinstance(value["error"], str)
        ):
            return None
        return Checkpoint(**value)
    except _CHECKPOINT_LOAD_ERRORS:
        return None


def write(path: Path, checkpoint: Checkpoint) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8"
    ) as handle:
        json.dump(asdict(checkpoint), handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "uninstalled"


def metadata() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "package_version": _version("thermodense"),
        "dependency_versions": {
            name: _version(name) for name in ("numpy", "polars", "pymsis", "tigramite")
        },
    }
