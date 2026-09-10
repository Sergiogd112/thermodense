"""Integrity checks and immutable publication for derived artifacts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path, PureWindowsPath
import re
import tempfile

from thermodense import checkpoints


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class ArtifactIntegrityError(ValueError):
    """An artifact cannot be accepted as the declared immutable content."""


def _relative_file(name: object) -> Path:
    if not isinstance(name, str) or not name:
        raise ArtifactIntegrityError("artifact filename must be a non-empty relative path")
    path = Path(name)
    windows_path = PureWindowsPath(name)
    if (
        path.is_absolute()
        or windows_path.is_absolute()
        or not path.parts
        or ".." in path.parts
        or ".." in windows_path.parts
    ):
        raise ArtifactIntegrityError("artifact filename must not be absolute or traverse its root")
    return path


def verify_files(
    root: Path, expected: Mapping[str, str], *, required: Iterable[str] = ()
) -> dict[str, str]:
    """Return verified digests for declared files rooted beneath ``root``.

    Each file is content-hashed through :func:`checkpoints.fingerprint`; metadata
    never substitutes for the content identity.
    """
    if not expected:
        raise ArtifactIntegrityError("artifact digest mapping must not be empty")
    root = root.resolve()
    if not root.is_dir():
        raise ArtifactIntegrityError("artifact root is not a directory")
    required_names = tuple(required)
    for name in required_names:
        _relative_file(name)
        if name not in expected:
            raise ArtifactIntegrityError(f"required artifact {name!r} is not declared")

    verified: dict[str, str] = {}
    for name, expected_digest in expected.items():
        relative = _relative_file(name)
        if not isinstance(expected_digest, str) or not _SHA256.fullmatch(expected_digest):
            raise ArtifactIntegrityError(f"artifact {name!r} has an invalid SHA-256 digest")
        try:
            path = (root / relative).resolve(strict=True)
            path.relative_to(root)
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
            raise ArtifactIntegrityError(f"artifact {name!r} is missing or escapes its root") from error
        fingerprint = checkpoints.fingerprint(path)
        if fingerprint.get("state") != "file":
            raise ArtifactIntegrityError(f"artifact {name!r} is not a file")
        digest = fingerprint.get("sha256")
        if digest != expected_digest:
            raise ArtifactIntegrityError(f"artifact {name!r} does not match its declared SHA-256 digest")
        verified[name] = expected_digest
    return verified


def publish_json_new(path: Path, payload: object) -> str:
    """Atomically publish JSON once, refusing to replace an existing artifact."""
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite immutable artifact {path}") from error
        fingerprint = checkpoints.fingerprint(path)
        digest = fingerprint.get("sha256")
        if not isinstance(digest, str):  # Defensive: a just-linked temporary must be a file.
            raise ArtifactIntegrityError("published artifact could not be fingerprinted")
        return digest
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


@dataclass(frozen=True)
class RecoveryCapability:
    """Declared recovery boundary for a named workflow, not checkpoint approval."""

    mode: str
    supports_fit_resume: bool
    requires_identity_match: bool
    caveat: str


_RECOVERY_CAPABILITIES = {
    "global-parcorr": RecoveryCapability(
        mode="derived-matrix re-export", supports_fit_resume=False, requires_identity_match=True,
        caveat="Saved association matrices can be re-exported, but diagnostic CI traces do not resume fitting.",
    ),
    "raw-36-parcorr": RecoveryCapability(
        mode="restart or final-result reuse", supports_fit_resume=False, requires_identity_match=True,
        caveat="There is no solver checkpoint; interrupted fitting must restart. Any saved final matrices require integrity validation before reuse.",
    ),
    "global-polynomial": RecoveryCapability(
        mode="compatible epoch resume", supports_fit_resume=True, requires_identity_match=True,
        caveat="Only identity-compatible epoch checkpoints may resume; checkpoint presence does not verify state.",
    ),
    "comparison": RecoveryCapability(
        mode="deterministic derived replay", supports_fit_resume=False, requires_identity_match=True,
        caveat="Replay requires verified immutable input and result identities; it never resumes fitting.",
    ),
}


def recovery_capability(kind: str) -> RecoveryCapability:
    """Return the declared recovery boundary for ``kind``."""
    try:
        return _RECOVERY_CAPABILITIES[kind]
    except KeyError as error:
        raise ValueError(f"unknown recovery capability {kind!r}") from error
