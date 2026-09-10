"""Public contracts for artifact integrity and immutable publication."""

import os
from pathlib import Path

import pytest

from thermodense.artifacts import (
    ArtifactIntegrityError,
    RecoveryCapability,
    publish_json_new,
    recovery_capability,
    verify_files,
)
from thermodense.checkpoints import fingerprint


def _digest(path: Path) -> str:
    value = fingerprint(path)["sha256"]
    assert isinstance(value, str)
    return value


def test_verify_files_hashes_contents_not_same_size_or_mtime(tmp_path):
    artifact = tmp_path / "metrics.json"
    artifact.write_bytes(b"original")
    expected = {"metrics.json": _digest(artifact)}
    original_stat = artifact.stat()
    artifact.write_bytes(b"tampered")
    os.utime(artifact, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    with pytest.raises(ArtifactIntegrityError, match="metrics.json.*declared"):
        verify_files(tmp_path, expected, required=("metrics.json",))


@pytest.mark.parametrize("expected", ({}, {"metrics.json": "not-a-digest"}))
def test_verify_files_rejects_empty_or_malformed_digest_mapping(tmp_path, expected):
    with pytest.raises(ArtifactIntegrityError):
        verify_files(tmp_path, expected)


def test_verify_files_rejects_traversal_and_symlink_escape(tmp_path):
    outside = tmp_path.parent / "outside-artifact"
    outside.write_text("outside")
    with pytest.raises(ArtifactIntegrityError, match="absolute or traverse"):
        verify_files(tmp_path, {"../outside-artifact": _digest(outside)})
    link = tmp_path / "escape.json"
    link.symlink_to(outside)
    with pytest.raises(ArtifactIntegrityError, match="escapes"):
        verify_files(tmp_path, {"escape.json": _digest(outside)})


def test_verify_files_rejects_a_missing_declared_file(tmp_path):
    with pytest.raises(ArtifactIntegrityError, match="missing"):
        verify_files(tmp_path, {"missing.json": "0" * 64}, required=("missing.json",))


def test_publish_json_new_is_no_clobber_and_cleans_failed_serialization(tmp_path):
    output = tmp_path / "comparison.json"
    digest = publish_json_new(output, {"value": 1})
    assert digest == _digest(output)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        publish_json_new(output, {"value": 2})
    assert output.read_text() == '{\n  "value": 1\n}\n'
    failed = tmp_path / "failed.json"
    with pytest.raises(ValueError, match="Out of range"):
        publish_json_new(failed, {"value": float("nan")})
    assert not failed.exists()
    assert not list(tmp_path.glob(".failed.json.*.tmp"))


def test_publish_json_new_cleans_temporary_after_failed_write(monkeypatch, tmp_path):
    output = tmp_path / "failed-write.json"

    def fail_fsync(_: int) -> None:
        raise OSError("simulated fsync failure")

    monkeypatch.setattr("thermodense.artifacts.os.fsync", fail_fsync)
    with pytest.raises(OSError, match="simulated fsync failure"):
        publish_json_new(output, {"value": 1})
    assert not output.exists()
    assert not list(tmp_path.glob(".failed-write.json.*.tmp"))


def test_recovery_capabilities_are_explicit_declarations():
    for kind in ("global-parcorr", "raw-36-parcorr", "global-polynomial", "comparison"):
        assert isinstance(recovery_capability(kind), RecoveryCapability)
    assert not recovery_capability("global-parcorr").supports_fit_resume
    assert recovery_capability("global-polynomial").supports_fit_resume
    with pytest.raises(ValueError, match="unknown recovery"):
        recovery_capability("unknown")
