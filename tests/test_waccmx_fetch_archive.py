"""Offline test of the WACCM-X archive streamer: stored ZIP in, byte-exact .xz members out."""

import importlib.util
import io
import json
import lzma
import subprocess
import zipfile
import zlib
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "fetch_archive", Path(__file__).resolve().parents[1] / "scripts/waccmx/fetch_archive.py"
)
fetch_archive = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch_archive)


def stored_zip(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


class FakeCurl:
    def __init__(self, payload: bytes):
        self.stdout = io.BytesIO(payload)

    def wait(self):
        return 0

    def kill(self):
        pass


def test_fetch_stores_every_member_byte_exact_with_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    members = {f"WACCM-X_2014-01-0{d}_16-25_200-210.nc": bytes([d]) * (1000 + d) for d in (1, 2, 3)}
    payload = stored_zip(members)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: FakeCurl(payload))
    monkeypatch.setattr(fetch_archive, "MIN_FREE_BYTES", 0)
    archive = "WACCM-X_2014-01-01_2014-12-31_16-25_200-210.zip"
    fetch_archive.fetch(archive, tmp_path, len(payload), "ftp://example.invalid/dir/")

    year = tmp_path / archive[:-4]
    complete = json.loads((year / "COMPLETE.json").read_text())
    manifest = json.loads((year / "manifest.json").read_text())
    assert complete["members"] == 3 and complete["bytes_read"] == len(payload)
    assert complete["trailing_bytes"] > 0  # the central directory follows the last member
    for entry in manifest:
        data = lzma.decompress((year / (entry["name"] + ".xz")).read_bytes())
        assert data == members[entry["name"]]
        assert entry["crc32"] == "%08x" % zlib.crc32(data)


def test_fetch_rejects_a_byte_count_that_disagrees_with_the_listing(tmp_path, monkeypatch):
    payload = stored_zip({"WACCM-X_2014-01-01_16-25_200-210.nc": b"x" * 100})
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: FakeCurl(payload))
    monkeypatch.setattr(fetch_archive, "MIN_FREE_BYTES", 0)
    with pytest.raises(ValueError, match="listing"):
        fetch_archive.fetch("WACCM-X_2014-01-01_2014-12-31_16-25_200-210.zip", tmp_path, len(payload) + 1,
                            "ftp://example.invalid/dir/")
    assert not list(tmp_path.glob("WACCM-X_*_200-210"))  # nothing published
