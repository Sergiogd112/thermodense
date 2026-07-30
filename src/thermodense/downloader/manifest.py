from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal


@dataclass
class ManifestEntry:
    """Record of a single file download operation."""

    path: str
    url: str
    status: Literal["downloaded", "skipped", "failed"]
    timestamp: str
    size_bytes: int | None = None
    error: str | None = None


@dataclass
class Manifest:
    """Manifest of download operations for a dataset."""

    dataset: str
    created_at: str
    entries: list[ManifestEntry]

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "created_at": self.created_at,
            "entries": [asdict(e) for e in self.entries],
        }

    def save(self, path: Path) -> None:
        """Save manifest to JSON file."""
        ensure_dir(path.parent)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> Manifest | None:
        """Load manifest from JSON file."""
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        entries = [ManifestEntry(**e) for e in data.get("entries", [])]
        return cls(
            dataset=data.get("dataset", "unknown"),
            created_at=data.get("created_at", ""),
            entries=entries,
        )

    def get_entry(self, file_path: str) -> ManifestEntry | None:
        """Get entry for a specific file path."""
        for entry in self.entries:
            if entry.path == file_path:
                return entry
        return None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    """Current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def should_download(
    file_path: Path,
    url: str,
    manifest_path: Path,
    *,
    overwrite: bool,
) -> tuple[bool, Manifest | None]:
    """Check if a file should be downloaded based on manifest and file existence.

    Returns:
        Tuple of (should_download, manifest).
    """
    manifest = Manifest.load(manifest_path)

    if overwrite:
        return True, manifest

    if file_path.exists() and file_path.stat().st_size > 0:
        # File exists, check if URL matches manifest
        if manifest:
            rel_path = str(file_path.relative_to(file_path.parent.parent))
            entry = manifest.get_entry(rel_path)
            if entry and entry.url == url and entry.status == "downloaded":
                return False, manifest
        return False, manifest

    return True, manifest


def create_or_update_manifest(
    dataset: str,
    manifest_path: Path,
    entries: list[ManifestEntry],
) -> Manifest:
    """Create new manifest or update existing one with new entries."""
    existing = Manifest.load(manifest_path)

    if existing:
        # Create a map of existing entries by path
        entry_map = {e.path: e for e in existing.entries}
        # Update with new entries
        for new_entry in entries:
            entry_map[new_entry.path] = new_entry
        all_entries = list(entry_map.values())
    else:
        all_entries = entries

    manifest = Manifest(
        dataset=dataset,
        created_at=now_iso(),
        entries=all_entries,
    )
    manifest.save(manifest_path)
    return manifest
