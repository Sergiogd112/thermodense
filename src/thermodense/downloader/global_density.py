from __future__ import annotations

import os
from pathlib import Path

from .common import (
    download_parallel,
    get_base_dir,
    parse_apache_index_filenames,
    safe_name,
    curl_text,
)
from .counter import Counters
from .manifest import ManifestEntry, create_or_update_manifest

BASE_DIR = get_base_dir()
REF_ROOT = Path("data") / "original"
MANIFEST_PATH = REF_ROOT / "global_density" / "manifest.json"
ALLOWED_SUFFIXES = (".zip", ".txt", ".csv", ".dat", ".tsv")


def _split_env_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [
        item.strip() for item in value.replace("\n", ",").split(",") if item.strip()
    ]


def _select_global_density_files(
    names: list[str],
    patterns: tuple[str, ...],
) -> list[str]:
    selected = [name for name in names if name.lower().endswith(ALLOWED_SUFFIXES)]
    if not patterns:
        return selected

    lowered_patterns = tuple(pattern.lower() for pattern in patterns)
    return [
        name
        for name in selected
        if any(pattern in name.lower() for pattern in lowered_patterns)
    ]


def download_global_density(
    *,
    urls: list[str] | None = None,
    index_urls: list[str] | None = None,
    filename_patterns: tuple[str, ...] = (),
    overwrite: bool = False,
    resume: bool = True,
    max_workers: int = 4,
) -> Counters:
    """Download daily global-average density files.

    This downloader is intended for files that provide daily global-average
    density as a function of altitude. Since the source path is not yet public
    in the repository, callers can provide either:

    - direct file URLs via ``urls``
    - directory index URLs via ``index_urls``, optionally filtered with
      ``filename_patterns``

    The same values can also be supplied via environment variables:

    - ``GLOBAL_DENSITY_URLS``: comma- or newline-separated direct URLs
    - ``GLOBAL_DENSITY_INDEX_URLS``: comma- or newline-separated index URLs
    """
    root = REF_ROOT / "global_density"
    root.mkdir(parents=True, exist_ok=True)

    resolved_urls = list(urls or _split_env_list(os.getenv("GLOBAL_DENSITY_URLS")))
    resolved_index_urls = list(
        index_urls or _split_env_list(os.getenv("GLOBAL_DENSITY_INDEX_URLS"))
    )

    if not resolved_urls and not resolved_index_urls:
        raise ValueError(
            "No global density source URLs were provided. "
            "Pass urls/index_urls explicitly or set GLOBAL_DENSITY_URLS / GLOBAL_DENSITY_INDEX_URLS."
        )

    all_downloads: list[tuple[str, Path]] = []
    all_entries: list[ManifestEntry] = []

    for url in resolved_urls:
        all_downloads.append((url, root / safe_name(url)))

    for index_url in resolved_index_urls:
        try:
            html = curl_text(index_url, retries=2, retry_delay=2, timeout_s=60)
        except Exception as exc:
            all_entries.append(
                ManifestEntry(
                    path=f"global_density/{safe_name(index_url)}",
                    url=index_url,
                    status="failed",
                    error=str(exc)[:200],
                )
            )
            print(f"  FAILED listing global density index {index_url}: {exc}")
            continue

        names = parse_apache_index_filenames(html)
        names = _select_global_density_files(names, filename_patterns)
        print(f"Global density index {index_url}: {len(names)} files selected")
        for name in names:
            url = index_url.rstrip("/") + "/" + name
            all_downloads.append((url, root / name))

    if all_downloads:
        entries, counters = download_parallel(
            all_downloads,
            REF_ROOT,
            overwrite=overwrite,
            resume=resume,
            max_workers=max_workers,
            timeout_s=60,
            desc=f"Downloading {len(all_downloads)} global density files",
        )
        all_entries.extend(entries)
    else:
        counters = Counters()

    manifest = create_or_update_manifest(
        dataset="global_density",
        manifest_path=MANIFEST_PATH,
        entries=all_entries,
    )
    print(f"Manifest saved: {MANIFEST_PATH}")
    print(f"  Total tracked files: {len(manifest.entries)}")
    print(f"  Downloaded: {counters.downloaded}")
    print(f"  Skipped: {counters.skipped_existing}")
    print(f"  Failed: {counters.failed}")

    return counters


sync_global_density = download_global_density
