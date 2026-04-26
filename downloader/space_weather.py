from __future__ import annotations

from pathlib import Path

from .common import download_parallel, ensure_dir, get_base_dir
from .counter import Counters
from .manifest import ManifestEntry, create_or_update_manifest

BASE_DIR = get_base_dir()
REF_ROOT = BASE_DIR / "data" / "original"
DEST_DIR = REF_ROOT / "space_weather"
MANIFEST_PATH = DEST_DIR / "manifest.json"

SOURCES: dict[str, str] = {
    "SW-Last5Years.txt": "https://celestrak.org/SpaceData/SW-Last5Years.txt",
    "SW-All.txt": "https://celestrak.org/SpaceData/SW-All.txt",
    "f107_nrcan_daily.txt": "https://www.spaceweather.gc.ca/solar_flux_data/daily_flux_values/fluxtable.txt",
    "omni2_all_years.dat": "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_all_years.dat",
}


def download_space_weather(
    *,
    overwrite: bool = False,
    resume: bool = True,
    max_workers: int = 4,
) -> Counters:
    """Download space weather indices for density modeling.

    Downloads solar flux and geomagnetic activity data from Celestrak,
    Canadian Space Weather, and NASA SPDF.

    Args:
        overwrite: If True, re-download existing files. If False, skip existing files.
        resume: If True, resume partial downloads using curl's continue feature.
        max_workers: Maximum number of concurrent download threads (default: 4).

    Returns:
        Counters object with downloaded, skipped_existing, and failed counts.

    Example:
        >>> counters = download_space_weather(overwrite=False, resume=True, max_workers=2)
        >>> print(f"Downloaded: {counters.downloaded}")
    """
    ensure_dir(DEST_DIR)

    # Prepare download list
    downloads: list[tuple[str, Path]] = []
    for name, url in SOURCES.items():
        out_path = DEST_DIR / name
        downloads.append((url, out_path))

    # Download all files in parallel
    entries, counters = download_parallel(
        downloads,
        REF_ROOT,
        overwrite=overwrite,
        resume=resume,
        max_workers=max_workers,
        timeout_s=180,
        desc="Downloading space weather data",
    )

    # Save manifest with all entries
    manifest = create_or_update_manifest(
        dataset="space_weather",
        manifest_path=MANIFEST_PATH,
        entries=entries,
    )
    print(f"Manifest saved: {MANIFEST_PATH}")
    print(f"  Total tracked files: {len(manifest.entries)}")
    print(f"  Downloaded: {counters.downloaded}")
    print(f"  Skipped: {counters.skipped_existing}")
    print(f"  Failed: {counters.failed}")

    return counters


# Backwards compatibility alias
sync_space_weather = download_space_weather
