from __future__ import annotations

from pathlib import Path

from .common import download_parallel, ensure_dir, get_base_dir
from .counter import Counters
from .manifest import ManifestEntry, create_or_update_manifest

BASE_DIR = get_base_dir()
REF_ROOT = BASE_DIR / "data" / "original"
DEST_DIR = REF_ROOT / "co2"
MANIFEST_PATH = DEST_DIR / "manifest.json"

BASE_URL = "https://gml.noaa.gov/webdata/ccgg/trends/co2"

FILES: dict[str, str] = {
    "co2_mm_mlo.csv": "mauna_loa_monthly.csv",
    "co2_annmean_mlo.csv": "mauna_loa_annual.csv",
    "co2_weekly_mlo.csv": "mauna_loa_weekly.csv",
    "co2_daily_mlo.csv": "mauna_loa_daily.csv",
    "co2_gr_mlo.csv": "mauna_loa_growth_rate.csv",
    "co2_mm_gl.csv": "global_monthly.csv",
    "co2_annmean_gl.csv": "global_annual.csv",
    "co2_gr_gl.csv": "global_growth_rate.csv",
    "co2_trend_gl.csv": "global_trend.csv",
}


def download_co2(
    *,
    overwrite: bool = False,
    resume: bool = True,
    max_workers: int = 4,
) -> Counters:
    """Download NOAA CO2 datasets.

    Downloads Mauna Loa and global CO2 measurements from NOAA GML,
    including monthly, annual, weekly, and daily data.

    Args:
        overwrite: If True, re-download existing files. If False, skip existing files.
        resume: If True, resume partial downloads using curl's continue feature.
        max_workers: Maximum number of concurrent download threads (default: 4).

    Returns:
        Counters object with downloaded, skipped_existing, and failed counts.

    Example:
        >>> counters = download_co2(overwrite=False, resume=True, max_workers=2)
        >>> print(f"Downloaded: {counters.downloaded}")
    """
    ensure_dir(DEST_DIR)

    # Prepare download list
    downloads: list[tuple[str, Path]] = []
    for src, dst in FILES.items():
        url = f"{BASE_URL}/{src}"
        out_path = DEST_DIR / dst
        downloads.append((url, out_path))

    # Download all files in parallel
    entries, counters = download_parallel(
        downloads,
        REF_ROOT,
        overwrite=overwrite,
        resume=resume,
        max_workers=max_workers,
        retries=4,
        retry_delay=2,
        timeout_s=120,
        desc="Downloading CO2 data",
    )

    # Save manifest with all entries
    manifest = create_or_update_manifest(
        dataset="co2",
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
sync_co2 = download_co2
