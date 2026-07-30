from __future__ import annotations

import re
from datetime import date, timedelta

from .common import curl_text, download_parallel, get_base_dir
from .manifest import create_or_update_manifest, ensure_dir

BASE_URL = (
    "https://data.gats-inc.com/saber/Version2_0/SABER_cooling/CO2_CoolingRate_Profiles/"
)
REF_ROOT = get_base_dir() / "data" / "original"
DEST_DIR = REF_ROOT / "saber" / "co2_cooling_profiles"
MANIFEST_PATH = DEST_DIR / "manifest.json"
FILENAME_RE = re.compile(r"SABER_CO2_PROFILE_FLUX_(\d{4})(\d{3})_V1\.[01]\.nc")


def doy_to_date(year: int, doy: int) -> date:
    return date(year, 1, 1) + timedelta(days=doy - 1)


def list_saber_co2_cooling_urls() -> list[tuple[date, str, str]]:
    index = curl_text(BASE_URL, timeout_s=120)
    seen: set[str] = set()
    rows: list[tuple[date, str, str]] = []
    for match in FILENAME_RE.finditer(index):
        filename = match.group(0)
        if filename in seen:
            continue
        seen.add(filename)
        year = int(match.group(1))
        doy = int(match.group(2))
        rows.append((doy_to_date(year, doy), filename, f"{BASE_URL}{filename}"))
    return sorted(rows, key=lambda row: row[0])


def download_saber_co2_cooling(
    *,
    start_date: date | None = date(2002, 1, 1),
    end_date: date | None = date(2025, 12, 31),
    overwrite: bool = False,
    resume: bool = True,
    max_workers: int = 4,
) -> None:
    """Download SABER CO2 cooling profile NetCDF files.

    Files are discovered from the GATS HTTP directory index so missing days in the
    public archive are skipped rather than guessed.
    """
    ensure_dir(DEST_DIR)
    rows = list_saber_co2_cooling_urls()
    if start_date is not None:
        rows = [row for row in rows if row[0] >= start_date]
    if end_date is not None:
        rows = [row for row in rows if row[0] <= end_date]

    downloads = [(url, DEST_DIR / filename) for _day, filename, url in rows]
    entries, _counters = download_parallel(
        downloads,
        get_base_dir(),
        overwrite=overwrite,
        resume=resume,
        max_workers=max_workers,
        timeout_s=180,
        desc="SABER CO2 cooling",
    )
    create_or_update_manifest("saber_co2_cooling_profiles", MANIFEST_PATH, entries)
    downloaded = sum(1 for entry in entries if entry.status == "downloaded")
    skipped = sum(1 for entry in entries if entry.status == "skipped")
    failed = sum(1 for entry in entries if entry.status == "failed")
    print(
        "SABER CO2 cooling download summary: "
        f"downloaded={downloaded}, "
        f"skipped_existing={skipped}, "
        f"failed={failed}"
    )


sync_saber_co2_cooling = download_saber_co2_cooling
