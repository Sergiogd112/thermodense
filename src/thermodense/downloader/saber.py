from __future__ import annotations

import json
import os
import re
from datetime import date, timedelta
from pathlib import Path

from .common import (
    curl_text,
    download_parallel,
    get_base_dir,
    parse_apache_index_filenames,
)
from .manifest import create_or_update_manifest, ensure_dir, now_iso

BASE_URL = (
    "https://data.gats-inc.com/saber/Version2_0/SABER_cooling/CO2_CoolingRate_Profiles/"
)
NO_BASE_URL = (
    "https://data.gats-inc.com/saber/Version2_0/SABER_cooling/NO_CoolingRate_Profiles/"
)
L2A_BASE_URL = "https://data.gats-inc.com/saber/Version2_0/Level2A/"
REF_ROOT = get_base_dir() / "data" / "original"
DEST_DIR = REF_ROOT / "saber" / "co2_cooling_profiles"
MANIFEST_PATH = DEST_DIR / "manifest.json"
FILENAME_RE = re.compile(r"SABER_CO2_PROFILE_FLUX_(\d{4})(\d{3})_V(1\.\d+)\.nc")
NO_FILENAME_RE = re.compile(r"SABER_NO_PROFILE_FLUX_(\d{4})(\d{3})_V(1\.\d+)\.nc")
L2A_FILENAME_RE = re.compile(r"SABER_L2A_(\d{4})(\d{3})_(\d+)_(02\.\d+)\.nc")
NO_DEST_DIR = REF_ROOT / "saber" / "no_cooling_profiles"
NO_MANIFEST_PATH = NO_DEST_DIR / "manifest.json"
L2A_DEST_DIR = REF_ROOT / "saber" / "level2a"
L2A_MANIFEST_PATH = L2A_DEST_DIR / "manifest.json"
INVENTORY_FILENAME = "remote_inventory.json"


def doy_to_date(year: int, doy: int) -> date:
    return date(year, 1, 1) + timedelta(days=doy - 1)


def _latest_daily_urls(
    base_url: str, pattern: re.Pattern[str]
) -> list[tuple[date, str, str]]:
    """Discover one highest-version cooling file per day from an Apache index."""
    latest: dict[date, tuple[tuple[int, ...], str]] = {}
    for filename in parse_apache_index_filenames(curl_text(base_url, timeout_s=120)):
        match = pattern.fullmatch(filename)
        if not match:
            continue
        day = doy_to_date(int(match.group(1)), int(match.group(2)))
        version = tuple(int(piece) for piece in match.group(3).split("."))
        if day not in latest or version > latest[day][0]:
            latest[day] = (version, filename)
    return [
        (day, filename, f"{base_url}{filename}")
        for day, (_, filename) in sorted(latest.items())
    ]


def list_saber_co2_cooling_urls() -> list[tuple[date, str, str]]:
    """List the latest official CO2 cooling file for each available UTC day."""
    return _latest_daily_urls(BASE_URL, FILENAME_RE)


def list_saber_no_cooling_urls() -> list[tuple[date, str, str]]:
    """List the latest official NO cooling file for each available UTC day."""
    return _latest_daily_urls(NO_BASE_URL, NO_FILENAME_RE)


def _write_inventory(
    directory: Path, dataset: str, source_url: str, rows: list[tuple[date, str, str]]
) -> None:
    """Atomically preserve the complete remote latest-per-day directory listing."""
    inventory = {
        "dataset": dataset,
        "source_url": source_url,
        "discovered_at": now_iso(),
        "files": [
            {"date": day.isoformat(), "filename": filename}
            for day, filename, _url in rows
        ],
        "available_through": rows[-1][0].isoformat() if rows else None,
    }
    path = directory / INVENTORY_FILENAME
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def saber_l2a_url_rows(
    day: date, event_ids: set[int] | None = None
) -> list[tuple[str, str]]:
    """List the required official L2A orbit files for one day.

    SABER changed the official L2A version from 02.07 to 02.08 on 2019-12-15.
    Older versions advertised in the directory are deliberately not selected.
    """
    def matches_in(directory_day: date) -> dict[int, list[tuple[str, str]]]:
        version = "02.07" if directory_day < date(2019, 12, 15) else "02.08"
        url = (
            f"{L2A_BASE_URL}{directory_day.year:04d}/"
            f"{directory_day.timetuple().tm_yday:03d}/"
        )
        matches: dict[int, list[tuple[str, str]]] = {}
        for filename in parse_apache_index_filenames(curl_text(url, timeout_s=120)):
            match = L2A_FILENAME_RE.fullmatch(filename)
            if not match or match.group(4) != version:
                continue
            event_id = int(match.group(3))
            if event_ids is not None and event_id not in event_ids:
                continue
            matches.setdefault(event_id, []).append((filename, f"{url}{filename}"))
        return matches

    same_day = matches_in(day)
    if event_ids is None:
        return [
            max(matches)
            for _event_id, matches in sorted(same_day.items())
        ]

    selected: dict[int, tuple[str, str]] = {}
    missing = []
    for event_id in sorted(event_ids):
        matches = same_day.get(event_id, [])
        if len(matches) == 1:
            selected[event_id] = matches[0]
        elif len(matches) == 0:
            missing.append(event_id)
        else:
            raise RuntimeError(
                f"Expected exactly one official Level2A file for {day} orbit "
                f"{event_id}, found {len(matches)}."
            )
    if missing:
        previous_day = matches_in(day - timedelta(days=1))
        next_day = matches_in(day + timedelta(days=1))
    for event_id in missing:
        matches = previous_day.get(event_id, []) + next_day.get(event_id, [])
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one official Level2A file for {day} orbit "
                f"{event_id}, found {len(matches)}."
            )
        selected[event_id] = matches[0]
    return [selected[event_id] for event_id in sorted(selected)]


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
    _write_inventory(DEST_DIR, "saber_co2_cooling_profiles", BASE_URL, rows)
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


def download_saber_no_cooling(
    *,
    start_date: date | None = date(2002, 1, 1),
    end_date: date | None = date(2025, 12, 31),
    overwrite: bool = False,
    resume: bool = True,
    max_workers: int = 4,
) -> None:
    """Download latest-version official daily SABER NO cooling files."""
    ensure_dir(NO_DEST_DIR)
    available = list_saber_no_cooling_urls()
    _write_inventory(NO_DEST_DIR, "saber_no_cooling_profiles", NO_BASE_URL, available)
    rows = [
        row
        for row in available
        if (start_date is None or row[0] >= start_date)
        and (end_date is None or row[0] <= end_date)
    ]
    entries, _ = download_parallel(
        [(url, NO_DEST_DIR / filename) for _, filename, url in rows],
        get_base_dir(),
        overwrite=overwrite,
        resume=resume,
        max_workers=max_workers,
        timeout_s=180,
        desc="SABER NO cooling",
    )
    create_or_update_manifest("saber_no_cooling_profiles", NO_MANIFEST_PATH, entries)


def download_saber_l2a(
    required_events: dict[date, set[int]],
    *,
    overwrite: bool = False,
    resume: bool = True,
    max_workers: int = 4,
) -> None:
    """Download only L2A orbit/event files identified by qualifying cooling scans."""
    ensure_dir(L2A_DEST_DIR)
    rows = [
        row
        for day, events in sorted(required_events.items())
        for row in saber_l2a_url_rows(day, events)
    ]
    entries, _ = download_parallel(
        [(url, L2A_DEST_DIR / filename) for filename, url in rows],
        get_base_dir(),
        overwrite=overwrite,
        resume=resume,
        max_workers=max_workers,
        timeout_s=180,
        desc="SABER Level2A",
    )
    create_or_update_manifest("saber_level2a_profiles", L2A_MANIFEST_PATH, entries)
