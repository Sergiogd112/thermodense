from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from .common import download_parallel, ensure_dir, get_base_dir
from .counter import Counters
from .manifest import create_or_update_manifest

BASE_DIR = get_base_dir()
REF_ROOT = Path("data") / "original"
DEST_DIR = REF_ROOT / "space_weather"
MANIFEST_PATH = DEST_DIR / "manifest.json"
SPACE_WEATHER_CSV_PATH = DEST_DIR / "SW-All.csv"

CSV_COLUMNS = (
    "DATE",
    "BSRN",
    "ND",
    "KP1",
    "KP2",
    "KP3",
    "KP4",
    "KP5",
    "KP6",
    "KP7",
    "KP8",
    "KP_SUM",
    "AP1",
    "AP2",
    "AP3",
    "AP4",
    "AP5",
    "AP6",
    "AP7",
    "AP8",
    "AP_AVG",
    "CP",
    "C9",
    "ISN",
    "F10.7_OBS",
    "F10.7_ADJ",
    "F10.7_DATA_TYPE",
    "F10.7_OBS_CENTER81",
    "F10.7_OBS_LAST81",
    "F10.7_ADJ_CENTER81",
    "F10.7_ADJ_LAST81",
)

_SECTION_DATA_TYPES = {
    "OBSERVED": "OBS",
    "DAILY_PREDICTED": "PRD",
    "MONTHLY_PREDICTED": "PRM",
}

_FIELD_WIDTHS = (
    4,
    3,
    3,
    5,
    3,
    *([3] * 8),
    4,
    *([4] * 8),
    4,
    4,
    2,
    4,
    6,
    2,
    *([6] * 5),
)

# CelesTrak's fixed-width format stores adjusted flux first (fields 26, 28,
# 29), followed by observed flux (fields 30, 31, 32). The compatibility CSV
# deliberately restores the historical observed-before-adjusted column order.

SOURCES: dict[str, str] = {
    "SW-Last5Years.txt": "https://celestrak.org/SpaceData/SW-Last5Years.txt",
    "SW-All.txt": "https://celestrak.org/SpaceData/SW-All.txt",
    "f107_nrcan_daily.txt": "https://www.spaceweather.gc.ca/solar_flux_data/daily_flux_values/fluxtable.txt",
    "omni2_all_years.dat": "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_all_years.dat",
}


def parse_celestrak_space_weather(lines: Iterable[str]) -> list[dict[str, str]]:
    """Convert CelesTrak's sectioned SW-All text format to the legacy CSV schema."""
    rows: list[dict[str, str]] = []
    data_type: str | None = None

    for line in lines:
        directive = line.split()
        if len(directive) == 2 and directive[0] == "BEGIN":
            data_type = _SECTION_DATA_TYPES.get(directive[1])
            continue
        if len(directive) == 2 and directive[0] == "END":
            data_type = None
            continue
        if data_type is None:
            continue

        fields = _parse_fixed_width_record(line)
        if fields is None:
            continue

        year, month, day = (int(value) for value in fields[:3])
        rows.append(
            dict(
                zip(
                    CSV_COLUMNS,
                    (
                        f"{year:04d}-{month:02d}-{day:02d}",
                        *fields[3:26],
                        fields[30],
                        fields[26],
                        data_type,
                        fields[31],
                        fields[32],
                        fields[28],
                        fields[29],
                    ),
                    strict=True,
                )
            )
        )
    return rows


def _parse_fixed_width_record(line: str) -> list[str] | None:
    """Parse the CelesTrak FORMAT record without collapsing blank fields."""
    record = line.rstrip()
    if len(record) > sum(_FIELD_WIDTHS):
        return None

    fields = []
    start = 0
    for width in _FIELD_WIDTHS:
        fields.append(record[start : start + width].strip())
        start += width
    return fields if all(fields[:3]) else None


def prepare_space_weather_csv(source: Path, destination: Path = SPACE_WEATHER_CSV_PATH) -> int:
    """Write the compatibility CSV consumed by retained scientific scripts."""
    rows = parse_celestrak_space_weather(source.read_text(encoding="utf-8").splitlines())
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


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

    source = DEST_DIR / "SW-All.txt"
    if source.exists():
        csv_path = DEST_DIR / "SW-All.csv"
        row_count = prepare_space_weather_csv(source, csv_path)
        print(f"Prepared {csv_path}: {row_count} rows")

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
