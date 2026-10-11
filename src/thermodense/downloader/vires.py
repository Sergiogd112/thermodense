"""ESA VirES HAPI mirror of the TU Delft (TOLEOS) thermosphere density products.

VirES serves the same CHAMP, GRACE, GRACE-FO and Swarm density data as
thermosphere.tudelft.nl, anonymously, through a HAPI 3.0 interface. Values are
on a UTC time grid (TU Delft text files are GPS for most missions), so treat
this as an alternative source, not a byte-identical reproduction. GOCE is not
served by VirES; it is on ESA's GOCE dissemination server (EO Sign In required).

Each month is stored gzip-compressed as
``data/original/vires/<mission>/<CODE>_DNS_<KIND>_YYYY_MM_vires.csv.gz``: the HAPI
JSON header (``#``-prefixed, including the data terms) followed by CSV rows. Data terms: https://vires.services/data_terms.
"""

from __future__ import annotations

import gzip
import json
import shutil
from datetime import date
from pathlib import Path
from urllib.parse import urlencode

from .common import curl_text, download_parallel, get_base_dir
from .counter import Counters
from .manifest import ManifestEntry, create_or_update_manifest

HAPI_URL = "https://vires.services/hapi"
REF_ROOT = get_base_dir() / "data" / "original"
DEST_DIR = REF_ROOT / "vires"
MANIFEST_PATH = DEST_DIR / "manifest.json"

# VirES collection -> (mission directory, TU Delft-style file code, product kind).
VIRES_COLLECTIONS: dict[str, tuple[str, str, str]] = {
    "CH_OPER_DNS_ACC_2_": ("champ", "CH", "ACC"),
    "GR_OPER_DNS1ACC_2_": ("grace", "GA", "ACC"),
    "GR_OPER_DNS2ACC_2_": ("grace", "GB", "ACC"),
    "GF_OPER_DNS1ACC_2_": ("grace_fo", "GC", "ACC"),
    "SW_OPER_DNSAPOD_2_": ("swarm", "SA", "POD"),
    "SW_OPER_DNSBPOD_2_": ("swarm", "SB", "POD"),
    "SW_OPER_DNSCPOD_2_": ("swarm", "SC", "POD"),
    "SW_OPER_DNSAACC_2_": ("swarm", "SWA", "ACC"),
    "SW_OPER_DNSBACC_2_": ("swarm", "SWB", "ACC"),
    "SW_OPER_DNSCACC_2_": ("swarm", "SWC", "ACC"),
}
# Swarm POD matches the TU Delft Swarm files already used; ACC is opt-in.
DEFAULT_COLLECTIONS = tuple(c for c in VIRES_COLLECTIONS if not c.startswith("SW_") or "POD" in c)


def collection_info(collection: str) -> dict:
    """Return the HAPI ``info`` document (coverage, cadence, parameters)."""
    query = urlencode({"dataset": collection})
    return json.loads(curl_text(f"{HAPI_URL}/info?{query}", timeout_s=120))


def month_starts(start: date, stop: date) -> list[date]:
    """First day of every month overlapping ``[start, stop]``."""
    months = []
    current = date(start.year, start.month, 1)
    while current <= stop:
        months.append(current)
        current = date(current.year + current.month // 12, current.month % 12 + 1, 1)
    return months


def month_url(collection: str, month: date) -> str:
    """HAPI CSV request for one calendar month, with the JSON header included."""
    next_month = date(month.year + month.month // 12, month.month % 12 + 1, 1)
    query = urlencode(
        {
            "dataset": collection,
            "start": f"{month.isoformat()}T00:00:00Z",
            "stop": f"{next_month.isoformat()}T00:00:00Z",
            "format": "csv",
            "include": "header",
        }
    )
    return f"{HAPI_URL}/data?{query}"


def month_path(collection: str, month: date, root: Path = DEST_DIR) -> Path:
    mission, code, kind = VIRES_COLLECTIONS[collection]
    return root / mission / f"{code}_DNS_{kind}_{month.year}_{month.month:02d}_vires.csv.gz"


def _compress(entry: ManifestEntry, csv_path: Path) -> ManifestEntry:
    """Replace a downloaded CSV with its gzip version and update the entry.

    The archive is written to a temporary name and renamed only when complete,
    so an interrupted compression never leaves a final-looking file that a
    later run would skip.
    """
    gz_path = csv_path.with_name(csv_path.name + ".gz")
    partial = gz_path.with_name(gz_path.name + ".partial")
    try:
        with csv_path.open("rb") as source, gzip.open(partial, "wb") as target:
            shutil.copyfileobj(source, target)
        partial.replace(gz_path)
    finally:
        partial.unlink(missing_ok=True)
    csv_path.unlink()
    entry.path = entry.path + ".gz"
    entry.size_bytes = gz_path.stat().st_size
    return entry


def _in_range(month: date, start_ym: tuple[int, int] | None, end_ym: tuple[int, int] | None) -> bool:
    key = (month.year, month.month)
    return (start_ym is None or key >= start_ym) and (end_ym is None or key <= end_ym)


def download_vires(
    collections: list[str] | None = None,
    start_ym: tuple[int, int] | None = None,
    end_ym: tuple[int, int] | None = None,
    *,
    overwrite: bool = False,
    max_workers: int = 4,
) -> Counters:
    """Download monthly VirES density files for the selected collections.

    Only months inside each collection's published HAPI coverage are requested
    (VirES answers HTTP 400 outside it). HAPI responses cannot be resumed, so an
    interrupted month is fetched again in full on the next run.

    Args:
        collections: VirES collection IDs (keys of ``VIRES_COLLECTIONS``).
            Defaults to CHAMP, GRACE A/B, GRACE-FO and Swarm A/B/C POD.
        start_ym: Optional inclusive (year, month) lower bound.
        end_ym: Optional inclusive (year, month) upper bound.
        overwrite: Re-download months that already exist.
        max_workers: Concurrent requests (keep modest; VirES is a shared service).

    Returns:
        Counters with downloaded, skipped_existing and failed counts.
    """
    downloads: list[tuple[str, Path]] = []
    entries: list[ManifestEntry] = []
    skipped = 0
    for collection in collections or list(DEFAULT_COLLECTIONS):
        if collection not in VIRES_COLLECTIONS:
            raise ValueError(f"Unknown VirES collection: {collection}")
        info = collection_info(collection)
        start = date.fromisoformat(info["startDate"][:10])
        stop = date.fromisoformat(info["stopDate"][:10])
        months = [m for m in month_starts(start, stop) if _in_range(m, start_ym, end_ym)]
        print(f"VirES {collection}: {len(months)} months ({start} to {stop})")
        for month in months:
            target = month_path(collection, month)
            if target.exists() and target.stat().st_size > 0 and not overwrite:
                skipped += 1
                continue
            # Download uncompressed (``download_parallel`` checks this path), then gzip.
            # A leftover .csv can only be a partial download, so never reuse it.
            partial = target.with_suffix("")
            partial.unlink(missing_ok=True)
            downloads.append((month_url(collection, month), partial))

    if downloads:
        entries, counters = download_parallel(
            downloads,
            REF_ROOT,
            overwrite=overwrite,
            resume=False,
            max_workers=max_workers,
            timeout_s=1800,
            desc=f"Downloading {len(downloads)} VirES months",
        )
        entries = [
            _compress(entry, path) if entry.status == "downloaded" else entry
            for entry, (_url, path) in zip(entries, downloads)
        ]
    else:
        counters = Counters()
    counters.skipped_existing += skipped

    manifest = create_or_update_manifest(
        dataset="vires", manifest_path=MANIFEST_PATH, entries=entries
    )
    print(f"Manifest saved: {MANIFEST_PATH}")
    print(f"  Total tracked files: {len(manifest.entries)}")
    print(f"  Downloaded: {counters.downloaded}")
    print(f"  Skipped: {counters.skipped_existing}")
    print(f"  Failed: {counters.failed}")
    return counters
