from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from .common import (
    curl_text,
    download_parallel,
    get_base_dir,
    in_ym_range,
    parse_apache_index_filenames,
)
from .counter import Counters
from .manifest import ManifestEntry, create_or_update_manifest

BASE_DIR = get_base_dir()
REF_ROOT = Path("data") / "original"
MANIFEST_PATH = REF_ROOT / "tudelft" / "manifest.json"

TUDELFT_MISSIONS: dict[str, dict[str, object]] = {
    "grace": {
        "index_urls": (
            "https://thermosphere.tudelft.nl/data/data/version_02/GRACE_data/",
        ),
        "dest_subdir": "grace",
        "prefixes": ("GA_DNS_ACC_", "GB_DNS_ACC_"),
        "fallback_dns_zip": False,
    },
    "grace_fo": {
        "index_urls": (
            "https://thermosphere.tudelft.nl/data/data/version_02/GRACE-FO_data/",
        ),
        "dest_subdir": "grace_fo",
        "prefixes": ("GC_DNS_ACC_", "GD_DNS_ACC_"),
        "fallback_dns_zip": False,
    },
    "champ": {
        "index_urls": (
            "https://thermosphere.tudelft.nl/data/data/version_02/CHAMP_data/",
        ),
        "dest_subdir": "champ",
        "prefixes": ("CH_DNS_ACC_", "CHAMP_DNS_ACC_"),
        "fallback_dns_zip": True,
    },
    "swarm": {
        "index_urls": (
            "https://thermosphere.tudelft.nl/data/data/version_02/SWARM_data/",
            "https://thermosphere.tudelft.nl/data/data/version_02/Swarm_data/",
            "https://thermosphere.tudelft.nl/data/data/version_02/swarm_data/",
            "https://thermosphere.tudelft.nl/data/data/version_02/SWARM/",
        ),
        "dest_subdir": "swarm",
        "prefixes": ("SWA_DNS_ACC_", "SWB_DNS_ACC_", "SWC_DNS_ACC_", "SW_DNS_ACC_","SA_DNS_POD_","SB_DNS_POD_","SC_DNS_POD_"),
        "fallback_dns_zip": True,
    },
    "goce": {
        "index_urls": (
            "https://thermosphere.tudelft.nl/data/data/version_02/GOCE_data/",
            "https://thermosphere.tudelft.nl/data/data/version_02/Goce_data/",
            "https://thermosphere.tudelft.nl/data/data/version_02/goce_data/",
            "https://thermosphere.tudelft.nl/data/data/version_02/GOCE/",
        ),
        "dest_subdir": "goce",
        "prefixes": ("GO_DNS_ACC_", "GOCE_DNS_ACC_"),
        "fallback_dns_zip": True,
    },
}


def _select_tudelft_files(
    names: list[str],
    prefixes: tuple[str, ...],
    start_ym: tuple[int, int] | None,
    end_ym: tuple[int, int] | None,
    fallback_dns_zip: bool,
) -> list[str]:
    selected = [
        n
        for n in names
        if n.endswith(".zip")
        and any(n.startswith(prefix) for prefix in prefixes)
        and in_ym_range(n, start_ym, end_ym)
    ]
    if selected or not fallback_dns_zip:
        return selected
    # Some TU mission folders use different filename roots. Fallback to DNS zip files.
    return [
        n
        for n in names
        if n.endswith(".zip")
        and "DNS" in n.upper()
        and in_ym_range(n, start_ym, end_ym)
    ]


def download_tudelft(
    missions: list[str] | None = None,
    start_ym: tuple[int, int] | None = None,
    end_ym: tuple[int, int] | None = None,
    *,
    overwrite: bool = False,
    resume: bool = True,
    max_workers: int = 4,
) -> Counters:
    """Download TU Delft thermosphere data for specified missions.

    Downloads accelerometer data from the TU Delft thermosphere repository
    for missions including GRACE, GRACE-FO, CHAMP, SWARM, and GOCE.

    Args:
        missions: List of mission names to download. If None, downloads all missions.
            Available: "grace", "grace_fo", "champ", "swarm", "goce".
        start_ym: Optional (year, month) tuple to filter files starting from this date.
        end_ym: Optional (year, month) tuple to filter files up to this date.
        overwrite: If True, re-download existing files. If False, skip existing files.
        resume: If True, resume partial downloads using curl's continue feature.
        max_workers: Maximum number of concurrent download threads (default: 4).

    Returns:
        Counters object with downloaded, skipped_existing, and failed counts.

    Example:
        >>> counters = download_tudelft(
        ...     missions=["grace", "swarm"],
        ...     start_ym=(2020, 1),
        ...     end_ym=(2020, 12),
        ...     overwrite=False,
        ...     resume=True,
        ...     max_workers=4,
        ... )
        >>> print(f"Downloaded: {counters.downloaded}")
    """
    root = REF_ROOT / "tudelft"
    root.mkdir(parents=True, exist_ok=True)

    # Collect all download tasks across all missions
    all_downloads: list[tuple[str, Path]] = []
    all_entries: list[ManifestEntry] = []

    if missions is None:
        missions = list(TUDELFT_MISSIONS.keys())

    for mission in tqdm(missions, desc="Processing missions"):
        cfg = TUDELFT_MISSIONS.get(mission)
        if cfg is None:
            print(f"  FAILED: unknown mission '{mission}'")
            # Create failed entry for tracking
            all_entries.append(
                ManifestEntry(
                    path=f"tudelft/{mission}",
                    url="",
                    status="failed",
                    error=f"Unknown mission: {mission}",
                )
            )
            continue

        index_urls = cfg.get("index_urls")
        prefixes = cfg.get("prefixes")
        dest_subdir = cfg.get("dest_subdir")
        fallback_dns_zip = cfg.get("fallback_dns_zip", False)

        if (
            not isinstance(index_urls, tuple)
            or not isinstance(prefixes, tuple)
            or not isinstance(dest_subdir, str)
            or not isinstance(fallback_dns_zip, bool)
        ):
            print(f"  FAILED: invalid TU Delft config for mission {mission}")
            continue

        print(f"TU Delft: {mission}")
        html = ""
        used_index_url = ""
        errors: list[str] = []

        for idx_url in index_urls:
            try:
                html = curl_text(idx_url, retries=2, retry_delay=2, timeout_s=60)
                used_index_url = idx_url
                break
            except Exception as exc:
                errors.append(f"{idx_url} -> {exc}")

        if not html:
            print(f"  FAILED listing mission {mission}. Tried:")
            for line in errors:
                print(f"    - {line}")
            continue

        names = parse_apache_index_filenames(html)
        names = _select_tudelft_files(
            names=names,
            prefixes=prefixes,
            start_ym=start_ym,
            end_ym=end_ym,
            fallback_dns_zip=fallback_dns_zip,
        )

        dest = root / dest_subdir
        dest.mkdir(parents=True, exist_ok=True)
        print(f"  Files selected: {len(names)}")

        # Queue downloads for this mission
        for name in names:
            url = used_index_url + name
            out_path = dest / name
            all_downloads.append((url, out_path))

    # Download all files in parallel
    if all_downloads:
        entries, counters = download_parallel(
            all_downloads,
            REF_ROOT,
            overwrite=overwrite,
            resume=resume,
            max_workers=max_workers,
            timeout_s=60,
            desc=f"Downloading {len(all_downloads)} TU Delft files",
        )
        all_entries.extend(entries)
    else:
        counters = Counters()

    # Save manifest with all entries
    manifest = create_or_update_manifest(
        dataset="tudelft",
        manifest_path=MANIFEST_PATH,
        entries=all_entries,
    )
    print(f"Manifest saved: {MANIFEST_PATH}")
    print(f"  Total tracked files: {len(manifest.entries)}")
    print(f"  Downloaded: {counters.downloaded}")
    print(f"  Skipped: {counters.skipped_existing}")
    print(f"  Failed: {counters.failed}")

    return counters


# Backwards compatibility alias
sync_tudelft = download_tudelft
