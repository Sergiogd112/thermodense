from __future__ import annotations

from .common import curl_download, ensure_dir, get_base_dir
from .counter import Counters

BASE_DIR = get_base_dir()
DEST_DIR = BASE_DIR / "data" / "original" / "space_weather"

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
) -> Counters:
    """Download space weather indices for density modeling.

    Downloads solar flux and geomagnetic activity data from Celestrak,
    Canadian Space Weather, and NASA SPDF.

    Args:
        overwrite: If True, re-download existing files. If False, skip existing files.
        resume: If True, resume partial downloads using curl's continue feature.

    Returns:
        Counters object with downloaded, skipped_existing, and failed counts.

    Example:
        >>> counters = download_space_weather(overwrite=False, resume=True)
        >>> print(f"Downloaded: {counters.downloaded}")
    """
    counters = Counters()
    ensure_dir(DEST_DIR)

    for name, url in SOURCES.items():
        curl_download(
            url,
            DEST_DIR / name,
            overwrite=overwrite,
            resume=resume,
            timeout_s=180,
            counters=counters,
        )

    return counters


# Backwards compatibility alias
sync_space_weather = download_space_weather
