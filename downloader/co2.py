from __future__ import annotations

from .common import curl_download, ensure_dir, get_base_dir
from .counter import Counters

BASE_DIR = get_base_dir()
DEST_DIR = BASE_DIR / "data" / "original" / "co2"
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
) -> Counters:
    """Download NOAA CO2 datasets.

    Downloads Mauna Loa and global CO2 measurements from NOAA GML,
    including monthly, annual, weekly, and daily data.

    Args:
        overwrite: If True, re-download existing files. If False, skip existing files.
        resume: If True, resume partial downloads using curl's continue feature.

    Returns:
        Counters object with downloaded, skipped_existing, and failed counts.

    Example:
        >>> counters = download_co2(overwrite=False, resume=True)
        >>> print(f"Downloaded: {counters.downloaded}")
    """
    counters = Counters()
    ensure_dir(DEST_DIR)

    for src, dst in FILES.items():
        url = f"{BASE_URL}/{src}"
        out = DEST_DIR / dst
        curl_download(
            url,
            out,
            overwrite=overwrite,
            resume=resume,
            retries=4,
            retry_delay=2,
            timeout_s=120,
            counters=counters,
        )

    return counters


# Backwards compatibility alias
sync_co2 = download_co2
