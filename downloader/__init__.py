"""Download utilities for ExoDense datasets.

This module provides parallel download functions for external datasets:
- TU Delft thermosphere data (GRACE, GRACE-FO, CHAMP, SWARM, GOCE)
- Space weather indices (solar flux, geomagnetic activity)
- CO2 measurements (NOAA global and Mauna Loa)

Each download function maintains a manifest file that tracks the status
of all files (downloaded, skipped, or failed) with timestamps and URLs.
Manifests are saved in the respective data directories as `manifest.json`.

Downloads run in parallel using multiple threads for faster operation.
Use `max_workers` to control the number of concurrent downloads.

Example:
    from downloader import download_tudelft, download_space_weather, download_co2
    from downloader.counter import Counters

    # Download TU Delft data with 4 parallel workers (default)
    counters = download_tudelft(
        missions=["grace", "grace_fo"],
        start_ym=(2020, 1),
        end_ym=(2020, 12),
        overwrite=False,
        resume=True,
        max_workers=4,
    )

    # Download space weather data with 2 workers
    counters = download_space_weather(overwrite=False, resume=True, max_workers=2)

    # Download CO2 data
    counters = download_co2(overwrite=False, resume=True, max_workers=2)

    # Access manifest to check download history
    from downloader.manifest import Manifest
    manifest = Manifest.load("data/original/co2/manifest.json")
    for entry in manifest.entries:
        print(f"{entry.path}: {entry.status} at {entry.timestamp}")
"""

from .counter import Counters
from .manifest import Manifest, ManifestEntry
from .tudelft import download_tudelft
from .space_weather import download_space_weather
from .co2 import download_co2

__all__ = [
    "Counters",
    "Manifest",
    "ManifestEntry",
    "download_tudelft",
    "download_space_weather",
    "download_co2",
]
