"""Download utilities for ExoDense datasets.

This module provides functions to download external datasets:
- TU Delft thermosphere data (GRACE, GRACE-FO, CHAMP, SWARM, GOCE)
- Space weather indices (solar flux, geomagnetic activity)
- CO2 measurements (NOAA global and Mauna Loa)

Example:
    from downloader import download_tudelft, download_space_weather, download_co2
    from downloader.counter import Counters

    # Download TU Delft data for specific missions and date range
    counters = download_tudelft(
        missions=["grace", "grace_fo"],
        start_ym=(2020, 1),
        end_ym=(2020, 12),
        overwrite=False,
        resume=True,
    )

    # Download space weather data
    counters = download_space_weather(overwrite=False, resume=True)

    # Download CO2 data
    counters = download_co2(overwrite=False, resume=True)
"""

from .counter import Counters
from .tudelft import download_tudelft
from .space_weather import download_space_weather
from .co2 import download_co2

__all__ = [
    "Counters",
    "download_tudelft",
    "download_space_weather",
    "download_co2",
]
