from dataclasses import dataclass


@dataclass
class Counters:
    """Counters for the download process."""

    downloaded: int = 0
    skipped_existing: int = 0
    failed: int = 0
