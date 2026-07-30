from dataclasses import dataclass, field
from threading import Lock


@dataclass
class Counters:
    """Thread-safe counters for the download process."""

    downloaded: int = 0
    skipped_existing: int = 0
    failed: int = 0
    _lock: Lock = field(default_factory=Lock, repr=False)

    def increment(self, field: str) -> None:
        """Thread-safe increment of a counter field."""
        with self._lock:
            if field == "downloaded":
                self.downloaded += 1
            elif field == "skipped_existing":
                self.skipped_existing += 1
            elif field == "failed":
                self.failed += 1

    @property
    def total(self) -> int:
        """Total number of files processed."""
        with self._lock:
            return self.downloaded + self.skipped_existing + self.failed
