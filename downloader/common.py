from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tqdm import tqdm

from .counter import Counters
from .manifest import ManifestEntry, create_or_update_manifest, ensure_dir, now_iso


@dataclass(frozen=True)
class DownloadResult:
    ok: bool
    path: Path
    url: str
    note: str = ""


def get_base_dir() -> Path:
    """Resolve repository root with optional environment override."""
    override = os.environ.get("EXODENSE_BASE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def run_cmd(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def curl_text(
    url: str,
    *,
    retries: int = 3,
    retry_delay: int = 2,
    timeout_s: int | None = None,
) -> str:
    cmd = [
        "curl",
        "-L",
        "-f",
        "--retry",
        str(retries),
        "--retry-delay",
        str(retry_delay),
    ]
    if timeout_s is not None:
        cmd.extend(["--connect-timeout", "20", "--max-time", str(timeout_s)])
    cmd.append(url)
    rc, out, err = run_cmd(cmd)
    if rc != 0:
        raise RuntimeError(f"curl failed for {url}: {err.strip()[:240]}")
    return out


def curl_download(
    url: str,
    out_path: Path,
    *,
    overwrite: bool,
    counters: Counters | None = None,
    auth: dict | None = None,
    retries: int = 5,
    retry_delay: int = 2,
    timeout_s: int | None = None,
    resume: bool = True,
) -> Literal["downloaded", "skipped", "failed"]:
    """Download a file with curl, tracking the result.

    Returns:
        "downloaded" if the file was successfully downloaded,
        "skipped" if the file already exists and was not overwritten,
        "failed" if the download failed.
    """
    ensure_dir(out_path.parent)
    if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
        if counters:
            counters.increment("skipped_existing")
        return "skipped"

    cmd = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        str(retries),
        "--retry-delay",
        str(retry_delay),
    ]
    if resume:
        cmd.extend(["-C", "-"])
    if timeout_s is not None:
        cmd.extend(["--connect-timeout", "20", "--max-time", str(timeout_s)])
    if auth:
        if auth.get("user") and auth.get("password"):
            cmd.extend(["-u", f"{auth['user']}:{auth['password']}"])
        if auth.get("bearer"):
            cmd.extend(["-H", f"Authorization: Bearer {auth['bearer']}"])
    cmd.extend(["-o", str(out_path), url])

    rc, _out, err = run_cmd(cmd)
    if rc == 0 and out_path.exists() and out_path.stat().st_size > 0:
        if counters:
            counters.increment("downloaded")
        return "downloaded"

    if counters:
        counters.increment("failed")
    # Don't print here to avoid interleaved output in parallel mode
    # Caller can handle error reporting
    try:
        if out_path.exists() and out_path.stat().st_size == 0:
            out_path.unlink()
    except OSError:
        pass
    return "failed"


def download_single(
    url: str,
    out_path: Path,
    base_dir: Path,
    *,
    overwrite: bool = False,
    resume: bool = True,
    retries: int = 5,
    retry_delay: int = 2,
    timeout_s: int | None = None,
) -> ManifestEntry:
    """Download a single file and return a manifest entry.

    This function is designed to be used with ThreadPoolExecutor for parallel downloads.
    """
    ensure_dir(out_path.parent)

    status: Literal["downloaded", "skipped", "failed"]
    error = None

    if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
        status = "skipped"
    else:
        cmd = [
            "curl",
            "-L",
            "--fail",
            "--retry",
            str(retries),
            "--retry-delay",
            str(retry_delay),
            "-s",  # Silent mode for parallel downloads
        ]
        if resume:
            cmd.extend(["-C", "-"])
        if timeout_s is not None:
            cmd.extend(["--connect-timeout", "20", "--max-time", str(timeout_s)])
        cmd.extend(["-o", str(out_path), url])

        rc, _out, err = run_cmd(cmd)
        if rc == 0 and out_path.exists() and out_path.stat().st_size > 0:
            status = "downloaded"
        else:
            status = "failed"
            error = err.strip()[:200] if err else None
            try:
                if out_path.exists() and out_path.stat().st_size == 0:
                    out_path.unlink()
            except OSError:
                pass

    # Build manifest entry
    try:
        rel_path = str(out_path.relative_to(base_dir))
    except ValueError:
        rel_path = str(out_path)

    size = None
    if out_path.exists():
        try:
            size = out_path.stat().st_size
        except OSError:
            pass

    return ManifestEntry(
        path=rel_path,
        url=url,
        status=status,
        timestamp=now_iso(),
        size_bytes=size,
        error=error,
    )


def download_parallel(
    downloads: list[tuple[str, Path]],
    base_dir: Path,
    *,
    overwrite: bool = False,
    resume: bool = True,
    max_workers: int = 4,
    retries: int = 5,
    retry_delay: int = 2,
    timeout_s: int | None = None,
    desc: str = "Downloading",
) -> tuple[list[ManifestEntry], Counters]:
    """Download multiple files in parallel using ThreadPoolExecutor.

    Args:
        downloads: List of (url, out_path) tuples
        base_dir: Base directory for computing relative paths in manifest
        overwrite: If True, re-download existing files
        resume: If True, resume partial downloads
        max_workers: Maximum number of concurrent download threads
        retries: Number of retries for each download
        retry_delay: Delay between retries in seconds
        timeout_s: Timeout in seconds, None for no timeout
        desc: Description for progress bar

    Returns:
        Tuple of (list of manifest entries, counters)
    """
    counters = Counters()
    entries: list[ManifestEntry | None] = [None] * len(downloads)

    if not downloads:
        return [], counters

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {}
        for idx, (url, out_path) in enumerate(downloads):
            future = executor.submit(
                download_single,
                url,
                out_path,
                base_dir,
                overwrite=overwrite,
                resume=resume,
                retries=retries,
                retry_delay=retry_delay,
                timeout_s=timeout_s,
            )
            future_to_idx[future] = idx

        # Collect results with progress bar
        for future in tqdm(as_completed(future_to_idx), total=len(downloads), desc=desc):
            idx = future_to_idx[future]
            try:
                entry = future.result()
                entries[idx] = entry
                counters.increment(entry.status)
                if entry.status == "failed" and entry.error:
                    print(f"  FAILED: {entry.url} -> {entry.path} ({entry.error})")
            except Exception as exc:
                # Should not happen, but handle gracefully
                url, out_path = downloads[idx]
                counters.increment("failed")
                print(f"  FAILED: {url} -> {out_path} ({exc})")
                entries[idx] = ManifestEntry(
                    path=str(out_path.relative_to(base_dir)) if out_path.is_relative_to(base_dir) else str(out_path),
                    url=url,
                    status="failed",
                    timestamp=now_iso(),
                    error=str(exc)[:200],
                )

    return [e for e in entries if e is not None], counters


def list_ftp_dir(url: str) -> list[str]:
    rc, out, err = run_cmd(["curl", "--list-only", url])
    if rc != 0:
        raise RuntimeError(f"Unable to list FTP directory {url}: {err.strip()[:240]}")
    return [x.strip() for x in out.splitlines() if x.strip()]


def parse_apache_index_filenames(html: str) -> list[str]:
    names = re.findall(r'href="([^"]+)"', html)
    files: list[str] = []
    for name in names:
        if name in ("../", "./") or name.endswith("/"):
            continue
        files.append(name)
    return sorted(set(files))


def parse_year_month_token(name: str) -> tuple[int, int] | None:
    m = re.search(r"_(\d{4})_(\d{2})_", name)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"_(\d{4})-(\d{2})\.", name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def in_ym_range(
    name: str, start_ym: tuple[int, int] | None, end_ym: tuple[int, int] | None
) -> bool:
    ym = parse_year_month_token(name)
    if ym is None:
        return True
    if start_ym and ym < start_ym:
        return False
    if end_ym and ym > end_ym:
        return False
    return True


def parse_ym_arg(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    m = re.fullmatch(r"(\d{4})-(\d{2})", value.strip())
    if not m:
        raise ValueError(f"Invalid YYYY-MM value: {value}")
    year, month = int(m.group(1)), int(m.group(2))
    if month < 1 or month > 12:
        raise ValueError(f"Invalid month in YYYY-MM value: {value}")
    return year, month


def safe_name(url: str) -> str:
    tail = url.rstrip("/").split("/")[-1]
    return tail or "download.bin"


def summarize_results(results: Iterable[DownloadResult]) -> dict:
    items = [
        {
            "ok": r.ok,
            "path": str(r.path),
            "url": r.url,
            "note": r.note,
        }
        for r in results
    ]
    ok = sum(1 for x in items if x["ok"])
    fail = len(items) - ok
    return {"ok": ok, "failed": fail, "items": items}
