from __future__ import annotations

import os
import time
from pathlib import Path

from .common import get_base_dir, run_cmd
from .counter import Counters
from .manifest import ManifestEntry, create_or_update_manifest, ensure_dir, now_iso

BASE_DIR = get_base_dir()
REF_ROOT = Path("data") / "original"
DEST_DIR = REF_ROOT / "hasdm"
MANIFEST_PATH = DEST_DIR / "manifest.json"
BASE_URL = "https://sol.spacenvironment.net/Hasdm_database/download"
DEFAULT_START_YM = (2000, 1)
DEFAULT_END_YM = (2025, 12)
DEFAULT_DELAY_S = 65
REQUEST_HEADERS = {
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,"
        "image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7"
    ),
    "Accept-Language": "en-US,en;q=0.9,es;q=0.8",
    "Referer": "https://sol.spacenvironment.net/Hasdm_database/",
    "Upgrade-Insecure-Requests": "1",
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"
    ),
}


def _iter_months(
    start_ym: tuple[int, int],
    end_ym: tuple[int, int],
) -> list[tuple[int, int]]:
    start_year, start_month = start_ym
    end_year, end_month = end_ym
    if start_month < 1 or start_month > 12:
        raise ValueError(f"Invalid start month: {start_month}")
    if end_month < 1 or end_month > 12:
        raise ValueError(f"Invalid end month: {end_month}")
    if start_ym > end_ym:
        raise ValueError(f"start_ym must be <= end_ym, got {start_ym} > {end_ym}")

    months: list[tuple[int, int]] = []
    year, month = start_year, start_month
    while (year, month) <= end_ym:
        months.append((year, month))
        month += 1
        if month == 13:
            year += 1
            month = 1
    return months


def _cookie_settings() -> tuple[str | None, Path | None]:
    cookie = os.environ.get("HASDM_COOKIE")
    cookie_file_value = os.environ.get("HASDM_COOKIE_FILE")
    cookie_file = Path(cookie_file_value) if cookie_file_value else None
    return cookie, cookie_file


def _manifest_path_for(out_path: Path) -> str:
    try:
        return str(out_path.relative_to(REF_ROOT))
    except ValueError:
        return str(out_path)


def _download_month(
    url: str,
    out_path: Path,
    *,
    overwrite: bool,
    resume: bool,
    cookie: str | None,
    cookie_file: Path | None,
) -> tuple[str, str | None]:
    ensure_dir(out_path.parent)

    if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
        return "skipped", None

    cmd = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        "2",
        "--retry-delay",
        "5",
        "--connect-timeout",
        "20",
        "--max-time",
        "600",
        "-sS",
        "-w",
        "__CURL_META__%{http_code} %{url_effective} %{content_type}",
    ]
    if resume:
        cmd.extend(["-C", "-"])
    for key, value in REQUEST_HEADERS.items():
        cmd.extend(["-H", f"{key}: {value}"])
    if cookie:
        cmd.extend(["-H", f"Cookie: {cookie}"])
    if cookie_file:
        cmd.extend(["-b", str(cookie_file)])
    cmd.extend(["-o", str(out_path), url])

    rc, out, err = run_cmd(cmd)
    meta = out.strip().split("__CURL_META__")[-1].strip() if out.strip() else ""
    parts = meta.split(maxsplit=2)
    http_code = parts[0] if len(parts) > 0 else ""
    effective_url = parts[1] if len(parts) > 1 else ""
    content_type = parts[2] if len(parts) > 2 else ""

    if rc != 0:
        error = err.strip()[:200] if err else f"curl failed with exit code {rc}"
        try:
            if out_path.exists() and out_path.stat().st_size == 0:
                out_path.unlink()
        except OSError:
            pass
        return "failed", error

    invalid_response = (
        "login.spacenvironment.net" in effective_url
        or content_type.lower().startswith("text/html")
        or http_code != "200"
    )
    if invalid_response:
        try:
            if out_path.exists():
                out_path.unlink()
        except OSError:
            pass
        error = (
            f"Unexpected response: http_code={http_code or 'unknown'}, "
            f"effective_url={effective_url or 'unknown'}, "
            f"content_type={content_type or 'unknown'}"
        )
        return "failed", error

    if out_path.exists() and out_path.stat().st_size > 0:
        return "downloaded", None

    return "failed", "curl completed without creating a non-empty file"


def download_hasdm(
    *,
    overwrite: bool = False,
    resume: bool = True,
    start_ym: tuple[int, int] = DEFAULT_START_YM,
    end_ym: tuple[int, int] = DEFAULT_END_YM,
    delay_s: int = DEFAULT_DELAY_S,
) -> Counters:
    """Download monthly HASDM archive files.

    The upstream archive currently redirects through `login.spacenvironment.net`,
    so authentication must be provided with one of:
    - `HASDM_COOKIE`: raw Cookie header value for an authenticated session
    - `HASDM_COOKIE_FILE`: curl-compatible cookie jar path

    Files are requested sequentially because the source is rate-limited to roughly
    one file per minute. A 65-second pause is applied between actual download
    attempts; skipped existing files do not incur a delay.
    """
    if delay_s < 0:
        raise ValueError(f"delay_s must be >= 0, got {delay_s}")

    ensure_dir(DEST_DIR)
    cookie, cookie_file = _cookie_settings()

    months = _iter_months(start_ym, end_ym)
    counters = Counters()
    entries: list[ManifestEntry] = []

    for idx, (year, month) in enumerate(months):
        url = f"{BASE_URL}/{year:04d}/{month:02d}"
        out_path = DEST_DIR / f"hasdm_{year:04d}_{month:02d}"

        status, error = _download_month(
            url,
            out_path,
            overwrite=overwrite,
            resume=resume,
            cookie=cookie,
            cookie_file=cookie_file,
        )
        counters.increment(
            "skipped_existing" if status == "skipped" else status
        )

        size = None
        if out_path.exists():
            try:
                size = out_path.stat().st_size
            except OSError:
                size = None

        if status == "failed" and not cookie and not cookie_file:
            auth_error = (
                "The HASDM archive currently requires an authenticated session; "
                "set HASDM_COOKIE or HASDM_COOKIE_FILE."
            )
            error = f"{auth_error} {error}" if error else auth_error

        entries.append(
            ManifestEntry(
                path=_manifest_path_for(out_path),
                url=url,
                status=status,
                timestamp=now_iso(),
                size_bytes=size,
                error=error,
            )
        )

        if status != "skipped":
            should_sleep = idx < len(months) - 1
            if should_sleep and delay_s > 0:
                print(
                    f"HASDM {year:04d}-{month:02d}: {status}. "
                    f"Sleeping {delay_s}s for rate limit."
                )
                time.sleep(delay_s)
        else:
            print(f"HASDM {year:04d}-{month:02d}: skipped existing file.")

    manifest = create_or_update_manifest(
        dataset="hasdm",
        manifest_path=MANIFEST_PATH,
        entries=entries,
    )
    print(f"Manifest saved: {MANIFEST_PATH}")
    print(f"  Total tracked files: {len(manifest.entries)}")
    print(f"  Downloaded: {counters.downloaded}")
    print(f"  Skipped: {counters.skipped_existing}")
    print(f"  Failed: {counters.failed}")

    return counters


sync_hasdm = download_hasdm
