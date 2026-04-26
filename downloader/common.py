from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .counter import Counters


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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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
) -> bool | None:
    ensure_dir(out_path.parent)
    if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
        if counters:
            counters.skipped_existing += 1
        return False

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
            counters.downloaded += 1
        return True

    if counters:
        counters.failed += 1
    print(f"  FAILED: {url} -> {out_path} ({err.strip()[:200]})")
    try:
        if out_path.exists() and out_path.stat().st_size == 0:
            out_path.unlink()
    except OSError:
        pass
    return None


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
