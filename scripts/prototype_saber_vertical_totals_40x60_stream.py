"""THROWAWAY: exact, resumable 40x60-degree SABER vertical-total stream.

Run from the repository root (temporary storage remains bounded by --workers):
    uv run python scripts/prototype_saber_vertical_totals_40x60_stream.py --resume

This is scratch feasibility work for the isolated PCMCI pilot.  It does not
change production SABER inputs or decoder outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock, Timeout
from scipy.io import netcdf_file
from thermodense.downloader.common import curl_text, parse_apache_index_filenames
from thermodense.downloader.saber import L2A_BASE_URL, L2A_FILENAME_RE
from thermodense.saber import (
    CO2_FILE_RE,
    MISSING_VALUE_LIMIT,
    NO_FILE_RE,
    _coverage,
    hasdm_longitudes_by_timestamp,
)

# Deliberately local prototype defaults: neither path is a production artifact.
CO2_DIR = Path("data/original/saber/co2_cooling_profiles")
NO_DIR = Path("data/original/saber/no_cooling_profiles")
L2A_DIR = Path("data/original/saber/level2a")
SAMPLES = Path(
    "outputs/figures/results/hasdm_msis_model_errors/data/"
    "hasdm_msis_errors_nearest_timestamp_grid_samples.parquet"
)
BASELINE = Path("data/decoded/saber/saber_hasdm_maunaloa_3hour.parquet")
DEFAULT_OUTPUT = Path("outputs/prototypes/saber_vertical_totals_40x60_stream")
DEFAULT_TEMP = Path("/tmp/thermodense-saber-vertical-totals-40x60")
START, END = date(2002, 1, 25), date(2025, 7, 20)
LATITUDE_CENTER, CELL_WIDTHS = 20.0, (40.0, 60.0)
ALTITUDE_BOUNDS_KM, MIN_ALTITUDE_SPAN_KM = (100.0, 140.0), 38.0
CHANNELS = ("CO2cool", "NOcool", "O2_1delta_ver", "OH_16_ver", "OH_20_ver")
PROXIES = CHANNELS[2:]
L2A_RE = re.compile(r"SABER_L2A_(\d{4})(\d{3})_(\d+)_(02\.\d+)\.nc")
FALLBACK_FILE_BYTES = int(252 * 2**30 / 42_127)
CHECKPOINT_SCHEMA = 3


def circular_delta(longitude: np.ndarray, center: np.ndarray) -> np.ndarray:
    return np.abs((longitude - center + 180.0) % 360.0 - 180.0)


def orient(values: np.ndarray, scans: int) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError(f"expected scan/altitude matrix, got {values.shape}")
    if values.shape[0] == scans:
        return values
    if values.shape[1] == scans:
        return values.T
    raise ValueError(f"cannot identify scan dimension in {values.shape}")


def file_day(ds) -> date:
    if "date" in ds.variables:
        token = str(int(np.asarray(ds.variables["date"].data).flat[0]))
        return date(int(token[:4]), 1, 1) + timedelta(days=int(token[4:]) - 1)
    return date(int(np.asarray(ds.variables["year"].data).item()), 1, 1) + timedelta(
        days=int(np.asarray(ds.variables["day"].data).item()) - 1
    )


def slots_for(day: date, milliseconds: np.ndarray) -> np.ndarray:
    return (
        (np.datetime64(day.isoformat(), "ms") + milliseconds.astype("timedelta64[ms]"))
        .astype("datetime64[h]")
        .astype("datetime64[3h]")
    )


def integrate_profiles(
    altitudes_km: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use only wholly finite native 100--140 km samples; never fill endpoints."""
    totals = np.full(values.shape[0], np.nan)
    spans = np.full(values.shape[0], np.nan)
    accepted = np.zeros(values.shape[0], dtype=bool)
    for scan in range(values.shape[0]):
        altitude = altitudes_km if altitudes_km.ndim == 1 else altitudes_km[scan]
        selected = np.isfinite(altitude) & (altitude >= 100.0) & (altitude <= 140.0)
        if selected.sum() < 2:
            continue
        z, y = altitude[selected], values[scan, selected]
        if not np.all(np.isfinite(y) & (y > MISSING_VALUE_LIMIT)):
            continue
        order = np.argsort(z)
        z, y = z[order], y[order]
        spans[scan] = z[-1] - z[0]
        if spans[scan] < MIN_ALTITUDE_SPAN_KM:
            continue
        totals[scan] = np.trapezoid(y, x=z * 1000.0)
        accepted[scan] = True
    return totals, spans, accepted


def native_120_indices(altitudes_km: np.ndarray, scans: int) -> np.ndarray:
    distances = np.where(
        np.isfinite(altitudes_km), np.abs(altitudes_km - 120.0), np.inf
    )
    if altitudes_km.ndim == 1:
        return np.full(scans, int(np.argmin(distances)))
    return np.argmin(distances, axis=1)


def add_profiles(
    day: date,
    values: np.ndarray,
    altitudes: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    milliseconds: np.ndarray,
    orbits: np.ndarray,
    centers: np.ndarray,
    index_by_slot: dict[np.datetime64, int],
    sums: np.ndarray,
    counts: np.ndarray,
    diagnostics: dict[str, int],
    accepted_spans: list[float],
    required: set[tuple[date, int]] | None,
) -> None:
    """Apply the expanded-cell prototype's native-nearest-120-km cell rule."""
    scans = values.shape[0]
    totals, spans, integrated = integrate_profiles(altitudes, values)
    level, rows = native_120_indices(altitudes, scans), np.arange(scans)
    lat120, lon120, time120 = (
        latitude[rows, level],
        longitude[rows, level],
        milliseconds[rows, level],
    )
    positions = np.array(
        [index_by_slot.get(value, -1) for value in slots_for(day, time120)]
    )
    center = np.where(positions >= 0, centers[np.maximum(positions, 0)], np.nan)
    keep = (
        integrated
        & np.isfinite(lat120)
        & np.isfinite(lon120)
        & np.isfinite(time120)
        & (positions >= 0)
        & (np.abs(lat120 - LATITUDE_CENTER) <= CELL_WIDTHS[0] / 2)
        & (circular_delta(lon120, center) <= CELL_WIDTHS[1] / 2)
    )
    diagnostics["profiles_seen"] += scans
    diagnostics["profiles_rejected_missing_or_insufficient_levels"] += int(
        np.sum(~np.isfinite(spans))
    )
    diagnostics["profiles_rejected_span_below_38km"] += int(
        np.sum(np.isfinite(spans) & ~integrated)
    )
    diagnostics["integrated_profiles_before_spatial_match"] += int(integrated.sum())
    diagnostics["integrated_profiles_rejected_geolocation_or_cell"] += int(
        np.sum(integrated & ~keep)
    )
    np.add.at(sums, positions[keep], totals[keep])
    np.add.at(counts, positions[keep], 1)
    accepted_spans.extend(spans[keep].tolist())
    if required is not None:
        required.update((day, int(orbit)) for orbit in orbits[keep] if orbit >= 0)


def add_file(
    path: Path,
    channels: Iterable[str],
    centers: np.ndarray,
    index_by_slot: dict[np.datetime64, int],
    sums: dict[str, np.ndarray],
    counts: dict[str, np.ndarray],
    diagnostics: dict[str, dict[str, int]],
    spans: dict[str, list[float]],
    required: set[tuple[date, int]] | None,
) -> None:
    """Decode one direct file or one Level2A file, once per proxy product set."""
    channels = tuple(channels)
    with netcdf_file(path, "r", mmap=False) as ds:
        day = file_day(ds)
        first = np.asarray(ds.variables[channels[0]].data, dtype=float)
        scans = first.shape[0]
        altitude_name = "altitude" if "altitude" in ds.variables else "tpaltitude"
        altitudes = np.asarray(ds.variables[altitude_name].data, dtype=float)
        if altitudes.ndim == 2:
            altitudes = orient(altitudes, scans)
        latitude = orient(
            np.asarray(ds.variables["tplatitude"].data, dtype=float), scans
        )
        longitude = (
            orient(np.asarray(ds.variables["tplongitude"].data, dtype=float), scans)
            % 360
        )
        milliseconds = orient(np.asarray(ds.variables["time"].data, dtype=float), scans)
        orbits = (
            np.asarray(ds.variables["orbit"].data, dtype=int)
            if "orbit" in ds.variables
            else np.full(scans, -1)
        )
        values = {
            channel: orient(np.asarray(ds.variables[channel].data, dtype=float), scans)
            * (0.1 if channel in PROXIES else 1.0)
            for channel in channels
        }
    for channel in channels:
        add_profiles(
            day,
            values[channel],
            altitudes,
            latitude,
            longitude,
            milliseconds,
            orbits,
            centers,
            index_by_slot,
            sums[channel],
            counts[channel],
            diagnostics[channel],
            spans[channel],
            required,
        )


def empty_diagnostics() -> dict[str, dict[str, int]]:
    return {
        channel: {
            "profiles_seen": 0,
            "profiles_rejected_missing_or_insufficient_levels": 0,
            "profiles_rejected_span_below_38km": 0,
            "integrated_profiles_before_spatial_match": 0,
            "integrated_profiles_rejected_geolocation_or_cell": 0,
        }
        for channel in CHANNELS
    }


def atomic_npz(path: Path, **arrays: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def save_checkpoint(
    path: Path,
    metadata: dict[str, object],
    sums: dict[str, np.ndarray],
    counts: dict[str, np.ndarray],
    spans: dict[str, list[float]],
    completed: set[str],
    required: set[tuple[date, int]],
    failures: dict[str, dict[str, str]],
    diagnostics: dict[str, dict[str, int]],
) -> None:
    atomic_npz(
        path,
        metadata=np.array([json.dumps(metadata, sort_keys=True)]),
        **{f"sum_{channel}": sums[channel] for channel in CHANNELS},
        **{f"count_{channel}": counts[channel] for channel in CHANNELS},
        **{f"spans_{channel}": np.asarray(spans[channel]) for channel in CHANNELS},
        completed=np.array(sorted(completed)),
        required=np.array(
            sorted(f"{day.isoformat()}:{orbit}" for day, orbit in required)
        ),
        failures=np.array([json.dumps(failures, sort_keys=True)]),
        diagnostics=np.array([json.dumps(diagnostics, sort_keys=True)]),
    )


def load_checkpoint(path: Path, metadata: dict[str, object], slots: int) -> tuple:
    if not path.exists():
        return (
            {channel: np.zeros(slots) for channel in CHANNELS},
            {channel: np.zeros(slots, dtype=np.int64) for channel in CHANNELS},
            {channel: [] for channel in CHANNELS},
            set(),
            set(),
            {},
            empty_diagnostics(),
        )
    with np.load(path, allow_pickle=False) as saved:
        try:
            checkpoint_metadata = json.loads(str(saved["metadata"][0]))
        except (KeyError, IndexError, json.JSONDecodeError) as error:
            raise RuntimeError(
                "Checkpoint schema is incompatible; use --reset."
            ) from error
        if checkpoint_metadata != metadata:
            raise RuntimeError("Checkpoint metadata/config differs; use --reset.")
        sums = {channel: saved[f"sum_{channel}"] for channel in CHANNELS}
        counts = {channel: saved[f"count_{channel}"] for channel in CHANNELS}
        spans = {channel: saved[f"spans_{channel}"].tolist() for channel in CHANNELS}
        completed = set(saved["completed"].tolist())
        required = {
            (date.fromisoformat(token[:10]), int(token[11:]))
            for token in saved["required"].tolist()
        }
        failures = json.loads(str(saved["failures"][0]))
        diagnostics = json.loads(str(saved["diagnostics"][0]))
    if any(len(array) != slots for array in [*sums.values(), *counts.values()]):
        raise RuntimeError(
            "Checkpoint calendar differs from the current baseline calendar."
        )
    return sums, counts, spans, completed, required, failures, diagnostics


def pair_key(pair: tuple[date, int]) -> str:
    return f"l2a:{pair[0].isoformat()}:{pair[1]}"


def artifact_key(filename: str) -> str:
    """Identify a physical orbit file independently of date/orbit aliases."""
    return f"artifact:{filename}"


def local_l2a_index(
    directory: Path,
) -> tuple[dict[tuple[date, int], list[Path]], list[int]]:
    """Index local files once; individual pair lookup must not rescan the archive."""
    result: dict[tuple[date, int], list[Path]] = {}
    sizes = []
    for path in directory.glob("SABER_L2A_*_*.nc"):
        match = L2A_RE.fullmatch(path.name)
        if not match or path.stat().st_size == 0:
            continue
        file_date = date(int(match.group(1)), 1, 1) + timedelta(
            days=int(match.group(2)) - 1
        )
        version = "02.07" if file_date < date(2019, 12, 15) else "02.08"
        if match.group(4) == version:
            result.setdefault((file_date, int(match.group(3))), []).append(path)
            sizes.append(path.stat().st_size)
    return result, sizes


def local_l2a(
    pair: tuple[date, int], index: dict[tuple[date, int], list[Path]]
) -> Path | None:
    """Apply the decoder's same/previous/next-day rule using the one-time index."""
    requested_day, orbit = pair
    candidates = [
        path
        for candidate_day in (
            requested_day - timedelta(days=1),
            requested_day,
            requested_day + timedelta(days=1),
        )
        for path in index.get((candidate_day, orbit), [])
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def listing(day: date, cache: Path) -> dict[int, tuple[str, str]]:
    """Cache one official directory request per UTC day across every requested orbit."""
    path = cache / f"{day.isoformat()}.json"
    if path.exists():
        rows = json.loads(path.read_text())
    else:
        url = f"{L2A_BASE_URL}{day.year:04d}/{day.timetuple().tm_yday:03d}/"
        version = "02.07" if day < date(2019, 12, 15) else "02.08"
        rows = [
            name
            for name in parse_apache_index_filenames(curl_text(url, timeout_s=120))
            if (match := L2A_FILENAME_RE.fullmatch(name)) and match.group(4) == version
        ]
        atomic_json(path, rows)
    url = f"{L2A_BASE_URL}{day.year:04d}/{day.timetuple().tm_yday:03d}/"
    result: dict[int, tuple[str, str]] = {}
    for name in rows:
        match = L2A_FILENAME_RE.fullmatch(name)
        if match:
            orbit = int(match.group(3))
            if orbit in result:
                raise RuntimeError(
                    f"Duplicate official Level2A orbit {orbit} in {day}."
                )
            result[orbit] = (name, f"{url}{name}")
    return result


def resolve_url(pair: tuple[date, int], cache: Path) -> tuple[str, str]:
    """Reuse downloader selection: same day first, then exactly one adjacent result."""
    day, orbit = pair
    same = listing(day, cache).get(orbit)
    if same is not None:
        return same
    matches = [
        found
        for candidate_day in (day - timedelta(days=1), day + timedelta(days=1))
        if (found := listing(candidate_day, cache).get(orbit)) is not None
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one official Level2A file for {pair}; found {len(matches)}."
        )
    return matches[0]


def resolve_urls_parallel(
    pairs: list[tuple[date, int]], cache: Path, workers: int
) -> dict[tuple[date, int], tuple[str, str] | str]:
    """Fetch each relevant daily listing once, concurrently, then resolve pairs."""
    days = sorted(
        {
            requested_day + timedelta(days=offset)
            for requested_day, _orbit in pairs
            for offset in (-1, 0, 1)
        }
    )
    listings: dict[date, dict[int, tuple[str, str]]] = {}
    errors: dict[date, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(listing, day, cache): day for day in days}
        for number, future in enumerate(as_completed(futures), 1):
            day = futures[future]
            try:
                listings[day] = future.result()
            except Exception as error:
                errors[day] = str(error)[:400]
            if number % 500 == 0 or number == len(days):
                print(
                    f"GATS listings: resolved={number - len(errors)}/{number}, "
                    f"daily_errors={len(errors)}, total_days={len(days)}",
                    flush=True,
                )

    resolved: dict[tuple[date, int], tuple[str, str] | str] = {}
    for pair in pairs:
        requested_day, orbit = pair
        if requested_day in errors:
            resolved[pair] = f"same-day listing failed: {errors[requested_day]}"
            continue
        same = listings[requested_day].get(orbit)
        if same is not None:
            resolved[pair] = same
            continue
        adjacent_days = (
            requested_day - timedelta(days=1),
            requested_day + timedelta(days=1),
        )
        failed_days = [day for day in adjacent_days if day in errors]
        if failed_days:
            resolved[pair] = (
                "adjacent-day listing failed: "
                + "; ".join(f"{day}: {errors[day]}" for day in failed_days)
            )[:400]
            continue
        matches = [
            found
            for day in adjacent_days
            if (found := listings[day].get(orbit)) is not None
        ]
        if len(matches) != 1:
            resolved[pair] = (
                f"Expected exactly one official Level2A file for {pair}; "
                f"found {len(matches)}."
            )
        else:
            resolved[pair] = matches[0]
    return resolved


def download(url: str, path: Path) -> str | None:
    """Resume a retained .part safely, then atomically publish a decodable file."""
    partial = path.with_suffix(f"{path.suffix}.part")

    def curl(resume: bool) -> subprocess.CompletedProcess[str]:
        command = [
            "curl",
            "-L",
            "--fail",
            "--retry",
            "3",
            "--retry-delay",
            "2",
            "--connect-timeout",
            "20",
            "--max-time",
            "600",
        ]
        if resume and partial.exists() and partial.stat().st_size:
            command.extend(["-C", "-"])
        command.extend(["-o", str(partial), url])
        return subprocess.run(command, capture_output=True, text=True)

    process = curl(resume=True)
    if process.returncode:
        partial.unlink(missing_ok=True)
        process = curl(resume=False)
    if process.returncode or not partial.is_file() or partial.stat().st_size == 0:
        error = (process.stderr or "curl produced no file").strip()[:400]
        partial.unlink(missing_ok=True)
        return error
    partial.replace(path)
    return None


def print_progress(
    total: int,
    completed: set[str],
    failures: dict[str, dict[str, str]],
    average_bytes: int,
) -> None:
    done = sum(key.startswith("l2a:") for key in completed)
    pending = total - done
    print(
        f"Level2A state: completed={done}/{total}, retryable_failures={len(failures)}, "
        f"remaining_files={pending}, estimated_remaining_bytes={pending * average_bytes:,}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--temp-dir", type=Path, default=DEFAULT_TEMP)
    available_cpus = os.cpu_count() or 1
    max_workers = available_cpus * 4
    default_workers = available_cpus * 2
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        choices=range(1, max_workers + 1),
        metavar=f"1..{max_workers}",
        help=(
            "Concurrent download+decode workers; defaults to two per available "
            f"CPU ({default_workers}) to overlap network waits."
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Durably checkpoint after this many completed/failed source files (default: 10).",
    )
    parser.add_argument(
        "--limit", type=int, help="Process at most N required Level2A pairs."
    )
    parser.add_argument(
        "--daily-limit",
        type=int,
        help="Validation only: inspect at most N local CO2 and NO daily files each.",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume the atomic checkpoint (default)."
    )
    parser.add_argument(
        "--reset", action="store_true", help="Discard this prototype checkpoint first."
    )
    parser.add_argument(
        "--dry-run",
        "--list-only",
        action="store_true",
        help="List the selected pair count only; never resolve, download, or decode Level2A files.",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Do not request GATS listings or download; process only resolvable local files.",
    )
    args = parser.parse_args()
    if any(
        limit is not None and limit < 1
        for limit in (args.limit, args.daily_limit, args.checkpoint_every)
    ):
        parser.error("--limit, --daily-limit, and --checkpoint-every must be positive")
    if args.reset and args.resume:
        parser.error("--reset and --resume cannot be combined")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "CHECKPOINT_atomic_state.npz"
    lock = FileLock(str(args.output_dir / ".prototype-run.lock"))
    try:
        lock.acquire(timeout=0)
    except Timeout as error:
        raise RuntimeError(f"Another prototype run holds {lock.lock_file}.") from error
    try:
        if args.reset:
            checkpoint.unlink(missing_ok=True)
            for name in (
                "final_arrays.npz",
                "final_report.json",
                "partial_arrays.npz",
                "partial_report.json",
            ):
                (args.output_dir / name).unlink(missing_ok=True)
        started = time.monotonic()
        baseline = pd.read_parquet(BASELINE)
        calendar = pd.DatetimeIndex(pd.to_datetime(baseline["timestamp"]))
        centers_by_timestamp = hasdm_longitudes_by_timestamp(SAMPLES)
        centers = np.array(
            [
                centers_by_timestamp.get(value.to_pydatetime(), np.nan)
                for value in calendar
            ]
        )
        index_by_slot = {
            value.to_datetime64(): index for index, value in enumerate(calendar)
        }
        metadata = {
            "schema": CHECKPOINT_SCHEMA,
            "calendar_sha256": hashlib.sha256(
                calendar.to_numpy(dtype="datetime64[ns]").tobytes()
            ).hexdigest(),
            # Use JSON-native containers so a saved checkpoint compares equal
            # after json.loads on every subsequent resume.
            "channels": list(CHANNELS),
            "cell_widths_degrees": list(CELL_WIDTHS),
            "altitude_bounds_km": list(ALTITUDE_BOUNDS_KM),
            "minimum_span_km": MIN_ALTITUDE_SPAN_KM,
            "daily_limit": args.daily_limit,
        }
        sums, counts, spans, completed, required, failures, diagnostics = (
            load_checkpoint(checkpoint, metadata, len(calendar))
        )
        dirty = 0

        def checkpoint_if_due(*, force: bool = False) -> None:
            nonlocal dirty
            if force or dirty >= args.checkpoint_every:
                save_checkpoint(
                    checkpoint,
                    metadata,
                    sums,
                    counts,
                    spans,
                    completed,
                    required,
                    failures,
                    diagnostics,
                )
                dirty = 0

        def decode_file(
            path: Path, channels: tuple[str, ...], collect_required: bool
        ) -> tuple[
            dict[str, np.ndarray],
            dict[str, np.ndarray],
            dict[str, dict[str, int]],
            dict[str, list[float]],
            set[tuple[date, int]] | None,
        ]:
            """Decode independently so worker threads cannot mutate durable state."""
            delta_sums = {channel: np.zeros(len(calendar)) for channel in channels}
            delta_counts = {
                channel: np.zeros(len(calendar), dtype=np.int64) for channel in channels
            }
            delta_diagnostics = empty_diagnostics()
            delta_spans = {channel: [] for channel in CHANNELS}
            delta_required: set[tuple[date, int]] | None = (
                set() if collect_required else None
            )
            add_file(
                path,
                channels,
                centers,
                index_by_slot,
                delta_sums,
                delta_counts,
                delta_diagnostics,
                delta_spans,
                delta_required,
            )
            return (
                delta_sums,
                delta_counts,
                delta_diagnostics,
                delta_spans,
                delta_required,
            )

        def commit_decoded(
            decoded: tuple[
                dict[str, np.ndarray],
                dict[str, np.ndarray],
                dict[str, dict[str, int]],
                dict[str, list[float]],
                set[tuple[date, int]] | None,
            ],
            channels: tuple[str, ...],
        ) -> None:
            """Serialize state updates before the next atomic checkpoint."""
            (
                delta_sums,
                delta_counts,
                delta_diagnostics,
                delta_spans,
                delta_required,
            ) = decoded
            for channel in channels:
                sums[channel] += delta_sums[channel]
                counts[channel] += delta_counts[channel]
                spans[channel].extend(delta_spans[channel])
                for name, value in delta_diagnostics[channel].items():
                    diagnostics[channel][name] += value
            if delta_required is not None:
                required.update(delta_required)

        def integrate_file(
            path: Path, channels: tuple[str, ...], collect_required: bool
        ) -> None:
            """Decode then commit a source only after every product succeeds."""
            commit_decoded(decode_file(path, channels, collect_required), channels)

        daily_files = {
            "CO2cool": _coverage(CO2_DIR, CO2_FILE_RE, START, END, "CO2 cooling"),
            "NOcool": _coverage(NO_DIR, NO_FILE_RE, START, END, "NO cooling"),
        }
        for channel, files in daily_files.items():
            selected_daily = files[: args.daily_limit]
            for number, path in enumerate(selected_daily, 1):
                key = f"direct:{channel}:{path.name}"
                if key in completed:
                    continue
                try:
                    integrate_file(path, (channel,), collect_required=True)
                except Exception as error:
                    failures[key] = {
                        "stage": "decode_integration",
                        "message": str(error)[:400],
                    }
                else:
                    completed.add(key)
                    failures.pop(key, None)
                dirty += 1
                checkpoint_if_due()
                if number % 100 == 0 or number == len(selected_daily):
                    print(
                        f"direct {channel}: inspected {number}/{len(selected_daily)}",
                        flush=True,
                    )
        direct_missing = sum(
            f"direct:{channel}:{path.name}" not in completed
            for channel, files in daily_files.items()
            for path in files[: args.daily_limit]
        )
        if direct_missing:
            raise RuntimeError(
                f"{direct_missing} direct files failed; fix/retry before Level2A selection."
            )
        checkpoint_if_due(force=True)

        pairs = sorted(required)
        if args.limit is not None:
            pairs = pairs[: args.limit]
        local_index, local_sizes = local_l2a_index(L2A_DIR)
        average_bytes = (
            int(np.mean(local_sizes)) if local_sizes else FALLBACK_FILE_BYTES
        )
        scope = "exact" if args.daily_limit is None else "validation subset; not exact"
        print(
            f"40x60 required pairs={len(required)} ({scope}); selected={len(pairs)}",
            flush=True,
        )
        print_progress(len(pairs), completed, failures, average_bytes)
        if args.dry_run:
            print(
                "dry-run: direct arrays/checkpoint updated; no Level2A URL resolution, download, or decode."
            )
            return

        args.temp_dir.mkdir(parents=True, exist_ok=True)
        cache = args.output_dir / "gats_daily_listing_cache"
        cache.mkdir(exist_ok=True)
        pending: dict[str, tuple[Path, str | None, list[tuple[date, int]]]] = {}

        def add_pending(
            filename: str, path: Path, url: str | None, pair: tuple[date, int]
        ) -> None:
            """Group all requirement aliases for one physical Level2A filename."""
            existing = pending.get(filename)
            if existing is None:
                pending[filename] = (path, url, [pair])
            else:
                if existing[:2] != (path, url):
                    raise RuntimeError(
                        f"Conflicting locations resolved for Level2A file {filename}."
                    )
                existing[2].append(pair)

        remote_pairs: list[tuple[date, int]] = []
        for pair in pairs:
            key = pair_key(pair)
            if key in completed:
                continue
            local = local_l2a(pair, local_index)
            if local is not None:
                if artifact_key(local.name) in completed:
                    completed.add(key)
                    failures.pop(key, None)
                    dirty += 1
                    checkpoint_if_due()
                    continue
                add_pending(local.name, local, None, pair)
                continue
            if args.local_only:
                continue
            remote_pairs.append(pair)

        if remote_pairs:
            resolved_urls = resolve_urls_parallel(remote_pairs, cache, args.workers)
            for pair in remote_pairs:
                key = pair_key(pair)
                resolved = resolved_urls[pair]
                if isinstance(resolved, str):
                    failures[key] = {
                        "stage": "url_resolution",
                        "message": resolved,
                    }
                    dirty += 1
                    checkpoint_if_due()
                    continue
                filename, url = resolved
                if artifact_key(filename) in completed:
                    completed.add(key)
                    failures.pop(key, None)
                    dirty += 1
                    checkpoint_if_due()
                    continue
                add_pending(filename, args.temp_dir / filename, url, pair)

        print(
            f"Level2A ready: physical_local_or_resolved={len(pending)}, unresolved_retryable={len(failures)}",
            flush=True,
        )

        def process(
            filename: str,
            aliases: list[tuple[date, int]],
            path: Path,
            temporary: bool,
        ) -> None:
            nonlocal dirty
            try:
                integrate_file(path, PROXIES, collect_required=False)
            except Exception as error:
                for pair in aliases:
                    failures[pair_key(pair)] = {
                        "stage": "decode_integration",
                        "message": str(error)[:400],
                    }
            else:
                completed.add(artifact_key(filename))
                for pair in aliases:
                    completed.add(pair_key(pair))
                    failures.pop(pair_key(pair), None)
            finally:
                if temporary:
                    path.unlink(missing_ok=True)
                dirty += 1
                checkpoint_if_due()

        local_pending = [
            (filename, *item) for filename, item in pending.items() if item[1] is None
        ]
        for number, (filename, path, _url, aliases) in enumerate(local_pending, 1):
            process(filename, aliases, path, False)
            if number % 50 == 0 or number == len(local_pending):
                print_progress(len(pairs), completed, failures, average_bytes)
        remote_pending = iter(
            (filename, *item)
            for filename, item in pending.items()
            if item[1] is not None
        )

        def download_and_decode(
            url: str, path: Path
        ) -> tuple[str | None, str | None, object | None]:
            """Use each worker for network I/O and independent CPU decoding."""
            error = download(url, path)
            if error is not None:
                return "download", error, None
            try:
                decoded = decode_file(path, PROXIES, collect_required=False)
            except Exception as caught:
                return "decode_integration", str(caught)[:400], None
            finally:
                path.unlink(missing_ok=True)
            return None, None, decoded

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            remote_processed = 0

            def submit_next() -> bool:
                try:
                    filename, path, url, aliases = next(remote_pending)
                except StopIteration:
                    return False
                futures[executor.submit(download_and_decode, url, path)] = (
                    filename,
                    aliases,
                    path,
                )
                return True

            for _ in range(args.workers):
                if not submit_next():
                    break
            while futures:
                future = next(as_completed(futures))
                filename, aliases, path = futures[future]
                del futures[future]
                try:
                    stage, error, decoded = future.result()
                except Exception as caught:  # Keep worker failures retryable too.
                    stage = "download_or_decode_worker"
                    error = str(caught)[:400]
                    decoded = None
                if error is not None:
                    for pair in aliases:
                        failures[pair_key(pair)] = {
                            "stage": stage,
                            "message": error,
                        }
                    dirty += 1
                    checkpoint_if_due()
                else:
                    commit_decoded(decoded, PROXIES)
                    completed.add(artifact_key(filename))
                    for pair in aliases:
                        completed.add(pair_key(pair))
                        failures.pop(pair_key(pair), None)
                    dirty += 1
                    checkpoint_if_due()
                remote_processed += 1
                if remote_processed % 50 == 0 or not futures:
                    print_progress(len(pairs), completed, failures, average_bytes)
                submit_next()

        checkpoint_if_due(force=True)
        exact = (
            args.daily_limit is None
            and args.limit is None
            and all(pair_key(pair) in completed for pair in required)
            and not any(key.startswith("l2a:") for key in failures)
        )
        totals = {
            channel: np.divide(
                sums[channel],
                counts[channel],
                out=np.full(len(calendar), np.nan),
                where=counts[channel] > 0,
            )
            for channel in CHANNELS
        }
        masks = np.array([np.isfinite(totals[channel]) for channel in CHANNELS])
        arrays_name = "final_arrays.npz" if exact else "partial_arrays.npz"
        report_name = "final_report.json" if exact else "partial_report.json"
        atomic_npz(
            args.output_dir / arrays_name,
            timestamps=calendar.to_numpy(dtype="datetime64[ns]"),
            channels=np.array(CHANNELS),
            values=np.array([totals[channel] for channel in CHANNELS]),
            counts=np.array([counts[channel] for channel in CHANNELS]),
            masks=masks,
            joint_all_five_mask=np.all(masks, axis=0),
            **{
                f"accepted_spans_km_{channel}": np.asarray(spans[channel])
                for channel in CHANNELS
            },
        )
        report = {
            "prototype": "THROWAWAY resumable 40x60 SABER Level2A vertical-total stream",
            "runtime_seconds": time.monotonic() - started,
            "cell": {"center_latitude_deg": LATITUDE_CENTER, "widths_deg": CELL_WIDTHS},
            "integration": "independent np.trapezoid native samples, 100--140 km inclusive, altitude metres, no interpolation/extrapolation, >=38 km observed span",
            "proxy_conversion": "O2_1delta_ver, OH_16_ver, OH_20_ver multiplied by 0.1 from ergs/cm3/sec to W/m3",
            "required_pairs_exact_40x60": len(required),
            "daily_limit": args.daily_limit,
            "exact_full_cell": exact,
            "output_status": "exact_complete" if exact else "partial_not_for_pcmci",
            "selected_pairs": len(pairs),
            "completed_pairs": sum(key.startswith("l2a:") for key in completed),
            "retryable_failures": failures,
            "per_product": {
                channel: {
                    "slot_count": int(masks[index].sum()),
                    "observation_count": int(counts[channel].sum()),
                    "diagnostics": diagnostics[channel],
                }
                for index, channel in enumerate(CHANNELS)
            },
            "joint_all_five_slot_count": int(np.all(masks, axis=0).sum()),
            "checkpoint": str(checkpoint),
        }
        atomic_json(args.output_dir / report_name, report)
        print(
            f"wrote {args.output_dir / arrays_name} and {report_name}",
            flush=True,
        )
    finally:
        lock.release()


if __name__ == "__main__":
    main()
