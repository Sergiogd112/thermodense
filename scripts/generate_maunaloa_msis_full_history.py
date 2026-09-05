"""Generate the reproducible full-history Mauna Loa MSIS density baseline.

The driver construction intentionally mirrors :func:`pymsis.utils.get_f107_ap`
without reading pymsis's bundled cache or allowing it to download data.  As in
pymsis 0.12.0, isolated non-physical F10.7 radio bursts (>400 or <=0) are
replaced by the source CSV's centered-81 value before their next-day use. The
seven-element Ap array is supplied, but no ``geomagnetic_activity`` switch is
passed: pymsis's default handling uses daily Ap (element zero); elements one
through six are used only with ``geomagnetic_activity=-1``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

MAUNA_LOA_LAT = 19.5362
MAUNA_LOA_LON_EAST = 204.4237
ALTITUDES_KM = tuple(range(125, 826, 25))
MODEL_VERSIONS = {
    "nrlmsise_00": "0",
    "nrlmsis_2p0": "2.0",
    "nrlmsis_2p1": "2.1",
}
REQUIRED_COLUMNS = (
    "F10.7_OBS",
    "F10.7_OBS_CENTER81",
    "AP_AVG",
    "AP1",
    "AP2",
    "AP3",
    "AP4",
    "AP5",
    "AP6",
    "AP7",
    "AP8",
)
SOURCE_PATH = Path("data/original/space_weather/SW-All.csv")
OUTPUT_PATH = Path(
    "outputs/figures/results/maunaloa_msis_density_baselines/data/"
    "maunaloa_msis_density_baselines_daily_wide.parquet"
)
HISTORY_START = date(1967, 1, 1)
HOURS_PER_DAY = 24

ModelExecutor = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str],
    np.ndarray,
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_space_weather(path: Path = SOURCE_PATH) -> dict[date, dict[str, float]]:
    """Load only complete daily driver rows from the frozen CelesTrak CSV."""
    frame = pl.read_csv(path, null_values=[""]).with_columns(
        pl.col("DATE").str.to_date("%Y-%m-%d").alias("date")
    )
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Space-weather CSV is missing columns: {sorted(missing)}")
    complete = frame.filter(
        pl.all_horizontal([pl.col(column).is_not_null() for column in REQUIRED_COLUMNS])
    ).select("date", *REQUIRED_COLUMNS)
    return {
        row["date"]: {column: float(row[column]) for column in REQUIRED_COLUMNS}
        for row in complete.iter_rows(named=True)
    }


def contiguous_coverage_end(drivers: dict[date, dict[str, float]], start: date) -> date:
    """Return the final uninterrupted complete-driver day beginning at ``start``."""
    current = start
    while current in drivers:
        current += timedelta(days=1)
    return current - timedelta(days=1)


def validate_requested_dates(
    drivers: dict[date, dict[str, float]], start: date, end: date
) -> None:
    if end < start:
        raise ValueError("--end must not precede --start")
    coverage_end = contiguous_coverage_end(drivers, HISTORY_START)
    if start < HISTORY_START or end > coverage_end:
        raise ValueError(
            f"Requested dates must be within {HISTORY_START.isoformat()} through "
            f"{coverage_end.isoformat()}, the contiguous complete source coverage."
        )
    # The first requested F10.7 is previous-day data and the Ap history reaches
    # 57 hours back, so verify the three preceding source days too.
    for offset in range(1, 4):
        required_day = start - timedelta(days=offset)
        if required_day not in drivers:
            raise ValueError(f"Missing required driver history for {required_day}")


def hourly_timestamps(start: date, end: date) -> np.ndarray:
    count = ((end - start).days + 1) * HOURS_PER_DAY
    return np.datetime64(start, "h") + np.arange(count).astype("timedelta64[h]")


def build_drivers(
    timestamps: np.ndarray, drivers: dict[date, dict[str, float]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build pymsis-compatible F10.7 and Ap inputs from the supplied CSV.

    This follows pymsis 0.12.0's ``get_f107_ap`` indexing: Ap bins are selected
    at the current 3-hour bin, then at -3/-6/-9 h, with means for bins -12..-33
    and -36..-57 h inclusive.
    """
    timestamps = np.asarray(timestamps, dtype="datetime64[h]")
    f107s = np.empty(len(timestamps), dtype=float)
    f107as = np.empty(len(timestamps), dtype=float)
    aps = np.empty((len(timestamps), 7), dtype=float)
    for index, timestamp in enumerate(timestamps):
        timestamp_seconds = int(timestamp.astype("datetime64[s]").astype(int))
        current_datetime = datetime.fromtimestamp(timestamp_seconds, tz=UTC)
        current_day = current_datetime.date()
        current = drivers[current_day]
        previous_day = drivers[current_day - timedelta(days=1)]
        previous_f107 = previous_day["F10.7_OBS"]
        # This is pymsis 0.12.0's get_f107_ap radio-burst safeguard, reproduced
        # from the supplied source rather than an implicit pymsis data download.
        f107s[index] = (
            previous_day["F10.7_OBS_CENTER81"]
            if previous_f107 <= 0 or previous_f107 > 400
            else previous_f107
        )
        f107as[index] = current["F10.7_OBS_CENTER81"]
        ap_bins = []
        for hours_before in range(0, 58, 3):
            bin_datetime = current_datetime - timedelta(hours=hours_before)
            bin_day = bin_datetime.date()
            ap_bin = bin_datetime.hour // 3 + 1
            ap_bins.append(drivers[bin_day][f"AP{ap_bin}"])
        aps[index] = (
            current["AP_AVG"],
            ap_bins[0],
            ap_bins[1],
            ap_bins[2],
            ap_bins[3],
            float(np.mean(ap_bins[4:12])),
            float(np.mean(ap_bins[12:20])),
        )
    return f107s, f107as, aps


def default_model_executor(
    timestamps: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    alts: np.ndarray,
    f107s: np.ndarray,
    f107as: np.ndarray,
    aps: np.ndarray,
    version: str,
) -> np.ndarray:
    """Execute pymsis without implicit driver lookup or switch overrides."""
    from pymsis import msis

    return msis.calculate(timestamps, lons, lats, alts, f107s, f107as, aps, version=version)


def daily_columns() -> list[str]:
    columns = ["date", "F10.7_OBS_CENTER81"]
    for model_name in MODEL_VERSIONS:
        for altitude in ALTITUDES_KM:
            columns.extend(
                [
                    f"{model_name}_log10rho_daily_mean_{altitude}km",
                    f"{model_name}_log10rho_daily_range_{altitude}km",
                ]
            )
    return columns


def aggregate_chunk(
    timestamps: np.ndarray,
    f107as: np.ndarray,
    model_densities: dict[str, np.ndarray],
) -> pl.DataFrame:
    """Validate hourly densities and produce the daily-wide output contract."""
    if len(timestamps) % HOURS_PER_DAY:
        raise ValueError("Hourly timestamps must contain whole UTC days")
    dates = timestamps[::HOURS_PER_DAY].astype("datetime64[D]").astype(object)
    rows: dict[str, object] = {
        "date": dates.tolist(),
        "F10.7_OBS_CENTER81": f107as.reshape(-1, HOURS_PER_DAY)[:, 0],
    }
    days = len(dates)
    for model_name, density in model_densities.items():
        density = np.asarray(density, dtype=float)
        expected_shape = (len(timestamps), len(ALTITUDES_KM))
        if density.shape != expected_shape:
            raise ValueError(f"{model_name} returned {density.shape}; expected {expected_shape}")
        if not np.all(np.isfinite(density) & (density > 0)):
            raise ValueError(f"{model_name} produced non-finite or non-positive density")
        log_density = np.log10(density).reshape(days, HOURS_PER_DAY, len(ALTITUDES_KM))
        for altitude_index, altitude in enumerate(ALTITUDES_KM):
            values = log_density[:, :, altitude_index]
            rows[f"{model_name}_log10rho_daily_mean_{altitude}km"] = values.mean(axis=1)
            rows[f"{model_name}_log10rho_daily_range_{altitude}km"] = (
                values.max(axis=1) - values.min(axis=1)
            )
    result = pl.DataFrame(rows).select(daily_columns())
    if result.height != days:
        raise ValueError("Expected one daily row per complete 24-hour block")
    return result


def month_chunks(start: date, end: date) -> Iterator[tuple[date, date]]:
    current = start.replace(day=1)
    while current <= end:
        next_month = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
        yield max(start, current), min(end, next_month - timedelta(days=1))
        current = next_month


def process_chunk(
    start: date,
    end: date,
    drivers: dict[date, dict[str, float]],
    executor: ModelExecutor,
) -> pl.DataFrame:
    timestamps = hourly_timestamps(start, end)
    f107s, f107as, aps = build_drivers(timestamps, drivers)
    hourly_count = len(timestamps)
    expanded_timestamps = np.repeat(timestamps, len(ALTITUDES_KM))
    densities: dict[str, np.ndarray] = {}
    for safe_name, version in MODEL_VERSIONS.items():
        output = executor(
            expanded_timestamps,
            np.full(hourly_count * len(ALTITUDES_KM), MAUNA_LOA_LON_EAST),
            np.full(hourly_count * len(ALTITUDES_KM), MAUNA_LOA_LAT),
            np.tile(ALTITUDES_KM, hourly_count),
            np.repeat(f107s, len(ALTITUDES_KM)),
            np.repeat(f107as, len(ALTITUDES_KM)),
            np.repeat(aps, len(ALTITUDES_KM), axis=0),
            version,
        )
        densities[safe_name] = np.asarray(output, dtype=float)[:, 0].reshape(
            hourly_count, len(ALTITUDES_KM)
        )
    return aggregate_chunk(timestamps, f107as, densities)


def generate(
    start: date,
    end: date,
    output: Path = OUTPUT_PATH,
    source: Path = SOURCE_PATH,
    executor: ModelExecutor = default_model_executor,
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Generate output in monthly checkpoints and return its table/provenance."""
    started = time.monotonic()
    drivers = load_space_weather(source)
    validate_requested_dates(drivers, start, end)
    source_hash = file_sha256(source)
    generator_hash = file_sha256(Path(__file__))
    try:
        import pymsis

        pymsis_version = pymsis.__version__
    except ImportError:
        pymsis_version = "unavailable (custom executor)"
    checkpoint_key = hashlib.sha256(
        json.dumps(
            {
                "source_sha256": source_hash,
                "generator_sha256": generator_hash,
                "start": str(start),
                "end": str(end),
                "latitude": MAUNA_LOA_LAT,
                "longitude_east": MAUNA_LOA_LON_EAST,
                "altitudes": ALTITUDES_KM,
                "models": MODEL_VERSIONS,
                "pymsis_version": pymsis_version,
                "driver_algorithm": "pymsis-0.12.0-radio-burst-replacement",
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    checkpoint_dir = output.parent / f".{output.stem}.checkpoints" / checkpoint_key
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    chunks = list(month_chunks(start, end))
    frames = []
    for chunk_start, chunk_end in chunks:
        checkpoint = checkpoint_dir / f"{chunk_start:%Y-%m}.parquet"
        if checkpoint.exists():
            frame = pl.read_parquet(checkpoint)
        else:
            frame = process_chunk(chunk_start, chunk_end, drivers, executor)
            frame.write_parquet(checkpoint, compression="lz4")
        frames.append(frame)
    result = pl.concat(frames).sort("date")
    expected_days = (end - start).days + 1
    if result.height != expected_days or result.select(pl.col("date").n_unique()).item() != expected_days:
        raise ValueError("Checkpoint output does not contain exactly one row per requested day")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(output, compression="lz4")
    runtime_seconds = time.monotonic() - started
    provenance: dict[str, object] = {
        "command": " ".join(sys.argv),
        "frozen_config": {
            "latitude_geodetic_degrees": MAUNA_LOA_LAT,
            "longitude_east_degrees": MAUNA_LOA_LON_EAST,
            "altitudes_km": list(ALTITUDES_KM),
            "cadence": "hourly UTC",
            "daily_aggregation": "mean and max-minus-min of log10(total mass density kg/m^3), exactly 24 samples/day",
            "models": MODEL_VERSIONS,
            "geomagnetic_handling": "pymsis default; no geomagnetic_activity switch; 7-element Ap supplied",
        },
        "requested_dates": {"start": str(start), "end": str(end)},
        "actual_dates": {"start": str(result["date"].min()), "end": str(result["date"].max())},
        "source": {"path": str(source), "sha256": source_hash},
        "generator_sha256": generator_hash,
        "pymsis_version": pymsis_version,
        "driver_construction": {
            "f107s": "previous-day F10.7_OBS; pymsis-0.12.0 radio-burst safeguard substitutes that day's F10.7_OBS_CENTER81 when F10.7_OBS <=0 or >400",
            "f107as": "current-day F10.7_OBS_CENTER81",
            "aps": "[AP_AVG, current, -3h, -6h, -9h, mean(-12..-33h), mean(-36..-57h)]",
        },
        "counts": {"daily_rows": result.height, "hourly_timestamps": result.height * HOURS_PER_DAY, "model_altitude_series": len(MODEL_VERSIONS) * len(ALTITUDES_KM)},
        "runtime_seconds": runtime_seconds,
        "output": {"path": str(output), "sha256": file_sha256(output), "columns": len(result.columns)},
    }
    output.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result, provenance


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat)
    parser.add_argument("--end", type=date.fromisoformat)
    parser.add_argument("--smoke", action="store_true", help="Run the final three source days.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    drivers = load_space_weather()
    coverage_end = contiguous_coverage_end(drivers, HISTORY_START)
    start = args.start or (coverage_end - timedelta(days=2) if args.smoke else HISTORY_START)
    end = args.end or coverage_end
    result, provenance = generate(start, end, args.output)
    print(
        f"Wrote {result.height} daily rows ({provenance['actual_dates']}) to {args.output} "
        f"in {provenance['runtime_seconds']:.1f} s"
    )


if __name__ == "__main__":
    main()
