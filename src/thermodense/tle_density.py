"""Generate TLE-derived density rows for the Figure 19 replica.

The implementation mirrors the convention used in Sylvester's MATLAB scripts:
successive TLE epochs are thinned to at least three days apart, each interval is
sampled with SGP4, and observed density is inferred from mean-motion decay and
the supplied ballistic coefficient.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from astropy import units as u
from astropy.coordinates import CartesianRepresentation, EarthLocation, ITRS, TEME
from astropy.time import Time
from pymsis import msis
from pymsis.utils import get_f107_ap
from sgp4.api import Satrec

MODELS = ("0", "2.0", "2.1")
DEFAULT_SYLVESTER_DIR = Path(
    "/home/sergiogd/Github/exodense/TLE Code and Data Sylvester/TLE data"
)
EARTH_ROTATION_RAD_S = 7.272e-5
EARTH_MU_KM3_S2 = 398600.4418
TLE_DENSITY_SCHEMA = {
    "timestamp": pl.Datetime,
    "satellite_id": pl.Utf8,
    "period": pl.Float64,
    "Altitude (m)": pl.Float64,
    "Latitude (deg)": pl.Float64,
    "Longitude (deg)": pl.Float64,
    "Density (kg/m^3)": pl.Float64,
    "msis_density_0": pl.Float64,
    "msis_density_2.0": pl.Float64,
    "msis_density_2.1": pl.Float64,
    "f107": pl.Float64,
    "f107a": pl.Float64,
    "ap": pl.Float64,
    "ln_density_ratio_0": pl.Float64,
    "ln_density_ratio_2.0": pl.Float64,
    "ln_density_ratio_2.1": pl.Float64,
}


@dataclass(frozen=True)
class SatelliteConfig:
    satellite_id: str
    ballistic_coefficient_m2_kg: float


@dataclass(frozen=True)
class TLERecord:
    line1: str
    line2: str
    epoch: datetime
    mean_motion_rev_day: float


def parse_satellite_list(path: str | Path) -> list[SatelliteConfig]:
    """Read `SAT_list_ALL.txt` style rows: NORAD id and ballistic coefficient."""
    configs = []
    with Path(path).open() as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            configs.append(SatelliteConfig(parts[0], float(parts[1])))
    return configs


def tle_epoch(line1: str) -> datetime:
    """Convert the epoch field in a TLE line 1 to a UTC datetime."""
    sat = Satrec.twoline2rv(line1, _dummy_line2_for(line1))
    return Time(sat.jdsatepoch + sat.jdsatepochF, format="jd", scale="utc").to_datetime(
        timezone=timezone.utc
    )


def _dummy_line2_for(line1: str) -> str:
    satnum = line1[2:7]
    return f"2 {satnum} 000.0000 000.0000 0000000 000.0000 000.0000 01.00000000    00"


def parse_tle_file(path: str | Path) -> list[TLERecord]:
    lines = [
        line.strip()
        for line in Path(path).read_text(errors="replace").splitlines()
        if line.strip()
    ]
    records = []
    i = 0
    while i < len(lines) - 1:
        if not lines[i].startswith("1 "):
            i += 1
            continue
        if not lines[i + 1].startswith("2 "):
            i += 1
            continue
        line1 = lines[i]
        line2 = lines[i + 1]
        sat = Satrec.twoline2rv(line1, line2)
        epoch = Time(
            sat.jdsatepoch + sat.jdsatepochF, format="jd", scale="utc"
        ).to_datetime(timezone=timezone.utc)
        records.append(
            TLERecord(
                line1=line1,
                line2=line2,
                epoch=epoch,
                mean_motion_rev_day=float(line2[52:63]),
            )
        )
        i += 2
    return sorted(records, key=lambda rec: rec.epoch)


def filter_tles_by_epoch_gap(
    records: Iterable[TLERecord], min_days: float = 3.0
) -> list[TLERecord]:
    selected = []
    min_seconds = min_days * 86400.0
    for record in sorted(records, key=lambda rec: rec.epoch):
        if not selected:
            selected.append(record)
            continue
        if (record.epoch - selected[-1].epoch).total_seconds() >= min_seconds:
            selected.append(record)
    return selected


def density_ratios(rho_obs: np.ndarray, model_densities: dict[str, np.ndarray]):
    """Return ln(rho_model / rho_obs) for each model version."""
    return {
        f"ln_density_ratio_{version}": np.log(values / rho_obs)
        for version, values in model_densities.items()
    }


def find_local_tle_file(satellite_id: str, sylvester_dir: Path) -> Path | None:
    candidates = [
        sylvester_dir / "TLE nuevos" / f"new{satellite_id}.txt",
        sylvester_dir / "TLE data" / "1U CubeSats" / f"{satellite_id}.txt",
        sylvester_dir / "TLE data" / "CelesTrak" / f"sat{int(satellite_id):09d}.txt",
        sylvester_dir / f"{satellite_id}.txt",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate

    matches = sorted(sylvester_dir.glob(f"**/*{satellite_id}*.txt"))
    for match in matches:
        if (
            match.stat().st_size > 0
            and "MSISE" not in match.name
            and "derived" not in match.name
            and "Bstar" not in match.name
        ):
            return match
    return None


def mirror_local_tle(satellite_id: str, source: Path, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / f"{satellite_id}.txt"
    if not dest.exists() or dest.read_bytes() != source.read_bytes():
        shutil.copyfile(source, dest)
    return dest


def download_spacetrack_tle(
    satellite_id: str, raw_dir: Path, cookie: str
) -> Path | None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    url = (
        "https://www.space-track.org/basicspacedata/query/class/gp_history/"
        f"NORAD_CAT_ID/{satellite_id}/orderby/EPOCH asc/format/tle"
    )
    request = urllib.request.Request(url, headers={"Cookie": cookie})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read().decode("utf-8", errors="replace")
    except urllib.error.URLError, TimeoutError:
        return None

    if "1 " not in payload or "2 " not in payload:
        return None
    dest = raw_dir / f"{satellite_id}.txt"
    dest.write_text(payload)
    return dest


def _sample_interval(
    first: TLERecord, second: TLERecord, sample_minutes: float
) -> tuple[np.ndarray, Time, np.ndarray, np.ndarray, np.ndarray]:
    sat = Satrec.twoline2rv(first.line1, first.line2)
    interval_seconds = (second.epoch - first.epoch).total_seconds()
    if interval_seconds <= 0:
        raise ValueError("TLE epochs must be strictly increasing")

    step_seconds = sample_minutes * 60.0
    offsets = np.arange(0.0, interval_seconds, step_seconds)
    if len(offsets) == 0 or offsets[-1] != interval_seconds:
        offsets = np.append(offsets, interval_seconds)

    start = Time(first.epoch)
    times = start + offsets * u.s
    errors, positions, velocities = sat.sgp4_array(times.jd1, times.jd2)
    valid = errors == 0
    return offsets[valid], times[valid], positions[valid], velocities[valid], errors


def _teme_to_geodetic(times: Time, positions_km: np.ndarray):
    teme = TEME(
        CartesianRepresentation(
            positions_km[:, 0] * u.km,
            positions_km[:, 1] * u.km,
            positions_km[:, 2] * u.km,
        ),
        obstime=times,
    )
    itrs = teme.transform_to(ITRS(obstime=times))
    loc = EarthLocation.from_geocentric(
        itrs.x.to_value(u.m), itrs.y.to_value(u.m), itrs.z.to_value(u.m), unit=u.m
    )
    lon, lat, height = loc.to_geodetic()
    return lat.deg, lon.deg, height.to_value(u.m)


def _orbit_weighted_mean(values: np.ndarray, weights: np.ndarray, seconds: np.ndarray):
    numerator = np.trapezoid(values * weights, seconds)
    denominator = np.trapezoid(weights, seconds)
    if denominator <= 0:
        return np.nan
    return numerator / denominator


def process_tle_interval(
    satellite_id: str,
    ballistic_coefficient_m2_kg: float,
    first: TLERecord,
    second: TLERecord,
    sample_minutes: float = 4.0,
) -> dict[str, float | str | datetime] | None:
    offsets, times, positions, velocities, errors = _sample_interval(
        first, second, sample_minutes
    )
    if len(offsets) < 2 or np.any(errors != 0):
        return None

    lats, lons, alts_m = _teme_to_geodetic(times, positions)
    keep = alts_m < 800_000.0
    if keep.sum() < 2:
        return None

    offsets = offsets[keep]
    py_datetimes = np.array(
        [
            dt.replace(tzinfo=None)
            for dt in times[keep].to_datetime(timezone=timezone.utc)
        ]
    )
    lats = lats[keep]
    lons = lons[keep]
    alts_m = alts_m[keep]
    positions = positions[keep]
    velocities = velocities[keep]

    speed_km_s = np.linalg.norm(velocities, axis=1)
    radius_km = np.linalg.norm(positions, axis=1)
    incl = Satrec.twoline2rv(first.line1, first.line2).inclo
    drag_geometry = (
        1.0 - ((radius_km * EARTH_ROTATION_RAD_S) / speed_km_s) * np.cos(incl)
    ) ** 2
    weights = speed_km_s**3 * drag_geometry

    integral = np.trapezoid(weights, offsets)
    if integral <= 0:
        return None

    ndiff = abs(second.mean_motion_rev_day - first.mean_motion_rev_day) * (
        2.0 * np.pi / 86400.0
    )
    nmean = ((second.mean_motion_rev_day + first.mean_motion_rev_day) / 2.0) * (
        2.0 * np.pi / 86400.0
    )
    bc_km2_kg = ballistic_coefficient_m2_kg * 1e-6
    if ndiff <= 0 or nmean <= 0 or bc_km2_kg <= 0:
        return None
    rho_obs = (
        (ndiff * (2.0 / 3.0) * EARTH_MU_KM3_S2 ** (2.0 / 3.0) * nmean ** (-1.0 / 3.0))
        / (integral * bc_km2_kg)
    ) / 1e9
    if not np.isfinite(rho_obs) or rho_obs <= 0:
        return None

    f107, f107a, aps, _ = get_f107_ap(py_datetimes)
    model_means = {}
    for version in MODELS:
        densities = msis.calculate(
            py_datetimes,
            lons,
            lats,
            alts_m / 1000.0,
            f107,
            f107a,
            aps,
            version=version,
        )[:, 0]
        model_means[version] = _orbit_weighted_mean(densities, weights, offsets)

    midpoint = first.epoch + (second.epoch - first.epoch) / 2
    row = {
        "timestamp": midpoint.replace(tzinfo=None),
        "satellite_id": satellite_id,
        "period": (second.epoch - first.epoch).total_seconds() / 86400.0,
        "Altitude (m)": float(np.average(alts_m, weights=weights)),
        "Latitude (deg)": float(np.average(lats, weights=weights)),
        "Longitude (deg)": float(np.average(lons, weights=weights)),
        "Density (kg/m^3)": float(rho_obs),
        "f107": float(np.mean(f107)),
        "f107a": float(np.mean(f107a)),
        "ap": float(np.mean(aps[:, 0])),
    }
    for version, density in model_means.items():
        row[f"msis_density_{version}"] = float(density)
    for name, value in density_ratios(
        np.array([rho_obs]), {k: np.array([v]) for k, v in model_means.items()}
    ).items():
        row[name] = float(value[0])
    return row


def append_manifest(path: Path, fieldnames: list[str], row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fieldnames})


def _rows_to_frame(rows: list[dict]) -> pl.DataFrame:
    return (
        pl.DataFrame(rows, schema=TLE_DENSITY_SCHEMA)
        if rows
        else pl.DataFrame(schema=TLE_DENSITY_SCHEMA)
    )


def _write_rows_batch(
    writer: pq.ParquetWriter | None,
    rows: list[dict],
    output_path: Path,
) -> pq.ParquetWriter | None:
    if not rows:
        return writer
    table = _rows_to_frame(rows).to_arrow()
    if writer is None:
        writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
    writer.write_table(table)
    rows.clear()
    return writer


def generate_tle_density(
    sat_list_path: Path,
    sylvester_dir: Path,
    output_dir: Path,
    max_satellites: int | None = None,
    sample_minutes: float = 4.0,
    start_date: str | None = None,
    end_date: str | None = None,
    batch_rows: int = 1000,
) -> pl.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    skipped_manifest = output_dir / "skipped_manifest.csv"
    processed_manifest = output_dir / "processing_manifest.csv"
    download_manifest = output_dir / "download_manifest.csv"
    rows = []
    output_path = output_dir / "tle_density.parquet"
    tmp_output_path = output_dir / "tle_density.parquet.tmp"
    if tmp_output_path.exists():
        tmp_output_path.unlink()
    writer = None
    total_rows = 0

    configs = parse_satellite_list(sat_list_path)
    if max_satellites is not None:
        configs = configs[:max_satellites]

    start_dt = (
        datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
        if start_date
        else None
    )
    end_dt = (
        datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
        if end_date
        else None
    )
    cookie = os.environ.get("SPACETRACK_COOKIE", "")

    for config in configs:
        source = find_local_tle_file(config.satellite_id, sylvester_dir)
        source_type = "local"
        if source is None and cookie:
            source = download_spacetrack_tle(config.satellite_id, raw_dir, cookie)
            source_type = "spacetrack" if source else "missing"
        if source is None:
            append_manifest(
                skipped_manifest,
                ["satellite_id", "reason", "finished_at_utc"],
                {
                    "satellite_id": config.satellite_id,
                    "reason": "no_local_or_spacetrack_tle",
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            continue

        raw_path = (
            source
            if source.parent == raw_dir
            else mirror_local_tle(config.satellite_id, source, raw_dir)
        )
        append_manifest(
            download_manifest,
            [
                "satellite_id",
                "source_type",
                "source_path",
                "raw_path",
                "finished_at_utc",
            ],
            {
                "satellite_id": config.satellite_id,
                "source_type": source_type,
                "source_path": str(source),
                "raw_path": str(raw_path),
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

        records = filter_tles_by_epoch_gap(parse_tle_file(raw_path), min_days=3.0)
        if start_dt or end_dt:
            records = [
                record
                for record in records
                if (start_dt is None or record.epoch >= start_dt)
                and (end_dt is None or record.epoch <= end_dt)
            ]
        processed = 0
        for first, second in zip(records, records[1:]):
            row = process_tle_interval(
                config.satellite_id,
                config.ballistic_coefficient_m2_kg,
                first,
                second,
                sample_minutes=sample_minutes,
            )
            if row is not None:
                rows.append(row)
                processed += 1
                total_rows += 1
                if len(rows) >= batch_rows:
                    writer = _write_rows_batch(writer, rows, tmp_output_path)

        append_manifest(
            processed_manifest,
            ["satellite_id", "tle_pairs", "rows", "raw_path", "finished_at_utc"],
            {
                "satellite_id": config.satellite_id,
                "tle_pairs": max(len(records) - 1, 0),
                "rows": processed,
                "raw_path": str(raw_path),
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    writer = _write_rows_batch(writer, rows, tmp_output_path)
    if writer is not None:
        writer.close()
        tmp_output_path.replace(output_path)
    else:
        _rows_to_frame([]).write_parquet(output_path, compression="zstd")
    (output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "sat_list_path": str(sat_list_path),
                "sylvester_dir": str(sylvester_dir),
                "rows": total_rows,
                "sample_minutes": sample_minutes,
                "start_date": start_date,
                "end_date": end_date,
                "batch_rows": batch_rows,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
    )
    return pl.read_parquet(output_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sat-list", type=Path, default=DEFAULT_SYLVESTER_DIR / "SAT_list_ALL.txt"
    )
    parser.add_argument("--sylvester-dir", type=Path, default=DEFAULT_SYLVESTER_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("data/original/TLE"))
    parser.add_argument("--max-satellites", type=int, default=None)
    parser.add_argument("--sample-minutes", type=float, default=4.0)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--batch-rows", type=int, default=1000)
    args = parser.parse_args(argv)

    df = generate_tle_density(
        sat_list_path=args.sat_list,
        sylvester_dir=args.sylvester_dir,
        output_dir=args.output_dir,
        max_satellites=args.max_satellites,
        sample_minutes=args.sample_minutes,
        start_date=args.start_date,
        end_date=args.end_date,
        batch_rows=args.batch_rows,
    )
    print(f"Wrote {len(df)} rows to {args.output_dir / 'tle_density.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
