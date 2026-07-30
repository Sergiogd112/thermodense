from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
from scipy.io import netcdf_file
from tqdm import tqdm

MAUNA_LOA_LAT = 19.5362
MAUNA_LOA_LON_EAST = 204.4237
INPUT_DIR = Path("data/original/saber/co2_cooling_profiles")
OUTPUT_DIR = Path("data/decoded/saber")
OUTPUT_PATH = OUTPUT_DIR / "saber_co2_cooling_maunaloa_daily.parquet"
MISSING_VALUE_LIMIT = -1.0e20


def circular_lon_delta(lon_a: np.ndarray, lon_b: float) -> np.ndarray:
    return np.abs((lon_a - lon_b + 180.0) % 360.0 - 180.0)


def date_from_year_doy(year: int, doy: int) -> date:
    return date(year, 1, 1) + timedelta(days=doy - 1)


def decode_saber_file(path: Path) -> pl.DataFrame:
    with netcdf_file(path, "r", mmap=False) as ds:
        year = int(np.asarray(ds.variables["year"].data).item())
        doy = int(np.asarray(ds.variables["day"].data).item())
        altitudes = np.asarray(ds.variables["altitude"].data, dtype=float)
        cooling = np.asarray(ds.variables["CO2cool"].data, dtype=float)
        latitudes = np.asarray(ds.variables["tplatitude"].data, dtype=float)
        longitudes = np.asarray(ds.variables["tplongitude"].data, dtype=float)
        times = np.asarray(ds.variables["time"].data, dtype=float)
        flux = np.asarray(ds.variables["flux"].data, dtype=float)

    # scipy returns NetCDF variables in file order: scans x altitude for these files.
    if (
        cooling.shape[1] != altitudes.shape[0]
        and cooling.shape[0] == altitudes.shape[0]
    ):
        cooling = cooling.T
        latitudes = latitudes.T
        longitudes = longitudes.T
        times = times.T

    rows = []
    file_date = date_from_year_doy(year, doy)
    for alt_idx, altitude_km in enumerate(altitudes):
        alt_cooling = cooling[:, alt_idx]
        alt_lat = latitudes[:, alt_idx]
        alt_lon = longitudes[:, alt_idx] % 360.0
        alt_time = times[:, alt_idx]
        finite = (
            np.isfinite(alt_cooling)
            & np.isfinite(alt_lat)
            & np.isfinite(alt_lon)
            & np.isfinite(alt_time)
            & (alt_cooling > MISSING_VALUE_LIMIT)
        )
        if not np.any(finite):
            continue

        lat_delta = alt_lat[finite] - MAUNA_LOA_LAT
        lon_delta = circular_lon_delta(alt_lon[finite], MAUNA_LOA_LON_EAST)
        distance_deg = np.hypot(
            lat_delta, lon_delta * np.cos(np.deg2rad(MAUNA_LOA_LAT))
        )
        nearest_idx = int(np.argmin(distance_deg))
        scan_indices = np.flatnonzero(finite)
        scan_idx = int(scan_indices[nearest_idx])
        millis = float(alt_time[scan_idx])
        rows.append(
            {
                "date": file_date,
                "timestamp": datetime.combine(file_date, datetime.min.time())
                + timedelta(milliseconds=millis),
                "altitude_km": float(altitude_km),
                "co2_cooling_rate_w_m3": float(alt_cooling[scan_idx]),
                "nearest_latitude_deg": float(alt_lat[scan_idx]),
                "nearest_longitude_deg_east": float(alt_lon[scan_idx]),
                "nearest_distance_deg": float(distance_deg[nearest_idx]),
                "scan_index": scan_idx,
                "flux_w_m2": (
                    float(flux[scan_idx]) if np.isfinite(flux[scan_idx]) else None
                ),
                "source_file": path.name,
            }
        )
    return pl.DataFrame(rows)


def decode_saber_dataset(
    input_dir: Path = INPUT_DIR,
    output_path: Path = OUTPUT_PATH,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pl.DataFrame:
    files = sorted(input_dir.glob("SABER_CO2_PROFILE_FLUX_*_V*.nc"))
    frames = []
    for path in tqdm(files, desc="Decoding SABER CO2 cooling"):
        try:
            frame = decode_saber_file(path)
        except Exception as exc:
            print(f"Skipping {path}: {exc}")
            continue
        if frame.is_empty():
            continue
        if start_date is not None:
            frame = frame.filter(pl.col("date") >= start_date)
        if end_date is not None:
            frame = frame.filter(pl.col("date") <= end_date)
        if not frame.is_empty():
            frames.append(frame)

    if not frames:
        raise RuntimeError(f"No SABER files decoded from {input_dir}")

    decoded = pl.concat(frames).sort(["date", "altitude_km"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    decoded.write_parquet(output_path, compression="lz4")
    return decoded


def parse_date(value: str | None) -> date | None:
    if value is None:
        return None
    return date.fromisoformat(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode SABER CO2 cooling profiles near Mauna Loa."
    )
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    args = parser.parse_args()

    decoded = decode_saber_dataset(
        args.input_dir,
        args.output,
        start_date=parse_date(args.start_date),
        end_date=parse_date(args.end_date),
    )
    print(
        decoded.select(
            pl.min("date").alias("date_min"),
            pl.max("date").alias("date_max"),
            pl.col("altitude_km").n_unique().alias("altitudes"),
            pl.len().alias("rows"),
        )
    )


if __name__ == "__main__":
    main()
