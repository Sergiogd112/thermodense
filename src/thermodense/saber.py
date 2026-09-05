"""Prepare spatially matched, 3-hour SABER products for the HASDM PCMCI input."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
from scipy.io import netcdf_file

MAUNA_LOA_LATITUDE = 19.5362
MAUNA_LOA_LONGITUDE_EAST = 204.4237
TARGET_ALTITUDES_KM = (100.0, 119.0, 139.0)
ALTITUDE_TOLERANCE_KM = 0.5
MISSION_START = date(2002, 1, 25)
DEFAULT_END = datetime(2025, 7, 20)
MISSING_VALUE_LIMIT = -1.0e20
CO2_FILE_RE = re.compile(r"SABER_CO2_PROFILE_FLUX_(\d{4})(\d{3})_V(\d+)\.(\d+)\.nc")
NO_FILE_RE = re.compile(r"SABER_NO_PROFILE_FLUX_(\d{4})(\d{3})_V(\d+)\.(\d+)\.nc")
DIRECT_PRODUCTS = {"CO2cool": "saber_co2cool", "NOcool": "saber_nocool"}
PROXY_PRODUCTS = {
    "O2_1delta_ver": "saber_o2_1delta_ver",
    "OH_16_ver": "saber_oh_16_ver",
    "OH_20_ver": "saber_oh_20_ver",
}


def circular_longitude_delta(longitude: np.ndarray, center: float) -> np.ndarray:
    return np.abs((longitude - center + 180.0) % 360.0 - 180.0)


def select_native_altitudes(altitudes: np.ndarray) -> dict[float, tuple[int, float]]:
    """Select a native altitude for every target or reject a non-nearby channel."""
    selected: dict[float, tuple[int, float]] = {}
    for target in TARGET_ALTITUDES_KM:
        index = int(np.argmin(np.abs(altitudes - target)))
        actual = float(altitudes[index])
        if abs(actual - target) > ALTITUDE_TOLERANCE_KM:
            raise ValueError(
                f"No native SABER altitude within 0.5 km of {target:g} km."
            )
        selected[target] = (index, actual)
    return selected


def _file_day(ds) -> date:
    if "date" in ds.variables:
        token = str(int(np.asarray(ds.variables["date"].data).flat[0]))
        return date(int(token[:4]), 1, 1) + timedelta(days=int(token[4:]) - 1)
    return date(int(np.asarray(ds.variables["year"].data).item()), 1, 1) + timedelta(
        days=int(np.asarray(ds.variables["day"].data).item()) - 1
    )


def _matrix(ds, name: str, scans: int) -> np.ndarray:
    values = np.asarray(ds.variables[name].data, dtype=float)
    return values.T if values.shape[0] != scans and values.shape[1] == scans else values


def _observations(
    path: Path, variable: str, prefix: str, *, convert_proxy: bool = False
) -> tuple[list[dict], date, set[int]]:
    with netcdf_file(path, "r", mmap=False) as ds:
        day = _file_day(ds)
        values = np.asarray(ds.variables[variable].data, dtype=float)
        scans = values.shape[0]
        altitude_name = "altitude" if "altitude" in ds.variables else "tpaltitude"
        altitudes = np.asarray(ds.variables[altitude_name].data, dtype=float)
        selected = select_native_altitudes(altitudes) if altitudes.ndim == 1 else None
        latitudes = _matrix(ds, "tplatitude", scans)
        longitudes = _matrix(ds, "tplongitude", scans) % 360.0
        times = _matrix(ds, "time", scans)
        values = _matrix(ds, variable, scans)
        altitudes = (
            _matrix(ds, altitude_name, scans) if altitudes.ndim == 2 else altitudes
        )
        orbits = (
            np.asarray(ds.variables.get("orbit").data, dtype=int)
            if "orbit" in ds.variables
            else np.full(scans, -1)
        )
    rows: list[dict] = []
    for target in TARGET_ALTITUDES_KM:
        for scan in range(scans):
            if selected is None:
                altitude_index = int(np.argmin(np.abs(altitudes[scan] - target)))
                actual_altitude = float(altitudes[scan, altitude_index])
                if abs(actual_altitude - target) > ALTITUDE_TOLERANCE_KM:
                    continue
            else:
                altitude_index, actual_altitude = selected[target]
            value = values[scan, altitude_index]
            latitude, longitude, milliseconds = (
                latitudes[scan, altitude_index],
                longitudes[scan, altitude_index],
                times[scan, altitude_index],
            )
            if (
                not (
                    np.isfinite(value)
                    and np.isfinite(latitude)
                    and np.isfinite(longitude)
                    and np.isfinite(milliseconds)
                )
                or value <= MISSING_VALUE_LIMIT
            ):
                continue
            rows.append(
                {
                    "timestamp": datetime.combine(day, datetime.min.time())
                    + timedelta(milliseconds=float(milliseconds)),
                    "latitude": float(latitude),
                    "longitude": float(longitude),
                    "value": float(value) * (0.1 if convert_proxy else 1.0),
                    "column": f"{prefix}_{int(target)}km_w_m3",
                    "actual_altitude_km": actual_altitude,
                    "orbit": int(orbits[scan]),
                }
            )
    return rows, day, set(int(value) for value in orbits if value >= 0)


def _finite_geolocation_time_footprints(
    path: Path, variable: str
) -> tuple[date, list[dict]]:
    """Return valid scan footprints without conditioning on the cooling value."""
    with netcdf_file(path, "r", mmap=False) as ds:
        day = _file_day(ds)
        scans = np.asarray(ds.variables[variable].data).shape[0]
        altitude_name = "altitude" if "altitude" in ds.variables else "tpaltitude"
        altitudes = np.asarray(ds.variables[altitude_name].data, dtype=float)
        selected = select_native_altitudes(altitudes) if altitudes.ndim == 1 else None
        latitudes = _matrix(ds, "tplatitude", scans)
        longitudes = _matrix(ds, "tplongitude", scans) % 360.0
        times = _matrix(ds, "time", scans)
        orbits = (
            np.asarray(ds.variables["orbit"].data, dtype=int)
            if "orbit" in ds.variables
            else np.full(scans, -1)
        )
    footprints = []
    for scan, orbit in enumerate(orbits):
        if orbit < 0:
            continue
        for target in TARGET_ALTITUDES_KM:
            if selected is None:
                altitude_index = int(np.argmin(np.abs(altitudes[scan] - target)))
                if (
                    abs(float(altitudes[scan, altitude_index]) - target)
                    > ALTITUDE_TOLERANCE_KM
                ):
                    continue
            else:
                altitude_index, _actual_altitude = selected[target]
            latitude, longitude, milliseconds = (
                latitudes[scan, altitude_index],
                longitudes[scan, altitude_index],
                times[scan, altitude_index],
            )
            if not all(
                np.isfinite(value) for value in (latitude, longitude, milliseconds)
            ):
                continue
            footprints.append(
                {
                    "timestamp": datetime.combine(day, datetime.min.time())
                    + timedelta(milliseconds=float(milliseconds)),
                    "latitude": float(latitude),
                    "longitude": float(longitude),
                    "orbit": int(orbit),
                }
            )
    return day, footprints


def derive_hasdm_grid(hasdm_source: Path) -> tuple[float, float, float]:
    source = pl.scan_parquet(hasdm_source)
    timestamp = source.select(pl.col("timestamp").min()).collect().item()
    altitude = (
        source.filter(pl.col("timestamp") == timestamp)
        .select(pl.col("Altitude (m)").min())
        .collect()
        .item()
    )
    grid = (
        source.filter(
            (pl.col("timestamp") == timestamp) & (pl.col("Altitude (m)") == altitude)
        )
        .select("Latitude (deg)", "Longitude (deg)")
        .unique()
        .collect()
    )
    latitudes = np.sort(
        source.select(pl.col("Latitude (deg)").unique())
        .collect()
        .to_series()
        .to_numpy()
    )
    longitudes = np.sort(
        grid["Longitude (deg)"].cast(pl.Float64).unique().to_numpy() % 360.0
    )
    latitude_spacing = float(np.median(np.diff(latitudes)))
    longitude_spacing = float(
        np.median(np.diff(np.r_[longitudes, longitudes[0] + 360.0]))
    )
    if not np.isclose(latitude_spacing, 10.0) or not np.isclose(
        longitude_spacing, 15.0
    ):
        raise ValueError(
            f"Unexpected HASDM grid spacing: {latitude_spacing:g} x {longitude_spacing:g} degrees."
        )
    if (
        len(grid) != len(latitudes) * 24
        or len(latitudes) != 19
        or len(longitudes) != 24
    ):
        raise ValueError(
            "HASDM reference slice does not have the 19x24 native-grid contract."
        )
    return (
        float(latitudes[np.argmin(np.abs(latitudes - MAUNA_LOA_LATITUDE))]),
        latitude_spacing,
        longitude_spacing,
    )


def hasdm_longitudes_by_timestamp(samples_path: Path) -> dict[datetime, float]:
    samples = pl.read_parquet(samples_path).select("timestamp", "Longitude (deg)")
    counts = samples.group_by("timestamp").agg(
        pl.col("Longitude (deg)").n_unique().alias("n")
    )
    if counts.filter(pl.col("n") != 1).height:
        raise ValueError(
            "Frozen direct-density analysis samples must have exactly one longitude per timestamp."
        )
    return {
        timestamp: float(longitude) % 360.0
        for timestamp, longitude in samples.unique(subset="timestamp").iter_rows()
    }


def _calendar_slots(start: date, end: date | datetime) -> list[datetime]:
    """Return 3-hour slots through an exact datetime or an inclusive end date."""
    current = datetime.combine(start, datetime.min.time())
    if isinstance(end, datetime):
        if (
            end.tzinfo is not None
            or end.minute
            or end.second
            or end.microsecond
            or end.hour % 3
        ):
            raise ValueError(
                "Calendar datetime end must be a timezone-naive 3-hour slot."
            )
        if end < current:
            raise ValueError("Calendar end precedes its start date.")
        stop, inclusive = end, True
    else:
        stop, inclusive = (
            datetime.combine(end + timedelta(days=1), datetime.min.time()),
            False,
        )
    slots = []
    while current <= stop if inclusive else current < stop:
        slots.append(current)
        current += timedelta(hours=3)
    return slots


def _require_longitude_coverage(
    calendar_slots: list[datetime], grid_longitudes: dict[datetime, float]
) -> None:
    if not grid_longitudes:
        raise ValueError(
            "HASDM longitude coverage is empty for the requested calendar."
        )
    available_start, available_end = min(grid_longitudes), max(grid_longitudes)
    if calendar_slots[0] < available_start:
        raise ValueError(
            "Requested calendar starts before available HASDM longitude timestamp "
            f"{available_start}."
        )
    if calendar_slots[-1] > available_end:
        raise ValueError(
            "Requested calendar ends after available HASDM longitude timestamp "
            f"{available_end}."
        )


def _end_argument(value: str) -> date | datetime:
    """Parse date-only CLI values as full days and datetimes as exact endpoints."""
    return (
        datetime.fromisoformat(value)
        if "T" in value or " " in value
        else date.fromisoformat(value)
    )


def _bin(timestamp: datetime) -> datetime:
    return timestamp.replace(
        hour=timestamp.hour - timestamp.hour % 3, minute=0, second=0, microsecond=0
    )


def required_l2a_orbits(
    co2_files: list[Path],
    no_files: list[Path],
    grid_longitudes: dict[datetime, float],
    latitude_center: float,
    latitude_spacing: float,
    longitude_spacing: float,
) -> dict[date, set[int]]:
    """Select cooling-supported orbits from CO2/NO finite spatial-time footprints."""
    required: dict[date, set[int]] = defaultdict(set)
    for files, variable in ((co2_files, "CO2cool"), (no_files, "NOcool")):
        for path in files:
            day, footprints = _finite_geolocation_time_footprints(path, variable)
            for row in footprints:
                center = grid_longitudes.get(_bin(row["timestamp"]))
                if (
                    center is not None
                    and abs(row["latitude"] - latitude_center) <= latitude_spacing / 2
                    and circular_longitude_delta(np.array([row["longitude"]]), center)[
                        0
                    ]
                    <= longitude_spacing / 2
                ):
                    required[day].add(row["orbit"])
    return dict(required)


def _coverage(
    directory: Path, pattern: re.Pattern[str], start: date, end: date, source: str
) -> list[Path]:
    """Require every officially inventoried requested daily archive locally."""
    inventory_path = directory / "remote_inventory.json"
    manifest_path = directory / "manifest.json"
    if not inventory_path.exists():
        raise RuntimeError(
            f"{source} requires remote inventory {inventory_path}; re-run the downloader."
        )
    if not manifest_path.exists():
        raise RuntimeError(f"{source} requires downloader manifest {manifest_path}.")
    try:
        inventory = json.loads(inventory_path.read_text())
        if not isinstance(inventory["dataset"], str) or not isinstance(
            inventory["source_url"], str
        ):
            raise ValueError("missing inventory provenance")
        datetime.fromisoformat(inventory["discovered_at"])
        available_through = date.fromisoformat(inventory["available_through"])
        inventory_rows = inventory["files"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Invalid {source} remote inventory; re-run the downloader."
        ) from error
    if end >= MISSION_START and available_through < end:
        raise RuntimeError(
            f"Incomplete {source} inventory; requested through {end}, available through "
            f"{available_through}. Re-run the downloader."
        )
    official: dict[date, str] = {}
    try:
        for row in inventory_rows:
            day, filename = date.fromisoformat(row["date"]), row["filename"]
            if not pattern.fullmatch(filename) or day in official:
                raise ValueError(filename)
            official[day] = filename
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"Invalid {source} remote inventory; re-run the downloader."
        ) from error
    if not official or available_through != max(official):
        raise RuntimeError(f"Invalid {source} remote inventory; re-run the downloader.")
    manifest_entries: dict[str, list[dict]] = defaultdict(list)
    for entry in json.loads(manifest_path.read_text()).get("entries", []):
        manifest_entries[Path(entry.get("path", "")).name].append(entry)
    paths = []
    for day, filename in sorted(official.items()):
        if not (start <= day <= end):
            continue
        matching, path = manifest_entries[filename], directory / filename
        if (
            len(matching) != 1
            or matching[0].get("status") not in {"downloaded", "skipped"}
            or not path.is_file()
            or path.stat().st_size == 0
        ):
            raise RuntimeError(
                f"Incomplete {source} coverage; failed or missing {filename}."
            )
        paths.append(path)
    return paths


def _required_l2a_files(
    directory: Path, required: dict[date, set[int]]
) -> dict[tuple[date, int], Path]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Level2A requires downloader manifest {manifest_path}.")
    manifest_files: list[tuple[date, int, str, str, dict]] = []
    for entry in json.loads(manifest_path.read_text()).get("entries", []):
        filename = Path(entry.get("path", "")).name
        match = re.fullmatch(r"SABER_L2A_(\d{4})(\d{3})_(\d+)_(02\.\d+)\.nc", filename)
        if match:
            day = date(int(match.group(1)), 1, 1) + timedelta(
                days=int(match.group(2)) - 1
            )
            manifest_files.append(
                (day, int(match.group(3)), match.group(4), filename, entry)
            )
    local_files: list[tuple[date, int, str, str]] = []
    for path in directory.glob("SABER_L2A_*_*.nc"):
        match = re.fullmatch(r"SABER_L2A_(\d{4})(\d{3})_(\d+)_(02\.\d+)\.nc", path.name)
        if match:
            day = date(int(match.group(1)), 1, 1) + timedelta(
                days=int(match.group(2)) - 1
            )
            local_files.append((day, int(match.group(3)), match.group(4), path.name))
    selected = {}
    for day, orbits in required.items():
        version = "02.07" if day < date(2019, 12, 15) else "02.08"
        token = f"{day.year:04d}{day.timetuple().tm_yday:03d}"
        candidate_days = {day - timedelta(days=1), day, day + timedelta(days=1)}
        for orbit in orbits:
            expected, pair = f"SABER_L2A_{token}_{orbit:05d}_{version}.nc", (day, orbit)
            matching = [
                (file_day, file_version, filename, entry)
                for file_day, file_orbit, file_version, filename, entry in manifest_files
                if file_day in candidate_days and file_orbit == orbit
            ]
            local = [
                filename
                for file_day, file_orbit, _file_version, filename in local_files
                if file_day in candidate_days and file_orbit == orbit
            ]
            if (
                len(matching) != 1
                or matching[0][1]
                != ("02.07" if matching[0][0] < date(2019, 12, 15) else "02.08")
                or matching[0][3].get("status") not in {"downloaded", "skipped"}
                or local != [matching[0][2]]
            ):
                raise RuntimeError(
                    f"Incomplete Level2A coverage for {pair}; require exactly one "
                    f"manifested official {expected}."
                )
            path = directory / matching[0][2]
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(
                    f"Incomplete Level2A coverage; failed or missing {expected}."
                )
            selected[pair] = path
    return selected


def prepare_hasdm_saber_3hour(
    *,
    co2_dir: Path,
    no_dir: Path,
    l2a_dir: Path,
    hasdm_samples: Path,
    hasdm_source: Path,
    output: Path,
    start: date = date(2000, 1, 1),
    end: date | datetime = DEFAULT_END,
) -> pl.DataFrame:
    """Write all calendar bins with unfilled SABER means and observation counts."""
    latitude_center, latitude_spacing, longitude_spacing = derive_hasdm_grid(
        hasdm_source
    )
    grid_longitudes = hasdm_longitudes_by_timestamp(hasdm_samples)
    calendar_slots = _calendar_slots(start, end)
    _require_longitude_coverage(calendar_slots, grid_longitudes)
    end_date = end.date() if isinstance(end, datetime) else end
    co2_files = _coverage(co2_dir, CO2_FILE_RE, start, end_date, "CO2 cooling")
    no_files = _coverage(no_dir, NO_FILE_RE, start, end_date, "NO cooling")
    required = required_l2a_orbits(
        co2_files,
        no_files,
        grid_longitudes,
        latitude_center,
        latitude_spacing,
        longitude_spacing,
    )
    l2a_by_day_orbit = _required_l2a_files(l2a_dir, required)
    buckets: dict[tuple[datetime, str], list[float]] = defaultdict(list)
    actual_altitudes: dict[str, set[float]] = defaultdict(set)

    def add_observations(path: Path, variable: str, prefix: str, proxy: bool) -> None:
        for row in _observations(path, variable, prefix, convert_proxy=proxy)[0]:
            timestamp = _bin(row["timestamp"])
            longitude = grid_longitudes.get(timestamp)
            if (
                longitude is None
                or abs(row["latitude"] - latitude_center) > latitude_spacing / 2
                or circular_longitude_delta(np.array([row["longitude"]]), longitude)[0]
                > longitude_spacing / 2
            ):
                continue
            buckets[(timestamp, row["column"])].append(row["value"])
            actual_altitudes[row["column"]].add(row["actual_altitude_km"])

    for path, variable, prefix, proxy in [
        (path, "CO2cool", "saber_co2cool", False) for path in co2_files
    ] + [(path, "NOcool", "saber_nocool", False) for path in no_files]:
        add_observations(path, variable, prefix, proxy)
    for path in sorted(
        {
            l2a_by_day_orbit[(day, orbit)]
            for day, orbits in required.items()
            for orbit in orbits
        }
    ):
        for variable, prefix in PROXY_PRODUCTS.items():
            add_observations(path, variable, prefix, True)
    columns = [
        f"{prefix}_{int(altitude)}km_w_m3"
        for prefix in [*DIRECT_PRODUCTS.values(), *PROXY_PRODUCTS.values()]
        for altitude in TARGET_ALTITUDES_KM
    ]
    records = []
    for timestamp in calendar_slots:
        record: dict[str, object] = {
            "timestamp": timestamp,
            "hasdm_longitude_deg_east": grid_longitudes.get(timestamp),
        }
        for column in columns:
            values = buckets[(timestamp, column)]
            record[column] = float(np.mean(values)) if values else None
            record[f"{column}_observations"] = len(values)
        records.append(record)
    result = pl.DataFrame(
        records,
        schema={
            "timestamp": pl.Datetime,
            "hasdm_longitude_deg_east": pl.Float64,
            **{column: pl.Float64 for column in columns},
            **{f"{column}_observations": pl.Int64 for column in columns},
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(output, compression="lz4")
    output.with_suffix(".provenance.json").write_text(
        json.dumps(
            {
                "hasdm_latitude_center_deg": latitude_center,
                "hasdm_latitude_spacing_deg": latitude_spacing,
                "hasdm_longitude_spacing_deg": longitude_spacing,
                "latitude_bounds_deg": [
                    latitude_center - latitude_spacing / 2,
                    latitude_center + latitude_spacing / 2,
                ],
                "longitude_policy": "circular center +/- half decoded HASDM spacing; center is direct-density sample longitude per 3-hour timestamp",
                "l2a_orbit_selection": "union of finite geolocation/time footprints from CO2cool and NOcool daily archives within the HASDM spatial-time sample; cooling values are not required finite",
                "altitude_policy": "nearest native sample within 0.5 km; actual values by column",
                "actual_altitudes_km": {
                    key: sorted(value) for key, value in actual_altitudes.items()
                },
                "proxy_conversion": "O2_1delta_ver, OH_16_ver, and OH_20_ver multiplied by 0.1 from ergs/cm3/sec to W/m^3",
                "missing_policy": "calendar slots and empty SABER bins are null; no filling or interpolation",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--co2-dir", type=Path, default=Path("data/original/saber/co2_cooling_profiles")
    )
    parser.add_argument(
        "--no-dir", type=Path, default=Path("data/original/saber/no_cooling_profiles")
    )
    parser.add_argument(
        "--l2a-dir", type=Path, default=Path("data/original/saber/level2a")
    )
    parser.add_argument(
        "--hasdm-samples",
        type=Path,
        default=Path(
            "outputs/figures/results/hasdm_msis_model_errors/data/hasdm_msis_errors_nearest_timestamp_grid_samples.parquet"
        ),
    )
    parser.add_argument(
        "--hasdm-source",
        type=Path,
        default=Path("data/decoded/hasdm/HASDM_2000_merged.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/decoded/saber/saber_hasdm_maunaloa_3hour.parquet"),
    )
    parser.add_argument("--start", type=date.fromisoformat, default=date(2000, 1, 1))
    parser.add_argument(
        "--end",
        type=_end_argument,
        default=DEFAULT_END,
        help="Inclusive date or exact timezone-naive 3-hour datetime endpoint.",
    )
    args = parser.parse_args()
    prepare_hasdm_saber_3hour(**vars(args))


if __name__ == "__main__":
    main()
