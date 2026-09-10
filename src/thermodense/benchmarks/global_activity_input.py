"""Immutable daily global-mean input for retrospective activity associations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl

VERSION = "global-activity-daily-v1"
ALTITUDES_KM = (250, 275, 325, 375, 400, 425, 475, 525, 550, 575)
DRIVER_NAMES = ("F10.7_OBS", "AP_AVG", "KP_SUM", "CO2_ppm")
NODE_NAMES = [*(f"log10rho_{altitude}" for altitude in ALTITUDES_KM), *DRIVER_NAMES]
TRANSFORMATIONS = {"detrending": "none", "deseasonalizing": "none", "standardizing": "none", "interpolation": "none"}
ACTIVITY_CLASSIFICATION = "F10.7_OBS_CENTER81: low <100, moderate 100<=F<160, high >=160"


@dataclass(frozen=True)
class GlobalActivityInput:
    timestamps: np.ndarray
    values: np.ndarray
    available: np.ndarray
    activity_f107_center81: np.ndarray
    node_names: list[str]
    manifest: dict[str, Any]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **cast(dict[str, Any], arrays))
    temporary.replace(path)


def group_mask(activity: np.ndarray, group: str) -> np.ndarray:
    """Classify target dates; ``all`` intentionally includes unavailable dates."""
    values = np.asarray(activity, dtype=float)
    if group == "all":
        return np.ones(values.shape, dtype=bool)
    if group == "low":
        return np.isfinite(values) & (values < 100)
    if group == "moderate":
        return np.isfinite(values) & (values >= 100) & (values < 160)
    if group == "high":
        return np.isfinite(values) & (values >= 160)
    raise ValueError("group must be one of all, low, moderate, high")


def _dates(frame: pl.DataFrame, source: str) -> dict[np.datetime64, dict[str, Any]]:
    if "date" not in frame.columns:
        raise ValueError(f"{source} requires a date column")
    result: dict[np.datetime64, dict[str, Any]] = {}
    for row in frame.to_dicts():
        date = np.datetime64(row["date"], "D")
        if np.isnat(date):
            raise ValueError(f"invalid {source} date")
        if date in result:
            raise ValueError(f"duplicate {source} date {date}")
        result[date] = row
    return result


def _finite(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def _co2_rows(path: Path) -> dict[np.datetime64, float]:
    result: dict[np.datetime64, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(line for line in handle if not line.lstrip().startswith("#")):
            if len(row) != 5:
                raise ValueError("MLO CO2 CSV requires five no-header columns")
            try:
                date = np.datetime64(f"{int(row[0]):04}-{int(row[1]):02}-{int(row[2]):02}", "D")
            except ValueError as error:
                raise ValueError("invalid MLO CO2 date") from error
            if date in result:
                raise ValueError(f"duplicate MLO CO2 date {date}")
            value = _finite(row[4])
            result[date] = np.nan if value < 0 else value
    return result


def assemble(density_parquet: Path, space_weather_csv: Path, co2_csv: Path, output: Path) -> Path:
    """Assemble local raw sources without filling, dropping, or transforming dates."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output {output}")
    density_parquet, space_weather_csv, co2_csv = (path.resolve() for path in (density_parquet, space_weather_csv, co2_csv))
    if not all(path.is_file() for path in (density_parquet, space_weather_csv, co2_csv)):
        raise FileNotFoundError("all three input sources must be local files")
    source_hashes = {path: sha256(path) for path in (density_parquet, space_weather_csv, co2_csv)}
    density = pl.read_parquet(density_parquet)
    required_density = {"date", *NODE_NAMES[:10]}
    if required_density - set(density.columns):
        raise ValueError(f"density parquet requires exactly these density columns: {NODE_NAMES[:10]}")
    density = density.select("date", *NODE_NAMES[:10])
    density_rows = _dates(density, "density")
    if not density_rows:
        raise ValueError("density parquet contains no dates")
    weather = pl.read_csv(space_weather_csv)
    weather_required = {"DATE", "F10.7_OBS", "F10.7_OBS_CENTER81", "AP_AVG", "KP_SUM"}
    if weather_required - set(weather.columns):
        raise ValueError("space-weather CSV requires DATE, F10.7_OBS, F10.7_OBS_CENTER81, AP_AVG, KP_SUM")
    weather = weather.select(pl.col("DATE").cast(pl.String).alias("date"), *sorted(weather_required - {"DATE"}))
    weather_rows = _dates(weather, "space-weather")
    co2_rows = _co2_rows(co2_csv)
    start, stop = min(density_rows), max(density_rows)
    timestamps = np.arange(start, stop + np.timedelta64(1, "D"), np.timedelta64(1, "D"), dtype="datetime64[D]")
    values = np.full((len(timestamps), len(NODE_NAMES)), np.nan, dtype=float)
    activity = np.full(len(timestamps), np.nan, dtype=float)
    for index, date in enumerate(timestamps):
        density_row = density_rows.get(date)
        if density_row is not None:
            for column, name in enumerate(NODE_NAMES[:10]):
                value = _finite(density_row[name])
                values[index, column] = np.nan if value < -200 else value
        weather_row = weather_rows.get(date)
        if weather_row is not None:
            for column, name in enumerate(DRIVER_NAMES[:3], start=10):
                values[index, column] = _finite(weather_row[name])
            activity[index] = _finite(weather_row["F10.7_OBS_CENTER81"])
        values[index, 13] = co2_rows.get(date, np.nan)
    available = np.isfinite(values)
    if any(sha256(path) != digest for path, digest in source_hashes.items()):
        raise ValueError("a source changed while the immutable input was assembled")
    output.mkdir(parents=True)
    _atomic_npz(output / "input.npz", timestamps=timestamps, values=values, available=available, activity_f107_center81=activity, node_names=np.asarray(NODE_NAMES))
    sources = {
        "density_parquet": {"path": str(density_parquet), "sha256": source_hashes[density_parquet], "schema": ["date", *NODE_NAMES[:10]]},
        "space_weather_csv": {"path": str(space_weather_csv), "sha256": source_hashes[space_weather_csv], "schema": ["DATE", "F10.7_OBS", "F10.7_OBS_CENTER81", "AP_AVG", "KP_SUM"]},
        "mlo_co2_csv": {"path": str(co2_csv), "sha256": source_hashes[co2_csv], "schema": ["year", "month", "day", "year_decimal", "CO2_ppm"]},
    }
    manifest = {"input_contract": VERSION, "core_npz": "input.npz", "core_sha256": sha256(output / "input.npz"), "node_names": NODE_NAMES, "calendar": {"cadence": "daily UTC", "start": str(start), "end": str(stop), "length": len(timestamps), "density_calendar": "full min..max reindex; no dropped dates"}, "transformations": TRANSFORMATIONS, "f107_fit": "F10.7_OBS raw", "activity_classification": ACTIVITY_CLASSIFICATION, "missing_data": {"density_less_than_-200": "unavailable legacy sentinel", "co2_negative": "unavailable", "interpolation": "none"}, "sources": sources, "provenance": {"module": str(Path(__file__).resolve()), "module_sha256": sha256(Path(__file__))}}
    atomic_json(output / "manifest.json", manifest)
    return output / "manifest.json"


def load_manifest(path: Path) -> GlobalActivityInput:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("input_contract") != VERSION or manifest.get("node_names") != NODE_NAMES:
        raise ValueError("manifest is not the global-activity daily input contract")
    for source in manifest.get("sources", {}).values():
        if not Path(source.get("path", "")).is_absolute() or not isinstance(source.get("sha256"), str):
            raise ValueError("manifest sources require absolute paths and recorded checksums")
    if manifest.get("transformations") != TRANSFORMATIONS or manifest.get("f107_fit") != "F10.7_OBS raw":
        raise ValueError("input must retain raw values without interpolation, detrending, or deseasonalizing")
    core = (path.parent / manifest.get("core_npz", "")).resolve()
    if core.parent != path.parent.resolve():
        raise ValueError("input NPZ must be alongside its manifest")
    if not core.is_file() or sha256(core) != manifest.get("core_sha256"):
        raise ValueError("input NPZ digest does not match manifest")
    with np.load(core, allow_pickle=False) as saved:
        timestamps_raw = np.asarray(saved["timestamps"])
        if not np.issubdtype(timestamps_raw.dtype, np.datetime64):
            raise ValueError("timestamps must retain datetime64 dtype")
        timestamps = timestamps_raw.astype("datetime64[D]")
        values = np.asarray(saved["values"], dtype=float)
        available = np.asarray(saved["available"])
        if available.dtype != np.dtype(bool):
            raise ValueError("available must retain boolean dtype")
        activity = np.asarray(saved["activity_f107_center81"], dtype=float)
        names = [str(item) for item in saved["node_names"].tolist()]
    if names != NODE_NAMES or timestamps.ndim != 1 or len(timestamps) == 0 or np.isnat(timestamps).any() or not np.all(np.diff(timestamps.astype("int64")) == 1):
        raise ValueError("input requires the exact nonempty consecutive daily UTC calendar and node order")
    if values.shape != (len(timestamps), 14) or available.shape != values.shape or activity.shape != (len(timestamps),):
        raise ValueError("input array shapes do not match the T x 14 daily contract")
    if not np.array_equal(available, np.isfinite(values)):
        raise ValueError("available must exactly equal finite raw values")
    calendar = {"cadence": "daily UTC", "start": str(timestamps[0]), "end": str(timestamps[-1]), "length": len(timestamps), "density_calendar": "full min..max reindex; no dropped dates"}
    if manifest.get("calendar") != calendar or manifest.get("activity_classification") != ACTIVITY_CLASSIFICATION:
        raise ValueError("manifest calendar or activity classification does not match input payload")
    return GlobalActivityInput(timestamps, values, available, activity, NODE_NAMES.copy(), manifest)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--density-parquet", type=Path, required=True)
    command.add_argument("--space-weather-csv", type=Path, required=True)
    command.add_argument("--co2-csv", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(assemble(args.density_parquet, args.space_weather_csv, args.co2_csv, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
