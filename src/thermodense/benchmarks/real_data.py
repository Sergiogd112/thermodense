"""Assemble the real-data five-node daily input for the PCMCI benchmark.

The five-node graph is: daily mean log10 HASDM density at 325 km and
825 km (from the **Mauna Loa HASDM subset**), Mauna Loa tropospheric CO2,
F10.7 (81-day centred solar proxy), and Ap (daily geomagnetic average).

Assembly follows the thesis language in CONTEXT.md:

- **Mauna Loa HASDM subset**: nearest available HASDM latitude, with the
  nearest available longitude selected independently for each timestamp
  and altitude (no spatial interpolation).
- **Bounded gap interpolation**: linear fill only for gaps of at most
  ``max_gap_steps`` bounded by real samples on both sides; longer gaps and
  edges stay missing.

Run as ``python -m thermodense.benchmarks.real_data`` (or import the
functions); the CSV is written once and then consumed by the frozen
benchmark harness via ``run-real``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from thermodense.downloader.space_weather import SPACE_WEATHER_CSV_PATH

REPO_ROOT = Path(__file__).resolve().parents[3]

MAUNA_LOA_LAT = 19.5362
MAUNA_LOA_LON_EAST = 204.4237  # -155.5763 deg W

HASDM_LAT_COL = "Latitude (deg)"
HASDM_LON_COL = "Longitude (deg)"
HASDM_ALT_COL = "Altitude (m)"
HASDM_DENSITY_COL = "Density (kg/m^3)"
MAX_VALID_HASDM_DENSITY = 1.0e-8

ALTITUDES_KM = [325, 825]
MAX_GAP_STEPS = 5
MAX_LAG_DAYS = 180

DECODED_HASDM_DIR = REPO_ROOT / "data/decoded/hasdm"
CO2_PATH = REPO_ROOT / "data/original/co2/co2_daily_mlo.csv"
SPACE_WEATHER_PATH = REPO_ROOT / SPACE_WEATHER_CSV_PATH
DEFAULT_OUTPUT = REPO_ROOT / "data/products/causal_discovery/five_node_daily.csv"

# Final column order in the five-node CSV. The two density channels are the
# PCMCI altitude group [325, 825]; CO2 is the explanatory target of interest.
NODE_COLUMNS = [
    "f10_7_center81",
    "ap_avg",
    "co2_ppm",
    "log10rho_325_daily_mean",
    "log10rho_825_daily_mean",
]
TARGET_COLUMNS = [
    "log10rho_325_daily_mean",
    "log10rho_825_daily_mean",
]
DATE_COLUMN = "date"
IMPUTATION_MASK_COLUMNS = [f"{column}_imputed" for column in NODE_COLUMNS]


def circular_lon_delta_expr(lon_col: str, target_lon: float) -> pl.Expr:
    return ((pl.col(lon_col) - target_lon + 180.0) % 360.0 - 180.0).abs()


def nearest_hasdm_latitude(decoded_dir: Path = DECODED_HASDM_DIR) -> float:
    paths = sorted(decoded_dir.glob("HASDM_*_merged.parquet"))
    if not paths:
        raise FileNotFoundError(f"No decoded HASDM parquet files in {decoded_dir}")
    latitudes = (
        pl.scan_parquet(str(paths[0]))
        .select(pl.col(HASDM_LAT_COL).unique())
        .collect()[HASDM_LAT_COL]
        .to_numpy()
        .astype(float)
    )
    return float(latitudes[np.argmin(np.abs(latitudes - MAUNA_LOA_LAT))])


def daily_hasdm_for_path(path: Path, grid_lat: float) -> pl.DataFrame:
    """Per (date, altitude_km) daily mean of rho with log10 columns.

    Mirrors the thesis pipeline's HASDM Mauna Loa subset: filter to the
    nearest latitude and valid densities, pick the nearest longitude per
    timestamp and altitude, then aggregate daily.
    """
    lf = pl.scan_parquet(str(path)).filter(
        (pl.col(HASDM_LAT_COL) == grid_lat)
        & (pl.col(HASDM_DENSITY_COL) > 0)
        & (pl.col(HASDM_DENSITY_COL) <= MAX_VALID_HASDM_DENSITY)
        & (pl.col(HASDM_ALT_COL) / 1000.0).is_in(ALTITUDES_KM)
    )
    selected = (
        lf.with_columns(
            circular_lon_delta_expr(HASDM_LON_COL, MAUNA_LOA_LON_EAST).alias(
                "lon_delta"
            )
        )
        .with_columns(
            pl.min("lon_delta")
            .over(["timestamp", HASDM_ALT_COL])
            .alias("nearest_lon_delta")
        )
        .filter(pl.col("lon_delta") == pl.col("nearest_lon_delta"))
        .group_by(["timestamp", HASDM_ALT_COL])
        .agg(pl.col(HASDM_DENSITY_COL).first().alias("rho_grid"))
        .select(
            "timestamp",
            (pl.col(HASDM_ALT_COL) / 1000.0).alias("altitude_km"),
            "rho_grid",
        )
    )
    daily = (
        selected.with_columns(pl.col("timestamp").dt.date().alias("date"))
        .group_by(["date", "altitude_km"])
        .agg(
            pl.col("rho_grid").mean().alias("rho_daily_mean"),
            pl.len().alias("n_samples"),
        )
        .with_columns(pl.col("rho_daily_mean").log10().alias("log10rho_daily_mean"))
    )
    return daily.collect()


def build_hasdm_maunaloa_daily(decoded_dir: Path) -> pl.DataFrame:
    """Wide daily frame: date + log10rho_{alt}_daily_mean for each altitude."""
    grid_lat = nearest_hasdm_latitude(decoded_dir)
    paths = sorted(decoded_dir.glob("HASDM_*_merged.parquet"))
    frames = [daily_hasdm_for_path(path, grid_lat) for path in paths]
    long_df = pl.concat(frames).sort(["date", "altitude_km"])
    altitudes = [int(a) for a in long_df["altitude_km"].unique().sort().to_list()]
    pivot = (
        long_df.with_columns(
            pl.col("altitude_km").cast(pl.Int64).cast(pl.Utf8).alias("altitude_label")
        )
        .select("date", "altitude_label", "log10rho_daily_mean")
        .pivot(
            index="date",
            on="altitude_label",
            values="log10rho_daily_mean",
            aggregate_function="first",
        )
    )
    rename = {
        str(alt): f"log10rho_{alt}_daily_mean"
        for alt in altitudes
        if str(alt) in pivot.columns
    }
    pivot = pivot.rename(rename)
    wide = pivot.with_columns(pl.col("date").cast(pl.Date)).sort("date")
    available = [
        alt for alt in ALTITUDES_KM if f"log10rho_{alt}_daily_mean" in wide.columns
    ]
    if set(available) != set(ALTITUDES_KM):
        raise RuntimeError(
            f"Expected HASDM altitudes {ALTITUDES_KM}, found {sorted(altitudes)}"
        )
    return wide.select("date", *[f"log10rho_{alt}_daily_mean" for alt in ALTITUDES_KM])


def load_co2_daily(path: Path = CO2_PATH) -> pl.DataFrame:
    schema = {
        "year": pl.Int32,
        "month": pl.Int32,
        "day": pl.Int32,
        "year_decimal": pl.Float32,
        "CO2_ppm": pl.Float64,
    }
    return (
        pl.read_csv(path, has_header=False, schema=schema, comment_prefix="#")
        .with_columns(pl.date("year", "month", "day").alias("date"))
        .select("date", pl.col("CO2_ppm").alias("co2_ppm"))
        .with_columns(
            pl.when(pl.col("co2_ppm") < 0)
            .then(None)
            .otherwise(pl.col("co2_ppm"))
            .alias("co2_ppm")
        )
    )


def load_space_weather_daily(path: Path = SPACE_WEATHER_PATH) -> pl.DataFrame:
    return (
        pl.read_csv(path)
        .with_columns(pl.col("DATE").str.to_date("%Y-%m-%d").alias("date"))
        .select(
            "date",
            pl.col("F10.7_OBS_CENTER81").alias("f10_7_center81"),
            pl.col("AP_AVG").cast(pl.Float64).alias("ap_avg"),
        )
    )


def as_date_index(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(pl.col("date").cast(pl.Date))
        .filter(pl.col("date").is_not_null())
        .unique(subset="date")
        .sort("date")
    )


def explicit_daily_coverage(hasdm: pl.DataFrame) -> pl.DataFrame:
    """Expose absent days between the first and last HASDM observations."""
    if hasdm.is_empty():
        return hasdm
    dates = pl.DataFrame(
        {
            DATE_COLUMN: pl.date_range(
                hasdm[DATE_COLUMN].min(),
                hasdm[DATE_COLUMN].max(),
                "1d",
                eager=True,
            )
        }
    )
    return dates.join(hasdm, on=DATE_COLUMN, how="left")


def bounded_gap_fill(values: np.ndarray, max_gap: int) -> tuple[np.ndarray, np.ndarray]:
    """Fill bounded interior NaN runs and return values plus an imputation mask."""
    out = np.asarray(values, dtype=float).copy()
    imputed = np.zeros(len(out), dtype=bool)
    n = len(out)
    index = 0
    while index < n:
        if not np.isnan(out[index]):
            index += 1
            continue
        start = index
        while index < n and np.isnan(out[index]):
            index += 1
        if index - start <= max_gap and start > 0 and index < n:
            left, right = out[start - 1], out[index]
            for k in range(start, index):
                weight = (k - (start - 1)) / (index - (start - 1))
                out[k] = left + (right - left) * weight
                imputed[k] = True
    return out, imputed


def build_five_node_daily(
    decoded_dir: Path = DECODED_HASDM_DIR,
    sw_path: Path = SPACE_WEATHER_PATH,
    co2_path: Path = CO2_PATH,
) -> pl.DataFrame:
    hasdm = explicit_daily_coverage(
        as_date_index(build_hasdm_maunaloa_daily(decoded_dir))
    )
    space_weather = as_date_index(load_space_weather_daily(sw_path))
    co2 = as_date_index(load_co2_daily(co2_path))
    # HASDM defines the benchmark's coverage.  Driver-only history must not
    # create rows outside that window.
    combined = hasdm.join(space_weather, on="date", how="left")
    combined = combined.join(co2, on="date", how="left")
    combined = combined.sort("date")

    filled = {DATE_COLUMN: combined[DATE_COLUMN]}
    for column in NODE_COLUMNS:
        values = combined[column].to_numpy().astype(float)
        if column in TARGET_COLUMNS:
            values, imputed = bounded_gap_fill(values, MAX_GAP_STEPS)
        else:
            imputed = np.zeros(len(values), dtype=bool)
        filled[column] = values
        filled[f"{column}_imputed"] = imputed
    result = pl.DataFrame(filled).select(
        DATE_COLUMN, *NODE_COLUMNS, *IMPUTATION_MASK_COLUMNS
    )
    return result.with_columns(pl.col(DATE_COLUMN).cast(pl.Date))


def describe(df: pl.DataFrame) -> str:
    missing = [
        pl.col(column).is_null() | pl.col(column).is_nan() for column in NODE_COLUMNS
    ]
    complete = df.filter(~pl.any_horizontal(missing))
    lines = [
        f"rows: {len(df)} (complete rows: {len(complete)})",
        f"date range: {df['date'].min()} .. {df['date'].max()}",
    ]
    for column in NODE_COLUMNS:
        missing_count = df.select(
            (pl.col(column).is_null() | pl.col(column).is_nan()).sum()
        ).item()
        imputed_count = (
            int(df[f"{column}_imputed"].sum())
            if f"{column}_imputed" in df.columns
            else 0
        )
        lines.append(
            f"  {column}: {len(df) - missing_count}/{len(df)} present; "
            f"{imputed_count} imputed"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m thermodense.benchmarks.real_data")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--decoded-hasdm-dir", type=Path, default=DECODED_HASDM_DIR)
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame = build_five_node_daily(args.decoded_hasdm_dir)
    frame.write_csv(args.output)
    print(args.output)
    print(describe(frame))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
