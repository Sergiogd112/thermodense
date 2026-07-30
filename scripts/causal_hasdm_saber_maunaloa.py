# Zed REPL-compatible Mauna Loa HASDM/SABER causal workflow.
# Cells are separated with # %% markers so they can be run incrementally.

# %%
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from tqdm import tqdm
from thermodense.downloader.space_weather import SPACE_WEATHER_CSV_PATH  # noqa: E402

# %%
MAUNA_LOA_LAT = 19.5362
MAUNA_LOA_LON_EAST = 204.4237
HASDM_LAT_COL = "Latitude (deg)"
HASDM_LON_COL = "Longitude (deg)"
HASDM_ALT_COL = "Altitude (m)"
HASDM_DENSITY_COL = "Density (kg/m^3)"
MAX_VALID_HASDM_DENSITY = 1.0e-8


def parse_env_date(name: str) -> date | None:
    value = os.environ.get(name)
    if not value:
        return None
    return date.fromisoformat(value)


ANALYSIS_START_DATE = parse_env_date("MAUNALOA_START_DATE")
ANALYSIS_END_DATE = parse_env_date("MAUNALOA_END_DATE")


def hasdm_year_from_path(path: Path) -> int:
    return int(path.stem.split("_")[1])


HASDM_PATHS = sorted(Path("data/decoded/hasdm").glob("HASDM_*_merged.parquet"))
if ANALYSIS_START_DATE is not None:
    HASDM_PATHS = [
        path
        for path in HASDM_PATHS
        if hasdm_year_from_path(path) >= ANALYSIS_START_DATE.year
    ]
if ANALYSIS_END_DATE is not None:
    HASDM_PATHS = [
        path
        for path in HASDM_PATHS
        if hasdm_year_from_path(path) <= ANALYSIS_END_DATE.year
    ]
SABER_PATH = Path("data/decoded/saber/saber_co2_cooling_maunaloa_daily.parquet")
CO2_PATH = Path("data/original/co2/co2_daily_mlo.csv")
SPACE_WEATHER_PATH = SPACE_WEATHER_CSV_PATH

OUTPUT_DIR = Path(
    "outputs/figures/results/set_hasdm/model_validations/causal_hasdm_saber_maunaloa"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

X_COL = "CO2_ppm"
Y_COLS = ["F10.7_OBS_CENTER81", "AP_AVG", "KP_SUM"]
DRIVER_COLS = [*Y_COLS, X_COL]
SABER_ALTITUDE_ROLE_COLS = {
    "min": "saber_co2cool_min_alt",
    "median": "saber_co2cool_median_alt",
    "max": "saber_co2cool_max_alt",
}
SABER_COLS = list(SABER_ALTITUDE_ROLE_COLS.values())
MIN_SAMPLES_PER_CELL = 5

MAX_LAG_DAYS = int(os.environ.get("MAUNALOA_MAX_LAG_DAYS", "180"))
BASE_LAGS = list(range(14)) + [14, 27, 54, 81, 120, 180]
PHYSICS_LAGS = [lag for lag in BASE_LAGS if lag <= MAX_LAG_DAYS]
ADJUSTMENT_LAGS = [lag for lag in BASE_LAGS if lag <= MAX_LAG_DAYS]
TARGET_AUTOREGRESSIVE_LAGS = [lag for lag in BASE_LAGS if lag <= MAX_LAG_DAYS]
BOOTSTRAPS = 50
BOOTSTRAP_BLOCK_DAYS = 27
USE_BLOCK_BOOTSTRAP = False
HAC_MAX_LAG_DAYS = 27
RANDOM_SEED = 42
RUN_ADJUSTED_EFFECTS = os.environ.get("MAUNALOA_RUN_ADJUSTED_EFFECTS", "0") == "1"


# %%
@dataclass
class Variant:
    name: str
    dates: np.ndarray
    data: dict[str, np.ndarray]
    description: str


def circular_lon_delta_expr(lon_col: str, target_lon: float) -> pl.Expr:
    return ((pl.col(lon_col) - target_lon + 180.0) % 360.0 - 180.0).abs()


def circular_lon_delta_np(lon: np.ndarray, target_lon: float) -> np.ndarray:
    return np.abs((lon - target_lon + 180.0) % 360.0 - 180.0)


def nearest_hasdm_latitude() -> float:
    if not HASDM_PATHS:
        raise FileNotFoundError("No decoded HASDM parquet files found.")
    latitudes = (
        pl.scan_parquet(str(HASDM_PATHS[0]))
        .select(pl.col(HASDM_LAT_COL).unique())
        .collect()[HASDM_LAT_COL]
        .to_numpy()
        .astype(float)
    )
    return float(latitudes[np.argmin(np.abs(latitudes - MAUNA_LOA_LAT))])


def nearest_hasdm_grid_point() -> tuple[float, float]:
    if not HASDM_PATHS:
        raise FileNotFoundError("No decoded HASDM parquet files found.")
    grid = (
        pl.scan_parquet(str(HASDM_PATHS[0]))
        .select(HASDM_LAT_COL, HASDM_LON_COL)
        .unique()
        .collect()
    )
    latitudes = grid[HASDM_LAT_COL].to_numpy().astype(float)
    longitudes = grid[HASDM_LON_COL].to_numpy().astype(float)
    lon_deltas = circular_lon_delta_np(longitudes, MAUNA_LOA_LON_EAST)
    # Use a simple local distance score; the HASDM grid is coarse enough that
    # this deterministically selects the nearest available grid cell.
    lat_scale = np.cos(np.deg2rad(MAUNA_LOA_LAT))
    distances = (latitudes - MAUNA_LOA_LAT) ** 2 + (lon_deltas * lat_scale) ** 2
    index = int(np.argmin(distances))
    return float(latitudes[index]), float(longitudes[index])


def hasdm_latitude_bounds() -> tuple[float, float]:
    if not HASDM_PATHS:
        raise FileNotFoundError("No decoded HASDM parquet files found.")
    latitudes = np.sort(
        pl.scan_parquet(str(HASDM_PATHS[0]))
        .select(pl.col(HASDM_LAT_COL).unique())
        .collect()[HASDM_LAT_COL]
        .to_numpy()
        .astype(float)
    )
    lower_candidates = latitudes[latitudes <= MAUNA_LOA_LAT]
    upper_candidates = latitudes[latitudes >= MAUNA_LOA_LAT]
    lower = (
        float(lower_candidates[-1]) if len(lower_candidates) else float(latitudes[0])
    )
    upper = (
        float(upper_candidates[0]) if len(upper_candidates) else float(latitudes[-1])
    )
    return lower, upper


def selected_altitudes_for_analysis(altitudes: list[int]) -> list[int]:
    """Pick min, lower-mid median, median, upper-mid median, and max altitudes."""
    values = np.asarray(sorted(altitudes), dtype=float)
    selected: list[int] = []
    for quantile in [0.0, 0.25, 0.5, 0.75, 1.0]:
        target = float(np.quantile(values, quantile))
        altitude = int(values[np.argmin(np.abs(values - target))])
        if altitude not in selected:
            selected.append(altitude)
    return selected


def density_cols_for_altitudes(altitudes: list[int]) -> list[str]:
    cols: list[str] = []
    for altitude in altitudes:
        cols.extend(
            [
                f"log10rho_{altitude}_daily_min",
                f"log10rho_{altitude}_daily_mean",
                f"log10rho_{altitude}_daily_max",
                f"log10rho_{altitude}_daily_range",
            ]
        )
    return cols


def density_mean_cols_for_altitudes(altitudes: list[int]) -> list[str]:
    return [f"log10rho_{altitude}_daily_mean" for altitude in altitudes]


def density_range_cols_for_altitudes(altitudes: list[int]) -> list[str]:
    return [f"log10rho_{altitude}_daily_range" for altitude in altitudes]


def daily_hasdm_for_path(path: Path, grid_lat: float) -> pl.DataFrame:
    start_timestamp = (
        datetime.combine(ANALYSIS_START_DATE, datetime.min.time())
        if ANALYSIS_START_DATE
        else None
    )
    end_timestamp = (
        datetime.combine(ANALYSIS_END_DATE, datetime.max.time())
        if ANALYSIS_END_DATE
        else None
    )
    lf = pl.scan_parquet(str(path)).filter(
        (pl.col(HASDM_LAT_COL) == grid_lat)
        & (pl.col(HASDM_DENSITY_COL) > 0)
        & (pl.col(HASDM_DENSITY_COL) <= MAX_VALID_HASDM_DENSITY)
    )
    if start_timestamp is not None:
        lf = lf.filter(pl.col("timestamp") >= start_timestamp)
    if end_timestamp is not None:
        lf = lf.filter(pl.col("timestamp") <= end_timestamp)

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
        .agg(
            pl.col(HASDM_DENSITY_COL).first().alias("rho_grid"),
        )
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
            pl.col("rho_grid").min().alias("rho_daily_min"),
            pl.col("rho_grid").mean().alias("rho_daily_mean"),
            pl.col("rho_grid").max().alias("rho_daily_max"),
            pl.len().alias("hasdm_samples"),
        )
        .with_columns(
            pl.col("rho_daily_min").log10().alias("log10rho_daily_min"),
            pl.col("rho_daily_mean").log10().alias("log10rho_daily_mean"),
            pl.col("rho_daily_max").log10().alias("log10rho_daily_max"),
            (pl.col("rho_daily_max").log10() - pl.col("rho_daily_min").log10()).alias(
                "log10rho_daily_range"
            ),
        )
    )
    return daily.collect()


def load_hasdm_maunaloa_daily() -> tuple[pl.DataFrame, list[str], list[int], list[int]]:
    grid_lat = nearest_hasdm_latitude()
    frames = [
        daily_hasdm_for_path(path, grid_lat)
        for path in tqdm(HASDM_PATHS, desc="HASDM daily")
    ]
    long_df = pl.concat(frames).sort(["date", "altitude_km"])
    long_df.write_parquet(
        OUTPUT_DIR / "hasdm_maunaloa_daily_long.parquet", compression="lz4"
    )

    altitudes = [int(alt) for alt in long_df["altitude_km"].unique().sort().to_list()]
    wide: pl.DataFrame | None = None
    all_density_cols: list[str] = []
    pivot_specs = [
        ("log10rho_daily_min", "log10rho_{altitude}_daily_min"),
        ("log10rho_daily_mean", "log10rho_{altitude}_daily_mean"),
        ("log10rho_daily_max", "log10rho_{altitude}_daily_max"),
        ("log10rho_daily_range", "log10rho_{altitude}_daily_range"),
    ]
    for value_col, output_template in pivot_specs:
        pivot = (
            long_df.with_columns(
                pl.col("altitude_km")
                .cast(pl.Int64)
                .cast(pl.Utf8)
                .alias("altitude_label")
            )
            .select("date", "altitude_label", value_col)
            .pivot(
                index="date",
                on="altitude_label",
                values=value_col,
                aggregate_function="first",
            )
        )
        rename_map = {
            str(altitude): output_template.format(altitude=altitude)
            for altitude in altitudes
            if str(altitude) in pivot.columns
        }
        pivot = pivot.rename(rename_map)
        all_density_cols.extend(rename_map.values())
        wide = (
            pivot
            if wide is None
            else wide.join(pivot, on="date", how="full", coalesce=True)
        )

    if wide is None:
        raise RuntimeError("No HASDM Mauna Loa data loaded.")
    wide = wide.sort("date")
    wide.write_parquet(
        OUTPUT_DIR / "hasdm_maunaloa_daily_wide.parquet", compression="lz4"
    )
    selected_altitudes = selected_altitudes_for_analysis(altitudes)
    selected_density_cols = [
        col
        for col in density_cols_for_altitudes(selected_altitudes)
        if col in all_density_cols
    ]
    (OUTPUT_DIR / "hasdm_selection.txt").write_text(
        f"Mauna Loa lat={MAUNA_LOA_LAT}, lon_east={MAUNA_LOA_LON_EAST}\n"
        f"Nearest HASDM latitude={grid_lat}; nearest available longitude is selected independently for each timestamp and altitude\n"
        f"HASDM densities > {MAX_VALID_HASDM_DENSITY:.1e} kg/m^3 are treated as invalid before aggregation\n"
        f"HASDM altitude km={altitudes}\n"
        f"Selected HASDM altitude km for analysis={selected_altitudes}\n",
        encoding="utf-8",
    )
    return wide, selected_density_cols, altitudes, selected_altitudes


def load_saber_maunaloa_daily() -> tuple[pl.DataFrame, dict[str, float]]:
    if not SABER_PATH.exists():
        raise FileNotFoundError(f"Decoded SABER file not found: {SABER_PATH}")
    long_df = pl.read_parquet(SABER_PATH)
    if ANALYSIS_START_DATE is not None:
        long_df = long_df.filter(pl.col("date") >= ANALYSIS_START_DATE)
    if ANALYSIS_END_DATE is not None:
        long_df = long_df.filter(pl.col("date") <= ANALYSIS_END_DATE)
    altitudes = long_df["altitude_km"].unique().sort().to_list()
    if not altitudes:
        raise RuntimeError("Decoded SABER file has no altitudes.")
    median_value = float(np.median(np.asarray(altitudes, dtype=float)))
    selected = {
        "min": float(min(altitudes)),
        "median": float(min(altitudes, key=lambda alt: abs(float(alt) - median_value))),
        "max": float(max(altitudes)),
    }
    frames = []
    for role, altitude in selected.items():
        frames.append(
            long_df.filter(pl.col("altitude_km") == altitude).select(
                "date",
                pl.col("co2_cooling_rate_w_m3").alias(SABER_ALTITUDE_ROLE_COLS[role]),
            )
        )
    wide = frames[0]
    for frame in frames[1:]:
        wide = wide.join(frame, on="date", how="full", coalesce=True)
    wide = wide.sort("date")
    wide.write_csv(OUTPUT_DIR / "saber_maunaloa_daily_selected_altitudes.csv")
    return wide, selected


def load_co2() -> pl.DataFrame:
    schema = {
        "year": pl.Int32,
        "month": pl.Int32,
        "day": pl.Int32,
        "year_decimal": pl.Float32,
        "CO2_ppm": pl.Float64,
    }
    return (
        pl.read_csv(CO2_PATH, has_header=False, schema=schema, comment_prefix="#")
        .with_columns(pl.date("year", "month", "day").alias("date"))
        .select("date", "CO2_ppm")
        .with_columns(
            pl.when(pl.col("CO2_ppm") < 0)
            .then(None)
            .otherwise(pl.col("CO2_ppm"))
            .alias("CO2_ppm")
        )
    )


def load_space_weather() -> pl.DataFrame:
    return (
        pl.read_csv(SPACE_WEATHER_PATH)
        .with_columns(pl.col("DATE").str.to_date("%Y-%m-%d").alias("date"))
        .select("date", *Y_COLS)
    )


def as_date_index(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(pl.col("date").cast(pl.Date))
        .filter(pl.col("date").is_not_null())
        .unique(subset="date")
        .sort("date")
    )


def combine_inputs() -> tuple[
    pl.DataFrame, pl.DataFrame, list[str], list[int], list[int], dict[str, float]
]:
    hasdm, density_cols, hasdm_altitudes, selected_hasdm_altitudes = (
        load_hasdm_maunaloa_daily()
    )
    saber, saber_altitudes = load_saber_maunaloa_daily()
    space_weather = as_date_index(load_space_weather())
    co2 = as_date_index(load_co2())
    hasdm = as_date_index(hasdm)
    saber = as_date_index(saber)

    all_cols = [*DRIVER_COLS, *density_cols, *SABER_COLS]
    start_date = max(
        hasdm.filter(pl.col(density_cols[0]).is_not_null())["date"].min(),
        saber.filter(pl.col(SABER_COLS[0]).is_not_null())["date"].min(),
        space_weather.filter(pl.col(Y_COLS[0]).is_not_null())["date"].min(),
        co2.filter(pl.col("CO2_ppm").is_not_null())["date"].min(),
    )
    end_date = min(
        hasdm.filter(pl.col(density_cols[0]).is_not_null())["date"].max(),
        saber.filter(pl.col(SABER_COLS[0]).is_not_null())["date"].max(),
        space_weather.filter(pl.col(Y_COLS[0]).is_not_null())["date"].max(),
        co2.filter(pl.col("CO2_ppm").is_not_null())["date"].max(),
    )

    combined = hasdm
    for dataset in [saber, space_weather, co2]:
        combined = combined.join(dataset, on="date", how="full", coalesce=True)
    combined = (
        combined.sort("date")
        .filter((pl.col("date") >= start_date) & (pl.col("date") <= end_date))
        .select("date", *all_cols)
    )
    missing_summary = combined.select(
        [pl.col(col).is_null().sum().alias(col) for col in ["date", *all_cols]]
    )
    interpolated = combined.with_columns(
        [
            pl.col(col)
            .interpolate()
            .fill_null(strategy="forward")
            .fill_null(strategy="backward")
            .alias(col)
            for col in all_cols
        ]
    ).drop_nulls(all_cols)
    return (
        interpolated,
        missing_summary,
        density_cols,
        hasdm_altitudes,
        selected_hasdm_altitudes,
        saber_altitudes,
    )


(
    analysis_df,
    missing_summary_df,
    DENSITY_COLS,
    HASDM_ALTITUDES,
    SELECTED_HASDM_ALTITUDES,
    SABER_ALTITUDES,
) = combine_inputs()
LOWEST_HASDM_ALTITUDE = min(HASDM_ALTITUDES)
LOWEST_DENSITY_COLS = [
    f"log10rho_{LOWEST_HASDM_ALTITUDE}_daily_{stat}" for stat in ["min", "mean", "max"]
] + [f"log10rho_{LOWEST_HASDM_ALTITUDE}_daily_range"]
SELECTED_DENSITY_MEAN_COLS = density_mean_cols_for_altitudes(SELECTED_HASDM_ALTITUDES)
SELECTED_DENSITY_RANGE_COLS = density_range_cols_for_altitudes(SELECTED_HASDM_ALTITUDES)
TARGET_COLS = [*DENSITY_COLS, *SABER_COLS]
LAG_EFFECT_TARGET_COLS = [
    *SELECTED_DENSITY_MEAN_COLS,
    *SELECTED_DENSITY_RANGE_COLS,
    *SABER_COLS,
]
ALL_COLS = [*DRIVER_COLS, *DENSITY_COLS, *SABER_COLS]


def label_for_col(col: str) -> str:
    if col.startswith("log10rho_") and col.endswith("_daily_range"):
        return col.replace("log10rho_", r"$\Delta\ell_\rho$ range ").replace(
            "_daily_range", " km daily max/min"
        )
    if col.startswith("log10rho_"):
        return col.replace("log10rho_", r"$\ell_\rho$ ").replace(
            "_daily_", " km daily "
        )
    return col


LABELS = {
    "F10.7_OBS_CENTER81": "F10.7 81d avg",
    "AP_AVG": "Ap",
    "KP_SUM": "Kp",
    "CO2_ppm": "CO2",
    **{col: label_for_col(col) for col in DENSITY_COLS},
    "saber_co2cool_min_alt": f"SABER CO2 cooling {SABER_ALTITUDES['min']:.0f} km",
    "saber_co2cool_median_alt": f"SABER CO2 cooling {SABER_ALTITUDES['median']:.0f} km",
    "saber_co2cool_max_alt": f"SABER CO2 cooling {SABER_ALTITUDES['max']:.0f} km",
}

missing_summary_df.write_csv(OUTPUT_DIR / "missing_summary_before_interpolation.csv")
analysis_df.write_csv(OUTPUT_DIR / "daily_analysis_dataset.csv")
print(
    analysis_df.select(
        pl.min("date").alias("date_min"),
        pl.max("date").alias("date_max"),
        pl.len().alias("rows"),
    )
)
print(missing_summary_df)


# %%
def finite_standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values)
    std = np.nanstd(values)
    if not np.isfinite(std) or std == 0:
        return values - mean
    return (values - mean) / std


def rolling_nanmean(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    window = max(1, min(window, len(values)))
    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0)
    kernel = np.ones(window, dtype=float)
    numerator = np.convolve(filled, kernel, mode="same")
    denominator = np.convolve(finite.astype(float), kernel, mode="same")
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(values, np.nan, dtype=float),
        where=denominator > 0,
    )


def day_of_year(dates: np.ndarray) -> np.ndarray:
    python_dates = dates.astype("datetime64[D]").astype(object)
    return np.array([value.timetuple().tm_yday for value in python_dates], dtype=int)


def seasonal_anomaly(values: np.ndarray, dates: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    doys = day_of_year(dates)
    climatology = np.full(367, np.nan, dtype=float)
    for doy in range(1, 367):
        mask = doys == doy
        if np.any(mask):
            climatology[doy] = np.nanmean(values[mask])
    global_mean = np.nanmean(values)
    climatology = np.where(np.isfinite(climatology), climatology, global_mean)
    return values - climatology[doys]


def detrended(values: np.ndarray, window: int = 365 * 3) -> np.ndarray:
    return values - rolling_nanmean(values, window)


def make_variants(df: pl.DataFrame) -> dict[str, Variant]:
    dates = df["date"].to_numpy()
    raw = {col: df[col].to_numpy().astype(float) for col in ALL_COLS}
    seasonal = {col: seasonal_anomaly(raw[col], dates) for col in ALL_COLS}
    detrended_seasonal = {col: detrended(seasonal[col]) for col in ALL_COLS}
    co2_preserved = {
        col: seasonal[col] if col != "CO2_ppm" else raw[col] for col in ALL_COLS
    }
    return {
        "raw_standardized": Variant(
            "raw_standardized",
            dates,
            {col: finite_standardize(raw[col]) for col in ALL_COLS},
            "Raw daily values after temporal gap filling, standardized.",
        ),
        "seasonal_anomaly": Variant(
            "seasonal_anomaly",
            dates,
            {col: finite_standardize(seasonal[col]) for col in ALL_COLS},
            "Day-of-year climatology removed, then standardized.",
        ),
        "detrended_anomaly": Variant(
            "detrended_anomaly",
            dates,
            {col: finite_standardize(detrended_seasonal[col]) for col in ALL_COLS},
            "Seasonal anomalies with a 3-year rolling mean removed.",
        ),
        "co2_preserved_anomaly": Variant(
            "co2_preserved_anomaly",
            dates,
            {col: finite_standardize(co2_preserved[col]) for col in ALL_COLS},
            "Seasonal anomalies, but CO2 kept as a slow standardized driver.",
        ),
    }


variants = make_variants(analysis_df)
for variant in variants.values():
    print(f"{variant.name}: {variant.description}")


# %%
def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 3:
        return np.nan
    x0 = x[mask] - np.mean(x[mask])
    y0 = y[mask] - np.mean(y[mask])
    denom = np.sqrt(np.sum(x0**2) * np.sum(y0**2))
    if denom == 0:
        return np.nan
    return float(np.sum(x0 * y0) / denom)


def plot_correlation_heatmap(variant: Variant) -> np.ndarray:
    matrix = np.array(
        [
            [pearsonr(variant.data[row], variant.data[col]) for col in ALL_COLS]
            for row in ALL_COLS
        ]
    )
    labels = [LABELS[col] for col in ALL_COLS]
    size = max(10, min(30, 0.32 * len(labels)))
    fig, ax = plt.subplots(figsize=(size, size), constrained_layout=True)
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=90, fontsize=6)
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=6)
    ax.set_title(f"Correlation matrix: {variant.name}")
    fig.colorbar(image, ax=ax, label="Pearson r")
    fig.savefig(OUTPUT_DIR / f"correlation_{variant.name}.png", dpi=200)
    plt.close(fig)
    return matrix


def write_lowest_density_top_saber_correlations() -> None:
    rows = []
    top_saber = SABER_ALTITUDE_ROLE_COLS["max"]
    for variant in variants.values():
        for density_col in LOWEST_DENSITY_COLS:
            rows.append(
                {
                    "variant": variant.name,
                    "density_col": density_col,
                    "saber_col": top_saber,
                    "correlation": pearsonr(
                        variant.data[density_col], variant.data[top_saber]
                    ),
                }
            )
    pl.DataFrame(rows).write_csv(
        OUTPUT_DIR / "lowest_hasdm_density_vs_top_saber_cooling_correlations.csv"
    )


def finite_xy(
    df: pl.DataFrame, x_col: str, y_col: str
) -> tuple[np.ndarray, np.ndarray]:
    data = df.select(x_col, y_col).drop_nulls()
    x = data[x_col].to_numpy().astype(float)
    y = data[y_col].to_numpy().astype(float)
    finite = np.isfinite(x) & np.isfinite(y)
    return x[finite], y[finite]


def plot_target_scatter_grid(
    df: pl.DataFrame,
    target_cols: list[str],
    x_cols: list[str],
    filename: str,
    title: str,
) -> None:
    if not target_cols:
        return
    fig, axes = plt.subplots(
        len(target_cols),
        len(x_cols),
        figsize=(3.1 * len(x_cols), 2.4 * len(target_cols)),
        sharex="col",
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    for row, target_col in enumerate(target_cols):
        for col, x_col in enumerate(x_cols):
            ax = axes[row, col]
            x, y = finite_xy(df, x_col, target_col)
            ax.scatter(x, y, s=6, alpha=0.25)
            if len(x) >= 3 and np.nanstd(x) > 0:
                slope, intercept = np.polyfit(x, y, 1)
                x_line = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 80)
                ax.plot(
                    x_line,
                    slope * x_line + intercept,
                    color="black",
                    linewidth=0.8,
                    alpha=0.75,
                )
            if row == 0:
                ax.set_title(LABELS[x_col], fontsize=8)
            if row == len(target_cols) - 1:
                ax.set_xlabel(LABELS[x_col], fontsize=8)
            if col == 0:
                ax.set_ylabel(LABELS[target_col], fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.2)
    fig.suptitle(title)
    fig.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.close(fig)


def plot_scatter_outputs() -> None:
    x_cols = [*DRIVER_COLS, *SABER_COLS]
    plot_target_scatter_grid(
        analysis_df,
        SELECTED_DENSITY_MEAN_COLS,
        x_cols,
        "scatter_density_mean_selected_altitudes.png",
        "Daily mean log10 HASDM density scatter by selected altitude",
    )
    plot_target_scatter_grid(
        analysis_df,
        SELECTED_DENSITY_RANGE_COLS,
        x_cols,
        "scatter_density_range_selected_altitudes.png",
        "Daily log10 HASDM density max/min range scatter by selected altitude",
    )
    plot_target_scatter_grid(
        analysis_df,
        SABER_COLS,
        [*DRIVER_COLS, *SELECTED_DENSITY_MEAN_COLS, *SELECTED_DENSITY_RANGE_COLS],
        "scatter_saber_cooling_vs_drivers_density.png",
        "SABER CO2 cooling scatter against drivers and selected HASDM density parameters",
    )


def plot_correlation_by_altitude(variant: Variant) -> pl.DataFrame:
    rows = []
    causes = [*DRIVER_COLS, *SABER_COLS]
    for metric, template in [
        ("daily_mean_log10_density", "log10rho_{altitude}_daily_mean"),
        ("daily_log10_density_range", "log10rho_{altitude}_daily_range"),
    ]:
        for altitude in SELECTED_HASDM_ALTITUDES:
            target_col = template.format(altitude=altitude)
            if target_col not in variant.data:
                continue
            for cause in causes:
                rows.append(
                    {
                        "variant": variant.name,
                        "density_metric": metric,
                        "altitude_km": altitude,
                        "cause": cause,
                        "correlation": pearsonr(
                            variant.data[target_col], variant.data[cause]
                        ),
                    }
                )
    table = pl.DataFrame(rows)
    if table.is_empty():
        return table

    fig, axes = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True, constrained_layout=True
    )
    for ax, metric in zip(
        axes, ["daily_mean_log10_density", "daily_log10_density_range"]
    ):
        metric_df = table.filter(pl.col("density_metric") == metric)
        for cause in causes:
            series = metric_df.filter(pl.col("cause") == cause).sort("altitude_km")
            if series.is_empty():
                continue
            ax.plot(
                series["altitude_km"],
                series["correlation"],
                marker="o",
                linewidth=1.4,
                label=LABELS[cause],
            )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylim(-1, 1)
        ax.set_ylabel("Pearson r")
        ax.set_title(metric.replace("_", " "))
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("HASDM altitude (km)")
    axes[0].legend(fontsize=8, ncols=2)
    fig.suptitle(f"Correlation by selected HASDM altitude: {variant.name}")
    fig.savefig(OUTPUT_DIR / f"correlation_by_altitude_{variant.name}.png", dpi=200)
    plt.close(fig)
    return table


def lag_correlations(variant: Variant, max_lag: int = MAX_LAG_DAYS) -> pl.DataFrame:
    rows = []
    cause_cols = [*DRIVER_COLS, *SABER_COLS]
    for target in LAG_EFFECT_TARGET_COLS:
        y = variant.data[target]
        for cause in cause_cols:
            x = variant.data[cause]
            if cause == target:
                continue
            for lag in range(max_lag + 1):
                corr = pearsonr(x[: len(x) - lag], y[lag:]) if lag else pearsonr(x, y)
                rows.append(
                    {
                        "variant": variant.name,
                        "target": target,
                        "cause": cause,
                        "lag_days": lag,
                        "correlation": corr,
                    }
                )
    return pl.DataFrame(rows)


def plot_lag_correlations(corr_df: pl.DataFrame, variant_name: str) -> None:
    for target in LAG_EFFECT_TARGET_COLS:
        target_df = corr_df.filter(pl.col("target") == target)
        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
        for cause in [*DRIVER_COLS, *SABER_COLS]:
            series = target_df.filter(pl.col("cause") == cause).sort("lag_days")
            if series.is_empty():
                continue
            ax.plot(
                series["lag_days"],
                series["correlation"],
                linewidth=1.5,
                label=LABELS[cause],
            )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Lag from cause to target (days)")
        ax.set_ylabel("Pearson r")
        ax.set_title(f"Lag correlations with {LABELS[target]}: {variant_name}")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.savefig(
            OUTPUT_DIR / f"lag_correlations_{variant_name}_{target}.png", dpi=200
        )
        plt.close(fig)


correlation_tables = []
lag_correlation_tables = []
altitude_correlation_tables = []
plot_scatter_outputs()
for variant in variants.values():
    corr = plot_correlation_heatmap(variant)
    correlation_tables.append(
        pl.DataFrame(corr, schema=ALL_COLS, orient="row").with_columns(
            pl.Series("row", ALL_COLS), pl.lit(variant.name).alias("variant")
        )
    )
    altitude_corr = plot_correlation_by_altitude(variant)
    if not altitude_corr.is_empty():
        altitude_correlation_tables.append(altitude_corr)
    lag_df = lag_correlations(variant)
    lag_df.write_csv(OUTPUT_DIR / f"lag_correlations_{variant.name}.csv")
    plot_lag_correlations(lag_df, variant.name)
    lag_correlation_tables.append(lag_df)

pl.concat(correlation_tables).write_csv(OUTPUT_DIR / "correlation_matrices.csv")
pl.concat(lag_correlation_tables).write_csv(
    OUTPUT_DIR / "lag_correlations_all_variants.csv"
)
if altitude_correlation_tables:
    pl.concat(altitude_correlation_tables).write_csv(
        OUTPUT_DIR / "correlations_by_selected_altitude.csv"
    )
write_lowest_density_top_saber_correlations()


# %%
def lagged_matrix(
    variant: Variant,
    predictors: list[tuple[str, int]],
    target: str,
    max_lag: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if max_lag is None:
        max_lag = max(lag for _, lag in predictors)
    n = len(variant.dates)
    y = variant.data[target][max_lag:]
    columns = []
    names = []
    for col, lag in predictors:
        columns.append(variant.data[col][max_lag - lag : n - lag])
        names.append(f"{col}_lag{lag}")
    x = np.column_stack(columns) if columns else np.empty((len(y), 0))
    mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    return x[mask], y[mask], names


def standardize_design(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_std = np.std(x, axis=0)
    x_std = np.where(x_std == 0, 1.0, x_std)
    y_std = np.std(y)
    if y_std == 0:
        y_std = 1.0
    return (x - np.mean(x, axis=0)) / x_std, (y - np.mean(y)) / y_std


def ols_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    x_aug = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(x_aug, y, rcond=None)[0]
    residual = y - x_aug @ beta
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return beta, 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def block_bootstrap_indices(
    n: int, block_len: int, rng: np.random.Generator
) -> np.ndarray:
    starts = rng.integers(
        0, max(1, n - block_len + 1), size=int(np.ceil(n / block_len))
    )
    indices = np.concatenate(
        [np.arange(start, min(start + block_len, n)) for start in starts]
    )
    if len(indices) < n:
        indices = np.concatenate([indices, rng.integers(0, n, size=n - len(indices))])
    return indices[:n]


def hac_standard_errors(
    x: np.ndarray, y: np.ndarray, beta: np.ndarray, max_lag: int
) -> np.ndarray:
    x_aug = np.column_stack([np.ones(len(x)), x])
    residual = y - x_aug @ beta
    xtx_inv = np.linalg.pinv(x_aug.T @ x_aug)
    xu = x_aug * residual[:, None]
    meat = xu.T @ xu
    for lag in range(1, min(max_lag, len(y) - 1) + 1):
        weight = 1 - lag / (max_lag + 1)
        gamma = xu[lag:].T @ xu[:-lag]
        meat += weight * (gamma + gamma.T)
    covariance = xtx_inv @ meat @ xtx_inv
    return np.sqrt(np.clip(np.diag(covariance), 0, np.inf))


def effect_predictors(
    driver: str, driver_lag: int, target: str
) -> list[tuple[str, int]]:
    predictors: list[tuple[str, int]] = [(driver, driver_lag)]
    for other in [*DRIVER_COLS, *SABER_COLS]:
        for lag in ADJUSTMENT_LAGS:
            candidate = (other, lag)
            if candidate not in predictors and other != target:
                predictors.append(candidate)
    for lag in TARGET_AUTOREGRESSIVE_LAGS:
        predictors.append((target, lag))
    return predictors


def estimate_adjusted_effect(
    variant: Variant,
    driver: str,
    target: str,
    driver_lag: int,
    bootstraps: int = BOOTSTRAPS,
) -> dict[str, float | str | int]:
    predictors = effect_predictors(driver, driver_lag, target)
    max_lag = max(lag for _, lag in predictors)
    x, y, names = lagged_matrix(variant, predictors, target, max_lag)
    x, y = standardize_design(x, y)
    beta, r2 = ols_fit(x, y)
    effect_idx = names.index(f"{driver}_lag{driver_lag}") + 1
    effect = float(beta[effect_idx])
    if USE_BLOCK_BOOTSTRAP:
        rng = np.random.default_rng(
            RANDOM_SEED + driver_lag + len(driver) + len(target)
        )
        boot = np.empty(bootstraps, dtype=float)
        for i in range(bootstraps):
            idx = block_bootstrap_indices(len(y), BOOTSTRAP_BLOCK_DAYS, rng)
            boot_beta, _ = ols_fit(x[idx], y[idx])
            boot[i] = boot_beta[effect_idx]
        ci_low, ci_high = np.percentile(boot, [5, 95])
        standard_error = float(np.nanstd(boot))
        interval_method = "block_bootstrap"
    else:
        standard_errors = hac_standard_errors(x, y, beta, HAC_MAX_LAG_DAYS)
        standard_error = float(standard_errors[effect_idx])
        ci_low = effect - 1.6448536269514722 * standard_error
        ci_high = effect + 1.6448536269514722 * standard_error
        interval_method = "newey_west_hac"
    return {
        "variant": variant.name,
        "target": target,
        "driver": driver,
        "lag_days": driver_lag,
        "effect_std_target_per_std_driver": effect,
        "ci90_low": float(ci_low),
        "ci90_high": float(ci_high),
        "standard_error": standard_error,
        "r2_adjusted_model": float(r2),
        "n_samples": int(len(y)),
        "method": f"linear_adjustment_{interval_method}",
    }


if RUN_ADJUSTED_EFFECTS:
    effect_rows = []
    for variant in variants.values():
        for target in LAG_EFFECT_TARGET_COLS:
            for driver in [*DRIVER_COLS, *SABER_COLS]:
                if driver == target:
                    continue
                for lag in PHYSICS_LAGS:
                    effect_rows.append(
                        estimate_adjusted_effect(variant, driver, target, lag)
                    )

    effect_df = pl.DataFrame(effect_rows)
    effect_df.write_csv(OUTPUT_DIR / "adjusted_effect_estimates.csv")
    print(effect_df.sort("effect_std_target_per_std_driver", descending=True).head(20))
else:
    print(
        "Skipping adjusted-effect estimates. Set MAUNALOA_RUN_ADJUSTED_EFFECTS=1 to enable them."
    )


# %%
def sigma_edges(values: np.ndarray) -> tuple[np.ndarray, float, float, np.ndarray]:
    mean = float(np.mean(values))
    std = float(np.std(values))
    if not np.isfinite(std) or std == 0:
        raise ValueError("Cannot build sigma bins for a constant or invalid variable.")
    z = (values - mean) / std
    edges = np.arange(np.floor(np.min(z)), np.ceil(np.max(z)) + 1, 1.0)
    if len(edges) < 2:
        edges = np.array([-0.5, 0.5])
    return edges, mean, std, z


def trim_empty_margins(
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    mean_values: np.ndarray,
    std_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(mean_values)
    if not np.any(finite):
        return x_edges, y_edges, mean_values, std_values
    row_indices = np.flatnonzero(np.any(finite, axis=1))
    col_indices = np.flatnonzero(np.any(finite, axis=0))
    row_start, row_stop = int(row_indices[0]), int(row_indices[-1]) + 1
    col_start, col_stop = int(col_indices[0]), int(col_indices[-1]) + 1
    return (
        x_edges[col_start : col_stop + 1],
        y_edges[row_start : row_stop + 1],
        mean_values[row_start:row_stop, col_start:col_stop],
        std_values[row_start:row_stop, col_start:col_stop],
    )


def binned_value_stats(
    df: pl.DataFrame, y_col: str, value_col: str
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]
]:
    data = df.select(X_COL, y_col, value_col).drop_nulls()
    x = data[X_COL].to_numpy().astype(float)
    y = data[y_col].to_numpy().astype(float)
    values = data[value_col].to_numpy().astype(float)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
    x, y, values = x[finite], y[finite], values[finite]
    x_edges, x_mean, x_std, x_z = sigma_edges(x)
    y_edges, y_mean, y_std, y_z = sigma_edges(y)
    counts, _, _ = np.histogram2d(y_z, x_z, bins=[y_edges, x_edges])
    sums, _, _ = np.histogram2d(y_z, x_z, bins=[y_edges, x_edges], weights=values)
    sums_sq, _, _ = np.histogram2d(y_z, x_z, bins=[y_edges, x_edges], weights=values**2)
    mean_values = np.divide(
        sums,
        counts,
        out=np.full_like(sums, np.nan, dtype=float),
        where=counts >= MIN_SAMPLES_PER_CELL,
    )
    variance = (
        np.divide(
            sums_sq,
            counts,
            out=np.full_like(sums_sq, np.nan, dtype=float),
            where=counts >= MIN_SAMPLES_PER_CELL,
        )
        - mean_values**2
    )
    std_values = np.sqrt(np.clip(variance, 0, np.inf))
    return (
        x_edges,
        y_edges,
        counts,
        mean_values,
        std_values,
        {
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": y_mean,
            "y_std": y_std,
            "value_mean": float(np.mean(values)),
            "value_std": float(np.std(values)),
        },
    )


def set_integer_sigma_ticks(
    ax: plt.Axes, x_edges: np.ndarray, y_edges: np.ndarray
) -> None:
    x_ticks = np.arange(
        int(np.ceil(np.min(x_edges))), int(np.floor(np.max(x_edges))) + 1, 1
    )
    y_ticks = np.arange(
        int(np.ceil(np.min(y_edges))), int(np.floor(np.max(y_edges))) + 1, 1
    )
    if len(x_ticks):
        ax.set_xticks(x_ticks)
    if len(y_ticks):
        ax.set_yticks(y_ticks)


def safe_name(text: str) -> str:
    return text.replace(".", "p").replace("/", "_").replace(" ", "_")


def plot_value_heatmaps(
    df: pl.DataFrame, value_col: str, output_prefix: str, colorbar_label: str
) -> None:
    stats_by_driver = {
        y_col: binned_value_stats(df, y_col, value_col) for y_col in Y_COLS
    }
    finite_means = np.concatenate(
        [
            ((mean_values - metadata["value_mean"]) / metadata["value_std"])[
                np.isfinite(mean_values)
            ]
            for _, _, _, mean_values, _, metadata in stats_by_driver.values()
        ]
    )
    max_abs = float(np.max(np.abs(finite_means))) if len(finite_means) else 1.0
    fig, axes = plt.subplots(
        1, len(Y_COLS), figsize=(5.4 * len(Y_COLS), 6.4), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    mesh = None
    for ax, y_col in zip(axes, Y_COLS):
        x_edges, y_edges, counts, mean_values, std_values, metadata = stats_by_driver[
            y_col
        ]
        x_edges, y_edges, mean_values, std_values = trim_empty_margins(
            x_edges, y_edges, mean_values, std_values
        )
        color_values = (mean_values - metadata["value_mean"]) / metadata["value_std"]
        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            color_values,
            shading="auto",
            cmap="coolwarm",
            vmin=-max_abs,
            vmax=max_abs,
        )
        for y_idx in range(len(y_edges) - 1):
            for x_idx in range(len(x_edges) - 1):
                if not np.isfinite(mean_values[y_idx, x_idx]):
                    continue
                ax.text(
                    0.5 * (x_edges[x_idx] + x_edges[x_idx + 1]),
                    0.5 * (y_edges[y_idx] + y_edges[y_idx + 1]),
                    f"{mean_values[y_idx, x_idx]:.2e}\n+/- {std_values[y_idx, x_idx]:.1e}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    bbox={
                        "boxstyle": "round,pad=0.15",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.65,
                    },
                )
        ax.set_xlabel(
            f"CO2 sigma bins\nmean={metadata['x_mean']:.2f} ppm, sigma={metadata['x_std']:.2f} ppm"
        )
        ax.set_ylabel(
            f"{LABELS[y_col]} sigma bins\nmean={metadata['y_mean']:.2f}, sigma={metadata['y_std']:.2f}"
        )
        set_integer_sigma_ticks(ax, x_edges, y_edges)
        ax.grid(True, color="white", alpha=0.25, linewidth=0.7)
    if mesh is not None:
        colorbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), shrink=0.9)
        colorbar.set_label(colorbar_label)
    fig.suptitle(f"{LABELS[value_col]}: mean +/- std by bin")
    fig.savefig(
        OUTPUT_DIR
        / f"{output_prefix}_{safe_name(value_col)}_combined_co2_vs_space_weather.png",
        dpi=200,
    )
    plt.close(fig)


for saber_col in SABER_COLS:
    plot_value_heatmaps(
        analysis_df, saber_col, "cooling_heatmap", "Mean SABER CO2 cooling rate sigma"
    )
for density_col in SELECTED_DENSITY_MEAN_COLS:
    plot_value_heatmaps(
        analysis_df,
        density_col,
        "density_heatmap",
        "Mean daily log10 HASDM density sigma",
    )
for density_range_col in SELECTED_DENSITY_RANGE_COLS:
    plot_value_heatmaps(
        analysis_df,
        density_range_col,
        "density_range_heatmap",
        "Mean daily log10 HASDM density range sigma",
    )


# %%
def write_method_notes() -> None:
    grid_lat = nearest_hasdm_latitude()
    notes = f"""# Mauna Loa HASDM/SABER Workflow Notes

Location:
- Mauna Loa Observatory latitude: {MAUNA_LOA_LAT}
- Mauna Loa longitude east: {MAUNA_LOA_LON_EAST}

Variables:
- Candidate drivers: {", ".join(DRIVER_COLS)}
- Focus targets: {", ".join(TARGET_COLS)}
- Lag/effect targets: {", ".join(LAG_EFFECT_TARGET_COLS)}
- HASDM full daily file includes min/mean/max/range for each altitude in {HASDM_ALTITUDES}
- HASDM nearest latitude: {grid_lat}; nearest available longitude is selected independently for each timestamp and altitude
- HASDM correlation/plot altitudes: {SELECTED_HASDM_ALTITUDES}
- HASDM analysis columns at selected altitudes: daily min/mean/max ell_rho and daily Delta ell_rho, `log10(max/min)`
- SABER selected altitudes: {SABER_ALTITUDES}
- Lag window: 0 to {MAX_LAG_DAYS} days

Preprocessing:
- HASDM is sampled at the nearest available grid point to Mauna Loa without spatial interpolation.
- HASDM densities above {MAX_VALID_HASDM_DENSITY:.1e} kg/m^3 are discarded before daily aggregation.
- HASDM daily density summaries are min, mean, max, and range. Min/mean/max are log10 transformed; range is `log10(max) - log10(min)`, equivalent to `log10(max/min)`.
- SABER uses the nearest daily tangent-point profile to Mauna Loa for each selected altitude.
- Daily gaps are linearly interpolated before standardized variants are generated.
"""
    for variant in variants.values():
        notes += f"- {variant.name}: {variant.description}\n"
    (OUTPUT_DIR / "README.md").write_text(notes, encoding="utf-8")


write_method_notes()
print(f"Saved Mauna Loa HASDM/SABER outputs to {OUTPUT_DIR}")
