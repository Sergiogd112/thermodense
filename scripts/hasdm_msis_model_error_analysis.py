from __future__ import annotations
# Ruff: configure_pgf() must run before pyplot imports; suppress intentional E402.
# ruff: noqa: E402

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from scripts.pgf_config import configure_pgf, fig_size

configure_pgf()

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.lines import Line2D
import numpy as np
import polars as pl
from pymsis import msis
from pymsis.utils import get_f107_ap
from thermodense.downloader.space_weather import SPACE_WEATHER_CSV_PATH  # noqa: E402

from scripts.stats_utils import ols_slope_ci, pearsonr_ci

MAUNA_LOA_LAT = 19.5362
MAUNA_LOA_LON_EAST = 204.4237
HASDM_LAT_COL = "Latitude (deg)"
HASDM_LON_COL = "Longitude (deg)"
HASDM_ALT_COL = "Altitude (m)"
HASDM_DENSITY_COL = "Density (kg/m^3)"
MAX_VALID_HASDM_DENSITY = 1.0e-8

HASDM_PATHS = sorted(Path("data/decoded/hasdm").glob("HASDM_*_merged.parquet"))
SABER_PATH = Path("data/decoded/saber/saber_co2_cooling_maunaloa_daily.parquet")
CO2_PATH = Path("data/original/co2/co2_daily_mlo.csv")
SPACE_WEATHER_PATH = SPACE_WEATHER_CSV_PATH

OUTPUT_ROOT = Path("outputs/figures/results/hasdm_msis_model_errors")
DATA_DIR = OUTPUT_ROOT / "data"

MODEL_VERSIONS = {
    "NRLMSISE-00": "0",
    "NRLMSIS 2.0": "2.0",
    "NRLMSIS 2.1": "2.1",
}
MODEL_ERROR_COLS = {
    name: f"error_{version.replace('.', 'p')}"
    for name, version in MODEL_VERSIONS.items()
}
SELECTED_ALTITUDES = [175, 325, 500, 650, 825]
FFT_HEATMAP_PERIOD_BINS_PER_DECADE = 24
FFT_HEATMAP_ALTITUDE_BIN_KM = 25
ANALYSIS_COLS = ["F10.7_OBS_CENTER81", "AP_AVG", "KP_SUM", "CO2_ppm"]
SABER_COLS = [
    "saber_co2cool_min_alt",
    "saber_co2cool_median_alt",
    "saber_co2cool_max_alt",
]
CORRELATION_DURATION_STEP_YEARS = 11
CORRELATION_DURATION_MARKERS = {
    0: "o",
    11: "s",
    22: "D",
    33: "^",
    44: "P",
    55: "X",
}


@dataclass(frozen=True)
class GridPoint:
    lat: float
    lon: float | None = None


def safe_name(value: str) -> str:
    return value.lower().replace(" ", "_").replace(".", "p").replace("-", "_")


def label_for_col(col: str) -> str:
    labels = {
        "F10.7_OBS_CENTER81": "$F_{10.7,81}$",
        "AP_AVG": "$A_p$",
        "KP_SUM": "$K_p$",
        "CO2_ppm": "CO$_2$",
        "saber_co2cool_min_alt": "SABER CO$_2$ 100 km",
        "saber_co2cool_median_alt": "SABER CO$_2$ 119 km",
        "saber_co2cool_max_alt": "SABER CO$_2$ 139 km",
    }
    return labels.get(col, col)


def circular_lon_delta(lon: np.ndarray, target_lon: float) -> np.ndarray:
    return np.abs((lon - target_lon + 180.0) % 360.0 - 180.0)


def nearest_hasdm_grid_point() -> GridPoint:
    if not HASDM_PATHS:
        raise FileNotFoundError("No decoded HASDM parquet files found.")
    grid = (
        pl.scan_parquet(str(HASDM_PATHS[0]))
        .select(HASDM_LAT_COL, HASDM_LON_COL)
        .unique()
        .collect()
    )
    lats = grid[HASDM_LAT_COL].to_numpy().astype(float)
    lons = grid[HASDM_LON_COL].to_numpy().astype(float)
    lon_deltas = circular_lon_delta(lons, MAUNA_LOA_LON_EAST)
    lat_scale = np.cos(np.deg2rad(MAUNA_LOA_LAT))
    distances = (lats - MAUNA_LOA_LAT) ** 2 + (lon_deltas * lat_scale) ** 2
    idx = int(np.argmin(distances))
    return GridPoint(float(lats[idx]), float(lons[idx]))


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


def selected_hasdm_for_path(path: Path, grid: GridPoint) -> pl.DataFrame:
    df = (
        pl.scan_parquet(str(path))
        .filter(
            (pl.col(HASDM_LAT_COL) == grid.lat)
            & (pl.col(HASDM_DENSITY_COL) > 0)
            & (pl.col(HASDM_DENSITY_COL) <= MAX_VALID_HASDM_DENSITY)
        )
        .with_columns(
            (((pl.col(HASDM_LON_COL) - MAUNA_LOA_LON_EAST + 180.0) % 360.0) - 180.0)
            .abs()
            .alias("lon_delta")
        )
        .with_columns(
            pl.min("lon_delta")
            .over(["timestamp", HASDM_ALT_COL])
            .alias("nearest_lon_delta")
        )
    )
    return (
        df.filter(pl.col("lon_delta") == pl.col("nearest_lon_delta"))
        .group_by(["timestamp", HASDM_ALT_COL])
        .agg(
            pl.col(HASDM_LON_COL).first().alias(HASDM_LON_COL),
            pl.col(HASDM_DENSITY_COL).first().alias(HASDM_DENSITY_COL),
        )
        .select("timestamp", HASDM_ALT_COL, HASDM_LON_COL, HASDM_DENSITY_COL)
        .collect()
        .sort(["timestamp", HASDM_ALT_COL])
    )


def add_msis_errors(df: pl.DataFrame, grid: GridPoint) -> pl.DataFrame:
    timestamps = df["timestamp"].to_numpy()
    lons = df[HASDM_LON_COL].to_numpy().astype(float)
    lats = np.full(len(df), grid.lat, dtype=float)
    alts = (df[HASDM_ALT_COL] / 1000.0).to_numpy().astype(float)
    rho_ref = df[HASDM_DENSITY_COL].to_numpy().astype(float)
    space_weather = get_f107_ap(timestamps)
    if len(space_weather) == 4:
        f107, f107a, aps, _ = space_weather
    else:
        f107, f107a, aps = space_weather
    out = df.with_columns(
        pl.Series("f107", f107),
        pl.Series("f107a", f107a),
        pl.Series("ap", aps[:, 0]),
    )
    for name, version in MODEL_VERSIONS.items():
        density = msis.calculate(
            timestamps, lons, lats, alts, f107, f107a, aps, version=version
        )[:, 0]
        out = out.with_columns(
            pl.Series(MODEL_ERROR_COLS[name], np.log(density / rho_ref))
        )
    return out


def compute_sample_errors() -> pl.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sample_path = DATA_DIR / "hasdm_msis_errors_nearest_timestamp_grid_samples.parquet"
    # if sample_path.exists():
    #     return pl.read_parquet(sample_path)
    grid = GridPoint(nearest_hasdm_latitude())
    frames = []
    for path in HASDM_PATHS:
        df = selected_hasdm_for_path(path, grid)
        if df.is_empty():
            continue
        frames.append(add_msis_errors(df, grid))
        print(f"Processed {path.name}: {df.height:,} nearest-grid samples")
    if not frames:
        raise RuntimeError("No HASDM nearest-grid samples found.")
    combined = pl.concat(frames).sort(["timestamp", HASDM_ALT_COL])
    combined.write_parquet(sample_path, compression="lz4")
    return combined


def aggregate_daily_errors(samples: pl.DataFrame) -> pl.DataFrame:
    daily_path = (
        DATA_DIR / "hasdm_msis_errors_nearest_timestamp_grid_daily_long.parquet"
    )
    agg_exprs = [
        (pl.col(HASDM_DENSITY_COL).mean()).alias("rho_hasdm_daily_mean"),
        (pl.col(HASDM_DENSITY_COL).min()).alias("rho_hasdm_daily_min"),
        (pl.col(HASDM_DENSITY_COL).max()).alias("rho_hasdm_daily_max"),
        pl.col("f107a").mean().alias("f107a"),
        pl.col("ap").mean().alias("ap"),
        pl.len().alias("samples"),
    ]
    for model_name, error_col in MODEL_ERROR_COLS.items():
        prefix = safe_name(model_name)
        agg_exprs.extend(
            [
                pl.col(error_col).min().alias(f"{prefix}_daily_min"),
                pl.col(error_col).mean().alias(f"{prefix}_daily_mean"),
                pl.col(error_col).max().alias(f"{prefix}_daily_max"),
                (pl.col(error_col).max() - pl.col(error_col).min()).alias(
                    f"{prefix}_daily_range"
                ),
            ]
        )
    daily = (
        samples.with_columns(
            pl.col("timestamp").dt.date().alias("date"),
            (pl.col(HASDM_ALT_COL) / 1000.0).cast(pl.Int64).alias("altitude_km"),
        )
        .group_by("date", "altitude_km")
        .agg(agg_exprs)
        .sort(["date", "altitude_km"])
    )
    daily.write_parquet(daily_path, compression="lz4")
    return daily


def pivot_daily(daily: pl.DataFrame) -> pl.DataFrame:
    wide_path = DATA_DIR / "hasdm_msis_errors_nearest_timestamp_grid_daily_wide.parquet"
    value_cols = [col for col in daily.columns if col not in {"date", "altitude_km"}]
    wide: pl.DataFrame | None = None
    for value_col in value_cols:
        pivot = daily.select(
            "date",
            pl.col("altitude_km").cast(pl.Utf8).alias("altitude_label"),
            value_col,
        ).pivot(
            index="date",
            on="altitude_label",
            values=value_col,
            aggregate_function="first",
        )
        rename = {col: f"{value_col}_{col}km" for col in pivot.columns if col != "date"}
        pivot = pivot.rename(rename)
        wide = (
            pivot
            if wide is None
            else wide.join(pivot, on="date", how="full", coalesce=True)
        )
    if wide is None:
        raise RuntimeError("No daily model-error data to pivot.")
    wide = wide.sort("date")
    wide.write_parquet(wide_path, compression="lz4")
    return wide


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
        .select("date", "F10.7_OBS_CENTER81", "AP_AVG", "KP_SUM")
    )


def load_saber() -> pl.DataFrame:
    long = pl.read_parquet(SABER_PATH)
    altitudes = long["altitude_km"].unique().sort().to_list()
    median_alt = float(
        min(
            altitudes, key=lambda value: abs(float(value) - float(np.median(altitudes)))
        )
    )
    selected = {
        "saber_co2cool_min_alt": float(min(altitudes)),
        "saber_co2cool_median_alt": median_alt,
        "saber_co2cool_max_alt": float(max(altitudes)),
    }
    frames = [
        long.filter(pl.col("altitude_km") == altitude).select(
            "date", pl.col("co2_cooling_rate_w_m3").alias(col)
        )
        for col, altitude in selected.items()
    ]
    out = frames[0]
    for frame in frames[1:]:
        out = out.join(frame, on="date", how="full", coalesce=True)
    return out.sort("date")


def combine_analysis_dataset(wide: pl.DataFrame) -> pl.DataFrame:
    analysis_path = DATA_DIR / "hasdm_msis_model_error_analysis_dataset.csv"
    combined = wide.with_columns(pl.col("date").cast(pl.Date))
    for dataset in [load_space_weather(), load_co2(), load_saber()]:
        combined = combined.join(
            dataset.with_columns(pl.col("date").cast(pl.Date)),
            on="date",
            how="full",
            coalesce=True,
        )
    required_error_cols = [
        col
        for col in combined.columns
        if col.endswith("_daily_mean_500km") and col.startswith("nrlms")
    ]
    model_error_cols = [
        col
        for col in combined.columns
        if col.startswith("nrlms") and "_daily_" in col and col.endswith("km")
    ]
    required = [*ANALYSIS_COLS, *required_error_cols]
    fill_cols = [
        col
        for col, dtype in combined.schema.items()
        if col != "date" and col not in model_error_cols and dtype.is_numeric()
    ]
    combined = (
        combined.sort("date")
        .with_columns(
            [
                pl.col(col)
                .interpolate()
                .fill_null(strategy="forward")
                .fill_null(strategy="backward")
                .alias(col)
                for col in fill_cols
            ]
        )
        .drop_nulls(required)
    )
    combined.write_csv(analysis_path)
    return combined


def save_figure(fig: plt.Figure, *parts: str) -> None:
    out = OUTPUT_ROOT.joinpath(*parts)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    if out.suffix == ".pgf":
        FigureCanvasAgg(fig)
        fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def add_outside_legend(
    fig: plt.Figure,
    handles: list,
    labels: list[str],
    *,
    loc: str = "upper center",
    title: str | None = None,
    ncols: int | None = None,
) -> None:
    if len(handles) <= 1:
        return
    bbox = (0.5, 0.995) if loc.startswith("upper") else (0.5, 0.005)
    fig.legend(
        handles,
        labels,
        title=title,
        loc=loc,
        bbox_to_anchor=bbox,
        ncols=ncols or min(5, len(handles)),
        fontsize=7,
        title_fontsize=8,
        borderaxespad=0.2,
    )
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        rect = (
            (0.0, 0.0, 1.0, 0.91) if loc.startswith("upper") else (0.0, 0.11, 1.0, 1.0)
        )
        layout_engine.set(rect=rect)


def error_col(model_name: str, altitude: int, stat: str = "mean") -> str:
    return f"{safe_name(model_name)}_daily_{stat}_{altitude}km"


def available_error_altitudes(
    df: pl.DataFrame, model_name: str, stat: str = "mean"
) -> list[int]:
    prefix = f"{safe_name(model_name)}_daily_{stat}_"
    suffix = "km"
    altitudes: list[int] = []
    for col in df.columns:
        if col.startswith(prefix) and col.endswith(suffix):
            try:
                altitudes.append(int(col[len(prefix) : -len(suffix)]))
            except ValueError:
                continue
    return sorted(set(altitudes))


def correlation_duration_bin(duration_years: float) -> int:
    if (
        not np.isfinite(duration_years)
        or duration_years < CORRELATION_DURATION_STEP_YEARS
    ):
        return 0
    return int(
        np.floor(duration_years / CORRELATION_DURATION_STEP_YEARS)
        * CORRELATION_DURATION_STEP_YEARS
    )


def correlation_duration_label(bin_start: int) -> str:
    if bin_start == 0:
        return f"<{CORRELATION_DURATION_STEP_YEARS} yr"
    return f"{bin_start}-{bin_start + CORRELATION_DURATION_STEP_YEARS} yr"


def correlation_duration_marker(bin_start: int) -> str:
    return CORRELATION_DURATION_MARKERS.get(bin_start, "*")


def correlation_and_duration(
    df: pl.DataFrame, x_col: str, y_col: str
) -> tuple[float, float, float, int | None]:
    if y_col not in df.columns or x_col not in df.columns:
        return np.nan, np.nan, np.nan, None
    data = (
        df.select("date", x_col, y_col)
        .drop_nulls()
        .group_by("date")
        .agg(pl.col(x_col).mean(), pl.col(y_col).mean())
        .sort("date")
    )
    if data.height < 3:
        return np.nan, np.nan, np.nan, None
    x = data[x_col].to_numpy().astype(float)
    y = data[y_col].to_numpy().astype(float)
    finite = np.isfinite(x) & np.isfinite(y)
    if int(np.count_nonzero(finite)) < 3:
        return np.nan, np.nan, np.nan, None
    r, r_lo, r_hi, _ = pearsonr_ci(x[finite], y[finite])
    if not np.isfinite(r):
        return np.nan, np.nan, np.nan, None
    dates = np.asarray(data["date"].to_list(), dtype=object)[finite]
    duration_years = (dates[-1] - dates[0]).days / 365.2425
    return r, r_lo, r_hi, correlation_duration_bin(duration_years)


def add_correlation_effect_size_bands(ax: plt.Axes) -> None:
    bands = [
        (-1.0, -0.5, "#cfe3f5"),
        (-0.5, -0.3, "#e3effa"),
        (-0.3, -0.1, "#f2f6fb"),
        (-0.1, 0.1, "#fff7e6"),
        (0.1, 0.3, "#f2f6fb"),
        (0.3, 0.5, "#e3effa"),
        (0.5, 1.0, "#cfe3f5"),
    ]
    for lower, upper, color in bands:
        ax.axhspan(lower, upper, color=color, alpha=1.0, zorder=0)
    for threshold in [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5]:
        ax.axhline(threshold, color="0.72", linewidth=0.9, zorder=1)


def add_record_length_legend(ax: plt.Axes, duration_bins: set[int]) -> None:
    if not duration_bins:
        return
    handles = [
        Line2D(
            [0],
            [0],
            marker=correlation_duration_marker(duration_bin),
            color="white",
            linestyle="None",
            markerfacecolor="0.55",
            markeredgecolor="black",
            markeredgewidth=0.45,
            markersize=8,
            label=correlation_duration_label(duration_bin),
        )
        for duration_bin in sorted(duration_bins)
    ]
    ax.legend(
        handles=handles,
        title="Record length",
        fontsize=8,
        title_fontsize=8,
        loc="upper right",
        bbox_to_anchor=(1, -0.18),
        borderaxespad=0,
        ncols=min(3, len(handles)),
    )


def record_length_handles(duration_bins: set[int]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker=correlation_duration_marker(duration_bin),
            color="white",
            linestyle="None",
            markerfacecolor="0.55",
            markeredgecolor="black",
            markeredgewidth=0.45,
            markersize=8,
            label=correlation_duration_label(duration_bin),
        )
        for duration_bin in sorted(duration_bins)
    ]


def date_edges_for_heatmap(dates: np.ndarray) -> np.ndarray:
    centers = mdates.date2num(dates)
    if len(centers) == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5])
    midpoints = (centers[:-1] + centers[1:]) / 2
    return np.concatenate(
        [
            [centers[0] - (midpoints[0] - centers[0])],
            midpoints,
            [centers[-1] + (centers[-1] - midpoints[-1])],
        ]
    )


def plot_model_time_heatmap(ax: plt.Axes, df: pl.DataFrame, model_name: str) -> None:
    altitudes = available_error_altitudes(df, model_name)
    cols = [error_col(model_name, altitude) for altitude in altitudes]
    if not cols:
        ax.text(
            0.5,
            0.5,
            "No time heatmap data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return
    data = df.select("date", *cols).sort("date").drop_nulls("date")
    dates = data["date"].to_numpy()
    alt_edges = altitude_bin_edges(altitudes)
    matrix_sum = np.zeros((len(alt_edges) - 1, len(dates)), dtype=float)
    matrix_count = np.zeros_like(matrix_sum)
    for altitude, col in zip(altitudes, cols):
        row = np.searchsorted(alt_edges, altitude, side="right") - 1
        values = data[col].to_numpy().astype(float)
        mask = np.isfinite(values)
        if 0 <= row < matrix_sum.shape[0]:
            matrix_sum[row, mask] += values[mask]
            matrix_count[row, mask] += 1
    matrix = np.divide(
        matrix_sum,
        matrix_count,
        out=np.full_like(matrix_sum, np.nan),
        where=matrix_count > 0,
    )
    max_abs = np.nanmax(np.abs(matrix)) if np.any(np.isfinite(matrix)) else 1.0
    mesh = ax.pcolormesh(
        date_edges_for_heatmap(dates),
        alt_edges,
        matrix,
        shading="auto",
        cmap="coolwarm",
        vmin=-max_abs,
        vmax=max_abs,
        rasterized=True,
    )
    if matrix.shape[0] > 1 and matrix.shape[1] > 1 and np.any(np.isfinite(matrix)):
        date_centers = mdates.date2num(dates)
        altitude_centers = 0.5 * (alt_edges[:-1] + alt_edges[1:])
        contour_levels = np.linspace(-max_abs, max_abs, 9)
        contour_levels = contour_levels[np.abs(contour_levels) > 1e-12]
        if len(contour_levels):
            ax.contour(
                date_centers,
                altitude_centers,
                matrix,
                levels=contour_levels,
                colors="black",
                linewidths=0.35,
                alpha=0.45,
            )
    ax.xaxis_date()
    ax.set_ylabel(f"{model_name}\naltitude (km)")
    ax.grid(True, alpha=0.18)
    ax.figure.colorbar(
        mesh, ax=ax, label=r"$\epsilon_m=\ln(\rho_m/\rho_\mathrm{ref})$", pad=0.01
    )


def plot_time_series(
    df: pl.DataFrame, x_limits: tuple[date, date] | None = None
) -> None:
    fig, axes = plt.subplots(
        len(MODEL_VERSIONS) * 2 + 2,
        1,
        figsize=fig_size(1.0, 1.55),
        sharex=True,
        constrained_layout=True,
    )
    model_axes = axes[: len(MODEL_VERSIONS) * 2 : 2]
    space_weather_ax = axes[-2]
    co2_ax = axes[-1]
    for model_idx, (ax, model_name) in enumerate(zip(model_axes, MODEL_VERSIONS)):
        for altitude in SELECTED_ALTITUDES:
            col = error_col(model_name, altitude)
            if col in df.columns:
                ax.plot(df["date"], df[col], linewidth=0.9, label=f"{altitude} km")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel(r"$\epsilon_m$")
        ax.set_title(model_name, loc="left")
        ax.grid(True, alpha=0.25)
        plot_model_time_heatmap(axes[model_idx * 2 + 1], df, model_name)
    space_weather_ax.plot(
        df["date"],
        df["F10.7_OBS_CENTER81"],
        color="darkred",
        linewidth=0.9,
        label="$F_{10.7,81}$",
    )
    ap_ax = space_weather_ax.twinx()
    ap_ax.plot(
        df["date"],
        df["AP_AVG"],
        color="purple",
        linewidth=0.8,
        alpha=0.75,
        label="$A_p$",
    )
    space_weather_ax.set_ylabel("$F_{10.7,81}$")
    ap_ax.set_ylabel("$A_p$")
    space_weather_ax.tick_params(axis="y", colors="darkred")
    ap_ax.tick_params(axis="y", colors="purple")
    space_weather_ax.yaxis.label.set_color("darkred")
    ap_ax.yaxis.label.set_color("purple")
    space_weather_ax.grid(True, alpha=0.25)
    co2_ax.plot(
        df["date"],
        df["CO2_ppm"],
        color="darkgreen",
        linewidth=0.9,
        label="Mauna Loa CO$_2$",
    )
    co2_ax.set_ylabel("CO$_2$ (ppm)")
    co2_ax.tick_params(axis="y", colors="darkgreen")
    co2_ax.yaxis.label.set_color("darkgreen")
    co2_ax.grid(True, alpha=0.25)
    if x_limits is not None:
        for ax in axes:
            ax.set_xlim(*x_limits)
    co2_ax.xaxis.set_major_locator(mdates.YearLocator(4))
    co2_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    handles, labels = model_axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.965),
            ncols=min(len(handles), len(SELECTED_ALTITUDES)),
            fontsize=7,
            borderaxespad=0,
            frameon=True,
        )
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(rect=(0.0, 0.0, 1.0, 0.90))
    fig.suptitle("Nearest-grid Mauna Loa HASDM model log-density-ratio errors", y=0.995)
    save_figure(fig, "fft_timeseries", "hasdm_msis_model_error_timeseries.pgf")


def fft_period_power(df: pl.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
    series = df.select("date", col).drop_nulls().sort("date")
    values = series[col].to_numpy().astype(float)
    if len(values) < 4:
        return np.array([]), np.array([])
    values = values - np.nanmean(values)
    days = series["date"].cast(pl.Int64).to_numpy()
    dt = float(np.median(np.diff(days)))
    frequencies = np.fft.rfftfreq(len(values), d=dt)
    amplitudes = np.abs(np.fft.rfft(values))
    valid = frequencies > 0
    return 1.0 / frequencies[valid], amplitudes[valid]


def log_period_edges(periods_years: np.ndarray) -> np.ndarray:
    finite = periods_years[np.isfinite(periods_years) & (periods_years > 0)]
    if len(finite) == 0:
        return np.array([])
    span_decades = np.log10(np.max(finite)) - np.log10(np.min(finite))
    n_bins = max(24, int(np.ceil(span_decades * FFT_HEATMAP_PERIOD_BINS_PER_DECADE)))
    return np.logspace(np.log10(np.min(finite)), np.log10(np.max(finite)), n_bins + 1)


def altitude_bin_edges(altitudes: list[int]) -> np.ndarray:
    lower = (
        np.floor(min(altitudes) / FFT_HEATMAP_ALTITUDE_BIN_KM)
        * FFT_HEATMAP_ALTITUDE_BIN_KM
    )
    upper = (
        np.ceil(max(altitudes) / FFT_HEATMAP_ALTITUDE_BIN_KM)
        * FFT_HEATMAP_ALTITUDE_BIN_KM
    )
    return np.arange(
        lower, upper + FFT_HEATMAP_ALTITUDE_BIN_KM, FFT_HEATMAP_ALTITUDE_BIN_KM
    )


def plot_model_fft_heatmap(ax: plt.Axes, df: pl.DataFrame, model_name: str) -> None:
    spectra = []
    all_periods = []
    for altitude in available_error_altitudes(df, model_name):
        col = error_col(model_name, altitude)
        if col not in df.columns:
            continue
        periods, amplitudes = fft_period_power(df, col)
        if len(periods) == 0:
            continue
        periods_years = periods / 365.25
        power = amplitudes**2
        spectra.append((altitude, periods_years, power))
        all_periods.extend(periods_years)
    if not spectra:
        ax.text(
            0.5,
            0.5,
            "No FFT heatmap data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return
    period_edges = log_period_edges(np.asarray(all_periods, dtype=float))
    period_centers = np.sqrt(period_edges[:-1] * period_edges[1:])
    target_log_periods = np.log10(period_centers)
    alt_edges = altitude_bin_edges([altitude for altitude, _, _ in spectra])
    matrix = np.full((len(alt_edges) - 1, len(period_centers)), np.nan)
    for altitude, periods_years, power in spectra:
        mask = (
            np.isfinite(periods_years)
            & np.isfinite(power)
            & (periods_years > 0)
            & (power > 0)
        )
        if np.sum(mask) < 2:
            continue
        order = np.argsort(periods_years[mask])
        interpolated = np.interp(
            target_log_periods,
            np.log10(periods_years[mask][order]),
            np.log10(power[mask][order]),
            left=np.nan,
            right=np.nan,
        )
        row = np.searchsorted(alt_edges, altitude, side="right") - 1
        if 0 <= row < matrix.shape[0]:
            matrix[row, :] = interpolated
    mesh = ax.pcolormesh(
        period_edges, alt_edges, matrix, shading="auto", cmap="magma", rasterized=True
    )
    ax.set_xscale("log")
    ax.set_ylabel(f"{model_name}\naltitude (km)")
    ax.grid(True, which="both", alpha=0.18)
    ax.figure.colorbar(mesh, ax=ax, label="log$_{10}$ FFT power")


def plot_fft(
    df: pl.DataFrame,
    model_versions: dict[str, str] | None = None,
    filename: str = "hasdm_msis_model_error_fft.pgf",
) -> None:
    model_versions = MODEL_VERSIONS if model_versions is None else model_versions
    n_rows = len(model_versions) * 2 + 3
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=fig_size(1.0, 0.22 * n_rows),
        sharex=True,
        constrained_layout=True,
    )
    ref_periods = np.array([7, 27, 183, 365.25, 365.25 * 11])
    for model_idx, model_name in enumerate(model_versions):
        ax = axes[model_idx * 2]
        for altitude in SELECTED_ALTITUDES:
            col = error_col(model_name, altitude)
            if col not in df.columns:
                continue
            periods, amplitudes = fft_period_power(df, col)
            if len(periods):
                ax.plot(
                    periods / 365.25, amplitudes, linewidth=0.9, label=f"{altitude} km"
                )
        for period in ref_periods:
            ax.axvline(period / 365.25, color="gray", linewidth=0.7, alpha=0.35)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel(model_name)
        ax.grid(True, which="both", alpha=0.25)
        heatmap_ax = axes[model_idx * 2 + 1]
        plot_model_fft_heatmap(heatmap_ax, df, model_name)
        for period in ref_periods:
            heatmap_ax.axvline(period / 365.25, color="gray", linewidth=0.7, alpha=0.35)
    solar_ax = axes[-3]
    geomag_ax = axes[-2]
    periods, amplitudes = fft_period_power(df, "F10.7_OBS_CENTER81")
    if len(periods) and np.nanmax(amplitudes) > 0:
        normalized = amplitudes / np.nanmax(amplitudes)
        normalized[normalized <= 0] = np.nan
        solar_ax.plot(
            periods / 365.25,
            normalized,
            color="darkred",
            linewidth=0.9,
            label="$F_{10.7,81}$",
        )
    for period in ref_periods:
        solar_ax.axvline(period / 365.25, color="gray", linewidth=0.7, alpha=0.35)
    solar_ax.set_xscale("log")
    solar_ax.set_yscale("log")
    solar_ax.set_ylabel("$F_{10.7,81}$\n(norm.)")
    solar_ax.grid(True, which="both", alpha=0.25)

    periods, amplitudes = fft_period_power(df, "AP_AVG")
    if len(periods) and np.nanmax(amplitudes) > 0:
        normalized = amplitudes / np.nanmax(amplitudes)
        normalized[normalized <= 0] = np.nan
        geomag_ax.plot(
            periods / 365.25, normalized, linewidth=0.9, label="$A_p$", color="purple"
        )
    for period in ref_periods:
        geomag_ax.axvline(period / 365.25, color="gray", linewidth=0.7, alpha=0.35)
    geomag_ax.set_xscale("log")
    geomag_ax.set_yscale("log")
    geomag_ax.set_ylabel("$A_p$\n(norm.)")
    geomag_ax.grid(True, which="both", alpha=0.25)
    co2_ax = axes[-1]
    periods, amplitudes = fft_period_power(df, "CO2_ppm")
    if len(periods) and np.nanmax(amplitudes) > 0:
        normalized = amplitudes / np.nanmax(amplitudes)
        normalized[normalized <= 0] = np.nan
        co2_ax.plot(
            periods / 365.25,
            normalized,
            color="darkgreen",
            linewidth=0.9,
            label="Mauna Loa CO$_2$",
        )
    for period in ref_periods:
        co2_ax.axvline(period / 365.25, color="gray", linewidth=0.7, alpha=0.35)
    co2_ax.set_xscale("log")
    co2_ax.set_yscale("log")
    co2_ax.set_ylabel("CO$_2$\n(norm.)")
    co2_ax.grid(True, which="both", alpha=0.25)
    axes[-1].set_xlabel("Period")
    axes[-1].set_xticks(ref_periods / 365.25)
    axes[-1].set_xticklabels(
        ["1 wk", "27 d", "6 mo", "1 y", "11 y"], rotation=45, ha="right"
    )
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.965),
            ncols=min(len(handles), len(SELECTED_ALTITUDES)),
            fontsize=7,
            borderaxespad=0,
        )
        layout_engine = fig.get_layout_engine()
        if layout_engine is not None:
            layout_engine.set(rect=(0.0, 0.0, 1.0, 0.90))
    fig.suptitle("HASDM model-error FFT spectra", y=0.995)
    save_figure(fig, "fft_timeseries", filename)


def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 3:
        return np.nan
    x0 = x[mask] - np.mean(x[mask])
    y0 = y[mask] - np.mean(y[mask])
    denom = np.sqrt(np.sum(x0**2) * np.sum(y0**2))
    return float(np.sum(x0 * y0) / denom) if denom else np.nan


def plot_correlation_by_altitude(df: pl.DataFrame) -> None:
    causes = [*ANALYSIS_COLS, *SABER_COLS]
    combined_fig, combined_axes = plt.subplots(
        len(MODEL_VERSIONS),
        1,
        figsize=fig_size(1.0, 1.2),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    combined_axes = np.asarray(combined_axes).reshape(len(MODEL_VERSIONS))
    combined_fig.subplots_adjust(
        left=0.18, right=0.98, top=0.86, bottom=0.22, hspace=0.35
    )
    combined_duration_bins: set[int] = set()
    for model_name in MODEL_VERSIONS:
        fig, ax = plt.subplots(figsize=fig_size(1.0, 0.3), constrained_layout=True)
        combined_ax = combined_axes[list(MODEL_VERSIONS).index(model_name)]
        add_correlation_effect_size_bands(ax)
        add_correlation_effect_size_bands(combined_ax)
        model_duration_bins: set[int] = set()
        altitudes = available_error_altitudes(df, model_name)
        for cause in causes:
            if cause not in df.columns:
                continue
            corr = []
            corr_los = []
            corr_his = []
            duration_bins = []
            for altitude in altitudes:
                col = error_col(model_name, altitude)
                point_corr, r_lo, r_hi, duration_bin = correlation_and_duration(
                    df, cause, col
                )
                corr.append(point_corr)
                corr_los.append(r_lo)
                corr_his.append(r_hi)
                duration_bins.append(duration_bin)
                if duration_bin is not None and np.isfinite(point_corr):
                    model_duration_bins.add(duration_bin)
                    combined_duration_bins.add(duration_bin)
            (line,) = ax.plot(
                altitudes, corr, linewidth=1.2, label=label_for_col(cause), zorder=3
            )
            combined_line = combined_ax.plot(
                altitudes, corr, linewidth=1.2, label=label_for_col(cause), zorder=3
            )[0]
            corr_arr = np.asarray(corr, dtype=float)
            lo_arr = np.asarray(corr_los, dtype=float)
            hi_arr = np.asarray(corr_his, dtype=float)
            finite_ci = (
                np.isfinite(corr_arr) & np.isfinite(lo_arr) & np.isfinite(hi_arr)
            )
            if np.any(finite_ci):
                ax.fill_between(
                    np.asarray(altitudes)[finite_ci],
                    lo_arr[finite_ci],
                    hi_arr[finite_ci],
                    alpha=0.15,
                    color=line.get_color(),
                    zorder=2,
                )
                combined_ax.fill_between(
                    np.asarray(altitudes)[finite_ci],
                    lo_arr[finite_ci],
                    hi_arr[finite_ci],
                    alpha=0.15,
                    color=combined_line.get_color(),
                    zorder=2,
                )
            for duration_bin in sorted(set(duration_bins) - {None}):
                mask = np.array([value == duration_bin for value in duration_bins])
                finite_corr = np.isfinite(np.asarray(corr, dtype=float))
                mask = mask & finite_corr
                if np.any(mask):
                    x_values = np.asarray(altitudes)[mask]
                    y_values = np.asarray(corr, dtype=float)[mask]
                    ax.scatter(
                        x_values,
                        y_values,
                        marker=correlation_duration_marker(int(duration_bin)),
                        s=32,
                        color=line.get_color(),
                        edgecolors="black",
                        linewidths=0.45,
                        zorder=4,
                    )
                    combined_ax.scatter(
                        x_values,
                        y_values,
                        marker=correlation_duration_marker(int(duration_bin)),
                        s=32,
                        color=combined_line.get_color(),
                        edgecolors="black",
                        linewidths=0.45,
                        zorder=4,
                    )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylim(-1, 1)
        ax.set_xlabel("Altitude (km)")
        ax.set_ylabel("Pearson r")
        ax.grid(True, alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        add_outside_legend(fig, handles, labels, loc="upper center", ncols=4)
        add_record_length_legend(ax, model_duration_bins)
        save_figure(
            fig,
            "correlation",
            f"{safe_name(model_name)}_hasdm_model_error_correlation_by_altitude.pgf",
        )
        combined_ax.axhline(0, color="black", linewidth=0.8)
        combined_ax.set_ylabel(f"{model_name}\nPearson r")
        combined_ax.grid(True, alpha=0.25)
        combined_ax.set_title(model_name, loc="left")
    combined_axes[-1].set_xlabel("Altitude (km)")
    combined_axes[0].set_ylim(-1, 1)
    handles, labels = combined_axes[0].get_legend_handles_labels()
    combined_fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncols=4,
        fontsize=7,
        borderaxespad=0,
    )
    record_handles = record_length_handles(combined_duration_bins)
    if record_handles:
        combined_fig.legend(
            record_handles,
            [handle.get_label() for handle in record_handles],
            title="Record length",
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
            ncols=min(3, len(record_handles)),
            fontsize=7,
            title_fontsize=8,
            borderaxespad=0,
        )
    combined_fig.suptitle("HASDM model-error correlations by altitude", y=0.985)
    save_figure(
        combined_fig,
        "correlation",
        "all_models_hasdm_model_error_correlation_by_altitude.pgf",
    )


def plot_binned(df: pl.DataFrame) -> None:
    rows = []
    for driver in ["F10.7_OBS_CENTER81", "AP_AVG", "KP_SUM"]:
        values = df[driver].to_numpy().astype(float)
        mean = float(np.nanmean(values))
        std = float(np.nanstd(values))
        if not np.isfinite(std) or std == 0:
            continue
        work = df.with_columns(((pl.col(driver) - mean) / std).alias("driver_sigma"))
        edges = np.arange(
            np.floor(np.nanmin((values - mean) / std)),
            np.ceil(np.nanmax((values - mean) / std)) + 1,
            1.0,
        )
        for model_name in MODEL_VERSIONS:
            for altitude in available_error_altitudes(df, model_name):
                col = error_col(model_name, altitude)
                if col not in df.columns:
                    continue
                source = work.select("date", "driver_sigma", "CO2_ppm", col)
                for idx in range(len(edges) - 1):
                    subset = source.filter(
                        (pl.col("driver_sigma") >= edges[idx])
                        & (pl.col("driver_sigma") < edges[idx + 1])
                    )
                    if subset.height < 20:
                        continue
                    mean_error = subset[col].mean()
                    std_error = subset[col].std()
                    if mean_error is None:
                        continue
                    co2 = subset["CO2_ppm"].to_numpy().astype(float)
                    error = subset[col].to_numpy().astype(float)
                    mask = np.isfinite(co2) & np.isfinite(error)
                    if np.sum(mask) >= 20 and np.std(co2[mask]) > 0:
                        (
                            slope,
                            slope_lo,
                            slope_hi,
                            _se,
                            _int,
                            co2_correlation,
                            _rmse,
                            _n,
                        ) = ols_slope_ci(co2[mask], error[mask])
                        co2_slope = float(slope)
                        _, corr_lo, corr_hi, _ = pearsonr_ci(co2[mask], error[mask])
                    else:
                        co2_correlation = np.nan
                        co2_slope = np.nan
                        slope_lo = np.nan
                        slope_hi = np.nan
                        corr_lo = np.nan
                        corr_hi = np.nan
                    valid_dates = (
                        subset.filter(
                            pl.col(col).is_not_null() & pl.col(col).is_finite()
                        )
                        .select("date")
                        .drop_nulls()
                        .sort("date")
                    )
                    duration_bin_years = np.nan
                    if valid_dates.height > 0:
                        duration_years = (
                            valid_dates["date"].max() - valid_dates["date"].min()
                        ).days / 365.2425
                        duration_bin_years = correlation_duration_bin(duration_years)
                    rows.append(
                        {
                            "model": model_name,
                            "altitude_km": altitude,
                            "driver": driver,
                            "driver_sigma_min": float(edges[idx]),
                            "driver_sigma_max": float(edges[idx + 1]),
                            "mean_error": float(mean_error),
                            "std_error": (
                                float(std_error) if std_error is not None else np.nan
                            ),
                            "co2_correlation": co2_correlation,
                            "co2_corr_lo": corr_lo,
                            "co2_corr_hi": corr_hi,
                            "co2_slope": co2_slope,
                            "co2_slope_lo": slope_lo,
                            "co2_slope_hi": slope_hi,
                            "n": subset.height,
                            "duration_bin_years": duration_bin_years,
                        }
                    )
    table = pl.DataFrame(rows)
    table.write_csv(
        OUTPUT_ROOT / "correlation" / "hasdm_msis_model_error_binned_stats.csv"
    )
    for model_name in MODEL_VERSIONS:
        fig, axes = plt.subplots(
            1, 3, figsize=fig_size(1.0, 0.56), sharey=True, constrained_layout=False
        )
        fig.subplots_adjust(left=0.11, right=0.98, top=0.7, bottom=0.32, wspace=0.08)
        for ax, driver in zip(axes, ["F10.7_OBS_CENTER81", "AP_AVG", "KP_SUM"]):
            subset = table.filter(
                (pl.col("model") == model_name) & (pl.col("driver") == driver)
            )
            sigma_bins = (
                subset.select("driver_sigma_min", "driver_sigma_max")
                .unique()
                .sort("driver_sigma_min")
            )
            for sigma_row in sigma_bins.iter_rows(named=True):
                series = subset.filter(
                    (pl.col("driver_sigma_min") == sigma_row["driver_sigma_min"])
                    & (pl.col("driver_sigma_max") == sigma_row["driver_sigma_max"])
                ).sort("altitude_km")
                if not series.is_empty():
                    (line,) = ax.plot(
                        series["altitude_km"],
                        series["co2_correlation"],
                        linewidth=1.0,
                        label=f"{sigma_row['driver_sigma_min']:g} to {sigma_row['driver_sigma_max']:g} sigma",
                        zorder=3,
                    )
                    for duration_bin in sorted(
                        set(series["duration_bin_years"].to_list())
                    ):
                        if not np.isfinite(duration_bin):
                            continue
                        points = series.filter(
                            pl.col("duration_bin_years") == duration_bin
                        )
                        ax.scatter(
                            points["altitude_km"],
                            points["co2_correlation"],
                            marker=correlation_duration_marker(int(duration_bin)),
                            s=32,
                            color=line.get_color(),
                            edgecolors="black",
                            linewidths=0.45,
                            zorder=4,
                        )
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_title(label_for_col(driver))
            ax.set_xlabel("Altitude (km)")
            ax.grid(True, alpha=0.25)
        axes[0].set_ylabel(r"Pearson r($\epsilon_m$, CO$_2$)")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                title="Activity bin",
                loc="upper center",
                bbox_to_anchor=(0.5, 0.985),
                ncols=3,
                fontsize=6.2,
                title_fontsize=7,
                borderaxespad=0,
                handlelength=1.8,
                columnspacing=0.9,
            )
        record_handles = record_length_handles(
            {
                int(value)
                for value in table["duration_bin_years"].drop_nulls().to_list()
                if np.isfinite(value)
            }
            if table.height
            else set()
        )
        if record_handles:
            fig.legend(
                record_handles,
                [handle.get_label() for handle in record_handles],
                title="Record length",
                loc="lower center",
                bbox_to_anchor=(0.5, 0.005),
                ncols=min(3, len(record_handles)),
                fontsize=7,
                title_fontsize=7,
                borderaxespad=0,
            )
        save_figure(
            fig, "correlation", f"{safe_name(model_name)}_hasdm_model_error_binned.pgf"
        )
    plot_co2_slope_binned(table)
    plot_co2_correlation_binned(table)


def plot_co2_slope_binned(table: pl.DataFrame) -> None:
    drivers = ["F10.7_OBS_CENTER81", "AP_AVG"]
    duration_bins = (
        {
            int(value)
            for value in table["duration_bin_years"].drop_nulls().to_list()
            if np.isfinite(value)
        }
        if table.height
        else set()
    )

    for driver in drivers:
        fig, axes = plt.subplots(
            len(MODEL_VERSIONS),
            1,
            figsize=fig_size(1.0, 1.05),
            sharex=True,
            sharey=True,
            constrained_layout=False,
        )
        fig.subplots_adjust(left=0.14, right=0.98, top=0.74, bottom=0.16, hspace=0.42)
        if len(MODEL_VERSIONS) == 1:
            axes = np.asarray([axes])

        for row_idx, model_name in enumerate(MODEL_VERSIONS):
            ax = axes[row_idx]
            subset = (
                table.filter(
                    (pl.col("model") == model_name) & (pl.col("driver") == driver)
                )
                .drop_nulls("co2_slope")
                .with_columns(
                    (pl.col("co2_slope") * pl.col("n")).alias("weighted_slope"),
                    (
                        pl.when(pl.col("co2_slope_lo").is_not_null())
                        .then(pl.col("co2_slope_lo") * pl.col("n"))
                        .otherwise(0.0)
                    ).alias("weighted_slope_lo"),
                    (
                        pl.when(pl.col("co2_slope_hi").is_not_null())
                        .then(pl.col("co2_slope_hi") * pl.col("n"))
                        .otherwise(0.0)
                    ).alias("weighted_slope_hi"),
                )
                .group_by("driver_sigma_min", "driver_sigma_max", "altitude_km")
                .agg(
                    pl.col("weighted_slope").sum().alias("weighted_slope"),
                    pl.col("weighted_slope_lo").sum().alias("weighted_slope_lo"),
                    pl.col("weighted_slope_hi").sum().alias("weighted_slope_hi"),
                    pl.col("n").sum().alias("n"),
                    pl.col("duration_bin_years").first().alias("duration_bin_years"),
                )
                .with_columns(
                    (pl.col("weighted_slope") / pl.col("n")).alias("co2_slope"),
                    (pl.col("weighted_slope_lo") / pl.col("n")).alias(
                        "co2_slope_lo_agg"
                    ),
                    (pl.col("weighted_slope_hi") / pl.col("n")).alias(
                        "co2_slope_hi_agg"
                    ),
                )
                .sort("driver_sigma_min", "altitude_km")
            )
            sigma_bins = (
                subset.select("driver_sigma_min", "driver_sigma_max")
                .unique()
                .sort("driver_sigma_min")
            )
            for sigma_row in sigma_bins.iter_rows(named=True):
                series = subset.filter(
                    (pl.col("driver_sigma_min") == sigma_row["driver_sigma_min"])
                    & (pl.col("driver_sigma_max") == sigma_row["driver_sigma_max"])
                ).sort("altitude_km")
                if series.is_empty():
                    continue
                (line,) = ax.plot(
                    series["altitude_km"],
                    series["co2_slope"],
                    linewidth=1.0,
                    label=f"{sigma_row['driver_sigma_min']:g} to {sigma_row['driver_sigma_max']:g} sigma",
                    zorder=3,
                )
                lo_vals = series["co2_slope_lo_agg"].to_numpy().astype(float)
                hi_vals = series["co2_slope_hi_agg"].to_numpy().astype(float)
                slope_vals = series["co2_slope"].to_numpy().astype(float)
                alt_vals = series["altitude_km"].to_numpy().astype(float)
                finite_ci = (
                    np.isfinite(slope_vals)
                    & np.isfinite(lo_vals)
                    & np.isfinite(hi_vals)
                )
                if np.any(finite_ci):
                    ax.fill_between(
                        alt_vals[finite_ci],
                        lo_vals[finite_ci],
                        hi_vals[finite_ci],
                        alpha=0.15,
                        color=line.get_color(),
                        zorder=2,
                    )
                for duration_bin in sorted(set(series["duration_bin_years"].to_list())):
                    if not np.isfinite(duration_bin):
                        continue
                    points = series.filter(pl.col("duration_bin_years") == duration_bin)
                    ax.scatter(
                        points["altitude_km"],
                        points["co2_slope"],
                        marker=correlation_duration_marker(int(duration_bin)),
                        s=32,
                        color=line.get_color(),
                        edgecolors="black",
                        linewidths=0.45,
                        zorder=4,
                    )
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_title(f"{model_name} by {label_for_col(driver)}")
            ax.grid(True, alpha=0.25)
            if row_idx == len(MODEL_VERSIONS) // 2:
                ax.set_ylabel(r"CO$_2$ fitted slope in $\epsilon_m$ per ppm")
            if row_idx == len(MODEL_VERSIONS) - 1:
                ax.set_xlabel("Altitude (km)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            title="Activity bin",
            loc="upper center",
            bbox_to_anchor=(0.5, 0.93),
            ncols=3,
            fontsize=7,
            title_fontsize=8,
            borderaxespad=0,
        )
        record_handles = record_length_handles(duration_bins)
        if record_handles:
            fig.legend(
                record_handles,
                [handle.get_label() for handle in record_handles],
                title="Record length",
                loc="lower center",
                bbox_to_anchor=(0.5, 0.005),
                ncols=min(3, len(record_handles)),
                fontsize=7,
                title_fontsize=8,
                borderaxespad=0,
            )
        fig.suptitle(
            f"HASDM model-error CO$_2$ slope by altitude and {label_for_col(driver)} activity bin",
            y=0.985,
        )
        save_figure(
            fig,
            "correlation",
            f"hasdm_msis_model_error_co2_slope_binned_{safe_name(driver)}.pgf",
        )


def plot_co2_correlation_binned(table: pl.DataFrame) -> None:
    drivers = ["F10.7_OBS_CENTER81", "AP_AVG"]
    fig, axes = plt.subplots(
        len(MODEL_VERSIONS),
        len(drivers),
        figsize=fig_size(1.0, 1.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(len(MODEL_VERSIONS), len(drivers))
    for row_idx, model_name in enumerate(MODEL_VERSIONS):
        for col_idx, driver in enumerate(drivers):
            ax = axes[row_idx, col_idx]
            subset = table.filter(
                (pl.col("model") == model_name) & (pl.col("driver") == driver)
            )
            sigma_bins = (
                subset.select("driver_sigma_min", "driver_sigma_max")
                .unique()
                .sort("driver_sigma_min")
            )
            for sigma_row in sigma_bins.iter_rows(named=True):
                series = subset.filter(
                    (pl.col("driver_sigma_min") == sigma_row["driver_sigma_min"])
                    & (pl.col("driver_sigma_max") == sigma_row["driver_sigma_max"])
                ).sort("altitude_km")
                if series.is_empty():
                    continue
                (line,) = ax.plot(
                    series["altitude_km"],
                    series["co2_correlation"],
                    linewidth=1.0,
                    label=f"{sigma_row['driver_sigma_min']:g} to {sigma_row['driver_sigma_max']:g} sigma",
                    zorder=3,
                )
                if "co2_corr_lo" in series.columns and "co2_corr_hi" in series.columns:
                    lo_vals = series["co2_corr_lo"].to_numpy().astype(float)
                    hi_vals = series["co2_corr_hi"].to_numpy().astype(float)
                    corr_vals = series["co2_correlation"].to_numpy().astype(float)
                    alt_vals = series["altitude_km"].to_numpy().astype(float)
                    finite_ci = (
                        np.isfinite(corr_vals)
                        & np.isfinite(lo_vals)
                        & np.isfinite(hi_vals)
                    )
                    if np.any(finite_ci):
                        ax.fill_between(
                            alt_vals[finite_ci],
                            lo_vals[finite_ci],
                            hi_vals[finite_ci],
                            alpha=0.15,
                            color=line.get_color(),
                            zorder=2,
                        )
                for duration_bin in sorted(set(series["duration_bin_years"].to_list())):
                    if not np.isfinite(duration_bin):
                        continue
                    points = series.filter(pl.col("duration_bin_years") == duration_bin)
                    ax.scatter(
                        points["altitude_km"],
                        points["co2_correlation"],
                        marker=correlation_duration_marker(int(duration_bin)),
                        s=32,
                        color=line.get_color(),
                        edgecolors="black",
                        linewidths=0.45,
                        zorder=4,
                    )
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_title(f"{model_name} by {label_for_col(driver)}")
            ax.grid(True, alpha=0.25)
            if col_idx == 0:
                ax.set_ylabel(r"Pearson r($\epsilon_m$, CO$_2$)")
            if row_idx == len(MODEL_VERSIONS) - 1:
                ax.set_xlabel("Altitude (km)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    add_outside_legend(
        fig, handles, labels, loc="upper center", title="Activity bin", ncols=3
    )
    add_record_length_legend(
        axes[0, 0],
        {
            int(value)
            for value in table["duration_bin_years"].drop_nulls().to_list()
            if np.isfinite(value)
        }
        if table.height
        else set(),
    )
    fig.suptitle("HASDM model-error CO$_2$ correlation by altitude and activity bin")
    save_figure(fig, "correlation", "hasdm_msis_model_error_co2_correlation_binned.pgf")


def write_causal_inputs(df: pl.DataFrame) -> None:
    causal_input_dir = Path("data/products/causal_discovery")
    causal_input_dir.mkdir(parents=True, exist_ok=True)
    cols = ["date", *ANALYSIS_COLS, *[col for col in SABER_COLS if col in df.columns]]
    for model_name in MODEL_VERSIONS:
        for stat in ["mean", "range"]:
            col = error_col(model_name, 500, stat=stat)
            if col in df.columns:
                cols.append(col)
    df.select(cols).write_csv(causal_input_dir / "hasdm_msis_500km_causal_input.csv")


def write_selection(grid: GridPoint) -> None:
    text = (
        f"Mauna Loa lat={MAUNA_LOA_LAT}, lon_east={MAUNA_LOA_LON_EAST}\n"
        f"Nearest HASDM latitude={grid.lat}; nearest available longitude is selected independently for each timestamp and altitude\n"
        f"Error definition=ln(rho_model/rho_HASDM)\n"
        "Plots use all available HASDM altitudes in the generated daily model-error columns; 500 km remains the causal/Kaggle export target.\n"
    )
    (OUTPUT_ROOT / "hasdm_msis_selection.txt").write_text(text, encoding="utf-8")


def plot_causal_target_summary() -> None:
    causal_dir = OUTPUT_ROOT / "causal"
    paths = sorted(
        causal_dir.glob("links_*_daily_range_500km_gpdctorch_pcmciplus_daily_7d.csv")
    )
    if not paths:
        return
    frames = [pl.read_csv(path) for path in paths]
    links = pl.concat(frames, how="vertical_relaxed")
    rows = []
    cause_order = ["F10.7_OBS_CENTER81", "AP_AVG", "CO2_ppm", "saber_co2cool_max_alt"]
    for model_name in MODEL_VERSIONS:
        target = error_col(model_name, 500, stat="range")
        target_links = links.filter(pl.col("target") == target)
        for cause in cause_order:
            value = target_links.filter(pl.col("cause") == cause)["mci_value"].max()
            rows.append(
                {
                    "model": model_name,
                    "cause": cause,
                    "max_mci": float(value) if value is not None else np.nan,
                }
            )
    table = pl.DataFrame(rows)
    if table.is_empty():
        return
    matrix = np.full((len(MODEL_VERSIONS), len(cause_order)), np.nan)
    for row_idx, model_name in enumerate(MODEL_VERSIONS):
        for col_idx, cause in enumerate(cause_order):
            value = table.filter(
                (pl.col("model") == model_name) & (pl.col("cause") == cause)
            )["max_mci"]
            if len(value):
                matrix[row_idx, col_idx] = value[0]
    fig, ax = plt.subplots(figsize=fig_size(0.82, 0.55), constrained_layout=True)
    image = ax.imshow(np.ma.masked_invalid(matrix), cmap="magma", vmin=0)
    ax.set_xticks(np.arange(len(cause_order)), cause_order, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(MODEL_VERSIONS)), list(MODEL_VERSIONS))
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if np.isfinite(matrix[row, col]):
                ax.text(
                    col,
                    row,
                    f"{matrix[row, col]:.3f}",
                    ha="center",
                    va="center",
                    color="white" if matrix[row, col] > 0.1 else "black",
                    fontsize=8,
                )
    fig.colorbar(image, ax=ax, label="maximum MCI into 500 km range-error target")
    ax.set_title("HASDM model-error causal target summary")
    save_figure(fig, "causal", "hasdm_msis_model_error_causal_target_summary.pgf")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "correlation").mkdir(parents=True, exist_ok=True)
    grid = GridPoint(nearest_hasdm_latitude())
    write_selection(grid)
    samples = compute_sample_errors()
    daily = aggregate_daily_errors(samples)
    wide = pivot_daily(daily)
    df = combine_analysis_dataset(wide)
    hasdm_date_limits = (wide["date"].min(), wide["date"].max())
    write_causal_inputs(df)
    plot_time_series(df, hasdm_date_limits)
    plot_fft(df)
    plot_fft(
        df,
        {"NRLMSIS 2.0": MODEL_VERSIONS["NRLMSIS 2.0"]},
        "hasdm_msis_model_error_fft_nrlmsis_2p0.pgf",
    )
    plot_correlation_by_altitude(df)
    plot_binned(df)
    plot_causal_target_summary()
    print(f"Generated HASDM MSIS model-error outputs in {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
