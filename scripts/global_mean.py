# Zed REPL-compatible export of causality.ipynb
# Ruff: configure_pgf() must run before pyplot imports; suppress intentional E402.
# ruff: noqa: E402
# Cells are separated with # %% markers so they can be run incrementally.

# %%
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from scripts.pgf_config import TEXTWIDTH_IN, configure_pgf, fig_size, page_fig_size

configure_pgf()

import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import polars as pl

from scripts.stats_utils import ols_slope_ci, pearsonr_ci
from thermodense.downloader.space_weather import SPACE_WEATHER_CSV_PATH  # noqa: E402

GLOBAL_MEAN_PATH = (
    "data/decoded/orbit_derived_global_mean/orbit-density-ds03-density-values.parquet"
)
CO2_PATH = "data/original/co2/co2_daily_mlo.csv"
MGII_PATH = "data/original/MgII/MgII_composite.dat"
SW_PATH = str(SPACE_WEATHER_CSV_PATH)
OUTPUT_ROOT = Path("outputs")
FIGURE_ROOT = Path("outputs/figures/results/global_mean")
LATEX_FIGURE_INDEX = "figures.tex"
FIGURE_EXTENSIONS = {".pdf", ".pgf", ".png", ".jpg", ".jpeg"}

F10_7_RANGES = [(0, 100), (100, 160), (160, 400), (0, 400)]  # (100, 400), (0, 160),
ANALYSIS_COLS = ["F10.7_OBS_CENTER81", "AP_AVG", "KP_SUM", "CO2_ppm"]
ANALYSIS_LABELS = ["F$_{10.7,81}$", "$A_p$", "$K_p$", "CO$_2$"]
CORRELATION_DURATION_STEP_YEARS = 11
CORRELATION_DURATION_MARKERS = {
    0: "o",
    11: "s",
    22: "D",
    33: "^",
    44: "P",
    55: "X",
}
SPACE_WEATHER_SIGMA_COLS = ["F10.7_OBS_CENTER81", "AP_AVG", "KP_SUM"]
SPACE_WEATHER_SIGMA_LABELS = {
    "F10.7_OBS_CENTER81": "F$_{10.7,81}$",
    "AP_AVG": "$A_p$",
    "KP_SUM": "$K_p$",
}
ALTITUDES = [250, 275, 325, 375, 400, 425, 475, 525, 550, 575]
SELECTED_SCATTER_ALTITUDES = [250, 425, 575]
RESULTS_HEATMAP_ALTITUDES = [250, 325, 400, 475, 575]
FFT_HEATMAP_PERIOD_BINS_PER_DECADE = 24
FFT_HEATMAP_ALTITUDE_BIN_KM = 50
MIN_SAMPLES_PER_HEATMAP_CELL = 10
HEATMAP_COLORBAR_HEIGHT_INCHES = 3.1
FIT_METRIC_NAMES = [
    "correlation",
    "slope",
    "zero_crossing",
    "error",
    "sample_count",
]
SPECIAL_PERIODS_YEARS = np.array([0.5, 1.0])
FFT_PERIOD_TICKS = [
    (2 / 365.25, "2 d"),
    (7 / 365.25, "1 wk"),
    (27 / 365.25, "27 d"),
    (0.5, "6 mo"),
    (1.0, "1 y"),
    # (11.0, "11 y"),
    # (14 / 365.25, "2 wk"),
    # (21 / 365.25, "3 wk"),
    # (28 / 365.25, "4 wk"),
    # (1 / 12.0, "1 mo"),
]
FFT_PERIOD_TICKS_YEARS = np.array([period for period, _ in FFT_PERIOD_TICKS])
SCATTER_RASTERIZE_PANEL_POINTS = 2_000
SCATTER_RASTERIZE_FIGURE_POINTS = 10_000


def format_altitude_axis(ax: plt.Axes, axis: str = "y") -> None:
    if axis == "y":
        ax.set_yscale("linear")
        target = ax.yaxis
    else:
        ax.set_xscale("linear")
        target = ax.xaxis
    target.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
    target.get_offset_text().set_visible(False)


def scatter_rasterized(panel_points: int, figure_points: int | None = None) -> bool:
    return panel_points > SCATTER_RASTERIZE_PANEL_POINTS or (
        figure_points is not None and figure_points > SCATTER_RASTERIZE_FIGURE_POINTS
    )


START_YEAR = 1966
END_YEAR = 2020


def load_global_mean() -> tuple[pl.DataFrame, list[str]]:
    df = pl.read_parquet(GLOBAL_MEAN_PATH)
    rho_cols = [c for c in df.columns if c.startswith("log10rho_")]
    df = df.with_columns([pl.col(c).interpolate().alias(c) for c in rho_cols])
    df = df.with_columns(
        [
            pl.when(pl.col(c) < -200).then(None).otherwise(pl.col(c)).alias(c)
            for c in df.columns
            if c != "date"
        ]
    )
    return df, rho_cols


def load_co2() -> pl.DataFrame:
    schema = {
        "year": pl.Int32,
        "month": pl.Int32,
        "day": pl.Int32,
        "year_decimal": pl.Float32,
        "CO2_ppm": pl.Float64,
    }
    return (
        pl.read_csv(
            CO2_PATH,
            has_header=False,
            schema=schema,
            comment_prefix="#",
        )
        .with_columns(
            pl.date(pl.col("year"), pl.col("month"), pl.col("day")).alias("date")
        )
        .drop("year", "month", "day", "year_decimal")
    )


def load_space_weather(f10_min: int, f10_max: int) -> pl.DataFrame:
    return (
        pl.read_csv(SW_PATH)
        .with_columns(pl.col("DATE").str.to_date("%Y-%m-%d").alias("date"))
        .filter(
            (pl.col("F10.7_OBS_CENTER81") > f10_min)
            & (pl.col("F10.7_OBS_CENTER81") < f10_max)
        )
        .drop("DATE")
    )


def load_mgii() -> pl.DataFrame:
    with open(MGII_PATH, "r") as f:
        cleaned_text = "\n".join(re.sub(r"(?<=\w)\s", ",", line) for line in f)

    schema = {
        "year_decimal": pl.Float32,
        "month": pl.Float32,
        "day": pl.Float32,
        "MgII": pl.Float32,
        "MgII_uncert": pl.Float32,
        "source_id": pl.Int32,
        "ignore": pl.String,
    }
    return (
        pl.read_csv(
            cleaned_text.encode(),
            separator=",",
            comment_prefix=";",
            has_header=False,
            schema=schema,
        )
        .drop("ignore")
        .with_columns(
            pl.date(
                pl.col("year_decimal").cast(pl.Int32), pl.col("month"), pl.col("day")
            ).alias("date")
        )
        .drop("year_decimal", "month", "day", "source_id")
    )


def as_date_index(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(pl.col("date").cast(pl.Date))
        .filter(pl.col("date").is_not_null())
        .unique(subset="date")
        .sort("date")
    )


def combined_dataset(
    df_global: pl.DataFrame,
    df_co2: pl.DataFrame,
    df_sw: pl.DataFrame,
    df_mgii: pl.DataFrame,
) -> pl.DataFrame:
    df_combined = as_date_index(df_global)
    for dataset in [df_co2, df_sw, df_mgii]:
        df_combined = df_combined.join(
            as_date_index(dataset),
            on="date",
            how="full",
            coalesce=True,
        )
    return df_combined.sort("date").select("date", pl.exclude("date"))


def finite_xy(
    df: pl.DataFrame, x_col: str, y_col: str
) -> tuple[np.ndarray, np.ndarray]:
    if x_col == y_col:
        pair = df.select(x_col).drop_nulls()
        x = pair[x_col].to_numpy()
        y = x
    else:
        pair = df.select(x_col, y_col).drop_nulls()
        x = pair[x_col].to_numpy()
        y = pair[y_col].to_numpy()
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


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
    return f"{bin_start} to {bin_start + CORRELATION_DURATION_STEP_YEARS} yr"


def correlation_duration_marker(bin_start: int) -> str:
    return CORRELATION_DURATION_MARKERS.get(bin_start, "*")


def correlation_and_duration(
    df: pl.DataFrame, x_col: str, y_col: str
) -> tuple[float, float, float, int | None]:
    data = df.select("date", x_col, y_col).drop_nulls().sort("date")
    if data.height < 3:
        return np.nan, np.nan, np.nan, None
    x = data[x_col].to_numpy().astype(float)
    y = data[y_col].to_numpy().astype(float)
    finite = np.isfinite(x) & np.isfinite(y)
    if int(np.count_nonzero(finite)) < 3:
        return np.nan, np.nan, np.nan, None
    x = x[finite]
    y = y[finite]
    if np.std(x) <= 0 or np.std(y) <= 0:
        return np.nan, np.nan, np.nan, None
    dates = np.asarray(data["date"].to_list(), dtype=object)[finite]
    duration_years = (dates[-1] - dates[0]).days / 365.2425
    r, r_lo, r_hi, _ = pearsonr_ci(x, y)
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
    duration_handles = [
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
        handles=duration_handles,
        title="Record length",
        fontsize=7,
        title_fontsize=7,
        loc="upper right",
        bbox_to_anchor=(1, -0.22),
        borderaxespad=0,
        ncol=min(2, len(duration_handles)),
    )


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


def sigma_bin_labels(edges: np.ndarray) -> list[str]:
    return [f"{edges[idx]:g}-{edges[idx + 1]:g}" for idx in range(len(edges) - 1)]


def linear_fit_stats(
    x: np.ndarray, y: np.ndarray
) -> tuple[float, float, float, float, float, float]:
    if len(x) < MIN_SAMPLES_PER_HEATMAP_CELL:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    slope, slope_lo, slope_hi, slope_se, intercept, correlation, error, _ = (
        ols_slope_ci(x, y)
    )
    zero_crossing = float(-intercept / slope) if slope != 0 else np.nan
    return correlation, float(slope), zero_crossing, error, slope_lo, slope_hi


def fit_annotation(
    correlation: float,
    slope: float,
    zero_crossing: float,
    error: float,
    count: int,
    slope_lo: float = np.nan,
    slope_hi: float = np.nan,
) -> str:
    ci_str = ""
    if np.isfinite(slope_lo) and np.isfinite(slope_hi):
        ci_str = f"\nCI=[{slope_lo:.2e}, {slope_hi:.2e}]"
    return (
        f"r={correlation:.2f}\n"
        f"m={slope:.2e}{ci_str}\n"
        f"x0={zero_crossing:.1f}\n"
        f"err={error:.3f}\n"
        f"n={count}"
    )


def driver_output_name(driver_col: str) -> str:
    return driver_col.replace(".", "p").replace("_", "-").lower()


def metric_plot_path(output_dir: Path, driver_col: str, metric_name: str) -> Path:
    output_name = driver_output_name(driver_col)
    return (
        output_dir
        / f"global_mean_density_co2_{metric_name}_by_altitude_for_{output_name}.png"
    )


def cleanup_metric_plots_for_driver(output_dir: Path, driver_col: str):
    for metric_name in FIT_METRIC_NAMES:
        path = metric_plot_path(output_dir, driver_col, metric_name)
        for candidate in [path, path.with_suffix(".pgf")]:
            if candidate.exists():
                candidate.unlink()


def plot_metric_by_altitude_and_sigma(
    output_dir: Path,
    driver_col: str,
    altitudes: list[int],
    row_labels: np.ndarray,
    metric_name: str,
    metric_label: str,
    values: np.ndarray,
    duration_bins: np.ndarray | None = None,
    ci_values: np.ndarray | None = None,
):
    output_path = metric_plot_path(output_dir, driver_col, metric_name)
    if values.size == 0 or not np.any(np.isfinite(values)):
        if output_path.exists():
            output_path.unlink()
        return

    if metric_name == "sample_count":
        counts = []
        for row in range(len(row_labels)):
            row_values = values[row].astype(float)
            finite = row_values[np.isfinite(row_values)]
            counts.append(float(finite[0]) if finite.size else np.nan)
        fig, ax = plt.subplots(figsize=fig_size(1.0, 0.5), constrained_layout=True)
        positions = np.arange(len(row_labels))
        colors = [plt.get_cmap("tab10")(idx) for idx in range(len(row_labels))]
        ax.bar(positions, counts, color=colors, alpha=0.8)
        ax.set_xticks(positions, row_labels, rotation=25, ha="right")
        ax.set_xlabel(f"{SPACE_WEATHER_SIGMA_LABELS[driver_col]} activity bin")
        ax.set_ylabel(metric_label)
        ax.grid(True, axis="y", alpha=0.3)
        save_and_close(fig, output_path)
        return

    fig, ax = plt.subplots(figsize=fig_size(1.0, 0.5), constrained_layout=True)
    observed_duration_bins: set[int] = set()
    for row, sigma_label in enumerate(row_labels):
        row_values = values[row].astype(float)
        if not np.any(np.isfinite(row_values)):
            continue
        (line,) = ax.plot(
            altitudes,
            row_values,
            linewidth=1.5,
            label=sigma_label,
            zorder=3,
        )
        if ci_values is not None:
            row_lo = ci_values[row, :, 0].astype(float)
            row_hi = ci_values[row, :, 1].astype(float)
            finite_ci = (
                np.isfinite(row_values) & np.isfinite(row_lo) & np.isfinite(row_hi)
            )
            if np.any(finite_ci):
                ax.fill_between(
                    np.asarray(altitudes)[finite_ci],
                    row_lo[finite_ci],
                    row_hi[finite_ci],
                    alpha=0.15,
                    color=line.get_color(),
                    zorder=2,
                )
        if duration_bins is not None:
            row_duration_bins = duration_bins[row]
            for duration_bin in sorted(set(row_duration_bins.tolist()) - {None}):
                if not np.isfinite(duration_bin):
                    continue
                mask = (row_duration_bins == duration_bin) & np.isfinite(row_values)
                if np.any(mask):
                    observed_duration_bins.add(int(duration_bin))
                    ax.scatter(
                        np.asarray(altitudes)[mask],
                        row_values[mask],
                        marker=correlation_duration_marker(int(duration_bin)),
                        s=32,
                        color=line.get_color(),
                        edgecolors="black",
                        linewidths=0.45,
                        zorder=4,
                    )

    ax.set_xlabel("Altitude (km)")
    ax.set_ylabel(metric_label)
    ax.grid(True, alpha=0.3)
    data_legend = ax.legend(
        title=f"{SPACE_WEATHER_SIGMA_LABELS[driver_col]} $\\sigma$ bin",
        fontsize=7,
        title_fontsize=7,
        loc="upper left",
        bbox_to_anchor=(-0.13, -0.22),
        ncol=5,
        borderaxespad=0,
    )
    ax.add_artist(data_legend)
    add_record_length_legend(ax, observed_duration_bins)
    save_and_close(fig, output_path)


def text_color_for_correlation(value: float) -> str:
    if not np.isfinite(value):
        return "black"
    return "white" if abs(value) > 0.55 else "black"


def fft_period_power(df: pl.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
    series = (
        df.select("date", col)
        .sort("date")
        .with_columns(
            pl.col(col)
            .interpolate()
            .fill_null(strategy="forward")
            .fill_null(strategy="backward")
        )
        .drop_nulls()
    )
    values = series[col].to_numpy()
    if len(values) < 3:
        return np.array([]), np.array([])
    values = values - np.mean(values)
    days = series["date"].cast(pl.Int64).to_numpy()
    sample_days = np.median(np.diff(days))
    frequencies = np.fft.rfftfreq(len(values), d=sample_days)
    amplitudes = np.abs(np.fft.rfft(values))
    valid = frequencies > 0
    return 1 / frequencies[valid], amplitudes[valid]


def altitude_from_density_col(col: str) -> int | None:
    match = re.search(r"log10rho_(\d+)", col)
    return int(match.group(1)) if match else None


def log_period_edges(periods_years: np.ndarray) -> np.ndarray:
    finite = periods_years[np.isfinite(periods_years) & (periods_years > 0)]
    if len(finite) == 0:
        return np.array([])
    span_decades = np.log10(np.max(finite)) - np.log10(np.min(finite))
    n_bins = max(24, int(np.ceil(span_decades * FFT_HEATMAP_PERIOD_BINS_PER_DECADE)))
    return np.logspace(np.log10(np.min(finite)), np.log10(np.max(finite)), n_bins + 1)


def altitude_edges(altitudes: list[int]) -> np.ndarray:
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


def plot_fft_altitude_heatmap(
    ax: plt.Axes, df: pl.DataFrame, rho_cols: list[str]
) -> None:
    spectra = []
    all_periods = []
    for col in rho_cols:
        altitude = altitude_from_density_col(col)
        if altitude is None:
            continue
        periods_days, amplitudes = fft_period_power(df, col)
        if len(periods_days) == 0:
            continue
        periods_years = periods_days / 365.25
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
    alts = sorted({altitude for altitude, _, _ in spectra})
    alt_edges = altitude_edges(alts)
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
    ax.set_ylabel("Altitude (km)")
    format_altitude_axis(ax)
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.18)
    ax.figure.colorbar(mesh, ax=ax, label="$\\bar{\\ell_\\rho}$")


def regular_daily_series(df: pl.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
    series = (
        df.select("date", col)
        .drop_nulls()
        .group_by("date")
        .agg(pl.col(col).mean())
        .sort("date")
    )
    days = series["date"].cast(pl.Int64).to_numpy()
    values = series[col].to_numpy()
    mask = np.isfinite(days) & np.isfinite(values)
    days = days[mask]
    values = values[mask]
    if len(days) < 3:
        return np.array([]), np.array([])
    full_days = np.arange(days[0], days[-1] + 1)
    full_values = np.interp(full_days, days, values)
    return full_days, full_values


def morlet_wavelet_power(
    df: pl.DataFrame,
    col: str,
    min_period_years: float = 0.25,
    max_period_years: float = 32,
    periods_per_octave: int = 8,
    omega0: float = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    days, values = regular_daily_series(df, col)
    if len(values) < 3:
        return np.array([]), np.array([]), np.array([[]]), np.array([])

    values = values - np.mean(values)
    std = np.std(values)
    if std > 0:
        values = values / std

    max_period_years = min(max_period_years, len(values) / (2 * 365.25))
    period_count = max(
        2, int(np.log2(max_period_years / min_period_years) * periods_per_octave)
    )
    periods_years = np.geomspace(min_period_years, max_period_years, period_count)
    periods_years = np.unique(
        np.concatenate(
            [
                periods_years,
                SPECIAL_PERIODS_YEARS[
                    (SPECIAL_PERIODS_YEARS >= min_period_years)
                    & (SPECIAL_PERIODS_YEARS <= max_period_years)
                ],
            ]
        )
    )
    periods_days = periods_years * 365.25
    fourier_factor = (4 * np.pi) / (omega0 + np.sqrt(2 + omega0**2))
    scales = periods_days / fourier_factor

    padded_n = 2 ** int(np.ceil(np.log2(len(values))))
    angular_frequencies = 2 * np.pi * np.fft.fftfreq(padded_n, d=1)
    values_fft = np.fft.fft(values, padded_n)

    wavelet = np.empty((len(scales), len(values)), dtype=complex)
    for row, scale in enumerate(scales):
        daughter = (
            np.pi ** (-0.25)
            * np.sqrt(2 * np.pi * scale)
            * np.exp(-0.5 * (scale * angular_frequencies - omega0) ** 2)
            * (angular_frequencies > 0)
        )
        wavelet[row] = np.fft.ifft(values_fft * daughter)[: len(values)]

    power = np.abs(wavelet) ** 2
    distance_from_edge_days = np.minimum(days - days[0], days[-1] - days)
    cone_of_influence_years = (
        fourier_factor * distance_from_edge_days / (np.sqrt(2) * 365.25)
    )
    return days, periods_years, power, cone_of_influence_years


def add_period_xticks(ax, stagger: bool = False):
    ticks = ax.get_xticks()
    ticks = ticks[(ticks > 0) & np.isfinite(ticks)]
    ticks = np.unique(
        np.concatenate([ticks, FFT_PERIOD_TICKS_YEARS, SPECIAL_PERIODS_YEARS])
    )
    ax.set_xticks(ticks)
    labels = [fft_period_tick_label(tick) for tick in ticks]
    if stagger:
        labels = [
            label if i % 2 == 0 else f"\n{label}" for i, label in enumerate(labels)
        ]
    ax.set_xticklabels(labels, rotation=60, ha="right")


def stagger_visible_xticklabels(ax, top: bool = False) -> None:
    labels = ax.xaxis.get_ticklabels(minor=False)
    for i, label in enumerate(labels):
        if not label.get_visible():
            continue
        label.set_y(1.02 + 0.08 * (i % 2) if top else -0.04 - 0.08 * (i % 2))


def fft_period_tick_label(tick: float) -> str:
    for period, label in FFT_PERIOD_TICKS:
        if np.isclose(tick, period):
            return label
    return period_label(tick)


def period_label(tick: float) -> str:
    if np.isclose(tick, 0.5):
        return "6 mo"
    if np.isclose(tick, 1.0):
        return "1 y"
    if np.isclose(tick, 11.0):
        return "11 y"
    return f"{tick:g} y"


def add_period_yticks(ax):
    ticks = ax.get_yticks()
    ticks = ticks[(ticks > 0) & np.isfinite(ticks)]
    ticks = np.unique(np.concatenate([ticks, SPECIAL_PERIODS_YEARS]))
    ax.set_yticks(ticks)
    ax.set_yticklabels([period_label(tick) for tick in ticks])


def save_and_close(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def unlink_output(path: Path) -> None:
    if path.exists():
        path.unlink()


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


def plot_density_time_heatmap(
    ax: plt.Axes, df_global: pl.DataFrame, rho_cols: list[str]
) -> None:
    pairs = sorted(
        (altitude_from_density_col(col), col)
        for col in rho_cols
        if altitude_from_density_col(col) is not None
    )

    if not pairs:
        ax.text(
            0.5,
            0.5,
            "No density heatmap data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    altitudes = [altitude for altitude, _ in pairs]
    cols = [col for _, col in pairs]

    data = df_global.select("date", *cols).sort("date").drop_nulls("date")

    dates = data["date"].to_numpy()
    alt_edges = altitude_edges(altitudes)

    matrix_sum = np.zeros((len(alt_edges) - 1, len(dates)), dtype=float)
    matrix_count = np.zeros_like(matrix_sum)

    for altitude, col in pairs:
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

    date_edges = date_edges_for_heatmap(dates)

    mesh = ax.pcolormesh(
        date_edges, alt_edges, matrix, shading="auto", cmap="viridis", rasterized=True
    )

    # Contours use cell centers, not edges
    date_centers = mdates.date2num(dates)
    alt_centers = 0.5 * (alt_edges[:-1] + alt_edges[1:])

    if matrix.shape[0] >= 2 and matrix.shape[1] >= 2 and np.isfinite(matrix).any():
        contour_data = np.ma.masked_invalid(matrix)

        contours = ax.contour(
            date_centers,
            alt_centers,
            contour_data,
            levels=[-10, -11, -12, -13, -14][::-1],
            colors="white",
            linewidths=0.7,
            alpha=0.75,
        )

        ax.clabel(
            contours,
            inline=True,
            fontsize=7,
            fmt="%.2f",
        )

    ax.xaxis_date()
    ax.set_ylabel("Altitude (km)")
    format_altitude_axis(ax)
    ax.grid(True, alpha=0.18)
    ax.figure.colorbar(mesh, ax=ax, label=r"$\bar{\ell}_\rho$")


def subsection_name(text: str) -> str:
    elements = text.split("_")
    return (
        f"{elements[0]} from {elements[1]} to {elements[2]}"
        if len(elements) == 3
        else text.replace("_", " ").title()
    )


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def figure_caption(path: Path) -> str:
    driver_labels = {
        "f10p7-obs-center81": r"\(F_{10.7,81}\)",
        "ap-avg": r"\(A_p\)",
        "kp-sum": r"\(K_p\)",
    }
    heatmap_prefix = "global_mean_density_co2_correlation_by_"
    if path.parent.name == "density_co2_correlation_heatmaps" and path.stem.startswith(
        heatmap_prefix
    ):
        driver_key = path.stem.removeprefix(heatmap_prefix)
        driver_label = driver_labels.get(driver_key, driver_key.replace("-", " "))
        return (
            rf"Density vs \coo correlation by {driver_label} sigma bin. "
            r"Cells show Pearson \(r\), linear-fit slope \(m\) for \(\ell_\rho\) per \coo ppm, "
            r"fitted \coo zero crossing \(x_0\), RMSE err, and sample count \(n\). "
            r"Empty sigma-bin rows are dropped automatically; displayed cells require "
            r"at least 1\% of the figure maximum sample count and \(|r| \ge 0.1\)."
        )
    metric_prefix = "global_mean_density_co2_"
    metric_separator = "_by_altitude_for_"
    if (
        path.parent.name == "density_co2_fit_metric_plots"
        and path.stem.startswith(metric_prefix)
        and metric_separator in path.stem
    ):
        metric_key, driver_key = path.stem.removeprefix(metric_prefix).split(
            metric_separator,
            maxsplit=1,
        )
        driver_label = driver_labels.get(driver_key, driver_key.replace("-", " "))
        metric_labels = {
            "correlation": "Pearson correlation r",
            "slope": r"linear-fit slope m for \(\ell_\rho\) per \coo ppm",
            "zero_crossing": r"fitted \coo zero crossing \(x_0\)",
            "error": "linear-fit RMSE err",
            "sample_count": "sample count n",
        }
        metric_label = metric_labels.get(metric_key, metric_key.replace("_", " "))
        return (
            rf"{metric_label} versus altitude for density and \coo fits grouped by "
            f"{driver_label} sigma bin. Empty sigma-bin rows are dropped automatically."
        )
    return latex_escape(path.stem.replace("_", " ").title())


def write_latex_figure_indexes(root: Path = FIGURE_ROOT, path_root: Path = OUTPUT_ROOT):
    directories = [root, *sorted(path for path in root.rglob("*") if path.is_dir())]
    for directory in sorted(
        directories, key=lambda path: len(path.parts), reverse=True
    ):
        figures = sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in FIGURE_EXTENSIONS
        )
        child_indexes = sorted(
            path / LATEX_FIGURE_INDEX
            for path in directory.iterdir()
            if path.is_dir() and (path / LATEX_FIGURE_INDEX).exists()
        )
        if not figures and not child_indexes:
            continue

        rel_dir = directory.relative_to(root)
        title = (
            "Global Mean Density Figures"
            if rel_dir == Path(".")
            else latex_escape(subsection_name(str(rel_dir)))
        )
        lines = [f"\\subsection*{{{title}}}", ""]
        for figure in figures:
            rel_path = figure.relative_to(path_root).as_posix()
            lines.extend(
                [
                    r"\begin{figure}[H]",
                    r"\centering",
                    rf"\includegraphics[width=\textwidth,height=\textheight,keepaspectratio]{{\detokenize{{{rel_path}}}}}",
                    rf"\caption{{{figure_caption(figure)}}}",
                    r"\end{figure}",
                    "",
                ]
            )
        for child_index in child_indexes:
            rel_path = child_index.relative_to(path_root).as_posix()
            lines.extend([rf"\input{{\detokenize{{{rel_path}}}}}", ""])

        (directory / LATEX_FIGURE_INDEX).write_text("\n".join(lines), encoding="utf-8")


def plot_time_series(
    df_global: pl.DataFrame,
    df_sw: pl.DataFrame,
    df_co2: pl.DataFrame,
    rho_cols: list[str],
    fig_dir: Path,
):
    global_plot = df_global.filter(
        (pl.col("date").dt.year() < END_YEAR) & (pl.col("date").dt.year() > START_YEAR)
    ).select("date", "log10rho_400")
    sw_plot = df_sw.filter(
        (pl.col("date").dt.year() < END_YEAR) & (pl.col("date").dt.year() > START_YEAR)
    ).select("date", "F10.7_OBS", "F10.7_OBS_CENTER81", "AP_AVG")
    co2_plot = df_co2.filter(
        (pl.col("date").dt.year() < END_YEAR) & (pl.col("date").dt.year() > START_YEAR)
    ).select("date", "CO2_ppm")

    fig, (ax_heatmap, ax1, ax2, ax3) = plt.subplots(
        4,
        1,
        figsize=fig_size(1.0, 0.92),
        sharex=True,
        constrained_layout=True,
    )
    minx = max(
        [global_plot["date"].min(), sw_plot["date"].min(), co2_plot["date"].min()]
    )
    maxx = min(
        [global_plot["date"].max(), sw_plot["date"].max(), co2_plot["date"].max()]
    )
    ax1.set_xlim((minx, maxx))
    heatmap_plot = df_global.filter(
        (pl.col("date").dt.year() < END_YEAR) & (pl.col("date").dt.year() > START_YEAR)
    )
    plot_density_time_heatmap(ax_heatmap, heatmap_plot, rho_cols)

    ax1.plot(
        sw_plot["date"],
        sw_plot["F10.7_OBS"],
        linewidth=1,
        label="Daily",
    )
    ax1.plot(
        sw_plot["date"],
        sw_plot["F10.7_OBS_CENTER81"],
        linewidth=1,
        label="81d avg",
    )
    ax1.legend(loc="upper right", fontsize=8, frameon=True)
    ax1.set_ylim(0, 400)
    ax1.set_ylabel("$F_{10.7}$")
    ax1.grid(True)

    ax2_ap = ax2
    ax2_ap.plot(
        sw_plot["date"],
        sw_plot["AP_AVG"],
        linewidth=0.8,
        color="darkviolet",
        label="$A_p$",
    )
    ax2_ap.set_ylabel("$A_p$")
    # ax2.set_yscale("log")
    ax2.grid(True)

    ax3.plot(
        co2_plot["date"],
        co2_plot["CO2_ppm"],
        linewidth=1,
        color="darkgreen",
        label="CO$_2$",
    )
    ax3.set_ylabel("CO$_2$ (ppm)")
    ax3.grid(True)
    ax3.legend(loc="upper right", fontsize=8, frameon=True)

    ax3.xaxis.set_major_locator(mdates.YearLocator(5))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax3.get_xticklabels(), rotation=90)
    save_and_close(fig, fig_dir / "global_mean_timeseries.png")


def plot_fft(
    df_global: pl.DataFrame,
    analysis_df: pl.DataFrame,
    rho_cols: list[str],
    fig_dir: Path,
):
    ANALYSIS_COLS_FFT = ["F10.7_OBS_CENTER81", "AP_AVG", "CO2_ppm"]
    ANALYSIS_LABELS_FFT = ["F$_{10.7,81}$", "$A_p$", "CO$_2$"]
    fig, fft_axes = plt.subplots(
        len(ANALYSIS_COLS_FFT) + 2,
        1,
        figsize=fig_size(1.0, 1.05),
        sharex=True,
        constrained_layout=True,
    )
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(rect=(0.0, 0.0, 1.0, 0.88))
    fft_x_limits = []

    for density_col in rho_cols:
        periods_days, amplitudes = fft_period_power(df_global, density_col)
        periods_years = periods_days / 365.25
        fft_x_limits.extend(periods_years)
        altitude = altitude_from_density_col(density_col)
        label = f"{altitude} km" if altitude is not None else density_col
        fft_axes[0].plot(periods_years, amplitudes, linewidth=1, label=label)
    fft_axes[0].set_ylabel(r"$\bar{\ell}_\rho$")
    handles, labels = fft_axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncols=5,
        fontsize="small",
        borderaxespad=0.2,
    )

    plot_fft_altitude_heatmap(fft_axes[1], df_global, rho_cols)

    for ax, col, label in zip(fft_axes[2:], ANALYSIS_COLS_FFT, ANALYSIS_LABELS_FFT):
        periods_days, amplitudes = fft_period_power(analysis_df, col)
        periods_years = periods_days / 365.25
        fft_x_limits.extend(periods_years)
        ax.plot(periods_years, amplitudes, linewidth=1.5)
        ax.set_ylabel(label)

    fft_x_limits = np.array([*fft_x_limits, *FFT_PERIOD_TICKS_YEARS])
    fft_x_limits = fft_x_limits[np.isfinite(fft_x_limits) & (fft_x_limits > 0)]
    for ax in fft_axes:
        ax.set_xscale("log")
        if ax is not fft_axes[1]:
            ax.set_yscale("log")
        ax.margins(x=0)
        ax.grid(True, which="both", alpha=0.3)
        add_period_xticks(ax, stagger=True)
        ax.tick_params(top=False, labeltop=False, bottom=False, labelbottom=False)
    fft_axes[0].tick_params(top=True, labeltop=True)
    fft_axes[-1].tick_params(bottom=True, labelbottom=True)
    stagger_visible_xticklabels(fft_axes[0], top=True)
    stagger_visible_xticklabels(fft_axes[-1], top=False)
    if len(fft_x_limits) > 0:
        fft_axes[-1].set_xlim(np.min(fft_x_limits), np.max(fft_x_limits))
    fft_axes[-1].set_xlabel("Period")
    save_and_close(fig, fig_dir / "global_mean_fft.png")


def plot_correlation_outputs(
    analysis_df: pl.DataFrame,
    fig_dir: Path,
    heatmap_root: Path,
) -> tuple[list[int], list[np.ndarray]]:
    corr_mats = []
    plotted_altitudes = []

    plot_correlation_heatmap(
        analysis_df.select(ANALYSIS_COLS).drop_nulls().corr().to_numpy(),
        ANALYSIS_LABELS,
        "analysis_variables",
        fig_dir,
    )
    plot_scatter_matrix(
        analysis_df.select(ANALYSIS_COLS).drop_nulls(),
        ANALYSIS_COLS,
        ANALYSIS_LABELS,
        "analysis_variables",
        fig_dir,
    )
    plot_density_co2_correlation_heatmaps(
        analysis_df,
        heatmap_root / "density_co2_correlation_heatmaps_all_altitudes",
        fig_dir / "density_co2_fit_metric_plots",
        ALTITUDES,
    )
    plot_density_co2_correlation_heatmaps(
        analysis_df,
        heatmap_root / "density_co2_correlation_heatmaps",
        fig_dir / "density_co2_fit_metric_plots",
        RESULTS_HEATMAP_ALTITUDES,
        write_metric_plots=False,
    )
    plot_density_scatter_by_altitude(analysis_df, fig_dir)

    for altitude in ALTITUDES:
        density_col = f"log10rho_{altitude}"
        if density_col not in analysis_df.columns:
            continue

        cols = [density_col] + ANALYSIS_COLS
        correlation_df = analysis_df.select(cols).drop_nulls()
        correlation_matrix = correlation_df.corr().to_numpy()
        corr_mats.append(correlation_matrix)
        plotted_altitudes.append(altitude)

    return plotted_altitudes, corr_mats


def plot_density_co2_correlation_heatmaps(
    analysis_df: pl.DataFrame,
    heatmap_dir: Path,
    metric_plot_dir: Path,
    altitudes: list[int] | None = None,
    write_metric_plots: bool = True,
):
    requested_altitudes = ALTITUDES if altitudes is None else altitudes
    available_altitudes = [
        altitude
        for altitude in requested_altitudes
        if f"log10rho_{altitude}" in analysis_df.columns
    ]
    if not available_altitudes:
        return

    heatmap_dir.mkdir(parents=True, exist_ok=True)
    metric_plot_dir.mkdir(parents=True, exist_ok=True)

    for driver_col in SPACE_WEATHER_SIGMA_COLS:
        output_name = driver_output_name(driver_col)
        output_path = (
            heatmap_dir / f"global_mean_density_co2_correlation_by_{output_name}.png"
        )
        driver_values = (
            analysis_df.select(driver_col)
            .drop_nulls()[driver_col]
            .to_numpy()
            .astype(float)
        )
        driver_values = driver_values[np.isfinite(driver_values)]
        if len(driver_values) < MIN_SAMPLES_PER_HEATMAP_CELL:
            if output_path.exists():
                output_path.unlink()
            if write_metric_plots:
                cleanup_metric_plots_for_driver(metric_plot_dir, driver_col)
            continue

        try:
            edges, driver_mean, driver_std, _driver_z = sigma_edges(driver_values)
        except ValueError:
            if output_path.exists():
                output_path.unlink()
                if write_metric_plots:
                    cleanup_metric_plots_for_driver(metric_plot_dir, driver_col)
                continue

        work_df = analysis_df.with_columns(
            ((pl.col(driver_col) - driver_mean) / driver_std).alias("__driver_sigma")
        )
        correlations = np.full((len(edges) - 1, len(available_altitudes)), np.nan)
        corr_los = np.full_like(correlations, np.nan)
        corr_his = np.full_like(correlations, np.nan)
        slopes = np.full_like(correlations, np.nan)
        zero_crossings = np.full_like(correlations, np.nan)
        errors = np.full_like(correlations, np.nan)
        slope_los = np.full_like(correlations, np.nan)
        slope_his = np.full_like(correlations, np.nan)
        counts = np.zeros_like(correlations, dtype=int)
        duration_bins = np.full_like(correlations, np.nan)

        for row in range(len(edges) - 1):
            low = edges[row]
            high = edges[row + 1]
            upper_filter = (
                pl.col("__driver_sigma") <= high
                if row == len(edges) - 2
                else pl.col("__driver_sigma") < high
            )
            bin_df = work_df.filter((pl.col("__driver_sigma") >= low) & upper_filter)

            for col, altitude in enumerate(available_altitudes):
                density_col = f"log10rho_{altitude}"
                co2, density = finite_xy(bin_df, "CO2_ppm", density_col)
                counts[row, col] = len(co2)
                (
                    correlations[row, col],
                    slopes[row, col],
                    zero_crossings[row, col],
                    errors[row, col],
                    slope_los[row, col],
                    slope_his[row, col],
                ) = linear_fit_stats(co2, density)
                r, r_lo, r_hi, _ = pearsonr_ci(co2, density)
                corr_los[row, col] = r_lo
                corr_his[row, col] = r_hi
                valid_dates = (
                    bin_df.filter(
                        pl.col("CO2_ppm").is_not_null()
                        & pl.col(density_col).is_not_null()
                        & pl.col("CO2_ppm").is_finite()
                        & pl.col(density_col).is_finite()
                    )
                    .select("date")
                    .drop_nulls()
                    .sort("date")
                )
                if valid_dates.height > 0:
                    duration_years = (
                        valid_dates["date"].max() - valid_dates["date"].min()
                    ).days / 365.2425
                    duration_bins[row, col] = correlation_duration_bin(duration_years)

        painted_rows = np.any(np.isfinite(correlations), axis=1)
        if not np.any(painted_rows):
            if output_path.exists():
                output_path.unlink()
            if write_metric_plots:
                cleanup_metric_plots_for_driver(metric_plot_dir, driver_col)
            continue

        row_labels = np.array(sigma_bin_labels(edges))[painted_rows]
        correlations = correlations[painted_rows]
        corr_los = corr_los[painted_rows]
        corr_his = corr_his[painted_rows]
        slopes = slopes[painted_rows]
        zero_crossings = zero_crossings[painted_rows]
        errors = errors[painted_rows]
        slope_los = slope_los[painted_rows]
        slope_his = slope_his[painted_rows]
        counts = counts[painted_rows]
        duration_bins = duration_bins[painted_rows]

        if write_metric_plots:
            corr_ci = np.stack([corr_los, corr_his], axis=-1)
            plot_metric_by_altitude_and_sigma(
                metric_plot_dir,
                driver_col,
                available_altitudes,
                row_labels,
                "correlation",
                r"Pearson r($\bar{\ell}_\rho$, CO$_2$)",
                correlations,
                duration_bins,
                ci_values=corr_ci,
            )
            slope_ci = np.stack([slope_los, slope_his], axis=-1)
            plot_metric_by_altitude_and_sigma(
                metric_plot_dir,
                driver_col,
                available_altitudes,
                row_labels,
                "slope",
                r"Linear fit slope m ($\bar{\ell}_\rho$ per CO$_2$ ppm)",
                slopes,
                duration_bins,
                ci_values=slope_ci,
            )
            # plot_metric_by_altitude_and_sigma(
            #     metric_plot_dir,
            #     driver_col,
            #     available_altitudes,
            #     row_labels,
            #     "zero_crossing",
            #     "Fitted CO$_2$ zero crossing x0 (ppm)",
            #     zero_crossings,
            # )
            plot_metric_by_altitude_and_sigma(
                metric_plot_dir,
                driver_col,
                available_altitudes,
                row_labels,
                "error",
                r"Linear fit RMSE err ($\ell_\rho$)",
                errors,
                duration_bins,
            )
            plot_metric_by_altitude_and_sigma(
                metric_plot_dir,
                driver_col,
                available_altitudes,
                row_labels,
                "sample_count",
                "Sample count n",
                counts.astype(float),
            )

        max_count = int(counts.max()) if counts.size else 0
        count_threshold = max(1, int(np.ceil(0.01 * max_count)))
        display_correlations = np.array(correlations, copy=True)
        display_correlations[counts < count_threshold] = np.nan
        display_correlations[np.abs(display_correlations) < 0.1] = np.nan
        painted_rows = np.any(np.isfinite(display_correlations), axis=1)
        if not np.any(painted_rows):
            if output_path.exists():
                output_path.unlink()
            continue

        row_labels = row_labels[painted_rows]
        correlations = correlations[painted_rows]
        display_correlations = display_correlations[painted_rows]
        slopes = slopes[painted_rows]
        zero_crossings = zero_crossings[painted_rows]
        errors = errors[painted_rows]
        slope_los = slope_los[painted_rows]
        slope_his = slope_his[painted_rows]
        counts = counts[painted_rows]

        aspect_ratio = max(
            0.45, min(0.9, (1.2 + 0.55 * len(row_labels)) / TEXTWIDTH_IN)
        )
        fig, ax = plt.subplots(
            figsize=fig_size(1.0, aspect_ratio), constrained_layout=True
        )
        cmap = plt.get_cmap("coolwarm").copy()
        cmap.set_bad(color="lightgray")
        heatmap = ax.imshow(
            np.ma.masked_invalid(display_correlations),
            aspect="auto",
            origin="lower",
            vmin=-1,
            vmax=1,
            cmap=cmap,
            rasterized=True,
        )

        ax.set_xticks(
            np.arange(len(available_altitudes)),
            [f"{altitude}" for altitude in available_altitudes],
            rotation=45,
            ha="right",
        )
        ax.set_yticks(np.arange(len(row_labels)), row_labels)
        ax.set_xlabel("Altitude (km)")
        ax.set_ylabel(
            f"{SPACE_WEATHER_SIGMA_LABELS[driver_col]} bins\n"
            f"mean={driver_mean:.2f}, sigma={driver_std:.2f}"
        )
        ax.set_xticks(np.arange(-0.5, len(available_altitudes), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)

        for row in range(display_correlations.shape[0]):
            for col in range(display_correlations.shape[1]):
                value = display_correlations[row, col]
                if np.isfinite(value):
                    ax.text(
                        col,
                        row,
                        fit_annotation(
                            value,
                            slopes[row, col],
                            zero_crossings[row, col],
                            errors[row, col],
                            counts[row, col],
                            slope_los[row, col],
                            slope_his[row, col],
                        ),
                        ha="center",
                        va="center",
                        fontsize=6,
                        color=text_color_for_correlation(value),
                    )
                elif counts[row, col] > 0:
                    ax.text(
                        col,
                        row,
                        f"n={counts[row, col]}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="dimgray",
                    )

        colorbar_shrink = min(1.0, HEATMAP_COLORBAR_HEIGHT_INCHES / fig.get_figheight())
        fig.colorbar(
            heatmap,
            ax=ax,
            label=r"Pearson r($\bar{\ell}_\rho$, CO$_2$)",
            shrink=colorbar_shrink,
        )
        save_and_close(fig, output_path)


def plot_correlation_heatmap(
    correlation_matrix: np.ndarray, labels: list[str], output_name: str, fig_dir: Path
):
    fig, ax = plt.subplots(figsize=fig_size(0.72, 0.85), constrained_layout=True)
    corr_plot = ax.imshow(
        correlation_matrix, vmin=-1, vmax=1, cmap="coolwarm", rasterized=True
    )
    ax.set_xticks(np.arange(len(labels)), labels, rotation=-45, ha="left", va="top")
    ax.set_yticks(np.arange(len(labels)), labels)
    for label in ax.get_xticklabels():
        label.set_rotation_mode("anchor")
    ax.tick_params(
        top=False,
        labeltop=False,
        bottom=True,
        labelbottom=True,
        right=False,
        labelright=False,
        left=True,
        labelleft=True,
    )
    for row in range(len(labels)):
        for col in range(len(labels)):
            ax.text(
                col,
                row,
                f"{correlation_matrix[row, col]:.2f}",
                ha="center",
                va="center",
                color="white" if abs(correlation_matrix[row, col]) > 0.55 else "black",
            )
    fig.colorbar(corr_plot, ax=ax, label="Pearson r")
    save_and_close(fig, fig_dir / f"global_mean_correlation_{output_name}.png")


def plot_scatter_matrix(
    correlation_df: pl.DataFrame,
    cols: list[str],
    labels: list[str],
    output_name: str,
    fig_dir: Path,
):
    total_points = sum(
        len(finite_xy(correlation_df, x, y)[0]) for y in cols for x in cols
    )
    fig, axes = plt.subplots(
        len(cols),
        len(cols),
        figsize=fig_size(1.0, 1.0),
        constrained_layout=True,
    )

    for row, y_col in enumerate(cols):
        for col, x_col in enumerate(cols):
            ax = axes[row, col]
            x, y = finite_xy(correlation_df, x_col, y_col)
            ax.scatter(
                x,
                y,
                s=8,
                alpha=0.35,
                rasterized=scatter_rasterized(len(x), total_points),
            )
            ax.tick_params(
                top=row == 0,
                labeltop=row == 0,
                bottom=row == len(cols) - 1,
                labelbottom=row == len(cols) - 1,
                right=col == len(cols) - 1,
                labelright=col == len(cols) - 1,
                left=col == 0,
                labelleft=col == 0,
            )
            if row == len(cols) - 1:
                ax.set_xlabel(labels[col])
            elif row != 0:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel(labels[row])
            elif col != len(cols) - 1:
                ax.set_yticklabels([])
            ax.grid(True, alpha=0.2)

    save_and_close(fig, fig_dir / f"global_mean_scatter_{output_name}.png")


def plot_density_scatter_by_altitude(analysis_df: pl.DataFrame, fig_dir: Path):
    available_altitudes = [
        altitude
        for altitude in SELECTED_SCATTER_ALTITUDES
        if f"log10rho_{altitude}" in analysis_df.columns
    ]
    if not available_altitudes:
        return

    split_fig_dir = fig_dir / "density_scatter_by_altitude"
    split_fig_dir.mkdir(parents=True, exist_ok=True)
    for stale in split_fig_dir.glob("global_mean_density_scatter_*km.*"):
        unlink_output(stale)

    total_points = sum(
        len(finite_xy(analysis_df, x_col, f"log10rho_{altitude}")[0])
        for altitude in available_altitudes
        for x_col in ANALYSIS_COLS
    )
    fig, axes = plt.subplots(
        len(available_altitudes),
        len(ANALYSIS_COLS),
        figsize=page_fig_size(1.0, 1.0, max_height_scale=0.95),
        sharex="col",
        sharey="row",
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)

    for row, altitude in enumerate(available_altitudes):
        density_col = f"log10rho_{altitude}"
        for col, (x_col, x_label) in enumerate(zip(ANALYSIS_COLS, ANALYSIS_LABELS)):
            ax = axes[row, col]
            x, y = finite_xy(analysis_df, x_col, density_col)
            ax.scatter(
                x,
                y,
                s=5,
                alpha=0.32,
                rasterized=scatter_rasterized(len(x), total_points),
            )
            if row == len(available_altitudes) - 1:
                ax.set_xlabel(x_label)
            if col == 0:
                ax.set_ylabel(f"{altitude} km")
            ax.grid(True, alpha=0.2)

    fig.supylabel(r"$\bar{\ell}_\rho$")

    save_and_close(
        fig,
        split_fig_dir / "global_mean_density_scatter_selected_altitudes.png",
    )


def plot_correlation_by_altitude(
    analysis_df: pl.DataFrame, altitudes: list[int], fig_dir: Path
):
    if not altitudes:
        return

    fig, ax = plt.subplots(figsize=fig_size(1.0, 0.62), constrained_layout=False)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.97, bottom=0.26)
    add_correlation_effect_size_bands(ax)
    observed_duration_bins: set[int] = set()
    for x_col, x_label in zip(ANALYSIS_COLS, ANALYSIS_LABELS):
        corr = []
        corr_los = []
        corr_his = []
        duration_bins = []
        for altitude in altitudes:
            density_col = f"log10rho_{altitude}"
            point_corr, r_lo, r_hi, duration_bin = correlation_and_duration(
                analysis_df, x_col, density_col
            )
            corr.append(point_corr)
            corr_los.append(r_lo)
            corr_his.append(r_hi)
            duration_bins.append(duration_bin)
            if duration_bin is not None and np.isfinite(point_corr):
                observed_duration_bins.add(duration_bin)
        (line,) = ax.plot(
            altitudes,
            corr,
            linewidth=1.4,
            label=x_label,
            zorder=3,
        )
        corr_arr = np.asarray(corr, dtype=float)
        lo_arr = np.asarray(corr_los, dtype=float)
        hi_arr = np.asarray(corr_his, dtype=float)
        finite_ci = np.isfinite(corr_arr) & np.isfinite(lo_arr) & np.isfinite(hi_arr)
        if np.any(finite_ci):
            ax.fill_between(
                np.asarray(altitudes)[finite_ci],
                lo_arr[finite_ci],
                hi_arr[finite_ci],
                alpha=0.15,
                color=line.get_color(),
                zorder=2,
            )
        for duration_bin in sorted(set(duration_bins) - {None}):
            mask = np.array([value == duration_bin for value in duration_bins])
            finite_corr = np.isfinite(np.asarray(corr, dtype=float))
            mask = mask & finite_corr
            if np.any(mask):
                ax.scatter(
                    np.asarray(altitudes)[mask],
                    np.asarray(corr, dtype=float)[mask],
                    marker=correlation_duration_marker(int(duration_bin)),
                    s=32,
                    color=line.get_color(),
                    edgecolors="black",
                    linewidths=0.45,
                    zorder=4,
                )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Altitude (km)")
    ax.set_ylim((-1, 1))
    ax.set_ylabel("Pearson Correlation")
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower left",
            bbox_to_anchor=(0.12, 0.04),
            ncols=min(4, len(handles)),
            fontsize=8,
            borderaxespad=0,
            handlelength=1.8,
            columnspacing=1.1,
        )
    duration_handles = [
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
        for duration_bin in sorted(observed_duration_bins)
    ]
    if duration_handles:
        fig.legend(
            duration_handles,
            [handle.get_label() for handle in duration_handles],
            title="Record length",
            loc="lower right",
            bbox_to_anchor=(0.98, 0.04),
            ncols=min(2, len(duration_handles)),
            fontsize=7,
            title_fontsize=8,
            borderaxespad=0,
        )
    save_and_close(fig, fig_dir / "global_mean_correlation_by_altitude.png")


def plot_wavelet(analysis_df: pl.DataFrame, fig_dir: Path):
    fig, axes = plt.subplots(
        len(ANALYSIS_COLS),
        1,
        figsize=page_fig_size(1.0, 0.35 * len(ANALYSIS_COLS), max_height_scale=0.95),
        sharex=True,
        constrained_layout=True,
    )

    for ax, col, label in zip(axes, ANALYSIS_COLS, ANALYSIS_LABELS):
        days, periods_years, power, cone_of_influence_years = morlet_wavelet_power(
            analysis_df, col
        )
        if len(days) == 0:
            ax.set_ylabel(label)
            continue

        date_numbers = mdates.date2num(np.datetime64("1970-01-01")) + days
        log_power = np.log10(power + np.finfo(float).eps)
        mesh = ax.pcolormesh(
            date_numbers,
            periods_years,
            log_power,
            shading="auto",
            cmap="magma",
            rasterized=True,
        )
        ax.plot(
            date_numbers, cone_of_influence_years, color="white", linewidth=1, alpha=0.8
        )
        ax.fill_between(
            date_numbers,
            cone_of_influence_years,
            np.max(periods_years),
            color="white",
            alpha=0.12,
        )
        ax.set_yscale("log")
        add_period_yticks(ax)
        ax.set_ylabel(label)
        ax.grid(True, which="both", alpha=0.15)
        fig.colorbar(mesh, ax=ax, label="log10 power")

    axes[-1].xaxis.set_major_locator(mdates.YearLocator(5))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(axes[-1].get_xticklabels(), rotation=45, ha="right")
    axes[-1].set_xlabel("Date")
    save_and_close(fig, fig_dir / "global_mean_wavelet.pgf")


def generate_figures_for_range(f10_range: tuple[int, int]) -> str:
    f10_min, f10_max = f10_range
    range_name = f"f107_{f10_min}_{f10_max}"
    time_frequency_dir = FIGURE_ROOT / "fft_timeseries" / range_name
    correlation_dir = FIGURE_ROOT / "correlation" / range_name
    heatmap_dir = FIGURE_ROOT / "heatmaps" / range_name
    for directory in [time_frequency_dir, correlation_dir, heatmap_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    df_global, rho_cols = load_global_mean()
    df_co2 = load_co2()
    df_sw = load_space_weather(f10_min, f10_max)
    df_mgii = load_mgii()
    df_combined = combined_dataset(df_global, df_co2, df_sw, df_mgii)
    analysis_df = df_combined.with_columns(
        pl.col("log10rho_400").alias("log10rho_mean")
    )

    plot_time_series(df_global, df_sw, df_co2, rho_cols, time_frequency_dir)
    plot_fft(df_global, analysis_df, rho_cols, time_frequency_dir)
    altitudes, _ = plot_correlation_outputs(analysis_df, correlation_dir, heatmap_dir)
    plot_correlation_by_altitude(analysis_df, altitudes, correlation_dir)
    # plot_wavelet(analysis_df, fig_dir)

    return f"Generated {FIGURE_ROOT / range_name}"


def main():
    max_workers = min(len(F10_7_RANGES), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(generate_figures_for_range, f10_range)
            for f10_range in F10_7_RANGES
        ]
        for future in as_completed(futures):
            print(future.result())
    write_latex_figure_indexes()


if __name__ == "__main__":
    main()
