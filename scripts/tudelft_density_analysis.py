from __future__ import annotations
# Ruff: configure_pgf() must run before pyplot imports; suppress intentional E402.
# ruff: noqa: E402

from dataclasses import dataclass
from pathlib import Path

from scripts.pgf_config import configure_pgf, fig_size, page_fig_size

configure_pgf()

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import polars as pl
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from scripts.stats_utils import ols_slope_ci, pearsonr_ci  # noqa: E402
from scripts.tudelft_model_error_analysis import (  # noqa: E402
    ACTIVITY_DRIVERS,
    CORRELATION_LABELS,
    DRIVER_COLS,
    FREQUENCY_HEATMAP_ALTITUDE_BIN_KM,
    MIN_SAMPLES_PER_HEATMAP_CELL,
    MISSION_ORDER,
    MISSIONS,
    MissionConfig,
    add_period_reference_lines,
    date_edges_for_heatmap,
    load_driver_data,
    lomb_scargle_spectrum,
    pearsonr,
    positive_power,
    sigma_edges,
)

OUTPUT_ROOT = Path("outputs/figures/results/tudelft_density")
CAUSAL_WORKFLOW_DIR = ("model_validations", "causal_tudelft_density")
DENSITY_COL = "log10_density"
CORRELATION_DRIVERS = ["f107_81d", "ap", "kp", "co2", "altitude_km"]
TIMESERIES_DRIVER_PANELS = [
    ("f107_81d", "$F_{10.7,81}$", "$F_{10.7,81}$", "darkred"),
    ("ap", "$A_p$", "$A_p$", "purple"),
    ("co2", "CO$_2$ (ppm)", "Mauna Loa CO$_2$", "darkgreen"),
]
CAUSAL_DRIVER_RENAMES = {
    "f107_81d": "F10.7_OBS_CENTER81",
    "ap": "AP_AVG",
    "kp": "KP_SUM",
    "co2": "CO2_ppm",
}
CAUSAL_Y_COLS = ["F10.7_OBS_CENTER81", "AP_AVG", "KP_SUM"]
CAUSAL_X_COL = "CO2_ppm"
CAUSAL_DRIVER_COLS = [*CAUSAL_Y_COLS, CAUSAL_X_COL]
CAUSAL_ALTITUDE_BIN_KM = 25
CAUSAL_MIN_SAMPLES_PER_CELL = 5
CAUSAL_MAX_LAG_DAYS = 180
CAUSAL_BASE_LAGS = list(range(14)) + [14, 27, 54, 81, 120, 180]
GLOBAL_STYLE_DRIVER_COLS = ["F10.7_OBS_CENTER81", "AP_AVG", "KP_SUM", "CO2_ppm"]
GLOBAL_STYLE_SPACE_WEATHER_COLS = ["F10.7_OBS_CENTER81", "AP_AVG", "KP_SUM"]
GLOBAL_STYLE_SPACE_WEATHER_LABELS = {
    "F10.7_OBS_CENTER81": r"$F_{10.7,81}$",
    "AP_AVG": "$A_p$",
    "KP_SUM": "$K_p$",
}
CORRELATION_DURATION_STEP_YEARS = 11
CORRELATION_DURATION_MARKERS = {
    0: "o",
    11: "s",
    22: "D",
    33: "^",
    44: "P",
    55: "X",
}
SCATTER_RASTERIZE_PANEL_POINTS = 2_000


def format_altitude_axis(ax: plt.Axes, axis: str = "y") -> None:
    if axis == "y":
        ax.set_yscale("linear")
        target = ax.yaxis
    else:
        ax.set_xscale("linear")
        target = ax.xaxis
    target.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
    target.get_offset_text().set_visible(False)


def scatter_rasterized(panel_points: int) -> bool:
    return panel_points > SCATTER_RASTERIZE_PANEL_POINTS


@dataclass
class Variant:
    name: str
    dates: np.ndarray
    data: dict[str, np.ndarray]
    description: str


def output_path(*parts: str) -> Path:
    path = OUTPUT_ROOT.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig: plt.Figure, *parts: str) -> None:
    out = output_path(*parts)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def save_result_figure(fig: plt.Figure, *parts: str) -> None:
    out = output_path(*parts)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def load_density_mission(
    config: MissionConfig, *, downsampled: bool = True
) -> pl.DataFrame:
    min_alt, max_alt = config.altitude_range_km
    source_path = config.downsampled_path if downsampled else config.path
    df = pl.read_parquet(source_path, low_memory=True).sort("timestamp")
    quality_filter = (pl.col("Altitude (m)") / 1000.0).is_between(min_alt, max_alt)

    if config.name not in {"Swarm-A", "Swarm-B", "Swarm-C"}:
        if "Anomalus Density (kg/m^3)" in df.columns:
            quality_filter = quality_filter & (pl.col("Anomalus Density (kg/m^3)") == 0)
        if "Anomalus Density Mean (kg/m^3)" in df.columns:
            quality_filter = quality_filter & (
                pl.col("Anomalus Density Mean (kg/m^3)") == 0
            )

    if config.name == "GOCE" and "Degraded Flag Thrusters" in df.columns:
        quality_filter = quality_filter & (pl.col("Degraded Flag Thrusters") == 0)

    joined_driver_cols = [col for col in DRIVER_COLS if col in df.columns]
    if joined_driver_cols:
        df = df.drop(joined_driver_cols)

    return (
        df.filter(quality_filter)
        .with_columns(pl.col("timestamp").dt.date().alias("date"))
        .join(load_driver_data(), on="date", how="left")
        .with_columns(
            (pl.col("Altitude (m)") / 1000.0).alias("altitude_km"),
            pl.col("Density (kg/m^3)").log10().alias(DENSITY_COL),
        )
        .select(
            "timestamp",
            "Density (kg/m^3)",
            DENSITY_COL,
            "altitude_km",
            "Latitude (deg)",
            "Longitude (deg)",
            "f107a",
            "f107_81d",
            "ap",
            "kp",
            "co2",
        )
        .drop_nulls([DENSITY_COL, "altitude_km"])
    )


def load_all_density() -> list[tuple[MissionConfig, pl.DataFrame]]:
    return [
        (MISSIONS[code], load_density_mission(MISSIONS[code])) for code in MISSION_ORDER
    ]


def daily_summary(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.group_by_dynamic(
            "timestamp", every="1d", period="1d", closed="left", label="left"
        )
        .agg(
            pl.col("Density (kg/m^3)").median().alias("density"),
            pl.col(DENSITY_COL).median().alias(DENSITY_COL),
            pl.col("altitude_km").mean().alias("altitude_km"),
            pl.col("f107a").mean().alias("f107a"),
            pl.col("f107_81d").mean().alias("f107_81d"),
            pl.col("ap").mean().alias("ap"),
            pl.col("kp").mean().alias("kp"),
            pl.col("co2").mean().alias("co2"),
        )
        .drop_nulls([DENSITY_COL, "altitude_km"])
        .sort("timestamp")
    )


def plot_altitude_time_heatmap(
    ax: plt.Axes,
    summaries: list[tuple[MissionConfig, pl.DataFrame]],
    value_col: str,
    title: str,
) -> None:
    frames = [
        daily.select("timestamp", "altitude_km", value_col)
        for _, daily in summaries
        if value_col in daily.columns and daily.height
    ]
    combined = pl.concat(frames).drop_nulls(["timestamp", "altitude_km", value_col])
    dates = combined["timestamp"].unique().sort().to_numpy()
    altitude_values = combined["altitude_km"].to_numpy().astype(float)
    lower = (
        np.floor(np.nanmin(altitude_values) / FREQUENCY_HEATMAP_ALTITUDE_BIN_KM)
        * FREQUENCY_HEATMAP_ALTITUDE_BIN_KM
    )
    upper = (
        np.ceil(np.nanmax(altitude_values) / FREQUENCY_HEATMAP_ALTITUDE_BIN_KM)
        * FREQUENCY_HEATMAP_ALTITUDE_BIN_KM
    )
    altitude_edges = np.arange(
        lower,
        upper + FREQUENCY_HEATMAP_ALTITUDE_BIN_KM,
        FREQUENCY_HEATMAP_ALTITUDE_BIN_KM,
    )
    date_index = {value: idx for idx, value in enumerate(dates)}
    matrix_sum = np.zeros((len(altitude_edges) - 1, len(dates)), dtype=float)
    matrix_count = np.zeros_like(matrix_sum)
    for timestamp, altitude, value in zip(
        combined["timestamp"], combined["altitude_km"], combined[value_col]
    ):
        date_idx = date_index[timestamp]
        altitude_idx = (
            np.searchsorted(altitude_edges, float(altitude), side="right") - 1
        )
        if 0 <= altitude_idx < matrix_sum.shape[0] and np.isfinite(value):
            matrix_sum[altitude_idx, date_idx] += float(value)
            matrix_count[altitude_idx, date_idx] += 1
    matrix = np.divide(
        matrix_sum,
        matrix_count,
        out=np.full_like(matrix_sum, np.nan),
        where=matrix_count > 0,
    )
    mesh = ax.pcolormesh(
        date_edges_for_heatmap(dates),
        altitude_edges,
        matrix,
        shading="auto",
        cmap="viridis",
        rasterized=True,
    )
    ax.xaxis_date()
    ax.set_ylabel("Altitude (km)")
    format_altitude_axis(ax)
    ax.grid(True, alpha=0.18)
    cax = inset_axes(
        ax,
        width="1.8%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(1.01, 0.0, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    colorbar = ax.figure.colorbar(
        mesh,
        cax=cax,
        label=r"$\bar{\ell}_\rho$",
    )
    colorbar.ax.tick_params(labelsize=8)
    colorbar.ax.yaxis.label.set_size(8)


def plot_timeseries(missions: list[tuple[MissionConfig, pl.DataFrame]]) -> None:
    summaries = [(config, daily_summary(df)) for config, df in missions]
    colors = plt.get_cmap("tab10", len(summaries))
    fig, axes = plt.subplots(
        6,
        1,
        figsize=page_fig_size(1.0, 1.45, 0.98),
        sharex=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.12, right=0.96, top=0.91, bottom=0.10, hspace=0.20)
    for idx, (config, daily) in enumerate(summaries):
        axes[0].plot(
            daily["timestamp"],
            daily[DENSITY_COL],
            linewidth=0.8,
            alpha=0.85,
            color=colors(idx),
            label=config.name,
        )
        axes[2].plot(
            daily["timestamp"],
            daily["altitude_km"],
            linewidth=0.8,
            alpha=0.85,
            color=colors(idx),
        )
    axes[0].set_ylabel(r"$\bar{\ell}_\rho$")
    axes[0].grid(True, alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncols=4,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.965),
        frameon=True,
    )
    plot_altitude_time_heatmap(
        axes[1], summaries, DENSITY_COL, "Observed density by altitude and time"
    )
    axes[2].set_ylabel("Altitude (km)")
    format_altitude_axis(axes[2])
    axes[2].grid(True, alpha=0.25)

    start = min(daily["timestamp"].min().date() for _, daily in summaries)
    end = max(daily["timestamp"].max().date() for _, daily in summaries)
    drivers = load_driver_data().filter(
        (pl.col("date") >= start) & (pl.col("date") <= end)
    )
    for ax, (driver_col, ylabel, label, color) in zip(
        axes[3:], TIMESERIES_DRIVER_PANELS
    ):
        ax.plot(
            drivers["date"],
            drivers[driver_col],
            linewidth=0.9,
            color=color,
            label=label,
        )
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="y", colors=color)
        ax.yaxis.label.set_color(color)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Date")
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    save_figure(fig, "fft_timeseries", "tudelft_density_timeseries_combined.pgf")


def binned_lomb_scargle_heatmap(
    summaries: list[tuple[MissionConfig, pl.DataFrame]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spectra = []
    all_periods = []
    for _, daily in summaries:
        if daily.height < 20:
            continue
        altitude = float(np.nanmedian(daily["altitude_km"].to_numpy().astype(float)))
        periods, power = lomb_scargle_spectrum(daily, DENSITY_COL)
        if len(periods) == 0:
            continue
        periods_years = periods / 365.25
        spectra.append((altitude, periods_years, positive_power(power)))
        all_periods.extend(periods_years)
    if not spectra:
        return np.array([]), np.array([]), np.array([[]])

    altitudes = np.array([row[0] for row in spectra], dtype=float)
    altitude_edges = np.arange(
        np.floor(np.nanmin(altitudes) / 25) * 25,
        np.ceil(np.nanmax(altitudes) / 25) * 25 + 25,
        25,
    )
    period_edges = np.logspace(
        np.log10(np.nanmin(all_periods)), np.log10(np.nanmax(all_periods)), 80
    )
    period_centers = np.sqrt(period_edges[:-1] * period_edges[1:])
    matrix = np.full((len(altitude_edges) - 1, len(period_centers)), np.nan)
    target_log_periods = np.log10(period_centers)
    for altitude_idx in range(len(altitude_edges) - 1):
        rows = []
        for altitude, periods_years, power in spectra:
            if not (
                altitude_edges[altitude_idx]
                <= altitude
                < altitude_edges[altitude_idx + 1]
            ):
                continue
            mask = np.isfinite(periods_years) & np.isfinite(power) & (power > 0)
            if np.sum(mask) < 2:
                continue
            order = np.argsort(periods_years[mask])
            rows.append(
                np.interp(
                    target_log_periods,
                    np.log10(periods_years[mask][order]),
                    np.log10(power[mask][order]),
                    left=np.nan,
                    right=np.nan,
                )
            )
        if rows:
            subset = np.vstack(rows)
            valid_cols = np.any(np.isfinite(subset), axis=0)
            matrix[altitude_idx, valid_cols] = np.nanmean(subset[:, valid_cols], axis=0)
    return period_edges, altitude_edges, matrix


def plot_frequency(missions: list[tuple[MissionConfig, pl.DataFrame]]) -> None:
    summaries = [(config, daily_summary(df)) for config, df in missions]
    fig, axes = plt.subplots(
        5,
        1,
        figsize=page_fig_size(1.0, 1.28, 0.95),
        constrained_layout=False,
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.15, 0.78, 0.78, 0.78]},
    )
    fig.subplots_adjust(left=0.12, right=0.96, top=0.91, bottom=0.12, hspace=0.18)
    colors = plt.get_cmap("tab10", len(summaries))
    for idx, (config, daily) in enumerate(summaries):
        periods, power = lomb_scargle_spectrum(daily, DENSITY_COL)
        if len(periods):
            axes[0].plot(
                periods / 365.25,
                positive_power(power),
                linewidth=0.8,
                alpha=0.85,
                color=colors(idx),
                label=config.name,
            )
    add_period_reference_lines(axes[0])
    axes[0].set_yscale("log")
    axes[0].set_ylim(1e-4, 1.05)
    axes[0].set_ylabel("Norm. power")
    axes[0].grid(True, which="both", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncols=4,
        fontsize=7,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.965),
        frameon=True,
    )

    period_edges, altitude_edges, log_power = binned_lomb_scargle_heatmap(summaries)
    if len(period_edges) and len(altitude_edges):
        mesh = axes[1].pcolormesh(
            period_edges,
            altitude_edges,
            log_power,
            shading="auto",
            cmap="magma",
            rasterized=True,
        )
        cax = inset_axes(
            axes[1],
            width="1.8%",
            height="100%",
            loc="lower left",
            bbox_to_anchor=(1.01, 0.0, 1.0, 1.0),
            bbox_transform=axes[1].transAxes,
            borderpad=0,
        )
        colorbar = fig.colorbar(
            mesh,
            cax=cax,
            label="log$_{10}$ normalized power",
        )
        colorbar.ax.tick_params(labelsize=8)
        colorbar.ax.yaxis.label.set_size(8)
    add_period_reference_lines(axes[1])
    axes[1].set_ylabel("Altitude (km)")
    format_altitude_axis(axes[1])
    axes[1].grid(True, which="both", alpha=0.18)

    start = min(daily["timestamp"].min().date() for _, daily in summaries)
    end = max(daily["timestamp"].max().date() for _, daily in summaries)
    drivers = load_driver_data().filter(
        (pl.col("date") >= start) & (pl.col("date") <= end)
    )
    for ax, (driver_col, label, _, _) in zip(axes[2:], TIMESERIES_DRIVER_PANELS):
        periods, power = lomb_scargle_spectrum(drivers, driver_col, time_col="date")
        if len(periods):
            ax.plot(
                periods / 365.25, positive_power(power), linewidth=1.2, color="black"
            )
        add_period_reference_lines(ax)
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1.05)
        ax.set_ylabel(f"Norm. {label}")
        ax.grid(True, which="both", alpha=0.25)
    axes[-1].set_xlabel("Period")
    save_figure(fig, "fft_timeseries", "tudelft_density_driver_spectra.pgf")


def correlation_table(
    missions: list[tuple[MissionConfig, pl.DataFrame]],
) -> pl.DataFrame:
    rows = []
    for config, df in missions:
        daily = daily_summary(df)
        for driver in CORRELATION_DRIVERS:
            rows.append(
                {
                    "mission": config.name,
                    "driver": driver,
                    "pearson_r": pearsonr(
                        daily[DENSITY_COL].to_numpy(), daily[driver].to_numpy()
                    ),
                    "n_days": daily.height,
                }
            )
    table = pl.DataFrame(rows)
    table.write_csv(output_path("correlation", "tudelft_density_correlations.csv"))
    return table


def plot_correlation_summary(table: pl.DataFrame) -> None:
    missions = [MISSIONS[code].name for code in MISSION_ORDER]
    matrix = np.full((len(CORRELATION_DRIVERS), len(missions)), np.nan)
    for row, driver in enumerate(CORRELATION_DRIVERS):
        for col, mission in enumerate(missions):
            value = table.filter(
                (pl.col("driver") == driver) & (pl.col("mission") == mission)
            )["pearson_r"]
            matrix[row, col] = value[0] if len(value) else np.nan

    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    image = ax.imshow(
        matrix, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto", rasterized=True
    )
    ax.set_xticks(np.arange(len(missions)), missions, rotation=45, ha="right")
    ax.set_yticks(
        np.arange(len(CORRELATION_DRIVERS)),
        [CORRELATION_LABELS[driver] for driver in CORRELATION_DRIVERS],
    )
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if np.isfinite(matrix[row, col]):
                ax.text(
                    col,
                    row,
                    f"{matrix[row, col]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if abs(matrix[row, col]) > 0.55 else "black",
                )
    fig.colorbar(image, ax=ax, label="Pearson r")
    ax.set_title(r"Mission-wise correlations with TU Delft $\ell_\rho$")
    save_figure(fig, "correlation", "tudelft_density_correlation_summary.pgf")


def binned_density_table(
    missions: list[tuple[MissionConfig, pl.DataFrame]],
) -> pl.DataFrame:
    rows = []
    for config, df in missions:
        altitude = df["altitude_km"].to_numpy().astype(float)
        alt_edges = np.arange(
            np.floor(np.nanmin(altitude) / 25) * 25,
            np.ceil(np.nanmax(altitude) / 25) * 25 + 25,
            25,
        )
        driver_edges = {}
        sigma_exprs = []
        for driver in ACTIVITY_DRIVERS:
            edges, mean, std = sigma_edges(df[driver].to_numpy().astype(float))
            driver_edges[driver] = edges
            sigma_exprs.append(
                ((pl.col(driver) - mean) / std).alias(f"{driver}_sigma")
                if std
                else pl.lit(0.0).alias(f"{driver}_sigma")
            )
        work = df.with_columns(*sigma_exprs)
        for driver, edges in driver_edges.items():
            sigma_col = f"{driver}_sigma"
            for sigma_idx in range(len(edges) - 1):
                sigma_filter = (pl.col(sigma_col) >= edges[sigma_idx]) & (
                    pl.col(sigma_col) < edges[sigma_idx + 1]
                )
                for alt_idx in range(len(alt_edges) - 1):
                    cell = work.filter(
                        sigma_filter
                        & (pl.col("altitude_km") >= alt_edges[alt_idx])
                        & (pl.col("altitude_km") < alt_edges[alt_idx + 1])
                    )
                    if cell.height < MIN_SAMPLES_PER_HEATMAP_CELL:
                        continue
                    x = cell["co2"].to_numpy().astype(float)
                    y = cell[DENSITY_COL].to_numpy().astype(float)
                    mask = np.isfinite(x) & np.isfinite(y)
                    x = x[mask]
                    y = y[mask]
                    if (
                        len(x) >= MIN_SAMPLES_PER_HEATMAP_CELL
                        and np.std(x) > 0
                        and np.std(y) > 0
                    ):
                        (
                            _slope,
                            slope_lo,
                            slope_hi,
                            _se,
                            _int,
                            co2_correlation,
                            _rmse,
                            _n,
                        ) = ols_slope_ci(x, y)
                        co2_slope = float(_slope)
                    else:
                        co2_slope = np.nan
                        co2_correlation = np.nan
                        slope_lo = np.nan
                        slope_hi = np.nan
                    rows.append(
                        {
                            "mission": config.name,
                            "driver": driver,
                            "driver_sigma_min": float(edges[sigma_idx]),
                            "driver_sigma_max": float(edges[sigma_idx + 1]),
                            "altitude_min_km": float(alt_edges[alt_idx]),
                            "altitude_max_km": float(alt_edges[alt_idx + 1]),
                            "mean_log10_density": float(cell[DENSITY_COL].mean()),
                            "std_log10_density": float(cell[DENSITY_COL].std()),
                            "co2_slope": co2_slope,
                            "co2_correlation": co2_correlation,
                            "co2_slope_lo": slope_lo,
                            "co2_slope_hi": slope_hi,
                            "n": cell.height,
                        }
                    )
    table = pl.DataFrame(rows)
    table.write_csv(output_path("binning", "tudelft_density_binned_stats.csv"))
    return table


def plot_binned_summary(table: pl.DataFrame) -> None:
    fig, axes = plt.subplots(
        1,
        len(ACTIVITY_DRIVERS),
        figsize=(16, 4.8),
        constrained_layout=True,
        sharey=True,
    )
    for col_idx, driver in enumerate(ACTIVITY_DRIVERS):
        ax = axes[col_idx]
        driver_df = (
            table.filter(pl.col("driver") == driver)
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
            .group_by("driver_sigma_min", "driver_sigma_max", "altitude_min_km")
            .agg(
                pl.col("weighted_slope").sum().alias("weighted_slope"),
                pl.col("weighted_slope_lo").sum().alias("weighted_slope_lo"),
                pl.col("weighted_slope_hi").sum().alias("weighted_slope_hi"),
                pl.col("n").sum().alias("n"),
            )
            .with_columns(
                (pl.col("weighted_slope") / pl.col("n")).alias("co2_slope"),
                (pl.col("weighted_slope_lo") / pl.col("n")).alias("co2_slope_lo_agg"),
                (pl.col("weighted_slope_hi") / pl.col("n")).alias("co2_slope_hi_agg"),
            )
            .sort("driver_sigma_min", "altitude_min_km")
        )
        sigma_bins = (
            driver_df.select("driver_sigma_min", "driver_sigma_max")
            .unique()
            .sort("driver_sigma_min")
        )
        for sigma_row in sigma_bins.iter_rows(named=True):
            series = driver_df.filter(
                (pl.col("driver_sigma_min") == sigma_row["driver_sigma_min"])
                & (pl.col("driver_sigma_max") == sigma_row["driver_sigma_max"])
            ).sort("altitude_min_km")
            label = f"{sigma_row['driver_sigma_min']:g} to {sigma_row['driver_sigma_max']:g} sigma"
            (line,) = ax.plot(
                series["altitude_min_km"],
                series["co2_slope"],
                marker="o",
                linewidth=1.0,
                label=label,
            )
            lo_vals = series["co2_slope_lo_agg"].to_numpy().astype(float)
            hi_vals = series["co2_slope_hi_agg"].to_numpy().astype(float)
            slope_vals = series["co2_slope"].to_numpy().astype(float)
            alt_vals = series["altitude_min_km"].to_numpy().astype(float)
            finite_ci = (
                np.isfinite(slope_vals) & np.isfinite(lo_vals) & np.isfinite(hi_vals)
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
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Altitude bin lower edge (km)")
        ax.set_title(f"Binned by {DRIVER_COLS[driver]}")
        ax.grid(True, alpha=0.25)
        if col_idx == 0:
            ax.set_ylabel(r"CO$_2$ fitted slope in $\ell_\rho$ per ppm")
    axes[0].legend(
        title="Activity bin",
        fontsize=7,
        title_fontsize=8,
        ncols=2,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
    )
    fig.suptitle("TuDelft observed density CO$_2$ slope by altitude and activity bin")
    save_figure(fig, "binning", "tudelft_density_altitude_activity_binned_summary.pgf")


def global_style_label(col: str) -> str:
    if col in GLOBAL_STYLE_SPACE_WEATHER_LABELS:
        return GLOBAL_STYLE_SPACE_WEATHER_LABELS[col]
    if col == "CO2_ppm":
        return "CO$_2$"
    if col.startswith("log10rho_") and col.endswith("_daily_range"):
        return col.replace("log10rho_", "$\\Delta\\ell_\\rho$ ").replace(
            "_daily_range", " km"
        )
    if col.startswith("log10rho_"):
        return col.replace("log10rho_", "$\\ell_\\rho$ ").replace("_daily_mean", " km")
    return col


def global_style_finite_xy(
    df: pl.DataFrame, x_col: str, y_col: str
) -> tuple[np.ndarray, np.ndarray]:
    if x_col == y_col:
        values = df.select(x_col).drop_nulls()[x_col].to_numpy().astype(float)
        mask = np.isfinite(values)
        return values[mask], values[mask]
    pair = df.select(x_col, y_col).drop_nulls()
    x = pair[x_col].to_numpy().astype(float)
    y = pair[y_col].to_numpy().astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def global_style_sigma_bin_labels(edges: np.ndarray) -> list[str]:
    return [
        f"{int(edges[idx])} to {int(edges[idx + 1])} sigma"
        for idx in range(len(edges) - 1)
    ]


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
            markeredgewidth=0.65,
            markersize=8,
            label=correlation_duration_label(duration_bin),
        )
        for duration_bin in sorted(duration_bins)
    ]


def global_style_linear_fit_stats(
    x: np.ndarray, y: np.ndarray
) -> tuple[float, float, float, float, float, float]:
    if len(x) < MIN_SAMPLES_PER_HEATMAP_CELL or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    slope, slope_lo, slope_hi, _se, intercept, correlation, error, _ = ols_slope_ci(
        x, y
    )
    zero_crossing = float(-intercept / slope) if slope != 0 else np.nan
    return correlation, float(slope), zero_crossing, error, slope_lo, slope_hi


def global_style_fit_annotation(
    correlation: float,
    slope: float,
    zero_crossing: float,
    error: float,
    count: int,
    year_range: str = "",
    slope_lo: float = np.nan,
    slope_hi: float = np.nan,
) -> str:
    ci_str = ""
    if np.isfinite(slope_lo) and np.isfinite(slope_hi):
        ci_str = f"\nCI=[{slope_lo:.2e}, {slope_hi:.2e}]"
    annotation = (
        f"r={correlation:.2f}\nm={slope:.2e}{ci_str}\nx0={zero_crossing:.1f}"
        f"\nerr={error:.3f}\nn={count}"
    )
    return f"{annotation}\n{year_range}" if year_range else annotation


def tudelft_density_mean_col(altitude: int) -> str:
    return f"log10rho_{altitude}_daily_mean"


def tudelft_density_range_col(altitude: int) -> str:
    return f"log10rho_{altitude}_daily_range"


def build_global_style_tables(
    missions: list[tuple[MissionConfig, pl.DataFrame]],
) -> tuple[pl.DataFrame, list[int], pl.DataFrame, list[str]]:
    altitude_frames = []
    mission_frames = []
    for config, df in missions:
        work = df.with_columns(pl.col("timestamp").dt.date().alias("date")).select(
            "date", "altitude_km", DENSITY_COL
        )
        altitude_frames.append(work)
        mission_frames.append(
            work.with_columns(pl.lit(config.name).alias("mission")).select(
                "date", "mission", DENSITY_COL
            )
        )
    combined = pl.concat(altitude_frames).drop_nulls(
        ["date", "altitude_km", DENSITY_COL]
    )
    long_altitude = (
        combined.with_columns(
            (
                (pl.col("altitude_km") / CAUSAL_ALTITUDE_BIN_KM).floor()
                * CAUSAL_ALTITUDE_BIN_KM
            )
            .cast(pl.Int32)
            .alias("altitude_bin_km")
        )
        .group_by("date", "altitude_bin_km")
        .agg(
            pl.col(DENSITY_COL).mean().alias("log10rho_daily_mean"),
            (pl.col(DENSITY_COL).max() - pl.col(DENSITY_COL).min()).alias(
                "log10rho_daily_range"
            ),
            pl.len().alias("samples"),
        )
        .filter(pl.col("samples") >= MIN_SAMPLES_PER_HEATMAP_CELL)
        .sort("date", "altitude_bin_km")
    )
    altitudes = [
        int(alt) for alt in long_altitude["altitude_bin_km"].unique().sort().to_list()
    ]
    wide_altitude: pl.DataFrame | None = None
    for value_col, template in [
        ("log10rho_daily_mean", "log10rho_{altitude}_daily_mean"),
        ("log10rho_daily_range", "log10rho_{altitude}_daily_range"),
    ]:
        pivot = (
            long_altitude.with_columns(
                pl.col("altitude_bin_km").cast(pl.Utf8).alias("altitude_label")
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
            str(altitude): template.format(altitude=altitude)
            for altitude in altitudes
            if str(altitude) in pivot.columns
        }
        pivot = pivot.rename(rename_map)
        wide_altitude = (
            pivot
            if wide_altitude is None
            else wide_altitude.join(pivot, on="date", how="full", coalesce=True)
        )
    if wide_altitude is None:
        raise RuntimeError("No TuDelft altitude table created.")

    mission_names = [MISSIONS[code].name for code in MISSION_ORDER]
    long_mission = (
        pl.concat(mission_frames)
        .drop_nulls(["date", "mission", DENSITY_COL])
        .group_by("date", "mission")
        .agg(
            pl.col(DENSITY_COL).mean().alias("density_mean"),
            (pl.col(DENSITY_COL).max() - pl.col(DENSITY_COL).min()).alias(
                "density_range"
            ),
            pl.len().alias("samples"),
        )
        .filter(pl.col("samples") >= MIN_SAMPLES_PER_HEATMAP_CELL)
        .sort("date", "mission")
    )
    drivers = (
        load_driver_data()
        .rename(CAUSAL_DRIVER_RENAMES)
        .select("date", *GLOBAL_STYLE_DRIVER_COLS)
        .unique(subset="date")
        .sort("date")
    )
    return (
        drivers.join(wide_altitude.sort("date"), on="date", how="inner"),
        altitudes,
        drivers.join(long_mission, on="date", how="inner"),
        mission_names,
    )


def plot_global_style_correlation_heatmap(
    df: pl.DataFrame, cols: list[str], filename: str, title: str
) -> None:
    corr_df = df.select(cols).drop_nulls()
    matrix = corr_df.corr().to_numpy()
    labels = [global_style_label(col) for col in cols]
    fig, ax = plt.subplots(
        figsize=(max(8, 0.55 * len(cols)), max(7, 0.55 * len(cols))),
        constrained_layout=True,
    )
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm", rasterized=True)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=7)
    for row in range(len(labels)):
        for col in range(len(labels)):
            ax.text(
                col,
                row,
                f"{matrix[row, col]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(matrix[row, col]) > 0.55 else "black",
            )
    fig.colorbar(image, ax=ax, label="Pearson r")
    ax.set_title(title)
    save_result_figure(fig, "correlation", filename)


def plot_global_style_scatter_matrix(
    df: pl.DataFrame, cols: list[str], filename: str, title: str
) -> None:
    labels = [global_style_label(col) for col in cols]
    total_points = sum(
        len(global_style_finite_xy(df, x, y)[0]) for y in cols for x in cols
    )
    fig, axes = plt.subplots(
        len(cols),
        len(cols),
        figsize=(2.4 * len(cols), 2.4 * len(cols)),
        sharex="col",
        sharey="row",
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    for row, y_col in enumerate(cols):
        for col, x_col in enumerate(cols):
            ax = axes[row, col]
            x, y = global_style_finite_xy(df, x_col, y_col)
            ax.scatter(
                x,
                y,
                s=5,
                alpha=0.22,
                rasterized=len(x) > SCATTER_RASTERIZE_PANEL_POINTS
                or total_points > 10_000,
            )
            if row == len(cols) - 1:
                ax.set_xlabel(labels[col], fontsize=7)
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel(labels[row], fontsize=7)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.15)
    fig.suptitle(title)
    save_result_figure(fig, "correlation", filename)


def plot_density_scatter_by_altitude(df: pl.DataFrame, altitudes: list[int]) -> None:
    for metric, col_fn, title_prefix in [
        ("mean", tudelft_density_mean_col, r"$\bar{\ell}_\rho$"),
        ("range", tudelft_density_range_col, r"$\Delta\ell_\rho$"),
    ]:
        for altitude in altitudes:
            y_col = col_fn(altitude)
            fig, axes = plt.subplots(
                1,
                len(GLOBAL_STYLE_DRIVER_COLS),
                figsize=fig_size(1.0, 0.36),
                sharey=True,
                constrained_layout=True,
            )
            axes = np.atleast_1d(axes)
            for ax, x_col in zip(axes, GLOBAL_STYLE_DRIVER_COLS):
                x, y = global_style_finite_xy(df, x_col, y_col)
                ax.scatter(x, y, s=6, alpha=0.25, rasterized=scatter_rasterized(len(x)))
                ax.set_xlabel(global_style_label(x_col), fontsize=8)
                ax.set_title(global_style_label(x_col), fontsize=8)
                ax.grid(True, alpha=0.2)
            axes[0].set_ylabel(global_style_label(y_col), fontsize=8)
            fig.suptitle(f"{title_prefix} scatter at {altitude} km")
            save_result_figure(
                fig,
                "correlation",
                "density_scatter_by_altitude",
                f"tudelft_density_{metric}_scatter_{altitude}km.pgf",
            )


def plot_density_scatter_by_mission(df: pl.DataFrame, missions: list[str]) -> None:
    for metric, value_col, title_prefix in [
        ("mean", "density_mean", r"$\bar{\ell}_\rho$"),
        ("range", "density_range", r"$\Delta\ell_\rho$"),
    ]:
        for mission in missions:
            mission_df = df.filter(pl.col("mission") == mission)
            if mission_df.is_empty():
                continue
            fig, axes = plt.subplots(
                1,
                len(GLOBAL_STYLE_DRIVER_COLS),
                figsize=fig_size(1.0, 0.36),
                sharey=True,
                constrained_layout=True,
            )
            axes = np.atleast_1d(axes)
            for ax, x_col in zip(axes, GLOBAL_STYLE_DRIVER_COLS):
                x, y = global_style_finite_xy(mission_df, x_col, value_col)
                ax.scatter(x, y, s=6, alpha=0.25, rasterized=scatter_rasterized(len(x)))
                ax.set_xlabel(global_style_label(x_col), fontsize=8)
                ax.set_title(global_style_label(x_col), fontsize=8)
                ax.grid(True, alpha=0.2)
            axes[0].set_ylabel(title_prefix, fontsize=8)
            fig.suptitle(f"{title_prefix} scatter for {mission}")
            save_result_figure(
                fig,
                "correlation",
                "density_scatter_by_mission",
                f"tudelft_density_{metric}_scatter_{safe_name(mission)}.pgf",
            )


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
        return f"0-{CORRELATION_DURATION_STEP_YEARS} yr"
    bin_end = bin_start + CORRELATION_DURATION_STEP_YEARS
    return f"{bin_start}-{bin_end} yr"


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
        (-1.0, -0.5, "large", "#cfe3f5"),
        (-0.5, -0.3, "medium", "#e3effa"),
        (-0.3, -0.1, "small", "#f2f6fb"),
        (-0.1, 0.1, "negligible", "#fff7e6"),
        (0.1, 0.3, "small", "#f2f6fb"),
        (0.3, 0.5, "medium", "#e3effa"),
        (0.5, 1.0, "large", "#cfe3f5"),
    ]
    for lower, upper, _, color in bands:
        ax.axhspan(lower, upper, color=color, alpha=1.0, zorder=0)
    for threshold in [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5]:
        ax.axhline(threshold, color="0.72", linewidth=0.9, zorder=1)


def plot_correlation_by_axis(
    df: pl.DataFrame,
    axis_values: list[int] | list[str],
    axis_name: str,
    filename: str,
) -> None:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=page_fig_size(1.0, 0.78, 0.95),
        sharex=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.11, right=0.98, top=0.80, bottom=0.12, hspace=0.16)
    observed_duration_bins: set[int] = set()
    for ax, title, metric in [
        (axes[0], r"Mean $\ell_\rho$", "mean"),
        (axes[1], r"$\Delta\ell_\rho$ max/min range", "range"),
    ]:
        add_correlation_effect_size_bands(ax)
        for cause in GLOBAL_STYLE_DRIVER_COLS:
            corr = []
            corr_los = []
            corr_his = []
            duration_bins = []
            for axis_value in axis_values:
                if axis_name == "altitude":
                    y_col = (
                        tudelft_density_mean_col(int(axis_value))
                        if metric == "mean"
                        else tudelft_density_range_col(int(axis_value))
                    )
                    point_corr, r_lo, r_hi, duration_bin = correlation_and_duration(
                        df, cause, y_col
                    )
                else:
                    y_col = "density_mean" if metric == "mean" else "density_range"
                    point_corr, r_lo, r_hi, duration_bin = correlation_and_duration(
                        df.filter(pl.col("mission") == axis_value), cause, y_col
                    )
                corr.append(point_corr)
                corr_los.append(r_lo)
                corr_his.append(r_hi)
                duration_bins.append(duration_bin)
                if duration_bin is not None and np.isfinite(point_corr):
                    observed_duration_bins.add(duration_bin)
            x_values = (
                np.arange(len(axis_values)) if axis_name == "mission" else axis_values
            )
            (line,) = ax.plot(
                x_values,
                corr,
                linewidth=1.4,
                label=global_style_label(cause),
                zorder=3,
            )
            corr_arr = np.asarray(corr, dtype=float)
            lo_arr = np.asarray(corr_los, dtype=float)
            hi_arr = np.asarray(corr_his, dtype=float)
            finite_ci = (
                np.isfinite(corr_arr) & np.isfinite(lo_arr) & np.isfinite(hi_arr)
            )
            if np.any(finite_ci):
                ax.fill_between(
                    np.asarray(x_values)[finite_ci],
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
                        np.asarray(x_values)[mask],
                        np.asarray(corr, dtype=float)[mask],
                        marker=correlation_duration_marker(int(duration_bin)),
                        s=32,
                        color=line.get_color(),
                        edgecolors="black",
                        linewidths=0.45,
                        zorder=4,
                    )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylim(-1, 1)
        ax.set_ylabel("Pearson r")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    if axis_name == "mission":
        axes[-1].set_xticks(
            np.arange(len(axis_values)), axis_values, rotation=45, ha="right"
        )
        axes[-1].set_xlabel("Mission")
    else:
        axes[-1].set_xlabel("Altitude bin lower edge (km)")
    driver_handles, driver_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        driver_handles,
        driver_labels,
        fontsize=8,
        ncols=len(driver_handles),
        loc="upper center",
        bbox_to_anchor=(0.50, 1.00),
        framealpha=0.9,
    )
    if observed_duration_bins:
        fig.legend(
            handles=record_length_handles(observed_duration_bins),
            title="Record length",
            fontsize=8,
            title_fontsize=8,
            ncols=len(observed_duration_bins),
            loc="upper center",
            bbox_to_anchor=(0.50, 1.13),
            framealpha=0.9,
        )
    save_result_figure(fig, "correlation", filename)


def metric_plot_path(
    axis_name: str, driver_col: str, metric_name: str, metric: str
) -> tuple[str, ...]:
    return (
        "correlation",
        "density_co2_fit_metric_plots",
        f"tudelft_density_{metric}_co2_{metric_name}_by_{axis_name}_for_{safe_name(driver_col)}.pgf",
    )


def plot_metric_by_axis_and_sigma(
    axis_name: str,
    driver_col: str,
    axis_values: list[int] | list[str],
    row_labels: np.ndarray,
    metric_name: str,
    metric_label: str,
    metric: str,
    values: np.ndarray,
    duration_bins: np.ndarray | None = None,
    ci_values: np.ndarray | None = None,
) -> None:
    if values.size == 0 or not np.any(np.isfinite(values)):
        return
    fig, ax = plt.subplots(
        figsize=page_fig_size(1.0, 0.58, 0.95), constrained_layout=False
    )
    fig.subplots_adjust(left=0.12, right=0.98, top=0.82, bottom=0.13)
    x_values = np.arange(len(axis_values)) if axis_name == "mission" else axis_values
    observed_duration_bins: set[int] = set()
    for row, sigma_label in enumerate(row_labels):
        row_values = values[row].astype(float)
        if np.any(np.isfinite(row_values)):
            (line,) = ax.plot(
                x_values,
                row_values,
                linewidth=2.2,
                marker="o",
                markersize=3.5,
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
                        np.asarray(x_values)[finite_ci],
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
                            np.asarray(x_values)[mask],
                            row_values[mask],
                            marker=correlation_duration_marker(int(duration_bin)),
                            s=58,
                            color=line.get_color(),
                            edgecolors="black",
                            linewidths=0.65,
                            zorder=4,
                        )
    if axis_name == "mission":
        ax.set_xticks(x_values, axis_values, rotation=45, ha="right")
        ax.set_xlabel("Mission", fontsize=12)
    else:
        ax.set_xlabel("Altitude bin lower edge (km)", fontsize=12)
    ax.set_ylabel(metric_label, fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, alpha=0.3)
    data_handles, data_labels = ax.get_legend_handles_labels()
    fig.legend(
        data_handles,
        data_labels,
        title=f"{GLOBAL_STYLE_SPACE_WEATHER_LABELS[driver_col]} bin",
        fontsize=9,
        title_fontsize=9,
        ncols=min(3, max(1, len(data_handles))),
        loc="upper center",
        bbox_to_anchor=(0.50, 1.03),
        framealpha=0.88,
    )
    if observed_duration_bins:
        fig.legend(
            handles=record_length_handles(observed_duration_bins),
            title="Record length",
            fontsize=9,
            title_fontsize=9,
            ncols=len(observed_duration_bins),
            loc="upper center",
            bbox_to_anchor=(0.50, 1.22),
            framealpha=0.88,
        )
    save_result_figure(
        fig, *metric_plot_path(axis_name, driver_col, metric_name, metric)
    )


def density_co2_metric_arrays(
    df: pl.DataFrame,
    axis_values: list[int],
    driver_col: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    driver_values = (
        df.select(driver_col).drop_nulls()[driver_col].to_numpy().astype(float)
    )
    driver_values = driver_values[np.isfinite(driver_values)]
    if len(driver_values) < MIN_SAMPLES_PER_HEATMAP_CELL:
        empty = np.empty((0, len(axis_values)))
        return np.array([], dtype=object), empty, empty.astype(int), empty, empty

    edges, driver_mean, driver_std, _ = causal_sigma_edges(driver_values)
    work_df = df.with_columns(
        ((pl.col(driver_col) - driver_mean) / driver_std).alias("__driver_sigma")
    )
    slopes = np.full((len(edges) - 1, len(axis_values)), np.nan)
    slope_los = np.full_like(slopes, np.nan)
    slope_his = np.full_like(slopes, np.nan)
    counts = np.zeros_like(slopes, dtype=int)
    year_spans = np.zeros(slopes.shape, dtype=int)

    for row in range(len(edges) - 1):
        upper = (
            pl.col("__driver_sigma") <= edges[row + 1]
            if row == len(edges) - 2
            else pl.col("__driver_sigma") < edges[row + 1]
        )
        bin_df = work_df.filter((pl.col("__driver_sigma") >= edges[row]) & upper)
        for col, axis_value in enumerate(axis_values):
            y_col = (
                tudelft_density_mean_col(int(axis_value))
                if metric == "mean"
                else tudelft_density_range_col(int(axis_value))
            )
            co2, density = global_style_finite_xy(bin_df, "CO2_ppm", y_col)
            counts[row, col] = len(co2)
            _, slopes[row, col], _, _, slope_los[row, col], slope_his[row, col] = (
                global_style_linear_fit_stats(co2, density)
            )
            if counts[row, col] > 0:
                valid_dates = (
                    bin_df.filter(
                        pl.col("CO2_ppm").is_not_null()
                        & pl.col(y_col).is_not_null()
                        & pl.col("CO2_ppm").is_finite()
                        & pl.col(y_col).is_finite()
                    )
                    .select("date")
                    .drop_nulls()
                )
                if valid_dates.height > 0:
                    year_spans[row, col] = (
                        valid_dates["date"].max().year - valid_dates["date"].min().year
                    )

    row_labels = np.array(global_style_sigma_bin_labels(edges))
    painted_rows = np.any(np.isfinite(slopes), axis=1) | np.any(counts > 0, axis=1)
    duration_bins = np.vectorize(correlation_duration_bin)(
        year_spans.astype(float)
    ).astype(float)
    slope_cis = (
        np.stack([slope_los[painted_rows], slope_his[painted_rows]], axis=-1)
        if np.any(painted_rows)
        else np.empty((0, 0, 2))
    )
    return (
        row_labels[painted_rows],
        slopes[painted_rows],
        counts[painted_rows],
        duration_bins[painted_rows],
        slope_cis,
    )


def draw_metric_summary_panel(
    ax: plt.Axes,
    x_values: list[int],
    row_labels: np.ndarray,
    values: np.ndarray,
    duration_bins: np.ndarray,
    ylabel: str,
    driver_label: str,
    show_bin_legend: bool,
    show_duration_legend: bool,
    ci_values: np.ndarray | None = None,
) -> set[int]:
    observed_duration_bins: set[int] = set()
    for row, sigma_label in enumerate(row_labels):
        row_values = values[row].astype(float)
        if not np.any(np.isfinite(row_values)):
            continue
        (line,) = ax.plot(
            x_values,
            row_values,
            linewidth=2.0,
            marker="o",
            markersize=3.5,
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
                    np.asarray(x_values)[finite_ci],
                    row_lo[finite_ci],
                    row_hi[finite_ci],
                    alpha=0.15,
                    color=line.get_color(),
                    zorder=2,
                )
        row_duration_bins = duration_bins[row]
        for duration_bin in sorted(set(row_duration_bins.tolist()) - {None}):
            if not np.isfinite(duration_bin):
                continue
            mask = (row_duration_bins == duration_bin) & np.isfinite(row_values)
            if np.any(mask):
                observed_duration_bins.add(int(duration_bin))
                ax.scatter(
                    np.asarray(x_values)[mask],
                    row_values[mask],
                    marker=correlation_duration_marker(int(duration_bin)),
                    s=54,
                    color=line.get_color(),
                    edgecolors="black",
                    linewidths=0.65,
                    zorder=4,
                )

    ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, alpha=0.3)
    if show_bin_legend:
        data_legend = ax.legend(
            title=f"{driver_label} bin",
            fontsize=8,
            title_fontsize=8,
            loc="lower left",
            bbox_to_anchor=(1.01, 0),
            borderaxespad=0,
            framealpha=0.88,
        )
        ax.add_artist(data_legend)
    if show_duration_legend and observed_duration_bins:
        duration_handles = [
            Line2D(
                [0],
                [0],
                marker=correlation_duration_marker(duration_bin),
                color="white",
                linestyle="None",
                markerfacecolor="0.55",
                markeredgecolor="black",
                markeredgewidth=0.65,
                markersize=9,
                label=correlation_duration_label(duration_bin),
            )
            for duration_bin in sorted(observed_duration_bins)
        ]
        ax.legend(
            handles=duration_handles,
            title="Record length",
            fontsize=8,
            title_fontsize=8,
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
            borderaxespad=0,
            framealpha=0.88,
        )
    return observed_duration_bins


def plot_tudelft_density_co2_metric_summary(
    df: pl.DataFrame, altitudes: list[int]
) -> None:
    drivers = ["AP_AVG", "F10.7_OBS_CENTER81"]
    fig, axes = plt.subplots(
        3,
        2,
        figsize=page_fig_size(1.0, 1.08, 0.98),
        sharex=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.10, right=0.98, top=0.80, bottom=0.10, hspace=0.16, wspace=0.18
    )
    all_duration_bins: set[int] = set()

    for col, driver_col in enumerate(drivers):
        driver_label = GLOBAL_STYLE_SPACE_WEATHER_LABELS[driver_col]
        mean_labels, mean_slopes, mean_counts, mean_duration_bins, mean_cis = (
            density_co2_metric_arrays(df, altitudes, driver_col, "mean")
        )
        range_labels, range_slopes, _, range_duration_bins, range_cis = (
            density_co2_metric_arrays(df, altitudes, driver_col, "range")
        )
        all_duration_bins.update(
            draw_metric_summary_panel(
                axes[0, col],
                altitudes,
                mean_labels,
                mean_slopes,
                mean_duration_bins,
                "Mean slope",
                driver_label,
                show_bin_legend=False,
                show_duration_legend=False,
                ci_values=mean_cis,
            )
        )
        all_duration_bins.update(
            draw_metric_summary_panel(
                axes[1, col],
                altitudes,
                range_labels,
                range_slopes,
                range_duration_bins,
                "Range slope",
                driver_label,
                show_bin_legend=False,
                show_duration_legend=False,
                ci_values=range_cis,
            )
        )
        all_duration_bins.update(
            draw_metric_summary_panel(
                axes[2, col],
                altitudes,
                mean_labels[np.any(np.isfinite(mean_slopes), axis=1)],
                mean_counts[np.any(np.isfinite(mean_slopes), axis=1)].astype(float),
                mean_duration_bins[np.any(np.isfinite(mean_slopes), axis=1)],
                "Sample count",
                driver_label,
                show_bin_legend=False,
                show_duration_legend=False,
            )
        )
        axes[2, col].set_xlabel("Altitude bin lower edge (km)", fontsize=11)
        if col > 0:
            for row in range(axes.shape[0]):
                axes[row, col].set_ylabel("")

        handles, labels = axes[0, col].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            title=f"{driver_label} bin",
            fontsize=7,
            title_fontsize=7,
            ncols=min(3, max(1, len(handles))),
            loc="upper center",
            bbox_to_anchor=(0.50, 1.00 - 0.10 * col),
            framealpha=0.9,
        )

    if all_duration_bins:
        fig.legend(
            handles=record_length_handles(all_duration_bins),
            title="Record length",
            fontsize=7,
            title_fontsize=7,
            ncols=len(all_duration_bins),
            loc="upper center",
            bbox_to_anchor=(0.50, 1.13),
            framealpha=0.9,
        )

    save_result_figure(
        fig,
        "correlation",
        "density_co2_fit_metric_plots",
        "tudelft_density_co2_slope_count_summary_by_altitude.pgf",
    )


def plot_density_co2_correlation_heatmaps_by_axis(
    df: pl.DataFrame,
    axis_values: list[int] | list[str],
    axis_name: str,
    heatmap_subdir: str,
    make_metric_plots: bool,
) -> None:
    for metric, ylabel in [
        ("mean", r"$\ell_\rho$"),
        ("range", r"$\Delta\ell_\rho$ max/min range"),
    ]:
        for driver_col in GLOBAL_STYLE_SPACE_WEATHER_COLS:
            driver_values = (
                df.select(driver_col).drop_nulls()[driver_col].to_numpy().astype(float)
            )
            driver_values = driver_values[np.isfinite(driver_values)]
            if len(driver_values) < MIN_SAMPLES_PER_HEATMAP_CELL:
                continue
            edges, driver_mean, driver_std, _ = causal_sigma_edges(driver_values)
            work_df = df.with_columns(
                ((pl.col(driver_col) - driver_mean) / driver_std).alias(
                    "__driver_sigma"
                )
            )
            correlations = np.full((len(edges) - 1, len(axis_values)), np.nan)
            corr_los = np.full_like(correlations, np.nan)
            corr_his = np.full_like(correlations, np.nan)
            slopes = np.full_like(correlations, np.nan)
            zero_crossings = np.full_like(correlations, np.nan)
            errors = np.full_like(correlations, np.nan)
            slope_los = np.full_like(correlations, np.nan)
            slope_his = np.full_like(correlations, np.nan)
            counts = np.zeros_like(correlations, dtype=int)
            year_ranges = np.full(correlations.shape, "", dtype=object)
            year_spans = np.zeros(correlations.shape, dtype=int)
            for row in range(len(edges) - 1):
                upper = (
                    pl.col("__driver_sigma") <= edges[row + 1]
                    if row == len(edges) - 2
                    else pl.col("__driver_sigma") < edges[row + 1]
                )
                bin_df = work_df.filter(
                    (pl.col("__driver_sigma") >= edges[row]) & upper
                )
                for col, axis_value in enumerate(axis_values):
                    if axis_name == "altitude":
                        y_col = (
                            tudelft_density_mean_col(int(axis_value))
                            if metric == "mean"
                            else tudelft_density_range_col(int(axis_value))
                        )
                        cell = bin_df
                        co2, density = global_style_finite_xy(cell, "CO2_ppm", y_col)
                    else:
                        y_col = "density_mean" if metric == "mean" else "density_range"
                        cell = bin_df.filter(pl.col("mission") == axis_value)
                        co2, density = global_style_finite_xy(cell, "CO2_ppm", y_col)
                    counts[row, col] = len(co2)
                    (
                        correlations[row, col],
                        slopes[row, col],
                        zero_crossings[row, col],
                        errors[row, col],
                        slope_los[row, col],
                        slope_his[row, col],
                    ) = global_style_linear_fit_stats(co2, density)
                    _, r_lo, r_hi, _ = pearsonr_ci(co2, density)
                    corr_los[row, col] = r_lo
                    corr_his[row, col] = r_hi
                    if counts[row, col] > 0:
                        valid_dates = (
                            cell.filter(
                                pl.col("CO2_ppm").is_not_null()
                                & pl.col(y_col).is_not_null()
                                & pl.col("CO2_ppm").is_finite()
                                & pl.col(y_col).is_finite()
                            )
                            .select("date")
                            .drop_nulls()
                        )
                        if valid_dates.height > 0:
                            start_year = valid_dates["date"].min().year
                            end_year = valid_dates["date"].max().year
                            year_spans[row, col] = end_year - start_year
                            year_ranges[row, col] = (
                                f"{start_year % 100:02d}-{end_year % 100:02d}"
                            )
            painted_rows = np.any(np.isfinite(correlations), axis=1)
            if not np.any(painted_rows):
                continue
            row_labels = np.array(global_style_sigma_bin_labels(edges))[painted_rows]
            correlations = correlations[painted_rows]
            corr_los = corr_los[painted_rows]
            corr_his = corr_his[painted_rows]
            slopes = slopes[painted_rows]
            zero_crossings = zero_crossings[painted_rows]
            errors = errors[painted_rows]
            slope_los = slope_los[painted_rows]
            slope_his = slope_his[painted_rows]
            counts = counts[painted_rows]
            year_ranges = year_ranges[painted_rows]
            year_spans = year_spans[painted_rows]
            duration_bins = np.vectorize(correlation_duration_bin)(
                year_spans.astype(float)
            ).astype(float)
            if make_metric_plots:
                corr_ci = np.stack([corr_los, corr_his], axis=-1)
                plot_metric_by_axis_and_sigma(
                    axis_name,
                    driver_col,
                    axis_values,
                    row_labels,
                    "correlation",
                    f"Pearson r({ylabel}, CO2)",
                    metric,
                    correlations,
                    duration_bins,
                    ci_values=corr_ci,
                )
                slope_ci = np.stack([slope_los, slope_his], axis=-1)
                plot_metric_by_axis_and_sigma(
                    axis_name,
                    driver_col,
                    axis_values,
                    row_labels,
                    "slope",
                    f"Linear fit slope ({ylabel} per CO2 ppm)",
                    metric,
                    slopes,
                    duration_bins,
                    ci_values=slope_ci,
                )
                plot_metric_by_axis_and_sigma(
                    axis_name,
                    driver_col,
                    axis_values,
                    row_labels,
                    "error",
                    f"Linear fit RMSE ({ylabel})",
                    metric,
                    errors,
                    duration_bins,
                )
                plot_metric_by_axis_and_sigma(
                    axis_name,
                    driver_col,
                    axis_values,
                    row_labels,
                    "sample_count",
                    "Sample count n",
                    metric,
                    counts.astype(float),
                    duration_bins,
                )

            max_count = int(counts.max()) if counts.size else 0
            count_threshold = max(1, int(np.ceil(0.01 * max_count)))
            display_correlations = np.array(correlations, copy=True)
            display_correlations[counts < count_threshold] = np.nan
            display_correlations[np.abs(display_correlations) < 0.1] = np.nan
            display_correlations[year_spans < 8] = np.nan
            painted_rows = np.any(np.isfinite(display_correlations), axis=1)
            if not np.any(painted_rows):
                continue

            if axis_name == "altitude":
                painted_cols = np.any(np.isfinite(display_correlations), axis=0)
                if np.any(painted_cols):
                    first_col = int(np.argmax(painted_cols))
                    last_col = len(painted_cols) - int(np.argmax(painted_cols[::-1]))
                    visible_cols = np.zeros_like(painted_cols, dtype=bool)
                    visible_cols[first_col:last_col] = True
                else:
                    visible_cols = np.ones(len(axis_values), dtype=bool)
            else:
                visible_cols = np.ones(len(axis_values), dtype=bool)

            visible_axis_values = np.array(axis_values, dtype=object)[visible_cols]
            row_labels = row_labels[painted_rows]
            display_correlations = display_correlations[
                np.ix_(painted_rows, visible_cols)
            ]
            slopes = slopes[np.ix_(painted_rows, visible_cols)]
            zero_crossings = zero_crossings[np.ix_(painted_rows, visible_cols)]
            errors = errors[np.ix_(painted_rows, visible_cols)]
            slope_los = slope_los[np.ix_(painted_rows, visible_cols)]
            slope_his = slope_his[np.ix_(painted_rows, visible_cols)]
            counts = counts[np.ix_(painted_rows, visible_cols)]
            year_ranges = year_ranges[np.ix_(painted_rows, visible_cols)]

            fig, ax = plt.subplots(
                figsize=(
                    max(9, 0.85 * len(visible_axis_values)),
                    max(4.5, 0.55 * len(row_labels)),
                ),
                constrained_layout=True,
            )
            cmap = plt.get_cmap("coolwarm").copy()
            cmap.set_bad(color="lightgray")
            image = ax.imshow(
                np.ma.masked_invalid(display_correlations),
                aspect="auto",
                origin="lower",
                vmin=-1,
                vmax=1,
                cmap=cmap,
            )
            axis_labels = (
                [f"{value} km" for value in visible_axis_values]
                if axis_name == "altitude"
                else [str(value) for value in visible_axis_values]
            )
            ax.set_xticks(
                np.arange(len(visible_axis_values)),
                axis_labels,
                rotation=45,
                ha="right",
            )
            ax.set_yticks(np.arange(len(row_labels)), row_labels)
            ax.set_xlabel("Altitude" if axis_name == "altitude" else "Mission")
            ax.set_ylabel(
                f"{GLOBAL_STYLE_SPACE_WEATHER_LABELS[driver_col]} bins\nmean={driver_mean:.2f}, sigma={driver_std:.2f}"
            )
            ax.set_xticks(np.arange(-0.5, len(visible_axis_values), 1), minor=True)
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
                            global_style_fit_annotation(
                                value,
                                slopes[row, col],
                                zero_crossings[row, col],
                                errors[row, col],
                                counts[row, col],
                                year_ranges[row, col],
                                slope_los[row, col],
                                slope_his[row, col],
                            ),
                            ha="center",
                            va="center",
                            fontsize=6,
                            color="white" if abs(value) > 0.55 else "black",
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
            fig.colorbar(image, ax=ax, label=f"Pearson r({ylabel}, CO2)")
            ax.set_title(
                f"TuDelft {ylabel} vs CO2 correlation by {GLOBAL_STYLE_SPACE_WEATHER_LABELS[driver_col]} bin"
            )
            save_result_figure(
                fig,
                "heatmaps",
                heatmap_subdir,
                f"tudelft_density_{metric}_co2_correlation_by_{safe_name(driver_col)}.pgf",
            )


def plot_global_style_correlation_and_heatmaps(
    missions: list[tuple[MissionConfig, pl.DataFrame]],
) -> None:
    altitude_df, altitudes, mission_df, mission_names = build_global_style_tables(
        missions
    )
    major_altitudes = selected_altitudes_for_analysis(altitudes)
    selected_cols = [
        *GLOBAL_STYLE_DRIVER_COLS,
        *[
            col
            for altitude in major_altitudes
            for col in [
                tudelft_density_mean_col(altitude),
                tudelft_density_range_col(altitude),
            ]
        ],
    ]
    plot_global_style_correlation_heatmap(
        altitude_df,
        GLOBAL_STYLE_DRIVER_COLS,
        "tudelft_correlation_analysis_variables.pgf",
        "TuDelft driver correlation",
    )
    plot_global_style_correlation_heatmap(
        altitude_df,
        selected_cols,
        "tudelft_correlation_all_selected_variables.pgf",
        "TuDelft selected-variable correlation",
    )
    plot_global_style_scatter_matrix(
        altitude_df,
        GLOBAL_STYLE_DRIVER_COLS,
        "tudelft_scatter_analysis_variables.pgf",
        "TuDelft driver scatter matrix",
    )
    plot_density_scatter_by_altitude(altitude_df, major_altitudes)
    plot_density_scatter_by_mission(mission_df, mission_names)
    plot_correlation_by_axis(
        altitude_df,
        altitudes,
        "altitude",
        "tudelft_correlation_by_altitude.pgf",
    )
    plot_correlation_by_axis(
        mission_df,
        mission_names,
        "mission",
        "tudelft_correlation_by_mission.pgf",
    )
    plot_density_co2_correlation_heatmaps_by_axis(
        altitude_df,
        altitudes,
        "altitude",
        "density_co2_correlation_heatmaps_all_altitudes",
        False,
    )
    plot_density_co2_correlation_heatmaps_by_axis(
        altitude_df,
        major_altitudes,
        "altitude",
        "density_co2_correlation_heatmaps",
        True,
    )
    plot_density_co2_correlation_heatmaps_by_axis(
        mission_df,
        mission_names,
        "mission",
        "density_co2_correlation_heatmaps_by_mission",
        True,
    )
    plot_tudelft_density_co2_metric_summary(altitude_df, major_altitudes)


def causal_output_path(*parts: str) -> Path:
    return output_path(*CAUSAL_WORKFLOW_DIR, *parts)


def save_causal_figure(fig: plt.Figure, *parts: str) -> None:
    out = causal_output_path(*parts)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def selected_altitudes_for_analysis(altitudes: list[int]) -> list[int]:
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
                f"rho_{altitude}_daily_range",
            ]
        )
    return cols


def density_mean_cols_for_altitudes(altitudes: list[int]) -> list[str]:
    return [f"log10rho_{altitude}_daily_mean" for altitude in altitudes]


def density_range_cols_for_altitudes(altitudes: list[int]) -> list[str]:
    return [f"log10rho_{altitude}_daily_range" for altitude in altitudes]


def rho_range_cols_for_altitudes(altitudes: list[int]) -> list[str]:
    return [f"rho_{altitude}_daily_range" for altitude in altitudes]


def label_for_causal_col(col: str) -> str:
    if col.startswith("rho_") and col.endswith("_daily_range"):
        return col.replace("rho_", "density range ").replace(
            "_daily_range", " km daily max-min"
        )
    if col.startswith("log10rho_") and col.endswith("_daily_range"):
        return col.replace("log10rho_", r"$\Delta\ell_\rho$ range ").replace(
            "_daily_range", " km daily max/min"
        )
    if col.startswith("log10rho_"):
        return col.replace("log10rho_", r"$\ell_\rho$ ").replace(
            "_daily_", " km daily "
        )
    return {
        "F10.7_OBS_CENTER81": "F10.7 81d avg",
        "AP_AVG": "Ap",
        "KP_SUM": "Kp",
        "CO2_ppm": "CO2",
    }.get(col, col)


def build_tudelft_causal_density_table(
    missions: list[tuple[MissionConfig, pl.DataFrame]],
) -> tuple[pl.DataFrame, list[str], list[int], list[int]]:
    combined = pl.concat(
        [
            df.with_columns(pl.col("timestamp").dt.date().alias("date")).select(
                "date", "altitude_km", DENSITY_COL, "Density (kg/m^3)"
            )
            for _, df in missions
            if df.height
        ]
    ).drop_nulls(["date", "altitude_km", DENSITY_COL, "Density (kg/m^3)"])
    long_df = (
        combined.with_columns(
            (
                (pl.col("altitude_km") / CAUSAL_ALTITUDE_BIN_KM).floor()
                * CAUSAL_ALTITUDE_BIN_KM
            )
            .cast(pl.Int32)
            .alias("altitude_bin_km")
        )
        .group_by("date", "altitude_bin_km")
        .agg(
            pl.col(DENSITY_COL).min().alias("log10rho_daily_min"),
            pl.col(DENSITY_COL).mean().alias("log10rho_daily_mean"),
            pl.col(DENSITY_COL).max().alias("log10rho_daily_max"),
            (pl.col("Density (kg/m^3)").max() - pl.col("Density (kg/m^3)").min()).alias(
                "rho_daily_range"
            ),
            pl.len().alias("samples"),
        )
        .filter(pl.col("samples") >= MIN_SAMPLES_PER_HEATMAP_CELL)
        .with_columns(
            (pl.col("log10rho_daily_max") - pl.col("log10rho_daily_min")).alias(
                "log10rho_daily_range"
            )
        )
        .sort("date", "altitude_bin_km")
    )
    long_df.write_parquet(
        causal_output_path("tudelft_daily_altitude_binned_long.parquet"),
        compression="lz4",
    )

    altitudes = [
        int(alt) for alt in long_df["altitude_bin_km"].unique().sort().to_list()
    ]
    selected_altitudes = selected_altitudes_for_analysis(altitudes)
    selected_density_cols = density_cols_for_altitudes(selected_altitudes)

    wide: pl.DataFrame | None = None
    pivot_specs = [
        ("log10rho_daily_min", "log10rho_{altitude}_daily_min"),
        ("log10rho_daily_mean", "log10rho_{altitude}_daily_mean"),
        ("log10rho_daily_max", "log10rho_{altitude}_daily_max"),
        ("log10rho_daily_range", "log10rho_{altitude}_daily_range"),
        ("rho_daily_range", "rho_{altitude}_daily_range"),
    ]
    selected_long = long_df.filter(pl.col("altitude_bin_km").is_in(selected_altitudes))
    for value_col, output_template in pivot_specs:
        pivot = (
            selected_long.with_columns(
                pl.col("altitude_bin_km").cast(pl.Utf8).alias("altitude_label")
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
            for altitude in selected_altitudes
            if str(altitude) in pivot.columns
        }
        pivot = pivot.rename(rename_map)
        wide = (
            pivot
            if wide is None
            else wide.join(pivot, on="date", how="full", coalesce=True)
        )
    if wide is None:
        raise RuntimeError("No TuDelft causal density columns were created.")
    wide = wide.sort("date")
    wide.write_parquet(
        causal_output_path("tudelft_daily_altitude_binned_wide.parquet"),
        compression="lz4",
    )
    causal_output_path("tudelft_causal_selection.txt").write_text(
        "TuDelft daily causal table uses observed-density samples binned by "
        f"{CAUSAL_ALTITUDE_BIN_KM} km altitude lower edge.\n"
        f"Available altitude bins km={altitudes}\n"
        f"Selected altitude bins km={selected_altitudes}\n",
        encoding="utf-8",
    )
    return wide, selected_density_cols, altitudes, selected_altitudes


def combine_causal_inputs(
    missions: list[tuple[MissionConfig, pl.DataFrame]],
) -> tuple[pl.DataFrame, pl.DataFrame, list[str], list[int], list[int]]:
    density, density_cols, altitude_bins, selected_altitudes = (
        build_tudelft_causal_density_table(missions)
    )
    drivers = (
        load_driver_data()
        .rename(CAUSAL_DRIVER_RENAMES)
        .select("date", *CAUSAL_DRIVER_COLS)
        .unique(subset="date")
        .sort("date")
    )
    density = (
        density.with_columns(pl.col("date").cast(pl.Date))
        .unique(subset="date")
        .sort("date")
    )
    density_start = min(
        density.filter(pl.col(col).is_not_null())["date"].min() for col in density_cols
    )
    density_end = max(
        density.filter(pl.col(col).is_not_null())["date"].max() for col in density_cols
    )
    start_date = max(
        density_start,
        drivers.filter(pl.col(CAUSAL_Y_COLS[0]).is_not_null())["date"].min(),
        drivers.filter(pl.col(CAUSAL_X_COL).is_not_null())["date"].min(),
    )
    end_date = min(
        density_end,
        drivers.filter(pl.col(CAUSAL_Y_COLS[0]).is_not_null())["date"].max(),
        drivers.filter(pl.col(CAUSAL_X_COL).is_not_null())["date"].max(),
    )
    all_cols = [*CAUSAL_DRIVER_COLS, *density_cols]
    combined = (
        density.join(drivers, on="date", how="full", coalesce=True)
        .sort("date")
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
    missing_summary.write_csv(
        causal_output_path("missing_summary_before_interpolation.csv")
    )
    interpolated.write_csv(causal_output_path("daily_analysis_dataset.csv"))
    return (
        interpolated,
        missing_summary,
        density_cols,
        altitude_bins,
        selected_altitudes,
    )


def combine_causal_mission_inputs(
    missions: list[tuple[MissionConfig, pl.DataFrame]],
) -> pl.DataFrame:
    frames = []
    for config, df in missions:
        frames.append(
            df.with_columns(pl.col("timestamp").dt.date().alias("date"))
            .group_by("date")
            .agg(
                pl.col(DENSITY_COL).mean().alias("log10_density_daily_mean"),
                (pl.col(DENSITY_COL).max() - pl.col(DENSITY_COL).min()).alias(
                    "log10_density_daily_range"
                ),
                (
                    pl.col("Density (kg/m^3)").max() - pl.col("Density (kg/m^3)").min()
                ).alias("density_daily_range"),
                pl.len().alias("samples"),
            )
            .filter(pl.col("samples") >= MIN_SAMPLES_PER_HEATMAP_CELL)
            .with_columns(pl.lit(config.name).alias("mission"))
        )
    mission_density = pl.concat(frames).sort("mission", "date")
    drivers = (
        load_driver_data()
        .rename(CAUSAL_DRIVER_RENAMES)
        .select("date", *CAUSAL_DRIVER_COLS)
        .unique(subset="date")
        .sort("date")
    )
    return drivers.join(mission_density, on="date", how="inner").drop_nulls(
        CAUSAL_DRIVER_COLS
    )


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


def make_variants(df: pl.DataFrame, all_cols: list[str]) -> dict[str, Variant]:
    dates = df["date"].to_numpy()
    raw = {col: df[col].to_numpy().astype(float) for col in all_cols}
    seasonal = {col: seasonal_anomaly(raw[col], dates) for col in all_cols}
    detrended_seasonal = {col: detrended(seasonal[col]) for col in all_cols}
    co2_preserved = {
        col: seasonal[col] if col != CAUSAL_X_COL else raw[col] for col in all_cols
    }
    return {
        "raw_standardized": Variant(
            "raw_standardized",
            dates,
            {col: finite_standardize(raw[col]) for col in all_cols},
            "Raw daily values after temporal gap filling, standardized.",
        ),
        "seasonal_anomaly": Variant(
            "seasonal_anomaly",
            dates,
            {col: finite_standardize(seasonal[col]) for col in all_cols},
            "Day-of-year climatology removed, then standardized.",
        ),
        "detrended_anomaly": Variant(
            "detrended_anomaly",
            dates,
            {col: finite_standardize(detrended_seasonal[col]) for col in all_cols},
            "Seasonal anomalies with a 3-year rolling mean removed.",
        ),
        "co2_preserved_anomaly": Variant(
            "co2_preserved_anomaly",
            dates,
            {col: finite_standardize(co2_preserved[col]) for col in all_cols},
            "Seasonal anomalies, but CO2 kept as a slow standardized driver.",
        ),
    }


def plot_causal_correlation_heatmap(
    variant: Variant, all_cols: list[str], labels: dict[str, str]
) -> np.ndarray:
    matrix = np.array(
        [
            [pearsonr(variant.data[row], variant.data[col]) for col in all_cols]
            for row in all_cols
        ]
    )
    tick_labels = [labels[col] for col in all_cols]
    size = max(10, min(30, 0.34 * len(tick_labels)))
    fig, ax = plt.subplots(figsize=(size, size), constrained_layout=True)
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm", rasterized=True)
    ax.set_xticks(np.arange(len(tick_labels)), tick_labels, rotation=90, fontsize=6)
    ax.set_yticks(np.arange(len(tick_labels)), tick_labels, fontsize=6)
    ax.set_title(f"Correlation matrix: {variant.name}")
    fig.colorbar(image, ax=ax, label="Pearson r")
    save_causal_figure(fig, f"correlation_{variant.name}.pgf")
    return matrix


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
    labels: dict[str, str],
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
            ax.scatter(x, y, s=6, alpha=0.25, rasterized=True)
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
                ax.set_title(labels[x_col], fontsize=8)
            if row == len(target_cols) - 1:
                ax.set_xlabel(labels[x_col], fontsize=8)
            if col == 0:
                ax.set_ylabel(labels[target_col], fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.2)
    fig.suptitle(title)
    save_causal_figure(fig, filename)


def plot_causal_correlation_by_altitude(
    variant: Variant,
    selected_altitudes: list[int],
    labels: dict[str, str],
) -> pl.DataFrame:
    rows = []
    for metric, template in [
        ("daily_mean_log10_density", "log10rho_{altitude}_daily_mean"),
        ("daily_log10_density_range", "log10rho_{altitude}_daily_range"),
    ]:
        for altitude in selected_altitudes:
            target_col = template.format(altitude=altitude)
            if target_col not in variant.data:
                continue
            for cause in CAUSAL_DRIVER_COLS:
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
        for cause in CAUSAL_DRIVER_COLS:
            series = metric_df.filter(pl.col("cause") == cause).sort("altitude_km")
            if series.is_empty():
                continue
            ax.plot(
                series["altitude_km"],
                series["correlation"],
                marker="o",
                linewidth=1.4,
                label=labels[cause],
            )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylim(-1, 1)
        ax.set_ylabel("Pearson r")
        ax.set_title(metric.replace("_", " "))
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("TuDelft altitude bin lower edge (km)")
    axes[0].legend(
        fontsize=8, ncols=2, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0
    )
    fig.suptitle(f"Correlation by selected TuDelft altitude bin: {variant.name}")
    save_causal_figure(fig, f"correlation_by_altitude_{variant.name}.pgf")
    return table


def causal_lag_correlations(
    variant: Variant, target_cols: list[str], max_lag: int = CAUSAL_MAX_LAG_DAYS
) -> pl.DataFrame:
    rows = []
    for target in target_cols:
        y = variant.data[target]
        for cause in CAUSAL_DRIVER_COLS:
            x = variant.data[cause]
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


def plot_causal_lag_correlations(
    corr_df: pl.DataFrame,
    variant_name: str,
    target_cols: list[str],
    labels: dict[str, str],
) -> None:
    for target in target_cols:
        target_df = corr_df.filter(pl.col("target") == target)
        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
        for cause in CAUSAL_DRIVER_COLS:
            series = target_df.filter(pl.col("cause") == cause).sort("lag_days")
            if series.is_empty():
                continue
            ax.plot(
                series["lag_days"],
                series["correlation"],
                linewidth=1.5,
                label=labels[cause],
            )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Lag from cause to target (days)")
        ax.set_ylabel("Pearson r")
        ax.set_title(f"Lag correlations with {labels[target]}: {variant_name}")
        ax.grid(True, alpha=0.25)
        ax.legend(
            fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0
        )
        save_causal_figure(fig, f"lag_correlations_{variant_name}_{target}.pgf")


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


def causal_sigma_edges(
    values: np.ndarray,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    mean = float(np.mean(values))
    std = float(np.std(values))
    if not np.isfinite(std) or std == 0:
        raise ValueError("Cannot build sigma bins for a constant or invalid variable.")
    z = (values - mean) / std
    edges = np.arange(np.floor(np.min(z)), np.ceil(np.max(z)) + 1, 1.0)
    if len(edges) < 2:
        edges = np.array([-0.5, 0.5])
    return edges, mean, std, z


def binned_causal_value_stats(
    df: pl.DataFrame, y_col: str, value_col: str
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]
]:
    data = df.select(CAUSAL_X_COL, y_col, value_col).drop_nulls()
    x = data[CAUSAL_X_COL].to_numpy().astype(float)
    y = data[y_col].to_numpy().astype(float)
    values = data[value_col].to_numpy().astype(float)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
    x, y, values = x[finite], y[finite], values[finite]
    x_edges, x_mean, x_std, x_z = causal_sigma_edges(x)
    y_edges, y_mean, y_std, y_z = causal_sigma_edges(y)
    counts, _, _ = np.histogram2d(y_z, x_z, bins=[y_edges, x_edges])
    sums, _, _ = np.histogram2d(y_z, x_z, bins=[y_edges, x_edges], weights=values)
    sums_sq, _, _ = np.histogram2d(y_z, x_z, bins=[y_edges, x_edges], weights=values**2)
    mean_values = np.divide(
        sums,
        counts,
        out=np.full_like(sums, np.nan, dtype=float),
        where=counts >= CAUSAL_MIN_SAMPLES_PER_CELL,
    )
    variance = (
        np.divide(
            sums_sq,
            counts,
            out=np.full_like(sums_sq, np.nan, dtype=float),
            where=counts >= CAUSAL_MIN_SAMPLES_PER_CELL,
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


def plot_causal_value_heatmaps(
    df: pl.DataFrame,
    value_col: str,
    labels: dict[str, str],
    output_prefix: str,
    colorbar_label: str,
) -> None:
    stats_by_driver = {
        y_col: binned_causal_value_stats(df, y_col, value_col)
        for y_col in CAUSAL_Y_COLS
    }
    finite_means = np.concatenate(
        [
            ((mean_values - metadata["value_mean"]) / metadata["value_std"])[
                np.isfinite(mean_values)
            ]
            for _, _, _, mean_values, _, metadata in stats_by_driver.values()
            if metadata["value_std"] > 0
        ]
    )
    max_abs = float(np.max(np.abs(finite_means))) if len(finite_means) else 1.0
    fig, axes = plt.subplots(
        1,
        len(CAUSAL_Y_COLS),
        figsize=(5.4 * len(CAUSAL_Y_COLS), 6.4),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    mesh = None
    for ax, y_col in zip(axes, CAUSAL_Y_COLS):
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
            rasterized=True,
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
            f"{labels[y_col]} sigma bins\nmean={metadata['y_mean']:.2f}, sigma={metadata['y_std']:.2f}"
        )
        set_integer_sigma_ticks(ax, x_edges, y_edges)
        ax.grid(True, color="white", alpha=0.25, linewidth=0.7)
    if mesh is not None:
        colorbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), shrink=0.9)
        colorbar.set_label(colorbar_label)
    fig.suptitle(f"{labels[value_col]}: mean +/- std by bin")
    safe_value_col = safe_name(value_col)
    if output_prefix == "density_range_heatmap_rho" and safe_value_col.startswith(
        "rho_"
    ):
        safe_value_col = safe_value_col.removeprefix("rho_")
    save_causal_figure(
        fig, f"{output_prefix}_{safe_value_col}_combined_co2_vs_space_weather.pgf"
    )


def run_causal_analysis(missions: list[tuple[MissionConfig, pl.DataFrame]]) -> None:
    (
        analysis_df,
        _missing_summary,
        density_cols,
        _altitude_bins,
        selected_altitudes,
    ) = combine_causal_inputs(missions)
    selected_density_mean_cols = density_mean_cols_for_altitudes(selected_altitudes)
    selected_density_range_cols = density_range_cols_for_altitudes(selected_altitudes)
    selected_rho_range_cols = rho_range_cols_for_altitudes(selected_altitudes)
    target_cols = [*selected_density_mean_cols, *selected_density_range_cols]
    all_cols = [*CAUSAL_DRIVER_COLS, *density_cols]
    labels = {col: label_for_causal_col(col) for col in all_cols}
    mission_analysis_df = combine_causal_mission_inputs(missions)
    mission_labels = {
        **{col: label_for_causal_col(col) for col in CAUSAL_DRIVER_COLS},
        "density_daily_range": "density daily max-min",
    }

    variants = make_variants(analysis_df, all_cols)
    variant_notes = "\n".join(
        f"{variant.name}: {variant.description}" for variant in variants.values()
    )
    causal_output_path("variant_notes.txt").write_text(variant_notes, encoding="utf-8")

    plot_target_scatter_grid(
        analysis_df,
        selected_density_mean_cols,
        CAUSAL_DRIVER_COLS,
        labels,
        "scatter_density_mean_selected_altitudes.pgf",
        r"Daily mean TU Delft $\ell_\rho$ scatter by selected altitude bin",
    )
    plot_target_scatter_grid(
        analysis_df,
        selected_density_range_cols,
        CAUSAL_DRIVER_COLS,
        labels,
        "scatter_density_range_selected_altitudes.pgf",
        r"Daily TU Delft $\Delta\ell_\rho$ max/min range scatter by selected altitude bin",
    )

    correlation_tables = []
    lag_correlation_tables = []
    altitude_correlation_tables = []
    for variant in variants.values():
        corr = plot_causal_correlation_heatmap(variant, all_cols, labels)
        correlation_tables.append(
            pl.DataFrame(corr, schema=all_cols, orient="row").with_columns(
                pl.Series("row", all_cols), pl.lit(variant.name).alias("variant")
            )
        )
        altitude_corr = plot_causal_correlation_by_altitude(
            variant, selected_altitudes, labels
        )
        if not altitude_corr.is_empty():
            altitude_correlation_tables.append(altitude_corr)
        lag_df = causal_lag_correlations(variant, target_cols)
        lag_df.write_csv(causal_output_path(f"lag_correlations_{variant.name}.csv"))
        plot_causal_lag_correlations(lag_df, variant.name, target_cols, labels)
        lag_correlation_tables.append(lag_df)

    pl.concat(correlation_tables).write_csv(
        causal_output_path("correlation_matrices.csv")
    )
    pl.concat(lag_correlation_tables).write_csv(
        causal_output_path("lag_correlations_all_variants.csv")
    )
    if altitude_correlation_tables:
        pl.concat(altitude_correlation_tables).write_csv(
            causal_output_path("correlations_by_selected_altitude.csv")
        )

    for density_col in selected_density_mean_cols:
        plot_causal_value_heatmaps(
            analysis_df,
            density_col,
            labels,
            "density_heatmap",
            r"Mean daily TU Delft $\ell_\rho$ sigma",
        )
    for density_range_col in selected_density_range_cols:
        plot_causal_value_heatmaps(
            analysis_df,
            density_range_col,
            labels,
            "density_range_heatmap",
            r"Mean daily TU Delft $\Delta\ell_\rho$ sigma",
        )
    for rho_range_col in selected_rho_range_cols:
        plot_causal_value_heatmaps(
            analysis_df,
            rho_range_col,
            labels,
            "density_range_heatmap_rho",
            "Mean daily TuDelft density range sigma",
        )
    for mission in [MISSIONS[code].name for code in MISSION_ORDER]:
        mission_df = mission_analysis_df.filter(pl.col("mission") == mission)
        if mission_df.is_empty():
            continue
        plot_causal_value_heatmaps(
            mission_df,
            "density_daily_range",
            mission_labels,
            f"density_range_heatmap_rho_{safe_name(mission)}",
            "Mean daily TuDelft density range sigma",
        )


def write_latex_index() -> None:
    captions = {
        "tudelft_density_timeseries_combined.pgf": "Daily TuDelft observed density and mission altitude coverage.",
        "tudelft_density_driver_spectra.pgf": "Normalized Lomb--Scargle spectra for TuDelft observed density and external drivers.",
        "tudelft_density_correlation_summary.pgf": "Mission-wise Pearson correlations between TuDelft observed density and external drivers.",
        "tudelft_density_altitude_activity_binned_summary.pgf": "TuDelft observed-density CO2 fitted slope grouped by altitude and one-sigma activity bins.",
    }
    root = OUTPUT_ROOT
    for directory in sorted(
        [root, *[path for path in root.rglob("*") if path.is_dir()]],
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        figures = sorted(
            path for path in directory.iterdir() if path.suffix.lower() == ".pgf"
        )
        child_indexes = sorted(
            path / "figures.tex"
            for path in directory.iterdir()
            if path.is_dir() and (path / "figures.tex").exists()
        )
        if not figures and not child_indexes:
            continue
        rel_dir = directory.relative_to(root)
        title = (
            "TuDelft Density Figures"
            if rel_dir == Path(".")
            else str(rel_dir).replace("_", " ").title()
        )
        lines = [f"\\subsection*{{{title}}}", ""]
        for figure in figures:
            rel_path = figure.relative_to("outputs").as_posix()
            lines.extend(
                [
                    r"\begin{figure}[H]",
                    r"\centering",
                    rf"\includegraphics[width=\textwidth,height=\textheight,keepaspectratio]{{\detokenize{{{rel_path}}}}}",
                    rf"\caption{{{captions.get(figure.name, figure.stem.replace('_', ' ').title())}}}",
                    r"\end{figure}",
                    "",
                ]
            )
        for child_index in child_indexes:
            rel_path = child_index.relative_to("outputs").as_posix()
            lines.extend([rf"\input{{\detokenize{{{rel_path}}}}}", ""])
        (directory / "figures.tex").write_text("\n".join(lines), encoding="utf-8")


def mission_summary(missions: list[tuple[MissionConfig, pl.DataFrame]]) -> None:
    rows = []
    for config, df in missions:
        rows.append(
            {
                "mission": config.name,
                "rows": df.height,
                "start": df["timestamp"].min(),
                "end": df["timestamp"].max(),
                "min_alt_km": float(df["altitude_km"].min()),
                "max_alt_km": float(df["altitude_km"].max()),
            }
        )
    table = pl.DataFrame(rows)
    table.write_csv(output_path("tudelft_density_mission_summary.csv"))


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    missions = load_all_density()
    mission_summary(missions)
    print("Plotting TuDelft density timeseries")
    plot_timeseries(missions)
    print("Plotting TuDelft density spectra")
    plot_frequency(missions)
    print("Plotting TuDelft density correlations")
    corr = correlation_table(missions)
    plot_correlation_summary(corr)
    print("Plotting TuDelft density binned summary")
    bins = binned_density_table(missions)
    plot_binned_summary(bins)
    print("Plotting TuDelft global-style correlation and heatmap figures")
    plot_global_style_correlation_and_heatmaps(missions)
    print("Running TuDelft density causal analysis")
    run_causal_analysis(missions)
    write_latex_index()
    print(f"Generated TuDelft density outputs in {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
