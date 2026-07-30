from __future__ import annotations

# ruff: noqa: E402

from pathlib import Path

from scripts.pgf_config import TEXTWIDTH_IN, configure_pgf, fig_size, page_fig_size

configure_pgf()

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import polars as pl

from scripts.hasdm_msis_model_error_analysis import (
    ANALYSIS_COLS,
    HASDM_ALT_COL,
    HASDM_DENSITY_COL,
    MODEL_ERROR_COLS,
    MODEL_VERSIONS,
    altitude_bin_edges,
    date_edges_for_heatmap,
    fft_period_power,
    label_for_col,
    log_period_edges,
    load_co2,
    load_space_weather,
    safe_name,
)
from scripts.stats_utils import ols_slope_ci, pearsonr_ci

SOURCE_SAMPLE_PATH = Path(
    "outputs/figures/results/hasdm_msis_model_errors/data/hasdm_msis_errors_nearest_timestamp_grid_samples.parquet"
)
OUTPUT_ROOT = Path("outputs/figures/results/maunaloa_msis_density_baselines")
DATA_DIR = OUTPUT_ROOT / "data"
ACTIVITY_DRIVERS = ["F10.7_OBS_CENTER81", "AP_AVG"]
ANALYSIS_LABELS = [label_for_col(col) for col in ANALYSIS_COLS]
ACTIVITY_DRIVER_LABELS = {driver: label_for_col(driver) for driver in ACTIVITY_DRIVERS}
FIT_METRIC_NAMES = ["correlation", "slope", "error", "sample_count"]
MIN_SAMPLES_PER_HEATMAP_CELL = 20
HEATMAP_COLORBAR_HEIGHT_INCHES = 3.1
FIGURE_EXTENSIONS = {".pdf", ".pgf", ".png", ".jpg", ".jpeg"}
CORRELATION_DURATION_STEP_YEARS = 11
CORRELATION_DURATION_MARKERS = {
    0: "o",
    11: "s",
    22: "D",
    33: "^",
    44: "P",
    55: "X",
}


def save_figure(fig: plt.Figure, *parts: str) -> None:
    out = OUTPUT_ROOT.joinpath(*parts)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    if out.suffix == ".pgf":
        fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def unlink_figure(*parts: str) -> None:
    path = OUTPUT_ROOT.joinpath(*parts)
    if path.exists():
        path.unlink()


def format_altitude_axis(ax: plt.Axes, axis: str = "y") -> None:
    target = ax.yaxis if axis == "y" else ax.xaxis
    target.get_offset_text().set_visible(False)


def finite_xy(
    df: pl.DataFrame, x_col: str, y_col: str
) -> tuple[np.ndarray, np.ndarray]:
    if x_col not in df.columns or y_col not in df.columns:
        return np.array([]), np.array([])
    pair = df.select(x_col, y_col).drop_nulls()
    x = pair[x_col].to_numpy().astype(float)
    y = pair[y_col].to_numpy().astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def density_col(model_name: str, altitude: int) -> str:
    return f"{safe_name(model_name)}_log10rho_daily_mean_{altitude}km"


def available_density_altitudes(df: pl.DataFrame, model_name: str) -> list[int]:
    prefix = f"{safe_name(model_name)}_log10rho_daily_mean_"
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
    return f"{bin_start} to {bin_start + CORRELATION_DURATION_STEP_YEARS} yr"


def correlation_duration_marker(bin_start: int) -> str:
    return CORRELATION_DURATION_MARKERS.get(bin_start, "*")


def correlation_and_duration(
    df: pl.DataFrame, x_col: str, y_col: str
) -> tuple[float, float, float, int | None]:
    if y_col not in df.columns:
        return np.nan, np.nan, np.nan, None
    data = df.select("date", x_col, y_col).drop_nulls().sort("date")
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
        fontsize=7,
        title_fontsize=7,
        loc="upper right",
        bbox_to_anchor=(1, -0.22),
        borderaxespad=0,
        ncol=min(2, len(handles)),
    )


def sigma_bin_labels(edges: np.ndarray) -> list[str]:
    return [f"{edges[idx]:g}-{edges[idx + 1]:g}" for idx in range(len(edges) - 1)]


def text_color_for_correlation(value: float) -> str:
    if not np.isfinite(value):
        return "black"
    return "white" if abs(value) > 0.55 else "black"


def linear_fit_stats(
    x: np.ndarray, y: np.ndarray
) -> tuple[float, float, float, float, float, float]:
    if len(x) < MIN_SAMPLES_PER_HEATMAP_CELL:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    slope, slope_lo, slope_hi, _se, _intercept, correlation, error, _n = ols_slope_ci(
        x, y
    )
    return correlation, float(slope), error, slope_lo, slope_hi, float(len(x))


def fit_annotation(
    correlation: float,
    slope: float,
    error: float,
    count: int,
    slope_lo: float = np.nan,
    slope_hi: float = np.nan,
) -> str:
    ci_str = ""
    if np.isfinite(slope_lo) and np.isfinite(slope_hi):
        ci_str = f"\nCI=[{slope_lo:.2e}, {slope_hi:.2e}]"
    return f"r={correlation:.2f}\nm={slope:.2e}{ci_str}\nerr={error:.3f}\nn={count}"


def metric_plot_filename(model_name: str, driver_col: str, metric_name: str) -> str:
    driver_name = safe_name(driver_col).replace("_", "-")
    return (
        f"{safe_name(model_name)}_msis_density_baseline_co2_"
        f"{metric_name}_by_altitude_for_{driver_name}.png"
    )


def cleanup_metric_plots_for_driver(model_name: str, driver_col: str) -> None:
    for metric_name in FIT_METRIC_NAMES:
        unlink_figure(
            "correlation",
            "density_co2_fit_metric_plots",
            metric_plot_filename(model_name, driver_col, metric_name),
        )


def plot_metric_by_altitude_and_sigma(
    model_name: str,
    driver_col: str,
    altitudes: list[int],
    row_labels: np.ndarray,
    metric_name: str,
    metric_label: str,
    values: np.ndarray,
    duration_bins: np.ndarray | None = None,
    ci_values: np.ndarray | None = None,
) -> None:
    parts = (
        "correlation",
        "density_co2_fit_metric_plots",
        metric_plot_filename(model_name, driver_col, metric_name),
    )
    if values.size == 0 or not np.any(np.isfinite(values)):
        unlink_figure(*parts)
        return

    if metric_name == "sample_count":
        counts = []
        for row in range(len(row_labels)):
            finite = values[row][np.isfinite(values[row])]
            counts.append(float(finite[0]) if finite.size else np.nan)
        fig, ax = plt.subplots(figsize=fig_size(1.0, 0.5), constrained_layout=True)
        positions = np.arange(len(row_labels))
        colors = [plt.get_cmap("tab10")(idx) for idx in range(len(row_labels))]
        ax.bar(positions, counts, color=colors, alpha=0.8)
        ax.set_xticks(positions, row_labels, rotation=25, ha="right")
        ax.set_xlabel(f"{ACTIVITY_DRIVER_LABELS[driver_col]} activity bin")
        ax.set_ylabel(metric_label)
        ax.grid(True, axis="y", alpha=0.3)
        save_figure(fig, *parts)
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
        title=f"{ACTIVITY_DRIVER_LABELS[driver_col]} $\\sigma$ bin",
        fontsize=7,
        title_fontsize=7,
        loc="upper left",
        bbox_to_anchor=(-0.13, -0.22),
        ncol=5,
        borderaxespad=0,
    )
    ax.add_artist(data_legend)
    add_record_length_legend(ax, observed_duration_bins)
    save_figure(fig, *parts)


def plot_density_co2_correlation_heatmaps(df: pl.DataFrame) -> None:
    for model_name in MODEL_VERSIONS:
        altitudes = available_density_altitudes(df, model_name)
        if not altitudes:
            continue

        for driver_col in ACTIVITY_DRIVERS:
            driver_values = (
                df.select(driver_col).drop_nulls()[driver_col].to_numpy().astype(float)
            )
            driver_values = driver_values[np.isfinite(driver_values)]
            output_name = safe_name(driver_col).replace("_", "-")
            output_parts = (
                "correlation",
                "density_co2_correlation_heatmaps",
                f"{safe_name(model_name)}_msis_density_baseline_co2_correlation_by_{output_name}.png",
            )
            if len(driver_values) < MIN_SAMPLES_PER_HEATMAP_CELL:
                unlink_figure(*output_parts)
                cleanup_metric_plots_for_driver(model_name, driver_col)
                continue

            edges, driver_mean, driver_std = sigma_edges(driver_values)
            if not np.isfinite(driver_std) or driver_std == 0 or len(edges) < 2:
                unlink_figure(*output_parts)
                cleanup_metric_plots_for_driver(model_name, driver_col)
                continue

            work = df.with_columns(
                ((pl.col(driver_col) - driver_mean) / driver_std).alias("driver_sigma")
            )
            correlations = np.full((len(edges) - 1, len(altitudes)), np.nan)
            corr_los = np.full_like(correlations, np.nan)
            corr_his = np.full_like(correlations, np.nan)
            slopes = np.full_like(correlations, np.nan)
            slope_los = np.full_like(correlations, np.nan)
            slope_his = np.full_like(correlations, np.nan)
            errors = np.full_like(correlations, np.nan)
            counts = np.zeros_like(correlations, dtype=int)
            duration_bins = np.full_like(correlations, np.nan)

            for row in range(len(edges) - 1):
                low = edges[row]
                high = edges[row + 1]
                upper_filter = (
                    pl.col("driver_sigma") <= high
                    if row == len(edges) - 2
                    else pl.col("driver_sigma") < high
                )
                bin_df = work.filter((pl.col("driver_sigma") >= low) & upper_filter)
                for col_idx, altitude in enumerate(altitudes):
                    y_col = density_col(model_name, altitude)
                    co2, density = finite_xy(bin_df, "CO2_ppm", y_col)
                    counts[row, col_idx] = len(co2)
                    (
                        correlations[row, col_idx],
                        slopes[row, col_idx],
                        errors[row, col_idx],
                        slope_los[row, col_idx],
                        slope_his[row, col_idx],
                        _count,
                    ) = linear_fit_stats(co2, density)
                    r, r_lo, r_hi, _ = pearsonr_ci(co2, density)
                    corr_los[row, col_idx] = r_lo
                    corr_his[row, col_idx] = r_hi
                    valid_dates = (
                        bin_df.filter(
                            pl.col("CO2_ppm").is_not_null()
                            & pl.col(y_col).is_not_null()
                            & pl.col("CO2_ppm").is_finite()
                            & pl.col(y_col).is_finite()
                        )
                        .select("date")
                        .drop_nulls()
                        .sort("date")
                    )
                    if valid_dates.height > 0:
                        duration_years = (
                            valid_dates["date"].max() - valid_dates["date"].min()
                        ).days / 365.2425
                        duration_bins[row, col_idx] = correlation_duration_bin(
                            duration_years
                        )

            painted_rows = np.any(np.isfinite(correlations), axis=1)
            if not np.any(painted_rows):
                unlink_figure(*output_parts)
                cleanup_metric_plots_for_driver(model_name, driver_col)
                continue

            row_labels = np.array(sigma_bin_labels(edges))[painted_rows]
            correlations = correlations[painted_rows]
            corr_los = corr_los[painted_rows]
            corr_his = corr_his[painted_rows]
            slopes = slopes[painted_rows]
            slope_los = slope_los[painted_rows]
            slope_his = slope_his[painted_rows]
            errors = errors[painted_rows]
            counts = counts[painted_rows]
            duration_bins = duration_bins[painted_rows]

            plot_metric_by_altitude_and_sigma(
                model_name,
                driver_col,
                altitudes,
                row_labels,
                "correlation",
                r"Pearson r($\bar{\ell}_{\rho_m}$, CO$_2$)",
                correlations,
                duration_bins,
                ci_values=np.stack([corr_los, corr_his], axis=-1),
            )
            plot_metric_by_altitude_and_sigma(
                model_name,
                driver_col,
                altitudes,
                row_labels,
                "slope",
                r"Linear fit slope m ($\bar{\ell}_{\rho_m}$ per CO$_2$ ppm)",
                slopes,
                duration_bins,
                ci_values=np.stack([slope_los, slope_his], axis=-1),
            )
            plot_metric_by_altitude_and_sigma(
                model_name,
                driver_col,
                altitudes,
                row_labels,
                "error",
                r"Linear fit RMSE err ($\bar{\ell}_{\rho_m}$)",
                errors,
                duration_bins,
            )
            plot_metric_by_altitude_and_sigma(
                model_name,
                driver_col,
                altitudes,
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
                unlink_figure(*output_parts)
                continue

            row_labels = row_labels[painted_rows]
            display_correlations = display_correlations[painted_rows]
            slopes = slopes[painted_rows]
            slope_los = slope_los[painted_rows]
            slope_his = slope_his[painted_rows]
            errors = errors[painted_rows]
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
                np.arange(len(altitudes)),
                [f"{altitude}" for altitude in altitudes],
                rotation=45,
                ha="right",
            )
            ax.set_yticks(np.arange(len(row_labels)), row_labels)
            ax.set_xlabel("Altitude (km)")
            ax.set_ylabel(
                f"{ACTIVITY_DRIVER_LABELS[driver_col]} bins\n"
                f"mean={driver_mean:.2f}, sigma={driver_std:.2f}"
            )
            ax.set_xticks(np.arange(-0.5, len(altitudes), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=0.8)
            ax.tick_params(which="minor", bottom=False, left=False)
            ax.set_title(model_name)
            for row in range(display_correlations.shape[0]):
                for col_idx in range(display_correlations.shape[1]):
                    value = display_correlations[row, col_idx]
                    if np.isfinite(value):
                        ax.text(
                            col_idx,
                            row,
                            fit_annotation(
                                value,
                                slopes[row, col_idx],
                                errors[row, col_idx],
                                counts[row, col_idx],
                                slope_los[row, col_idx],
                                slope_his[row, col_idx],
                            ),
                            ha="center",
                            va="center",
                            fontsize=6,
                            color=text_color_for_correlation(value),
                        )
                    elif counts[row, col_idx] > 0:
                        ax.text(
                            col_idx,
                            row,
                            f"n={counts[row, col_idx]}",
                            ha="center",
                            va="center",
                            fontsize=7,
                            color="dimgray",
                        )
            colorbar_shrink = min(
                1.0, HEATMAP_COLORBAR_HEIGHT_INCHES / fig.get_figheight()
            )
            fig.colorbar(
                heatmap,
                ax=ax,
                label=r"Pearson r($\bar{\ell}_{\rho_m}$, CO$_2$)",
                shrink=colorbar_shrink,
            )
            save_figure(fig, *output_parts)


def load_daily_baselines() -> pl.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    daily_path = DATA_DIR / "maunaloa_msis_density_baselines_daily_wide.parquet"
    if not SOURCE_SAMPLE_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SOURCE_SAMPLE_PATH}. Run scripts/hasdm_msis_model_error_analysis.py first."
        )
    samples = pl.read_parquet(SOURCE_SAMPLE_PATH).with_columns(
        pl.col("timestamp").dt.date().alias("date"),
        (pl.col(HASDM_ALT_COL) / 1000.0).cast(pl.Int64).alias("altitude_km"),
    )
    model_exprs = []
    for model_name, error_col in MODEL_ERROR_COLS.items():
        model_exprs.append(
            (pl.col(HASDM_DENSITY_COL) * pl.col(error_col).exp())
            .log10()
            .alias(f"{safe_name(model_name)}_log10rho")
        )
    long = samples.with_columns(model_exprs)
    value_cols = [f"{safe_name(model)}_log10rho" for model in MODEL_VERSIONS]
    daily = (
        long.group_by("date", "altitude_km")
        .agg([pl.col(col).mean().alias(f"{col}_daily_mean") for col in value_cols])
        .sort("date", "altitude_km")
    )
    wide: pl.DataFrame | None = None
    for value_col in [
        col for col in daily.columns if col not in {"date", "altitude_km"}
    ]:
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
        raise RuntimeError("No MSIS density baseline data to pivot.")
    drivers = load_space_weather().join(load_co2(), on="date", how="left")
    wide = drivers.join(wide.sort("date"), on="date", how="inner").sort("date")
    wide.write_parquet(daily_path, compression="lz4")
    wide.write_csv(DATA_DIR / "maunaloa_msis_density_baselines_daily_wide.csv")
    return wide


def plot_time_heatmap(ax: plt.Axes, df: pl.DataFrame, model_name: str) -> None:
    altitudes = available_density_altitudes(df, model_name)
    cols = [
        density_col(model_name, altitude)
        for altitude in altitudes
        if density_col(model_name, altitude) in df.columns
    ]
    if not cols:
        ax.text(
            0.5,
            0.5,
            "No density data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return
    dates = df["date"].to_numpy()
    alt_edges = altitude_bin_edges(altitudes)
    matrix = np.full((len(alt_edges) - 1, len(dates)), np.nan)
    for altitude in altitudes:
        col = density_col(model_name, altitude)
        if col not in df.columns:
            continue
        row = np.searchsorted(alt_edges, altitude, side="right") - 1
        if 0 <= row < matrix.shape[0]:
            matrix[row, :] = df[col].to_numpy().astype(float)
    mesh = ax.pcolormesh(
        date_edges_for_heatmap(dates),
        alt_edges,
        matrix,
        shading="auto",
        cmap="viridis",
        rasterized=True,
    )
    if matrix.shape[0] > 1 and matrix.shape[1] > 1 and np.any(np.isfinite(matrix)):
        date_centers = mdates.date2num(dates)
        altitude_centers = 0.5 * (alt_edges[:-1] + alt_edges[1:])
        levels = [-10, -11, -12, -13, -14][::-1]
        if np.nanmin(matrix) <= max(levels) and np.nanmax(matrix) >= min(levels):
            contours = ax.contour(
                date_centers,
                altitude_centers,
                np.ma.masked_invalid(matrix),
                levels=levels,
                colors="white",
                linewidths=0.7,
                alpha=0.75,
            )
            ax.clabel(contours, inline=True, fontsize=7, fmt="%.2f")
    ax.xaxis_date()
    ax.set_ylabel(f"{model_name}\naltitude (km)")
    format_altitude_axis(ax)
    ax.grid(True, alpha=0.18)
    ax.figure.colorbar(mesh, ax=ax, label=r"$\bar{\ell}_{\rho_m}$", pad=0.01)


def plot_time_series(df: pl.DataFrame) -> None:
    fig, axes = plt.subplots(
        len(MODEL_VERSIONS) + 3,
        1,
        figsize=page_fig_size(1.0, 1.08, max_height_scale=0.88),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1, 1, 1, 0.75, 0.65, 0.75]},
    )
    for idx, model_name in enumerate(MODEL_VERSIONS):
        plot_time_heatmap(axes[idx], df, model_name)
    axes[-3].plot(
        df["date"],
        df["F10.7_OBS_CENTER81"],
        color="darkred",
        linewidth=0.9,
    )
    axes[-3].set_ylabel("F10.7")
    axes[-3].grid(True, alpha=0.25)
    axes[-2].plot(df["date"], df["AP_AVG"], color="purple", linewidth=0.8)
    axes[-2].set_ylabel("Ap")
    axes[-2].grid(True, alpha=0.25)
    axes[-1].plot(
        df["date"],
        df["CO2_ppm"],
        color="darkgreen",
        linewidth=0.9,
    )
    axes[-1].set_ylabel("CO$_2$ (ppm)")
    axes[-1].grid(True, alpha=0.25)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("Mauna Loa MSIS density baselines")
    save_figure(fig, "fft_timeseries", "maunaloa_msis_density_baseline_timeseries.pgf")


def plot_density_fft_heatmap(ax: plt.Axes, df: pl.DataFrame, model_name: str) -> None:
    spectra = []
    all_periods = []
    for altitude in available_density_altitudes(df, model_name):
        col = density_col(model_name, altitude)
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
    ax.set_ylabel("altitude (km)")
    format_altitude_axis(ax)
    ax.grid(True, which="both", alpha=0.18)
    ax.figure.colorbar(mesh, ax=ax, label="log$_{10}$ FFT power")


def plot_fft(df: pl.DataFrame) -> None:
    fig, axes = plt.subplots(
        len(MODEL_VERSIONS) * 2 + 3,
        1,
        figsize=page_fig_size(1.0, 2.25, max_height_scale=0.98),
        sharex=True,
        constrained_layout=True,
    )
    ref_periods = np.array([7, 27, 183, 365.25, 365.25 * 11])
    for model_idx, model_name in enumerate(MODEL_VERSIONS):
        ax = axes[model_idx * 2]
        for altitude in available_density_altitudes(df, model_name):
            col = density_col(model_name, altitude)
            if col in df.columns:
                periods, amplitudes = fft_period_power(df, col)
                if len(periods):
                    ax.plot(
                        periods / 365.25,
                        amplitudes,
                        linewidth=0.9,
                        label=f"{altitude} km",
                    )
        for period in ref_periods:
            ax.axvline(period / 365.25, color="gray", linewidth=0.7, alpha=0.35)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel(r"$\bar{\ell}_{\rho_m}$")
        ax.text(
            0.01,
            0.92,
            model_name,
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.0},
        )
        ax.grid(True, which="both", alpha=0.25)
        heatmap_ax = axes[model_idx * 2 + 1]
        plot_density_fft_heatmap(heatmap_ax, df, model_name)
        for period in ref_periods:
            heatmap_ax.axvline(period / 365.25, color="gray", linewidth=0.7, alpha=0.35)
    for ax, specs, ylabel in [
        (
            axes[-3],
            [("F10.7_OBS_CENTER81", "F10.7 81d avg", "darkred")],
            "F10.7\n(norm.)",
        ),
        (
            axes[-2],
            [("AP_AVG", "Ap", "purple")],
            "Ap\n(norm.)",
        ),
        (axes[-1], [("CO2_ppm", "Mauna Loa CO$_2$", "darkgreen")], "CO$_2$\n(norm.)"),
    ]:
        plotted = 0
        for col, label, color in specs:
            periods, amplitudes = fft_period_power(df, col)
            if len(periods) and np.nanmax(amplitudes) > 0:
                normalized = amplitudes / np.nanmax(amplitudes)
                normalized[normalized <= 0] = np.nan
                ax.plot(
                    periods / 365.25,
                    normalized,
                    linewidth=0.9,
                    color=color,
                    label=label,
                )
                plotted += 1
        for period in ref_periods:
            ax.axvline(period / 365.25, color="gray", linewidth=0.7, alpha=0.35)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.25)
    axes[-1].set_xlabel("Period")
    axes[-1].set_xticks(ref_periods / 365.25)
    axes[-1].set_xticklabels(
        ["1 wk", "27 d", "6 mo", "1 y", "11 y"], rotation=45, ha="right"
    )
    legend_cols = min(
        6, max(1, len(available_density_altitudes(df, next(iter(MODEL_VERSIONS)))))
    )
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles,
        labels,
        ncols=legend_cols,
        fontsize=7,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        borderaxespad=0,
        frameon=True,
    )
    fig.suptitle("Mauna Loa MSIS density baseline FFT spectra")
    save_figure(fig, "fft_timeseries", "maunaloa_msis_density_baseline_fft.pgf")


def plot_correlations(
    df: pl.DataFrame,
    model_versions: dict[str, str] = MODEL_VERSIONS,
    height_fraction: float = 0.9,
    filename: str = "maunaloa_msis_density_baseline_correlation_by_altitude.pgf",
) -> None:
    fig, axes = plt.subplots(
        len(model_versions),
        1,
        figsize=fig_size(1.0, height_fraction),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(len(model_versions))
    observed_duration_bins: set[int] = set()
    for idx, model_name in enumerate(model_versions):
        ax = axes[idx]
        add_correlation_effect_size_bands(ax)
        altitudes = available_density_altitudes(df, model_name)
        for cause, cause_label in zip(ANALYSIS_COLS, ANALYSIS_LABELS):
            corr = []
            corr_los = []
            corr_his = []
            duration_bins = []
            for altitude in altitudes:
                col = density_col(model_name, altitude)
                point_corr, r_lo, r_hi, duration_bin = correlation_and_duration(
                    df, cause, col
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
                linewidth=1.2,
                label=rf"$\bar{{\ell}}_{{\rho_m}}$ vs {cause_label}",
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
        ax.set_ylabel(f"{model_name}\nPearson r")
        ax.set_ylim(-1, 1)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Altitude (km)")
    driver_legend = axes[0].legend(
        fontsize=7,
        ncols=2,
        loc="upper left",
        bbox_to_anchor=(-0.13, -0.22),
        borderaxespad=0,
    )
    axes[0].add_artist(driver_legend)
    add_record_length_legend(axes[0], observed_duration_bins)
    fig.suptitle("Mauna Loa MSIS density baseline correlations")
    save_figure(fig, "correlation", filename)


def sigma_edges(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values))
    if not np.isfinite(std) or std == 0:
        raise ValueError("Cannot build sigma bins for a constant or invalid variable.")
    z = (values - mean) / std
    edges = np.arange(np.floor(np.nanmin(z)), np.ceil(np.nanmax(z)) + 1, 1.0)
    if len(edges) < 2:
        edges = np.array([-0.5, 0.5])
    return edges, mean, std


def plot_binned_slopes(df: pl.DataFrame) -> None:
    rows = []
    for driver in ACTIVITY_DRIVERS:
        values = df[driver].to_numpy().astype(float)
        try:
            edges, mean, std = sigma_edges(values[np.isfinite(values)])
        except ValueError:
            continue
        work = df.with_columns(((pl.col(driver) - mean) / std).alias("driver_sigma"))
        for model_name in MODEL_VERSIONS:
            for altitude in available_density_altitudes(df, model_name):
                col = density_col(model_name, altitude)
                if col not in df.columns:
                    continue
                for idx in range(len(edges) - 1):
                    subset = work.filter(
                        (pl.col("driver_sigma") >= edges[idx])
                        & (pl.col("driver_sigma") < edges[idx + 1])
                    )
                    co2 = subset["CO2_ppm"].to_numpy().astype(float)
                    y = subset[col].to_numpy().astype(float)
                    mask = np.isfinite(co2) & np.isfinite(y)
                    if np.sum(mask) < 20 or np.std(co2[mask]) == 0:
                        continue
                    dates = np.asarray(subset["date"].to_list(), dtype=object)[mask]
                    duration_years = (dates[-1] - dates[0]).days / 365.2425
                    slope, slope_lo, slope_hi, _se, _int, _r, _rmse, _n = ols_slope_ci(
                        co2[mask], y[mask]
                    )
                    rows.append(
                        {
                            "driver": driver,
                            "model": model_name,
                            "altitude_km": altitude,
                            "driver_sigma_min": float(edges[idx]),
                            "driver_sigma_max": float(edges[idx + 1]),
                            "co2_slope": float(slope),
                            "co2_slope_lo": slope_lo,
                            "co2_slope_hi": slope_hi,
                            "n": int(np.sum(mask)),
                            "duration_bin_years": correlation_duration_bin(
                                duration_years
                            ),
                        }
                    )
    table = pl.DataFrame(rows)
    (OUTPUT_ROOT / "correlation").mkdir(parents=True, exist_ok=True)
    table.write_csv(
        OUTPUT_ROOT
        / "correlation"
        / "maunaloa_msis_density_baseline_co2_slope_binned_stats.csv"
    )
    if table.is_empty():
        return
    display_models = ["NRLMSIS 2.0"]
    fig, axes = plt.subplots(
        len(display_models),
        len(ACTIVITY_DRIVERS),
        figsize=fig_size(1.0, 0.58),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.12, right=0.98, top=0.86, bottom=0.34, wspace=0.08)
    axes = np.asarray(axes).reshape(len(display_models), len(ACTIVITY_DRIVERS))
    for row_idx, model_name in enumerate(display_models):
        for col_idx, driver in enumerate(ACTIVITY_DRIVERS):
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
                (line,) = ax.plot(
                    series["altitude_km"],
                    series["co2_slope"],
                    linewidth=1.0,
                    label=f"{sigma_row['driver_sigma_min']:g} to {sigma_row['driver_sigma_max']:g} sigma",
                    zorder=3,
                )
                if (
                    "co2_slope_lo" in series.columns
                    and "co2_slope_hi" in series.columns
                ):
                    lo_vals = series["co2_slope_lo"].to_numpy().astype(float)
                    hi_vals = series["co2_slope_hi"].to_numpy().astype(float)
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
            ax.set_title(f"{model_name} by {ACTIVITY_DRIVER_LABELS[driver]}")
            ax.grid(True, alpha=0.25)
            if col_idx == 0:
                ax.set_ylabel(r"CO$_2$ fitted slope in $\bar{\ell}_{\rho_m}$ per ppm")
            if row_idx == len(display_models) - 1:
                ax.set_xlabel("Altitude (km)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            title="Activity bin",
            fontsize=7,
            title_fontsize=8,
            ncols=3,
            loc="lower left",
            bbox_to_anchor=(0.12, 0.04),
            borderaxespad=0,
            handlelength=1.7,
            columnspacing=0.9,
        )
    display_table = table.filter(pl.col("model").is_in(display_models))
    duration_bins = (
        {
            int(value)
            for value in display_table["duration_bin_years"].drop_nulls().to_list()
            if np.isfinite(value)
        }
        if display_table.height
        else set()
    )
    record_handles = [
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
    if record_handles:
        fig.legend(
            record_handles,
            [handle.get_label() for handle in record_handles],
            title="Record length",
            fontsize=7,
            title_fontsize=8,
            ncols=min(2, len(record_handles)),
            loc="lower right",
            bbox_to_anchor=(0.98, 0.04),
            borderaxespad=0,
        )
    save_figure(
        fig, "correlation", "maunaloa_msis_density_baseline_co2_slope_binned.pgf"
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
    stem = path.stem
    if stem == "maunaloa_msis_density_baseline_timeseries":
        return (
            r"Mauna Loa MSIS density baselines and analysis drivers. "
            r"Density panels show daily mean model log-density "
            r"\(\bar{\ell}_{\rho_m}\) by altitude for NRLMSISE-00, "
            r"NRLMSIS 2.0, and NRLMSIS 2.1."
        )
    if stem == "maunaloa_msis_density_baseline_fft":
        return (
            r"Frequency-domain comparison for the Mauna Loa MSIS density baselines, "
            r"with altitude-resolved model-density spectra and normalized driver spectra."
        )
    if stem == "maunaloa_msis_density_baseline_correlation_by_altitude":
        return (
            r"Pearson correlations between Mauna Loa MSIS density baselines and "
            r"solar, geomagnetic, and \coo{} drivers as a function of altitude."
        )
    if stem == "maunaloa_msis_density_baseline_co2_slope_binned":
        return (
            r"COO-related fitted slopes for Mauna Loa MSIS density baselines grouped "
            r"by activity-sigma bins."
        ).replace("COO", r"\coo{}")
    if "_msis_density_baseline_co2_correlation_by_" in stem:
        return (
            r"Mauna Loa MSIS density baseline versus \coo{} correlation by "
            r"activity-sigma bin. Cells show Pearson \(r\), linear-fit slope "
            r"\(m\) for \(\bar{\ell}_{\rho_m}\) per \coo{} ppm, RMSE err, "
            r"and sample count \(n\)."
        )
    metric_separator = "_msis_density_baseline_co2_"
    if metric_separator in stem and "_by_altitude_for_" in stem:
        metric_key = stem.split(metric_separator, maxsplit=1)[1].split(
            "_by_altitude_for_", maxsplit=1
        )[0]
        metric_labels = {
            "correlation": r"Pearson correlation \(r\)",
            "slope": r"linear-fit slope \(m\) for \(\bar{\ell}_{\rho_m}\) per \coo{} ppm",
            "error": "linear-fit RMSE err",
            "sample_count": r"sample count \(n\)",
        }
        return (
            f"{metric_labels.get(metric_key, latex_escape(metric_key))} versus altitude "
            r"for Mauna Loa MSIS density baseline and \coo{} fits grouped by "
            r"activity-sigma bin."
        )
    return latex_escape(stem.replace("_", " ").title())


def write_latex_index() -> None:
    root = OUTPUT_ROOT
    for directory in sorted(
        [root, *[path for path in root.rglob("*") if path.is_dir()]],
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        figures = sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in FIGURE_EXTENSIONS
        )
        child_indexes = sorted(
            path / "figures.tex"
            for path in directory.iterdir()
            if path.is_dir() and (path / "figures.tex").exists()
        )
        if not figures and not child_indexes:
            continue
        title = (
            "Mauna Loa MSIS Density Baseline Figures"
            if directory == root
            else str(directory.relative_to(root)).replace("_", " ").title()
        )
        lines = [f"\\subsection*{{{title}}}", ""]
        for figure in figures:
            rel_path = figure.relative_to("outputs").as_posix()
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
            rel_path = child_index.relative_to("outputs").as_posix()
            lines.extend([rf"\input{{\detokenize{{{rel_path}}}}}", ""])
        (directory / "figures.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    df = load_daily_baselines()
    plot_time_series(df)
    plot_fft(df)
    plot_correlations(df)
    plot_correlations(
        df,
        model_versions={"NRLMSIS 2.0": MODEL_VERSIONS["NRLMSIS 2.0"]},
        height_fraction=0.72,
        filename="maunaloa_msis_density_baseline_correlation_by_altitude_nrlmsis_2p0.png",
    )
    plot_density_co2_correlation_heatmaps(df)
    plot_binned_slopes(df)
    write_latex_index()
    print(f"Generated Mauna Loa MSIS density baseline outputs in {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
