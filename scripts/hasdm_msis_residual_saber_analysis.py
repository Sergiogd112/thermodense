from __future__ import annotations
# Ruff: configure_pgf() must run before pyplot imports; suppress intentional E402.
# ruff: noqa: E402

from dataclasses import dataclass
from pathlib import Path

from scripts.pgf_config import configure_pgf

configure_pgf()

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from scripts.hasdm_msis_model_error_analysis import (  # noqa: E402
    ANALYSIS_COLS,
    MODEL_VERSIONS,
    SABER_COLS,
    SELECTED_ALTITUDES,
    load_co2,
    load_saber,
    load_space_weather,
    safe_name,
)

DAILY_WIDE_PATH = Path(
    "outputs/figures/results/hasdm_msis_model_errors/data/"
    "hasdm_msis_errors_nearest_timestamp_grid_daily_wide.parquet"
)

OUTPUT_ROOT = Path(
    "outputs/figures/results/hasdm_msis_model_errors/model_validations/"
    "causal_hasdm_saber_msis_residuals"
)
DATA_DIR = Path("data")
DRIVER_COLS = ANALYSIS_COLS
Y_COLS = ["F10.7_OBS_CENTER81", "AP_AVG", "KP_SUM"]
X_COL = "CO2_ppm"
MIN_SAMPLES_PER_CELL = 5
CO2_DAILY_MEASUREMENT_ONLY_1SIGMA_PPM = "0.14-0.16"
CO2_DAILY_REPRESENTATIVENESS_1SIGMA_PPM = 0.38
CO2_DAILY_INCLUSIVE_1SIGMA_PPM = 0.41
NOAA_GML_WEEKLY_URL = "https://gml.noaa.gov/ccgg/trends/weekly.html"
CELESTRAK_SPACEWX_URL = "https://celestrak.org/SpaceData/SpaceWx-format.php"
NRCAN_F107_METHOD_URL = (
    "https://www.spaceweather.gc.ca/forecast-prevision/solar-solaire/"
    "solarflux/sx-3-en.php"
)
NRCAN_F107_DAILY_URL = (
    "https://spaceweather.gc.ca/forecast-prevision/solar-solaire/"
    "solarflux/sx-5-flux-en.php"
)
NRCAN_F107_TABLE_URL = (
    "https://spaceweather.gc.ca/solar_flux_data/daily_flux_values/fluxtable.txt"
)
GFZ_KP_DATA_URL = "https://kp.gfz.de/en/data"
GFZ_KP_METHOD_URL = "https://kp.gfz.de/en/about-kp"
GFZ_KP_ARCHIVE_URL = (
    "https://www-app3.gfz-potsdam.de/kp_index/Kp_ap_Ap_SN_F107_since_1932.txt"
)
TAPPING_F107_DOI = "https://doi.org/10.1002/swe.20064"
MATZKA_KP_DOI = "https://doi.org/10.1029/2020SW002641"
STORZ_HASDM_DOI = "https://doi.org/10.1016/j.asr.2004.02.020"

SABER_LABELS = {
    "saber_co2cool_min_alt": "SABER CO2 cooling min altitude",
    "saber_co2cool_median_alt": "SABER CO2 cooling median altitude",
    "saber_co2cool_max_alt": "SABER CO2 cooling max altitude",
}


@dataclass(frozen=True)
class Variant:
    name: str
    dates: np.ndarray
    data: dict[str, np.ndarray]
    description: str


def output_path(*parts: str | Path) -> Path:
    return OUTPUT_ROOT.joinpath(*parts)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, *parts: str) -> None:
    out = output_path(*parts)
    ensure_parent(out)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def write_csv(df: pl.DataFrame, *parts: str) -> None:
    out = output_path(*parts)
    ensure_parent(out)
    df.write_csv(out)


def write_parquet(df: pl.DataFrame, *parts: str) -> None:
    out = output_path(*parts)
    ensure_parent(out)
    df.write_parquet(out, compression="lz4")


def write_text(text: str, *parts: str) -> None:
    out = output_path(*parts)
    ensure_parent(out)
    out.write_text(text, encoding="utf-8")


def source_daily_wide_path() -> Path:
    if DAILY_WIDE_PATH.exists():
        return DAILY_WIDE_PATH
    raise FileNotFoundError(
        f"Missing HASDM MSIS residual daily-wide cache: {DAILY_WIDE_PATH}."
    )


def residual_col(model_name: str, altitude: int) -> str:
    return f"{safe_name(model_name)}_daily_mean_{altitude}km"


def residual_cols_for_model(model_name: str) -> list[str]:
    return [residual_col(model_name, altitude) for altitude in SELECTED_ALTITUDES]


def all_residual_cols() -> list[str]:
    return [
        residual_col(model_name, altitude)
        for model_name in MODEL_VERSIONS
        for altitude in SELECTED_ALTITUDES
    ]


def safe_file(text: str) -> str:
    return (
        text.lower()
        .replace(" ", "_")
        .replace(".", "p")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


def label_for_col(col: str) -> str:
    for model_name in MODEL_VERSIONS:
        prefix = f"{safe_name(model_name)}_daily_mean_"
        if col.startswith(prefix) and col.endswith("km"):
            altitude = col[len(prefix) : -2]
            return f"{model_name} mean error {altitude} km"
    if col in SABER_LABELS:
        return SABER_LABELS[col]
    if col == "F10.7_OBS_CENTER81":
        return "F10.7 81d avg"
    if col == "AP_AVG":
        return "Ap"
    if col == "KP_SUM":
        return "Kp"
    if col == "CO2_ppm":
        return "CO2"
    return col


def load_daily_residuals() -> pl.DataFrame:
    path = source_daily_wide_path()
    columns = ["date", *all_residual_cols()]
    missing = [
        col
        for col in columns
        if col not in pl.scan_parquet(path).collect_schema().names()
    ]
    if missing:
        raise ValueError(f"Residual daily-wide cache is missing columns: {missing}")
    return pl.read_parquet(path).select(columns)


def as_date_index(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(pl.col("date").cast(pl.Date))
        .filter(pl.col("date").is_not_null())
        .unique(subset="date")
        .sort("date")
    )


def combine_inputs() -> tuple[pl.DataFrame, pl.DataFrame]:
    residuals = as_date_index(load_daily_residuals())
    drivers = as_date_index(load_space_weather()).join(
        as_date_index(load_co2()), on="date", how="full", coalesce=True
    )
    saber = as_date_index(load_saber())

    all_cols = [*DRIVER_COLS, *all_residual_cols(), *SABER_COLS]
    target_valid = residuals.drop_nulls(all_residual_cols())
    saber_valid = saber.drop_nulls(SABER_COLS)

    start_date = max(
        target_valid["date"].min(),
        saber_valid["date"].min(),
        drivers.filter(pl.col("F10.7_OBS_CENTER81").is_not_null())["date"].min(),
        drivers.filter(pl.col("CO2_ppm").is_not_null())["date"].min(),
    )
    end_date = min(
        target_valid["date"].max(),
        saber_valid["date"].max(),
        drivers.filter(pl.col("F10.7_OBS_CENTER81").is_not_null())["date"].max(),
        drivers.filter(pl.col("CO2_ppm").is_not_null())["date"].max(),
    )

    combined = residuals.join(drivers, on="date", how="full", coalesce=True)
    combined = combined.join(saber, on="date", how="full", coalesce=True)
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
    return interpolated, missing_summary


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
    cols = [*DRIVER_COLS, *all_residual_cols(), *SABER_COLS]
    dates = df["date"].to_numpy()
    raw = {col: df[col].to_numpy().astype(float) for col in cols}
    seasonal = {col: seasonal_anomaly(raw[col], dates) for col in cols}
    detrended_seasonal = {col: detrended(seasonal[col]) for col in cols}
    co2_preserved = {
        col: seasonal[col] if col != "CO2_ppm" else raw[col] for col in cols
    }
    return {
        "raw_standardized": Variant(
            "raw_standardized",
            dates,
            {col: finite_standardize(raw[col]) for col in cols},
            "Raw daily values after temporal gap filling, standardized.",
        ),
        "seasonal_anomaly": Variant(
            "seasonal_anomaly",
            dates,
            {col: finite_standardize(seasonal[col]) for col in cols},
            "Day-of-year climatology removed, then standardized.",
        ),
        "detrended_anomaly": Variant(
            "detrended_anomaly",
            dates,
            {col: finite_standardize(detrended_seasonal[col]) for col in cols},
            "Seasonal anomalies with a 3-year rolling mean removed.",
        ),
        "co2_preserved_anomaly": Variant(
            "co2_preserved_anomaly",
            dates,
            {col: finite_standardize(co2_preserved[col]) for col in cols},
            "Seasonal anomalies, but CO2 kept as a slow standardized driver.",
        ),
    }


def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 3:
        return np.nan
    x0 = x[mask] - np.mean(x[mask])
    y0 = y[mask] - np.mean(y[mask])
    denom = np.sqrt(np.sum(x0**2) * np.sum(y0**2))
    return float(np.sum(x0 * y0) / denom) if denom else np.nan


def finite_xy(
    df: pl.DataFrame, x_col: str, y_col: str
) -> tuple[np.ndarray, np.ndarray]:
    data = df.select(x_col, y_col).drop_nulls()
    x = data[x_col].to_numpy().astype(float)
    y = data[y_col].to_numpy().astype(float)
    finite = np.isfinite(x) & np.isfinite(y)
    return x[finite], y[finite]


def draw_scatter(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    ax.scatter(x, y, s=5, alpha=0.22, rasterized=True)
    if len(x) >= 3 and np.nanstd(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 80)
        ax.plot(x_line, slope * x_line + intercept, color="black", linewidth=0.75)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.2)


def plot_residual_scatter_grid(df: pl.DataFrame) -> None:
    x_cols = [*DRIVER_COLS, *SABER_COLS]
    n_rows = len(MODEL_VERSIONS) * len(SELECTED_ALTITUDES)
    fig, axes = plt.subplots(
        n_rows,
        len(x_cols),
        figsize=(3.0 * len(x_cols), 1.85 * n_rows),
        sharex="col",
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    row = 0
    for model_name in MODEL_VERSIONS:
        for altitude in SELECTED_ALTITUDES:
            target_col = residual_col(model_name, altitude)
            for col_idx, x_col in enumerate(x_cols):
                ax = axes[row, col_idx]
                x, y = finite_xy(df, x_col, target_col)
                draw_scatter(ax, x, y)
                if row == 0:
                    ax.set_title(label_for_col(x_col), fontsize=8)
                if col_idx == 0:
                    ax.set_ylabel(f"{model_name}\n{altitude} km", fontsize=7)
            row += 1
    fig.suptitle("Daily mean HASDM MSIS residual scatter by selected altitude")
    save_figure(fig, "scatter_residual_mean_selected_altitudes.pgf")


def plot_saber_scatter_grid(df: pl.DataFrame) -> None:
    n_cols = len(DRIVER_COLS) + len(SELECTED_ALTITUDES)
    n_rows = len(MODEL_VERSIONS) * len(SABER_COLS)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.0 * n_cols, 1.95 * n_rows),
        sharex=False,
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    row = 0
    for model_name in MODEL_VERSIONS:
        x_cols = [*DRIVER_COLS, *residual_cols_for_model(model_name)]
        for saber_col in SABER_COLS:
            for col_idx, x_col in enumerate(x_cols):
                ax = axes[row, col_idx]
                x, y = finite_xy(df, x_col, saber_col)
                draw_scatter(ax, x, y)
                if row == 0:
                    ax.set_title(label_for_col(x_col), fontsize=8)
                if col_idx == 0:
                    ax.set_ylabel(
                        f"{model_name}\n{label_for_col(saber_col)}", fontsize=7
                    )
            row += 1
    fig.suptitle("SABER CO2 cooling scatter against drivers and MSIS residuals")
    save_figure(fig, "scatter_saber_cooling_vs_drivers_residuals.pgf")


def correlation_matrix(variant: Variant, cols: list[str]) -> np.ndarray:
    return np.array(
        [
            [pearsonr(variant.data[row], variant.data[col]) for col in cols]
            for row in cols
        ]
    )


def plot_correlation_matrices(variants: dict[str, Variant]) -> pl.DataFrame:
    rows = []
    for variant in variants.values():
        fig, axes = plt.subplots(
            1,
            len(MODEL_VERSIONS),
            figsize=(7.4 * len(MODEL_VERSIONS), 7.4),
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes)
        for ax, model_name in zip(axes, MODEL_VERSIONS):
            cols = [*DRIVER_COLS, *residual_cols_for_model(model_name), *SABER_COLS]
            matrix = correlation_matrix(variant, cols)
            labels = [label_for_col(col) for col in cols]
            image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm", rasterized=True)
            ax.set_xticks(np.arange(len(labels)), labels, rotation=90, fontsize=6)
            ax.set_yticks(np.arange(len(labels)), labels, fontsize=6)
            ax.set_title(model_name, fontsize=10)
            for row_idx, row_col in enumerate(cols):
                for col_idx, col in enumerate(cols):
                    rows.append(
                        {
                            "variant": variant.name,
                            "model": model_name,
                            "row": row_col,
                            "col": col,
                            "correlation": float(matrix[row_idx, col_idx]),
                        }
                    )
        fig.colorbar(image, ax=axes.ravel().tolist(), label="Pearson r", shrink=0.8)
        fig.suptitle(f"MSIS residual correlation matrices: {variant.name}")
        save_figure(fig, f"correlation_{variant.name}.pgf")
    return pl.DataFrame(rows)


def plot_correlation_by_altitude(variants: dict[str, Variant]) -> pl.DataFrame:
    rows = []
    causes = [*DRIVER_COLS, *SABER_COLS]
    for variant in variants.values():
        fig, axes = plt.subplots(
            len(MODEL_VERSIONS),
            1,
            figsize=(11, 8.5),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes)
        for ax, model_name in zip(axes, MODEL_VERSIONS):
            for cause in causes:
                corr = []
                for altitude in SELECTED_ALTITUDES:
                    target_col = residual_col(model_name, altitude)
                    value = pearsonr(variant.data[target_col], variant.data[cause])
                    corr.append(value)
                    rows.append(
                        {
                            "variant": variant.name,
                            "model": model_name,
                            "altitude_km": altitude,
                            "cause": cause,
                            "correlation": value,
                        }
                    )
                ax.plot(
                    SELECTED_ALTITUDES,
                    corr,
                    marker="o",
                    linewidth=1.2,
                    label=label_for_col(cause),
                )
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_ylim(-1, 1)
            ax.set_ylabel(f"{model_name}\nPearson r")
            ax.grid(True, alpha=0.25)
        axes[-1].set_xlabel("HASDM altitude (km)")
        axes[0].legend(fontsize=8, ncols=2)
        fig.suptitle(f"Residual correlation by selected HASDM altitude: {variant.name}")
        save_figure(fig, f"correlation_by_altitude_{variant.name}.pgf")
    return pl.DataFrame(rows)


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


def trim_empty_rows(
    y_edges: np.ndarray,
    *matrices: np.ndarray,
) -> tuple[np.ndarray, *tuple[np.ndarray, ...]]:
    if not matrices:
        return (y_edges,)
    finite = np.zeros(matrices[0].shape[0], dtype=bool)
    for matrix in matrices:
        if np.issubdtype(matrix.dtype, np.floating):
            finite |= np.any(np.isfinite(matrix), axis=1)
    if not np.any(finite):
        return (y_edges, *matrices)
    row_indices = np.flatnonzero(finite)
    row_start, row_stop = int(row_indices[0]), int(row_indices[-1]) + 1
    return (
        y_edges[row_start : row_stop + 1],
        *(matrix[row_start:row_stop, :] for matrix in matrices),
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


def annotate_heatmap(
    ax: plt.Axes,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    mean_values: np.ndarray,
    std_values: np.ndarray,
    fmt: str,
) -> None:
    for y_idx in range(len(y_edges) - 1):
        for x_idx in range(len(x_edges) - 1):
            if not np.isfinite(mean_values[y_idx, x_idx]):
                continue
            ax.text(
                0.5 * (x_edges[x_idx] + x_edges[x_idx + 1]),
                0.5 * (y_edges[y_idx] + y_edges[y_idx + 1]),
                f"mean {format(mean_values[y_idx, x_idx], fmt)}\n"
                f"sd {format(std_values[y_idx, x_idx], fmt)}",
                ha="center",
                va="center",
                fontsize=6,
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.62,
                },
            )


def altitude_bin_edges(altitudes: list[int]) -> np.ndarray:
    values = np.asarray(altitudes, dtype=float)
    if len(values) == 1:
        return np.array([values[0] - 12.5, values[0] + 12.5])
    midpoints = 0.5 * (values[:-1] + values[1:])
    first_width = midpoints[0] - values[0]
    last_width = values[-1] - midpoints[-1]
    return np.concatenate(
        [[values[0] - first_width], midpoints, [values[-1] + last_width]]
    )


def sigma_bin_file_label(lower: float, upper: float) -> str:
    def part(value: float) -> str:
        prefix = "minus" if value < 0 else "plus"
        return f"{prefix}{abs(value):g}".replace(".", "p")

    return f"{part(lower)}_to_{part(upper)}"


def bin_mask(
    values: np.ndarray, lower: float, upper: float, is_last: bool
) -> np.ndarray:
    if is_last:
        return (values >= lower) & (values <= upper)
    return (values >= lower) & (values < upper)


def plot_saber_heatmaps(df: pl.DataFrame) -> None:
    for saber_col in SABER_COLS:
        stats_by_driver = {
            y_col: binned_value_stats(df, y_col, saber_col) for y_col in Y_COLS
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
            1, len(Y_COLS), figsize=(16, 5.8), constrained_layout=True
        )
        axes = np.atleast_1d(axes)
        mesh = None
        for ax, y_col in zip(axes, Y_COLS):
            x_edges, y_edges, _, mean_values, std_values, metadata = stats_by_driver[
                y_col
            ]
            x_edges, y_edges, mean_values, std_values = trim_empty_margins(
                x_edges, y_edges, mean_values, std_values
            )
            color_values = (mean_values - metadata["value_mean"]) / metadata[
                "value_std"
            ]
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
            annotate_heatmap(ax, x_edges, y_edges, mean_values, std_values, ".2e")
            ax.set_xlabel(
                f"CO2 sigma bins\nmean={metadata['x_mean']:.2f} ppm, "
                f"sigma={metadata['x_std']:.2f} ppm"
            )
            ax.set_ylabel(
                f"{label_for_col(y_col)} sigma bins\nmean={metadata['y_mean']:.2f}, "
                f"sigma={metadata['y_std']:.2f}"
            )
            set_integer_sigma_ticks(ax, x_edges, y_edges)
            ax.grid(True, color="white", alpha=0.25, linewidth=0.7)
        if mesh is not None:
            fig.colorbar(mesh, ax=axes.ravel().tolist(), shrink=0.9).set_label(
                "Mean SABER CO2 cooling rate sigma"
            )
        fig.suptitle(f"{label_for_col(saber_col)}: mean and empirical std by bin")
        save_figure(
            fig,
            f"cooling_heatmap_{safe_file(saber_col)}_combined_co2_vs_space_weather.pgf",
        )


def plot_residual_heatmaps(df: pl.DataFrame) -> None:
    for altitude in SELECTED_ALTITUDES:
        stats = {
            (model_name, y_col): binned_value_stats(
                df, y_col, residual_col(model_name, altitude)
            )
            for model_name in MODEL_VERSIONS
            for y_col in Y_COLS
        }
        finite_means = []
        for _, _, _, mean_values, _, metadata in stats.values():
            if metadata["value_std"] == 0:
                continue
            finite_means.extend(
                ((mean_values - metadata["value_mean"]) / metadata["value_std"])[
                    np.isfinite(mean_values)
                ]
            )
        max_abs = float(np.max(np.abs(finite_means))) if len(finite_means) else 1.0
        fig, axes = plt.subplots(
            len(MODEL_VERSIONS),
            len(Y_COLS),
            figsize=(16, 12),
            constrained_layout=True,
            sharex=False,
            sharey=False,
        )
        mesh = None
        for row_idx, model_name in enumerate(MODEL_VERSIONS):
            for col_idx, y_col in enumerate(Y_COLS):
                ax = axes[row_idx, col_idx]
                x_edges, y_edges, _, mean_values, std_values, metadata = stats[
                    (model_name, y_col)
                ]
                x_edges, y_edges, mean_values, std_values = trim_empty_margins(
                    x_edges, y_edges, mean_values, std_values
                )
                color_values = (mean_values - metadata["value_mean"]) / metadata[
                    "value_std"
                ]
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
                annotate_heatmap(ax, x_edges, y_edges, mean_values, std_values, ".2f")
                if row_idx == 0:
                    ax.set_title(label_for_col(y_col), fontsize=10)
                if col_idx == 0:
                    ax.set_ylabel(
                        f"{model_name}\n{label_for_col(y_col)} sigma bins",
                        fontsize=9,
                    )
                ax.set_xlabel("CO2 sigma bins")
                set_integer_sigma_ticks(ax, x_edges, y_edges)
                ax.grid(True, color="white", alpha=0.25, linewidth=0.7)
        if mesh is not None:
            fig.colorbar(mesh, ax=axes.ravel().tolist(), shrink=0.9).set_label(
                "Mean ln(model/HASDM) residual sigma"
            )
        fig.suptitle(
            "Daily mean MSIS log-density-ratio residual: "
            f"mean and empirical std by bin at {altitude} km"
        )
        save_figure(
            fig,
            f"residual_heatmap_{altitude}km_combined_co2_vs_space_weather.pgf",
        )


def altitude_driver_bin_stats(
    df: pl.DataFrame,
    model_name: str,
    driver_col: str,
    co2_z: np.ndarray,
    co2_lower: float,
    co2_upper: float,
    co2_is_last: bool,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]
]:
    driver_values = df[driver_col].to_numpy().astype(float)
    driver_edges, driver_mean, driver_std, driver_z = sigma_edges(driver_values)
    means = np.full((len(driver_edges) - 1, len(SELECTED_ALTITUDES)), np.nan)
    stds = np.full_like(means, np.nan)
    counts = np.zeros_like(means, dtype=int)
    colors = np.full_like(means, np.nan)
    rows: list[dict[str, object]] = []
    co2_mask = bin_mask(co2_z, co2_lower, co2_upper, co2_is_last)

    for x_idx, altitude in enumerate(SELECTED_ALTITUDES):
        target_col = residual_col(model_name, altitude)
        values = df[target_col].to_numpy().astype(float)
        target_mean = float(np.nanmean(values))
        target_std = float(np.nanstd(values))
        for y_idx in range(len(driver_edges) - 1):
            driver_mask = bin_mask(
                driver_z,
                float(driver_edges[y_idx]),
                float(driver_edges[y_idx + 1]),
                y_idx == len(driver_edges) - 2,
            )
            mask = co2_mask & driver_mask & np.isfinite(values)
            count = int(np.sum(mask))
            if count >= MIN_SAMPLES_PER_CELL:
                mean_value = float(np.mean(values[mask]))
                std_value = float(np.std(values[mask]))
                color_value = (
                    (mean_value - target_mean) / target_std
                    if np.isfinite(target_std) and target_std > 0
                    else np.nan
                )
                means[y_idx, x_idx] = mean_value
                stds[y_idx, x_idx] = std_value
                colors[y_idx, x_idx] = color_value
            else:
                mean_value = np.nan
                std_value = np.nan
                color_value = np.nan
            counts[y_idx, x_idx] = count
            rows.append(
                {
                    "co2_sigma_min": co2_lower,
                    "co2_sigma_max": co2_upper,
                    "model": model_name,
                    "driver": driver_col,
                    "altitude_km": altitude,
                    "driver_sigma_min": float(driver_edges[y_idx]),
                    "driver_sigma_max": float(driver_edges[y_idx + 1]),
                    "residual_mean": mean_value,
                    "residual_std": std_value,
                    "residual_mean_sigma": color_value,
                    "n": count,
                    "driver_mean": driver_mean,
                    "driver_std": driver_std,
                }
            )
    return driver_edges, means, stds, counts, colors, rows


def annotate_altitude_heatmap(
    ax: plt.Axes,
    altitudes: list[int],
    y_edges: np.ndarray,
    mean_values: np.ndarray,
    std_values: np.ndarray,
    counts: np.ndarray,
) -> None:
    for y_idx in range(len(y_edges) - 1):
        y_center = 0.5 * (y_edges[y_idx] + y_edges[y_idx + 1])
        for x_idx, altitude in enumerate(altitudes):
            if not np.isfinite(mean_values[y_idx, x_idx]):
                continue
            ax.text(
                altitude,
                y_center,
                f"mean {mean_values[y_idx, x_idx]:.2f}\n"
                f"sd {std_values[y_idx, x_idx]:.2f}\n"
                f"n={counts[y_idx, x_idx]}",
                ha="center",
                va="center",
                fontsize=6,
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.65,
                },
            )


def plot_residual_altitude_heatmaps_by_co2_bin(df: pl.DataFrame) -> None:
    co2_values = df[X_COL].to_numpy().astype(float)
    co2_edges, co2_mean, co2_std, co2_z = sigma_edges(co2_values)
    altitude_edges = altitude_bin_edges(SELECTED_ALTITUDES)
    all_rows = []

    for co2_idx in range(len(co2_edges) - 1):
        co2_lower = float(co2_edges[co2_idx])
        co2_upper = float(co2_edges[co2_idx + 1])
        co2_is_last = co2_idx == len(co2_edges) - 2
        stats = {}
        figure_rows = []
        for model_name in MODEL_VERSIONS:
            for driver_col in Y_COLS:
                result = altitude_driver_bin_stats(
                    df,
                    model_name,
                    driver_col,
                    co2_z,
                    co2_lower,
                    co2_upper,
                    co2_is_last,
                )
                stats[(model_name, driver_col)] = result[:-1]
                figure_rows.extend(result[-1])
        all_rows.extend(figure_rows)

        finite_color_parts = [
            colors[np.isfinite(colors)]
            for _, _, _, _, colors in stats.values()
            if np.any(np.isfinite(colors))
        ]
        finite_colors = (
            np.concatenate(finite_color_parts)
            if finite_color_parts
            else np.array([], dtype=float)
        )
        max_abs = float(np.max(np.abs(finite_colors))) if len(finite_colors) else 1.0
        fig, axes = plt.subplots(
            len(MODEL_VERSIONS),
            len(Y_COLS),
            figsize=(16, 12),
            constrained_layout=True,
            sharex=True,
            sharey=False,
        )
        mesh = None
        for row_idx, model_name in enumerate(MODEL_VERSIONS):
            for col_idx, driver_col in enumerate(Y_COLS):
                ax = axes[row_idx, col_idx]
                driver_edges, means, stds, counts, colors = stats[
                    (model_name, driver_col)
                ]
                driver_edges, means, stds, counts, colors = trim_empty_rows(
                    driver_edges, means, stds, counts, colors
                )
                mesh = ax.pcolormesh(
                    altitude_edges,
                    driver_edges,
                    colors,
                    shading="auto",
                    cmap="coolwarm",
                    vmin=-max_abs,
                    vmax=max_abs,
                    rasterized=True,
                )
                annotate_altitude_heatmap(
                    ax, SELECTED_ALTITUDES, driver_edges, means, stds, counts
                )
                if row_idx == 0:
                    ax.set_title(label_for_col(driver_col), fontsize=10)
                if col_idx == 0:
                    ax.set_ylabel(
                        f"{model_name}\n{label_for_col(driver_col)} sigma bins",
                        fontsize=9,
                    )
                ax.set_xticks(SELECTED_ALTITUDES)
                ax.set_xlabel("Altitude (km)")
                y_ticks = np.arange(
                    int(np.ceil(np.min(driver_edges))),
                    int(np.floor(np.max(driver_edges))) + 1,
                    1,
                )
                if len(y_ticks):
                    ax.set_yticks(y_ticks)
                ax.set_xticks(SELECTED_ALTITUDES)
                ax.grid(True, color="white", alpha=0.25, linewidth=0.7)
        if mesh is not None:
            fig.colorbar(mesh, ax=axes.ravel().tolist(), shrink=0.9).set_label(
                "Mean ln(model/HASDM) residual sigma"
            )
        fig.suptitle(
            "Daily mean MSIS log-density-ratio residual by altitude: "
            f"CO2 {co2_lower:g} to {co2_upper:g} sigma "
            f"(mean={co2_mean:.2f} ppm, sigma={co2_std:.2f} ppm)"
        )
        save_figure(
            fig,
            "residual_altitude_heatmap_"
            f"co2_sigma_{sigma_bin_file_label(co2_lower, co2_upper)}.pgf",
        )

    write_csv(
        pl.DataFrame(all_rows),
        "residual_altitude_heatmap_by_co2_sigma_bin_stats.csv",
    )


def write_lowest_residual_top_saber_correlations(variants: dict[str, Variant]) -> None:
    rows = []
    top_saber = "saber_co2cool_max_alt"
    lowest_altitude = min(SELECTED_ALTITUDES)
    for variant in variants.values():
        for model_name in MODEL_VERSIONS:
            target_col = residual_col(model_name, lowest_altitude)
            rows.append(
                {
                    "variant": variant.name,
                    "model": model_name,
                    "residual_col": target_col,
                    "saber_col": top_saber,
                    "correlation": pearsonr(
                        variant.data[target_col], variant.data[top_saber]
                    ),
                }
            )
    write_csv(
        pl.DataFrame(rows),
        "lowest_altitude_residual_vs_top_saber_cooling_correlations.csv",
    )


def write_source_uncertainty_notes() -> None:
    rows = [
        {
            "quantity": "CO2_ppm",
            "source": "NOAA/GML Mauna Loa daily CO2 input",
            "local_file": "data/original/co2/co2_daily_mlo.csv",
            "uncertainty_status": "daily input has no explicit uncertainty column; values below are external assumptions",
            "uncertainty_value": "",
            "uncertainty_units": "ppm",
            "note": "Daily means are background-selected dry-air mole fractions. The local daily CSV contains date and CO2_ppm only.",
            "source_url": NOAA_GML_WEEKLY_URL,
        },
        {
            "quantity": "CO2_ppm_measurement_only",
            "source": "NOAA/GML Mauna Loa daily CO2 input",
            "local_file": "data/original/co2/co2_daily_mlo.csv",
            "uncertainty_status": "inferred measurement-only daily uncertainty assumption",
            "uncertainty_value": CO2_DAILY_MEASUREMENT_ONLY_1SIGMA_PPM,
            "uncertainty_units": "ppm, 1-sigma",
            "note": "Use only when a measurement-only CO2 error bar is required; this does not include background representativeness scatter.",
            "source_url": NOAA_GML_WEEKLY_URL,
        },
        {
            "quantity": "CO2_ppm_representativeness",
            "source": "NOAA/GML weekly Mauna Loa trends page",
            "local_file": "data/original/co2/co2_daily_mlo.csv",
            "uncertainty_status": "author-stated day-to-day representativeness scatter",
            "uncertainty_value": f"{CO2_DAILY_REPRESENTATIVENESS_1SIGMA_PPM:.2f}",
            "uncertainty_units": "ppm, standard deviation",
            "note": "NOAA/GML states that the average standard deviation of day-to-day variability, calculated relative to the appropriate weekly mean, equals 0.38 ppm for the entire record.",
            "source_url": NOAA_GML_WEEKLY_URL,
        },
        {
            "quantity": "CO2_ppm_inclusive_daily",
            "source": "NOAA/GML Mauna Loa daily CO2 input plus representativeness term",
            "local_file": "data/original/co2/co2_daily_mlo.csv",
            "uncertainty_status": "inferred inclusive daily uncertainty assumption",
            "uncertainty_value": f"{CO2_DAILY_INCLUSIVE_1SIGMA_PPM:.2f}",
            "uncertainty_units": "ppm, 1-sigma",
            "note": "Preferred daily CO2 error bar when both measurement and background representativeness uncertainty should be visible. It remains an assumption because the daily source file does not provide propagated per-day uncertainties.",
            "source_url": NOAA_GML_WEEKLY_URL,
        },
        {
            "quantity": "F10.7_OBS_CENTER81",
            "source": "CelesTrak space-weather input assembled from NRCan/DRAO F10.7 products",
            "local_file": "data/original/space_weather/SW-All.csv; data/original/space_weather/SW-All.txt",
            "uncertainty_status": "provider accuracy scale exists, but no per-day propagated uncertainty field is present in the CelesTrak CSV",
            "uncertainty_value": "about 1; about 0.4%; recent within-day sd 1.57",
            "uncertainty_units": "sfu; single-monitor percent fluctuation; sfu empirical variability",
            "note": "Use these as source-characterization scales, not propagated error bars. The workflow uses the centered 81-day CelesTrak field.",
            "source_url": TAPPING_F107_DOI,
        },
        {
            "quantity": "F10.7_DATA_TYPE",
            "source": "CelesTrak space-weather input",
            "local_file": "data/original/space_weather/SW-All.csv; data/original/space_weather/SW-All.txt",
            "uncertainty_status": "quality/provenance flag available locally but not loaded into the analysis dataset",
            "uncertainty_value": "OBS, INT, PRD, PRM",
            "uncertainty_units": "categorical flag",
            "note": "CelesTrak exposes F10.7 provenance/quality states. The analysis loader currently selects only date, F10.7_OBS_CENTER81, AP_AVG, and KP_SUM, so these flags are documented here but not used for filtering or weighting.",
            "source_url": CELESTRAK_SPACEWX_URL,
        },
        {
            "quantity": "KP_SUM",
            "source": "CelesTrak space-weather input assembled from GFZ Kp products",
            "local_file": "data/original/space_weather/SW-All.csv; data/original/space_weather/SW-All.txt",
            "uncertainty_status": "no numeric daily error-bar field; uncertainty is communicated through preliminary/definitive status, observatory completeness, and nowcast validation",
            "uncertainty_value": "about 70% exact nowcast-definitive agreement; usually 1/3 Kp; rarely 2/3 Kp",
            "uncertainty_units": "Kp category difference",
            "note": "Use this as an operational uncertainty scale, not a fixed daily sigma. KP_SUM summarizes the eight 3-hour Kp values.",
            "source_url": MATZKA_KP_DOI,
        },
        {
            "quantity": "AP_AVG",
            "source": "CelesTrak space-weather input assembled from GFZ ap/Ap products",
            "local_file": "data/original/space_weather/SW-All.csv; data/original/space_weather/SW-All.txt",
            "uncertainty_status": "no standalone daily Ap error-bar field; Ap inherits Kp production and nonlinear Kp-to-ap conversion uncertainty",
            "uncertainty_value": "recent within-day sd 5.54; recent day-to-day difference sd 8.03",
            "uncertainty_units": "ap units empirical variability",
            "note": "Ap uncertainty is state dependent because ap is derived nonlinearly from Kp. The empirical values are variability diagnostics, not formal measurement uncertainty.",
            "source_url": MATZKA_KP_DOI,
        },
        {
            "quantity": "Kp/ap empirical recent-window variability",
            "source": "GFZ recent hybrid Kp/ap/Ap/F10.7 file as summarized from official recent-window extraction",
            "local_file": "data/original/space_weather/SW-All.csv; data/original/space_weather/SW-All.txt",
            "uncertainty_status": "empirical variability supplement, not measurement uncertainty",
            "uncertainty_value": "Kp within-day sd 0.73; Kp day-to-day difference sd 0.82; ap within-day sd 5.54; Ap day-to-day difference sd 8.03",
            "uncertainty_units": "Kp and ap/Ap units",
            "note": "Recent-window diagnostics mix geophysical variability with measurement and processing uncertainty. They should not replace GFZ preliminary/definitive status and Matzka et al. validation as the primary uncertainty characterization.",
            "source_url": "; ".join([GFZ_KP_DATA_URL, GFZ_KP_ARCHIVE_URL]),
        },
        {
            "quantity": ", ".join(SABER_COLS),
            "source": "SABER CO2 cooling input",
            "local_file": "data/decoded/saber/saber_co2_cooling_maunaloa_daily.parquet; data/original/saber/co2_cooling_profiles/*.nc",
            "uncertainty_status": "no numeric uncertainty variable found in inspected local files",
            "uncertainty_value": "",
            "uncertainty_units": "",
            "note": "Heatmap sd annotations for SABER quantities are empirical within-bin scatter, not author-provided uncertainty.",
            "source_url": "",
        },
        {
            "quantity": "HASDM total neutral mass density",
            "source": "SET HASDM public database and official HASDM validation literature",
            "local_file": "data/decoded/hasdm/HASDM_*_merged.parquet",
            "uncertainty_status": "assimilative density nowcast product; public local files do not include per-grid-cell uncertainty or covariance fields",
            "uncertainty_value": "2-10 for a given epoch; less than 4 overall validation error; about 2 during latest solar maximum; about 4 calibration and 6-8 evaluation in early demonstration",
            "uncertainty_units": "percent density error, 1-sigma or validation residual as reported by source context",
            "note": "HASDM should be treated as an assimilative neutral mass density reference, not direct truth. Official sources characterize uncertainty with empirical residual and validation metrics rather than released local-time-resolved uncertainty grids.",
            "source_url": STORZ_HASDM_DOI,
        },
        {
            "quantity": "HASDM altitude/activity engineering prior",
            "source": "SET HASDM uncertainty summaries and validation literature",
            "local_file": "data/decoded/hasdm/HASDM_*_merged.parquet",
            "uncertainty_status": "engineering prior only; not a machine-readable uncertainty field in the local HASDM data",
            "uncertainty_value": "3-5 moderate/high solar activity at 200-800 km; 5-10 solar minimum through most of public grid; larger near 700-825 km and storm recovery",
            "uncertainty_units": "percent density, approximate 1-sigma multiplicative prior",
            "note": "Use this as a cautious reference-density uncertainty scale when interpreting ln(rho_model/rho_HASDM). Since ln(1+p) is the equivalent log-ratio scale, 5% density uncertainty is about 0.049 in natural-log residual units and 10% is about 0.095.",
            "source_url": STORZ_HASDM_DOI,
        },
        {
            "quantity": "HASDM external validation spread",
            "source": "Independent GOCE, CHAMP, and GRACE validation studies summarized in HASDM uncertainty review",
            "local_file": "data/decoded/hasdm/HASDM_*_merged.parquet",
            "uncertainty_status": "external model-measurement spread; not an internal HASDM per-sample sigma",
            "uncertainty_value": "GOCE daily averages about 3 after scale factor; CHAMP/GRACE orbit-averaged differences often about 16-31",
            "uncertainty_units": "percent density difference",
            "note": "External comparisons can be materially larger than internal HASDM uncertainty because they include independent-density retrieval uncertainty, aerodynamic scale differences, storm recovery behavior, and sampling differences. Treat HASDM as best-available operational prior rather than zero-error truth.",
            "source_url": STORZ_HASDM_DOI,
        },
        {
            "quantity": "model log-density-ratio error",
            "source": "Mauna Loa HASDM subset and Mauna Loa MSIS density baselines",
            "local_file": source_daily_wide_path().as_posix(),
            "uncertainty_status": "no propagated HASDM/MSIS residual uncertainty attached to daily residual cache; HASDM reference-density uncertainty is documented in separate rows",
            "uncertainty_value": "",
            "uncertainty_units": "",
            "note": "Residual heatmap sd annotations are empirical within-bin scatter of ln(rho_model/rho_HASDM), not author-provided error bars. HASDM percent uncertainty affects the interpretation of zero-relative-to-reference model error but is not propagated into plotted cells.",
            "source_url": STORZ_HASDM_DOI,
        },
    ]
    write_csv(pl.DataFrame(rows), "source_uncertainty_notes.csv")


def write_source_uncertainty_details() -> None:
    details = f"""# Source Uncertainty Details

This text expands the summary rows in `source_uncertainty_notes.csv`. The CSV is intentionally short: each note cell is a summary, while this file carries the interpretation caveats and source context.

## Mauna Loa CO2

The daily Mauna Loa CO2 input file used by this workflow contains date and `CO2_ppm`, but no propagated per-day uncertainty column. NOAA/GML describes the daily means as background-selected dry-air mole fractions and reports on the weekly trends page that the average standard deviation of day-to-day variability relative to the appropriate weekly mean is {CO2_DAILY_REPRESENTATIVENESS_1SIGMA_PPM:.2f} ppm for the full record.

For plotting or reporting an uncertainty scale, use {CO2_DAILY_INCLUSIVE_1SIGMA_PPM:.2f} ppm (1-sigma) when both measurement and background representativeness uncertainty should be visible. Use {CO2_DAILY_MEASUREMENT_ONLY_1SIGMA_PPM} ppm (1-sigma) only for measurement-only context, and state that the daily source file itself does not provide that propagated uncertainty.

Primary reference: {NOAA_GML_WEEKLY_URL}

## F10.7

CelesTrak republishes F10.7 values assembled from primary NRCan/DRAO products rather than independently measuring them. The prepared `data/original/space_weather/SW-All.csv` file includes `F10.7_DATA_TYPE`, but this workflow only loads `F10.7_OBS_CENTER81`, so the quality flag is documented rather than used for filtering or weighting.

Tapping reports routine F10.7 values accurate to about 1 solar flux unit and about 0.4% single-monitor fluctuation. Practical daily representativeness can be larger because the operational series uses three determinations per day and can be affected by rapid solar variability or burst contamination.

Primary references: {CELESTRAK_SPACEWX_URL}; {NRCAN_F107_METHOD_URL}; {NRCAN_F107_DAILY_URL}; {NRCAN_F107_TABLE_URL}; {TAPPING_F107_DOI}

## Kp and Ap

CelesTrak republishes geomagnetic products assembled from GFZ Kp/ap/Ap sources. The prepared `data/original/space_weather/SW-All.csv` file contains the eight 3-hour Kp and ap components, but this workflow loads only `KP_SUM` and `AP_AVG` for the residual analysis.

GFZ does not provide a universal numeric daily error bar for each Kp or Ap value. The defensible uncertainty characterization is based on preliminary versus definitive status, observatory completeness, and nowcast validation; Matzka et al. report about 70% exact nowcast-definitive agreement in 2019, with most differences one-third of a Kp unit and rare differences two-thirds of a unit.

Ap inherits Kp production uncertainty and the nonlinear Kp-to-ap conversion, so a fixed daily Ap sigma is not physically well defined. The recent-window Kp/ap values listed in the CSV are empirical variability diagnostics, not formal measurement uncertainties.

Primary references: {CELESTRAK_SPACEWX_URL}; {GFZ_KP_DATA_URL}; {GFZ_KP_METHOD_URL}; {GFZ_KP_ARCHIVE_URL}; {MATZKA_KP_DOI}

## HASDM Reference Density

The Mauna Loa HASDM subset is an assimilative neutral mass density reference, not a direct sensor record or zero-error truth model. The local HASDM files and the daily residual cache do not include per-grid-cell covariance, local-time-resolved uncertainty grids, or propagated residual uncertainty.

For interpreting `ln(rho_model / rho_HASDM)`, use the HASDM percent uncertainty as a reference-density interpretation scale rather than a plotted cell error bar. A practical prior is about 3-5% 1-sigma for moderate-to-high solar activity over roughly 200-800 km, about 5-10% under solar-minimum conditions through most of the public grid, and extra caution near 700-825 km or during storm recovery.

In the natural-log residual scale, a 5% density uncertainty corresponds to about 0.049 and a 10% density uncertainty corresponds to about 0.095. External GOCE, CHAMP, and GRACE comparisons can show larger differences because they include independent-density retrieval uncertainty, aerodynamic scale differences, storm recovery behavior, and sampling differences.

Primary reference available in this repository: {STORZ_HASDM_DOI}

## SABER Cooling And Residual Scatter

No numeric uncertainty variable was found in the inspected local SABER cooling files used by this workflow. The `sd` values printed in heatmap cells are empirical within-bin scatter of the plotted quantity, not author-provided uncertainty.

The same distinction applies to model log-density-ratio error heatmaps. Cell `sd` values describe within-bin residual variability in `ln(rho_model / rho_HASDM)` and should not be read as propagated HASDM, MSIS, SABER, CO2, or space-weather error bars.
"""
    write_text(details, "source_uncertainty_details.md")


def write_method_notes(variants: dict[str, Variant]) -> None:
    notes = f"""# Mauna Loa HASDM/SABER MSIS Residual Workflow Notes

Variables:
- Residual definition: model log-density-ratio error, ln(rho_model / rho_HASDM).
- MSIS baselines: {", ".join(MODEL_VERSIONS)}.
- Residual statistic: daily mean residual only.
- Residual altitudes: {", ".join(str(alt) for alt in SELECTED_ALTITUDES)} km.
- Candidate drivers: {", ".join(DRIVER_COLS)}.
- SABER cooling columns: {", ".join(SABER_COLS)}.

Inputs:
- Residual daily-wide source: {source_daily_wide_path()}.
- Residual source is read from the HASDM MSIS outputs cache.
- SABER, CO2, and space-weather drivers are loaded with the same helpers used by the HASDM MSIS model-error analysis.

Preprocessing:
- Daily inputs are joined on their common valid date range.
- Numeric gaps are linearly interpolated, then forward/backward filled before standardized variants are generated.
- Figures use separate subplots for NRLMSISE-00, NRLMSIS 2.0, and NRLMSIS 2.1 where residuals are model-specific.
- Heatmap annotations report cell means and empirical within-bin standard deviations. These standard deviations are sample scatter, not author-provided dataset uncertainty or formal error bars.
- Altitude-axis residual heatmaps are generated separately for each CO2 sigma bin and annotate mean, empirical standard deviation, and sample count per cell.
- Source uncertainty notes are written to source_uncertainty_notes.csv as a short summary table. Full explanatory text is written to source_uncertainty_details.md.
- CelesTrak space-weather drivers have provenance and quality metadata rather than universal daily numeric error bars. For F10.7, the source-characterization scale is about 1 sfu routine accuracy and about 0.4% single-monitor fluctuation, with larger practical day-level representativeness possible during bursts or rapid solar variability. For Kp/Ap, the defensible uncertainty characterization is GFZ preliminary/definitive status, observatory completeness, and nowcast-vs-definitive validation; recent-window within-day scatter is treated only as empirical variability.
- The workflow currently loads F10.7_OBS_CENTER81, AP_AVG, and KP_SUM from `data/original/space_weather/SW-All.csv`. F10.7_DATA_TYPE and the eight 3-hour Kp/ap components are present in the local source file but are not currently used for filtering, weighting, or error propagation.
- HASDM is treated as an assimilative neutral mass density reference, not zero-error truth. The local HASDM files and daily residual cache do not carry per-grid-cell uncertainty fields. Use about 3-5% 1-sigma as a typical moderate/high-activity 200-800 km reference-density prior, about 5-10% under solar-minimum conditions through most of the public grid, and larger caution near the top of the altitude range or during storm recovery. In natural-log residual units, 5% density uncertainty is about 0.049 and 10% is about 0.095.
"""
    for variant in variants.values():
        notes += f"- {variant.name}: {variant.description}\n"
    write_text(notes, "README.md")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    df, missing_summary = combine_inputs()
    write_csv(
        missing_summary, DATA_DIR.as_posix(), "missing_summary_before_interpolation.csv"
    )
    write_csv(df, DATA_DIR.as_posix(), "daily_analysis_dataset.csv")
    write_parquet(df, DATA_DIR.as_posix(), "daily_analysis_dataset.parquet")

    variants = make_variants(df)
    plot_residual_scatter_grid(df)
    plot_saber_scatter_grid(df)
    write_csv(plot_correlation_matrices(variants), "correlation_matrices.csv")
    write_csv(
        plot_correlation_by_altitude(variants), "correlations_by_selected_altitude.csv"
    )
    plot_saber_heatmaps(df)
    plot_residual_heatmaps(df)
    plot_residual_altitude_heatmaps_by_co2_bin(df)
    write_lowest_residual_top_saber_correlations(variants)
    write_source_uncertainty_notes()
    write_source_uncertainty_details()
    write_method_notes(variants)
    print(f"Saved HASDM/SABER MSIS residual outputs to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
