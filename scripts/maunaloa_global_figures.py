from __future__ import annotations

# ruff: noqa: E402

import re
from pathlib import Path

from scripts.pgf_config import configure_pgf, fig_size, page_fig_size

configure_pgf()

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.lines import Line2D
import numpy as np
import polars as pl

from scripts.stats_utils import ols_slope_ci, pearsonr_ci

OUTPUT_ROOT = Path("outputs")
FIGURE_ROOT = Path("outputs/figures/results/set_hasdm")
OUTPUT_DIR = FIGURE_ROOT
FFT_TIMESERIES_DIR = FIGURE_ROOT / "fft_timeseries"
CORRELATION_DIR = FIGURE_ROOT / "correlation"
HEATMAP_DIR = FIGURE_ROOT / "heatmaps"
MODEL_VALIDATION_DIR = FIGURE_ROOT / "model_validations" / "causal_hasdm_saber_maunaloa"
DATASET_PATH = MODEL_VALIDATION_DIR / "daily_analysis_dataset.csv"
HASDM_WIDE_PATH = MODEL_VALIDATION_DIR / "hasdm_maunaloa_daily_wide.parquet"
SABER_PATH = Path("data/decoded/saber/saber_co2_cooling_maunaloa_daily.parquet")
LATEX_FIGURE_INDEX = "figures.tex"
FIGURE_EXTENSIONS = {".pdf", ".pgf", ".jpg", ".jpeg"}
LATEX_EXCLUDED_FIGURE_PREFIXES = (
    "lag_correlations_co2_preserved_anomaly_",
    "lag_correlations_detrended_anomaly_",
    "lag_correlations_raw_standardized_",
    "lag_correlations_seasonal_anomaly_",
)
MIN_SAMPLES_PER_HEATMAP_CELL = 10
ANALYSIS_COLS = ["F10.7_OBS_CENTER81", "AP_AVG", "KP_SUM", "CO2_ppm"]
ANALYSIS_LABELS = ["F$_{10.7,81}$", "$A_p$", "$K_p$", "CO$_2$"]
CORRELATION_DURATION_STEP_YEARS = 11
MIN_PLOTTED_RECORD_LENGTH_YEARS = CORRELATION_DURATION_STEP_YEARS
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
SABER_COLS = [
    "saber_co2cool_min_alt",
    "saber_co2cool_median_alt",
    "saber_co2cool_max_alt",
]
PAPER_CANDIDATE_FIGURE_3_ROWS = [
    "log10rho_175_daily_mean",
    "log10rho_825_daily_mean",
    "log10rho_175_daily_range",
    "log10rho_825_daily_range",
]
PAPER_CANDIDATE_FIGURE_3_COLUMNS = [
    "F10.7_OBS_CENTER81",
    "AP_AVG",
    "CO2_ppm",
    "saber_co2cool_max_alt",
]
PAPER_CANDIDATE_FIGURE_4_CAUSES = PAPER_CANDIDATE_FIGURE_3_COLUMNS
SABER_LABELS_FALLBACK = {
    "saber_co2cool_min_alt": "SABER CO$_2$ cooling min altitude",
    "saber_co2cool_median_alt": "SABER CO$_2$ cooling median altitude",
    "saber_co2cool_max_alt": "SABER CO$_2$ cooling max altitude",
}
SABER_ALTITUDE_LABELS_FALLBACK = {
    "saber_co2cool_min_alt": "100 km",
    "saber_co2cool_median_alt": "119 km",
    "saber_co2cool_max_alt": "139 km",
}
_SABER_LABELS: dict[str, str] | None = None
SPECIAL_PERIODS_YEARS = np.array([0.5, 1.0, 11.0])
FFT_PERIOD_TICKS = [
    (2 / 365.25, "2 d"),
    (7 / 365.25, "1 wk"),
    (27 / 365.25, "27 d"),
    (0.5, "6 mo"),
    (1.0, "1 y"),
    # (11.0, "11 y"),
]
FFT_PERIOD_TICKS_YEARS = np.array([period for period, _ in FFT_PERIOD_TICKS])
FFT_HEATMAP_PERIOD_BINS_PER_DECADE = 24
FFT_HEATMAP_ALTITUDE_BIN_KM = 50
SABER_HEATMAP_ALTITUDE_BIN_KM = 1
MAJOR_HASDM_ALTITUDES = [175, 250, 325, 400, 500, 600, 700, 800, 825]
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


def safe_name(text: str) -> str:
    return text.replace(".", "p").replace("/", "_").replace(" ", "_")


def altitude_from_col(col: str) -> int:
    match = re.search(r"log10rho_(\d+)_daily_", col)
    if not match:
        raise ValueError(f"No altitude in column {col}")
    return int(match.group(1))


def density_mean_col(altitude: int) -> str:
    return f"log10rho_{altitude}_daily_mean"


def density_range_col(altitude: int) -> str:
    return f"log10rho_{altitude}_daily_range"


def saber_cooling_col(altitude: int) -> str:
    return f"saber_co2cool_{altitude}km"


def load_dataset() -> tuple[pl.DataFrame, list[int]]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATASET_PATH}. Run `uv run python -m scripts.causal_hasdm_saber_maunaloa` first."
        )
    df = pl.read_csv(DATASET_PATH, try_parse_dates=True).sort("date")
    altitudes = sorted(
        {
            altitude_from_col(col)
            for col in df.columns
            if col.startswith("log10rho_") and col.endswith("_daily_mean")
        }
    )
    return df, altitudes


def load_all_altitude_correlation_dataset(
    base_df: pl.DataFrame,
) -> tuple[pl.DataFrame, list[int]]:
    if not HASDM_WIDE_PATH.exists():
        raise FileNotFoundError(
            f"Missing {HASDM_WIDE_PATH}. Run `uv run python -m scripts.causal_hasdm_saber_maunaloa` first."
        )
    wide = pl.read_parquet(HASDM_WIDE_PATH).with_columns(pl.col("date").cast(pl.Date))
    altitudes = sorted(
        {
            altitude_from_col(col)
            for col in wide.columns
            if col.startswith("log10rho_") and col.endswith("_daily_mean")
        }
        & {
            altitude_from_col(col)
            for col in wide.columns
            if col.startswith("log10rho_") and col.endswith("_daily_range")
        }
    )
    cols = [*ANALYSIS_COLS, *SABER_COLS]
    drivers = base_df.select("date", *cols).with_columns(pl.col("date").cast(pl.Date))
    density_cols = [
        col
        for altitude in altitudes
        for col in [density_mean_col(altitude), density_range_col(altitude)]
    ]
    return (
        drivers.join(wide.select("date", *density_cols), on="date", how="inner"),
        altitudes,
    )


def load_saber_cooling_heatmap_dataset() -> tuple[pl.DataFrame, list[int]]:
    if not SABER_PATH.exists():
        raise FileNotFoundError(f"Missing {SABER_PATH}.")
    long = pl.read_parquet(SABER_PATH).select(
        pl.col("date").cast(pl.Date),
        pl.col("altitude_km").cast(pl.Int64).alias("altitude_km"),
        "co2_cooling_rate_w_m3",
    )
    altitudes = long["altitude_km"].unique().sort().to_list()
    wide = long.pivot(
        index="date",
        on="altitude_km",
        values="co2_cooling_rate_w_m3",
        aggregate_function="mean",
    )
    wide = wide.rename(
        {str(altitude): saber_cooling_col(int(altitude)) for altitude in altitudes}
    ).sort("date")
    return wide, [int(altitude) for altitude in altitudes]


def saber_labels() -> dict[str, str]:
    global _SABER_LABELS
    if _SABER_LABELS is not None:
        return _SABER_LABELS
    if not SABER_PATH.exists():
        _SABER_LABELS = SABER_LABELS_FALLBACK
        return _SABER_LABELS
    altitudes = (
        pl.read_parquet(SABER_PATH, columns=["altitude_km"])["altitude_km"]
        .unique()
        .sort()
        .to_list()
    )
    if not altitudes:
        _SABER_LABELS = SABER_LABELS_FALLBACK
        return _SABER_LABELS
    median_value = float(np.median(np.asarray(altitudes, dtype=float)))
    selected = {
        "saber_co2cool_min_alt": float(min(altitudes)),
        "saber_co2cool_median_alt": float(
            min(altitudes, key=lambda alt: abs(float(alt) - median_value))
        ),
        "saber_co2cool_max_alt": float(max(altitudes)),
    }
    _SABER_LABELS = {
        col: f"CO$_2$ cl.\n{altitude:.0f} km" for col, altitude in selected.items()
    }
    return _SABER_LABELS


def label_for_col(col: str) -> str:
    if col in SPACE_WEATHER_SIGMA_LABELS:
        return SPACE_WEATHER_SIGMA_LABELS[col]
    if col == "CO2_ppm":
        return "CO$_2$"
    if col in SABER_COLS:
        return saber_labels().get(col, SABER_LABELS_FALLBACK[col])
    if col.startswith("log10rho_") and col.endswith("_daily_range"):
        return rf"$\Delta\ell_\rho$ {altitude_from_col(col)} km"
    if col.startswith("log10rho_"):
        return rf"$\bar{{\ell}}_\rho$ {altitude_from_col(col)} km"
    return col


def saber_altitude_label(col: str) -> str:
    label = saber_labels().get(col, SABER_LABELS_FALLBACK[col])
    altitude = re.search(r"(\d+(?:\.\d+)?)\s*km", label)
    return (
        f"{altitude.group(1)} km" if altitude else SABER_ALTITUDE_LABELS_FALLBACK[col]
    )


def compact_label_for_col(col: str) -> str:
    if col == "F10.7_OBS_CENTER81":
        return "F$_{10.7,81}$"
    if col == "AP_AVG":
        return "$A_p$"
    if col == "KP_SUM":
        return "$K_p$"
    if col == "CO2_ppm":
        return "CO$_2$"
    if col in SABER_COLS:
        label = saber_labels().get(col, SABER_LABELS_FALLBACK[col])
        altitude = label.rsplit(" ", 2)[-2]
        return f"SABER {altitude} km"
    return label_for_col(col)


def finite_xy(
    df: pl.DataFrame, x_col: str, y_col: str
) -> tuple[np.ndarray, np.ndarray]:
    if x_col == y_col:
        series = df.select(x_col).drop_nulls()[x_col].to_numpy().astype(float)
        mask = np.isfinite(series)
        return series[mask], series[mask]
    pair = df.select(x_col, y_col).drop_nulls()
    x = pair[x_col].to_numpy().astype(float)
    y = pair[y_col].to_numpy().astype(float)
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
        return f"$<${CORRELATION_DURATION_STEP_YEARS} yr"
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
    if duration_years < MIN_PLOTTED_RECORD_LENGTH_YEARS:
        return np.nan, np.nan, np.nan, None
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


def add_record_length_legend(
    ax: plt.Axes,
    duration_bins: set[int],
    *,
    loc: str = "upper left",
    bbox_to_anchor: tuple[float, float] = (1.01, 1),
    ncols: int | None = None,
) -> None:
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
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        borderaxespad=0,
        ncols=ncols or 1,
    )


def sigma_edges(values: np.ndarray) -> tuple[np.ndarray, float, float, np.ndarray]:
    mean = float(np.mean(values))
    std = float(np.std(values))
    if not np.isfinite(std) or std == 0:
        raise ValueError("Cannot build sigma bins for a constant variable.")
    z = (values - mean) / std
    edges = np.arange(np.floor(np.min(z)), np.ceil(np.max(z)) + 1, 1.0)
    if len(edges) < 2:
        edges = np.array([-0.5, 0.5])
    return edges, mean, std, z


def sigma_bin_labels(edges: np.ndarray) -> list[str]:
    return [
        f"{int(edges[idx])} to {int(edges[idx + 1])}" for idx in range(len(edges) - 1)
    ]


def linear_fit_stats(
    x: np.ndarray, y: np.ndarray
) -> tuple[float, float, float, float, float, float]:
    if len(x) < MIN_SAMPLES_PER_HEATMAP_CELL or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    slope, slope_lo, slope_hi, slope_se, intercept, correlation, error, _ = (
        ols_slope_ci(x, y)
    )
    zero_crossing = float(-intercept / slope) if slope != 0 else np.nan
    return correlation, float(slope), zero_crossing, error, slope_lo, slope_hi


def save_and_close(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    if path.suffix == ".pgf":
        FigureCanvasAgg(fig)
        fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


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


def subsection_name(text: str) -> str:
    return text.replace("_", " ").title()


def figure_caption(path: Path) -> str:
    return latex_escape(path.stem.replace("_", " ").title())


def include_latex_figure(path: Path) -> bool:
    return not any(
        path.stem.startswith(prefix) for prefix in LATEX_EXCLUDED_FIGURE_PREFIXES
    )


def write_latex_figure_indexes(
    root: Path = FIGURE_ROOT, path_root: Path = OUTPUT_ROOT
) -> None:
    directories = [root, *sorted(path for path in root.rglob("*") if path.is_dir())]
    for directory in sorted(
        directories, key=lambda path: len(path.parts), reverse=True
    ):
        figures = sorted(
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.suffix.lower() in FIGURE_EXTENSIONS
            and include_latex_figure(path)
        )
        child_indexes = sorted(
            path / LATEX_FIGURE_INDEX
            for path in directory.iterdir()
            if path.is_dir() and (path / LATEX_FIGURE_INDEX).exists()
        )
        if directory == HEATMAP_DIR:
            child_indexes = [
                path
                for path in child_indexes
                if path.parent.name == "density_co2_correlation_heatmaps_all_altitudes"
            ]
        if not figures and not child_indexes:
            continue

        rel_dir = directory.relative_to(root)
        title = (
            "Mauna Loa HASDM/SABER Figures"
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


def date_xlim(df: pl.DataFrame) -> tuple[object, object]:
    dates = df.select("date").drop_nulls()["date"]
    return dates.min(), dates.max()


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


def plot_altitude_time_heatmap(
    ax: plt.Axes,
    df: pl.DataFrame,
    altitudes: list[int],
    col_fn,
    ylabel: str,
    colorbar_label: str,
    altitude_bin_km: float = FFT_HEATMAP_ALTITUDE_BIN_KM,
) -> object | None:
    available = [altitude for altitude in altitudes if col_fn(altitude) in df.columns]
    if not available:
        ax.text(
            0.5,
            0.5,
            "No altitude heatmap data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return None
    data = (
        df.select("date", *[col_fn(altitude) for altitude in available])
        .sort("date")
        .drop_nulls("date")
    )
    dates = data["date"].to_numpy()
    altitude_edges = altitude_bin_edges(available, altitude_bin_km)
    matrix_sum = np.zeros((len(altitude_edges) - 1, len(dates)), dtype=float)
    matrix_count = np.zeros_like(matrix_sum)
    for altitude in available:
        row = np.searchsorted(altitude_edges, altitude, side="right") - 1
        values = data[col_fn(altitude)].to_numpy().astype(float)
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
    mesh = ax.pcolormesh(
        date_edges_for_heatmap(dates),
        altitude_edges,
        matrix,
        shading="auto",
        cmap="viridis",
        rasterized=True,
    )
    if matrix.shape[0] > 1 and matrix.shape[1] > 1 and np.any(np.isfinite(matrix)):
        date_centers = mdates.date2num(dates)
        altitude_centers = 0.5 * (altitude_edges[:-1] + altitude_edges[1:])
        levels = np.linspace(np.nanmin(matrix), np.nanmax(matrix), 4)
        if len(np.unique(levels)) > 1:
            ax.contour(
                date_centers,
                altitude_centers,
                matrix,
                levels=levels,
                colors="black",
                linewidths=0.35,
                alpha=0.45,
            )
    ax.xaxis_date()
    ax.set_ylabel(ylabel)
    format_altitude_axis(ax)
    ax.grid(True, alpha=0.18)
    return mesh


def plot_time_series(
    df: pl.DataFrame,
    altitudes: list[int],
    heatmap_df: pl.DataFrame | None = None,
    heatmap_altitudes: list[int] | None = None,
    saber_heatmap_df: pl.DataFrame | None = None,
    saber_heatmap_altitudes: list[int] | None = None,
) -> None:
    fig, axes = plt.subplots(8, 1, figsize=page_fig_size(1.0, 1.45, 0.98), sharex=True)
    fig.subplots_adjust(left=0.08, right=0.84, top=0.80, bottom=0.08, hspace=0.12)
    heatmap_source = heatmap_df if heatmap_df is not None else df
    heatmap_alts = heatmap_altitudes if heatmap_altitudes is not None else altitudes
    for altitude in altitudes:
        axes[0].plot(
            df["date"],
            df[density_mean_col(altitude)],
            linewidth=0.9,
            label=f"{altitude} km",
        )
    axes[0].set_ylabel(r"$\bar{\ell}_\rho$")
    axes[0].grid(True, alpha=0.25)

    density_mean_mesh = plot_altitude_time_heatmap(
        axes[1],
        heatmap_source,
        heatmap_alts,
        density_mean_col,
        "$\\bar{\\ell}_\\rho$\naltitude (km)",
        r"$\bar{\ell}_\rho$",
    )

    for altitude in altitudes:
        axes[2].plot(
            df["date"],
            df[density_range_col(altitude)],
            linewidth=0.9,
            label=f"{altitude} km",
        )
    axes[2].set_ylabel(r"$\Delta\ell_\rho$")
    axes[2].grid(True, alpha=0.25)

    density_handles, density_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        density_handles,
        density_labels,
        ncols=2,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.30, 0.965),
        borderaxespad=0,
        frameon=True,
    )

    density_range_mesh = plot_altitude_time_heatmap(
        axes[3],
        heatmap_source,
        heatmap_alts,
        density_range_col,
        "$\\Delta\\ell_\\rho$\naltitude (km)",
        r"$\Delta\ell_\rho$",
    )

    axes[4].plot(
        df["date"], df["F10.7_OBS_CENTER81"], color="darkred", label="$F_{10.7,81}$"
    )
    axes2_ap = axes[4].twinx()
    axes2_ap.plot(
        df["date"], df["AP_AVG"], color="darkviolet", alpha=0.75, label="$A_p$"
    )
    axes[4].set_ylabel("$F_{10.7,81}$")
    axes2_ap.set_ylabel("$A_p$")
    axes[4].grid(True, alpha=0.25)

    for col in SABER_COLS:
        axes[5].plot(
            df["date"],
            df[col],
            linewidth=0.9,
            label=saber_altitude_label(col),
        )
    axes[5].set_ylabel("CO$_2$ cooling")
    axes[5].grid(True, alpha=0.25)
    saber_handles, saber_labels = axes[5].get_legend_handles_labels()
    fig.legend(
        saber_handles,
        saber_labels,
        ncols=2,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.75, 0.965),
        borderaxespad=0,
        frameon=True,
    )

    if saber_heatmap_df is not None and saber_heatmap_altitudes is not None:
        saber_source = df.select("date").join(saber_heatmap_df, on="date", how="left")
        saber_mesh = plot_altitude_time_heatmap(
            axes[6],
            saber_source,
            saber_heatmap_altitudes,
            saber_cooling_col,
            "SABER cooling\naltitude (km)",
            "W m$^{-3}$",
            SABER_HEATMAP_ALTITUDE_BIN_KM,
        )
    else:
        saber_mesh = None
        axes[6].text(
            0.5,
            0.5,
            "No SABER cooling heatmap data",
            ha="center",
            va="center",
            transform=axes[6].transAxes,
        )

    axes[7].plot(df["date"], df["CO2_ppm"], color="darkgreen", label="CO$_2$")
    axes[7].set_ylabel("CO$_2$ (ppm)")
    axes[7].grid(True, alpha=0.25)
    axes[7].set_xlim(*date_xlim(df))
    axes[7].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[7].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(axes[7].get_xticklabels(), rotation=45, ha="right")
    for ax, mesh, label in [
        (axes[1], density_mean_mesh, r"$\bar{\ell}_\rho$"),
        (axes[3], density_range_mesh, r"$\Delta\ell_\rho$"),
        (axes[6], saber_mesh, "W m$^{-3}$"),
    ]:
        if mesh is None:
            continue
        pos = ax.get_position()
        cax = fig.add_axes([0.87, pos.y0, 0.018, pos.height])
        fig.colorbar(mesh, cax=cax, label=label)
    fig.suptitle("Mauna Loa HASDM/SABER time series", y=0.995)
    save_and_close(fig, FFT_TIMESERIES_DIR / "maunaloa_timeseries.pgf")


def fft_period_power(df: pl.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
    series = df.select("date", col).sort("date").drop_nulls()
    values = series[col].to_numpy().astype(float)
    if len(values) < 3:
        return np.array([]), np.array([])
    values = values - np.mean(values)
    days = series["date"].cast(pl.Int64).to_numpy()
    sample_days = np.median(np.diff(days))
    frequencies = np.fft.rfftfreq(len(values), d=sample_days)
    amplitudes = np.abs(np.fft.rfft(values))
    valid = frequencies > 0
    return 1 / frequencies[valid], amplitudes[valid]


def fft_period_tick_label(tick: float) -> str:
    for period, label in FFT_PERIOD_TICKS:
        if np.isclose(tick, period):
            return label
    return f"{tick:g} y"


def add_period_xticks(ax: plt.Axes) -> None:
    ticks = ax.get_xticks()
    ticks = ticks[(ticks > 0) & np.isfinite(ticks)]
    ticks = np.unique(np.concatenate([ticks, FFT_PERIOD_TICKS_YEARS]))
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [fft_period_tick_label(tick) for tick in ticks], rotation=45, ha="right"
    )


def period_bin_edges(periods_years: np.ndarray) -> np.ndarray:
    finite = periods_years[np.isfinite(periods_years) & (periods_years > 0)]
    if len(finite) == 0:
        return np.array([])
    span_decades = np.log10(np.max(finite)) - np.log10(np.min(finite))
    n_bins = max(12, int(np.ceil(span_decades * FFT_HEATMAP_PERIOD_BINS_PER_DECADE)))
    return np.logspace(np.log10(np.min(finite)), np.log10(np.max(finite)), n_bins + 1)


def altitude_bin_edges(
    altitudes: list[int], bin_km: float = FFT_HEATMAP_ALTITUDE_BIN_KM
) -> np.ndarray:
    if not altitudes:
        return np.array([])
    min_alt = float(min(altitudes)) - bin_km / 2
    max_alt = float(max(altitudes)) + bin_km / 2
    return np.arange(min_alt, max_alt + 1e-9, bin_km)


def binned_fft_power_by_altitude(
    df: pl.DataFrame,
    altitudes: list[int],
    col_fn,
    altitude_bin_km: float = FFT_HEATMAP_ALTITUDE_BIN_KM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spectra = []
    all_periods = []
    for altitude in altitudes:
        col = col_fn(altitude)
        if col not in df.columns:
            continue
        periods_days, amplitudes = fft_period_power(df, col)
        if len(periods_days) == 0:
            continue
        periods_years = periods_days / 365.25
        power = amplitudes**2
        spectra.append((altitude, periods_years, power))
        all_periods.extend(periods_years)
    if not spectra:
        return np.array([]), np.array([]), np.array([[]])

    period_edges = period_bin_edges(np.asarray(all_periods, dtype=float))
    altitude_edges = altitude_bin_edges(
        [altitude for altitude, _, _ in spectra], altitude_bin_km
    )
    period_count = len(period_edges) - 1
    altitude_count = len(altitude_edges) - 1
    per_altitude_power = np.full((len(spectra), period_count), np.nan)
    spectrum_altitudes = np.array([altitude for altitude, _, _ in spectra], dtype=float)

    for row, (_, periods_years, power) in enumerate(spectra):
        period_bins = np.digitize(periods_years, period_edges) - 1
        for period_idx in range(period_count):
            values = power[period_bins == period_idx]
            if len(values):
                per_altitude_power[row, period_idx] = float(np.nanmean(values))

    period_centers = np.sqrt(period_edges[:-1] * period_edges[1:])
    target_log_periods = np.log10(period_centers)
    for row in range(per_altitude_power.shape[0]):
        values = per_altitude_power[row, :]
        mask = np.isfinite(values) & (values > 0)
        if np.sum(mask) >= 2:
            per_altitude_power[row, :] = 10 ** np.interp(
                target_log_periods,
                target_log_periods[mask],
                np.log10(values[mask]),
                left=np.nan,
                right=np.nan,
            )

    matrix = np.full((altitude_count, period_count), np.nan)
    for altitude_idx in range(altitude_count):
        lower = altitude_edges[altitude_idx]
        upper = altitude_edges[altitude_idx + 1]
        if altitude_idx == altitude_count - 1:
            mask = (spectrum_altitudes >= lower) & (spectrum_altitudes <= upper)
        else:
            mask = (spectrum_altitudes >= lower) & (spectrum_altitudes < upper)
        if np.any(mask):
            values = per_altitude_power[mask, :]
            finite_cols = np.any(np.isfinite(values), axis=0)
            if np.any(finite_cols):
                matrix[altitude_idx, finite_cols] = np.nanmean(
                    values[:, finite_cols], axis=0
                )
    return period_edges, altitude_edges, matrix


def plot_fft_heatmap(
    ax: plt.Axes,
    df: pl.DataFrame,
    altitudes: list[int],
    col_fn,
    ylabel: str,
    altitude_bin_km: float = FFT_HEATMAP_ALTITUDE_BIN_KM,
) -> None:
    period_edges, altitude_edges, power = binned_fft_power_by_altitude(
        df, altitudes, col_fn, altitude_bin_km
    )
    if len(period_edges) == 0 or len(altitude_edges) == 0:
        ax.text(
            0.5,
            0.5,
            "No all-altitude FFT data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return
    log_power = np.log10(power + np.finfo(float).eps)
    mesh = ax.pcolormesh(
        period_edges,
        altitude_edges,
        log_power,
        shading="auto",
        cmap="magma",
        rasterized=True,
    )
    ax.set_xscale("log")
    format_altitude_axis(ax)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.18)
    bbox = ax.get_position()
    cax = ax.figure.add_axes([bbox.x1 + 0.006, bbox.y0, 0.012, bbox.height])
    ax.figure.colorbar(mesh, cax=cax, label="log10 FFT power")


def plot_fft(
    df: pl.DataFrame,
    altitudes: list[int],
    heatmap_df: pl.DataFrame | None = None,
    heatmap_altitudes: list[int] | None = None,
    saber_heatmap_df: pl.DataFrame | None = None,
    saber_heatmap_altitudes: list[int] | None = None,
) -> None:
    groups = [
        (
            r"$|\mathcal{F}(\bar{\ell}_\rho)|$",
            [density_mean_col(altitude) for altitude in altitudes],
        ),
        (
            r"$|\mathcal{F}(\Delta\ell_\rho)|$",
            [density_range_col(altitude) for altitude in altitudes],
        ),
        (r"$|\mathcal{F}(\mathrm{SABER})|$", SABER_COLS),
        (r"$|\mathcal{F}(F_{10.7,81})|$", ["F10.7_OBS_CENTER81"]),
        (r"$|\mathcal{F}(A_p)|$", ["AP_AVG"]),
        (r"$|\mathcal{F}(\mathrm{CO_2})|$", ["CO2_ppm"]),
    ]
    fig, axes = plt.subplots(
        len(groups) + 2,
        1,
        figsize=page_fig_size(1.0, 1.45, 0.98),
        sharex=True,
        constrained_layout=False,
        gridspec_kw={"height_ratios": [1.0, 1.05, 1.0, 0.9, 1.05, 0.9, 0.9, 0.9]},
    )
    fig.subplots_adjust(left=0.14, right=0.78, top=0.93, bottom=0.08, hspace=0.20)
    limits = []

    for ax, (title, cols) in [(axes[0], groups[0])]:
        for col in cols:
            periods_days, amplitudes = fft_period_power(df, col)
            periods_years = periods_days / 365.25
            limits.extend(periods_years)
            if len(periods_years):
                ax.plot(
                    periods_years,
                    amplitudes,
                    linewidth=1.0,
                    label=f"{altitude_from_col(col)}",
                )
        ax.set_ylabel(title)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
    density_handles, density_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        density_handles,
        density_labels,
        title="$\\bar{\\ell_\\rho}$ altitude (km)",
        fontsize=6.0,
        title_fontsize=6.6,
        ncols=1,
        loc="upper left",
        bbox_to_anchor=(0.805, 0.93),
        borderaxespad=0,
        framealpha=0.85,
    )

    plot_fft_heatmap(
        axes[1],
        heatmap_df if heatmap_df is not None else df,
        heatmap_altitudes if heatmap_altitudes is not None else altitudes,
        density_mean_col,
        "Altitude (km)",
    )

    for ax, (title, cols) in zip(
        [axes[2], axes[3], axes[5], axes[6], axes[7]], groups[1:]
    ):
        for col in cols:
            periods_days, amplitudes = fft_period_power(df, col)
            periods_years = periods_days / 365.25
            limits.extend(periods_years)
            if len(periods_years):
                if col in SABER_COLS:
                    match = re.search(r"(\d+)\s*km", label_for_col(col))
                    label = f"{match.group(1)} km" if match else label_for_col(col)
                elif col.startswith("log10rho_"):
                    label = f"{altitude_from_col(col)} km"
                else:
                    label = label_for_col(col)
                ax.plot(periods_years, amplitudes, linewidth=1.0, label=label)
        ax.set_ylabel(title)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        if cols == SABER_COLS:
            ax.legend(
                fontsize=5.8,
                ncols=1,
                loc="upper left",
                bbox_to_anchor=(1.01, 1),
                borderaxespad=0,
                framealpha=0.85,
            )

    if saber_heatmap_df is not None and saber_heatmap_altitudes is not None:
        saber_source = df.select("date").join(saber_heatmap_df, on="date", how="left")
        plot_fft_heatmap(
            axes[4],
            saber_source,
            saber_heatmap_altitudes,
            saber_cooling_col,
            "Altitude (km)",
            SABER_HEATMAP_ALTITUDE_BIN_KM,
        )
    else:
        axes[4].text(
            0.5,
            0.5,
            "No SABER CO$_2$ cooling FFT heatmap data",
            ha="center",
            va="center",
            transform=axes[4].transAxes,
        )
        axes[4].set_xscale("log")
    axes[-1].set_xlabel("Period")
    add_period_xticks(axes[-1])
    finite_limits = np.asarray(limits, dtype=float)
    finite_limits = finite_limits[np.isfinite(finite_limits) & (finite_limits > 0)]
    if len(finite_limits):
        axes[-1].set_xlim(np.min(finite_limits), np.max(finite_limits))
    fig.suptitle("Mauna Loa HASDM/SABER FFT spectra")
    save_and_close(fig, FFT_TIMESERIES_DIR / "maunaloa_fft.pgf")


def plot_correlation_heatmap(
    df: pl.DataFrame, cols: list[str], filename: str, title: str
) -> None:
    corr_df = df.select(cols).drop_nulls()
    matrix = corr_df.corr().to_numpy()
    labels = [compact_label_for_col(col) for col in cols]
    fig, ax = plt.subplots(figsize=(5.4 * 0.9, 4.8 * 0.9), constrained_layout=True)
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm", rasterized=True)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=9)
    for row in range(len(labels)):
        for col in range(len(labels)):
            ax.text(
                col,
                row,
                f"{matrix[row, col]:.2f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color="white" if abs(matrix[row, col]) > 0.55 else "black",
            )
    colorbar = fig.colorbar(image, ax=ax, label="Pearson r")
    colorbar.ax.tick_params(labelsize=9)
    colorbar.set_label("Pearson r", fontsize=10)
    ax.set_title(title, fontsize=11)
    save_and_close(fig, CORRELATION_DIR / filename)


def plot_scatter_matrix(
    df: pl.DataFrame, cols: list[str], filename: str, title: str
) -> None:
    labels = [label_for_col(col) for col in cols]
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
            x, y = finite_xy(df, x_col, y_col)
            ax.scatter(x, y, s=5, alpha=0.22, rasterized=scatter_rasterized(len(x)))
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
    save_and_close(fig, CORRELATION_DIR / filename)


def plot_density_scatter_by_altitude(df: pl.DataFrame, altitudes: list[int]) -> None:
    out_dir = CORRELATION_DIR / "density_scatter_by_altitude"
    scatter_cols = [*ANALYSIS_COLS, *SABER_COLS]
    omit_kp_cases = {("mean", 825), ("range", 175)}
    for metric, col_fn, title_prefix in [
        ("mean", density_mean_col, r"$\bar{\ell_\rho}$"),
        ("range", density_range_col, r"$\Delta\ell_\rho$"),
    ]:
        for altitude in altitudes:
            cols = [
                col
                for col in scatter_cols
                if not ((metric, altitude) in omit_kp_cases and col == "KP_SUM")
            ]
            y_col = col_fn(altitude)
            fig, axes = plt.subplots(
                1,
                len(cols),
                figsize=(7, 1.8),
                sharey=True,
                constrained_layout=True,
            )
            axes = np.atleast_1d(axes)
            for ax, x_col in zip(axes, cols):
                x, y = finite_xy(df, x_col, y_col)
                ax.scatter(x, y, s=6, alpha=0.25, rasterized=scatter_rasterized(len(x)))
                # ax.set_xlabel(label_for_col(x_col), fontsize=8)
                ax.set_title(label_for_col(x_col), fontsize=8)
                ax.grid(True, alpha=0.2)
            axes[0].set_ylabel(label_for_col(y_col), fontsize=8)
            fig.suptitle(f"{title_prefix} scatter at {altitude} km")
            save_and_close(
                fig, out_dir / f"maunaloa_density_{metric}_scatter_{altitude}km.pgf"
            )


def paper_candidate_figure_3_layout() -> tuple[list[str], list[str]]:
    """Return the fixed target and driver order selected for Figure 3."""
    return [*PAPER_CANDIDATE_FIGURE_3_ROWS], [*PAPER_CANDIDATE_FIGURE_3_COLUMNS]


def require_columns(df: pl.DataFrame, cols: list[str], figure_name: str) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"{figure_name} requires columns: {', '.join(missing)}")


def plot_paper_candidate_figure_3(df: pl.DataFrame, altitudes: list[int]) -> None:
    rows, columns = paper_candidate_figure_3_layout()
    required_altitudes = [175, 825]
    missing_altitudes = [
        altitude for altitude in required_altitudes if altitude not in altitudes
    ]
    if missing_altitudes:
        raise ValueError(
            "Paper candidate Figure 3 requires HASDM altitudes: "
            + ", ".join(f"{altitude} km" for altitude in missing_altitudes)
        )
    require_columns(df, [*rows, *columns], "Paper candidate Figure 3")
    fig, axes = plt.subplots(4, 4, figsize=(8.4, 7.6), constrained_layout=True)
    for row, y_col in enumerate(rows):
        for col, x_col in enumerate(columns):
            ax = axes[row, col]
            x, y = finite_xy(df, x_col, y_col)
            ax.scatter(x, y, s=5, alpha=0.22, rasterized=scatter_rasterized(len(x)))
            if row == 0:
                ax.set_title(label_for_col(x_col), fontsize=9)
            if col == 0:
                ax.set_ylabel(label_for_col(y_col), fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.15)
    fig.suptitle(
        "Paper candidate Figure 3: HASDM driver scatter composite", fontsize=11
    )
    save_and_close(
        fig, CORRELATION_DIR / "paper_candidate_figure_3_hasdm_scatter_composite.pgf"
    )


def plot_correlation_by_altitude(
    df: pl.DataFrame,
    altitudes: list[int],
    causes: list[str] | None = None,
    filename: str = "maunaloa_correlation_by_altitude.pgf",
    title: str = "Mauna Loa correlation by HASDM altitude",
) -> None:
    causes = [*ANALYSIS_COLS, *SABER_COLS] if causes is None else causes
    require_columns(df, causes, title)
    fig, axes = plt.subplots(
        2, 1, figsize=(7.2, 5.2), sharex=True, constrained_layout=True
    )
    observed_duration_bins: set[int] = set()
    for ax, panel_title, col_fn in [
        (axes[0], r"$\bar{\ell_\rho}$", density_mean_col),
        (axes[1], r"$\Delta\ell_\rho$", density_range_col),
    ]:
        add_correlation_effect_size_bands(ax)
        for cause in causes:
            corr = []
            corr_los = []
            corr_his = []
            duration_bins = []
            for altitude in altitudes:
                point_corr, r_lo, r_hi, duration_bin = correlation_and_duration(
                    df, cause, col_fn(altitude)
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
                linewidth=1.7,
                label=compact_label_for_col(cause).replace("\n", " "),
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
                        s=42,
                        color=line.get_color(),
                        edgecolors="black",
                        linewidths=0.55,
                        zorder=4,
                    )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylim(-1, 1)
        ax.set_ylabel("Pearson r", fontsize=10)
        ax.set_title(panel_title, fontsize=10)
        ax.tick_params(axis="both", labelsize=9)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("HASDM altitude (km)", fontsize=10)
    driver_legend = axes[0].legend(
        fontsize=7.5,
        ncols=2,
        loc="lower left",
        bbox_to_anchor=(1.01, 0),
        borderaxespad=0,
    )
    axes[0].add_artist(driver_legend)
    add_record_length_legend(axes[0], observed_duration_bins)
    fig.suptitle(title, fontsize=11)
    save_and_close(fig, CORRELATION_DIR / filename)


def metric_plot_path(
    output_dir: Path, driver_col: str, metric_name: str, metric: str
) -> Path:
    return (
        output_dir
        / f"maunaloa_density_{metric}_co2_{metric_name}_by_altitude_for_{safe_name(driver_col)}.pgf"
    )


def slope_count_summary_path(output_dir: Path, driver_col: str, metric: str) -> Path:
    return (
        output_dir
        / f"maunaloa_density_{metric}_co2_slope_count_summary_by_altitude_for_{safe_name(driver_col)}.pgf"
    )


def representative_row_counts(counts: np.ndarray) -> list[float]:
    row_counts = []
    for row in range(counts.shape[0]):
        row_values = counts[row].astype(float)
        finite = row_values[np.isfinite(row_values)]
        row_counts.append(float(finite[0]) if finite.size else np.nan)
    return row_counts


def mask_short_record_values(
    values: np.ndarray,
    duration_bins: np.ndarray | None,
) -> np.ndarray:
    plotted_values = values.astype(float, copy=True)
    if duration_bins is not None:
        plotted_values[duration_bins < MIN_PLOTTED_RECORD_LENGTH_YEARS] = np.nan
    return plotted_values


def plot_slope_count_summary_by_altitude_and_sigma(
    output_dir: Path,
    driver_col: str,
    altitudes: list[int],
    row_labels: np.ndarray,
    metric_label: str,
    metric: str,
    slopes: np.ndarray,
    counts: np.ndarray,
    duration_bins: np.ndarray | None = None,
    ci_values: np.ndarray | None = None,
) -> None:
    if slopes.size == 0 or not np.any(np.isfinite(slopes)):
        return
    row_counts = representative_row_counts(counts)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=fig_size(1.0, 0.45),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [2.05, 1.35]},
    )
    slope_ax, count_ax = axes
    observed_duration_bins: set[int] = set()
    colors = [plt.get_cmap("tab10")(idx) for idx in range(len(row_labels))]
    for row, (sigma_label, sample_count, color) in enumerate(
        zip(row_labels, row_counts, colors, strict=False)
    ):
        row_values = mask_short_record_values(
            slopes[row], duration_bins[row] if duration_bins is not None else None
        )
        if not np.any(np.isfinite(row_values)):
            continue
        count_label = f"n={sample_count:.0f}" if np.isfinite(sample_count) else "n=NA"
        (line,) = slope_ax.plot(
            altitudes,
            row_values,
            linewidth=1.5,
            color=color,
            label=f"{sigma_label} ({count_label})",
            zorder=3,
        )
        if ci_values is not None:
            row_lo = ci_values[row, :, 0].astype(float)
            row_hi = ci_values[row, :, 1].astype(float)
            finite_ci = (
                np.isfinite(row_values) & np.isfinite(row_lo) & np.isfinite(row_hi)
            )
            if np.any(finite_ci):
                slope_ax.fill_between(
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
                    slope_ax.scatter(
                        np.asarray(altitudes)[mask],
                        row_values[mask],
                        marker=correlation_duration_marker(int(duration_bin)),
                        s=32,
                        color=line.get_color(),
                        edgecolors="black",
                        linewidths=0.45,
                        zorder=4,
                    )

    slope_ax.set_xlabel("Altitude (km)")
    slope_ax.set_ylabel(metric_label)
    slope_ax.grid(True, alpha=0.3)
    add_record_length_legend(
        count_ax,
        observed_duration_bins,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        ncols=1,
    )
    positions = np.arange(len(row_labels))
    count_ax.bar(positions, row_counts, color=colors, alpha=0.8)
    count_ax.set_xticks(positions, row_labels, rotation=35, ha="right")
    count_ax.set_xlabel(f"{SPACE_WEATHER_SIGMA_LABELS[driver_col]} activity bin")
    count_ax.set_ylabel("Sample count n")
    count_ax.grid(True, axis="y", alpha=0.3)
    save_and_close(fig, slope_count_summary_path(output_dir, driver_col, metric))


def plot_metric_by_altitude_and_sigma(
    output_dir: Path,
    driver_col: str,
    altitudes: list[int],
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
    if metric_name == "sample_count":
        counts = []
        for row in range(len(row_labels)):
            row_values = values[row].astype(float)
            finite = row_values[np.isfinite(row_values)]
            counts.append(float(finite[0]) if finite.size else np.nan)
        fig, ax = plt.subplots(figsize=fig_size(0.45, 0.72), constrained_layout=True)
        positions = np.arange(len(row_labels))
        colors = [plt.get_cmap("tab10")(idx) for idx in range(len(row_labels))]
        ax.bar(positions, counts, color=colors, alpha=0.8)
        ax.set_xticks(positions, row_labels, rotation=35, ha="right")
        ax.set_xlabel(f"{SPACE_WEATHER_SIGMA_LABELS[driver_col]} activity bin")
        ax.set_ylabel(metric_label)
        ax.grid(True, axis="y", alpha=0.3)
        save_and_close(
            fig, metric_plot_path(output_dir, driver_col, metric_name, metric)
        )
        return
    fig, ax = plt.subplots(figsize=fig_size(1.0, 0.52), constrained_layout=True)
    observed_duration_bins: set[int] = set()
    for row, sigma_label in enumerate(row_labels):
        row_values = mask_short_record_values(
            values[row], duration_bins[row] if duration_bins is not None else None
        )
        if np.any(np.isfinite(row_values)):
            (line,) = ax.plot(
                altitudes, row_values, linewidth=1.5, label=sigma_label, zorder=3
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
        title=f"{SPACE_WEATHER_SIGMA_LABELS[driver_col]} bin",
        fontsize="small",
        loc="lower left",
        bbox_to_anchor=(1.01, 0),
        borderaxespad=0,
    )
    ax.add_artist(data_legend)
    add_record_length_legend(ax, observed_duration_bins)
    save_and_close(fig, metric_plot_path(output_dir, driver_col, metric_name, metric))


def fit_annotation(
    correlation: float,
    slope: float,
    error: float,
    slope_lo: float = np.nan,
    slope_hi: float = np.nan,
) -> str:
    ci_str = ""
    if np.isfinite(slope_lo) and np.isfinite(slope_hi):
        ci_str = f"\nCI=[{slope_lo:.2e}, {slope_hi:.2e}]"
    return f"r={correlation:.2f}\nm={slope:.2e}{ci_str}\nerr={error:.3f}"


def plot_density_co2_correlation_heatmaps(
    df: pl.DataFrame,
    altitudes: list[int],
    heatmap_dir: Path | None = None,
    metric_dir: Path | None | bool = None,
) -> None:
    heatmap_dir = (
        HEATMAP_DIR / "density_co2_correlation_heatmaps"
        if heatmap_dir is None
        else heatmap_dir
    )
    if metric_dir is None:
        metric_dir = CORRELATION_DIR / "density_co2_fit_metric_plots"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    if metric_dir:
        metric_dir.mkdir(parents=True, exist_ok=True)
    for metric, col_fn, ylabel in [
        ("mean", density_mean_col, r"$\bar{\ell_\rho}$"),
        ("range", density_range_col, r"$\Delta\ell_\rho$"),
    ]:
        for driver_col in SPACE_WEATHER_SIGMA_COLS:
            driver_values = (
                df.select(driver_col).drop_nulls()[driver_col].to_numpy().astype(float)
            )
            driver_values = driver_values[np.isfinite(driver_values)]
            if len(driver_values) < MIN_SAMPLES_PER_HEATMAP_CELL:
                continue
            edges, driver_mean, driver_std, _ = sigma_edges(driver_values)
            work_df = df.with_columns(
                ((pl.col(driver_col) - driver_mean) / driver_std).alias(
                    "__driver_sigma"
                )
            )
            correlations = np.full((len(edges) - 1, len(altitudes)), np.nan)
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
                upper = (
                    pl.col("__driver_sigma") <= edges[row + 1]
                    if row == len(edges) - 2
                    else pl.col("__driver_sigma") < edges[row + 1]
                )
                bin_df = work_df.filter(
                    (pl.col("__driver_sigma") >= edges[row]) & upper
                )
                for col, altitude in enumerate(altitudes):
                    co2, density = finite_xy(bin_df, "CO2_ppm", col_fn(altitude))
                    counts[row, col] = len(co2)
                    (
                        correlations[row, col],
                        slopes[row, col],
                        zero_crossings[row, col],
                        errors[row, col],
                        slope_los[row, col],
                        slope_his[row, col],
                    ) = linear_fit_stats(co2, density)
                    _, r_lo, r_hi, _ = pearsonr_ci(co2, density)
                    corr_los[row, col] = r_lo
                    corr_his[row, col] = r_hi
                    y_col = col_fn(altitude)
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
                        duration_bins[row, col] = correlation_duration_bin(
                            duration_years
                        )
            painted_rows = np.any(np.isfinite(correlations), axis=1)
            if not np.any(painted_rows):
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
            if metric_dir:
                corr_ci = np.stack([corr_los, corr_his], axis=-1)
                plot_metric_by_altitude_and_sigma(
                    metric_dir,
                    driver_col,
                    altitudes,
                    row_labels,
                    "correlation",
                    f"Pearson r({ylabel}, CO2)",
                    metric,
                    correlations,
                    duration_bins,
                    ci_values=corr_ci,
                )
                slope_ci = np.stack([slope_los, slope_his], axis=-1)
                plot_slope_count_summary_by_altitude_and_sigma(
                    metric_dir,
                    driver_col,
                    altitudes,
                    row_labels,
                    f"Linear fit slope ({ylabel} per CO2 ppm)",
                    metric,
                    slopes,
                    counts.astype(float),
                    duration_bins,
                    ci_values=slope_ci,
                )
                plot_metric_by_altitude_and_sigma(
                    metric_dir,
                    driver_col,
                    altitudes,
                    row_labels,
                    "slope",
                    f"Linear fit slope ({ylabel} per CO2 ppm)",
                    metric,
                    slopes,
                    duration_bins,
                    ci_values=slope_ci,
                )
                plot_metric_by_altitude_and_sigma(
                    metric_dir,
                    driver_col,
                    altitudes,
                    row_labels,
                    "error",
                    f"Linear fit RMSE ({ylabel})",
                    metric,
                    errors,
                    duration_bins,
                )
                plot_metric_by_altitude_and_sigma(
                    metric_dir,
                    driver_col,
                    altitudes,
                    row_labels,
                    "sample_count",
                    "Sample count n",
                    metric,
                    counts.astype(float),
                )

            max_count = int(counts.max()) if counts.size else 0
            count_threshold = max(1, int(np.ceil(0.01 * max_count)))
            display_correlations = np.array(correlations, copy=True)
            display_correlations[counts < count_threshold] = np.nan
            display_correlations[np.abs(display_correlations) < 0.1] = np.nan
            painted_rows = np.any(np.isfinite(display_correlations), axis=1)
            painted_cols = np.any(np.isfinite(display_correlations), axis=0)
            if not np.any(painted_rows) or not np.any(painted_cols):
                continue

            row_labels = row_labels[painted_rows]
            plotted_altitudes = list(np.array(altitudes)[painted_cols])
            correlations = correlations[np.ix_(painted_rows, painted_cols)]
            display_correlations = display_correlations[
                np.ix_(painted_rows, painted_cols)
            ]
            slopes = slopes[np.ix_(painted_rows, painted_cols)]
            zero_crossings = zero_crossings[np.ix_(painted_rows, painted_cols)]
            errors = errors[np.ix_(painted_rows, painted_cols)]
            slope_los = slope_los[np.ix_(painted_rows, painted_cols)]
            slope_his = slope_his[np.ix_(painted_rows, painted_cols)]
            counts = counts[np.ix_(painted_rows, painted_cols)]

            fig, ax = plt.subplots(
                figsize=(
                    max(8, 0.75 * len(plotted_altitudes)),
                    max(4, 0.5 * len(row_labels)),
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
                rasterized=True,
            )
            ax.set_xticks(
                np.arange(len(plotted_altitudes)),
                [f"{altitude} km" for altitude in plotted_altitudes],
                rotation=45,
                ha="right",
            )
            ax.set_yticks(np.arange(len(row_labels)), row_labels)
            ax.set_xlabel("Altitude")
            ax.set_ylabel(
                f"{SPACE_WEATHER_SIGMA_LABELS[driver_col]} bins\nmean={driver_mean:.2f}, sigma={driver_std:.2f}"
            )
            ax.set_xticks(np.arange(-0.5, len(plotted_altitudes), 1), minor=True)
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
                                errors[row, col],
                                slope_los[row, col],
                                slope_his[row, col],
                            ),
                            ha="center",
                            va="center",
                            fontsize=6,
                            color="white" if abs(value) > 0.55 else "black",
                        )
            fig.colorbar(image, ax=ax, label="Pearson r")
            ax.set_title(
                f"Mauna Loa {ylabel} vs CO2 correlation by {SPACE_WEATHER_SIGMA_LABELS[driver_col]} bin"
            )
            save_and_close(
                fig,
                heatmap_dir
                / f"maunaloa_density_{metric}_co2_correlation_by_{safe_name(driver_col)}.pgf",
            )


def regular_daily_series(df: pl.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
    series = (
        df.select("date", col)
        .drop_nulls()
        .group_by("date")
        .agg(pl.col(col).mean())
        .sort("date")
    )
    days = series["date"].cast(pl.Int64).to_numpy()
    values = series[col].to_numpy().astype(float)
    mask = np.isfinite(days) & np.isfinite(values)
    days, values = days[mask], values[mask]
    if len(days) < 3:
        return np.array([]), np.array([])
    full_days = np.arange(days[0], days[-1] + 1)
    full_values = np.interp(full_days, days, values)
    return full_days, full_values


def morlet_wavelet_power(
    df: pl.DataFrame,
    col: str,
    min_period_years: float = 0.25,
    max_period_years: float = 16,
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


def add_period_yticks(ax: plt.Axes) -> None:
    ticks = ax.get_yticks()
    ticks = ticks[(ticks > 0) & np.isfinite(ticks)]
    ticks = np.unique(np.concatenate([ticks, SPECIAL_PERIODS_YEARS]))
    ax.set_yticks(ticks)
    ax.set_yticklabels([period_label(tick) for tick in ticks])


def period_label(tick: float) -> str:
    if np.isclose(tick, 0.5):
        return "6 mo"
    if np.isclose(tick, 1.0):
        return "1 y"
    if np.isclose(tick, 11.0):
        return "11 y"
    return f"{tick:g} y"


def plot_wavelet(df: pl.DataFrame, altitudes: list[int]) -> None:
    cols = [
        density_mean_col(altitudes[len(altitudes) // 2]),
        density_range_col(altitudes[len(altitudes) // 2]),
        *SABER_COLS,
        *ANALYSIS_COLS,
    ]
    fig, axes = plt.subplots(
        len(cols),
        1,
        figsize=(13, 2.4 * len(cols)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for ax, col in zip(axes, cols):
        days, periods_years, power, coi = morlet_wavelet_power(df, col)
        if len(days) == 0:
            ax.set_ylabel(label_for_col(col))
            continue
        date_numbers = mdates.date2num(np.datetime64("1970-01-01")) + days
        mesh = ax.pcolormesh(
            date_numbers,
            periods_years,
            np.log10(power + np.finfo(float).eps),
            shading="auto",
            cmap="magma",
            rasterized=True,
        )
        ax.plot(date_numbers, coi, color="white", linewidth=1, alpha=0.8)
        ax.fill_between(
            date_numbers, coi, np.max(periods_years), color="white", alpha=0.12
        )
        ax.set_yscale("log")
        add_period_yticks(ax)
        ax.set_ylabel(label_for_col(col), fontsize=8)
        ax.grid(True, which="both", alpha=0.15)
        fig.colorbar(mesh, ax=ax, label="log10 power")
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlim(*date_xlim(df))
    plt.setp(axes[-1].get_xticklabels(), rotation=45, ha="right")
    axes[-1].set_xlabel("Date")
    fig.suptitle("Mauna Loa Morlet wavelet power spectra")
    save_and_close(fig, FFT_TIMESERIES_DIR / "maunaloa_wavelet.pgf")


def main() -> None:
    for directory in [OUTPUT_DIR, FFT_TIMESERIES_DIR, CORRELATION_DIR, HEATMAP_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    df, altitudes = load_dataset()
    selected_cols = [
        *ANALYSIS_COLS,
        *SABER_COLS,
        *[density_mean_col(alt) for alt in altitudes],
        *[density_range_col(alt) for alt in altitudes],
    ]
    all_altitude_df, all_altitudes = load_all_altitude_correlation_dataset(df)
    saber_heatmap_df, saber_heatmap_altitudes = load_saber_cooling_heatmap_dataset()
    major_altitudes = [
        altitude for altitude in MAJOR_HASDM_ALTITUDES if altitude in all_altitudes
    ]
    plot_time_series(
        all_altitude_df,
        major_altitudes,
        all_altitude_df,
        all_altitudes,
        saber_heatmap_df,
        saber_heatmap_altitudes,
    )
    plot_fft(
        df,
        altitudes,
        all_altitude_df,
        all_altitudes,
        saber_heatmap_df,
        saber_heatmap_altitudes,
    )
    plot_correlation_heatmap(
        df,
        [*ANALYSIS_COLS, *SABER_COLS],
        "maunaloa_correlation_analysis_variables.pgf",
        "Mauna Loa driver correlation",
    )
    plot_correlation_heatmap(
        df,
        selected_cols,
        "maunaloa_correlation_all_selected_variables.pgf",
        "Mauna Loa selected-variable correlation",
    )
    plot_scatter_matrix(
        df,
        [*ANALYSIS_COLS, *SABER_COLS],
        "maunaloa_scatter_analysis_variables.pgf",
        "Mauna Loa driver/cooling scatter matrix",
    )
    plot_density_scatter_by_altitude(df, altitudes)
    plot_correlation_by_altitude(all_altitude_df, all_altitudes)
    plot_paper_candidate_figure_3(all_altitude_df, all_altitudes)
    plot_correlation_by_altitude(
        all_altitude_df,
        all_altitudes,
        PAPER_CANDIDATE_FIGURE_4_CAUSES,
        "paper_candidate_figure_4_altitude_relationship_summary.pgf",
        "Paper candidate Figure 4: altitude relationship summary",
    )
    plot_density_co2_correlation_heatmaps(
        all_altitude_df,
        all_altitudes,
        HEATMAP_DIR / "density_co2_correlation_heatmaps_all_altitudes",
        False,
    )
    plot_density_co2_correlation_heatmaps(all_altitude_df, major_altitudes)
    plot_wavelet(df, altitudes)
    write_latex_figure_indexes()
    print(f"Generated Mauna Loa global-style figures in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
