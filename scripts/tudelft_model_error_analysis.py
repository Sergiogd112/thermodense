from __future__ import annotations
# Ruff: configure_pgf() must run before pyplot imports; suppress intentional E402.
# ruff: noqa: E402

from dataclasses import dataclass
from datetime import date
import math
from pathlib import Path

from scripts.pgf_config import configure_pgf, fig_size, page_fig_size

configure_pgf()

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.lines import Line2D
import numpy as np
import polars as pl
from astropy.timeseries import LombScargle

from thermodense.figure19 import create_figure_19
from thermodense.downloader.space_weather import SPACE_WEATHER_CSV_PATH  # noqa: E402
from scripts.stats_utils import ols_slope_ci, pearsonr_ci

DATA_ROOT = Path("data/analyzed/tudelft")
DOWNSAMPLED_ROOT = Path("data/analyzed_downsampled/tudelft")
OUTPUT_ROOT = Path("outputs/figures/results/tudelft_model_errors")

MODEL_COLS = {
    "NRLMSISE-00": "ln_density_ratio_0",
    "NRLMSIS 2.0": "ln_density_ratio_2.0",
    "NRLMSIS 2.1": "ln_density_ratio_2.1",
}
LAT = 19.5362
LON = 204.4237 - 360
LAT_RANGE = 10.0
LON_RANGE = 15.0

MISSION_ORDER = ["CH", "GO", "GA", "GB", "GC", "SA", "SB", "SC"]
SPACE_WEATHER_PATH = SPACE_WEATHER_CSV_PATH
CO2_PATH = Path("data/original/co2/co2_daily_mlo.csv")
MIN_SAMPLES_PER_HEATMAP_CELL = 20
CORRELATION_DURATION_STEP_YEARS = 11
MIN_CORRELATION_DURATION_YEARS = CORRELATION_DURATION_STEP_YEARS
CORRELATION_DURATION_MARKERS = {
    0: "o",
    11: "s",
    22: "D",
    33: "^",
    44: "P",
    55: "X",
}
SPECTRUM_GRID_SIZE = 1600
SPECTRUM_LOG_Y_MIN = 1e-4
FREQUENCY_HEATMAP_ALTITUDE_BIN_KM = 10
CO2_HEATMAP_ALTITUDE_BIN_KM = 50
FREQUENCY_HEATMAP_PERIOD_BINS_PER_DECADE = 24


DRIVER_COLS = {
    "f107_81d": "$F_{10.7,81}$",
    "ap": "$A_p$",
    "kp": "$K_p$",
    "co2": "CO$_2$",
}
ACTIVITY_DRIVERS = ["f107_81d", "ap", "kp"]
CORRELATION_DRIVERS = ["f107_81d", "ap", "kp", "co2", "altitude_km"]
CORRELATION_LABELS = {
    **DRIVER_COLS,
    "altitude_km": "Altitude",
}


@dataclass(frozen=True)
class MissionConfig:
    code: str
    name: str
    path: Path
    downsampled_path: Path
    altitude_range_km: tuple[float, float]


MISSIONS = {
    "CH": MissionConfig(
        "CH",
        "CHAMP",
        DATA_ROOT / "champ" / "CH_analyzed.parquet",
        DOWNSAMPLED_ROOT / "champ" / "CH_analyzed_downsampled.parquet",
        (300, 500),
    ),
    "GO": MissionConfig(
        "GO",
        "GOCE",
        DATA_ROOT / "goce" / "GO_analyzed.parquet",
        DOWNSAMPLED_ROOT / "goce" / "GO_analyzed_downsampled.parquet",
        (225, 300),
    ),
    "GA": MissionConfig(
        "GA",
        "GRACE-A",
        DATA_ROOT / "grace" / "GA_analyzed.parquet",
        DOWNSAMPLED_ROOT / "grace" / "GA_analyzed_downsampled.parquet",
        (425, 500),
    ),
    "GB": MissionConfig(
        "GB",
        "GRACE-B",
        DATA_ROOT / "grace" / "GB_analyzed.parquet",
        DOWNSAMPLED_ROOT / "grace" / "GB_analyzed_downsampled.parquet",
        (425, 500),
    ),
    "GC": MissionConfig(
        "GC",
        "GRACE-FO",
        DATA_ROOT / "grace_fo" / "GC_analyzed.parquet",
        DOWNSAMPLED_ROOT / "grace_fo" / "GC_analyzed_downsampled.parquet",
        (475, 525),
    ),
    "SA": MissionConfig(
        "SA",
        "Swarm-A",
        DATA_ROOT / "swarm" / "SA_analyzed.parquet",
        DOWNSAMPLED_ROOT / "swarm" / "SA_analyzed_downsampled.parquet",
        (425, 500),
    ),
    "SB": MissionConfig(
        "SB",
        "Swarm-B",
        DATA_ROOT / "swarm" / "SB_analyzed.parquet",
        DOWNSAMPLED_ROOT / "swarm" / "SB_analyzed_downsampled.parquet",
        (500, 525),
    ),
    "SC": MissionConfig(
        "SC",
        "Swarm-C",
        DATA_ROOT / "swarm" / "SC_analyzed.parquet",
        DOWNSAMPLED_ROOT / "swarm" / "SC_analyzed_downsampled.parquet",
        (450, 525),
    ),
}


def safe_name(value: str) -> str:
    return value.lower().replace(" ", "_").replace(".", "p").replace("-", "_")


def load_space_weather_drivers() -> pl.DataFrame:
    return (
        pl.read_csv(SPACE_WEATHER_PATH)
        .with_columns(pl.col("DATE").str.to_date("%Y-%m-%d").alias("date"))
        .select(
            "date",
            pl.col("F10.7_OBS_CENTER81").alias("f107_81d"),
            pl.col("AP_AVG").alias("ap"),
            pl.col("KP_SUM").alias("kp"),
        )
    )


def load_co2() -> pl.DataFrame:
    schema = {
        "year": pl.Int32,
        "month": pl.Int32,
        "day": pl.Int32,
        "year_decimal": pl.Float32,
        "co2": pl.Float64,
    }
    return (
        pl.read_csv(CO2_PATH, has_header=False, schema=schema, comment_prefix="#")
        .with_columns(pl.date("year", "month", "day").alias("date"))
        .select("date", "co2")
        .with_columns(
            pl.when(pl.col("co2") < 0).then(None).otherwise(pl.col("co2")).alias("co2")
        )
        .drop_nulls("co2")
        .sort("date")
    )


def load_driver_data() -> pl.DataFrame:
    return (
        load_space_weather_drivers()
        .join(load_co2(), on="date", how="left")
        .sort("date")
    )


def output_path(*parts: str) -> Path:
    path = OUTPUT_ROOT.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def figure_caption(figure: Path) -> str:
    caption = figure.stem.replace("_", " ").replace("co2", "CO2").title()
    replacements = {
        "Nrlmsise 00": "NRLMSISE-00",
        "Nrlmsis 2P0": "NRLMSIS 2.0",
        "Nrlmsis 2P1": "NRLMSIS 2.1",
        "F107 81D": "F10.7 81-day",
        "F107": "F10.7",
        "Kp": "Kp",
        "Ap": "Ap",
        "Tudelft": "TuDelft",
        "Co2": "CO2",
        "Figure19": "Figure 19",
    }
    for old, new in replacements.items():
        caption = caption.replace(old, new)
    return caption


def index_title(rel_dir: Path) -> str:
    if rel_dir == Path("."):
        return "TuDelft Model-Error Figures"
    title = str(rel_dir).replace("_", " ").title()
    replacements = {
        "Tudelft": "TuDelft",
        "Co2": "CO2",
        "Figure19": "Figure 19",
    }
    for old, new in replacements.items():
        title = title.replace(old, new)
    return title


def load_mission(
    config: MissionConfig, *, figure19: bool = False, downsampled: bool = True
) -> pl.DataFrame:
    min_alt, max_alt = config.altitude_range_km
    source_path = config.downsampled_path if downsampled else config.path
    df = pl.read_parquet(source_path, low_memory=True).sort("timestamp")
    quality_filter = (pl.col("Altitude (m)") / 1000.0).is_between(min_alt, max_alt)

    if (
        config.name != "Swarm-A"
        and config.name != "Swarm-B"
        and config.name != "Swarm-C"
    ):
        if "Anomalus Density (kg/m^3)" in df.columns:
            quality_filter = quality_filter & (pl.col("Anomalus Density (kg/m^3)") == 0)
        if "Anomalus Density Mean (kg/m^3)" in df.columns:
            quality_filter = quality_filter & (
                pl.col("Anomalus Density Mean (kg/m^3)") == 0
            )

    if config.name == "GOCE" and "Degraded Flag Thrusters" in df.columns:
        quality_filter = quality_filter & (pl.col("Degraded Flag Thrusters") == 0)

    if figure19 and config.code == "CH":
        quality_filter = quality_filter & (
            (pl.col("timestamp") < date(2006, 1, 1))
            | (pl.col("timestamp") > date(2009, 12, 31))
        )

    df = df.filter(quality_filter)
    joined_driver_cols = [col for col in DRIVER_COLS if col in df.columns]
    if joined_driver_cols:
        df = df.drop(joined_driver_cols)
    df = (
        df.with_columns(pl.col("timestamp").dt.date().alias("date"))
        .join(load_driver_data(), on="date", how="left")
        .drop("date")
    )

    cols = [
        "timestamp",
        "Density (kg/m^3)",
        "Altitude (m)",
        "Latitude (deg)",
        "Longitude (deg)",
        "f107",
        "f107a",
        "ap",
        "kp",
        "f107_81d",
        "co2",
        *MODEL_COLS.values(),
    ]
    return df.select([col for col in cols if col in df.columns]).drop_nulls(
        list(MODEL_COLS.values())
    )


def load_all(
    *, figure19: bool = False, downsampled: bool = True
) -> list[tuple[MissionConfig, pl.DataFrame]]:
    return [
        (
            MISSIONS[code],
            load_mission(MISSIONS[code], figure19=figure19, downsampled=downsampled),
        )
        for code in MISSION_ORDER
    ]


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
            if path.suffix.lower() in {".pgf", ".pdf", ".jpg", ".png"}
        )
        child_indexes = sorted(
            path / "figures.tex"
            for path in directory.iterdir()
            if path.is_dir() and (path / "figures.tex").exists()
        )
        if not figures and not child_indexes:
            continue
        rel_dir = directory.relative_to(root)
        title = index_title(rel_dir)
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


def save_figure(fig: plt.Figure, *parts: str) -> None:
    out = output_path(*parts)
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
        borderaxespad=0.1,
    )
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        rect = (
            (0.0, 0.0, 1.0, 0.91) if loc.startswith("upper") else (0.0, 0.11, 1.0, 1.0)
        )
        layout_engine.set(rect=rect)


def add_density_reference_uncertainty_bands(
    fig: plt.Figure, missions: list[tuple[MissionConfig, pl.DataFrame]]
) -> None:
    """Add one conservative reference-density uncertainty scale per mission row."""
    uncertainty_scales = {
        "CH": (0.15, r"15\%"),
        "GO": (0.07, r"7\%"),
        "GA": (0.20, r"20\%"),
        "GB": (0.20, r"20\%"),
        "GC": (0.20, r"20\%"),
        "SA": (0.50, r"50\%"),
        "SB": (0.50, r"$\geq 50\%$"),
        "SC": (0.50, r"50\%"),
    }
    axes = np.asarray(fig.axes[: len(missions) * 3]).reshape(len(missions), 3)
    band_color = "#8ebad1"
    edge_color = "#477b96"
    for row, (config, _) in enumerate(missions):
        relative_uncertainty, display_value = uncertainty_scales[config.code]
        log_scale = math.log1p(relative_uncertainty)
        for ax in axes[row]:
            ax.axhspan(
                -log_scale,
                log_scale,
                color=band_color,
                alpha=0.18,
                zorder=0,
            )
            ax.axhline(log_scale, color=edge_color, linewidth=0.9, alpha=0.9, zorder=1)
            ax.axhline(-log_scale, color=edge_color, linewidth=0.9, alpha=0.9, zorder=1)
        axes[row, 0].text(
            0.03,
            0.05,
            rf"upper $1\sigma$ $u_{{\rho,\mathrm{{ref}}}}$: {display_value}",
            transform=axes[row, 0].transAxes,
            fontsize=6.5,
            color="#315d72",
            ha="left",
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1.5},
            zorder=10,
        )


def figure19(missions: list[tuple[MissionConfig, pl.DataFrame]]) -> None:
    dfs = [df for _, df in missions]
    names = []
    for config, df in missions:
        start = df["timestamp"].min().date().year
        end = df["timestamp"].max().date().year
        min_alt, max_alt = config.altitude_range_km
        if config.code == "CH":
            names.append(
                f"{config.name}\n{min_alt:.0f}-{max_alt:.0f} km, {start % 100}-{end % 100}\n(excl. 06-09)"
            )
        else:
            names.append(
                f"{config.name}\n{min_alt:.0f}-{max_alt:.0f} km, {start % 100}-{end % 100}"
            )
    fig = create_figure_19(
        dfs=dfs,
        mission_names=names,
        msis_00_col=MODEL_COLS["NRLMSISE-00"],
        msis_20_col=MODEL_COLS["NRLMSIS 2.0"],
        msis_21_col=MODEL_COLS["NRLMSIS 2.1"],
        matlab_col=None,
        errorbar_mode="uncertainty_of_mean",
        figsize=(8, 12),
    )
    # fig.suptitle(
    #     "TU Delft recreation of Emmert et al. Figure 19\n"
    #     "with density-reference uncertainty scales",
    #     y=1.015,
    # )
    add_density_reference_uncertainty_bands(fig, missions)
    fig.savefig(
        output_path("figure19", "tudelft_figure19_all_missions.pdf"),
        bbox_inches="tight",
    )
    save_figure(fig, "figure19", "tudelft_figure19_all_missions.pgf")


def daily_summary(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.group_by_dynamic(
            "timestamp", every="1d", period="1d", closed="left", label="left"
        )
        .agg(
            (pl.col("Altitude (m)").mean() / 1000.0).alias("altitude_km"),
            pl.col("f107a").mean().alias("f107a"),
            pl.col("f107_81d").mean().alias("f107_81d"),
            pl.col("ap").mean().alias("ap"),
            pl.col("kp").mean().alias("kp"),
            pl.col("co2").mean().alias("co2"),
            *[pl.col(col).mean().alias(col) for col in MODEL_COLS.values()],
        )
        .drop_nulls(list(MODEL_COLS.values()))
        .sort("timestamp")
    )


def plot_combined_timeseries(
    missions: list[tuple[MissionConfig, pl.DataFrame]],
) -> None:
    summaries = [(config, daily_summary(df)) for config, df in missions]
    colors = plt.get_cmap("tab10", len(summaries))
    fig, axes = plt.subplots(
        len(MODEL_COLS) * 2 + 1,
        1,
        figsize=fig_size(1.0, 1.35),
        sharex=True,
        constrained_layout=True,
    )
    for model_idx, (model_name, col) in enumerate(MODEL_COLS.items()):
        ax = axes[model_idx * 2]
        for idx, (config, daily) in enumerate(summaries):
            ax.plot(
                daily["timestamp"],
                daily[col],
                linewidth=0.8,
                alpha=0.85,
                color=colors(idx),
                label=config.name,
            )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel(r"$\epsilon_m$")
        ax.set_title(model_name, loc="left")
        ax.grid(True, alpha=0.25)
        plot_mission_altitude_time_heatmap(
            axes[model_idx * 2 + 1], summaries, col, f"{model_name} error by altitude"
        )
    alt_ax = axes[-1]
    for idx, (config, daily) in enumerate(summaries):
        alt_ax.plot(
            daily["timestamp"],
            daily["altitude_km"],
            linewidth=0.8,
            alpha=0.85,
            color=colors(idx),
            label=config.name,
        )
    alt_ax.set_ylabel("Altitude (km)")
    alt_ax.set_xlabel("Date")
    alt_ax.grid(True, alpha=0.25)
    alt_ax.xaxis.set_major_locator(mdates.YearLocator(2))
    alt_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    handles, labels = axes[0].get_legend_handles_labels()
    add_outside_legend(fig, handles, labels, loc="upper center", ncols=4)
    save_figure(fig, "timeseries", "tudelft_model_error_timeseries_combined.pgf")


def plot_per_mission_timeseries(
    missions: list[tuple[MissionConfig, pl.DataFrame]],
) -> None:
    for config, df in missions:
        daily = daily_summary(df)
        fig, axes = plt.subplots(
            2, 1, figsize=fig_size(1.0, 0.72), sharex=True, constrained_layout=True
        )
        for model_name, col in MODEL_COLS.items():
            axes[0].plot(
                daily["timestamp"], daily[col], linewidth=0.9, label=model_name
            )
        axes[0].axhline(0, color="black", linewidth=0.8)
        axes[0].set_ylabel(r"$\epsilon_m$")
        handles, labels = axes[0].get_legend_handles_labels()
        add_outside_legend(fig, handles, labels, loc="upper center", ncols=3)
        axes[0].grid(True, alpha=0.25)
        axes[1].plot(
            daily["timestamp"], daily["altitude_km"], color="dimgray", linewidth=0.9
        )
        axes[1].set_ylabel("Altitude (km)")
        axes[1].set_xlabel("Date")
        axes[1].grid(True, alpha=0.25)
        axes[1].xaxis.set_major_locator(mdates.YearLocator(2))
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.suptitle(f"{config.name} model log-density-ratio errors")
        save_figure(
            fig,
            "timeseries",
            "per_mission",
            f"{safe_name(config.name)}_model_error_timeseries.pgf",
        )


def lomb_scargle_spectrum(
    df: pl.DataFrame,
    col: str,
    *,
    time_col: str = "timestamp",
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    data = df.select(time_col, col).drop_nulls().sort(time_col)
    if data.height < 20:
        return np.array([]), np.array([])
    times = (
        data[time_col].cast(pl.Datetime).dt.epoch("s").to_numpy().astype(float)
        / 86400.0
    )
    times = times - times.min()
    values = data[col].to_numpy().astype(float)
    mask = np.isfinite(times) & np.isfinite(values)
    times = times[mask]
    values = values[mask] - np.mean(values[mask])
    if len(times) < 20 or np.std(values) == 0:
        return np.array([]), np.array([])
    if normalize:
        values = values / np.std(values)
    baseline = times.max() - times.min()
    if baseline <= 1:
        return np.array([]), np.array([])
    min_frequency = 1.0 / baseline
    median_dt = np.median(np.diff(np.unique(times)))
    max_frequency = 0.5 / median_dt if np.isfinite(median_dt) and median_dt > 0 else 2.0
    max_frequency = min(max_frequency, 12.0)
    frequency = np.geomspace(min_frequency, max_frequency, SPECTRUM_GRID_SIZE)
    power = LombScargle(times, values).power(frequency)
    if normalize and np.nanmax(power) > 0:
        power = power / np.nanmax(power)
    return 1.0 / frequency, power


def positive_power(power: np.ndarray) -> np.ndarray:
    out = np.asarray(power, dtype=float).copy()
    out[out <= 0] = np.nan
    return out


def log_period_edges(periods_years: np.ndarray) -> np.ndarray:
    finite = periods_years[np.isfinite(periods_years) & (periods_years > 0)]
    if len(finite) == 0:
        return np.array([])
    span_decades = np.log10(np.max(finite)) - np.log10(np.min(finite))
    n_bins = max(
        24, int(np.ceil(span_decades * FREQUENCY_HEATMAP_PERIOD_BINS_PER_DECADE))
    )
    return np.logspace(np.log10(np.min(finite)), np.log10(np.max(finite)), n_bins + 1)


def altitude_edges_for_missions(
    missions: list[tuple[MissionConfig, pl.DataFrame]],
) -> np.ndarray:
    altitudes = []
    for _, df in missions:
        if "altitude_km" in df.columns:
            values = df["altitude_km"].to_numpy().astype(float)
            altitudes.extend(values[np.isfinite(values)])
        elif "Altitude (m)" in df.columns:
            values = (df["Altitude (m)"] / 1000.0).to_numpy().astype(float)
            altitudes.extend(values[np.isfinite(values)])
    if not altitudes:
        return np.array([])
    lower = (
        np.floor(np.nanmin(altitudes) / FREQUENCY_HEATMAP_ALTITUDE_BIN_KM)
        * FREQUENCY_HEATMAP_ALTITUDE_BIN_KM
    )
    upper = (
        np.ceil(np.nanmax(altitudes) / FREQUENCY_HEATMAP_ALTITUDE_BIN_KM)
        * FREQUENCY_HEATMAP_ALTITUDE_BIN_KM
    )
    return np.arange(
        lower,
        upper + FREQUENCY_HEATMAP_ALTITUDE_BIN_KM,
        FREQUENCY_HEATMAP_ALTITUDE_BIN_KM,
    )


def binned_lomb_scargle_heatmap(
    missions: list[tuple[MissionConfig, pl.DataFrame]],
    model_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spectra = []
    all_periods = []
    for _, df in missions:
        if model_col not in df.columns:
            continue
        if "altitude_km" in df.columns:
            altitude = float(np.nanmedian(df["altitude_km"].to_numpy().astype(float)))
        elif "Altitude (m)" in df.columns:
            altitude = float(
                np.nanmedian((df["Altitude (m)"] / 1000.0).to_numpy().astype(float))
            )
        else:
            continue
        periods, power = lomb_scargle_spectrum(df, model_col)
        if len(periods) == 0:
            continue
        periods_years = periods / 365.25
        spectra.append((altitude, periods_years, positive_power(power)))
        all_periods.extend(periods_years)

    if not spectra:
        return np.array([]), np.array([]), np.array([[]])
    altitude_edges = altitude_edges_for_missions(missions)
    if len(altitude_edges) < 2:
        return np.array([]), np.array([]), np.array([[]])
    period_edges = log_period_edges(np.asarray(all_periods, dtype=float))
    if len(period_edges) < 2:
        return np.array([]), np.array([]), np.array([[]])
    period_centers = np.sqrt(period_edges[:-1] * period_edges[1:])
    mission_matrix = np.full((len(spectra), len(period_centers)), np.nan)
    target_log_periods = np.log10(period_centers)
    spectrum_altitudes = np.array([altitude for altitude, _, _ in spectra], dtype=float)
    for row, (_, periods_years, power) in enumerate(spectra):
        mask = (
            np.isfinite(periods_years)
            & np.isfinite(power)
            & (periods_years > 0)
            & (power > 0)
        )
        if np.sum(mask) < 2:
            continue
        order = np.argsort(periods_years[mask])
        source_log_periods = np.log10(periods_years[mask][order])
        source_log_power = np.log10(power[mask][order])
        mission_matrix[row, :] = np.interp(
            target_log_periods,
            source_log_periods,
            source_log_power,
            left=np.nan,
            right=np.nan,
        )

    matrix = np.full((len(altitude_edges) - 1, len(period_centers)), np.nan)
    for altitude_idx in range(len(altitude_edges) - 1):
        lower = altitude_edges[altitude_idx]
        upper = altitude_edges[altitude_idx + 1]
        if altitude_idx == len(altitude_edges) - 2:
            mask = (spectrum_altitudes >= lower) & (spectrum_altitudes <= upper)
        else:
            mask = (spectrum_altitudes >= lower) & (spectrum_altitudes < upper)
        if np.any(mask):
            subset = mission_matrix[mask, :]
            valid_cols = np.any(np.isfinite(subset), axis=0)
            if np.any(valid_cols):
                matrix[altitude_idx, valid_cols] = np.nanmean(
                    subset[:, valid_cols], axis=0
                )
    return period_edges, altitude_edges, matrix


def period_label(days: float) -> str:
    if days < 60:
        return f"{days:g} d"
    years = days / 365.25
    if years < 2:
        return f"{years:.1f} y"
    return f"{years:.0f} y"


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


def plot_mission_altitude_time_heatmap(
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
    if not frames:
        ax.text(
            0.5,
            0.5,
            "No time heatmap data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return
    combined = (
        pl.concat(frames)
        .drop_nulls(["timestamp", "altitude_km", value_col])
        .sort("timestamp")
    )
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
    max_abs = np.nanmax(np.abs(matrix)) if np.any(np.isfinite(matrix)) else 1.0
    mesh = ax.pcolormesh(
        date_edges_for_heatmap(dates),
        altitude_edges,
        matrix,
        shading="auto",
        cmap="coolwarm",
        vmin=-max_abs,
        vmax=max_abs,
        rasterized=True,
    )
    ax.xaxis_date()
    ax.set_ylabel("Altitude (km)")
    ax.set_title(title, loc="left")
    ax.grid(True, alpha=0.18)
    ax.figure.colorbar(mesh, ax=ax, label=r"$\epsilon_m=\ln(\rho_m/\rho_\mathrm{ref})$")


def add_period_reference_lines(ax: plt.Axes) -> None:
    for period in np.array([7, 27, 183, 365.25, 365.25 * 11]):
        ax.axvline(period / 365.25, color="gray", linewidth=0.7, alpha=0.35)
    ax.set_xscale("log")
    ax.set_xticks(np.array([7, 27, 183, 365.25, 365.25 * 11]) / 365.25)
    ax.set_xticklabels(
        [period_label(day) for day in [7, 27, 183, 365.25, 365.25 * 11]],
        rotation=45,
        ha="right",
    )


def plot_frequency_overview(
    missions: list[tuple[MissionConfig, pl.DataFrame]],
    model_cols: dict[str, str] | None = None,
    filename: str = "tudelft_model_error_driver_spectra.pgf",
) -> None:
    model_cols = MODEL_COLS if model_cols is None else model_cols
    summaries = [(config, daily_summary(df)) for config, df in missions]
    n_model_rows = len(model_cols)
    driver_cols = [col for col in DRIVER_COLS if col != "kp"]
    n_panels = 2 * n_model_rows + len(driver_cols)
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=page_fig_size(1.0, 0.242 * n_panels, max_height_scale=0.86),
        constrained_layout=True,
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    mission_colors = plt.get_cmap("tab10", len(summaries))

    for row_idx, (model_name, model_col) in enumerate(model_cols.items()):
        ax = axes[2 * row_idx]
        for mission_idx, (config, df) in enumerate(summaries):
            periods, power = lomb_scargle_spectrum(df, model_col)
            if len(periods):
                ax.plot(
                    periods / 365.25,
                    positive_power(power),
                    linewidth=0.8,
                    alpha=0.85,
                    color=mission_colors(mission_idx),
                    label=config.name,
                )
        add_period_reference_lines(ax)
        ax.set_yscale("log")
        ax.set_ylim(SPECTRUM_LOG_Y_MIN, 1.05)
        ax.set_title("Model-error spectra")
        ax.set_ylabel("Norm. power")
        ax.grid(True, which="both", alpha=0.25)

        heatmap_ax = axes[2 * row_idx + 1]
        period_edges, altitude_edges, log_power = binned_lomb_scargle_heatmap(
            summaries, model_col
        )
        if len(period_edges) and len(altitude_edges):
            mesh = heatmap_ax.pcolormesh(
                period_edges,
                altitude_edges,
                log_power,
                shading="auto",
                cmap="magma",
                rasterized=True,
            )
            fig.colorbar(mesh, ax=heatmap_ax, label="log$_{10}$ normalized power")
        else:
            heatmap_ax.text(
                0.5,
                0.5,
                "No heatmap data",
                ha="center",
                va="center",
                transform=heatmap_ax.transAxes,
            )
        add_period_reference_lines(heatmap_ax)
        heatmap_ax.set_yscale("linear")
        heatmap_ax.set_title("Model-error power by altitude")
        heatmap_ax.set_ylabel("Altitude (km)")
        heatmap_ax.grid(True, which="both", alpha=0.18)

    start = min(df["timestamp"].min().date() for _, df in summaries)
    end = max(df["timestamp"].max().date() for _, df in summaries)
    drivers = load_driver_data().filter(
        (pl.col("date") >= start) & (pl.col("date") <= end)
    )
    driver_axes = axes[2 * n_model_rows :]
    for ax, driver_col in zip(driver_axes, driver_cols):
        periods, power = lomb_scargle_spectrum(drivers, driver_col, time_col="date")
        if len(periods):
            ax.plot(
                periods / 365.25, positive_power(power), linewidth=1.2, color="black"
            )
        add_period_reference_lines(ax)
        ax.set_yscale("log")
        ax.set_ylim(SPECTRUM_LOG_Y_MIN, 1.05)
        ax.set_title(f"{DRIVER_COLS[driver_col]} spectrum")
        ax.set_ylabel(f"Norm. power\n{DRIVER_COLS[driver_col]}")
        ax.grid(True, which="both", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.965),
            ncols=min(4, len(handles)),
            fontsize=7,
            borderaxespad=0,
        )
        layout_engine = fig.get_layout_engine()
        if layout_engine is not None:
            layout_engine.set(rect=(0.0, 0.0, 1.0, 0.86))
    axes[-1].set_xlabel("Period")
    model_label = ", ".join(model_cols)
    fig.suptitle(
        f"TU Delft {model_label} model-error and driver Lomb-Scargle spectra", y=0.995
    )
    save_figure(fig, "frequency_domain", filename)


def plot_spectra(missions: list[tuple[MissionConfig, pl.DataFrame]]) -> None:
    reference_days = np.array([7, 27, 183, 365.25, 365.25 * 11])
    for config, df in missions:
        fig, ax = plt.subplots(figsize=fig_size(1.0, 0.62), constrained_layout=True)
        for model_name, col in MODEL_COLS.items():
            periods, power = lomb_scargle_spectrum(df, col, normalize=False)
            if len(periods):
                ax.plot(periods / 365.25, power, linewidth=1.0, label=model_name)
        for period in reference_days:
            ax.axvline(period / 365.25, color="gray", linewidth=0.7, alpha=0.35)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Period")
        ax.set_ylabel("Lomb-Scargle power")
        ax.set_title(f"{config.name} full-resolution model-error spectra")
        ax.set_xticks(reference_days / 365.25)
        ax.set_xticklabels(
            [period_label(day) for day in reference_days], rotation=45, ha="right"
        )
        ax.grid(True, which="both", alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        add_outside_legend(fig, handles, labels, loc="upper center", ncols=3)
        save_figure(
            fig,
            "frequency_domain",
            f"{safe_name(config.name)}_lomb_scargle_model_errors.pgf",
        )


def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 3:
        return np.nan
    x = x[mask] - np.mean(x[mask])
    y = y[mask] - np.mean(y[mask])
    denom = np.sqrt(np.sum(x**2) * np.sum(y**2))
    return float(np.sum(x * y) / denom) if denom else np.nan


def valid_duration_years(df: pl.DataFrame, *cols: str) -> float:
    valid = df.filter(
        pl.all_horizontal(
            [pl.col(col).is_not_null() & pl.col(col).is_finite() for col in cols]
        )
    ).select("timestamp")
    if valid.height < 2:
        return np.nan
    return (valid["timestamp"].max() - valid["timestamp"].min()).days / 365.2425


def trim_empty_edges(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if np.any(rows):
        first_row = int(np.argmax(rows))
        last_row = len(rows) - int(np.argmax(rows[::-1]))
        visible_rows = np.zeros_like(rows, dtype=bool)
        visible_rows[first_row:last_row] = True
    else:
        visible_rows = rows
    if np.any(cols):
        first_col = int(np.argmax(cols))
        last_col = len(cols) - int(np.argmax(cols[::-1]))
        visible_cols = np.zeros_like(cols, dtype=bool)
        visible_cols[first_col:last_col] = True
    else:
        visible_cols = cols
    return visible_rows, visible_cols


def correlation_tables(
    missions: list[tuple[MissionConfig, pl.DataFrame]],
) -> pl.DataFrame:
    rows = []
    for config, df in missions:
        daily = daily_summary(df)
        for model_name, col in MODEL_COLS.items():
            for driver in CORRELATION_DRIVERS:
                rows.append(
                    {
                        "mission": config.name,
                        "model": model_name,
                        "driver": driver,
                        "pearson_r": pearsonr(
                            daily[col].to_numpy(), daily[driver].to_numpy()
                        ),
                        "n_days": daily.height,
                        "duration_years": valid_duration_years(daily, col, driver),
                    }
                )
    table = pl.DataFrame(rows)
    output_path("correlation", "tudelft_model_error_correlations.csv")
    table.write_csv(output_path("correlation", "tudelft_model_error_correlations.csv"))
    return table


def plot_correlation_summary(
    table: pl.DataFrame,
    models: dict[str, str] | None = None,
    filename: str = "all_models_correlation_summary.png",
    figsize: tuple[float, float] | None = None,
) -> None:
    models = MODEL_COLS if models is None else models
    missions = [MISSIONS[code].name for code in MISSION_ORDER]
    fig, axes = plt.subplots(
        len(models),
        1,
        figsize=fig_size(0.82, 1.42) if figsize is None else figsize,
        constrained_layout=True,
        sharex=True,
    )
    axes = np.asarray(axes).reshape(len(models))
    model_matrices = []
    for model_name in models:
        matrix = np.full((len(CORRELATION_DRIVERS), len(missions)), np.nan)
        for row, driver in enumerate(CORRELATION_DRIVERS):
            for col, mission in enumerate(missions):
                cell = table.filter(
                    (pl.col("model") == model_name)
                    & (pl.col("driver") == driver)
                    & (pl.col("mission") == mission)
                )
                if (
                    cell.height
                    and cell["duration_years"][0] >= MIN_CORRELATION_DURATION_YEARS
                ):
                    matrix[row, col] = cell["pearson_r"][0]
        model_matrices.append(matrix)
    visible_rows, visible_cols = trim_empty_edges(
        np.any(np.isfinite(np.stack(model_matrices)), axis=0)
    )
    if not np.any(visible_rows) or not np.any(visible_cols):
        plt.close(fig)
        return
    visible_missions = np.array(missions)[visible_cols]
    visible_drivers = np.array(CORRELATION_DRIVERS)[visible_rows]
    image = None
    for model_idx, (model_name, matrix) in enumerate(zip(models, model_matrices)):
        ax = axes[model_idx]
        matrix = matrix[np.ix_(visible_rows, visible_cols)]
        image = ax.imshow(
            matrix, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto", rasterized=True
        )
        ax.set_xticks(
            np.arange(len(visible_missions)),
            visible_missions,
            rotation=45,
            ha="right",
            fontsize=9,
        )
        ax.set_yticks(
            np.arange(len(visible_drivers)),
            [CORRELATION_LABELS[driver] for driver in visible_drivers],
            fontsize=9,
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
                        fontsize=7.5,
                        color="white" if abs(matrix[row, col]) > 0.55 else "black",
                    )
        ax.set_title(model_name, fontsize=11.5)
    colorbar = fig.colorbar(image, ax=axes, label="Pearson r", shrink=0.82)
    colorbar.ax.tick_params(labelsize=9)
    colorbar.ax.yaxis.label.set_size(9.5)
    save_figure(fig, "correlation", filename)


def altitude_correlation_table(
    missions: list[tuple[MissionConfig, pl.DataFrame]], altitude_bin_km: int = 25
) -> pl.DataFrame:
    combined = pl.concat(
        [
            daily_summary(df).with_columns(pl.lit(config.name).alias("mission"))
            for config, df in missions
        ],
        how="vertical_relaxed",
    )
    altitude = combined["altitude_km"].to_numpy().astype(float)
    alt_edges = np.arange(
        np.floor(np.nanmin(altitude) / altitude_bin_km) * altitude_bin_km,
        np.ceil(np.nanmax(altitude) / altitude_bin_km) * altitude_bin_km
        + altitude_bin_km,
        altitude_bin_km,
    )
    rows = []
    for alt_idx in range(len(alt_edges) - 1):
        altitude_filter = (pl.col("altitude_km") >= alt_edges[alt_idx]) & (
            pl.col("altitude_km") < alt_edges[alt_idx + 1]
        )
        cell = combined.filter(altitude_filter)
        if cell.height < MIN_SAMPLES_PER_HEATMAP_CELL:
            continue
        for model_name, col in MODEL_COLS.items():
            for driver in CORRELATION_DRIVERS[:-1]:
                rows.append(
                    {
                        "altitude_min_km": float(alt_edges[alt_idx]),
                        "altitude_max_km": float(alt_edges[alt_idx + 1]),
                        "model": model_name,
                        "driver": driver,
                        "pearson_r": pearsonr(
                            cell[col].to_numpy(), cell[driver].to_numpy()
                        ),
                        "n_days": cell.height,
                        "duration_years": valid_duration_years(cell, col, driver),
                    }
                )
    table = pl.DataFrame(rows)
    table.write_csv(
        output_path("correlation", "tudelft_model_error_correlations_by_altitude.csv")
    )
    return table


def plot_altitude_correlation_summary(
    table: pl.DataFrame,
    models: dict[str, str] | None = None,
    filename: str = "all_models_correlation_by_altitude_summary.png",
    figsize: tuple[float, float] | None = None,
) -> None:
    models = MODEL_COLS if models is None else models
    altitude_bins = (
        table.select("altitude_min_km", "altitude_max_km")
        .unique()
        .sort("altitude_min_km")
    )
    labels = [
        f"{row['altitude_min_km']:.0f}-{row['altitude_max_km']:.0f}"
        for row in altitude_bins.iter_rows(named=True)
    ]
    drivers = CORRELATION_DRIVERS[:-1]
    fig, axes = plt.subplots(
        len(models),
        1,
        figsize=fig_size(0.9, 1.32) if figsize is None else figsize,
        constrained_layout=True,
        sharex=True,
    )
    axes = np.asarray(axes).reshape(len(models))
    model_matrices = []
    for model_name in models:
        matrix = np.full((len(drivers), len(labels)), np.nan)
        for row, driver in enumerate(drivers):
            for col, alt_row in enumerate(altitude_bins.iter_rows(named=True)):
                cell = table.filter(
                    (pl.col("model") == model_name)
                    & (pl.col("driver") == driver)
                    & (pl.col("altitude_min_km") == alt_row["altitude_min_km"])
                    & (pl.col("altitude_max_km") == alt_row["altitude_max_km"])
                )
                if (
                    cell.height
                    and cell["duration_years"][0] >= MIN_CORRELATION_DURATION_YEARS
                ):
                    matrix[row, col] = cell["pearson_r"][0]
        model_matrices.append(matrix)
    visible_rows, visible_cols = trim_empty_edges(
        np.any(np.isfinite(np.stack(model_matrices)), axis=0)
    )
    if not np.any(visible_rows) or not np.any(visible_cols):
        plt.close(fig)
        return
    visible_labels = np.array(labels)[visible_cols]
    visible_drivers = np.array(drivers)[visible_rows]
    image = None
    for model_idx, (model_name, matrix) in enumerate(zip(models, model_matrices)):
        ax = axes[model_idx]
        matrix = matrix[np.ix_(visible_rows, visible_cols)]
        image = ax.imshow(
            matrix, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto", rasterized=True
        )
        ax.set_xticks(
            np.arange(len(visible_labels)),
            visible_labels,
            rotation=45,
            ha="right",
            fontsize=9,
        )
        ax.set_yticks(
            np.arange(len(visible_drivers)),
            [CORRELATION_LABELS[driver] for driver in visible_drivers],
            fontsize=9,
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
                        fontsize=7.5,
                        color="white" if abs(matrix[row, col]) > 0.55 else "black",
                    )
        ax.set_title(model_name, fontsize=11.5)
    axes[-1].set_xlabel("Altitude bin (km)", fontsize=10)
    colorbar = fig.colorbar(image, ax=axes, label="Pearson r", shrink=0.82)
    colorbar.ax.tick_params(labelsize=9)
    colorbar.ax.yaxis.label.set_size(9.5)
    save_figure(fig, "correlation", filename)


def sigma_edges(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values))
    if not np.isfinite(std) or std == 0:
        return np.array([-0.5, 0.5]), mean, std
    z = (values - mean) / std
    edges = np.arange(np.floor(np.nanmin(z)), np.ceil(np.nanmax(z)) + 1, 1.0)
    return edges if len(edges) >= 2 else np.array([-0.5, 0.5]), mean, std


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
    return f"{bin_start}-<{bin_start + CORRELATION_DURATION_STEP_YEARS} yr"


def correlation_duration_marker(bin_start: int) -> str:
    return CORRELATION_DURATION_MARKERS.get(bin_start, "*")


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


def binned_table(missions: list[tuple[MissionConfig, pl.DataFrame]]) -> pl.DataFrame:
    rows = []
    for config, df in missions:
        altitude = df["Altitude (m)"].to_numpy().astype(float) / 1000.0
        alt_edges = np.arange(
            np.floor(np.nanmin(altitude) / 25) * 25,
            np.ceil(np.nanmax(altitude) / 25) * 25 + 25,
            25,
        )
        work = df.with_columns((pl.col("Altitude (m)") / 1000.0).alias("altitude_km"))
        driver_edges = {}
        sigma_exprs = []
        for driver in ACTIVITY_DRIVERS:
            edges, mean, std = sigma_edges(work[driver].to_numpy().astype(float))
            driver_edges[driver] = edges
            sigma_exprs.append(
                ((pl.col(driver) - mean) / std).alias(f"{driver}_sigma")
                if std
                else pl.lit(0.0).alias(f"{driver}_sigma")
            )
        work = work.with_columns(*sigma_exprs)
        for model_name, model_col in MODEL_COLS.items():
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
                        if cell.height < 20:
                            continue
                        start = cell["timestamp"].min()
                        end = cell["timestamp"].max()
                        duration_years = (end - start).days / 365.2425
                        co2 = cell["co2"].to_numpy().astype(float)
                        error_vals = cell[model_col].to_numpy().astype(float)
                        _corr, _slope, _rmse, slope_lo, slope_hi = linear_fit_stats(
                            co2, error_vals
                        )
                        _, corr_lo, corr_hi, _ = pearsonr_ci(co2, error_vals)
                        rows.append(
                            {
                                "mission": config.name,
                                "model": model_name,
                                "driver": driver,
                                "driver_sigma_min": float(edges[sigma_idx]),
                                "driver_sigma_max": float(edges[sigma_idx + 1]),
                                "altitude_min_km": float(alt_edges[alt_idx]),
                                "altitude_max_km": float(alt_edges[alt_idx + 1]),
                                "mean_error": float(cell[model_col].mean()),
                                "std_error": float(cell[model_col].std()),
                                "co2_correlation": _corr,
                                "co2_corr_lo": corr_lo,
                                "co2_corr_hi": corr_hi,
                                "co2_slope": _slope,
                                "co2_slope_lo": slope_lo,
                                "co2_slope_hi": slope_hi,
                                "n": cell.height,
                                "duration_bin_years": correlation_duration_bin(
                                    duration_years
                                ),
                            }
                        )
    table = pl.DataFrame(rows)
    table.write_csv(output_path("binning", "tudelft_model_error_binned_stats.csv"))
    return table


def plot_binned_summary(table: pl.DataFrame) -> None:
    fig, axes = plt.subplots(
        len(MODEL_COLS),
        len(ACTIVITY_DRIVERS),
        figsize=fig_size(1.0, 1.25),
        constrained_layout=True,
        sharey=True,
    )
    for row_idx, model_name in enumerate(MODEL_COLS):
        model_df = table.filter(pl.col("model") == model_name)
        for col_idx, driver in enumerate(ACTIVITY_DRIVERS):
            ax = axes[row_idx, col_idx]
            driver_df = (
                model_df.filter(pl.col("driver") == driver)
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
                    pl.col("duration_bin_years").max().alias("duration_bin_years"),
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
                    linewidth=1.0,
                    label=label,
                    zorder=3,
                )
                lo_vals = series["co2_slope_lo_agg"].to_numpy().astype(float)
                hi_vals = series["co2_slope_hi_agg"].to_numpy().astype(float)
                slope_vals = series["co2_slope"].to_numpy().astype(float)
                alt_vals = series["altitude_min_km"].to_numpy().astype(float)
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
                        points["altitude_min_km"],
                        points["co2_slope"],
                        marker=correlation_duration_marker(int(duration_bin)),
                        s=32,
                        color=line.get_color(),
                        edgecolors="black",
                        linewidths=0.45,
                        zorder=4,
                    )
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Altitude bin lower edge (km)")
            if col_idx == 0:
                ax.set_ylabel(
                    model_name + "\n" + r"CO$_2$ fitted slope in $\epsilon_m$ per ppm"
                )
            ax.set_title(f"Binned by {DRIVER_COLS[driver]}")
            ax.grid(True, alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    add_outside_legend(
        fig, handles, labels, loc="upper center", title="Activity bin", ncols=3
    )
    add_record_length_legend(
        axes[0, 0],
        set(table["duration_bin_years"].drop_nulls().to_list())
        if table.height
        else set(),
    )
    fig.suptitle("TuDelft altitude- and activity-binned CO$_2$ model-error slope")
    save_figure(fig, "binning", "all_models_altitude_activity_binned_summary.pgf")


def plot_co2_correlation_binned(table: pl.DataFrame) -> None:
    fig, axes = plt.subplots(
        len(MODEL_COLS),
        len(ACTIVITY_DRIVERS),
        figsize=fig_size(1.0, 1.25),
        constrained_layout=True,
        sharey=True,
    )
    for row_idx, model_name in enumerate(MODEL_COLS):
        model_df = table.filter(pl.col("model") == model_name)
        for col_idx, driver in enumerate(ACTIVITY_DRIVERS):
            ax = axes[row_idx, col_idx]
            driver_df = (
                model_df.filter(pl.col("driver") == driver)
                .drop_nulls("co2_correlation")
                .with_columns(
                    (pl.col("co2_correlation") * pl.col("n")).alias("weighted_corr"),
                    (
                        pl.when(pl.col("co2_corr_lo").is_not_null())
                        .then(pl.col("co2_corr_lo") * pl.col("n"))
                        .otherwise(0.0)
                    ).alias("weighted_corr_lo"),
                    (
                        pl.when(pl.col("co2_corr_hi").is_not_null())
                        .then(pl.col("co2_corr_hi") * pl.col("n"))
                        .otherwise(0.0)
                    ).alias("weighted_corr_hi"),
                )
                .group_by("driver_sigma_min", "driver_sigma_max", "altitude_min_km")
                .agg(
                    pl.col("weighted_corr").sum().alias("weighted_corr"),
                    pl.col("weighted_corr_lo").sum().alias("weighted_corr_lo"),
                    pl.col("weighted_corr_hi").sum().alias("weighted_corr_hi"),
                    pl.col("n").sum().alias("n"),
                    pl.col("duration_bin_years").max().alias("duration_bin_years"),
                )
                .with_columns(
                    (pl.col("weighted_corr") / pl.col("n")).alias("co2_correlation"),
                    (pl.col("weighted_corr_lo") / pl.col("n")).alias("co2_corr_lo_agg"),
                    (pl.col("weighted_corr_hi") / pl.col("n")).alias("co2_corr_hi_agg"),
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
                    series["co2_correlation"],
                    linewidth=1.0,
                    label=label,
                    zorder=3,
                )
                lo_vals = series["co2_corr_lo_agg"].to_numpy().astype(float)
                hi_vals = series["co2_corr_hi_agg"].to_numpy().astype(float)
                corr_vals = series["co2_correlation"].to_numpy().astype(float)
                alt_vals = series["altitude_min_km"].to_numpy().astype(float)
                finite_ci = (
                    np.isfinite(corr_vals) & np.isfinite(lo_vals) & np.isfinite(hi_vals)
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
                        points["altitude_min_km"],
                        points["co2_correlation"],
                        marker=correlation_duration_marker(int(duration_bin)),
                        s=32,
                        color=line.get_color(),
                        edgecolors="black",
                        linewidths=0.45,
                        zorder=4,
                    )
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Altitude bin lower edge (km)")
            if col_idx == 0:
                ax.set_ylabel(model_name + "\n" + r"Pearson r($\epsilon_m$, CO$_2$)")
            ax.set_title(f"Binned by {DRIVER_COLS[driver]}")
            ax.grid(True, alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    add_outside_legend(
        fig, handles, labels, loc="upper center", title="Activity bin", ncols=3
    )
    add_record_length_legend(
        axes[0, 0],
        set(table["duration_bin_years"].drop_nulls().to_list())
        if table.height
        else set(),
    )
    fig.suptitle("TuDelft altitude- and activity-binned CO$_2$ model-error correlation")
    save_figure(fig, "binning", "all_models_altitude_activity_binned_correlation.pgf")


def sigma_bin_labels(edges: np.ndarray) -> list[str]:
    return [
        f"{edges[idx]:g} to {edges[idx + 1]:g} sigma" for idx in range(len(edges) - 1)
    ]


def finite_xy(
    df: pl.DataFrame, x_col: str, y_col: str
) -> tuple[np.ndarray, np.ndarray]:
    data = df.select(x_col, y_col).drop_nulls()
    if data.height == 0:
        return np.array([]), np.array([])
    x = data[x_col].to_numpy().astype(float)
    y = data[y_col].to_numpy().astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def linear_fit_stats(
    x: np.ndarray, y: np.ndarray
) -> tuple[float, float, float, float, float]:
    if len(x) < MIN_SAMPLES_PER_HEATMAP_CELL or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    slope, slope_lo, slope_hi, _se, _intercept, correlation, rmse, _n = ols_slope_ci(
        x, y
    )
    return correlation, float(slope), rmse, slope_lo, slope_hi


def heatmap_annotation(
    correlation: float,
    slope: float,
    rmse: float,
    count: int,
    year_range: str,
    slope_lo: float = np.nan,
    slope_hi: float = np.nan,
) -> str:
    return f"r={correlation:.2f}\nn={count}"


def text_color_for_correlation(value: float) -> str:
    return "white" if abs(value) > 0.55 else "black"


def plot_co2_error_heatmap_grid(
    df: pl.DataFrame,
    driver: str,
    output_parts: tuple[str, ...],
    title: str,
) -> None:
    if df.is_empty():
        return
    driver_values = df.select(driver).drop_nulls()[driver].to_numpy().astype(float)
    driver_values = driver_values[np.isfinite(driver_values)]
    if len(driver_values) < MIN_SAMPLES_PER_HEATMAP_CELL:
        return
    edges, driver_mean, driver_std = sigma_edges(driver_values)
    if not np.isfinite(driver_std) or driver_std == 0:
        return
    altitude = df["altitude_km"].to_numpy().astype(float)
    altitude = altitude[np.isfinite(altitude)]
    if len(altitude) < MIN_SAMPLES_PER_HEATMAP_CELL:
        return
    alt_edges = np.arange(
        np.floor(np.nanmin(altitude) / CO2_HEATMAP_ALTITUDE_BIN_KM)
        * CO2_HEATMAP_ALTITUDE_BIN_KM,
        np.ceil(np.nanmax(altitude) / CO2_HEATMAP_ALTITUDE_BIN_KM)
        * CO2_HEATMAP_ALTITUDE_BIN_KM
        + CO2_HEATMAP_ALTITUDE_BIN_KM,
        CO2_HEATMAP_ALTITUDE_BIN_KM,
    )
    if len(alt_edges) < 2:
        return

    work = df.with_columns(
        ((pl.col(driver) - driver_mean) / driver_std).alias("driver_sigma")
    )
    row_labels = sigma_bin_labels(edges)
    col_labels = [
        f"{alt_edges[idx]:.0f}-{alt_edges[idx + 1]:.0f}"
        for idx in range(len(alt_edges) - 1)
    ]
    figure_height = 0.88 if driver == "ap" else 0.92
    fig, axes = plt.subplots(
        1,
        len(MODEL_COLS),
        figsize=fig_size(1.0, figure_height),
        constrained_layout=True,
        sharey=True,
    )
    axes = np.asarray(axes).reshape(len(MODEL_COLS))
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad(color="lightgray")
    image = None
    panel_data = []

    for model_idx, (model_name, model_col) in enumerate(MODEL_COLS.items()):
        correlations = np.full((len(edges) - 1, len(alt_edges) - 1), np.nan)
        slopes = np.full_like(correlations, np.nan)
        errors = np.full_like(correlations, np.nan)
        slope_los = np.full_like(correlations, np.nan)
        slope_his = np.full_like(correlations, np.nan)
        counts = np.zeros_like(correlations, dtype=int)
        year_ranges = np.full(correlations.shape, "", dtype=object)
        year_spans = np.zeros(correlations.shape, dtype=int)

        for row in range(len(edges) - 1):
            low = edges[row]
            high = edges[row + 1]
            upper_filter = (
                pl.col("driver_sigma") <= high
                if row == len(edges) - 2
                else pl.col("driver_sigma") < high
            )
            sigma_filter = (pl.col("driver_sigma") >= low) & upper_filter
            for col in range(len(alt_edges) - 1):
                altitude_filter = (pl.col("altitude_km") >= alt_edges[col]) & (
                    pl.col("altitude_km") < alt_edges[col + 1]
                )
                cell = work.filter(sigma_filter & altitude_filter)
                x, y = finite_xy(cell, "co2", model_col)
                counts[row, col] = len(x)
                (
                    correlations[row, col],
                    slopes[row, col],
                    errors[row, col],
                    slope_los[row, col],
                    slope_his[row, col],
                ) = linear_fit_stats(x, y)
                if counts[row, col] > 0:
                    time_col = "date" if "date" in cell.columns else "timestamp"
                    valid_dates = (
                        cell.filter(
                            pl.col("co2").is_not_null()
                            & pl.col(model_col).is_not_null()
                            & pl.col("co2").is_finite()
                            & pl.col(model_col).is_finite()
                        )
                        .select(time_col)
                        .drop_nulls()
                    )
                    if valid_dates.height > 0:
                        start_year_full = valid_dates[time_col].min().year
                        end_year_full = valid_dates[time_col].max().year
                        year_spans[row, col] = end_year_full - start_year_full
                        year_ranges[row, col] = (
                            f"{start_year_full % 100:02d}-{end_year_full % 100:02d}"
                        )

        panel_data.append(
            {
                "model_idx": model_idx,
                "model_name": model_name,
                "correlations": correlations,
                "slopes": slopes,
                "errors": errors,
                "slope_los": slope_los,
                "slope_his": slope_his,
                "counts": counts,
                "year_ranges": year_ranges,
                "year_spans": year_spans,
            }
        )

    max_count = max(int(panel["counts"].max()) for panel in panel_data)
    count_threshold = max(1, int(np.ceil(0.01 * max_count)))
    filtered_correlations = []
    for panel in panel_data:
        correlations = np.array(panel["correlations"], copy=True)
        counts = np.asarray(panel["counts"])
        year_spans = np.asarray(panel["year_spans"])
        correlations[counts < count_threshold] = np.nan
        correlations[np.abs(correlations) < 0.1] = np.nan
        correlations[year_spans < 8] = np.nan
        filtered_correlations.append(correlations)

    finite_columns = np.any(
        np.stack([np.isfinite(matrix) for matrix in filtered_correlations]), axis=(0, 1)
    )
    if np.any(finite_columns):
        first_col = int(np.argmax(finite_columns))
        last_col = len(finite_columns) - int(np.argmax(finite_columns[::-1]))
        painted_cols = np.zeros_like(finite_columns, dtype=bool)
        painted_cols[first_col:last_col] = True
    else:
        painted_cols = np.ones(len(col_labels), dtype=bool)
    visible_col_labels = np.array(col_labels)[painted_cols]

    for panel, prefiltered_correlations in zip(panel_data, filtered_correlations):
        model_idx = int(panel["model_idx"])
        model_name = str(panel["model_name"])
        correlations = np.array(prefiltered_correlations, copy=True)
        slopes = np.asarray(panel["slopes"])
        errors = np.asarray(panel["errors"])
        slope_los = np.asarray(panel["slope_los"])
        slope_his = np.asarray(panel["slope_his"])
        counts = np.asarray(panel["counts"])
        year_ranges = np.asarray(panel["year_ranges"])

        painted_rows = np.any(
            np.isfinite(correlations) & (counts >= count_threshold), axis=1
        )
        if not np.any(painted_rows):
            continue
        ax = axes[model_idx]
        visible_correlations = correlations[np.ix_(painted_rows, painted_cols)]
        visible_slopes = slopes[np.ix_(painted_rows, painted_cols)]
        visible_errors = errors[np.ix_(painted_rows, painted_cols)]
        visible_slope_los = slope_los[np.ix_(painted_rows, painted_cols)]
        visible_slope_his = slope_his[np.ix_(painted_rows, painted_cols)]
        visible_counts = counts[np.ix_(painted_rows, painted_cols)]
        visible_year_ranges = year_ranges[np.ix_(painted_rows, painted_cols)]
        visible_row_labels = np.array(row_labels)[painted_rows]
        image = ax.imshow(
            np.ma.masked_invalid(visible_correlations),
            aspect="auto",
            origin="lower",
            vmin=-1,
            vmax=1,
            cmap=cmap,
        )
        ax.set_title(model_name, fontsize=13)
        ax.set_xticks(
            np.arange(len(visible_col_labels)),
            visible_col_labels,
            rotation=35,
            ha="right",
        )
        ax.set_xlabel("Altitude bin (km)", fontsize=11.5)
        ax.tick_params(axis="both", labelsize=10.5)
        if model_idx == 0:
            ax.set_yticks(np.arange(len(visible_row_labels)), visible_row_labels)
            ax.set_ylabel(
                f"{DRIVER_COLS[driver]} bins\nmean={driver_mean:.2f}, sigma={driver_std:.2f}",
                fontsize=11.5,
            )
        ax.set_xticks(np.arange(-0.5, len(visible_col_labels), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(visible_row_labels), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        for row in range(visible_correlations.shape[0]):
            for col in range(visible_correlations.shape[1]):
                value = visible_correlations[row, col]
                if np.isfinite(value):
                    ax.text(
                        col,
                        row,
                        heatmap_annotation(
                            value,
                            visible_slopes[row, col],
                            visible_errors[row, col],
                            visible_counts[row, col],
                            visible_year_ranges[row, col],
                            visible_slope_los[row, col],
                            visible_slope_his[row, col],
                        ),
                        ha="center",
                        va="center",
                        fontsize=10.5,
                        color=text_color_for_correlation(value),
                    )

    if image is None:
        plt.close(fig)
        return
    colorbar = fig.colorbar(
        image,
        ax=axes,
        label=r"Pearson r($\epsilon_m$, CO$_2$)",
        shrink=0.58,
        pad=0.012,
    )
    colorbar.ax.tick_params(labelsize=10.5)
    colorbar.ax.yaxis.label.set_size(10.5)
    fig.suptitle(title, fontsize=13)
    save_figure(fig, *output_parts)


def plot_co2_error_heatmaps(missions: list[tuple[MissionConfig, pl.DataFrame]]) -> None:
    combined = pl.concat(
        [
            df.with_columns(
                pl.lit(config.name).alias("mission"),
                (pl.col("Altitude (m)") / 1000.0).alias("altitude_km"),
            )
            for config, df in missions
        ],
        how="vertical_relaxed",
    )
    for driver in ACTIVITY_DRIVERS:
        plot_co2_error_heatmap_grid(
            combined,
            driver,
            ("binning", "co2_heatmaps", f"aggregate_error_co2_by_{driver}.pgf"),
            f"Aggregate TuDelft model-error versus CO$_2$ by {DRIVER_COLS[driver]} and altitude",
        )
        for config, df in missions:
            mission_df = df.with_columns(
                (pl.col("Altitude (m)") / 1000.0).alias("altitude_km")
            )
            plot_co2_error_heatmap_grid(
                mission_df,
                driver,
                (
                    "binning",
                    "co2_heatmaps",
                    "per_mission",
                    f"{safe_name(config.name)}_error_co2_by_{driver}.pgf",
                ),
                f"{config.name} model-error versus CO$_2$ by {DRIVER_COLS[driver]} and altitude",
            )


def mission_summary(missions: list[tuple[MissionConfig, pl.DataFrame]]) -> None:
    rows = []
    for config, df in missions:
        rows.append(
            {
                "mission": config.name,
                "rows": df.height,
                "start": df["timestamp"].min(),
                "end": df["timestamp"].max(),
                "min_alt_km": float((df["Altitude (m)"] / 1000.0).min()),
                "max_alt_km": float((df["Altitude (m)"] / 1000.0).max()),
            }
        )
    table = pl.DataFrame(rows)
    table.write_csv(output_path("tudelft_mission_summary.csv"))


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    missions = load_all(figure19=False)
    figure19_missions = load_all(figure19=True)
    mission_summary(missions)
    figure19(figure19_missions)
    # missions = [
    #     (
    #         config,
    #         df.filter(
    #             ((LAT - LAT_RANGE) <= pl.col("Latitude (deg)"))
    #             & (pl.col("Latitude (deg)") <= (LAT + LAT_RANGE))
    #             & ((LON - LON_RANGE) <= pl.col("Longitude (deg)"))
    #             & (pl.col("Longitude (deg)") <= (LON + LON_RANGE))
    #         ),
    #     )
    #     for config, df in missions
    # ]
    print("Plotting Timeseries")
    plot_combined_timeseries(missions)
    plot_per_mission_timeseries(missions)
    print("Plotting Spectra")
    plot_frequency_overview(missions)
    plot_frequency_overview(
        missions,
        {"NRLMSIS 2.0": MODEL_COLS["NRLMSIS 2.0"]},
        "tudelft_model_error_driver_spectra_nrlmsis_2p0.pgf",
    )
    plot_spectra(missions)
    print("Correlation")
    corr = correlation_tables(missions)
    plot_correlation_summary(corr)
    plot_correlation_summary(
        corr,
        {"NRLMSIS 2.0": MODEL_COLS["NRLMSIS 2.0"]},
        "nrlmsis_2p0_correlation_summary.png",
        (4, 3),
    )
    alt_corr = altitude_correlation_table(missions)
    plot_altitude_correlation_summary(alt_corr)
    plot_altitude_correlation_summary(
        alt_corr,
        {"NRLMSIS 2.0": MODEL_COLS["NRLMSIS 2.0"]},
        "nrlmsis_2p0_correlation_by_altitude_summary.png",
        (4, 3),
    )
    print("Binning")
    bins = binned_table(missions)
    plot_binned_summary(bins)
    plot_co2_correlation_binned(bins)
    plot_co2_error_heatmaps(missions)
    write_latex_index()
    print(f"Generated TuDelft model-error outputs in {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
