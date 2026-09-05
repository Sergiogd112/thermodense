"""Render the six first-paper figures from already prepared result artifacts.

This is deliberately a rendering-only workflow: it never invokes PCMCI or a
model executable.  ``--research-root`` may point at a checkout containing the
large ignored artifacts while this script and its publication output remain in
the source checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import polars as pl


FIGURES = tuple(f"fig{i}" for i in range(1, 7))
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00")
MARKERS = ("o", "s", "^", "D", "P", "X")
F107 = "F10.7_OBS_CENTER81"
AP = "AP_AVG"
FIG2_REQUIRED_SERIES = (
    "Global mean",
    "HASDM",
    "NRLMSISE-00",
    "NRLMSIS 2.0",
    "NRLMSIS 2.1",
    "JB2006",
    "JB2008",
)
TUDELFT_MISSIONS = (
    "CHAMP",
    "GOCE",
    "GRACE-A",
    "GRACE-B",
    "GRACE-FO",
    "Swarm-A",
    "Swarm-B",
    "Swarm-C",
)
FIG5_LEAKAGE_METRICS = ("forcing_f107_obs_center81", "forcing_ap_avg")
SABER_139_COLUMN = "SABER_CO2_COOLING_139KM"


class PrerequisiteError(RuntimeError):
    """A required prepared/result artifact is unavailable or incompatible."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(path: Path, description: str) -> Path:
    if not path.is_file():
        raise PrerequisiteError(f"Missing {description}: {path}")
    return path


def first_existing(root: Path, candidates: Iterable[str], description: str) -> Path:
    for relative in candidates:
        path = root / relative
        if path.is_file():
            return path
    raise PrerequisiteError(
        f"Missing {description}. Looked for: "
        + ", ".join(str(root / p) for p in candidates)
    )


def read_frame(path: Path) -> pl.DataFrame:
    if path.suffix == ".parquet":
        return pl.read_parquet(path)
    return pl.read_csv(path, try_parse_dates=True)


def date_column(frame: pl.DataFrame) -> str:
    for name in ("date", "DATE", "timestamp", "time"):
        if name in frame.columns:
            return name
    raise PrerequisiteError("Artifact has no recognised date column")


def finite_pair(frame: pl.DataFrame, x: str, y: str) -> tuple[np.ndarray, np.ndarray]:
    if x not in frame.columns or y not in frame.columns:
        return np.array([]), np.array([])
    values = frame.select(x, y).drop_nulls().to_numpy().astype(float)
    values = values[np.isfinite(values).all(axis=1)]
    return values[:, 0], values[:, 1]


def density_columns(frame: pl.DataFrame, diagnostic: str) -> dict[int, str]:
    suffix = f"_daily_{diagnostic}"
    found: dict[int, str] = {}
    for name in frame.columns:
        match = re.fullmatch(r"log10rho_(\d+)_daily_(mean|range)", name)
        if match and name.endswith(suffix):
            found[int(match.group(1))] = name
    return found


def correlation_ci(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, int]:
    n = len(x)
    if n < 4 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan, np.nan, n
    r = float(np.corrcoef(x, y)[0, 1])
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    delta = 1.959963984540054 / np.sqrt(n - 3)
    return r, float(np.tanh(z - delta)), float(np.tanh(z + delta)), n


def source_record(
    root: Path, path: Path, frame: pl.DataFrame | None = None
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path.relative_to(root)),
        "sha256": sha256(path),
    }
    if frame is not None:
        record["rows"] = frame.height
        try:
            col = date_column(frame)
            dates = frame.select(pl.col(col).drop_nulls()).to_series()
            if len(dates):
                record["date_coverage"] = {
                    "start": str(dates.min()),
                    "end": str(dates.max()),
                }
        except PrerequisiteError:
            pass
    return record


def save_figure(fig: plt.Figure, output: Path, name: str) -> list[str]:
    paths = []
    for suffix, options in ((".png", {"dpi": 300}), (".pdf", {})):
        path = output / f"{name}{suffix}"
        fig.savefig(path, bbox_inches="tight", **options)
        paths.append(path.name)
    plt.close(fig)
    return paths


def label(column: str) -> str:
    labels = {
        F107: "centered-81 F10.7",
        AP: "Ap",
        "KP_SUM": "Kp",
        "CO2_ppm": "Mauna Loa CO2",
    }
    if column in labels:
        return labels[column]
    if "saber" in column.lower():
        return "SABER CO2 cooling\nat 139 km"
    return column


def coverage_record(
    name: str, path: Path, altitude_columns: bool = True
) -> tuple[dict[str, Any], pl.DataFrame]:
    frame = read_frame(path)
    time = date_column(frame)
    dates = frame.get_column(time).drop_nulls()
    altitudes: list[float] = []
    if "Altitude (m)" in frame.columns:
        altitudes = (frame.get_column("Altitude (m)").drop_nulls() / 1000).to_list()
    elif "altitude_km" in frame.columns:
        altitudes = frame.get_column("altitude_km").drop_nulls().to_list()
    elif altitude_columns:
        altitudes = [
            float(value)
            for column in frame.columns
            for pattern in (
                r"log10rho_(\d+)(?:km)?$",
                r"log10rho_(\d+)_daily_(?:min|mean|max|range)$",
                r"log10rho_daily_(?:mean|range)_(\d+)km$",
            )
            for value in re.findall(pattern, column)
        ]
    if not len(dates) or not altitudes:
        raise PrerequisiteError(
            f"{name} does not provide dated altitude coverage: {path}"
        )
    return {
        "series": name,
        "start": str(dates.min()),
        "end": str(dates.max()),
        "altitude_km": [min(altitudes), max(altitudes)],
        "rows": frame.height,
    }, frame


def coverage_figure(root: Path, output: Path) -> dict[str, Any]:
    regular = {
        "Global mean": "data/decoded/orbit_derived_global_mean/orbit-density-ds03-density-values.parquet",
        "HASDM": "outputs/figures/results/set_hasdm/model_validations/causal_hasdm_saber_maunaloa/hasdm_maunaloa_daily_wide.parquet",
        "NRLMSIS family": "outputs/figures/results/maunaloa_msis_density_baselines/data/maunaloa_msis_density_baselines_daily_wide.parquet",
        "JB2006/JB2008 paired": "outputs/figures/results/maunaloa_jb_density_baselines/data/maunaloa_jb_density_baselines_daily_wide.parquet",
        "SABER CO2 cooling": "data/decoded/saber/saber_co2_cooling_maunaloa_daily.parquet",
    }
    missions = {
        "CHAMP": "data/analyzed/tudelft/champ/CH_analyzed.parquet",
        "GOCE": "data/analyzed/tudelft/goce/GO_analyzed.parquet",
        "GRACE-A": "data/analyzed/tudelft/grace/GA_analyzed.parquet",
        "GRACE-B": "data/analyzed/tudelft/grace/GB_analyzed.parquet",
        "GRACE-FO": "data/analyzed/tudelft/grace_fo/GC_analyzed.parquet",
        "Swarm-A": "data/analyzed/tudelft/swarm/SA_analyzed.parquet",
        "Swarm-B": "data/analyzed/tudelft/swarm/SB_analyzed.parquet",
        "Swarm-C": "data/analyzed/tudelft/swarm/SC_analyzed.parquet",
    }
    fig, axes = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True, constrained_layout=True
    )
    records, sources = [], []
    for ax, title, items in (
        (axes[0], "(a) Regular/gridded and baseline products", regular),
        (axes[1], "(b) TU Delft mission coverage", missions),
    ):
        local, handles = [], []
        for index, (name, relative) in enumerate(items.items()):
            path = require(root / relative, name)
            record, frame = coverage_record(name, path)
            record["panel"] = title
            local.append(record)
            records.append(record)
            sources.append(source_record(root, path, frame))
            start = np.datetime64(record["start"])
            end = np.datetime64(record["end"])
            low, high = record["altitude_km"]
            hatch = ("//", "\\\\", "..", "xx", "++")[index % 5]
            ax.fill_betweenx(
                [low, high],
                start,
                end,
                color=COLORS[index % len(COLORS)],
                alpha=0.55,
                hatch=hatch,
                edgecolor="black",
                linewidth=0.5,
            )
            handles.append(
                Patch(
                    facecolor=COLORS[index % len(COLORS)],
                    hatch=hatch,
                    edgecolor="black",
                    label=name,
                    alpha=0.55,
                )
            )
        ax.set(title=title, ylabel="Altitude (km)", ylim=(0, 900))
        ax.grid(alpha=0.25)
        ax.legend(
            handles=handles, loc="upper left", fontsize=7, ncol=2, framealpha=0.92
        )
    axes[1].set_xlabel(
        "Date; rectangles show available date and altitude extent only, not an effect"
    )
    return {
        "files": save_figure(fig, output, "fig1"),
        "sources": sources,
        "panels": ["regular/gridded products", "eight TU Delft missions"],
        "series": records,
        "results": {"coverage_records": len(records)},
    }


def main_dataset(root: Path) -> tuple[Path, pl.DataFrame]:
    path = first_existing(
        root,
        [
            "outputs/figures/results/set_hasdm/model_validations/causal_hasdm_saber_maunaloa/daily_analysis_dataset.csv",
            "outputs/figures/results/hasdm_msis_model_errors/model_validations/causal_hasdm_saber_msis_residuals/data/daily_analysis_dataset.csv",
        ],
        "Mauna Loa/HASDM daily driver artifact",
    )
    wide_path = first_existing(
        root,
        [
            "outputs/figures/results/set_hasdm/model_validations/causal_hasdm_saber_maunaloa/hasdm_maunaloa_daily_wide.parquet"
        ],
        "HASDM altitude-wide daily artifact",
    )
    return path, read_frame(wide_path).join(read_frame(path), on="date", how="inner")


def add_saber_139(root: Path, daily: pl.DataFrame) -> tuple[Path, pl.DataFrame, float]:
    path = require(
        root / "data/decoded/saber/saber_co2_cooling_maunaloa_daily.parquet",
        "SABER CO2 cooling",
    )
    saber = read_frame(path)
    altitude = nearest_saber_altitude(
        saber.get_column("altitude_km").unique().to_list()
    )
    cooling = (
        saber.filter(pl.col("altitude_km") == altitude)
        .group_by("date")
        .agg(pl.col("co2_cooling_rate_w_m3").mean().alias("SABER_CO2_COOLING_139KM"))
    )
    return path, daily.join(cooling, on="date", how="left"), float(altitude)


def nearest_saber_altitude(altitudes: Iterable[float]) -> float:
    values = list(altitudes)
    if not values:
        raise PrerequisiteError("SABER cooling artifact has no altitude values")
    return float(min(values, key=lambda value: abs(value - 139)))


def response_figure(root: Path, output: Path, daily: pl.DataFrame) -> dict[str, Any]:
    paths = {
        "Global mean": "data/decoded/orbit_derived_global_mean/orbit-density-ds03-density-values.parquet",
        "HASDM": "outputs/figures/results/set_hasdm/model_validations/causal_hasdm_saber_maunaloa/hasdm_maunaloa_daily_wide.parquet",
        "NRLMSISE-00": "outputs/figures/results/maunaloa_msis_density_baselines/data/maunaloa_msis_density_baselines_daily_wide.parquet",
        "NRLMSIS 2.0": "outputs/figures/results/maunaloa_msis_density_baselines/data/maunaloa_msis_density_baselines_daily_wide.parquet",
        "NRLMSIS 2.1": "outputs/figures/results/maunaloa_msis_density_baselines/data/maunaloa_msis_density_baselines_daily_wide.parquet",
        "JB2006": "outputs/figures/results/maunaloa_jb_density_baselines/data/maunaloa_jb_density_baselines_daily_wide.parquet",
        "JB2008": "outputs/figures/results/maunaloa_jb_density_baselines/data/maunaloa_jb_density_baselines_daily_wide.parquet",
    }
    driver = daily.select("date", F107)
    fig, axes = plt.subplots(
        1, 2, figsize=(10, 4.5), sharey=True, constrained_layout=True
    )
    results, series, sources = {}, [], []
    for ax, diagnostic, title in zip(
        axes,
        ("mean", "range"),
        (r"daily mean $\bar{\ell}_\rho$", r"daily range $\Delta\ell_\rho$"),
        strict=True,
    ):
        for index, (name, relative) in enumerate(paths.items()):
            path = require(root / relative, name)
            frame = read_frame(path)
            if F107 not in frame.columns:
                frame = frame.join(driver, on="date", how="inner")
            prefix = {
                "NRLMSISE-00": "nrlmsise_00_",
                "NRLMSIS 2.0": "nrlmsis_2p0_",
                "NRLMSIS 2.1": "nrlmsis_2p1_",
                "JB2006": "jb2006_",
                "JB2008": "jb2008_",
            }.get(name, "")
            pattern = re.compile(
                rf"{re.escape(prefix)}(?:log10rho_)?(?:log10rho_)?(?:daily_)?{diagnostic}_?(\d+)(?:km)?$"
            )
            columns = {}
            for col in frame.columns:
                match = pattern.fullmatch(col)
                if match:
                    columns[int(match.group(1))] = col
            if not columns and not prefix:
                columns = density_columns(frame, diagnostic)
                if name == "Global mean" and diagnostic == "mean":
                    columns = {
                        int(c.removeprefix("log10rho_")): c
                        for c in frame.columns
                        if re.fullmatch(r"log10rho_\d+", c)
                    }
            points = []
            for altitude, col in sorted(columns.items()):
                r, low, high, count = correlation_ci(*finite_pair(frame, F107, col))
                if np.isfinite(r):
                    points.append((altitude, r, low, high, count))
            if points:
                values = np.asarray(points)
                ax.errorbar(
                    values[:, 1],
                    values[:, 0],
                    xerr=np.vstack(
                        (values[:, 1] - values[:, 2], values[:, 3] - values[:, 1])
                    ),
                    fmt=MARKERS[index % len(MARKERS)] + "-",
                    color=COLORS[index % len(COLORS)],
                    label=name,
                    capsize=1.5,
                )
                results.setdefault(diagnostic, []).extend(
                    {
                        "product": name,
                        "altitude_km": int(p[0]),
                        "pearson_r": float(p[1]),
                        "ci95": [float(p[2]), float(p[3])],
                        "samples": int(p[4]),
                    }
                    for p in points
                )
                if name not in series:
                    series.append(name)
            if not any(item["path"] == str(path.relative_to(root)) for item in sources):
                sources.append(source_record(root, path, frame))
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("F10.7 response (Pearson r)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Altitude (km)")
    axes[0].legend(fontsize=6, ncol=2)
    return {
        "files": save_figure(fig, output, "fig2"),
        "sources": sources,
        "panels": ["mean", "range"],
        "series": series,
        "results": results,
        "omissions": [
            "TU Delft is omitted because a directly comparable daily altitude diagnostic would require a new mission-aware aggregation; its irregular mission sampling is shown in Fig1.",
            "Global mean provides daily mean only; no daily within-day range is fabricated.",
        ],
    }


def scatter_figure(
    root: Path, output: Path, daily: pl.DataFrame, source: Path
) -> dict[str, Any]:
    targets = [
        "log10rho_175_daily_mean",
        "log10rho_825_daily_mean",
        "log10rho_175_daily_range",
        "log10rho_825_daily_range",
    ]
    cooling = SABER_139_COLUMN
    drivers = [F107, AP, "CO2_ppm", cooling]
    if any(c is None or c not in daily.columns for c in drivers) or any(
        c not in daily.columns for c in targets
    ):
        raise PrerequisiteError(
            "Fig3 requires 175/825 mean/range, F10.7, Ap, CO2, and SABER 139-km cooling columns"
        )
    fig, axes = plt.subplots(4, 4, figsize=(10, 9), constrained_layout=True)
    counts = []
    for row, target in enumerate(targets):
        for col, driver in enumerate(drivers):
            x, y = finite_pair(daily, driver, target)
            counts.append(int(len(x)))
            if driver == SABER_139_COLUMN:
                x = x * 1e9
            axes[row, col].scatter(x, y, s=4, alpha=0.28, color="#0072B2", marker="o")
            if row == 3:
                xlabel = label(driver)
                if driver == SABER_139_COLUMN:
                    xlabel += " (nW m$^{-3}$)"
                axes[row, col].set_xlabel(xlabel, fontsize=8)
            if col == 0:
                axes[row, col].set_ylabel(
                    target.replace("log10rho_", "").replace("_daily_", " ") + " km",
                    fontsize=8,
                )
            axes[row, col].grid(alpha=0.18)
    wide_path = (
        root
        / "outputs/figures/results/set_hasdm/model_validations/causal_hasdm_saber_maunaloa/hasdm_maunaloa_daily_wide.parquet"
    )
    saber_path = root / "data/decoded/saber/saber_co2_cooling_maunaloa_daily.parquet"
    return {
        "files": save_figure(fig, output, "fig3"),
        "sources": [
            source_record(root, source),
            source_record(root, wide_path),
            source_record(root, saber_path),
        ],
        "panels": [f"{t} × {d}" for t in targets for d in drivers],
        "series": [str(x) for x in drivers],
        "results": {"pair_sample_counts": counts, "saber_cooling_column": cooling},
    }


def relationship_figure(
    root: Path, output: Path, daily: pl.DataFrame, source: Path
) -> dict[str, Any]:
    fig, axes = plt.subplots(
        1, 2, figsize=(10, 4.8), sharey=True, constrained_layout=True
    )
    numerical = []
    drivers = [
        c
        for c in ("CO2_ppm", SABER_139_COLUMN, F107, AP, "KP_SUM")
        if c in daily.columns
    ]
    for ax, diagnostic in zip(axes, ("mean", "range"), strict=True):
        for index, driver in enumerate(drivers):
            points = []
            for altitude, target in sorted(density_columns(daily, diagnostic).items()):
                r, lo, hi, n = correlation_ci(*finite_pair(daily, driver, target))
                if np.isfinite(r):
                    points.append((altitude, r, lo, hi, n))
            if points:
                values = np.asarray(points)
                ax.errorbar(
                    values[:, 1],
                    values[:, 0],
                    xerr=np.vstack(
                        (values[:, 1] - values[:, 2], values[:, 3] - values[:, 1])
                    ),
                    fmt=MARKERS[index % len(MARKERS)] + "-",
                    color=COLORS[index],
                    label=label(driver),
                    alpha=0.9,
                )
                numerical.extend(
                    {
                        "diagnostic": diagnostic,
                        "driver": driver,
                        "altitude_km": int(p[0]),
                        "r": float(p[1]),
                        "ci95": [float(p[2]), float(p[3])],
                        "samples": int(p[4]),
                    }
                    for p in points
                )
        ax.axvspan(-0.1, 0.1, color="0.5", alpha=0.12)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(diagnostic)
        ax.set_xlabel("Pearson r (95% interval)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Altitude (km)")
    axes[1].legend(fontsize=7)
    wide_path = (
        root
        / "outputs/figures/results/set_hasdm/model_validations/causal_hasdm_saber_maunaloa/hasdm_maunaloa_daily_wide.parquet"
    )
    saber_path = root / "data/decoded/saber/saber_co2_cooling_maunaloa_daily.parquet"
    return {
        "files": save_figure(fig, output, "fig4"),
        "sources": [
            source_record(root, source),
            source_record(root, wide_path),
            source_record(root, saber_path),
        ],
        "panels": ["daily mean", "daily range"],
        "series": drivers,
        "results": numerical,
    }


def model_figure(root: Path, output: Path) -> dict[str, Any]:
    path = require(
        root / "outputs/prototypes/empirical_model_figure5/metrics.csv",
        "Fig5 precomputed empirical-model metrics",
    )
    frame = read_frame(path)
    needed = {"altitude_km"}
    if not needed <= set(frame.columns):
        raise PrerequisiteError("Fig5 metrics.csv lacks altitude_km")
    model_col = next((c for c in ("model", "model_name") if c in frame.columns), None)
    bias = next(
        (
            c
            for c in frame.columns
            if "signed_median" in c or ("median" in c and "bias" in c)
        ),
        None,
    )
    absolute = next(
        (c for c in frame.columns if "median" in c and "absolute" in c), None
    )
    leakage = list(FIG5_LEAKAGE_METRICS)
    required = [
        model_col,
        bias,
        absolute,
        *(
            f"{metric}_ci_{side}"
            for metric in (bias, absolute)
            for side in ("low", "high")
        ),
        *leakage,
        *(f"{metric}_detected" for metric in leakage),
    ]
    if any(value is None or value not in frame.columns for value in required):
        raise PrerequisiteError(
            "Fig5 metrics.csv lacks exact centered-F10.7/Ap leakage or accuracy intervals"
        )
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    models = frame[model_col].unique().sort().to_list()
    for i, model in enumerate(models):
        part = frame.filter(pl.col(model_col) == model).sort("altitude_km")
        for ax, metric, title in (
            (axes[0, 0], bias, "signed median log-density-ratio bias"),
            (axes[0, 1], absolute, "median absolute log error"),
        ):
            values = part[metric].to_numpy()
            low = part[f"{metric}_ci_low"].to_numpy()
            high = part[f"{metric}_ci_high"].to_numpy()
            ax.errorbar(
                values,
                part["altitude_km"],
                xerr=np.vstack((values - low, high - values)),
                marker=MARKERS[i % len(MARKERS)],
                color=COLORS[i % len(COLORS)],
                label=part["model_display"][0],
            )
            ax.set_title(title)
            ax.grid(alpha=0.25)
            ax.axvline(0, color="black", lw=0.8)
        for ax, metric, title in zip(
            axes[1], leakage, ("centered-81 F10.7 leakage", "Ap leakage"), strict=True
        ):
            detected = part[f"{metric}_detected"].to_numpy()
            values = part[metric].to_numpy()
            altitude = part["altitude_km"].to_numpy()
            ax.scatter(
                values[~detected],
                altitude[~detected],
                marker=MARKERS[i % len(MARKERS)],
                facecolors="none",
                edgecolors=COLORS[i % len(COLORS)],
                s=28,
                label=part["model_display"][0],
            )
            ax.scatter(
                values[detected],
                altitude[detected],
                marker=MARKERS[i % len(MARKERS)],
                facecolors=COLORS[i % len(COLORS)],
                edgecolors="black",
                s=32,
            )
            ax.set_title(title)
            ax.axvline(0, color="black", lw=0.8)
            ax.grid(alpha=0.25)
    for ax in axes.flat:
        ax.set_ylabel("Altitude (km)")
    axes[0, 1].legend(fontsize=7)
    summaries = {
        str(model): {
            metric: float(frame.filter(pl.col(model_col) == model)[metric].median())
            for metric in [bias, absolute, *leakage]
        }
        for model in models
    }
    retained_counts = {
        str(model): {
            metric: int(
                frame.filter(pl.col(model_col) == model)[f"{metric}_detected"].sum()
            )
            for metric in leakage
        }
        for model in models
    }
    return {
        "files": save_figure(fig, output, "fig5"),
        "sources": [source_record(root, path, frame)],
        "panels": ["bias", "absolute error", "centered F10.7 leakage", "Ap leakage"],
        "series": [str(x) for x in models],
        "results": {
            "rows": frame.height,
            "metrics": [bias, absolute, *leakage],
            "median_by_model": summaries,
            "fdr_retained_altitude_counts": retained_counts,
        },
        "caveat": "Filled circles are FDR-retained leakage links and open circles are not retained; zero means no retained link, not independence. Nearer-zero leakage is descriptive; no universal model ranking is claimed.",
    }


def trend_figure(root: Path, output: Path) -> dict[str, Any]:
    path = first_existing(
        root,
        [
            "outputs/figures/results/current_density_trends/current_density_trends_by_dataset_altitude.csv"
        ],
        "current trend CSV for Fig6",
    )
    brown = [root / "data/derived/literature/brown_2024_figure2_digitized.csv"]
    if not brown:
        raise PrerequisiteError(
            "Fig6 requires the Brown digitized CSV; no *brown*.csv file was found"
        )
    frame = read_frame(path)
    required = {"dataset", "altitude_km", "trend_percent_per_decade"}
    if not required <= set(frame.columns):
        raise PrerequisiteError("Fig6 trend CSV has incompatible columns")
    paired = {"JB2006", "JB2008"}
    labels = " ".join(frame["dataset"].unique().to_list())
    if not all(name in labels for name in paired):
        raise PrerequisiteError(
            "Fig6 requires paired JB2006/JB2008 trends in the current trend CSV"
        )
    brown_frame = read_frame(brown[0])
    brown_required = {
        "study",
        "series_type",
        "sequence",
        "density_trend_pct_per_decade",
        "altitude_km",
        "include_in_plot",
        "color_hex",
        "line_style",
        "marker",
    }
    if not brown_required <= set(brown_frame.columns) or brown_frame.height < 427:
        raise PrerequisiteError("Fig6 Brown digitized CSV is incompatible")
    fig, axes = plt.subplots(
        1, 2, figsize=(14, 6), sharey=True, constrained_layout=True
    )
    ax_a, ax = axes
    included = brown_frame.filter(
        pl.col("include_in_plot").cast(pl.String).str.to_lowercase() == "true"
    )
    for group in included.filter(pl.col("series_type") == "profile").partition_by(
        ["study", "variant"], as_dict=False
    ):
        group = group.sort("sequence")
        ax_a.plot(
            group["density_trend_pct_per_decade"],
            group["altitude_km"],
            color=group["color_hex"][0],
            linestyle={"solid": "-", "dash": "--", "dot": ":"}.get(
                group["line_style"][0], "-"
            ),
            lw=1,
        )
    for row in included.filter(
        pl.col("marker").cast(pl.String).str.to_lowercase() == "true"
    ).iter_rows(named=True):
        ax_a.scatter(
            row["density_trend_pct_per_decade"],
            row["altitude_km"],
            color=row["color_hex"],
            marker="o",
            s=16,
            edgecolor="black",
            linewidth=0.25,
        )
    ax_a.set(
        xlim=(-7, 1),
        ylim=(0, 850),
        xlabel="Reported density trend (%/decade)",
        ylabel="Altitude (km)",
        title="(a) Brown et al. (2024) literature reconstruction",
    )
    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[i % len(COLORS)],
            marker=MARKERS[i % len(MARKERS)],
            label=study,
            lw=1,
        )
        for i, study in enumerate(included["study"].unique().sort().to_list())
    ]
    ax_a.legend(
        handles=handles,
        title="16 studies",
        ncol=4,
        loc="upper center",
        fontsize=6,
        title_fontsize=7,
        framealpha=0.94,
    )
    for i, dataset in enumerate(frame["dataset"].unique().sort().to_list()):
        part = frame.filter(pl.col("dataset") == dataset).sort("altitude_km")
        values = part["trend_percent_per_decade"].to_numpy()
        low = part["trend_percent_per_decade_hac_95_ci_lower"].to_numpy()
        high = part["trend_percent_per_decade_hac_95_ci_upper"].to_numpy()
        ax.errorbar(
            values,
            part["altitude_km"],
            xerr=np.vstack((values - low, high - values)),
            marker=MARKERS[i % len(MARKERS)],
            color=COLORS[i % len(COLORS)],
            label=dataset,
            lw=1,
            capsize=1.5,
        )
    ax.axvline(0, color="black", lw=0.8)
    ax.set(
        xlabel="Solar-adjusted density trend (%/decade)",
        ylabel="Altitude (km)",
        ylim=(0, 850),
        title="(b) Updated estimates (HAC 95% intervals)",
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6, ncol=2, loc="lower left", framealpha=0.94)
    results = {
        dataset: {
            "median_trend_percent_per_decade": float(
                part["trend_percent_per_decade"].median()
            ),
            "altitude_count": part.height,
        }
        for dataset in frame["dataset"].unique().to_list()
        for part in [frame.filter(pl.col("dataset") == dataset)]
    }
    return {
        "files": save_figure(fig, output, "fig6"),
        "sources": [
            source_record(root, path, frame),
            source_record(root, brown[0], brown_frame),
        ],
        "panels": ["Brown literature reconstruction", "updated HAC trends"],
        "series": frame["dataset"].unique().to_list(),
        "results": results,
        "brown": {
            "doi": "10.1029/2024JA032659",
            "license": "CC BY 4.0",
            "expected_sha256": "1fafa2718250adcd01677d4c9257cef4f72d3e7d654a7a14accc2c8cdc216583",
            "studies": included["study"].n_unique(),
        },
        "caveat": "Panel A uses plot-precision digitized values from Brown Figure 2, not replacement study data. Panel B trends are descriptive, method- and sampling-dependent, and not deconfounded CO2-only estimates.",
    }


CAPTIONS = {
    "fig1": "Date–altitude availability rectangles for canonical regular/gridded density products and each of the eight TU Delft missions. Rectangle extent denotes available records only and is not evidence of a density response.",
    "fig2": "Pearson correlations of centered 81-day F10.7 with daily mean log10 density and daily within-day log10-density range by altitude. Horizontal bars are Fisher 95% intervals; product-specific date and row support are retained. TU Delft is omitted because a comparable mission-aware daily aggregation was not precomputed.",
    "fig3": "Points-only descriptive scatter matrix. Rows are HASDM daily mean log10 density and within-day log10 range at 175 and 825 km; columns are centered-81 F10.7, Ap, Mauna Loa CO2, and SABER CO2 cooling aggregated at the available altitude nearest 139 km. Pairwise missing values are omitted.",
    "fig4": "Altitude profiles of pairwise Pearson correlations for daily mean and range diagnostics with CO2, SABER cooling nearest 139 km, solar, and geomagnetic context. Shading marks the near-zero reference band and horizontal intervals are Fisher 95% intervals; these are descriptive associations.",
    "fig5": "On the exact common five-model sample, signed median log-density-ratio error and median absolute error are plotted with precomputed 95% block-bootstrap intervals. Centered-81 F10.7 and Ap residual leakage panels use filled circles for FDR-retained links and open circles for non-retained links; zero means no retained link, not independence or a universal ranking.",
    "fig6": "Two-panel density-trend synthesis. Panel A vector-renders plot-precision values digitized from Brown et al. (2024) Figure 2 under CC BY 4.0 and is not replacement study data. Panel B shows solar-adjusted log10-density trends with 27-day HAC 95% intervals and paired JB2006/JB2008; trends are descriptive, not CO2-only estimates.",
}
ALTS = {
    "fig1": "Two date versus altitude panels: five canonical product rectangles above and eight separately labelled TU Delft mission rectangles below.",
    "fig2": "Two altitude profiles compare F10.7 correlations for Global mean, HASDM, three NRLMSIS products, JB2006 and JB2008, using color, marker, and line style.",
    "fig3": "A 4 by 4 grid of blue points showing four density targets against F10.7, Ap, CO2, and explicitly labelled nearest-139-km SABER cooling.",
    "fig4": "Two altitude panels show colored, marked correlation profiles with 95 percent horizontal intervals and a gray near-zero band.",
    "fig5": "Four altitude-profile panels show five empirical models, error intervals, and filled circles identifying FDR-retained centered-F10.7 or Ap leakage links; open circles are non-retained.",
    "fig6": "Two equal-width panels share a 0 to 850 km altitude axis: a Brown literature reconstruction with 16-study legend and updated project trends with HAC intervals including both JB models.",
}


def validate_summary(summary: dict[str, Any]) -> None:
    if summary.get("schema_version") != 1 or set(summary.get("figures", {})) != set(
        FIGURES
    ):
        raise ValueError("Summary must contain exactly the six contracted figures")
    for name, item in summary["figures"].items():
        if set(item.get("files", ())) != {f"{name}.png", f"{name}.pdf"}:
            raise ValueError(f"{name} lacks its PNG/PDF pair")
        if (
            not isinstance(item.get("sources"), list)
            or not item.get("caption")
            or not item.get("alt_text")
        ):
            raise ValueError(f"{name} lacks required provenance or accessible text")


def build(root: Path, output: Path) -> dict[str, Any]:
    root, output = root.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source, daily = main_dataset(root)
    _, daily, saber_altitude = add_saber_139(root, daily)
    figures = {
        "fig1": coverage_figure(root, output),
        "fig2": response_figure(root, output, daily),
        "fig3": scatter_figure(root, output, daily, source),
        "fig4": relationship_figure(root, output, daily, source),
        "fig5": model_figure(root, output),
        "fig6": trend_figure(root, output),
    }
    figures["fig3"]["results"]["saber_selected_altitude_km"] = saber_altitude
    figures["fig4"]["saber_selected_altitude_km"] = saber_altitude
    for name, item in figures.items():
        item["caption"], item["alt_text"] = CAPTIONS[name], ALTS[name]
        (output / f"{name}.caption.txt").write_text(item["caption"] + "\n")
        (output / f"{name}.alt.txt").write_text(item["alt_text"] + "\n")
    summary = {
        "schema_version": 1,
        "builder": "scripts/build_first_paper_figures.py",
        "generated_on": str(date.today()),
        "research_root": str(root),
        "figures": figures,
    }
    validate_summary(summary)
    (output / "first-paper-figures-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("manuscript/figures"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = build(args.research_root, args.output_dir)
    print(f"Rendered {len(summary['figures'])} figures to {args.output_dir}")
