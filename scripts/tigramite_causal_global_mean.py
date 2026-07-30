from __future__ import annotations
# Ruff: configure_pgf() must run before pyplot imports; suppress intentional E402.
# ruff: noqa: E402

import argparse
from dataclasses import dataclass
from pathlib import Path

from scripts.pgf_config import configure_pgf

configure_pgf()

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from tqdm import tqdm
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.independence_tests.cmiknn import CMIknn
from tigramite.independence_tests.gpdc import GPDC
from tigramite.independence_tests.gpdc_torch import GPDCtorch
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI
from thermodense.downloader.space_weather import SPACE_WEATHER_CSV_PATH  # noqa: E402

GLOBAL_MEAN_PATH = Path(
    "data/decoded/orbit_derived_global_mean/orbit-density-ds03-density-values.parquet"
)
CO2_PATH = Path("data/original/co2/co2_daily_mlo.csv")
SPACE_WEATHER_PATH = SPACE_WEATHER_CSV_PATH
OUTPUT_DIR = Path("outputs/figures/results")
MAUNALOA_HASDM_WIDE_PATH = Path(
    "outputs/figures/results/set_hasdm/model_validations/causal_hasdm_saber_maunaloa/hasdm_maunaloa_daily_wide.parquet"
)
HASDM_MSIS_RESIDUAL_WIDE_PATH = Path(
    "outputs/figures/results/hasdm_msis_model_errors/data/"
    "hasdm_msis_errors_nearest_timestamp_grid_daily_wide.parquet"
)
MAUNALOA_SABER_PATH = Path(
    "data/decoded/saber/saber_co2_cooling_maunaloa_daily.parquet"
)

ALL_ALTITUDES = [250, 275, 325, 375, 400, 425, 475, 525, 550, 575]
SIX_MONTH_PERIOD_DAYS = 365.25 / 2
SOLAR_CYCLE_PERIOD_DAYS = 11.4 * 365.25

BASE_LABELS = {
    "F10.7_OBS_CENTER81": "F10.7 81d avg",
    "F10.7_OBS": "F10.7 raw obs",
    "AP_AVG": "Ap",
    "KP_SUM": "Kp",
    "CO2_ppm": "CO2",
}
MSIS_MODEL_SLUGS = {
    "nrlmsise_00": "NRLMSISE-00",
    "nrlmsis_2p0": "NRLMSIS 2.0",
    "nrlmsis_2p1": "NRLMSIS 2.1",
}


@dataclass(frozen=True)
class Variant:
    name: str
    dates: np.ndarray
    data: dict[str, np.ndarray]
    description: str


@dataclass(frozen=True)
class CiTestConfig:
    name: str
    display_name: str
    edge_label: str
    signed_measure: bool
    fdr_method: str


@dataclass(frozen=True)
class LagProfile:
    name: str
    unit: str
    tau_max: int
    days_per_step: float
    description: str


@dataclass(frozen=True)
class AnalysisSelection:
    dataset_kind: str
    dataset_name: str
    altitudes: list[int]
    target_cols: list[str]
    target_labels: dict[str, str]
    cooling_altitudes: list[int]


LAG_PROFILES = {
    "daily_7d": LagProfile("daily_7d", "days", 7, 1.0, "Daily data, lags 0-7 days."),
    "weekly_4w": LagProfile(
        "weekly_4w", "weeks", 4, 7.0, "Weekly means, lags 0-4 weeks."
    ),
    "monthly_12m": LagProfile(
        "monthly_12m", "months", 12, 365.25 / 12, "Monthly means, lags 0-12 months."
    ),
    "yearly_10y": LagProfile(
        "yearly_10y", "years", 10, 365.25, "Yearly means, lags 0-10 years."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Tigramite PCMCI+ on the causal global mean dataset."
    )
    parser.add_argument(
        "--dataset",
        choices=["global_mean", "maunaloa", "maunaloa_msis_residuals"],
        default="global_mean",
        help="Dataset to analyze. Default keeps the original global-mean workflow.",
    )
    parser.add_argument(
        "--altitudes",
        nargs="+",
        default=["250", "400", "575"],
        help="Use 'all' or provide one or more altitude values in km. Default: 250 400 575.",
    )
    parser.add_argument(
        "--hasdm-altitudes",
        nargs="+",
        default=["175", "500", "825"],
        help="For --dataset maunaloa, use 'all', 'selected', or HASDM altitude values in km.",
    )
    parser.add_argument(
        "--density-metric",
        choices=["mean", "range", "both"],
        default="mean",
        help="For --dataset maunaloa choose density metrics; for maunaloa_msis_residuals choose model-error metrics.",
    )
    parser.add_argument(
        "--msis-models",
        nargs="+",
        choices=list(MSIS_MODEL_SLUGS),
        default=list(MSIS_MODEL_SLUGS),
        help="For --dataset maunaloa_msis_residuals, select MSIS model-error baselines.",
    )
    parser.add_argument(
        "--cooling-altitudes",
        nargs="+",
        default=["min", "median", "max"],
        help="For --dataset maunaloa, use 'none', 'all', min/median/max, or SABER cooling altitude values in km.",
    )
    parser.add_argument(
        "--ci-tests",
        nargs="+",
        choices=["parcorr", "cmiknn", "gpdc", "gpdctorch"],
        default=["parcorr"],
        help="Conditional independence tests to run.",
    )
    parser.add_argument("--variant", default="detrended_anomaly")
    parser.add_argument(
        "--f107",
        choices=["average", "raw"],
        default="average",
        help="Use F10.7_OBS_CENTER81 ('average') or F10.7_OBS ('raw') as the F10.7 driver.",
    )
    parser.add_argument(
        "--kp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include KP_SUM as a geomagnetic driver. Use --no-kp to exclude it.",
    )
    parser.add_argument(
        "--lag-profiles",
        nargs="+",
        choices=list(LAG_PROFILES),
        default=list(LAG_PROFILES),
        help="Lag/cadence figure sets to run.",
    )
    parser.add_argument(
        "--forbid-f107-causes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forbid every variable except F10.7 itself as a cause of the selected F10.7 driver.",
    )
    parser.add_argument(
        "--tau-max-days",
        type=int,
        default=None,
        help="Override tau_max for the daily lag profile only.",
    )
    parser.add_argument(
        "--cmiknn-tau-max-days",
        type=int,
        default=None,
        help="Override tau_max for CMIknn daily lag-profile runs only.",
    )
    parser.add_argument(
        "--gpdctorch-tau-max-days",
        type=int,
        default=None,
        help="Override tau_max for GPDCtorch daily lag-profile runs only.",
    )
    parser.add_argument("--gpdctorch-max-samples", type=int, default=3000)
    parser.add_argument("--gpdctorch-sig-samples", type=int, default=100)
    parser.add_argument("--pc-alpha", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write graph and lag-dependency PNGs.",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=1,
        help="Script verbosity. Default 1 prints concise dataset and result summaries.",
    )
    parser.add_argument(
        "--tigramite-verbosity",
        type=int,
        default=None,
        help="Override Tigramite's internal verbosity. Defaults to min(--verbosity, 1).",
    )
    parser.add_argument("--cmiknn-sig-samples", type=int, default=20)
    parser.add_argument("--cmiknn-sig-blocklength", type=int, default=4)
    parser.add_argument("--cmiknn-knn", type=float, default=0.1)
    parser.add_argument("--cmiknn-shuffle-neighbors", type=int, default=5)
    parser.add_argument("--surrogate-seed", type=int, default=20260601)
    parser.add_argument(
        "--surrogate-types",
        nargs="+",
        choices=["white", "6mo", "11p4yr"],
        default=["white", "6mo", "11p4yr"],
        help="Subset of surrogate families to include.",
    )
    parser.add_argument("--white-surrogates", type=int, default=5)
    parser.add_argument("--six-month-surrogates", type=int, default=3)
    parser.add_argument("--solar-cycle-surrogates", type=int, default=3)
    parser.add_argument(
        "--surrogates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include white-noise and sinusoidal surrogate controls.",
    )
    parser.add_argument("--workers", type=int, default=-1)
    return parser.parse_args()


def parse_altitudes(values: list[str]) -> list[int]:
    if len(values) == 1 and values[0].lower() == "all":
        return ALL_ALTITUDES

    altitudes = [int(value) for value in values]
    unknown = sorted(set(altitudes) - set(ALL_ALTITUDES))
    if unknown:
        raise ValueError(
            f"Unknown altitude(s): {unknown}. Known altitudes: {ALL_ALTITUDES}"
        )
    return altitudes


def available_maunaloa_hasdm_altitudes() -> list[int]:
    if not MAUNALOA_HASDM_WIDE_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MAUNALOA_HASDM_WIDE_PATH}. Run `uv run python -m scripts.causal_hasdm_saber_maunaloa` first."
        )
    columns = pl.scan_parquet(MAUNALOA_HASDM_WIDE_PATH).collect_schema().names()
    altitudes = sorted(
        {
            int(col.split("_")[1])
            for col in columns
            if col.startswith("log10rho_") and col.endswith("_daily_mean")
        }
    )
    return altitudes


def hasdm_msis_residual_wide_path() -> Path:
    if HASDM_MSIS_RESIDUAL_WIDE_PATH.exists():
        return HASDM_MSIS_RESIDUAL_WIDE_PATH
    raise FileNotFoundError(
        "Missing HASDM MSIS residual daily-wide cache. Run "
        "scripts/hasdm_msis_model_error_analysis.py first."
    )


def available_hasdm_msis_residual_altitudes() -> list[int]:
    columns = pl.scan_parquet(hasdm_msis_residual_wide_path()).collect_schema().names()
    altitudes = sorted(
        {
            int(col.rsplit("_", 1)[-1].replace("km", ""))
            for col in columns
            if col.startswith("nrlms") and "_daily_mean_" in col and col.endswith("km")
        }
    )
    return altitudes


def selected_maunaloa_hasdm_altitudes() -> list[int]:
    altitudes = available_maunaloa_hasdm_altitudes()
    values = np.asarray(altitudes, dtype=float)
    selected: list[int] = []
    for quantile in [0.0, 0.25, 0.5, 0.75, 1.0]:
        target = float(np.quantile(values, quantile))
        altitude = int(values[np.argmin(np.abs(values - target))])
        if altitude not in selected:
            selected.append(altitude)
    return selected


def parse_maunaloa_hasdm_altitudes(values: list[str]) -> list[int]:
    available = available_maunaloa_hasdm_altitudes()
    if len(values) == 1 and values[0].lower() == "all":
        return available
    if len(values) == 1 and values[0].lower() == "selected":
        return selected_maunaloa_hasdm_altitudes()
    altitudes = [int(value) for value in values]
    unknown = sorted(set(altitudes) - set(available))
    if unknown:
        raise ValueError(
            f"Unknown HASDM altitude(s): {unknown}. Available: {available}"
        )
    return altitudes


def parse_hasdm_msis_residual_altitudes(values: list[str]) -> list[int]:
    available = available_hasdm_msis_residual_altitudes()
    if len(values) == 1 and values[0].lower() == "all":
        return available
    altitudes = [int(value) for value in values]
    unknown = sorted(set(altitudes) - set(available))
    if unknown:
        raise ValueError(
            f"Unknown HASDM MSIS residual altitude(s): {unknown}. Available: {available}"
        )
    return altitudes


def available_saber_cooling_altitudes() -> list[int]:
    if not MAUNALOA_SABER_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MAUNALOA_SABER_PATH}. Run `uv run python -m scripts.decode_saber` first."
        )
    return [
        int(value)
        for value in pl.scan_parquet(MAUNALOA_SABER_PATH)
        .select(pl.col("altitude_km").unique().sort())
        .collect()["altitude_km"]
        .to_list()
    ]


def parse_saber_cooling_altitudes(values: list[str]) -> list[int]:
    if len(values) == 1 and values[0].lower() == "none":
        return []
    available = available_saber_cooling_altitudes()
    selectors = {value.lower() for value in values}
    if "all" in selectors:
        return available
    selected: list[int] = []
    role_map = {
        "min": min(available),
        "median": int(
            available[
                int(np.argmin(np.abs(np.asarray(available) - np.median(available))))
            ]
        ),
        "max": max(available),
    }
    for value in values:
        lowered = value.lower()
        altitude = role_map[lowered] if lowered in role_map else int(value)
        nearest = int(
            available[int(np.argmin(np.abs(np.asarray(available) - altitude)))]
        )
        if nearest not in selected:
            selected.append(nearest)
    return selected


def maunaloa_density_col(altitude: int, metric: str) -> str:
    suffix = "daily_mean" if metric == "mean" else "daily_range"
    return f"log10rho_{altitude}_{suffix}"


def maunaloa_cooling_col(altitude: int) -> str:
    return f"saber_co2cool_{altitude}km"


def hasdm_msis_residual_col(model_slug: str, altitude: int, metric: str) -> str:
    return f"{model_slug}_daily_{metric}_{altitude}km"


def build_analysis_selection(args: argparse.Namespace) -> AnalysisSelection:
    if args.dataset == "global_mean":
        altitudes = parse_altitudes(args.altitudes)
        target_cols = [density_col(altitude) for altitude in altitudes]
        return AnalysisSelection(
            dataset_kind="global_mean",
            dataset_name=dataset_name_for_altitudes(altitudes),
            altitudes=altitudes,
            target_cols=target_cols,
            target_labels={
                density_col(altitude): rf"$\ell_\rho$ {altitude} km"
                for altitude in altitudes
            },
            cooling_altitudes=[],
        )

    if args.dataset == "maunaloa_msis_residuals":
        altitudes = parse_hasdm_msis_residual_altitudes(args.hasdm_altitudes)
        cooling_altitudes = parse_saber_cooling_altitudes(args.cooling_altitudes)
        metrics = (
            ["mean", "range"]
            if args.density_metric == "both"
            else [args.density_metric]
        )
        target_cols = [
            hasdm_msis_residual_col(model_slug, altitude, metric)
            for model_slug in args.msis_models
            for altitude in altitudes
            for metric in metrics
        ]
        target_cols.extend(
            maunaloa_cooling_col(altitude) for altitude in cooling_altitudes
        )
        target_labels = {}
        for model_slug in args.msis_models:
            model_label = MSIS_MODEL_SLUGS[model_slug]
            for altitude in altitudes:
                if "mean" in metrics:
                    target_labels[
                        hasdm_msis_residual_col(model_slug, altitude, "mean")
                    ] = f"{model_label} error {altitude} km"
                if "range" in metrics:
                    target_labels[
                        hasdm_msis_residual_col(model_slug, altitude, "range")
                    ] = f"{model_label} error range {altitude} km"
        for altitude in cooling_altitudes:
            target_labels[maunaloa_cooling_col(altitude)] = (
                f"SABER CO2 cooling {altitude} km"
            )
        model_part = "_".join(args.msis_models)
        altitude_part = "_".join(str(altitude) for altitude in altitudes)
        cooling_part = (
            "none"
            if not cooling_altitudes
            else "_".join(str(altitude) for altitude in cooling_altitudes)
        )
        return AnalysisSelection(
            dataset_kind="maunaloa_msis_residuals",
            dataset_name=(
                f"maunaloa_hasdm_msis_residuals_{args.density_metric}_{model_part}_"
                f"{altitude_part}_saber_{cooling_part}"
            ),
            altitudes=altitudes,
            target_cols=target_cols,
            target_labels=target_labels,
            cooling_altitudes=cooling_altitudes,
        )

    hasdm_altitudes = parse_maunaloa_hasdm_altitudes(args.hasdm_altitudes)
    cooling_altitudes = parse_saber_cooling_altitudes(args.cooling_altitudes)
    metrics = (
        ["mean", "range"] if args.density_metric == "both" else [args.density_metric]
    )
    target_cols = [
        maunaloa_density_col(altitude, metric)
        for altitude in hasdm_altitudes
        for metric in metrics
    ]
    target_cols.extend(maunaloa_cooling_col(altitude) for altitude in cooling_altitudes)
    target_labels = {}
    for altitude in hasdm_altitudes:
        if "mean" in metrics:
            target_labels[maunaloa_density_col(altitude, "mean")] = (
                rf"HASDM mean $\ell_\rho$ {altitude} km"
            )
        if "range" in metrics:
            target_labels[maunaloa_density_col(altitude, "range")] = (
                f"HASDM log10 max/min density range {altitude} km"
            )
    for altitude in cooling_altitudes:
        target_labels[maunaloa_cooling_col(altitude)] = (
            f"SABER CO2 cooling {altitude} km"
        )
    metric_part = args.density_metric
    altitude_part = "_".join(str(altitude) for altitude in hasdm_altitudes)
    cooling_part = (
        "none"
        if not cooling_altitudes
        else "_".join(str(altitude) for altitude in cooling_altitudes)
    )
    return AnalysisSelection(
        dataset_kind="maunaloa",
        dataset_name=f"maunaloa_hasdm_{metric_part}_{altitude_part}_saber_{cooling_part}",
        altitudes=hasdm_altitudes,
        target_cols=target_cols,
        target_labels=target_labels,
        cooling_altitudes=cooling_altitudes,
    )


def ci_test_config(name: str) -> CiTestConfig:
    if name == "parcorr":
        return CiTestConfig(
            name="parcorr",
            display_name='ParCorr(significance="analytic")',
            edge_label="MCI partial correlation",
            signed_measure=True,
            fdr_method="fdr_bh",
        )
    if name == "cmiknn":
        return CiTestConfig(
            name="cmiknn",
            display_name='CMIknn(significance="shuffle_test")',
            edge_label="MCI conditional mutual information",
            signed_measure=False,
            fdr_method="none",
        )
    if name == "gpdc":
        return CiTestConfig(
            name="gpdc",
            display_name='GPDC(significance="analytic")',
            edge_label="MCI GPDC distance correlation",
            signed_measure=False,
            fdr_method="fdr_bh",
        )
    if name == "gpdctorch":
        return CiTestConfig(
            name="gpdctorch",
            display_name='GPDCtorch(significance="analytic", experimental)',
            edge_label="MCI GPDCtorch distance correlation",
            signed_measure=False,
            fdr_method="fdr_bh",
        )
    raise ValueError(f"Unsupported conditional independence test: {name}")


def effective_profile(
    profile: LagProfile, ci_config: CiTestConfig, args: argparse.Namespace
) -> LagProfile:
    override = args.tau_max_days
    if ci_config.name == "cmiknn" and args.cmiknn_tau_max_days is not None:
        override = args.cmiknn_tau_max_days
    if ci_config.name == "gpdctorch" and args.gpdctorch_tau_max_days is not None:
        override = args.gpdctorch_tau_max_days
    if override is None or profile.name != "daily_7d":
        return profile
    if override < 0:
        raise ValueError("Daily tau_max override must be non-negative.")
    return LagProfile(
        name=f"daily_{override}d",
        unit="days",
        tau_max=override,
        days_per_step=1.0,
        description=f"Daily data, lags 0-{override} days.",
    )


def make_cond_ind_test(config: CiTestConfig, args: argparse.Namespace):
    if config.name == "parcorr":
        return ParCorr(significance="analytic")
    if config.name == "cmiknn":
        return CMIknn(
            significance="shuffle_test",
            sig_samples=args.cmiknn_sig_samples,
            sig_blocklength=args.cmiknn_sig_blocklength,
            knn=args.cmiknn_knn,
            shuffle_neighbors=args.cmiknn_shuffle_neighbors,
            workers=args.workers,
        )
    if config.name == "gpdc":
        return GPDC(significance="analytic")
    if config.name == "gpdctorch":
        return GPDCtorch(
            significance="analytic",
            sig_samples=args.gpdctorch_sig_samples,
        )
    raise ValueError(f"Unsupported conditional independence test: {config.name}")


def selected_f107_col(args: argparse.Namespace) -> str:
    return "F10.7_OBS" if args.f107 == "raw" else "F10.7_OBS_CENTER81"


def driver_cols(f107_col: str, include_kp: bool) -> list[str]:
    cols = [f107_col, "AP_AVG"]
    if include_kp:
        cols.append("KP_SUM")
    return [*cols, "CO2_ppm"]


def build_link_assumptions(
    all_cols: list[str],
    tau_max: int,
    forbid_f107_causes: bool,
    f107_col: str,
) -> dict[int, dict[tuple[int, int], str]] | None:
    if not forbid_f107_causes:
        return None

    link_assumptions: dict[int, dict[tuple[int, int], str]] = {
        j: {} for j in range(len(all_cols))
    }
    f107_idx = all_cols.index(f107_col) if f107_col in all_cols else None

    for target_idx in range(len(all_cols)):
        for cause_idx in range(len(all_cols)):
            for lag in range(1, tau_max + 1):
                if (
                    f107_idx is not None
                    and target_idx == f107_idx
                    and cause_idx != f107_idx
                ):
                    continue
                link_assumptions[target_idx][(cause_idx, -lag)] = "-?>"

    for cause_idx in range(len(all_cols)):
        for target_idx in range(cause_idx + 1, len(all_cols)):
            if f107_idx is not None and cause_idx == f107_idx:
                link_assumptions[target_idx][(f107_idx, 0)] = "-?>"
            elif f107_idx is not None and target_idx == f107_idx:
                link_assumptions[cause_idx][(f107_idx, 0)] = "-?>"
            else:
                link_assumptions[target_idx][(cause_idx, 0)] = "o?o"

    return link_assumptions


def density_col(altitude: int) -> str:
    return f"log10rho_{altitude}"


def load_global_mean(target_cols: list[str]) -> pl.DataFrame:
    df = pl.read_parquet(GLOBAL_MEAN_PATH).select("date", *target_cols)
    return df.with_columns(
        [
            pl.when(pl.col(col) < -200)
            .then(None)
            .otherwise(pl.col(col))
            .interpolate()
            .alias(col)
            for col in target_cols
        ]
    )


def load_maunaloa_hasdm(target_cols: list[str]) -> pl.DataFrame:
    hasdm_cols = [col for col in target_cols if col.startswith("log10rho_")]
    if not hasdm_cols:
        return pl.read_parquet(MAUNALOA_HASDM_WIDE_PATH).select("date")
    return pl.read_parquet(MAUNALOA_HASDM_WIDE_PATH).select("date", *hasdm_cols)


def load_hasdm_msis_residuals(target_cols: list[str]) -> pl.DataFrame:
    residual_cols = [col for col in target_cols if col.startswith("nrlms")]
    if not residual_cols:
        return pl.read_parquet(hasdm_msis_residual_wide_path()).select("date")
    return pl.read_parquet(hasdm_msis_residual_wide_path()).select(
        "date", *residual_cols
    )


def load_maunaloa_saber(cooling_altitudes: list[int]) -> pl.DataFrame:
    if not cooling_altitudes:
        return pl.read_parquet(MAUNALOA_HASDM_WIDE_PATH).select("date")
    long_df = pl.read_parquet(MAUNALOA_SABER_PATH)
    available = sorted(long_df["altitude_km"].unique().to_list())
    frames = []
    for altitude in cooling_altitudes:
        nearest = float(
            available[
                int(np.argmin(np.abs(np.asarray(available, dtype=float) - altitude)))
            ]
        )
        frames.append(
            long_df.filter(pl.col("altitude_km") == nearest).select(
                "date",
                pl.col("co2_cooling_rate_w_m3").alias(
                    maunaloa_cooling_col(int(nearest))
                ),
            )
        )
    saber = frames[0]
    for frame in frames[1:]:
        saber = saber.join(frame, on="date", how="full", coalesce=True)
    return saber.sort("date")


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
        .with_columns(pl.date("year", "month", "day").alias("date"))
        .select("date", "CO2_ppm")
        .with_columns(
            pl.when(pl.col("CO2_ppm") < 0)
            .then(None)
            .otherwise(pl.col("CO2_ppm"))
            .alias("CO2_ppm")
        )
    )


def load_space_weather(f107_col: str, include_kp: bool) -> pl.DataFrame:
    cols = ["date", f107_col, "AP_AVG"]
    if include_kp:
        cols.append("KP_SUM")
    return (
        pl.read_csv(SPACE_WEATHER_PATH)
        .with_columns(pl.col("DATE").str.to_date("%Y-%m-%d").alias("date"))
        .select(*cols)
    )


def as_date_index(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(pl.col("date").cast(pl.Date))
        .filter(pl.col("date").is_not_null())
        .unique(subset="date")
        .sort("date")
    )


def combine_inputs(
    selection: AnalysisSelection, f107_col: str, include_kp: bool
) -> tuple[pl.DataFrame, pl.DataFrame]:
    all_cols = [*driver_cols(f107_col, include_kp), *selection.target_cols]
    if selection.dataset_kind == "global_mean":
        target_data = as_date_index(load_global_mean(selection.target_cols))
    elif selection.dataset_kind == "maunaloa":
        target_data = as_date_index(load_maunaloa_hasdm(selection.target_cols)).join(
            as_date_index(load_maunaloa_saber(selection.cooling_altitudes)),
            on="date",
            how="full",
            coalesce=True,
        )
    elif selection.dataset_kind == "maunaloa_msis_residuals":
        target_data = as_date_index(
            load_hasdm_msis_residuals(selection.target_cols)
        ).join(
            as_date_index(load_maunaloa_saber(selection.cooling_altitudes)),
            on="date",
            how="full",
            coalesce=True,
        )
    else:
        raise ValueError(f"Unsupported dataset kind: {selection.dataset_kind}")
    space_weather = as_date_index(load_space_weather(f107_col, include_kp))
    co2 = as_date_index(load_co2())

    target_valid = target_data.drop_nulls(selection.target_cols)

    start_date = max(
        target_valid["date"].min(),
        space_weather.filter(pl.col(f107_col).is_not_null())["date"].min(),
        co2.filter(pl.col("CO2_ppm").is_not_null())["date"].min(),
    )
    end_date = min(
        target_valid["date"].max(),
        space_weather.filter(pl.col(f107_col).is_not_null())["date"].max(),
        co2.filter(pl.col("CO2_ppm").is_not_null())["date"].max(),
    )

    combined = target_data.join(space_weather, on="date", how="full", coalesce=True)
    combined = combined.join(co2, on="date", how="full", coalesce=True)

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


def resample_for_profile(df: pl.DataFrame, profile: LagProfile) -> pl.DataFrame:
    value_cols = [col for col in df.columns if col != "date"]
    if profile.unit == "days":
        return df

    if profile.name == "weekly_4w":
        start_day = int(df.select(pl.col("date").cast(pl.Int64).min()).item())
        return (
            df.with_columns(
                ((pl.col("date").cast(pl.Int64) - start_day) // 7).alias("period_index")
            )
            .group_by("period_index")
            .agg(
                pl.col("date").min().alias("date"),
                *[pl.col(col).mean().alias(col) for col in value_cols],
            )
            .sort("period_index")
            .drop("period_index")
        )

    if profile.name == "monthly_12m":
        return (
            df.with_columns(
                pl.col("date").dt.year().alias("year"),
                pl.col("date").dt.month().alias("month"),
            )
            .group_by("year", "month")
            .agg(
                pl.col("date").min().alias("date"),
                *[pl.col(col).mean().alias(col) for col in value_cols],
            )
            .sort("year", "month")
            .drop("year", "month")
        )

    if profile.name == "yearly_10y":
        return (
            df.with_columns(pl.col("date").dt.year().alias("year"))
            .group_by("year")
            .agg(
                pl.col("date").min().alias("date"),
                *[pl.col(col).mean().alias(col) for col in value_cols],
            )
            .sort("year")
            .drop("year")
        )

    raise ValueError(f"Unsupported lag profile: {profile.name}")


def finite_standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values)
    std = np.nanstd(values)
    if not np.isfinite(std) or std == 0:
        return values - mean
    return (values - mean) / std


def make_surrogate_data(
    dates: np.ndarray,
    seed: int,
    surrogate_types: set[str],
    white_count: int,
    six_month_count: int,
    solar_cycle_count: int,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    days = dates.astype("datetime64[D]").astype("int64")
    days = days - days[0]
    rng = np.random.default_rng(seed)

    data: dict[str, np.ndarray] = {}
    labels: dict[str, str] = {}

    if "white" in surrogate_types:
        for index in range(1, white_count + 1):
            name = f"surrogate_white_noise_{index}"
            data[name] = rng.normal(0.0, 1.0, size=len(days))
            labels[name] = f"White noise {index}"

    for prefix, period_days, count, label in [
        ("surrogate_sine_6mo", SIX_MONTH_PERIOD_DAYS, six_month_count, "6-month sine"),
        (
            "surrogate_sine_11p4yr",
            SOLAR_CYCLE_PERIOD_DAYS,
            solar_cycle_count,
            "11.4-year sine",
        ),
    ]:
        if prefix == "surrogate_sine_6mo" and "6mo" not in surrogate_types:
            continue
        if prefix == "surrogate_sine_11p4yr" and "11p4yr" not in surrogate_types:
            continue
        for index in range(1, count + 1):
            phase = rng.uniform(0.0, 2 * np.pi)
            sine = np.sin(2 * np.pi * days / period_days + phase)
            sine_power = float(np.nanvar(sine))
            noise_std = np.sqrt(0.25 * sine_power)
            noise = rng.normal(0.0, noise_std, size=len(days))
            name = f"{prefix}_{index}"
            data[name] = sine + noise
            labels[name] = f"{label} + noise {index}"

    return data, labels


def with_surrogates(
    df: pl.DataFrame,
    seed: int,
    surrogate_types: set[str],
    white_count: int,
    six_month_count: int,
    solar_cycle_count: int,
) -> tuple[pl.DataFrame, list[str], dict[str, str]]:
    surrogate_data, surrogate_labels = make_surrogate_data(
        df["date"].to_numpy(),
        seed,
        surrogate_types,
        white_count,
        six_month_count,
        solar_cycle_count,
    )
    columns = [pl.Series(name, values) for name, values in surrogate_data.items()]
    if not columns:
        return df, [], {}
    return df.with_columns(columns), list(surrogate_data), surrogate_labels


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
    return np.array([date.timetuple().tm_yday for date in python_dates], dtype=int)


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
    trend = rolling_nanmean(values, window)
    return values - trend


def make_variants(df: pl.DataFrame, all_cols: list[str]) -> dict[str, Variant]:
    dates = df["date"].to_numpy()
    raw = {col: df[col].to_numpy().astype(float) for col in all_cols}
    seasonal = {col: seasonal_anomaly(raw[col], dates) for col in all_cols}
    detrended_seasonal = {col: detrended(seasonal[col]) for col in all_cols}
    co2_preserved = {
        col: raw[col] if col == "CO2_ppm" else detrended_seasonal[col]
        for col in all_cols
    }

    return {
        "raw_standardized": Variant(
            name="raw_standardized",
            dates=dates,
            data={col: finite_standardize(raw[col]) for col in all_cols},
            description="Daily values, interpolated and standardized.",
        ),
        "seasonal_anomaly": Variant(
            name="seasonal_anomaly",
            dates=dates,
            data={col: finite_standardize(seasonal[col]) for col in all_cols},
            description="Day-of-year climatology removed, then standardized.",
        ),
        "detrended_anomaly": Variant(
            name="detrended_anomaly",
            dates=dates,
            data={col: finite_standardize(detrended_seasonal[col]) for col in all_cols},
            description="Daily seasonal anomalies with a 3-year rolling mean removed; best match for PCMCI+ stationarity.",
        ),
        "co2_preserved_anomaly": Variant(
            name="co2_preserved_anomaly",
            dates=dates,
            data={col: finite_standardize(co2_preserved[col]) for col in all_cols},
            description="Daily detrended seasonal anomalies, but CO2 kept as a slow standardized raw driver.",
        ),
    }


def run_pcmciplus(
    variant: Variant,
    all_cols: list[str],
    labels: dict[str, str],
    ci_config: CiTestConfig,
    args: argparse.Namespace,
    tau_max: int,
    f107_col: str,
    pc_alpha: float,
    verbosity: int,
) -> tuple[PCMCI, dict[str, np.ndarray], np.ndarray]:
    data = np.column_stack([variant.data[col] for col in all_cols])
    var_names = [labels[col] for col in all_cols]
    dataframe = pp.DataFrame(data, datatime=np.arange(len(data)), var_names=var_names)
    cond_ind_test = make_cond_ind_test(ci_config, args)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=verbosity)

    lag_dependencies = pcmci.run_bivci(tau_max=tau_max, val_only=True)["val_matrix"]
    results = pcmci.run_pcmciplus(
        link_assumptions=build_link_assumptions(
            all_cols,
            tau_max,
            args.forbid_f107_causes,
            f107_col,
        ),
        tau_min=0,
        tau_max=tau_max,
        pc_alpha=pc_alpha,
        contemp_collider_rule="majority",
        conflict_resolution=True,
        fdr_method=ci_config.fdr_method,
    )
    return pcmci, results, lag_dependencies


def save_tigramite_plots(
    results: dict[str, np.ndarray],
    lag_dependencies: np.ndarray,
    variant_name: str,
    dataset_name: str,
    ci_config: CiTestConfig,
    var_names: list[str],
    columns: list[str],
    real_columns: list[str],
    profile: LagProfile,
    output_dir: Path,
) -> None:
    prefix = f"{ci_config.name}_{variant_name}_{dataset_name}_{profile.name}"
    lag_array = np.arange(profile.tau_max + 1)
    plot_lag_dependencies = np.array(lag_dependencies, copy=True)
    plot_results = dict(results)
    plot_results["val_matrix"] = np.array(results["val_matrix"], copy=True)
    if not ci_config.signed_measure:
        plot_lag_dependencies = np.clip(plot_lag_dependencies, 0.0, None)
        plot_results["val_matrix"] = np.clip(plot_results["val_matrix"], 0.0, None)
    finite_lag_values = plot_lag_dependencies[np.isfinite(plot_lag_dependencies)]
    if ci_config.signed_measure:
        lag_minimum = -1.0
        lag_maximum = 1.0
        vmin_edges = -1.0
        vmax_edges = 1.0
        cmap_edges = "RdBu_r"
        y_tick_step = 0.25
        colorbar_tick_step = 0.5
    else:
        lag_minimum = 0.0
        lag_maximum = (
            float(np.nanmax(finite_lag_values)) if finite_lag_values.size else 1.0
        )
        lag_maximum = max(lag_maximum, 0.01)
        graph_values = plot_results["val_matrix"][plot_results["graph"] != ""]
        graph_values = graph_values[np.isfinite(graph_values)]
        vmin_edges = 0.0
        vmax_edges = (
            float(np.nanmax(graph_values)) if graph_values.size else lag_maximum
        )
        vmax_edges = max(vmax_edges, 0.01)
        cmap_edges = "viridis"
        y_tick_step = lag_maximum / 4
        colorbar_tick_step = vmax_edges / 4

    tp.plot_lagfuncs(
        val_matrix=plot_lag_dependencies,
        name=str(output_dir / f"lag_dependencies_{prefix}.pgf"),
        setup_args={
            "var_names": var_names,
            "figsize": (14, 10),
            "x_base": max(1, profile.tau_max // 5),
            "y_base": y_tick_step,
            "minimum": lag_minimum,
            "maximum": lag_maximum,
            "lag_units": profile.unit,
            "lag_array": lag_array,
        },
    )
    plt.close("all")

    def subset_results(
        selected_columns: list[str],
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        indices = [columns.index(col) for col in selected_columns]
        subset = dict(plot_results)
        subset["graph"] = plot_results["graph"][
            np.ix_(indices, indices, np.arange(plot_results["graph"].shape[2]))
        ]
        subset["val_matrix"] = plot_results["val_matrix"][
            np.ix_(indices, indices, np.arange(plot_results["val_matrix"].shape[2]))
        ]
        subset["p_matrix"] = plot_results["p_matrix"][
            np.ix_(indices, indices, np.arange(plot_results["p_matrix"].shape[2]))
        ]
        return subset, [var_names[index] for index in indices]

    plot_sets = [("with_surrogates", plot_results, var_names)]
    if real_columns != columns:
        no_surrogate_results, no_surrogate_names = subset_results(real_columns)
        plot_sets.append(
            ("without_surrogates", no_surrogate_results, no_surrogate_names)
        )

    for graph_suffix, graph_results, graph_var_names in plot_sets:
        graph_values = graph_results["val_matrix"][graph_results["graph"] != ""]
        graph_values = graph_values[np.isfinite(graph_values)]
        if not ci_config.signed_measure and graph_values.size:
            graph_vmax_edges = max(float(np.nanmax(graph_values)), 0.01)
            graph_colorbar_tick_step = graph_vmax_edges / 4
        else:
            graph_vmax_edges = vmax_edges
            graph_colorbar_tick_step = colorbar_tick_step

        fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
        tp.plot_graph(
            fig_ax=(fig, ax),
            graph=graph_results["graph"],
            val_matrix=graph_results["val_matrix"],
            var_names=graph_var_names,
            link_colorbar_label=f"{ci_config.edge_label} (edges)",
            node_colorbar_label=f"auto-{ci_config.edge_label} (nodes)",
            vmin_edges=vmin_edges,
            vmax_edges=graph_vmax_edges,
            vmin_nodes=vmin_edges,
            vmax_nodes=graph_vmax_edges,
            cmap_edges=cmap_edges,
            cmap_nodes=cmap_edges,
            edge_ticks=graph_colorbar_tick_step,
            node_ticks=graph_colorbar_tick_step,
            node_size=0.28,
            arrow_linewidth=5.0,
        )
        fig.savefig(
            output_dir / f"process_graph_{prefix}_{graph_suffix}.pgf",
        )
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
        tp.plot_time_series_graph(
            fig_ax=(fig, ax),
            graph=graph_results["graph"],
            val_matrix=graph_results["val_matrix"],
            var_names=graph_var_names,
            link_colorbar_label=ci_config.edge_label,
            vmin_edges=vmin_edges,
            vmax_edges=graph_vmax_edges,
            cmap_edges=cmap_edges,
            edge_ticks=graph_colorbar_tick_step,
            node_size=0.045,
            arrow_linewidth=2.2,
        )
        fig.savefig(
            output_dir / f"time_series_graph_{prefix}_{graph_suffix}.pgf",
        )
        plt.close(fig)


def fdr_filtered_results(
    results: dict[str, np.ndarray], alpha_level: float
) -> dict[str, np.ndarray]:
    filtered = dict(results)
    graph = np.array(results["graph"], copy=True)
    graph[results["p_matrix"] > alpha_level] = ""
    filtered["graph"] = graph
    return filtered


def link_rows(
    results: dict[str, np.ndarray],
    all_cols: list[str],
    labels: dict[str, str],
    ci_config: CiTestConfig,
    profile: LagProfile,
) -> list[dict[str, str | int | float]]:
    graph = results["graph"]
    val_matrix = results["val_matrix"]
    p_matrix = results["p_matrix"]
    rows: list[dict[str, str | int | float]] = []
    for cause_idx, cause in enumerate(all_cols):
        for target_idx, target in enumerate(all_cols):
            for lag in range(graph.shape[2]):
                link_type = graph[cause_idx, target_idx, lag]
                if link_type == "":
                    continue
                if lag == 0 and link_type == "<--":
                    continue
                if lag == 0 and link_type in {"o-o", "x-x"} and cause_idx > target_idx:
                    continue
                if lag > 0 and link_type != "-->":
                    continue

                weight = float(val_matrix[cause_idx, target_idx, lag])
                rows.append(
                    {
                        "cause": cause,
                        "cause_label": labels[cause],
                        "target": target,
                        "target_label": labels[target],
                        "lag_index": lag,
                        "lag_units": profile.unit,
                        "lag_value": lag,
                        "lag_days_approx": lag * profile.days_per_step,
                        "link_type": str(link_type),
                        "mci_value": weight,
                        "abs_mci_value": abs(weight),
                        "p_value": float(p_matrix[cause_idx, target_idx, lag]),
                        "p_value_method": ci_config.fdr_method,
                    }
                )
    return sorted(rows, key=lambda row: row["abs_mci_value"], reverse=True)


def save_weight_heatmap(
    rows: list[dict[str, str | int | float]],
    all_cols: list[str],
    labels: dict[str, str],
    variant_name: str,
    dataset_name: str,
    ci_config: CiTestConfig,
    profile: LagProfile,
    output_dir: Path,
) -> None:
    matrix = np.full((len(all_cols), len(all_cols)), np.nan)
    lag_labels = [["" for _ in all_cols] for _ in all_cols]
    col_index = {col: idx for idx, col in enumerate(all_cols)}

    for row in rows:
        i = col_index[str(row["cause"])]
        j = col_index[str(row["target"])]
        current = matrix[i, j]
        weight = float(row["mci_value"])
        if not np.isfinite(current) or abs(weight) > abs(current):
            matrix[i, j] = weight
            unit_suffix = {"days": "d", "weeks": "w", "months": "mo", "years": "y"}[
                str(row["lag_units"])
            ]
            lag_labels[i][j] = f"{int(row['lag_value'])}{unit_suffix}"

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    if ci_config.signed_measure:
        image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
        colorbar_ticks = np.linspace(-1, 1, 5)
    else:
        matrix = np.clip(matrix, 0.0, None)
        finite_values = matrix[np.isfinite(matrix)]
        vmax = float(np.nanmax(finite_values)) if finite_values.size else 1.0
        vmax = max(vmax, 0.01)
        image = ax.imshow(matrix, vmin=0, vmax=vmax, cmap="viridis")
        colorbar_ticks = np.linspace(0, vmax, 5)
    display_names = [labels[col] for col in all_cols]
    ax.set_xticks(np.arange(len(all_cols)), display_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(all_cols)), display_names)
    ax.set_xlabel("Target")
    ax.set_ylabel("Cause")
    ax.set_title(f"Strongest discovered {ci_config.edge_label} by variable pair")
    for i in range(len(all_cols)):
        for j in range(len(all_cols)):
            if np.isfinite(matrix[i, j]):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.2f}\n{lag_labels[i][j]}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if abs(matrix[i, j]) > 0.45 else "black",
                )
    colorbar = fig.colorbar(image, ax=ax, label=ci_config.edge_label)
    colorbar.set_ticks(colorbar_ticks)
    fig.savefig(
        output_dir
        / f"mci_weight_heatmap_{ci_config.name}_{variant_name}_{dataset_name}_{profile.name}.pgf",
    )
    plt.close(fig)


def write_notes(
    output_dir: Path,
    variant: Variant,
    selection: AnalysisSelection,
    all_cols: list[str],
    ci_config: CiTestConfig,
    profile: LagProfile,
    pc_alpha: float,
    n_rows: int,
    link_count: int,
    surrogate_enabled: bool,
    surrogate_description: str,
    forbid_f107_causes: bool,
    include_kp: bool,
) -> None:
    notes = f"""# Tigramite Causal Global Mean Setup

Best setup used here:
- Daily merged density, space-weather, and CO2 dataset.
- Surrogate controls: {surrogate_description if surrogate_enabled else "disabled"}.
- Dataset: `{selection.dataset_kind}`.
- HASDM/global density altitudes in the joint graph: {", ".join(str(altitude) for altitude in selection.altitudes)} km.
- SABER cooling altitudes in the joint graph: {", ".join(str(altitude) for altitude in selection.cooling_altitudes) if selection.cooling_altitudes else "none"} km.
- Variables in the joint graph: {", ".join(all_cols)}.
- KP_SUM driver included: {include_kp}.
- Variant: `{variant.name}`. {variant.description}
- Method: PCMCI+ with `{ci_config.display_name}`.
- Edge weight: {ci_config.edge_label}.
- Lag profile: `{profile.name}`. {profile.description}
- External causes of F10.7 forbidden by link assumptions: {forbid_f107_causes}. F10.7 self-lags remain allowed.
- GPDCtorch sample cap, if applicable: {"not applicable" if ci_config.name != "gpdctorch" else "see --gpdctorch-max-samples; rows are evenly spaced before preprocessing"}.
- `tau_min=0`, `contemp_collider_rule="majority"`, `conflict_resolution=True`, and `fdr_method="{ci_config.fdr_method}"`.
- Saved graph outputs are filtered to links with p-values <= `pc_alpha`; for ParCorr/GPDC these are FDR-adjusted, for CMIknn they are shuffle-test p-values.
- `pc_alpha={pc_alpha}`.
- Samples in this profile: {n_rows}.
- Discovered links in exported table: {link_count}.

Why this setup:
- Tigramite tutorials recommend plotting lagged dependencies first, then choosing `tau_max` from domain knowledge and lag peaks.
- PCMCI+ is preferred over standard PCMCI when same-day and lagged links may both exist.
- By default this script runs ParCorr. `gpdctorch` is available as an experimental GPU-oriented nonlinear option via `--ci-tests gpdctorch`.
- `detrended_anomaly` is the default because PCMCI+ assumes causal stationarity; it also detrends CO2 rather than preserving its raw long-term trend.
- Sinusoidal surrogate controls use additive white noise with variance equal to 0.25 of the sine variance.

Outputs:
- `lag_dependencies_*.png`: bivariate conditional lag scan used to assess lag structure.
- `process_graph_*.png`: aggregated causal graph, colored by MCI edge weights.
- `time_series_graph_*.png`: lag-resolved causal graph.
- `mci_weight_heatmap_*.png`: strongest discovered edge weight per cause-target pair.
- `links_*.csv`: link table with lag, graph mark, MCI value, p-value, and p-value method.
"""
    (output_dir / "README.md").write_text(notes, encoding="utf-8")


def links_dataframe(rows: list[dict[str, str | int | float]]) -> pl.DataFrame:
    schema = {
        "cause": pl.String,
        "cause_label": pl.String,
        "target": pl.String,
        "target_label": pl.String,
        "lag_index": pl.Int64,
        "lag_units": pl.String,
        "lag_value": pl.Int64,
        "lag_days_approx": pl.Float64,
        "link_type": pl.String,
        "mci_value": pl.Float64,
        "abs_mci_value": pl.Float64,
        "p_value": pl.Float64,
        "p_value_method": pl.String,
    }
    return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)


def dataset_name_for_altitudes(altitudes: list[int]) -> str:
    return "altitudes_" + "_".join(str(altitude) for altitude in altitudes)


def limit_samples(df: pl.DataFrame, max_samples: int) -> pl.DataFrame:
    if max_samples <= 0 or len(df) <= max_samples:
        return df
    indices = np.unique(np.linspace(0, len(df) - 1, max_samples, dtype=int))
    return (
        df.with_row_index("_row_index")
        .filter(pl.col("_row_index").is_in(indices.tolist()))
        .drop("_row_index")
    )


def surrogate_description(args: argparse.Namespace) -> str:
    if not args.surrogates:
        return "disabled"
    parts = []
    if "white" in args.surrogate_types and args.white_surrogates > 0:
        parts.append(f"{args.white_surrogates} white-noise")
    if "6mo" in args.surrogate_types and args.six_month_surrogates > 0:
        parts.append(f"{args.six_month_surrogates} six-month sine + white noise")
    if "11p4yr" in args.surrogate_types and args.solar_cycle_surrogates > 0:
        parts.append(f"{args.solar_cycle_surrogates} 11.4-year sine + white noise")
    return "enabled (" + ", ".join(parts) + ")" if parts else "enabled (none selected)"


def causal_output_root(output_root: Path, selection: AnalysisSelection) -> Path:
    result_section = (
        "global_mean" if selection.dataset_kind == "global_mean" else "set_hasdm"
    )
    return output_root / result_section / "causal"


def run_one_analysis(
    selection: AnalysisSelection,
    ci_config: CiTestConfig,
    profile: LagProfile,
    args: argparse.Namespace,
) -> dict[str, str | int | float | None]:
    f107_col = selected_f107_col(args)
    dataset_name = selection.dataset_name
    output_dir = (
        causal_output_root(args.output_dir, selection)
        / ci_config.name
        / dataset_name
        / profile.name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    target_cols = selection.target_cols
    real_cols = [*driver_cols(f107_col, args.kp), *target_cols]
    labels = {**BASE_LABELS, **selection.target_labels}

    daily_df, missing_summary = combine_inputs(selection, f107_col, args.kp)
    if ci_config.name == "gpdctorch":
        original_len = len(daily_df)
        daily_df = limit_samples(daily_df, args.gpdctorch_max_samples)
        if args.verbosity >= 1 and len(daily_df) != original_len:
            tqdm.write(
                f"GPDCtorch sample limit: using {len(daily_df)} evenly spaced "
                f"samples from {original_len} daily rows."
            )
    if args.surrogates:
        daily_df, surrogate_cols, surrogate_labels = with_surrogates(
            daily_df,
            args.surrogate_seed,
            set(args.surrogate_types),
            args.white_surrogates,
            args.six_month_surrogates,
            args.solar_cycle_surrogates,
        )
        all_cols = [*real_cols, *surrogate_cols]
        labels = {**labels, **surrogate_labels}
    else:
        all_cols = real_cols

    missing_summary.write_csv(output_dir / f"missing_summary_{dataset_name}.csv")
    daily_df.write_csv(output_dir / f"daily_analysis_dataset_{dataset_name}.csv")
    profile_df = resample_for_profile(daily_df, profile)
    profile_df.write_csv(
        output_dir / f"analysis_dataset_{dataset_name}_{profile.name}.csv"
    )

    variants = make_variants(profile_df, all_cols)
    if args.variant not in variants:
        raise ValueError(
            f"Unknown variant {args.variant!r}. Choose one of: {', '.join(variants)}"
        )
    variant = variants[args.variant]

    if args.verbosity >= 1:
        tqdm.write(
            f"Dataset {dataset_name}/{profile.name}: {len(variant.dates)} samples, "
            f"{len(all_cols)} variables, tau_max={profile.tau_max} {profile.unit}, "
            f"method={ci_config.name}, F10.7={f107_col}, "
            f"KP_SUM={'enabled' if args.kp else 'disabled'}, "
            f"surrogates={surrogate_description(args)}"
        )
        if args.verbosity >= 2:
            tqdm.write("Variables: " + ", ".join(labels[col] for col in all_cols))

    pcmci, results, lag_dependencies = run_pcmciplus(
        variant=variant,
        all_cols=all_cols,
        labels=labels,
        ci_config=ci_config,
        args=args,
        tau_max=profile.tau_max,
        f107_col=f107_col,
        pc_alpha=args.pc_alpha,
        verbosity=(
            args.tigramite_verbosity
            if args.tigramite_verbosity is not None
            else min(args.verbosity, 1)
        ),
    )
    _ = pcmci
    filtered_results = fdr_filtered_results(results, args.pc_alpha)

    var_names = [labels[col] for col in all_cols]
    if args.plot:
        save_tigramite_plots(
            results=filtered_results,
            lag_dependencies=lag_dependencies,
            variant_name=variant.name,
            dataset_name=dataset_name,
            ci_config=ci_config,
            var_names=var_names,
            columns=all_cols,
            real_columns=real_cols,
            profile=profile,
            output_dir=output_dir,
        )

    rows = link_rows(filtered_results, all_cols, labels, ci_config, profile)
    density_rows = [row for row in rows if row["target"] in target_cols]
    surrogate_rows = [
        row
        for row in rows
        if str(row["cause"]).startswith("surrogate_")
        or str(row["target"]).startswith("surrogate_")
    ]
    if args.verbosity >= 1:
        tqdm.write(
            f"{ci_config.name} discovered {len(rows)} links "
            f"({len(density_rows)} into density variables, "
            f"{len(surrogate_rows)} involving surrogates)."
        )
    links_df = links_dataframe(rows)
    links_df.write_csv(
        output_dir
        / f"links_{ci_config.name}_{variant.name}_{dataset_name}_{profile.name}.csv"
    )
    if args.plot:
        save_weight_heatmap(
            rows,
            all_cols,
            labels,
            variant.name,
            dataset_name,
            ci_config,
            profile,
            output_dir,
        )
    write_notes(
        output_dir=output_dir,
        variant=variant,
        selection=selection,
        all_cols=all_cols,
        ci_config=ci_config,
        profile=profile,
        pc_alpha=args.pc_alpha,
        n_rows=len(variant.dates),
        link_count=len(rows),
        surrogate_enabled=args.surrogates,
        surrogate_description=surrogate_description(args),
        forbid_f107_causes=args.forbid_f107_causes,
        include_kp=args.kp,
    )

    print(f"Saved Tigramite causal graph outputs to {output_dir}")
    if rows:
        top_n = min(20, len(rows))
        tqdm.write(f"Top {top_n} links for {ci_config.name} on {dataset_name}:")
        print(links_df.head(top_n))
    top_density = density_rows[0] if density_rows else None
    return {
        "ci_test": ci_config.name,
        "dataset": dataset_name,
        "lag_profile": profile.name,
        "altitudes_km": ",".join(str(altitude) for altitude in selection.altitudes),
        "cooling_altitudes_km": ",".join(
            str(altitude) for altitude in selection.cooling_altitudes
        ),
        "variant": variant.name,
        "f107": f107_col,
        "kp_enabled": args.kp,
        "tau_max": profile.tau_max,
        "lag_units": profile.unit,
        "tau_max_days_approx": profile.tau_max * profile.days_per_step,
        "samples": len(variant.dates),
        "variable_count": len(all_cols),
        "link_count": len(rows),
        "density_link_count": len(density_rows),
        "surrogate_link_count": len(surrogate_rows),
        "top_density_target": (
            None if top_density is None else str(top_density["target"])
        ),
        "top_density_cause": None if top_density is None else str(top_density["cause"]),
        "top_density_lag_value": (
            None if top_density is None else int(top_density["lag_value"])
        ),
        "top_density_lag_units": (
            None if top_density is None else str(top_density["lag_units"])
        ),
        "top_density_lag_days_approx": (
            None if top_density is None else float(top_density["lag_days_approx"])
        ),
        "top_density_link_type": (
            None if top_density is None else str(top_density["link_type"])
        ),
        "top_density_mci_value": (
            None if top_density is None else float(top_density["mci_value"])
        ),
        "top_density_p_value": (
            None if top_density is None else float(top_density["p_value"])
        ),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection = build_analysis_selection(args)
    output_root = causal_output_root(args.output_dir, selection)
    output_root.mkdir(parents=True, exist_ok=True)
    ci_configs = [ci_test_config(name) for name in args.ci_tests]
    profiles = [LAG_PROFILES[name] for name in args.lag_profiles]

    jobs = [
        (ci_config, effective_profile(profile, ci_config, args))
        for ci_config in ci_configs
        for profile in profiles
    ]
    summary_rows = []
    for ci_config, profile in tqdm(jobs, desc="Tigramite analyses", unit="run"):
        tqdm.write(
            f"Running {ci_config.name} on {selection.dataset_name}/{profile.name}"
        )
        summary_rows.append(run_one_analysis(selection, ci_config, profile, args))

    summary_df = pl.DataFrame(summary_rows)
    methods = "_".join(config.name for config in ci_configs)
    profile_names = "_".join(profile.name for _, profile in jobs)
    summary_path = output_root / f"summary_{args.variant}_{methods}_{profile_names}.csv"
    summary_df.write_csv(summary_path)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
