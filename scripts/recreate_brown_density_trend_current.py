"""Recreate the current direct calendar-time density trend figure.

The script compares altitude-resolved fitted density trends for three product
families: Global mean thermospheric density, the TU Delft satellite density
dataset, the Mauna Loa HASDM subset, and Mauna Loa MSIS density baselines. All
input density series are expected to be daily mean ell_rho columns. The
fitted slope is a linear calendar-time trend in log10(rho) after removing the
dominant solar-cycle dependence with F10.7_81 and F10.7_81 squared terms.
Newey-West/HAC 95% confidence intervals use Bartlett weights over a 27-day
maximum lag. Log10-slope interval endpoints are converted separately to
percent per decade for plotting and CSV export.

The resulting trend is descriptive rather than a deconfounded secular cooling
estimate. It preserves each product's sampling interval and altitude coverage,
while reducing the largest solar-cycle sequencing bias in a way that is closer
to the trend definitions summarized by Brown et al. (2024).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from scripts.pgf_config import configure_pgf
from thermodense.downloader.space_weather import SPACE_WEATHER_CSV_PATH

configure_pgf()

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

GLOBAL_MEAN_PATH = Path(
    "data/decoded/orbit_derived_global_mean/orbit-density-ds03-density-values.parquet"
)
HASDM_PATH = Path(
    "outputs/figures/results/set_hasdm/model_validations/causal_hasdm_saber_maunaloa/hasdm_maunaloa_daily_long.parquet"
)
TUDELFT_PATH = Path(
    "outputs/figures/results/tudelft_density/model_validations/causal_tudelft_density/tudelft_daily_altitude_binned_long.parquet"
)
MSIS_PATH = Path(
    "outputs/figures/results/maunaloa_msis_density_baselines/data/maunaloa_msis_density_baselines_daily_wide.parquet"
)
JB_PATH = Path(
    "outputs/figures/results/maunaloa_jb_density_baselines/data/"
    "maunaloa_jb_density_baselines_daily_wide.parquet"
)
SPACE_WEATHER_PATH = SPACE_WEATHER_CSV_PATH

OUTPUT_ROOT = Path("outputs/figures/results/current_density_trends")

MIN_SAMPLES = 365
MIN_DURATION_YEARS = 11.0
DURATION_INCREMENT_YEARS = 11
TUDELFT_ALTITUDE_BIN_KM = 25
TUDELFT_MISSION_MIN_DAILY_SAMPLES = 20
HAC_MAX_LAG_DAYS = 27
HAC_CONFIDENCE_Z_95 = 1.959963984540054

F107_COL = "F10.7_OBS_CENTER81"
TUDELFT_MISSIONS = (
    (
        "CHAMP",
        Path("data/analyzed/tudelft/champ/CH_analyzed.parquet"),
        (300.0, 500.0),
    ),
    ("GOCE", Path("data/analyzed/tudelft/goce/GO_analyzed.parquet"), (225.0, 300.0)),
    (
        "GRACE-A",
        Path("data/analyzed/tudelft/grace/GA_analyzed.parquet"),
        (425.0, 500.0),
    ),
    (
        "GRACE-B",
        Path("data/analyzed/tudelft/grace/GB_analyzed.parquet"),
        (425.0, 500.0),
    ),
    (
        "GRACE-FO",
        Path("data/analyzed/tudelft/grace_fo/GC_analyzed.parquet"),
        (475.0, 525.0),
    ),
    (
        "Swarm-A",
        Path("data/analyzed/tudelft/swarm/SA_analyzed.parquet"),
        (425.0, 500.0),
    ),
    (
        "Swarm-B",
        Path("data/analyzed/tudelft/swarm/SB_analyzed.parquet"),
        (500.0, 525.0),
    ),
    (
        "Swarm-C",
        Path("data/analyzed/tudelft/swarm/SC_analyzed.parquet"),
        (450.0, 525.0),
    ),
)
DATASET_LINESTYLES = {
    "Global mean thermospheric density": "-",
    "TU Delft satellite density dataset": (0, (5, 1)),
    "Mauna Loa HASDM subset": "--",
    "NRLMSISE-00 Mauna Loa baseline": ":",
    "NRLMSIS 2.0 Mauna Loa baseline": "-.",
    "NRLMSIS 2.1 Mauna Loa baseline": (0, (3, 1, 1, 1, 1, 1)),
    "JB2006 Mauna Loa baseline": ":",
    "JB2008 Mauna Loa baseline": (0, (5, 2, 1, 2)),
}
DATASET_LABELS = {
    "Global mean thermospheric density": "Gbl. Mean",
    "TU Delft satellite density dataset": "TU Delft",
    "Mauna Loa HASDM subset": "HASDM",
    "NRLMSISE-00 Mauna Loa baseline": "NRLMSISE-00",
    "NRLMSIS 2.0 Mauna Loa baseline": "NRLMSIS 2.0",
    "NRLMSIS 2.1 Mauna Loa baseline": "NRLMSIS 2.1",
    "JB2006 Mauna Loa baseline": "JB2006",
    "JB2008 Mauna Loa baseline": "JB2008",
}

DATASET_COLORS = {
    "Global mean thermospheric density": "#1f77b4",
    "TU Delft satellite density dataset": "#2ca02c",
    "Mauna Loa HASDM subset": "#ff7f0e",
    "NRLMSISE-00 Mauna Loa baseline": "#9467bd",
    "NRLMSIS 2.0 Mauna Loa baseline": "#8c564b",
    "NRLMSIS 2.1 Mauna Loa baseline": "#d62728",
    "JB2006 Mauna Loa baseline": "#17becf",
    "JB2008 Mauna Loa baseline": "#bcbd22",
}

DURATION_MARKERS = {
    0: "o",
    11: "s",
    22: "D",
    33: "^",
    44: "P",
    55: "X",
}

plt.rcParams.update(
    {
        "font.size": 14,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.labelsize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 14,
        "axes.titlesize": 14,
    }
)


@dataclass(frozen=True)
class SeriesSpec:
    """Metadata needed to fit one altitude-resolved log-density series.

    Attributes:
        dataset: Canonical dataset label used in outputs and plotting styles.
        altitude_km: Altitude represented by ``value_col``.
        date_col: Date column name in ``df``.
        value_col: Column containing daily mean ell_rho values.
        df: Data frame containing the selected density series and drivers.
        f107_col: Solar-activity column used for solar-cycle adjustment.
        categorical_col: Optional categorical offset term, used for TU Delft
            mission identity when fitting pooled altitude-bin trends.
    """

    dataset: str
    altitude_km: float
    date_col: str
    value_col: str
    df: pl.DataFrame
    f107_col: str = F107_COL
    categorical_col: str | None = None


def decimal_year(dates: np.ndarray) -> np.ndarray:
    """Convert date objects to decimal years for least-squares fitting."""

    years = np.array([d.year for d in dates], dtype=float)
    year_starts = np.array(
        [d.replace(month=1, day=1).toordinal() for d in dates], dtype=float
    )
    next_year_starts = np.array(
        [d.replace(year=d.year + 1, month=1, day=1).toordinal() for d in dates],
        dtype=float,
    )
    ordinals = np.array([d.toordinal() for d in dates], dtype=float)
    return years + (ordinals - year_starts) / (next_year_starts - year_starts)


def newey_west_covariance(
    design: np.ndarray,
    residuals: np.ndarray,
    day_index: np.ndarray,
    max_lag: int = HAC_MAX_LAG_DAYS,
) -> np.ndarray:
    """Return the Newey-West covariance matrix for an OLS design matrix.

    Row score vectors are summed by ``day_index``, then reindexed across the
    complete calendar-day range with zeros for unobserved dates. The estimator
    consequently uses Bartlett weights through ``max_lag`` calendar days even
    when observations have duplicate or missing dates. Moore-Penrose
    pseudoinverses form the bread matrix. The design may contain arbitrary
    regressors; this script places its intercept in the final column. A 27-day
    default captures approximately one solar rotation while avoiding an overly
    long bandwidth for daily records.
    """

    design = np.asarray(design, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    day_index = np.asarray(day_index, dtype=np.int64)
    if (
        design.ndim != 2
        or residuals.ndim != 1
        or day_index.ndim != 1
        or len(design) != len(residuals)
        or len(design) != len(day_index)
    ):
        raise ValueError(
            "design, residuals, and day_index must have matching row counts"
        )
    if max_lag < 0:
        raise ValueError("max_lag must be non-negative")
    if len(day_index) == 0:
        raise ValueError("day_index must contain at least one calendar day")

    scores = design * residuals[:, None]
    unique_days, inverse = np.unique(day_index, return_inverse=True)
    daily_scores = np.zeros((unique_days.size, design.shape[1]))
    np.add.at(daily_scores, inverse, scores)
    complete_daily_scores = np.zeros(
        (int(unique_days[-1] - unique_days[0]) + 1, design.shape[1])
    )
    complete_daily_scores[unique_days - unique_days[0]] = daily_scores

    meat = complete_daily_scores.T @ complete_daily_scores
    effective_lag = min(max_lag, len(complete_daily_scores) - 1)
    for lag in range(1, effective_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        lagged_cross_product = (
            complete_daily_scores[lag:].T @ complete_daily_scores[:-lag]
        )
        meat += weight * (lagged_cross_product + lagged_cross_product.T)

    bread = np.linalg.pinv(design.T @ design)
    return bread @ meat @ bread


def log10_slope_ci_to_percent_per_decade(
    lower_log10_per_year: float, upper_log10_per_year: float
) -> tuple[float, float]:
    """Convert ordered log10 slope endpoints to percent-per-decade endpoints.

    Transforming each endpoint preserves the asymmetric interval induced by the
    exponential density conversion rather than applying a delta-method error.
    """

    if lower_log10_per_year > upper_log10_per_year:
        raise ValueError("lower log10 slope endpoint must not exceed upper endpoint")
    return tuple(
        float((10 ** (endpoint * 10.0) - 1.0) * 100.0)
        for endpoint in (lower_log10_per_year, upper_log10_per_year)
    )


def duration_bin(duration_years: float) -> int:
    """Map a record duration to the marker-bin lower edge used in the plot."""

    if not np.isfinite(duration_years) or duration_years < DURATION_INCREMENT_YEARS:
        return 0
    return int(
        np.floor(duration_years / DURATION_INCREMENT_YEARS) * DURATION_INCREMENT_YEARS
    )


def duration_label(bin_start: int) -> str:
    """Return the legend label for a duration-bin lower edge."""

    if bin_start == 0:
        return "$<$11 yr"
    return f"{bin_start} to {bin_start + DURATION_INCREMENT_YEARS} yr"


def load_space_weather() -> pl.DataFrame:
    """Load daily F10.7 values used for solar-cycle adjustment."""

    return (
        pl.read_csv(SPACE_WEATHER_PATH)
        .with_columns(pl.col("DATE").str.to_date("%Y-%m-%d").alias("date"))
        .select("date", F107_COL)
    )


def fit_trend(spec: SeriesSpec) -> dict[str, object] | None:
    """Fit one solar-adjusted log10-density trend.

    Returns ``None`` when the series does not have enough finite samples or does
    not span the required minimum duration. The regression model is
    ``log10rho ~ time + F10.7_81 + F10.7_81^2``. The reported percent trend is
    the multiplicative density change implied by the fitted log10 time slope
    over one decade: ``(10 ** (slope_log10_per_year * 10) - 1) * 100``. Its
    95% interval uses the 27-day Bartlett Newey-West/HAC covariance estimate;
    both log10 interval endpoints are transformed independently.
    """

    selected_cols = [spec.date_col, spec.value_col]
    if spec.categorical_col is not None and spec.categorical_col in spec.df.columns:
        selected_cols.append(spec.categorical_col)
    if spec.f107_col not in spec.df.columns:
        return None
    selected_cols.append(spec.f107_col)
    df = (
        spec.df.select(selected_cols)
        .rename({spec.f107_col: F107_COL})
        .drop_nulls()
        .filter(pl.col(spec.value_col).is_finite())
        .sort(spec.date_col)
    )
    if df.height < MIN_SAMPLES:
        return None

    dates = np.array(df[spec.date_col].to_list(), dtype=object)
    values = df[spec.value_col].to_numpy().astype(float)
    f107 = df[F107_COL].to_numpy().astype(float)
    finite = np.isfinite(values) & (values > -100.0) & np.isfinite(f107)
    categories = None
    if spec.categorical_col is not None and spec.categorical_col in df.columns:
        categories = np.array(df[spec.categorical_col].to_list(), dtype=object)[finite]
    dates = dates[finite]
    values = values[finite]
    f107 = f107[finite]
    if len(values) < MIN_SAMPLES:
        return None

    start = dates[0]
    end = dates[-1]
    duration_years = (end - start).days / 365.2425
    if duration_years < MIN_DURATION_YEARS:
        return None

    x = decimal_year(dates)
    if np.std(x) == 0 or np.std(values) == 0:
        return None

    centered_year = x - np.mean(x)
    centered_f107 = f107 - np.mean(f107)
    design_cols = [centered_year, centered_f107, centered_f107**2]
    if categories is not None:
        for category in sorted(set(categories))[1:]:
            design_cols.append((categories == category).astype(float))
    design_cols.append(np.ones_like(centered_year))
    design = np.column_stack(design_cols)

    # Values are log10(rho). The time coefficient is therefore a logarithmic
    # density trend after accounting for the dominant solar-cycle dependence.
    params = np.linalg.lstsq(design, values, rcond=None)[0]
    slope_log10_per_year = params[0]
    trend_percent_per_decade = (10 ** (slope_log10_per_year * 10.0) - 1.0) * 100.0
    residuals = values - design @ params
    rmse_log10 = float(np.sqrt(np.mean(residuals**2)))
    day_index = np.array([date.toordinal() for date in dates], dtype=int)
    hac_covariance = newey_west_covariance(design, residuals, day_index)
    slope_hac_stderr = float(np.sqrt(max(0.0, hac_covariance[0, 0])))
    slope_hac_ci_lower = float(
        slope_log10_per_year - HAC_CONFIDENCE_Z_95 * slope_hac_stderr
    )
    slope_hac_ci_upper = float(
        slope_log10_per_year + HAC_CONFIDENCE_Z_95 * slope_hac_stderr
    )
    trend_hac_ci_lower, trend_hac_ci_upper = log10_slope_ci_to_percent_per_decade(
        slope_hac_ci_lower, slope_hac_ci_upper
    )
    bin_start = duration_bin(duration_years)
    return {
        "dataset": spec.dataset,
        "altitude_km": spec.altitude_km,
        "start_date": start,
        "end_date": end,
        "duration_years": duration_years,
        "duration_bin_years": bin_start,
        "duration_label": duration_label(bin_start),
        "f107_mean": float(np.mean(f107)),
        "samples": len(values),
        "slope_log10_per_year": float(slope_log10_per_year),
        "hac_max_lag_days": HAC_MAX_LAG_DAYS,
        "slope_log10_per_year_hac_stderr": slope_hac_stderr,
        "slope_log10_per_year_hac_95_ci_lower": slope_hac_ci_lower,
        "slope_log10_per_year_hac_95_ci_upper": slope_hac_ci_upper,
        "trend_percent_per_decade": float(trend_percent_per_decade),
        "trend_percent_per_decade_hac_95_ci_lower": trend_hac_ci_lower,
        "trend_percent_per_decade_hac_95_ci_upper": trend_hac_ci_upper,
        "rmse_log10": rmse_log10,
    }


def safe_model_name(value: str) -> str:
    """Convert a model display name to the prefix used in MSIS output columns."""

    return value.lower().replace(" ", "_").replace(".", "p").replace("-", "_")


def global_mean_specs() -> list[SeriesSpec]:
    """Build trend specifications for Global mean thermospheric density."""

    df = (
        load_space_weather()
        .join(
            pl.read_parquet(GLOBAL_MEAN_PATH).with_columns(
                pl.col("date").cast(pl.Date)
            ),
            on="date",
            how="inner",
        )
        .sort("date")
    )
    specs: list[SeriesSpec] = []
    for col in df.columns:
        if not col.startswith("log10rho_"):
            continue
        altitude_label = col.removeprefix("log10rho_")
        altitude = float(altitude_label)
        specs.append(
            SeriesSpec(
                "Global mean thermospheric density",
                altitude,
                "date",
                col,
                df,
            )
        )
    return specs


def long_altitude_specs(
    path: Path,
    dataset: str,
    altitude_col: str,
    value_col: str,
) -> list[SeriesSpec]:
    """Build per-altitude trend specifications from a long daily data table.

    The current caller uses this for the Mauna Loa HASDM subset. The density
    column is expected to already contain daily mean ell_rho values.
    """

    df = (
        load_space_weather()
        .join(
            pl.read_parquet(path).with_columns(pl.col("date").cast(pl.Date)),
            on="date",
            how="inner",
        )
        .sort("date")
    )
    specs: list[SeriesSpec] = []
    for altitude in df[altitude_col].unique().sort().to_list():
        alt_df = df.filter(pl.col(altitude_col) == altitude).select(
            "date", F107_COL, value_col
        )
        specs.append(SeriesSpec(dataset, float(altitude), "date", value_col, alt_df))
    return specs


def tudelft_specs() -> list[SeriesSpec]:
    """Build trend specifications for long TU Delft altitude bins.

    Only altitude bins with more than 11 years between the first and last valid
    daily mean sample are retained. Mission identity is preserved as a categorical
    offset in the fit to avoid treating inter-mission offsets as secular trends.
    """

    mission_frames = []
    for mission, path, (min_altitude, max_altitude) in TUDELFT_MISSIONS:
        scan = pl.scan_parquet(path)
        quality_filter = (pl.col("Altitude (m)") / 1000.0).is_between(
            min_altitude, max_altitude
        ) & (pl.col("Density (kg/m^3)") > 0)
        if mission not in {"Swarm-A", "Swarm-B", "Swarm-C"}:
            if "Anomalus Density (kg/m^3)" in scan.collect_schema().names():
                quality_filter = quality_filter & (
                    pl.col("Anomalus Density (kg/m^3)") == 0
                )
            if "Anomalus Density Mean (kg/m^3)" in scan.collect_schema().names():
                quality_filter = quality_filter & (
                    pl.col("Anomalus Density Mean (kg/m^3)") == 0
                )
        if (
            mission == "GOCE"
            and "Degraded Flag Thrusters" in scan.collect_schema().names()
        ):
            quality_filter = quality_filter & (pl.col("Degraded Flag Thrusters") == 0)

        mission_frames.append(
            scan.filter(quality_filter)
            .with_columns(
                pl.col("timestamp").dt.date().alias("date"),
                (pl.col("Density (kg/m^3)").log10()).alias("log10rho_daily_mean"),
                (
                    (
                        (pl.col("Altitude (m)") / 1000.0) / TUDELFT_ALTITUDE_BIN_KM
                    ).floor()
                    * TUDELFT_ALTITUDE_BIN_KM
                )
                .cast(pl.Int32)
                .alias("altitude_bin_km"),
                pl.lit(mission).alias("mission"),
            )
            .group_by("date", "altitude_bin_km", "mission")
            .agg(
                pl.col("log10rho_daily_mean").mean(),
                pl.len().alias("daily_samples"),
            )
            .filter(pl.col("daily_samples") >= TUDELFT_MISSION_MIN_DAILY_SAMPLES)
            .collect()
        )

    df = (
        load_space_weather()
        .join(pl.concat(mission_frames), on="date", how="inner")
        .sort("date", "altitude_bin_km", "mission")
    )
    eligible_altitudes = (
        df.group_by("altitude_bin_km", "mission")
        .agg(pl.col("date").min().alias("start"), pl.col("date").max().alias("end"))
        .with_columns(
            ((pl.col("end") - pl.col("start")).dt.total_days() / 365.2425).alias(
                "duration_years"
            )
        )
        .filter(pl.col("duration_years") > MIN_DURATION_YEARS)
        .select("altitude_bin_km")
        .unique()
        .get_column("altitude_bin_km")
        .sort()
        .to_list()
    )

    specs: list[SeriesSpec] = []
    for altitude in eligible_altitudes:
        alt_df = df.filter(pl.col("altitude_bin_km") == altitude).select(
            "date", F107_COL, "log10rho_daily_mean", "mission"
        )
        specs.append(
            SeriesSpec(
                "TU Delft satellite density dataset",
                float(altitude),
                "date",
                "log10rho_daily_mean",
                alt_df,
                categorical_col="mission",
            )
        )
    return specs


def msis_specs() -> list[SeriesSpec]:
    """Build trend specifications for Mauna Loa MSIS density baselines."""

    df = pl.read_parquet(MSIS_PATH).with_columns(pl.col("date").cast(pl.Date))
    specs: list[SeriesSpec] = []
    models = {
        "NRLMSISE-00 Mauna Loa baseline": "NRLMSISE-00",
        "NRLMSIS 2.0 Mauna Loa baseline": "NRLMSIS 2.0",
        "NRLMSIS 2.1 Mauna Loa baseline": "NRLMSIS 2.1",
    }
    for dataset, model in models.items():
        prefix = f"{safe_model_name(model)}_log10rho_daily_mean_"
        for col in df.columns:
            if not col.startswith(prefix) or not col.endswith("km"):
                continue
            altitude = float(col[len(prefix) : -2])
            specs.append(SeriesSpec(dataset, altitude, "date", col, df))
    return specs


def jb_specs(path: Path = JB_PATH) -> list[SeriesSpec]:
    """Build paired JB trend specifications from externally generated outputs.

    This seam deliberately consumes provider-model output only. It neither
    executes, translates, nor supplies JB2006/JB2008 software or indices.
    """

    df = pl.read_parquet(path)
    required = {"date", F107_COL}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"JB daily-wide Parquet is missing required columns: {', '.join(missing)}"
        )
    df = df.with_columns(pl.col("date").cast(pl.Date))
    models = {
        "JB2006 Mauna Loa baseline": "jb2006_log10rho_daily_mean_",
        "JB2008 Mauna Loa baseline": "jb2008_log10rho_daily_mean_",
    }
    columns_by_dataset: dict[str, dict[float, str]] = {}
    for dataset, prefix in models.items():
        columns: dict[float, str] = {}
        for column in df.columns:
            if not column.startswith(prefix) or not column.endswith("km"):
                continue
            try:
                altitude = float(column[len(prefix) : -2])
            except ValueError as error:
                raise ValueError(f"Invalid JB altitude column: {column}") from error
            columns[altitude] = column
        columns_by_dataset[dataset] = columns

    jb2006, jb2008 = (columns_by_dataset[dataset] for dataset in models)
    if not jb2006 or not jb2008:
        missing_models = [
            dataset for dataset, columns in columns_by_dataset.items() if not columns
        ]
        raise ValueError(
            "JB daily-wide Parquet requires the paired JB2006/JB2008 model families; "
            f"missing: {', '.join(missing_models)}"
        )
    if set(jb2006) != set(jb2008):
        raise ValueError(
            "JB2006 and JB2008 daily-wide columns must have identical altitude sets"
        )
    return [
        SeriesSpec(dataset, altitude, "date", columns[altitude], df)
        for dataset, columns in columns_by_dataset.items()
        for altitude in sorted(columns)
    ]


def required_input_paths(
    require_jb: bool = False, jb_path: Path = JB_PATH
) -> tuple[Path, ...]:
    """Return the processed inputs read by ``collect_trends``."""
    paths = (
        GLOBAL_MEAN_PATH,
        HASDM_PATH,
        *(path for _, path, _ in TUDELFT_MISSIONS),
        MSIS_PATH,
        SPACE_WEATHER_PATH,
    )
    return (*paths, jb_path) if require_jb else paths


def collect_trends(require_jb: bool = False, jb_path: Path = JB_PATH) -> pl.DataFrame:
    """Load all products, fit solar-adjusted trends, and return a table."""

    missing = [
        str(path)
        for path in required_input_paths(require_jb=require_jb, jb_path=jb_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required processed data files: " + ", ".join(missing)
        )

    specs: list[SeriesSpec] = [
        *global_mean_specs(),
        *long_altitude_specs(
            HASDM_PATH,
            "Mauna Loa HASDM subset",
            "altitude_km",
            "log10rho_daily_mean",
        ),
        *tudelft_specs(),
        *msis_specs(),
    ]
    # An available external JB file is always treated as a paired input. This
    # rejects a singleton even for a draft invocation, rather than plotting it.
    if require_jb or jb_path.exists():
        specs.extend(jb_specs(jb_path))
    rows = [trend for spec in specs if (trend := fit_trend(spec)) is not None]
    if not rows:
        raise RuntimeError("No trend estimates were produced.")
    return pl.DataFrame(rows).sort(["dataset", "altitude_km"])


def plot_trends(
    ax: plt.Axes,
    trends: pl.DataFrame,
    altitude_limits: tuple[float, float] | None = None,
) -> tuple[list[Line2D], list[Line2D]]:
    """Plot trends, optionally using supplied altitude limits, and return handles."""
    x_all = np.concatenate(
        [
            trends["trend_percent_per_decade"].to_numpy().astype(float),
            trends["trend_percent_per_decade_hac_95_ci_lower"].to_numpy().astype(float),
            trends["trend_percent_per_decade_hac_95_ci_upper"].to_numpy().astype(float),
        ]
    )
    x_all = x_all[np.isfinite(x_all)]
    x_min = float(np.min(x_all)) if len(x_all) else -20.0
    x_max = float(np.max(x_all)) if len(x_all) else 20.0
    x_pad = 0.08 * max(1.0, x_max - x_min)
    x_left = min(-2.0, x_min - x_pad)
    x_right = max(2.0, x_max + x_pad)

    ax.axvspan(0.0, x_right, color="0.5", alpha=0.08, zorder=0)
    ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.9, zorder=1)

    for dataset in DATASET_LABELS:
        subset = trends.filter(pl.col("dataset") == dataset).sort("altitude_km")
        if subset.is_empty():
            continue

        color = DATASET_COLORS[dataset]
        linestyle = DATASET_LINESTYLES.get(dataset, "-")
        x = subset["trend_percent_per_decade"].to_numpy().astype(float)
        y = subset["altitude_km"].to_numpy().astype(float)

        ax.plot(
            x,
            y,
            color=color,
            linestyle=linestyle,
            linewidth=1.9,
            alpha=0.88,
            zorder=2,
        )

        for bin_start in sorted(set(subset["duration_bin_years"].to_list())):
            points = subset.filter(pl.col("duration_bin_years") == bin_start)
            xvals = points["trend_percent_per_decade"].to_numpy().astype(float)
            yvals = points["altitude_km"].to_numpy().astype(float)
            ci_lower = (
                points["trend_percent_per_decade_hac_95_ci_lower"]
                .to_numpy()
                .astype(float)
            )
            ci_upper = (
                points["trend_percent_per_decade_hac_95_ci_upper"]
                .to_numpy()
                .astype(float)
            )
            xerr = np.vstack((xvals - ci_lower, ci_upper - xvals))
            err_mask = np.all(np.isfinite(xerr), axis=0) & np.isfinite(xvals)
            if np.any(err_mask):
                ax.errorbar(
                    xvals[err_mask],
                    yvals[err_mask],
                    xerr=xerr[:, err_mask],
                    fmt="none",
                    ecolor=color,
                    elinewidth=0.9,
                    capsize=2.0,
                    alpha=0.55,
                    zorder=2,
                )

            ax.scatter(
                points["trend_percent_per_decade"].to_numpy(),
                points["altitude_km"].to_numpy(),
                marker=DURATION_MARKERS.get(int(bin_start), "*"),
                s=42,
                color=color,
                edgecolor="black",
                linewidth=0.35,
                zorder=3,
            )

    ax.set_xlim(x_left, x_right)
    if altitude_limits is None:
        ax.set_ylim(bottom=150.0)
    else:
        ax.set_ylim(*altitude_limits)
    ax.set_xlabel("Solar-adjusted density trend (%/dec)")
    ax.set_ylabel("Altitude (km)")
    ax.grid(True, which="major", alpha=0.25)
    ax.minorticks_on()
    ax.grid(True, which="minor", alpha=0.10)

    dataset_handles = [
        Line2D(
            [0],
            [0],
            color=DATASET_COLORS[dataset],
            lw=2.2,
            linestyle=DATASET_LINESTYLES.get(dataset, "-"),
            label=DATASET_LABELS[dataset],
        )
        for dataset in DATASET_LABELS
        if not trends.filter(pl.col("dataset") == dataset).is_empty()
    ]
    duration_handles = [
        Line2D(
            [0],
            [0],
            marker=DURATION_MARKERS.get(int(bin_start), "*"),
            color="white",
            markerfacecolor="0.55",
            markeredgecolor="black",
            linestyle="None",
            markersize=7,
            label=duration_label(int(bin_start)),
        )
        for bin_start in sorted(set(trends["duration_bin_years"].to_list()))
    ]
    return dataset_handles, duration_handles


def save_outputs(trends: pl.DataFrame) -> None:
    """Write trends and plot all estimates with asymmetric HAC 95% intervals."""

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    trends.write_csv(OUTPUT_ROOT / "current_density_trends_by_dataset_altitude.csv")

    fig, ax = plt.subplots(figsize=(6, 7), constrained_layout=False)
    fig.subplots_adjust(left=0.18, right=0.96, bottom=0.09, top=0.72)
    dataset_handles, duration_handles = plot_trends(ax, trends)

    dataset_legend = fig.legend(
        handles=dataset_handles,
        title="Dataset",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        frameon=True,
        framealpha=0.92,
        borderaxespad=0.0,
        fontsize=11,
        title_fontsize=11,
    )
    fig.add_artist(dataset_legend)
    fig.legend(
        handles=duration_handles,
        title="Record length",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.85),
        ncol=2,
        frameon=True,
        framealpha=0.92,
        borderaxespad=0.0,
        fontsize=11,
        title_fontsize=11,
    )

    fig.savefig(
        OUTPUT_ROOT / "current_density_trends_by_dataset_altitude.pgf",
        bbox_inches="tight",
    )
    fig.savefig(
        OUTPUT_ROOT / "current_density_trends_by_dataset_altitude.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the trend-estimator command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-jb", action="store_true")
    parser.add_argument("--jb-path", type=Path, default=JB_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""

    args = parse_args(argv)
    trends = collect_trends(require_jb=args.require_jb, jb_path=args.jb_path)
    save_outputs(trends)
    print(f"Saved {trends.height} trend estimates to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
