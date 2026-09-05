"""THROWAWAY PROTOTYPE: Figure 5 empirical-model comparison in three layouts.

Run with `.venv/bin/python -m scripts.prototype_empirical_model_figure5`.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

OUTPUT_DIR = Path("outputs/prototypes/empirical_model_figure5")
THESIS_FIGURE_5_53_PDF = Path(
    "outputs/figures/results/tudelft_model_errors/figure19/"
    "tudelft_figure19_all_missions_with_jb.pdf"
)
ALTITUDES = np.arange(175, 826, 25)
MODELS = {
    "nrlmsise_00": ("NRLMSISE-00", "#0072B2"),
    "nrlmsis_2p0": ("NRLMSIS 2.0", "#E69F00"),
    "nrlmsis_2p1": ("NRLMSIS 2.1", "#009E73"),
    "jb2006": ("JB2006", "#CC79A7"),
    "jb2008": ("JB2008", "#D55E00"),
}
N_BOOTSTRAP = 250
BLOCK_DAYS = 27
PCMCI_TAU_MAX = 180
PCMCI_PC_ALPHA = 0.05
MISSING_FLAG = -9999.0
CACHE_VERSION = "figure5-pcmci-v3"
DRIVER_LABELS = {
    "F10.7_OBS": "F10.7 (raw)",
    "F10.7_OBS_CENTER81": "F10.7 (81-day centered)",
    "AP_AVG": "Ap",
    "KP_SUM": "Kp",
    "CO2_ppm": "Mauna Loa CO2",
    "SABER_CO2_COOLING_139KM": "SABER CO2 cooling (139 km)",
}
RUN_SPECS = {
    "primary": ["F10.7_OBS", "AP_AVG", "CO2_ppm"],
    "centered_f107_robustness": ["F10.7_OBS_CENTER81", "AP_AVG", "CO2_ppm"],
    "kp_geomagnetic_variant": ["F10.7_OBS", "KP_SUM", "CO2_ppm"],
    "saber_extension": [
        "F10.7_OBS",
        "AP_AVG",
        "CO2_ppm",
        "SABER_CO2_COOLING_139KM",
    ],
    "saber_overlap_core_control": ["F10.7_OBS", "AP_AVG", "CO2_ppm"],
}
FORCING_RUNS = {
    "F10.7_OBS": "primary",
    "F10.7_OBS_CENTER81": "centered_f107_robustness",
    "AP_AVG": "primary",
    "KP_SUM": "kp_geomagnetic_variant",
    "CO2_ppm": "primary",
    "SABER_CO2_COOLING_139KM": "saber_extension",
}
AVAILABILITY = (
    ("F10.7_OBS", True, "data/original/space_weather/SW-All.csv", "raw daily F10.7"),
    (
        "F10.7_OBS_CENTER81",
        True,
        "data/original/space_weather/SW-All.csv",
        "centered 81-day F10.7",
    ),
    ("AP_AVG", True, "data/original/space_weather/SW-All.csv", "daily Ap"),
    ("KP_SUM", True, "data/original/space_weather/SW-All.csv", "daily Kp"),
    ("CO2_ppm", True, "data/original/co2/co2_daily_mlo.csv", "Mauna Loa daily CO2"),
    (
        "SABER_CO2_COOLING_139KM",
        True,
        "data/decoded/saber/saber_co2_cooling_maunaloa_daily.parquet",
        "nearest available SABER altitude to 139 km",
    ),
    (
        "F30",
        False,
        "",
        "requested product is not locally available; no result fabricated",
    ),
    (
        "SABER_NO_COOLING",
        False,
        "",
        "requested product is not locally available; no result fabricated",
    ),
    (
        "SABER_OH_EMISSION",
        False,
        "",
        "requested product is not locally available; no result fabricated",
    ),
    (
        "SABER_O2_EMISSION",
        False,
        "",
        "requested product is not locally available; no result fabricated",
    ),
)

SUPERVISOR_CAPTIONS = {
    "A": (
        "Conventional profile atlas",
        "Each panel overlays altitude profiles for all five empirical models. The first "
        "two panels show median log-density bias and median absolute log error relative "
        "to HASDM; the remaining panels show PCMCI+ residual-dependence diagnostics for six "
        "driver/variant products from separate matched runs. "
        "Accuracy bars are 95% intervals; leakage points and markers are FDR detections. This familiar layout makes direct model and "
        "altitude comparisons easiest, although overlapping series can become visually dense.",
    ),
    "C": (
        "Decision matrix",
        "Heatmaps encode each estimate by model and altitude, with an independent color "
        "scale for each metric. Open circles mark accuracy interval cues or retained FDR PCMCI+ links. This is the "
        "most compact overview and makes broad altitude patterns easy to scan, but exact "
        "interval magnitudes and small between-model differences are less visible.",
    ),
}


def supervisor_methods_note(tau_max: int, pcmci_altitudes: np.ndarray) -> str:
    """Describe the actual render configuration rather than hard-coding smoke settings."""

    altitude_scope = (
        "all 27 altitudes"
        if len(pcmci_altitudes) == len(ALTITUDES)
        else f"{len(pcmci_altitudes)} selected altitude(s)"
    )
    return (
        "Common accuracy basis: 9,301 observed common dates from 2000-01-01 to "
        "2025-07-20 for all five models and all 27 altitudes (175-825 km), embedded in "
        "the complete 9,333-day calendar so missing dates remain missing while paired "
        "27-calendar-day circular bootstrap blocks preserve physical time. Error is "
        "epsilon = ln(rho_model/rho_HASDM); the median accuracy diagnostics are descriptive. "
        f"Residual dependence uses PCMCI+ at {altitude_scope} (ParCorr analytic; daily lags "
        f"0-{tau_max}; pc_alpha=.05; majority collider rule, conflict resolution and FDR-BH "
        "within each altitude/run graph, including lag zero) after calendar-day seasonal "
        "anomaly, centered 3-year rolling-mean removal, and standardization. The six displayed "
        "diagnostics are raw and centered-81 F10.7, Ap, Kp, Mauna Loa CO2, and SABER CO2 cooling "
        "at 139 km; SABER uses its shorter supported window. Its overlap artifact lists the full, "
        "overlap-control and extension runs but does not estimate an extension effect. Points are "
        "the signed partial r at the maximum retained absolute partial r across tested lags; zero "
        "means no retained FDR link, not independence. This single-profile exploratory diagnostic "
        "does not claim stationarity qualification, full 2x2 sensitivity agreement, or a causal "
        "effect. Unavailable F30 and SABER NO/OH/O2 products are recorded only in provenance."
    )


THESIS_FIGURE_5_53_CAPTION = (
    "TU Delft recreation and extension of the seasonal model log-density-ratio "
    "comparison from Figure 19 of Emmert et al. (2021), extended here to include "
    "paired JB2006 and JB2008 alongside the three MSIS-family models. Dates are shown "
    "as the last two digits of the year. Each light-blue band gives the upper "
    "recommended mission-specific reference-density standard-uncertainty scale, "
    "transformed approximately to +/-ln(1+u_rho) in model log-density-ratio units. "
    "The bands are per-observation attribution guides, independent of the plotted "
    "standard errors of the seasonal means; they are not confidence intervals or "
    "strict error bounds."
)

THESIS_FIGURE_5_53_SCOPE_NOTE = (
    "Scope note: unlike Options A-C, this recreation uses TU Delft satellite missions "
    "rather than the Mauna Loa HASDM altitude profiles. Rows are mission-specific and "
    "columns divide the data into low, moderate, and high 81-day mean F10.7 regimes. "
    "JB2006 and JB2008 are evaluated at each retained mission timestamp, location, and "
    "altitude. CHAMP retains the thesis exclusion of 2006-2009, and the shared vertical "
    "scale is expanded to avoid clipping the JB2008 high-activity departures."
)


def model_column(model: str, altitude: int) -> str:
    return f"{model}_log10rho_daily_mean_{altitude}km"


def complete_daily_calendar(
    frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Expose absent source dates without filling any measurement values."""

    calendar = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    return calendar.merge(frame, on="date", how="left")


def common_data() -> pd.DataFrame:
    """Build daily data, retaining source missingness for PCMCI+.

    ``accuracy_sample`` below deliberately preserves the historical strict sample.
    PCMCI+ receives this wider table and Tigramite's missing flag instead of interpolation.
    """

    hasdm = pd.read_parquet(
        "outputs/figures/results/set_hasdm/model_validations/"
        "causal_hasdm_saber_maunaloa/hasdm_maunaloa_daily_wide.parquet"
    )
    msis = pd.read_parquet(
        "outputs/figures/results/maunaloa_msis_density_baselines/data/"
        "maunaloa_msis_density_baselines_daily_wide.parquet"
    )
    jb = pd.read_parquet(
        "outputs/figures/results/maunaloa_jb_density_baselines/data/"
        "maunaloa_jb_density_baselines_daily_wide.parquet"
    )
    drivers = pd.read_csv(
        "data/original/space_weather/SW-All.csv",
        usecols=["DATE", "F10.7_OBS", "F10.7_OBS_CENTER81", "AP_AVG", "KP_SUM"],
    ).rename(columns={"DATE": "date"})
    co2 = pd.read_csv(
        "data/original/co2/co2_daily_mlo.csv",
        comment="#",
        header=None,
        names=["year", "month", "day", "year_decimal", "CO2_ppm"],
    )
    co2["date"] = pd.to_datetime(co2[["year", "month", "day"]])
    co2.loc[co2.CO2_ppm < 0, "CO2_ppm"] = np.nan

    required = [f"log10rho_{altitude}_daily_mean" for altitude in ALTITUDES]
    hasdm = hasdm[["date", *required]].copy()
    hasdm.columns = ["date", *[f"hasdm_{altitude}" for altitude in ALTITUDES]]
    msis_columns = [
        "date",
        *[
            model_column(model, altitude)
            for model in MODELS
            if model.startswith("nrl")
            for altitude in ALTITUDES
        ],
    ]
    jb_columns = [
        "date",
        *[
            model_column(model, altitude)
            for model in MODELS
            if model.startswith("jb")
            for altitude in ALTITUDES
        ],
    ]
    for frame in (hasdm, msis, jb, drivers):
        frame["date"] = pd.to_datetime(frame["date"])
    targets = hasdm.merge(msis[msis_columns], on="date", how="inner")
    targets = targets.merge(jb[jb_columns], on="date", how="inner")
    data = complete_daily_calendar(targets, hasdm.date.min(), hasdm.date.max())
    data = data.merge(drivers, on="date", how="left")
    return (
        data.merge(co2[["date", "CO2_ppm"]], on="date", how="left")
        .sort_values("date")
        .reset_index(drop=True)
    )


def accuracy_sample(data: pd.DataFrame) -> pd.DataFrame:
    """The existing exact five-model accuracy sample, including its historical drivers."""

    required = [
        *[f"hasdm_{altitude}" for altitude in ALTITUDES],
        *[model_column(model, altitude) for model in MODELS for altitude in ALTITUDES],
        "F10.7_OBS_CENTER81",
        "AP_AVG",
    ]
    return data.loc[np.isfinite(data[required]).all(axis=1)].reset_index(drop=True)


def circular_block_indices(n: int) -> np.ndarray:
    """Return deterministic paired circular-block bootstrap indices for every metric."""

    rng = np.random.default_rng(37)
    starts = rng.integers(0, n, size=(N_BOOTSTRAP, int(np.ceil(n / BLOCK_DAYS))))
    offsets = np.arange(BLOCK_DAYS)
    return ((starts[..., None] + offsets).reshape(N_BOOTSTRAP, -1)[:, :n]) % n


def bootstrap_interval(
    values: np.ndarray, indices: np.ndarray, absolute: bool
) -> tuple[float, float]:
    sampled = values[indices]
    if absolute:
        sampled = np.abs(sampled)
    estimates = np.nanmedian(sampled, axis=1)
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def calculate_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate common-sample medians on a complete calendar with missing placeholders."""

    indices = circular_block_indices(len(data))
    rows = []
    for model, (display_name, _) in MODELS.items():
        for altitude in ALTITUDES:
            epsilon = np.log(10) * (
                data[model_column(model, altitude)].to_numpy()
                - data[f"hasdm_{altitude}"].to_numpy()
            )
            bias_low, bias_high = bootstrap_interval(epsilon, indices, absolute=False)
            mae_low, mae_high = bootstrap_interval(epsilon, indices, absolute=True)
            finite = np.isfinite(epsilon)
            row = {
                "model": model,
                "model_display": display_name,
                "altitude_km": altitude,
                "signed_median_epsilon": float(np.nanmedian(epsilon)),
                "signed_median_epsilon_ci_low": bias_low,
                "signed_median_epsilon_ci_high": bias_high,
                "median_absolute_epsilon": float(np.nanmedian(np.abs(epsilon))),
                "median_absolute_epsilon_ci_low": mae_low,
                "median_absolute_epsilon_ci_high": mae_high,
                "n": int(finite.sum()),
                "date_start": data.loc[finite, "date"].min().date().isoformat(),
                "date_end": data.loc[finite, "date"].max().date().isoformat(),
                "bootstrap_method": "paired deterministic circular calendar-day block bootstrap",
                "bootstrap_repetitions": N_BOOTSTRAP,
                "bootstrap_block_days": BLOCK_DAYS,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def seasonal_detrended_standardized(values: np.ndarray, dates: pd.Series) -> np.ndarray:
    """Apply the repository's registered primary PCMCI preprocessing profile."""

    from thermodense.benchmarks.pcmci_real import DETRENDED_ANOMALY, preprocess

    matrix = np.asarray(values, dtype=float)[:, None]
    return preprocess(
        matrix,
        pd.to_datetime(dates).to_numpy(dtype="datetime64[D]"),
        DETRENDED_ANOMALY,
    )[:, 0]


def build_link_assumptions(
    columns: list[str], target_columns: list[str], tau_max: int
) -> dict[int, dict[tuple[int, int], str]]:
    """Allow driver dynamics and driver/self history, never peer-error conditioning."""

    assumptions: dict[int, dict[tuple[int, int], str]] = {
        j: {} for j in range(len(columns))
    }
    targets = set(target_columns)
    for target_idx, target in enumerate(columns):
        for cause_idx, cause in enumerate(columns):
            for lag in range(1, tau_max + 1):
                allowed = (
                    (target not in targets and cause not in targets)  # driver dynamics
                    or (target in targets and cause not in targets)  # driver -> target
                    or cause == target  # target self-history
                )
                if allowed:
                    assumptions[target_idx][(cause_idx, -lag)] = "-?>"
            if (
                target not in targets
                and cause not in targets
                and cause_idx < target_idx
            ):
                assumptions[target_idx][(cause_idx, 0)] = "o?o"
            elif target in targets and cause not in targets:
                assumptions[target_idx][(cause_idx, 0)] = "-?>"
    return assumptions


def retained_link_table(
    graph: np.ndarray,
    values: np.ndarray,
    raw_pvalues: np.ndarray,
    qvalues: np.ndarray,
    columns: list[str],
    target_columns: list[str],
    run: str,
    altitude: int,
) -> pd.DataFrame:
    """Return every FDR-retained driver-to-error lag, rather than a lag summary."""

    rows = []
    for source_idx, source in enumerate(columns):
        if source in target_columns:
            continue
        for target in target_columns:
            target_idx = columns.index(target)
            for lag, mark in enumerate(graph[source_idx, target_idx]):
                correctly_oriented = (lag == 0 and str(mark).endswith(">")) or (
                    lag > 0 and mark == "-->"
                )
                if mark and correctly_oriented:
                    rows.append(
                        {
                            "run": run,
                            "altitude_km": altitude,
                            "source": source,
                            "target": target,
                            "lag_days": lag,
                            "partial_r": float(values[source_idx, target_idx, lag]),
                            "absolute_r": abs(
                                float(values[source_idx, target_idx, lag])
                            ),
                            "raw_p_value": float(
                                raw_pvalues[source_idx, target_idx, lag]
                            ),
                            "q_value": float(qvalues[source_idx, target_idx, lag]),
                            "graph_mark": mark,
                        }
                    )
    return pd.DataFrame(
        rows,
        columns=[
            "run",
            "altitude_km",
            "source",
            "target",
            "lag_days",
            "partial_r",
            "absolute_r",
            "raw_p_value",
            "q_value",
            "graph_mark",
        ],
    )


def retain_fdr_links(graph: np.ndarray, qvalues: np.ndarray) -> np.ndarray:
    """Keep PCMCI+ graph marks only where the once-corrected BH q-value passes."""

    return np.where(qvalues <= PCMCI_PC_ALPHA, graph, "")


def fdr_bh_qvalues(
    pcmci: object,
    raw_pvalues: np.ndarray,
    tau_max: int,
    assumptions: dict[int, dict[tuple[int, int], str]],
) -> np.ndarray:
    """Correct the complete lag family once, including contemporaneous links."""

    return pcmci.get_corrected_pvalues(
        raw_pvalues,
        tau_min=0,
        tau_max=tau_max,
        fdr_method="fdr_bh",
        link_assumptions=assumptions,
        exclude_contemporaneous=False,
    )


def strongest_links(
    retained_links: pd.DataFrame,
    sources: list[str],
    target_columns: list[str],
    run: str,
    altitude: int,
) -> pd.DataFrame:
    """Summarize retained lags; absence is an explicit zero-display non-detection."""

    rows = []
    for source in sources:
        for target in target_columns:
            candidates = retained_links[
                (retained_links.source == source) & (retained_links.target == target)
            ]
            if candidates.empty:
                rows.append(
                    {
                        "run": run,
                        "altitude_km": altitude,
                        "source": source,
                        "target": target,
                        "partial_r": 0.0,
                        "absolute_r": 0.0,
                        "lag_days": None,
                        "raw_p_value": None,
                        "q_value": None,
                        "graph_mark": "",
                        "detected": False,
                        "display_value": 0.0,
                    }
                )
                continue
            strongest = candidates.loc[candidates.absolute_r.idxmax()]
            rows.append(
                {
                    **strongest.to_dict(),
                    "detected": True,
                    "display_value": strongest.partial_r,
                }
            )
    return pd.DataFrame(rows)


def load_saber_139() -> pd.DataFrame:
    """Return the locally supported SABER CO2-cooling series nearest 139 km."""

    saber = pd.read_parquet(
        "data/decoded/saber/saber_co2_cooling_maunaloa_daily.parquet"
    )
    altitude = saber.loc[(saber.altitude_km - 139).abs().idxmin(), "altitude_km"]
    return (
        saber.loc[saber.altitude_km == altitude, ["date", "co2_cooling_rate_w_m3"]]
        .groupby("date", as_index=False)
        .mean()
        .rename(columns={"co2_cooling_rate_w_m3": "SABER_CO2_COOLING_139KM"})
    )


def prepare_saber_window(data: pd.DataFrame, saber: pd.DataFrame) -> pd.DataFrame:
    """Keep the complete daily SABER window, including absent-product days as NaN."""

    saber = saber.copy()
    saber["date"] = pd.to_datetime(saber["date"])
    calendar = pd.DataFrame(
        {"date": pd.date_range(saber.date.min(), saber.date.max(), freq="D")}
    )
    windowed = calendar.merge(data, on="date", how="left")
    return windowed.merge(saber, on="date", how="left")


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Publish a CSV only after its complete same-filesystem write succeeds."""

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_write_json(payload: object, path: Path) -> None:
    """Publish JSON atomically for resumable caches."""

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def pcmci_fingerprint(
    run_data: dict[str, pd.DataFrame], altitudes: np.ndarray, tau_max: int
) -> str:
    """Identity includes run windows, selected altitudes, and every merged input value."""

    payload = {
        "version": CACHE_VERSION,
        "altitudes": list(map(int, altitudes)),
        "run_hashes": {
            run: hashlib.sha256(
                pd.util.hash_pandas_object(frame, index=True).values.tobytes()
            ).hexdigest()
            for run, frame in run_data.items()
        },
        "run_windows": {
            run: [str(frame.date.min()), str(frame.date.max()), len(frame)]
            for run, frame in run_data.items()
        },
        "tau_max": tau_max,
        "pc_alpha": PCMCI_PC_ALPHA,
        "settings": {
            "preprocessing": "registered detrended_anomaly profile: calendar-month/day anomaly; centered 3-year rolling mean removal; population standardization",
            "missing_flag": MISSING_FLAG,
            "remove_missing_upto_maxlag": False,
            "collider_rule": "majority",
            "conflict_resolution": True,
            "fdr": "BH once within each altitude/run graph after raw PCMCI+ p-values, including lag zero",
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def run_pcmci_task(
    task: tuple[str, int, pd.DataFrame, list[str], int],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Picklable one-altitude PCMCI+ task; input is preloaded by the parent process."""

    run, altitude, local, drivers, tau_max = task
    from tigramite import data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI
    from thermodense.benchmarks.pcmci_real import validate_daily_dates

    target_columns = [f"epsilon_{model}" for model in MODELS]
    columns = [*target_columns, *drivers]
    for model in MODELS:
        local[f"epsilon_{model}"] = np.log(10) * (
            local[model_column(model, altitude)] - local[f"hasdm_{altitude}"]
        )
    validate_daily_dates(local.date.to_numpy())
    values = np.column_stack(
        [
            seasonal_detrended_standardized(local[column].to_numpy(), local.date)
            for column in columns
        ]
    )
    finite_counts = {
        column: int(count)
        for column, count in zip(columns, np.isfinite(values).sum(axis=0), strict=True)
    }
    values[~np.isfinite(values)] = MISSING_FLAG
    assumptions = build_link_assumptions(columns, target_columns, tau_max)
    frame = pp.DataFrame(
        values,
        datatime=np.arange(len(local)),
        var_names=columns,
        missing_flag=MISSING_FLAG,
        remove_missing_upto_maxlag=False,
    )
    pcmci = PCMCI(frame, cond_ind_test=ParCorr(significance="analytic"), verbosity=0)
    result = pcmci.run_pcmciplus(
        link_assumptions=assumptions,
        tau_min=0,
        tau_max=tau_max,
        pc_alpha=PCMCI_PC_ALPHA,
        contemp_collider_rule="majority",
        conflict_resolution=True,
        fdr_method="none",
    )
    raw_pvalues = result["p_matrix"]
    qvalues = fdr_bh_qvalues(pcmci, raw_pvalues, tau_max, assumptions)
    retained_graph = retain_fdr_links(result["graph"], qvalues)
    retained = retained_link_table(
        retained_graph,
        result["val_matrix"],
        raw_pvalues,
        qvalues,
        columns,
        target_columns,
        run,
        altitude,
    )
    summary = strongest_links(retained, drivers, target_columns, run, altitude)
    return (
        retained,
        summary,
        {
            "run": run,
            "altitude_km": altitude,
            "finite_counts": finite_counts,
        },
    )


def pcmci_run_data(data: pd.DataFrame, saber: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build matched windows once; SABER and its control share the complete calendar."""

    saber_window = prepare_saber_window(data, saber)
    return {
        "primary": data.copy(),
        "centered_f107_robustness": data.copy(),
        "kp_geomagnetic_variant": data.copy(),
        "saber_extension": saber_window,
        "saber_overlap_core_control": saber_window.drop(
            columns=["SABER_CO2_COOLING_139KM"]
        ),
    }


def run_provenance(frame: pd.DataFrame, drivers: list[str]) -> dict[str, object]:
    """Document every run's calendar and raw finite counts before preprocessing."""

    node_columns = [
        *[model_column(model, altitude) for model in MODELS for altitude in ALTITUDES],
        *drivers,
    ]
    return {
        "daily_rows": len(frame),
        "date_start": frame.date.min().date().isoformat(),
        "date_end": frame.date.max().date().isoformat(),
        "finite_counts_raw": {
            column: int(frame[column].notna().sum()) for column in node_columns
        },
        "remove_missing_upto_maxlag": False,
    }


def pcmci_artifacts(
    data: pd.DataFrame,
    altitudes: np.ndarray,
    tau_max: int,
    recompute: bool,
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Read/write identity-keyed cache plus stable canonical retained and summary tables."""

    run_data = pcmci_run_data(data, load_saber_139())
    fingerprint = pcmci_fingerprint(run_data, altitudes, tau_max)
    cache_dir = OUTPUT_DIR / "pcmci_cache"
    cache_dir.mkdir(exist_ok=True)
    cache_prefix = cache_dir / fingerprint
    retained_cache = cache_prefix.with_name(f"{fingerprint}_retained.csv")
    summary_cache = cache_prefix.with_name(f"{fingerprint}_summary.csv")
    provenance_cache = cache_prefix.with_name(f"{fingerprint}_provenance.json")
    if not recompute and all(
        path.exists() for path in (retained_cache, summary_cache, provenance_cache)
    ):
        retained = pd.read_csv(retained_cache)
        summary = pd.read_csv(summary_cache)
        atomic_write_csv(retained, OUTPUT_DIR / "pcmci_retained_links.csv")
        atomic_write_csv(summary, OUTPUT_DIR / "forcing_summary.csv")
        return retained, summary, json.loads(provenance_cache.read_text())

    task_cache_dir = cache_dir / fingerprint
    task_cache_dir.mkdir(exist_ok=True)
    tasks = []
    for run, drivers in RUN_SPECS.items():
        for altitude_value in altitudes:
            altitude = int(altitude_value)
            task_columns = [
                "date",
                f"hasdm_{altitude}",
                *[model_column(model, altitude) for model in MODELS],
                *drivers,
            ]
            tasks.append(
                (
                    run,
                    altitude,
                    run_data[run][task_columns].copy(),
                    drivers,
                    tau_max,
                )
            )

    def task_paths(
        task: tuple[str, int, pd.DataFrame, list[str], int],
    ) -> tuple[Path, Path, Path]:
        run, altitude, _, _, _ = task
        prefix = task_cache_dir / f"{run}_{altitude}km"
        return (
            prefix.with_name(f"{prefix.name}_retained.csv"),
            prefix.with_name(f"{prefix.name}_summary.csv"),
            prefix.with_name(f"{prefix.name}_provenance.json"),
        )

    def load_task(task: tuple[str, int, pd.DataFrame, list[str], int]):
        retained_path, summary_path, task_provenance_path = task_paths(task)
        if not recompute and all(
            path.exists()
            for path in (retained_path, summary_path, task_provenance_path)
        ):
            return (
                pd.read_csv(retained_path),
                pd.read_csv(summary_path),
                json.loads(task_provenance_path.read_text()),
            )
        return None

    def save_task(task, output) -> None:
        retained_path, summary_path, task_provenance_path = task_paths(task)
        retained_frame, summary_frame, task_metadata = output
        atomic_write_csv(retained_frame, retained_path)
        atomic_write_csv(summary_frame, summary_path)
        atomic_write_json(task_metadata, task_provenance_path)

    outputs: list[tuple[pd.DataFrame, pd.DataFrame, dict[str, object]] | None] = [
        load_task(task) for task in tasks
    ]
    pending_indices = [index for index, output in enumerate(outputs) if output is None]
    if workers > 1 and pending_indices:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_indices = {
                executor.submit(run_pcmci_task, tasks[index]): index
                for index in pending_indices
            }
            for future in as_completed(future_indices):
                index = future_indices[future]
                output = future.result()
                save_task(tasks[index], output)
                outputs[index] = output
    else:
        for index in pending_indices:
            output = run_pcmci_task(tasks[index])
            save_task(tasks[index], output)
            outputs[index] = output
    if any(output is None for output in outputs):
        raise RuntimeError("PCMCI+ task execution left incomplete outputs")
    completed_outputs = [output for output in outputs if output is not None]
    retained_frames, summary_frames, task_provenance = zip(
        *completed_outputs, strict=True
    )
    retained = pd.concat(retained_frames, ignore_index=True)
    summary = pd.concat(summary_frames, ignore_index=True)
    provenance = {
        "cache_version": CACHE_VERSION,
        "fingerprint": fingerprint,
        "tau_max": tau_max,
        "altitudes": list(map(int, altitudes)),
        "method": "PCMCI+ ParCorr analytic; raw p then one within-graph BH q-value correction including lag zero",
        "interpretation": {
            "status": "single-profile exploratory residual-dependence diagnostic",
            "stationarity_qualification": "not run",
            "full_2x2_f107_timing_by_preprocessing_matrix": "not run",
            "causal_interpretation_eligible": False,
        },
        "runs": {
            run: {
                "role": run,
                "drivers": drivers,
                **run_provenance(run_data[run], drivers),
            }
            for run, drivers in RUN_SPECS.items()
        },
        "task_finite_counts_after_preprocessing": list(task_provenance),
    }
    atomic_write_csv(retained, retained_cache)
    atomic_write_csv(summary, summary_cache)
    atomic_write_json(provenance, provenance_cache)
    atomic_write_csv(retained, OUTPUT_DIR / "pcmci_retained_links.csv")
    atomic_write_csv(summary, OUTPUT_DIR / "forcing_summary.csv")
    return retained, summary, provenance


def add_figure_note(figure: plt.Figure, data: pd.DataFrame) -> None:
    layout_engine = figure.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(rect=(0.0, 0.075, 1.0, 0.90))
    figure.text(
        0.5,
        0.012,
        "Exact common sample across all five models and 27 altitudes; rankings are metric-, altitude-, and interval-specific—not universal.\n"
        f"n={len(data):,}, {data.date.min():%Y-%m-%d} to {data.date.max():%Y-%m-%d}. "
        "PCMCI+ zero = no retained FDR link, not proven independence; residual dependence diagnostic, not causal effect.",
        ha="center",
        va="bottom",
        fontsize=8,
        wrap=True,
    )


def save(figure: plt.Figure, variant: str) -> None:
    for suffix in ("png", "pdf"):
        figure.savefig(
            OUTPUT_DIR / f"figure5_variant_{variant}.{suffix}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(figure)


def forcing_metric(source: str) -> str:
    return f"forcing_{source.lower().replace('.', '').replace(' ', '_')}"


def metric_specs() -> list[tuple[str, str, bool, bool]]:
    """Accuracy first, followed by all locally available forcing/context diagnostics."""

    return [
        ("signed_median_epsilon", "Bias: median ε", True, False),
        ("median_absolute_epsilon", "Median |ε|", False, False),
        *[
            (forcing_metric(source), f"{label}: residual dependence", True, True)
            for source, label in DRIVER_LABELS.items()
        ],
    ]


def plot_profile_atlas(metrics: pd.DataFrame, data: pd.DataFrame) -> None:
    specs = metric_specs()
    figure, axes = plt.subplots(
        2, 4, figsize=(15, 10), sharey=True, layout="constrained"
    )
    for axis, (metric, label, zero_line, is_pcmci) in zip(
        axes.flat, specs, strict=True
    ):
        for model, (name, color) in MODELS.items():
            part = metrics[metrics.model == model]
            estimate = part[metric].to_numpy()
            if is_pcmci:
                detected = part[f"{metric}_detected"].to_numpy(bool)
                axis.plot(
                    estimate,
                    part.altitude_km,
                    color=color,
                    marker="o",
                    ms=3.2,
                    lw=1.2,
                    label=name,
                )
                axis.scatter(
                    estimate[detected],
                    part.altitude_km.to_numpy()[detected],
                    facecolors="none",
                    edgecolors="black",
                    s=28,
                    zorder=3,
                )
            else:
                low, high = (
                    part[f"{metric}_ci_low"].to_numpy(),
                    part[f"{metric}_ci_high"].to_numpy(),
                )
                axis.errorbar(
                    estimate,
                    part.altitude_km,
                    xerr=[estimate - low, high - estimate],
                    color=color,
                    marker="o",
                    ms=3.2,
                    lw=1.2,
                    capsize=1.8,
                    label=name,
                )
        if zero_line:
            axis.axvline(0, color="0.3", lw=0.8, zorder=0)
        axis.set_title(label, fontsize=9)
        axis.grid(axis="x", alpha=0.25)
        axis.set_xlabel("partial r" if is_pcmci else "accuracy ± 95% interval")
    axes[0, 0].set_ylabel("Altitude (km)")
    axes[1, 0].set_ylabel("Altitude (km)")
    axes[0, 0].set_yticks(ALTITUDES[::2])
    axes[0, 0].legend(loc="upper left", fontsize=7, frameon=False)
    figure.suptitle(
        "A. Conventional profile atlas: accuracy and conditional residual dependence",
        fontsize=13,
        fontweight="bold",
    )
    add_figure_note(figure, data)
    save(figure, "A")


def plot_scorecards(metrics: pd.DataFrame, data: pd.DataFrame) -> None:
    specs = metric_specs()
    figure, axes = plt.subplots(
        10, 4, figsize=(13, 20), sharey=True, layout="constrained"
    )
    for model_index, (model, (name, color)) in enumerate(MODELS.items()):
        part = metrics[metrics.model == model]
        for metric_index, (metric, heading, zero_line, is_pcmci) in enumerate(specs):
            row = model_index * 2 + metric_index // 4
            column = metric_index % 4
            axis = axes[row, column]
            estimate = part[metric].to_numpy()
            axis.plot(estimate, part.altitude_km, color=color, lw=1.5)
            axis.fill_betweenx(
                part.altitude_km,
                part[f"{metric}_ci_low"] if not is_pcmci else estimate,
                part[f"{metric}_ci_high"] if not is_pcmci else estimate,
                color=color,
                alpha=0.2 if not is_pcmci else 0,
            )
            if is_pcmci:
                detected = part[f"{metric}_detected"].to_numpy(bool)
                axis.scatter(
                    estimate[detected],
                    part.altitude_km.to_numpy()[detected],
                    facecolors="none",
                    edgecolors="black",
                    s=25,
                )
            if zero_line:
                axis.axvline(0, color="0.35", lw=0.7)
            axis.set_title(heading, fontweight="bold", fontsize=8)
            if column == 0:
                continuation = "" if metric_index < 4 else " (continued)"
                axis.set_ylabel(
                    f"{name}{continuation}\nAltitude (km)",
                    fontsize=8,
                )
            if row == len(MODELS) * 2 - 1:
                axis.set_xlabel(
                    "partial r" if is_pcmci else "accuracy ± 95%", fontsize=7
                )
            axis.grid(axis="x", alpha=0.2)
            axis.tick_params(labelsize=7)
    figure.suptitle(
        "B. Model-centric scorecards: read each row for conditional, altitude-specific trade-offs",
        fontsize=13,
        fontweight="bold",
    )
    add_figure_note(figure, data)
    save(figure, "B")


def plot_heatmap(metrics: pd.DataFrame, data: pd.DataFrame) -> None:
    specs = metric_specs()
    figure, axes = plt.subplots(8, 1, figsize=(14, 15), layout="constrained")
    for axis, (metric, heading, centered, is_pcmci) in zip(axes, specs, strict=True):
        matrix = np.array(
            [
                metrics[metrics.model == model]
                .sort_values("altitude_km")[metric]
                .to_numpy()
                for model in MODELS
            ]
        )
        limit = max(
            float(np.nanmax(np.abs(matrix)) if centered else np.nanmax(matrix)), 0.01
        )
        norm = TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit) if centered else None
        image = axis.pcolormesh(
            np.arange(matrix.shape[1] + 1) - 0.5,
            np.arange(matrix.shape[0] + 1) - 0.5,
            matrix,
            cmap="RdBu_r" if centered else "viridis",
            norm=norm,
            shading="flat",
        )
        axis.set_xlim(-0.5, matrix.shape[1] - 0.5)
        axis.set_ylim(matrix.shape[0] - 0.5, -0.5)
        for row, model in enumerate(MODELS):
            part = metrics[metrics.model == model].sort_values("altitude_km")
            if is_pcmci:
                marked = part[f"{metric}_detected"].to_numpy(bool)
            else:
                low, high = (
                    part[f"{metric}_ci_low"].to_numpy(),
                    part[f"{metric}_ci_high"].to_numpy(),
                )
                if centered:
                    marked = (low > 0) | (high < 0)
                else:
                    widths = high - low
                    marked = widths >= np.quantile(widths, 0.75)
            axis.scatter(
                np.where(marked)[0],
                np.full(marked.sum(), row),
                marker="o",
                s=18,
                facecolors="none",
                edgecolors="black",
                linewidths=0.7,
            )
        axis.set_title(
            f"{heading}  (open circles: {'FDR-retained PCMCI+ link' if is_pcmci else ('95% interval excludes zero' if centered else 'upper-quartile interval width')})",
            loc="left",
            fontsize=10,
            fontweight="bold",
        )
        axis.set_yticks(range(len(MODELS)), [value[0] for value in MODELS.values()])
        axis.set_xticks(range(0, len(ALTITUDES), 2), ALTITUDES[::2])
        axis.set_xlabel("Altitude (km)")
        colorbar = figure.colorbar(image, ax=axis, pad=0.01, label="estimate")
        colorbar.solids.set_rasterized(False)
    figure.suptitle(
        "C. Decision matrix: independent metric scales; uncertainty cues prevent rank-like reading",
        fontsize=13,
        fontweight="bold",
    )
    add_figure_note(figure, data)
    save(figure, "C")


def write_index() -> None:
    (OUTPUT_DIR / "index.html").write_text(
        """<!doctype html><meta charset=\"utf-8\"><title>Figure 5 layout prototype</title>
<style>body{margin:0;background:#f5f5f5;font:16px system-ui;text-align:center}img{max-width:100%;height:auto}.switcher{position:fixed;bottom:18px;left:50%;transform:translateX(-50%);background:#111;color:white;padding:10px 15px;border-radius:24px;box-shadow:0 2px 8px #777}button,a{margin:0 5px;color:white;background:#333;border:0;padding:6px 10px;border-radius:14px;text-decoration:none;cursor:pointer}</style>
<img id=figure alt=\"Scientific figure layout prototype\"><div class=switcher><button onclick=\"move(-1)\">← Previous</button><span id=label></span><button onclick=\"move(1)\">Next →</button><a id=pdf>PDF</a></div>
<script>const v=['A','C'];let i=Math.max(0,v.indexOf(new URLSearchParams(location.search).get('variant')));function show(){let x=v[i];figure.src=`figure5_variant_${x}.png`;label.textContent=`Variant ${x}`;pdf.href=`figure5_variant_${x}.pdf`;history.replaceState(null,'',`?variant=${x}`)}function move(n){i=(i+n+v.length)%v.length;show()}addEventListener('keydown',e=>{if(e.key==='ArrowLeft')move(-1);if(e.key==='ArrowRight')move(1)});show()</script>"""
    )


def remove_dropped_layout_b_artifacts() -> None:
    """Prevent stale Option B files from looking like current candidates."""

    for path in (
        OUTPUT_DIR / "figure5_variant_B.png",
        OUTPUT_DIR / "figure5_variant_B.pdf",
    ):
        path.unlink(missing_ok=True)
    build_dir = OUTPUT_DIR / "supervisor_packet_build"
    if build_dir.exists():
        for path in build_dir.glob("page_B.*"):
            path.unlink()


def latex_escape(value: str) -> str:
    """Escape plain supervisor-facing prose for insertion into LaTeX."""

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
    return "".join(replacements.get(character, character) for character in value)


def write_vector_review_page(
    *,
    source: Path,
    title: str,
    caption: str,
    note: str,
    footer: str,
    page_size: tuple[float, float],
    figure_height: float,
    build_dir: Path,
    page_name: str,
) -> Path:
    """Create one captioned page while preserving the source PDF as vector art."""

    if not source.exists():
        raise FileNotFoundError(f"Missing vector source figure: {source}")
    width, height = page_size
    tex_path = build_dir / f"{page_name}.tex"
    tex_path.write_text(
        rf"""\documentclass[10pt]{{article}}
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage[paperwidth={width}in,paperheight={height}in,margin=0.42in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{xcolor}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0pt}}
\pagestyle{{empty}}
\begin{{document}}
\begin{{minipage}}[t][\textheight][t]{{\textwidth}}
\centering
\includegraphics[width=\textwidth,height={figure_height}in,keepaspectratio]{{\detokenize{{{source.resolve().as_posix()}}}}}
\par\vspace{{0.10in}}
\raggedright
{{\fontsize{{14}}{{16}}\selectfont\bfseries {latex_escape(title)}\par}}
\vspace{{0.08in}}
{{\fontsize{{9.3}}{{11.3}}\selectfont {latex_escape(caption)}\par}}
\vfill
{{\fontsize{{7.8}}{{9.4}}\selectfont\color{{gray}} {latex_escape(note)}\par}}
\vspace{{0.06in}}
\hfill{{\fontsize{{7.8}}{{9.4}}\selectfont\color{{gray}} {latex_escape(footer)}}}
\end{{minipage}}
\end{{document}}
""",
        encoding="utf-8",
    )
    subprocess.run(
        [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-jobname={page_name}",
            tex_path.name,
        ],
        cwd=build_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    return build_dir / f"{page_name}.pdf"


def write_supervisor_review_pdf(methods_note: str) -> None:
    """Bundle vector source figures and captions without rasterizing any page."""

    build_dir = OUTPUT_DIR / "supervisor_packet_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    pages = []
    for variant in ("A", "C"):
        heading, caption = SUPERVISOR_CAPTIONS[variant]
        pages.append(
            write_vector_review_page(
                source=OUTPUT_DIR / f"figure5_variant_{variant}.pdf",
                title=f"Option {variant}: {heading}",
                caption=caption,
                note=methods_note,
                footer=f"Layout option {'1' if variant == 'A' else '2'} of 2 ({variant})",
                page_size=((12.0, 16.5) if variant == "C" else (14.0, 10.5)),
                figure_height=(11.8 if variant == "C" else 7.0),
                build_dir=build_dir,
                page_name=f"page_{variant}",
            )
        )
    pages.append(
        write_vector_review_page(
            source=THESIS_FIGURE_5_53_PDF,
            title="Thesis Figure 5.53 extended with JB2006 and JB2008",
            caption=THESIS_FIGURE_5_53_CAPTION,
            note=THESIS_FIGURE_5_53_SCOPE_NOTE,
            footer="Thesis reference figure",
            page_size=(11.0, 14.5),
            figure_height=10.1,
            build_dir=build_dir,
            page_name="page_thesis_5_53",
        )
    )
    output = OUTPUT_DIR / "figure5_layout_options_for_supervisor.pdf"
    subprocess.run(
        [
            "qpdf",
            "--empty",
            "--pages",
            *(str(page) for page in pages),
            "--",
            str(output),
        ],
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Figure 5 PCMCI+ residual-dependence prototype"
    )
    parser.add_argument(
        "--recompute-pcmci", action="store_true", help="ignore compatible PCMCI+ cache"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel PCMCI+ worker processes (default: 1)",
    )
    parser.add_argument(
        "--smoke-altitude", type=int, help="run PCMCI+ for one altitude only"
    )
    parser.add_argument(
        "--tau-max",
        type=int,
        default=PCMCI_TAU_MAX,
        help="daily maximum lag (production default: 180)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.tau_max < 0:
        raise ValueError("--tau-max must be non-negative")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = common_data()
    exact = accuracy_sample(data)
    metrics = calculate_metrics(data)
    selected_altitudes = ALTITUDES
    if args.smoke_altitude is not None:
        if args.smoke_altitude not in ALTITUDES:
            raise ValueError("--smoke-altitude must be one of the production altitudes")
        selected_altitudes = np.array([args.smoke_altitude])
    _, summary, provenance = pcmci_artifacts(
        data, selected_altitudes, args.tau_max, args.recompute_pcmci, args.workers
    )
    for source, run in FORCING_RUNS.items():
        metric = forcing_metric(source)
        selected = summary[(summary.run == run) & (summary.source == source)].copy()
        selected["model"] = selected.target.str.removeprefix("epsilon_")
        selected = selected[
            ["model", "altitude_km", "display_value", "detected"]
        ].rename(columns={"display_value": metric, "detected": f"{metric}_detected"})
        metrics = metrics.merge(selected, on=["model", "altitude_km"], how="left")
    atomic_write_csv(metrics, OUTPUT_DIR / "metrics.csv")
    availability = pd.DataFrame(
        AVAILABILITY, columns=["product", "available_locally", "source", "note"]
    )
    atomic_write_csv(availability, OUTPUT_DIR / "forcing_availability.csv")
    saber_comparison = summary[
        summary.run.isin(["primary", "saber_overlap_core_control", "saber_extension"])
    ].copy()
    atomic_write_csv(
        saber_comparison,
        OUTPUT_DIR / "saber_extension_overlap_comparison.csv",
    )
    provenance.update(
        {
            "accuracy_sample": {
                "n": len(exact),
                "calendar_rows": len(data),
                "calendar_missing_dates": len(data) - len(exact),
                "date_start": exact.date.min().date().isoformat(),
                "date_end": exact.date.max().date().isoformat(),
                "method": "paired deterministic circular calendar-day block bootstrap",
                "sample_contract": "point estimates use the 9,301 finite common dates; the complete 9,333-day calendar retains 32 missing placeholders solely so 27-day resampling blocks preserve physical calendar time",
            },
            "availability_artifact": "forcing_availability.csv",
            "saber_overlap_control_artifact": "saber_extension_overlap_comparison.csv",
        }
    )
    atomic_write_json(provenance, OUTPUT_DIR / "run_sample_provenance.json")
    plot_profile_atlas(metrics, exact)
    plot_heatmap(metrics, exact)
    write_index()
    remove_dropped_layout_b_artifacts()
    write_supervisor_review_pdf(
        supervisor_methods_note(args.tau_max, selected_altitudes)
    )
    print(
        f"Exact common rows: {len(exact):,}; {exact.date.min():%Y-%m-%d} to {exact.date.max():%Y-%m-%d}"
    )
    for column, _, _, _ in metric_specs():
        print(f"{column}: {metrics[column].min():.3f} to {metrics[column].max():.3f}")
    print(f"Wrote {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
