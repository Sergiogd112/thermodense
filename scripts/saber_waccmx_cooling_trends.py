"""Solar-adjusted trends of lower-thermosphere CO2 15-um cooling: SABER against WACCM-X.

The density test (``scripts/co2_response_bracketing.py``) finds that observed density
trends weaken less towards 200-250 km than Brown et al.'s (2024) modelled CO2
response. The CO2 cooling that drives the contraction acts mainly at 100-140 km.
This script compares its observed and modelled trends over the same region
(16-25 N, 200-210 E) and years:

- **SABER**: the official per-scan CO2 cooling flux integrated over 100-139 km
  (W/m2), from the regionally verified extraction (13,902 scans, 2002-01-25 to
  2019-12-01). Primary fit: daily means with the density estimator
  (``fit_trend``: log10 y ~ time + F10.7_81 + F10.7_81^2, 27-day HAC). Because
  SABER drifts through local time on a ~60-day yaw cycle and cooling depends on
  local time, a second fit uses every scan with two harmonics of solar local time
  and of day of year added, and the same HAC covariance summed by day.
- **WACCM-X v2.1 run 3** (regional daily means, settled window from 2006): a
  cooling proxy from the model's own CO2, O, T and density,
  ``L(z) ~ n_CO2 n_O exp(-960 K / T)`` (collisional excitation of CO2(v2) by O in
  the optically thin limit; constants cancel in a relative trend), integrated
  over 100-140 km geopotential (~102-143 km geometric). Daily means of the
  inputs are used, so the diurnal covariance of T and O is ignored. The proxy
  trend is decomposed into its CO2, O, number-density and temperature factors.

Both are fitted on the common window 2006-01-01 to 2019-12-01 and on each
record's own window. This is a fingerprint comparison, not an attribution: the
model trend includes all of its forcings.

Run:
    uv run python -m scripts.saber_waccmx_cooling_trends
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from scripts.recreate_brown_density_trend_current import (
    F107_COL,
    HAC_CONFIDENCE_Z_95,
    SeriesSpec,
    decimal_year,
    fit_trend,
    load_space_weather,
    newey_west_covariance,
)
from scripts.waccmx_density_trends import geometric_km, load

SABER_SCANS = Path(
    "data/products/co2-paper-execution-20261006/saber-fixed-region-2002-2019-regional-v1/scans.parquet"
)
WACCMX_DAILY = Path("data/derived/waccmx_regional_daily_v2")
OUTPUT = Path("outputs/figures/results/saber_waccmx_cooling")
RUN3_BASE = 19970101
COMMON = ("2006-01-01", "2019-12-01")
AMU = {"CO2": 44.0, "O": 16.0, "O2": 32.0, "N2_vmr": 28.0, "He": 4.0}
LAYER_KM = (100, 110, 120, 130, 140)  # geopotential
REGION_COLUMN = 1  # latitude-weighted 25-cell mean


def saber_scans() -> pl.DataFrame:
    scans = pl.read_parquet(SABER_SCANS).filter(pl.col("valid_flux") & (pl.col("official_flux_w_m2") > 0))
    sw = load_space_weather()
    return (
        scans.with_columns(pl.col("timestamp_utc").dt.date().alias("date"))
        .join(sw, on="date", how="left")
        .select("date", "timestamp_utc", "slt_hours", "official_flux_w_m2", F107_COL)
        .drop_nulls()
        .sort("timestamp_utc")
    )


def waccmx_proxy() -> pl.DataFrame:
    meta, d = load(WACCMX_DAILY)
    sel = d["base_date"] == RUN3_BASE
    fh = d["fixed_heights"][sel][..., REGION_COLUMN].astype(float)  # (day, var, height)
    var = {name: i for i, name in enumerate(meta["height_vars"])}
    idx = [meta["heights_km_geopotential"].index(h) for h in LAYER_KM]
    z_m = np.array([geometric_km(h) for h in LAYER_KM]) * 1e3
    get = {name: fh[:, var[name]][:, idx] for name in var}
    mean_mass = sum(get[name] * AMU[name] for name in AMU) * 1.66053906660e-27  # kg per molecule
    n = get["DEN"] / mean_mass  # m^-3
    factors = {"co2_vmr": get["CO2"], "o_vmr": get["O"], "n_total_sq": n**2, "boltzmann": np.exp(-960.0 / get["T"])}
    local = factors["co2_vmr"] * factors["o_vmr"] * factors["n_total_sq"] * factors["boltzmann"]
    flux = np.trapezoid(local, z_m, axis=1)
    out = {"date": d["dates"][sel].astype("datetime64[D]"), F107_COL: d["drivers"][sel, meta["drivers"].index("f107a")],
           "proxy": flux}
    # Layer-mean log factors, weighted by each level's share of the column cooling, for a decomposition.
    w = local / local.sum(axis=1, keepdims=True)
    for name, f in factors.items():
        out[f"ln_{name}"] = (w * np.log(f)).sum(axis=1)
    return pl.DataFrame(out).with_columns(pl.col("date").cast(pl.Date)).filter(pl.col("proxy").is_finite())


def window(df: pl.DataFrame, first: str, last: str) -> pl.DataFrame:
    return df.filter((pl.col("date") >= pl.lit(first).str.to_date()) & (pl.col("date") <= pl.lit(last).str.to_date()))


def fit(dataset: str, df: pl.DataFrame, value: str, label: str) -> dict[str, object]:
    frame = df.with_columns(pl.col(value).log10().alias("y")) if not value.startswith("ln_") else \
        df.with_columns((pl.col(value) / np.log(10.0)).alias("y"))
    r = fit_trend(SeriesSpec(dataset, 120.0, "date", "y", frame))
    return {"series": dataset, "quantity": label, "estimator": "daily means, time + F10.7_81 + F10.7_81^2",
            "start": str(r["start_date"]), "end": str(r["end_date"]), "samples": r["samples"],
            "trend_pct_per_decade": r["trend_percent_per_decade"],
            "ci95_lower": r["trend_percent_per_decade_hac_95_ci_lower"],
            "ci95_upper": r["trend_percent_per_decade_hac_95_ci_upper"]}


def fit_scans(scans: pl.DataFrame) -> dict[str, object]:
    """Per-scan fit with local-time and seasonal harmonics; HAC scores summed by day."""
    dates = np.array(scans["date"].to_list(), dtype=object)
    t = decimal_year(dates)
    f = scans[F107_COL].to_numpy().astype(float)
    slt = 2 * np.pi * scans["slt_hours"].to_numpy() / 24.0
    doy = 2 * np.pi * (t % 1.0)
    cols = [t - t.mean(), f - f.mean(), (f - f.mean()) ** 2]
    for k in (1, 2):
        cols += [np.sin(k * slt), np.cos(k * slt), np.sin(k * doy), np.cos(k * doy)]
    x = np.column_stack([*cols, np.ones_like(t)])
    y = np.log10(scans["official_flux_w_m2"].to_numpy())
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    cov = newey_west_covariance(x, y - x @ beta, np.array([d.toordinal() for d in dates]))
    se = np.sqrt(cov[0, 0])
    pct = lambda b: float((10 ** (10 * b) - 1) * 100)  # noqa: E731
    return {"series": "SABER (regional scans)", "quantity": "100-139 km CO2 cooling flux",
            "estimator": "scans, + 2 SLT and 2 day-of-year harmonics", "start": str(dates[0]), "end": str(dates[-1]),
            "samples": len(y), "trend_pct_per_decade": pct(beta[0]),
            "ci95_lower": pct(beta[0] - HAC_CONFIDENCE_Z_95 * se), "ci95_upper": pct(beta[0] + HAC_CONFIDENCE_Z_95 * se)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    scans = saber_scans()
    daily = scans.group_by("date").agg(pl.col("official_flux_w_m2").mean(), pl.col(F107_COL).first()).sort("date")
    model = waccmx_proxy()
    rows = [
        fit("SABER", daily, "official_flux_w_m2", "100-139 km CO2 cooling flux"),
        fit("SABER", window(daily, *COMMON), "official_flux_w_m2", "100-139 km CO2 cooling flux"),
        fit_scans(scans),
        fit_scans(window(scans, *COMMON)),
        fit("WACCM-X run 3", window(model, "2006-01-01", "2024-12-31"), "proxy", "100-140 km cooling proxy"),
        fit("WACCM-X run 3", window(model, *COMMON), "proxy", "100-140 km cooling proxy"),
    ]
    for name in ("co2_vmr", "o_vmr", "n_total_sq", "boltzmann"):
        rows.append(fit("WACCM-X run 3", window(model, *COMMON), f"ln_{name}", f"proxy factor: {name}"))
    table = pl.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.write_csv(args.output_dir / "cooling_trends.csv")
    with pl.Config(tbl_rows=50, tbl_cols=12, tbl_width_chars=220, fmt_str_lengths=60):
        print(table)


if __name__ == "__main__":
    main()
