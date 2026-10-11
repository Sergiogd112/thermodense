"""Solar-adjusted WACCM-X density trends with the Figure 6 estimator.

Reads the extracted regional daily series (``data/derived/waccmx_regional_daily_v2``,
full column 1-700 km; 25-cell latitude-weighted mean DEN at fixed geopotential
heights, produced by ``scripts/waccmx/extract_regional_daily.py --heights v2``) and fits each
height with ``scripts.recreate_brown_density_trend_current.fit_trend``: OLS
``log10 rho ~ time + F10.7_81 + F10.7_81^2`` with 27-day Newey-West/HAC 95%
intervals. F10.7_81 is the model's own centred 81-day ``f107a`` forcing.

Runs are never joined. Each restart re-initialised CO2/CO, and tracer ages need
~8 years to settle (see the 11 Oct lag receipt), so only the settled windows are
fitted: run 2 1982-1997 and run 3 2006-2024 (2022 missing). A day is missing
at a height if any hour lies above the model column top; heights covering fewer
than 95% of a window's days are not reported, because the top drops mostly at
solar minimum and the remaining days would bias the trend. Altitudes are
reported as geometric heights converted from geopotential.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from scripts.recreate_brown_density_trend_current import F107_COL, SeriesSpec, fit_trend

EARTH_RADIUS_KM = 6371.22  # CAM/WACCM-X rearth
MIN_COVERAGE = 0.95
# Days the provider never delivered (see the 10 Oct download receipt); any other
# missing day means an incomplete extraction, not a model-top gap.
DOCUMENTED_MISSING = (
    {np.datetime64("2022-01-01") + i for i in range(365)}
    | {np.datetime64("2023-01-31"), np.datetime64("2023-02-03")}
)
MODEL_CO2_COL = "model_co2vmr_ppm"
RUN_WINDOWS = {
    "WACCM-X run 3 (2006-2024)": (19970101, "2006-01-01", "2024-12-31"),
    "WACCM-X run 2 (1982-1997)": (19730101, "1982-01-01", "1997-12-31"),
}


def geometric_km(geopotential_km: float) -> float:
    return EARTH_RADIUS_KM * geopotential_km / (EARTH_RADIUS_KM - geopotential_km)


def load(directory: Path) -> tuple[dict, dict[str, np.ndarray]]:
    files = sorted(directory.glob("[12][0-9][0-9][0-9].npz"))
    if not files:
        raise FileNotFoundError(f"no extracted WACCM-X years in {directory}")
    meta = json.loads(files[0].with_suffix(".json").read_text())
    parts = [np.load(f) for f in files]
    keys = ("dates", "base_date", "drivers", "fixed_heights")
    return meta, {k: np.concatenate([p[k] for p in parts]) for k in keys}


def specs(directory: Path) -> list[tuple[SeriesSpec, float]]:
    """Return (spec, day coverage) for every height and settled run window."""

    meta, d = load(directory)
    heights = meta["heights_km_geopotential"]
    den = meta["height_vars"].index("DEN")
    f107a = meta["drivers"].index("f107a")
    co2vmr = meta["drivers"].index("co2vmr")
    dates = d["dates"].astype("datetime64[D]")
    out = []
    for dataset, (base, first, last) in RUN_WINDOWS.items():
        sel = (d["base_date"] == base) & (dates >= np.datetime64(first)) & (dates <= np.datetime64(last))
        expected = set(np.arange(np.datetime64(first), np.datetime64(last) + 1)) - DOCUMENTED_MISSING
        present = set(dates[sel])
        if present != expected:
            missing = sorted(expected - present)
            raise ValueError(f"{dataset}: extraction is incomplete ({len(missing)} undocumented missing "
                             f"days, first {missing[:3]}); re-run the extractor")
        # The model's own prescribed surface CO2 (global mean, mole fraction -> ppm), kept so
        # CO2-response comparisons use the forcing this run actually saw.
        columns = {"date": dates[sel], F107_COL: d["drivers"][sel, f107a],
                   MODEL_CO2_COL: d["drivers"][sel, co2vmr] * 1e6}
        for h in heights:
            k = meta["heights_km_geopotential"].index(h)
            with np.errstate(divide="ignore", invalid="ignore"):
                columns[f"waccmx_log10rho_{h}km"] = np.log10(d["fixed_heights"][sel, den, k, 1])
        df = pl.DataFrame(columns).with_columns(pl.col("date").cast(pl.Date))
        for h in heights:
            column = f"waccmx_log10rho_{h}km"
            coverage = float(np.isfinite(columns[column]).mean())
            out.append((SeriesSpec(dataset, round(geometric_km(h), 1), "date", column, df), coverage))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/derived/waccmx_regional_daily_v2"))
    p.add_argument("--out", type=Path,
                   default=Path("data/derived/waccmx_density_trends_v1/waccmx_density_trends.csv"))
    a = p.parse_args()
    rows, skipped = [], []
    for spec, coverage in specs(a.data):
        if coverage < MIN_COVERAGE:
            skipped.append((spec.dataset, spec.altitude_km, round(coverage, 3)))
            continue
        if (trend := fit_trend(spec)) is not None:
            rows.append({**trend, "day_coverage": coverage})
    table = pl.DataFrame(rows).sort(["dataset", "altitude_km"])
    print("skipped (coverage < %.0f%%):" % (100 * MIN_COVERAGE), skipped)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(a.out)
    print(table.select("dataset", "altitude_km", "samples", "duration_years", "trend_percent_per_decade",
                       "trend_percent_per_decade_hac_95_ci_lower", "trend_percent_per_decade_hac_95_ci_upper"))


if __name__ == "__main__":
    main()
