"""How strong would the CO2 response have to be? Bracketing the without-CO2 test.

``scripts/co2_only_expected_trends.py`` removes one model's CO2 response (Brown et
al. 2024, WACCM-X v1.2.2) from each observed density series and finds no net
trend above ~275 km but a significant decline at 200-250 km. That test treats
the response as exact. This script asks how the conclusion depends on the
response's strength, scaling Brown's response by a factor ``k`` at fixed
altitude shape:

1. **Required scaling.** For each series and altitude, ``k*`` is the factor that
   leaves no net trend, and the ``k`` range is the set of factors whose
   without-CO2 95% HAC interval contains zero. The fit is linear in ``k``: the
   time slope of ``obs - k s`` is ``b_obs - k b_s`` and its HAC variance is a
   quadratic in ``k``, recovered exactly from refits at k = 0, 1, 2. If ``k*``
   is roughly constant with altitude, the observations differ from the model in
   magnitude; if it rises at low altitude, they differ in altitude shape.
2. **Published model responses.** Brown et al. (2024, Table 2) compile modelled
   historical 400-km trends (Qian 2006; Solomon 2015, 2018, 2019; Cnossen 2020;
   Brown 2021). Each is divided by Brown's grid evaluated under the same
   conditions (same F10.7 and CO2 growth over the study period, or, for
   all-solar-activity values, the grid applied to the period's daily F10.7_81
   and CO2 and fitted with the same estimator) to give that study's ``k``
   relative to Brown 2024 at 400 km. Applying it at all altitudes assumes
   Brown's altitude shape; only the 400-km magnitude comes from each study.
   Cnossen (2020) is an all-forcing trend, not a CO2-only response.

Run (h5py is not a project dependency):
    uv run --with h5py python -m scripts.co2_response_bracketing
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from scripts.co2_only_expected_trends import WACCMX_DAILY, ScalingGrid, surface_co2
from scripts.recreate_brown_density_trend_current import (
    F107_COL,
    HASDM_PATH,
    HAC_CONFIDENCE_Z_95,
    SeriesSpec,
    fit_trend,
    global_mean_specs,
    load_space_weather,
    long_altitude_specs,
)

OUTPUT = Path("outputs/figures/results/co2_response_bracketing")
CO2_MLO_ANNUAL = Path("data/original/co2/mauna_loa_annual.csv")
LN10 = np.log(10.0)

# Brown et al. (2024) Table 2: modelled historical neutral-density trends at 400 km
# (%/decade). f107=None means all solar activity. Qian (2006) values at 70 and 210 sfu
# are their 350/450-km average; Solomon (2015) gives two CO2-O quenching rates.
PUBLISHED = [
    ("Qian et al. (2006), 70 sfu", "TIME-GCM 1D", 70.0, 1970, 2000, -2.5, True),
    ("Qian et al. (2006), all", "TIME-GCM 1D", None, 1970, 2000, -1.7, True),
    ("Qian et al. (2006), 210 sfu", "TIME-GCM 1D", 210.0, 1970, 2000, -0.75, True),
    ("Solomon et al. (2015), 70 sfu, kq=1.5e-12", "TIME-GCM", 70.0, 1996, 2008, -4.9, True),
    ("Solomon et al. (2015), 70 sfu, kq=3.0e-12", "TIME-GCM", 70.0, 1996, 2008, -6.8, True),
    ("Solomon et al. (2015), 200 sfu, kq=1.5e-12", "TIME-GCM", 200.0, 1996, 2008, -1.8, True),
    ("Solomon et al. (2015), 200 sfu, kq=3.0e-12", "TIME-GCM", 200.0, 1996, 2008, -2.1, True),
    ("Solomon et al. (2018), 70 sfu", "WACCM-X", 70.0, 1974, 2003, -3.9, True),
    ("Solomon et al. (2019), 200 sfu", "WACCM-X", 200.0, 1974, 2003, -1.7, True),
    ("Cnossen (2020), all", "WACCM-X 2.0", None, 1950, 2015, -2.8, False),
    ("Brown et al. (2021), 70 sfu", "WACCM-X", 70.0, 1975, 2005, -5.8, True),
]


SHAPE_PAIRS = [("Global mean thermospheric density", 250.0), ("Mauna Loa HASDM subset", 200.0),
               ("Mauna Loa HASDM subset", 250.0)]


def annual_co2() -> tuple[np.ndarray, np.ndarray]:
    lines = [line for line in CO2_MLO_ANNUAL.read_text().splitlines() if line and not line.startswith("#")]
    df = pl.read_csv("\n".join(lines).encode())
    return df["year"].to_numpy().astype(float), df["mean"].to_numpy().astype(float)


def model_co2_daily() -> pl.DataFrame:
    """WACCM-X prescribed surface CO2 (CMIP6 historical), one value per date across runs."""
    from scripts.waccmx_density_trends import load

    meta, d = load(WACCMX_DAILY)
    co2 = d["drivers"][:, meta["drivers"].index("co2vmr")] * 1e6
    return (
        pl.DataFrame({"date": d["dates"].astype("datetime64[D]"), "co2": co2})
        .with_columns(pl.col("date").cast(pl.Date))
        .group_by("date").agg(pl.col("co2").first()).sort("date")
    )


def brown_equivalent(grid: ScalingGrid, f107: float | None, start: int, end: int) -> float:
    """Brown (2024) grid trend at 400 km (%/decade) under one study's conditions."""

    if f107 is not None:
        years, ppm = annual_co2()
        sel = (years >= start) & (years <= end)
        rate = np.polyfit(years[sel], ppm[sel], 1)[0]  # ppm/yr over the study period
        ln_r = grid.ln_ratio(400.0, np.array([grid.co2[0], grid.co2[0] + 10 * rate]), np.full(2, f107))
        return float(100 * (np.exp(ln_r[1] - ln_r[0]) - 1))
    sw = load_space_weather().filter(
        (pl.col("date") >= pl.date(start, 1, 1)) & (pl.col("date") <= pl.date(end, 12, 31))
    ).drop_nulls()
    if start < 1959:  # before Mauna Loa: the CMIP6 forcing the WACCM-X runs prescribe
        sw = sw.join(model_co2_daily(), on="date", how="inner")
        co2 = sw["co2"].to_numpy()
    else:
        co2 = surface_co2(sw["date"].to_numpy())[0]
    synthetic = grid.ln_ratio(400.0, co2, sw[F107_COL].to_numpy().astype(float)) / LN10
    fit = fit_trend(SeriesSpec("brown-equivalent", 400.0, "date", "s", sw.with_columns(pl.Series("s", synthetic))))
    return float(fit["trend_percent_per_decade"])


def published_scalings(grid: ScalingGrid) -> pl.DataFrame:
    rows = []
    for study, model, f107, start, end, trend, co2_only in PUBLISHED:
        eq = brown_equivalent(grid, f107, start, end)
        # Compare log-slopes, so k scales Brown's ln-density response exactly.
        k = np.log1p(trend / 100) / np.log1p(eq / 100)
        rows.append({"study": study, "model": model, "f107": f107, "period": f"{start}-{end}",
                     "published_400km_pct_per_decade": trend, "co2_only": co2_only,
                     "brown2024_equivalent_pct_per_decade": eq, "k_relative_to_brown2024": float(k)})
    return pl.DataFrame(rows)


def pct(log10_slope: float) -> float:
    return float((10 ** (10 * log10_slope) - 1) * 100)


def series_and_signal(spec: SeriesSpec, grid: ScalingGrid) -> pl.DataFrame:
    """The fitted series ``y`` (log10 density) and Brown's CO2-only signal ``s`` on its own days."""
    from scripts.waccmx_density_trends import MODEL_CO2_COL

    extra = [MODEL_CO2_COL] if MODEL_CO2_COL in spec.df.columns else []
    df = (
        spec.df.select(spec.date_col, spec.value_col, spec.f107_col, *extra)
        .drop_nulls(subset=[spec.date_col, spec.value_col, spec.f107_col])
        .filter(pl.col(spec.value_col).is_finite() & (pl.col(spec.value_col) > -100.0))
    )
    co2 = df[MODEL_CO2_COL].to_numpy() if extra else surface_co2(df[spec.date_col].to_numpy())[0]
    s = grid.ln_ratio(spec.altitude_km, co2, df[spec.f107_col].to_numpy().astype(float)) / LN10
    return df.select(pl.col(spec.date_col).alias("date"), pl.col(spec.f107_col).alias(F107_COL),
                     pl.col(spec.value_col).alias("y"), pl.Series("s", s))


class Bracket:
    """Trend of ``y - k s`` for any k, from three refits of the same estimator."""

    def __init__(self, dataset: str, altitude_km: float, df: pl.DataFrame):
        fits = []
        for k in (0.0, 1.0, 2.0):
            frame = df.with_columns((pl.col("y") - k * pl.col("s")).alias("y_k"))
            fits.append(fit_trend(SeriesSpec(dataset, altitude_km, "date", "y_k", frame)))
        if any(f is None for f in fits) or len({f["samples"] for f in fits}) != 1:
            raise ValueError(f"unusable fit for {dataset} {altitude_km}")
        self.fit0 = fits[0]
        self.b0 = fits[0]["slope_log10_per_year"]
        self.bs = self.b0 - fits[1]["slope_log10_per_year"]  # time slope of the CO2-only signal
        v = [f["slope_log10_per_year_hac_stderr"] ** 2 for f in fits]
        # var(k) = a + b k + c k^2 through k = 0, 1, 2
        self.var = np.array([v[0], (4 * v[1] - v[2] - 3 * v[0]) / 2, (v[2] - 2 * v[1] + v[0]) / 2])

    def at(self, k: float) -> tuple[float, float, float]:
        b = self.b0 - k * self.bs
        se = np.sqrt(max(0.0, self.var[0] + self.var[1] * k + self.var[2] * k * k))
        z = HAC_CONFIDENCE_Z_95
        return pct(b), pct(b - z * se), pct(b + z * se)

    def k_star(self) -> tuple[float, float, float]:
        """Scaling leaving no net trend, and the range whose interval contains zero."""
        ks = np.linspace(-1, 4, 50001)
        b = self.b0 - ks * self.bs
        se = np.sqrt(np.maximum(0.0, self.var[0] + self.var[1] * ks + self.var[2] * ks**2))
        ok = np.abs(b) <= HAC_CONFIDENCE_Z_95 * se
        lo, hi = (float(ks[ok].min()), float(ks[ok].max())) if ok.any() else (np.nan, np.nan)
        return float(self.b0 / self.bs), lo, hi


def plot(required: pl.DataFrame, scalings: pl.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {"text.usetex": False, "font.family": "sans-serif", "font.size": 9, "axes.labelsize": 9.5,
             "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8}
    with matplotlib.rc_context(style):
        ink, muted, grid_c = "#1f1f1e", "#6b6a63", "#e6e5df"
        series = [("Global mean thermospheric density", "Orbit-derived global mean, 1967-2019", "#1f77b4", -6),
                  ("Mauna Loa HASDM subset", "HASDM (Mauna Loa), 2000-2025", "#ff7f0e", 6)]
        fig, ax = plt.subplots(figsize=(6.6, 5.4))
        ax.grid(color=grid_c, linewidth=0.8, zorder=0)
        pub = scalings.filter(pl.col("co2_only"))["k_relative_to_brown2024"].to_numpy()
        ax.axvspan(pub.min(), pub.max(), color=muted, alpha=0.12, linewidth=0, zorder=0,
                   label="CO₂-only models in Brown et al. (2024, Table 2), 400 km")
        for k in pub:
            ax.axvline(k, color=muted, linewidth=0.6, alpha=0.6, zorder=1)
        ax.axvline(1.0, color=ink, linewidth=1.2, zorder=1, label="Brown et al. (2024) response as published")
        for dataset, label, color, dz in series:
            sub = required.filter(pl.col("dataset") == dataset).sort("altitude_km")
            z = sub["altitude_km"].to_numpy() + dz
            k = sub["k_star"].to_numpy()
            ax.errorbar(k, z, xerr=[k - sub["k_lower"].to_numpy(), sub["k_upper"].to_numpy() - k], fmt="o",
                        color=color, markersize=4.5, capsize=2, linewidth=1.2, label=label, zorder=3)
        ax.set_xlabel("scaling k of the Brown et al. (2024) CO₂ response that leaves no net trend\n"
                      "(point: zero residual; bar: k values whose 95% residual interval contains zero)", color=ink)
        ax.set_ylabel("altitude (km)", color=ink)
        ax.set_xlim(0, 2.2)
        ax.tick_params(colors=muted)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=1)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path.with_suffix(".png"), dpi=180)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    grid = ScalingGrid(slope_points=4)
    scalings = published_scalings(grid)
    specs = [
        *global_mean_specs(),
        *long_altitude_specs(HASDM_PATH, "Mauna Loa HASDM subset", "altitude_km", "log10rho_daily_mean"),
    ]
    required, applied, frames = [], [], {}
    for spec in specs:
        if not grid.alt[0] <= spec.altitude_km <= grid.alt[-1] or fit_trend(spec) is None:
            continue
        frames[(spec.dataset, spec.altitude_km)] = frame = series_and_signal(spec, grid)
        br = Bracket(spec.dataset, spec.altitude_km, frame)
        k_star, k_lo, k_hi = br.k_star()
        obs = br.at(0.0)
        required.append({"dataset": spec.dataset, "altitude_km": spec.altitude_km,
                         "observed_pct_per_decade": obs[0], "co2_only_brown2024_pct_per_decade": pct(br.bs),
                         "k_star": k_star, "k_lower": k_lo, "k_upper": k_hi})
        for row in scalings.iter_rows(named=True):
            mid, lo, hi = br.at(row["k_relative_to_brown2024"])
            applied.append({"dataset": spec.dataset, "altitude_km": spec.altitude_km, "study": row["study"],
                            "k": row["k_relative_to_brown2024"], "without_co2_pct_per_decade": mid,
                            "without_co2_ci95_lower": lo, "without_co2_ci95_upper": hi,
                            "zero_in_interval": lo <= 0 <= hi})
    # Altitude shape: the trend of log10(rho_low / rho_400) against Brown's predicted change in
    # that ratio. Its k range is independent of the overall magnitude of the response.
    shape = []
    for dataset, low in SHAPE_PAIRS:
        lo_f, hi_f = frames[(dataset, low)], frames[(dataset, 400.0)]
        both = lo_f.join(hi_f.select("date", pl.col("y").alias("y_hi"), pl.col("s").alias("s_hi")), on="date")
        diff = both.select("date", F107_COL, (pl.col("y") - pl.col("y_hi")).alias("y"),
                           (pl.col("s") - pl.col("s_hi")).alias("s"))
        br = Bracket(dataset, low, diff)
        k_star, k_lo, k_hi = br.k_star()
        shape.append({"dataset": dataset, "pair": f"{low:g}/400 km", "samples": br.fit0["samples"],
                      "observed_ratio_trend_pct_per_decade": br.at(0.0)[0],
                      "observed_ratio_ci95_lower": br.at(0.0)[1], "observed_ratio_ci95_upper": br.at(0.0)[2],
                      "brown2024_ratio_trend_pct_per_decade": pct(br.bs),
                      "residual_at_k1_pct_per_decade": br.at(1.0)[0],
                      "residual_at_k1_ci95_lower": br.at(1.0)[1], "residual_at_k1_ci95_upper": br.at(1.0)[2],
                      "k_star": k_star, "k_lower": k_lo, "k_upper": k_hi})
    shape_df = pl.DataFrame(shape)
    required_df = pl.DataFrame(required).sort(["dataset", "altitude_km"])
    applied_df = pl.DataFrame(applied).sort(["dataset", "altitude_km", "k"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scalings.write_csv(args.output_dir / "published_scalings.csv")
    required_df.write_csv(args.output_dir / "required_scaling.csv")
    shape_df.write_csv(args.output_dir / "altitude_shape_test.csv")
    applied_df.write_csv(args.output_dir / "without_co2_by_published_response.csv")
    plot(required_df, scalings, args.output_dir / "co2_response_bracketing")
    with pl.Config(tbl_rows=100, tbl_cols=20, tbl_width_chars=200):
        print(scalings)
        print(required_df)
        print(shape_df)


if __name__ == "__main__":
    main()
