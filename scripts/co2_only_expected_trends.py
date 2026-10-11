"""Expected CO2-only density trends from the Brown et al. (2024) WACCM-X scaling factors.

Brown et al. (2024, JGR Space Physics, doi:10.1029/2024JA032659) publish global-mean
WACCM-X v1.2.2 neutral densities relative to the year 2000 for fixed surface CO2
(368.9-890 ppm), fixed F10.7 (70-200 sfu) and altitude (200-1000 km), with Kp = 0
(``CO2_only`` in doi:10.25500/edata.bham.00001075, CC BY 4.0). These are the
response to CO2 alone.

For each observed or modelled density series, this script builds the CO2-only
counterpart on exactly the days used in its trend fit,
``log10 R(CO2(t), F10.7_81(t), z)``, and fits it with the same estimator
(``fit_trend``: time + F10.7_81 + F10.7_81^2, 27-day HAC). Comparing the two trends
asks how much of each observed trend this model's CO2 response accounts for.
The complementary "without CO2" trend removes the CO2-only signal from the series
day by day and refits it, which keeps the observed HAC interval: an interval
containing zero means no trend is left once this model's CO2 response is removed.
It assumes additivity in log density and ignores uncertainty in the CO2 response.

Assumptions, stated in the outputs:
- surface CO2: NOAA Mauna Loa deseasonalized monthly means, shifted to the NOAA
  global mean by their mean 1979-2026 difference, interpolated to days and
  extrapolated linearly past the last month;
- ln R is taken as linear in CO2 (a constant %/ppm, Brown et al.'s own Table 3
  convention), with the slope fitted by least squares over the grid's first
  four CO2 levels (369-431 ppm, spanning the historical record); the first
  three levels (369-402 ppm) give a sensitivity variant. Single grid segments
  are too noisy at 200-300 km under low solar activity to use piecewise;
- F10.7 outside 70-200 sfu is clamped to the grid edge; altitudes outside
  200-1000 km are skipped; interpolation in altitude and F10.7 is linear.

Run (h5py is not a project dependency):
    uv run --with h5py python -m scripts.co2_only_expected_trends
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import polars as pl

from scripts.recreate_brown_density_trend_current import (
    HASDM_PATH,
    SeriesSpec,
    fit_trend,
    global_mean_specs,
    long_altitude_specs,
)

GRID_PATH = Path("data/original/brown2024_co2_scaling/DEN-CO2scaling2000-2500_v2.nc")
GRID_MD5 = "9373291f93771249565c8157897a967a"
CO2_MLO_MONTHLY = Path("data/original/co2/mauna_loa_monthly.csv")
CO2_GLOBAL_MONTHLY = Path("data/original/co2/global_monthly.csv")
WACCMX_DAILY = Path("data/derived/waccmx_regional_daily_v2")
OUTPUT = Path("outputs/figures/results/co2_only_expected_trends")


class ScalingGrid:
    """ln(rho / rho_2000) on (altitude, CO2, F10.7), interpolated linearly in ln R."""

    def __init__(self, path: Path = GRID_PATH, slope_points: int = 4):
        import h5py

        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest != GRID_MD5:
            raise ValueError(f"unexpected scaling-grid checksum {digest}")
        with h5py.File(path, "r") as f:
            self.alt = f["alt"][()].astype(float)
            self.co2 = f["CO2"][()].astype(float)
            self.f107 = f["F107"][()].astype(float)
            self.ln_r = np.log(f["CO2_only"][()].astype(float))  # (alt, co2, f107)
        if not np.isfinite(self.ln_r).all():
            raise ValueError("non-finite CO2_only scaling factors")
        # Constant d ln R / d CO2 per (altitude, F10.7), through R = 1 at the year-2000 level.
        x = self.co2[:slope_points] - self.co2[0]
        self.slope = np.einsum("c,acf->af", x, self.ln_r[:, :slope_points]) / (x @ x)
        self.slope_points = slope_points

    def ln_ratio(self, altitude_km: float, co2_ppm: np.ndarray, f107: np.ndarray) -> np.ndarray:
        if not self.alt[0] <= altitude_km <= self.alt[-1]:
            raise ValueError("altitude outside the 200-1000 km grid")
        ia = min(np.searchsorted(self.alt, altitude_km, side="right") - 1, len(self.alt) - 2)
        wa = (altitude_km - self.alt[ia]) / (self.alt[ia + 1] - self.alt[ia])
        slope = (1 - wa) * self.slope[ia] + wa * self.slope[ia + 1]  # (f107,)
        if np.max(co2_ppm) > self.co2[3] + 5:  # 431 ppm: historical range of the main fit
            raise ValueError("CO2 beyond the historical range of the constant-sensitivity fit")
        by_f = np.outer(co2_ppm - self.co2[0], slope)
        f = np.clip(f107, self.f107[0], self.f107[-1])
        jf = np.clip(np.searchsorted(self.f107, f, side="right") - 1, 0, len(self.f107) - 2)
        wf = (f - self.f107[jf]) / (self.f107[jf + 1] - self.f107[jf])
        rows = np.arange(len(f))
        return (1 - wf) * by_f[rows, jf] + wf * by_f[rows, jf + 1]


def _noaa(path: Path, value: str) -> pl.DataFrame:
    lines = [line for line in path.read_text().splitlines() if line and not line.startswith("#")]
    df = pl.read_csv("\n".join(lines).encode())
    return df.select(pl.col("year").cast(pl.Int32), pl.col("month").cast(pl.Int32), pl.col(value).alias("ppm"))


def surface_co2(dates: np.ndarray) -> tuple[np.ndarray, float]:
    """Daily surface CO2 (ppm) on the NOAA global scale; returns values and the MLO-global offset."""
    mlo = _noaa(CO2_MLO_MONTHLY, "deseasonalized").filter(pl.col("ppm") > 0)
    glb = _noaa(CO2_GLOBAL_MONTHLY, "trend")
    both = mlo.join(glb, on=["year", "month"], suffix="_global")
    offset = float((both["ppm"] - both["ppm_global"]).mean())
    t = (mlo["year"] + (mlo["month"] - 0.5) / 12).to_numpy()
    y = mlo["ppm"].to_numpy() - offset
    days = np.asarray(dates, dtype="datetime64[D]")
    years = days.astype("datetime64[Y]").astype(int) + 1970
    start = days.astype("datetime64[Y]").astype("datetime64[D]")
    td = years + (days - start).astype(float) / 365.25
    out = np.interp(td, t, y)
    late = td > t[-1]
    if late.any():
        slope = np.polyfit(t[-24:], y[-24:], 1)[0]
        out[late] = y[-1] + slope * (td[late] - t[-1])
    return out, offset


def waccmx_specs() -> list[SeriesSpec]:
    from scripts.waccmx_density_trends import MIN_COVERAGE, specs

    return [spec for spec, coverage in specs(WACCMX_DAILY) if coverage >= MIN_COVERAGE]


def expected_rows(
    spec: SeriesSpec, grid: ScalingGrid, variant: ScalingGrid | None = None
) -> dict[str, object] | None:
    observed = fit_trend(spec)
    if observed is None or not grid.alt[0] <= spec.altitude_km <= grid.alt[-1]:
        return None
    df = (
        spec.df.select(spec.date_col, spec.value_col, spec.f107_col)
        .drop_nulls()
        .filter(pl.col(spec.value_col).is_finite() & (pl.col(spec.value_col) > -100.0))
    )
    co2, _ = surface_co2(df[spec.date_col].to_numpy())
    f107 = df[spec.f107_col].to_numpy().astype(float)
    def co2_only(g: ScalingGrid) -> dict[str, object]:
        synthetic = g.ln_ratio(spec.altitude_km, co2, f107) / np.log(10.0)
        frame = df.with_columns(pl.Series("co2_only_log10_ratio", synthetic))
        fit = fit_trend(
            SeriesSpec(spec.dataset, spec.altitude_km, spec.date_col, "co2_only_log10_ratio", frame, spec.f107_col)
        )
        if fit is None or fit["samples"] != observed["samples"]:
            raise ValueError(f"sample mismatch for {spec.dataset} {spec.altitude_km}")
        return fit

    expected = co2_only(grid)
    alternative = co2_only(variant)["trend_percent_per_decade"] if variant else float("nan")
    # Without CO2: remove the CO2-only signal day by day and refit (assumes additivity in log density).
    without = df.with_columns(
        (pl.col(spec.value_col) - grid.ln_ratio(spec.altitude_km, co2, f107) / np.log(10.0)).alias("without_co2")
    )
    residual = fit_trend(SeriesSpec(spec.dataset, spec.altitude_km, spec.date_col, "without_co2", without, spec.f107_col))
    obs, exp = observed["trend_percent_per_decade"], expected["trend_percent_per_decade"]
    return {
        "dataset": spec.dataset,
        "altitude_km": spec.altitude_km,
        "start_date": observed["start_date"],
        "end_date": observed["end_date"],
        "samples": observed["samples"],
        "co2_ppm_start": float(co2[0]),
        "co2_ppm_end": float(co2[-1]),
        "f107_mean": observed["f107_mean"],
        "observed_trend_pct_per_decade": obs,
        "observed_ci95_lower": observed["trend_percent_per_decade_hac_95_ci_lower"],
        "observed_ci95_upper": observed["trend_percent_per_decade_hac_95_ci_upper"],
        "co2_only_expected_pct_per_decade": exp,
        "co2_only_expected_3pt_slope_pct_per_decade": alternative,
        "co2_only_share_of_observed": exp / obs if obs else float("nan"),
        "without_co2_pct_per_decade": residual["trend_percent_per_decade"],
        "without_co2_ci95_lower": residual["trend_percent_per_decade_hac_95_ci_lower"],
        "without_co2_ci95_upper": residual["trend_percent_per_decade_hac_95_ci_upper"],
    }


def plot(table: pl.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # The estimator module switches Matplotlib to LaTeX text, where "%" starts a comment.
    with matplotlib.rc_context({"text.usetex": False, "font.family": "sans-serif"}):
        _plot(table, path, plt)


def _plot(table: pl.DataFrame, path: Path, plt: object) -> None:

    panels = [
        ("Global mean thermospheric density", "Orbit-derived global mean, 1967-2019", "#1f77b4"),
        ("Mauna Loa HASDM subset", "HASDM (Mauna Loa), 2000-2025", "#ff7f0e"),
        ("WACCM-X run 3 (2006-2024)", "WACCM-X v2.1 regional, 2006-2024", "#e377c2"),
        ("WACCM-X run 2 (1982-1997)", "WACCM-X v2.1 regional, 1982-1997", "#e377c2"),
    ]
    ink, muted, grid_c = "#1f1f1e", "#6b6a63", "#e6e5df"
    fig, axes = plt.subplots(1, 4, figsize=(14, 5.2), sharey=True)
    for ax, (dataset, title, color) in zip(axes, panels):
        sub = table.filter(pl.col("dataset") == dataset).sort("altitude_km")
        ax.grid(color=grid_c, linewidth=0.8)
        ax.axvline(0, color=muted, linewidth=0.8)
        if sub.is_empty():
            ax.text(0.5, 0.5, "no overlap with 200-1000 km grid", transform=ax.transAxes, ha="center", color=muted)
        else:
            z = sub["altitude_km"].to_numpy()
            ax.fill_betweenx(z, sub["observed_ci95_lower"].to_numpy(), sub["observed_ci95_upper"].to_numpy(),
                             color=color, alpha=0.15, linewidth=0)
            ax.plot(sub["observed_trend_pct_per_decade"].to_numpy(), z, color=color, linewidth=2,
                    marker="o", markersize=4, label="observed / modelled total")
            ax.plot(sub["co2_only_expected_pct_per_decade"].to_numpy(), z, color=ink, linewidth=2,
                    linestyle=(0, (4, 2)), marker="s", markersize=4, markerfacecolor="white",
                    label="CO₂-only expectation")
        ax.set_title(title, fontsize=10, color=ink, loc="left")
        ax.set_xlabel("density trend (%/decade)", color=ink)
        ax.tick_params(colors=muted)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("altitude (km)", color=ink)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.text(0.01, 0.01, "CO₂-only expectation: Brown et al. (2024) WACCM-X v1.2.2 CO2_only scaling factors "
             "(CC BY 4.0) on each series' own days, CO2 and F10.7, fitted with the same solar-adjusted "
             "estimator. Shading: 95% HAC interval of the observed/modelled trend.", color=muted, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    grid, variant = ScalingGrid(slope_points=4), ScalingGrid(slope_points=3)
    specs = [
        *global_mean_specs(),
        *long_altitude_specs(HASDM_PATH, "Mauna Loa HASDM subset", "altitude_km", "log10rho_daily_mean"),
        *(waccmx_specs() if WACCMX_DAILY.exists() else []),
    ]
    rows = [row for spec in specs if (row := expected_rows(spec, grid, variant)) is not None]
    table = pl.DataFrame(rows).sort(["dataset", "altitude_km"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.write_csv(args.output_dir / "co2_only_expected_trends.csv")
    plot(table, args.output_dir / "co2_only_expected_trends")
    _, offset = surface_co2(np.array(["2000-01-01"], dtype="datetime64[D]"))
    print(f"MLO - global CO2 offset: {offset:.2f} ppm")
    print(table.select("dataset", "altitude_km", "observed_trend_pct_per_decade",
                       "co2_only_expected_pct_per_decade", "co2_only_expected_3pt_slope_pct_per_decade",
                       "co2_only_share_of_observed"))


if __name__ == "__main__":
    main()
