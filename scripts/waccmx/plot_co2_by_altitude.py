"""CO2 time series by altitude from the extracted regional WACCM-X daily series.

Writes two figures to ``data/derived/waccmx_co2_plots_v1``:
- ``co2_by_altitude_overview``: annual means of five representative altitudes on
  one log axis (ordinal blue ramp, darker = higher);
- ``co2_by_altitude_panels``: one panel per altitude with its own ppm axis,
  monthly means and the centred 12-month running mean.
Run restarts (1974, 1998), the excluded restart-relaxation years and the
missing 2022 are marked. 25-cell latitude-weighted regional mean; surface is the
prescribed boundary ``co2vmr``.
"""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import co2_lag_analysis as base  # noqa: E402

INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a63", "#e6e5df", "#fcfcfb"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # validated ordinal ramp (light surface)
OVERVIEW = ["surface", 60, 90, 110, 140]
PANELS = ["surface", 20, 40, 60, 70, 80, 90, 100, 110, 120, 140]
RESTARTS = (1974, 1998)
RELAXATION = ((1950, 1958), (1974, 1982), (1998, 2006))


def monthly_series(directory):
    meta, d = base.load(directory)
    dates = d["dates"].astype("datetime64[D]")
    H = meta["heights_km_geopotential"]
    j = meta["height_vars"].index("CO2")
    keys, t, surf = base.monthly(dates, d["drivers"][:, 0])
    _, _, z = base.monthly(dates, d["fixed_heights"][:, j, :, 1])
    series = {"surface": surf * 1e6}
    series.update({h: z[:, H.index(h)] * 1e6 for h in PANELS if h != "surface"})
    # Running means never bridge the 2022 gap or a run restart.
    run = np.select([keys < np.datetime64("1974-01"), keys < np.datetime64("1998-01")], [0, 1], 2)
    gap = np.r_[False, np.diff(keys.astype(int)) > 1]
    segment = run * 10 + np.cumsum(gap)
    smooth = {}
    for k, v in series.items():
        smooth[k] = np.concatenate([base.running12(v[segment == s]) for s in np.unique(segment)])
    return keys, t, series, smooth


def with_gap(keys, t, y):
    """Insert a NaN where months are missing so lines break across 2022."""
    gap = np.flatnonzero(np.diff(keys.astype(int)) > 1)
    return np.insert(t, gap + 1, np.nan), np.insert(y, gap + 1, np.nan)


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    for a, b in RELAXATION:
        ax.axvspan(a, b, color=GRID, alpha=0.6, linewidth=0, zorder=0)
    for x in RESTARTS:
        ax.axvline(x, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
    ax.axvspan(2022, 2023, color="#ffffff", linewidth=0, zorder=0)


def overview(keys, t, series, path):
    import matplotlib.pyplot as plt
    years = np.floor(t).astype(int)
    yrs = np.arange(years.min(), years.max() + 1)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ends = []
    for color, h in zip(RAMP, OVERVIEW):
        y = np.array([np.nanmean(series[h][years == yr]) if (years == yr).sum() >= 11 else np.nan for yr in yrs])
        label = "surface (prescribed)" if h == "surface" else "%d km" % h
        ax.plot(yrs + 0.5, y, color=color, linewidth=2, marker="o", markersize=3, label=label, zorder=3)
        last = np.flatnonzero(np.isfinite(y))[-1]
        ends.append([np.log10(y[last]), label, yrs[last] + 0.5, y[last]])
    ax.set_yscale("log")
    # Direct labels at the line ends, nudged apart (in log10 units) so near-equal curves stay readable.
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.09)
    for ly, label, x, y in ends:
        ax.annotate(label, (x, y), xytext=(x + 1.2, 10 ** ly), textcoords="data", color=INK, fontsize=8.5,
                    va="center", arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.6))
    ax.set_ylabel("CO2 (ppm, log scale)", color=INK)
    ax.set_xlabel("year", color=INK)
    ax.set_xlim(1949, 2034)
    ax.set_xticks(range(1950, 2030, 10))
    for x, name in ((1950, "run 1"), (1974, "run 2"), (1998, "run 3")):
        ax.text(x + 0.4, 1.01, name, transform=ax.get_xaxis_transform(), color=MUTED, fontsize=8)
    ax.set_title("WACCM-X CO2 by altitude, annual means (16-25N, 200-210E)", color=INK, fontsize=10.5, loc="left",
                 pad=16)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="center left", bbox_to_anchor=(0.01, 0.62), ncol=5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.text(0.01, 0.01, "Shaded: restart relaxation (first 8 years of each run, not usable for transport). "
             "Dashed: run restarts. 2022 not delivered.", color=MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, path)


def panels(keys, t, series, smooth, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 4, figsize=(13, 8.2), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, h in zip(axes.flat, PANELS):
        style(ax)
        x, y = with_gap(keys, t, series[h])
        xs, ys = with_gap(keys, t, smooth[h])
        ax.plot(x, y, color="#9ec5f4", linewidth=0.8, label="monthly mean", zorder=2)
        ax.plot(xs, ys, color="#1c5cab", linewidth=2, label="12-month running mean", zorder=3)
        ax.set_title("surface (prescribed co2vmr)" if h == "surface" else "%d km" % h, color=INK, fontsize=9.5,
                     loc="left")
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("CO2 (ppm)", color=INK)
    axes.flat[0].legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    note = axes.flat[-1]
    note.axis("off")
    note.text(0, 0.95, "Each panel has its own ppm axis.\n\nShaded: restart relaxation\n(first 8 years of each run).\n"
              "Dashed: run restarts (1974, 1998).\nWhite gap: 2022 not delivered.\n\n"
              "Above ~80 km CO2 is depleted by\nphotolysis and diffusive separation,\nand follows the solar cycle.",
              color=MUTED, fontsize=8.5, va="top", transform=note.transAxes)
    fig.suptitle("WACCM-X CO2 by altitude, 1950-2024 (25-cell mean, 16-25N, 200-210E)", color=INK, fontsize=11,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, path)


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=170, facecolor=SURFACE)
    fig.savefig(path.with_suffix(".pdf"), facecolor=SURFACE)


def main():
    import matplotlib
    matplotlib.use("Agg")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/derived/waccmx_regional_daily_v1"))
    p.add_argument("--out", type=Path, default=Path("data/derived/waccmx_co2_plots_v1"))
    a = p.parse_args()
    keys, t, series, smooth = monthly_series(a.data)
    overview(keys, t, series, a.out / "co2_by_altitude_overview")
    panels(keys, t, series, smooth, a.out / "co2_by_altitude_panels")


if __name__ == "__main__":
    main()
