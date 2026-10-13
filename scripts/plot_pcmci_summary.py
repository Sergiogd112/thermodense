"""Paper figure: links PCMCI+ retains into each density target, by driver and lag.

Reads the retained-link table exported from the thesis causal-discovery runs
(``causal_links.csv``: method, target_family, altitude_km, cause, target, lag_value,
lag_units, link_type, value) and draws one panel per target: rows are the candidate
drivers, columns the daily lags 0-7, cells the ParCorr partial correlation of each
retained incoming link. Dots mark driver-lag pairs that CMIknn retains. Self-links
(autoregression) are omitted. An empty row means no retained link from that driver.

Run:
    uv run python -m scripts.plot_pcmci_summary --links presentation/_site/data/causal_links.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

OUTPUT = Path("outputs/figures/results/pcmci_summary")
TARGETS = [
    ("Global mean", 325, "Global mean, 325 km"),
    ("HASDM mean (SABER 119 km)", 325, "HASDM mean, 325 km"),
    ("HASDM mean (SABER 119 km)", 825, "HASDM mean, 825 km"),
    ("NRLMSIS 2.0 error", 325, "NRLMSIS 2.0 error, 325 km"),
    ("NRLMSIS 2.0 error", 825, "NRLMSIS 2.0 error, 825 km"),
]
SOURCES = [
    ("F10.7_OBS_CENTER81", "F10.7 (81-day)"),
    ("AP_AVG", "Ap"),
    ("CO2_ppm", "Surface CO₂"),
    ("saber_co2cool_119km", "SABER cooling 119 km"),
    ("surrogate_white_noise_1", "White noise 1"),
    ("surrogate_white_noise_2", "White noise 2"),
]
LAGS = range(8)


def incoming(links: pl.DataFrame, method: str, family: str, altitude: int) -> pl.DataFrame:
    sel = links.filter(
        (pl.col("method") == method) & (pl.col("target_family") == family) & (pl.col("altitude_km") == altitude)
    )
    target = sel.filter(pl.col("cause") == pl.col("target"))["target"]
    targets = [t for t in sel["target"].unique().to_list() if "log10rho" in t or "nrlmsis" in t]
    if len(targets) != 1:
        raise ValueError(f"expected one density target for {method} {family} {altitude}, got {targets} {target}")
    return sel.filter((pl.col("target") == targets[0]) & (pl.col("cause") != targets[0]))


def plot(links: pl.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {"font.family": "sans-serif", "font.size": 8, "axes.titlesize": 8.5}
    with matplotlib.rc_context(style):
        fig, axes = plt.subplots(1, len(TARGETS), figsize=(13, 3.0), sharey=True)
        cmap = plt.get_cmap("RdBu_r")
        for ax, (family, altitude, title) in zip(axes, TARGETS):
            grid = np.full((len(SOURCES), len(LAGS)), np.nan)
            par = incoming(links, "ParCorr", family, altitude)
            for row in par.iter_rows(named=True):
                i = next(k for k, (s, _) in enumerate(SOURCES) if s == row["cause"])
                grid[i, row["lag_value"]] = row["value"]
            ax.imshow(grid, cmap=cmap, vmin=-0.7, vmax=0.7, aspect="auto")
            for i in range(len(SOURCES)):
                for j in LAGS:
                    if np.isfinite(grid[i, j]):
                        ax.text(j, i, f"{grid[i, j]:+.2f}".replace("+0.", "+.").replace("-0.", "−."),
                                ha="center", va="center", fontsize=6.0)
            cmi = incoming(links, "CMIknn", family, altitude)
            for row in cmi.iter_rows(named=True):
                i = next(k for k, (s, _) in enumerate(SOURCES) if s == row["cause"])
                ax.plot(row["lag_value"] + 0.3, i - 0.3, marker="o", markersize=2.6, color="#1f1f1e")
            ax.set_xticks(list(LAGS))
            ax.set_xlabel("lag (days)")
            ax.set_title(title)
            ax.set_yticks(range(len(SOURCES)), [label for _, label in SOURCES])
            ax.set_xticks(np.arange(-0.5, len(LAGS)), minor=True)
            ax.set_yticks(np.arange(-0.5, len(SOURCES)), minor=True)
            ax.grid(which="minor", color="#e6e5df", linewidth=0.6)
            ax.tick_params(which="minor", length=0)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-0.7, 0.7))
        fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.01, label="ParCorr partial correlation")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
        plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--links", type=Path, default=Path("presentation/_site/data/causal_links.csv"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    links = pl.read_csv(args.links)
    co2 = links.filter((pl.col("cause") == "CO2_ppm") & (pl.col("target") != "CO2_ppm"))
    print(f"retained links leaving surface CO2 toward another variable: {co2.height}")
    plot(links, args.output_dir / "pcmci_summary")


if __name__ == "__main__":
    main()
