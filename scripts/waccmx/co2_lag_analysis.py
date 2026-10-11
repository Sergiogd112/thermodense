"""Surface-to-upper-atmosphere CO2 transport lag (tracer age) in the regional WACCM-X history.

Exploratory model diagnostic on the extracted daily series
(``data/derived/waccmx_regional_daily_v1``), 25-cell latitude-weighted mean.
Tracer age at height z: ``age(t) = t - S^-1(c_z(t))``, where S is the
deseasonalized prescribed surface ``co2vmr`` (monotonic since 1950) and c_z the
deseasonalized CO2 at z. Valid only where CO2 is chemically inert (<= 70 km
here); above, the ratio c_z / S(t) reports depletion instead. Runs are never
joined, and the first years after each restart are excluded: the provider
initialised every run with year-2000 CO2/CO fields scaled to the start year.
A joint ``a * S(t - tau)`` fit was tried first and abandoned: with
near-linear growth a and tau are not separately identified.
"""
import argparse
import json
from pathlib import Path

import numpy as np

RUNS = {19490101: "run1 (1950-1973)", 19730101: "run2 (1974-1997)", 19970101: "run3 (1998-2024)"}


def load(directory):
    parts = [np.load(f) for f in sorted(Path(directory).glob("[12][0-9][0-9][0-9].npz"))]
    meta = json.loads(sorted(Path(directory).glob("[12][0-9][0-9][0-9].json"))[0].read_text())
    cat = lambda k: np.concatenate([p[k] for p in parts])
    return meta, {k: cat(k) for k in ("dates", "base_date", "date_written", "drivers", "profiles", "fixed_heights")}


def monthly(dates, values):
    """Calendar-month means; returns decimal-year month centres and (n_months, ...) means."""
    months = dates.astype("datetime64[M]")
    keys, inverse = np.unique(months, return_inverse=True)
    out = np.stack([np.nanmean(values[inverse == i], axis=0) for i in range(len(keys))])
    year = keys.astype(int) // 12 + 1970
    t = year + (keys.astype(int) % 12 + 0.5) / 12
    return keys, t, out


def running12(x):
    """Centred 12-month (2x12) running mean along axis 0; NaN at the ends."""
    w = np.r_[0.5, np.ones(11), 0.5] / 12
    out = np.full_like(x, np.nan, dtype=float)
    for i in range(6, len(x) - 6):
        out[i] = np.tensordot(w, x[i - 6:i + 7], axes=(0, 0))
    return out


def deseasonalize(keys, x):
    """2x12 running mean that never bridges a missing-month gap (2022 in run 3)."""
    gap = np.r_[False, np.diff(keys.astype(int)) > 1]
    segs = np.cumsum(gap)
    return np.concatenate([running12(x[segs == k]) for k in np.unique(segs)])


AGE_HEIGHTS = (20, 40, 60, 70, 80)
RATIO_HEIGHTS = (80, 90, 100, 110, 120)
GROWTH_HEIGHTS = (20, 40, 60, 80, 90, 100, 110, 120, 140)


def growth(t, y, f107a):
    """Trend of ln CO2 in %/decade, raw and with a linear F10.7a (81-day) adjustment."""
    ok = np.isfinite(y) & np.isfinite(f107a)
    raw = np.polyfit(t[ok], np.log(y[ok]), 1)[0]
    X = np.column_stack([np.ones(ok.sum()), t[ok] - t[ok].mean(), f107a[ok] - f107a[ok].mean()])
    b = np.linalg.lstsq(X, np.log(y[ok]), rcond=None)[0]
    return {"raw_pct_per_decade": float(1000 * raw), "solar_adjusted_pct_per_decade": float(1000 * b[1]),
            "pct_per_100_sfu": float(1e4 * b[2])}


def block_starts(t, values, block=24):
    """Start positions of blocks of `block` consecutive months that are all finite.

    Blocks never cross a missing month (the 2022 gap) or a NaN left by the
    running mean, so every block is an uninterrupted 24-month stretch.
    """
    t = np.asarray(t, float)
    ok = np.isfinite(values)
    consecutive = np.r_[np.abs(np.diff(t) - 1 / 12) < 1e-6, False]  # month i+1 follows month i
    starts = []
    for s in range(len(values) - block + 1):
        if ok[s:s + block].all() and consecutive[s:s + block - 1].all():
            starts.append(s)
    if not starts:
        raise ValueError("no uninterrupted block of %d finite months" % block)
    return np.array(starts)


def supported_months(t, values, block=24):
    """Months that lie in at least one valid bootstrap block.

    Estimates and their intervals are computed on these months only, so both
    describe the same sample (e.g. run 3's 12 usable 2023-24 months, too short
    for any 24-month block, are excluded from both).
    """
    keep = np.zeros(len(values), dtype=bool)
    for s in block_starts(t, values, block):
        keep[s:s + block] = True
    return keep


def block_ci(t, values, rng, boot=1000, block=24, stat=np.nanmean):
    """Moving-block bootstrap 95% interval of stat over a monthly series."""
    starts = block_starts(t, values, block)
    nb = max(1, np.isfinite(values).sum() // block)
    draws = [stat(np.concatenate([values[s:s + block] for s in rng.choice(starts, nb)]))
             for _ in range(boot)]
    return list(map(float, np.nanpercentile(draws, [2.5, 97.5])))


def trend_ci(t, y, rng, boot=1000, block=24):
    """Moving-block bootstrap 95% interval of the linear trend (years per decade)."""
    starts = block_starts(t, y, block)
    nb = max(1, np.isfinite(y).sum() // block)
    draws = []
    for _ in range(boot):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.choice(starts, nb)])
        draws.append(slope_per_decade(t[idx], y[idx]))
    return list(map(float, np.nanpercentile(draws, [2.5, 97.5])))


def tracer_age(t, c, T, S):
    """Years since the deseasonalized surface series S(T) last had the value c(t); NaN if never."""
    return t - np.interp(c, S, T, left=np.nan, right=np.nan)


def slope_per_decade(t, y):
    ok = np.isfinite(y)
    return float(np.polyfit(t[ok], y[ok], 1)[0] * 10) if ok.sum() > 24 else float("nan")


def series(directory):
    """Monthly deseasonalized surface S and per-run monthly tracer ages / depletion ratios."""
    meta, d = load(directory)
    dates = d["dates"].astype("datetime64[D]")
    heights = meta["heights_km_geopotential"]
    j = meta["height_vars"].index("CO2")
    sk, st, ss = monthly(dates, d["drivers"][:, 0])
    s12 = deseasonalize(sk, ss)
    good = np.isfinite(s12)
    T, S = st[good], s12[good]
    if not np.all(np.diff(S) > 0):
        raise ValueError("deseasonalized surface CO2 is not monotonic; tracer age undefined")
    runs = {}
    for base, label in RUNS.items():
        sel = d["base_date"] == base
        keys, t, z = monthly(dates[sel], d["fixed_heights"][sel, j, :, 1])
        Z = deseasonalize(keys, z)
        s_now = np.interp(t, T, S, left=np.nan, right=np.nan)
        _, _, f = monthly(dates[sel], d["drivers"][sel, meta["drivers"].index("f107a")])
        _, _, sr = monthly(dates[sel], d["drivers"][sel, 0])
        runs[label] = {"keys": keys, "t": t, "f107a": deseasonalize(keys, f), "surface": deseasonalize(keys, sr),
                       "co2": {h: Z[:, heights.index(h)] for h in GROWTH_HEIGHTS},
                       "age": {h: tracer_age(t, Z[:, heights.index(h)], T, S)
                               for h in AGE_HEIGHTS},
                       "ratio": {h: Z[:, heights.index(h)] / s_now for h in RATIO_HEIGHTS}}
    return meta, (T, S), runs


def analyse(directory, out, skip_years=(6, 8, 10), boot=1000, seed=20261011):
    meta, surface, runs = series(directory)
    rng = np.random.default_rng(seed)
    result = {"version": "waccmx-co2-age-v1", "method": __doc__.strip(),
              "surface": {"first_month_centre": float(surface[0][0]), "last_month_centre": float(surface[0][-1]),
                          "ppm": [float(surface[1][0] * 1e6), float(surface[1][-1] * 1e6)]},
              "deseasonalization": "centred 2x12-month running mean, never bridging missing months (2022)",
              "bootstrap_draws": boot, "bootstrap_seed": seed, "bootstrap_block_months": 24,
              "uncertainty": f"24-month moving-block bootstrap, {boot} draws, blocks never crossing missing "
                             "months; describes month-to-month variability "
                             "of one free-running model realisation, not model or forcing uncertainty",
              "co2_growth_note": "descriptive point estimates without intervals; monthly residuals are "
                                 "autocorrelated and the solar adjustment is a single linear F10.7a term",
              "runs": {}}
    for label, r in runs.items():
        t = r["t"]
        years = np.floor(t - t[0]).astype(int)
        entry = {"first": str(r["keys"][0]), "last": str(r["keys"][-1]), "months": int(len(t)),
                 "age_by_year_since_start": {str(h): [float(np.nanmean(a[years == y])) if np.isfinite(a[years == y]).any()
                                                      else None for y in range(years.max() + 1)]
                                             for h, a in r["age"].items()},
                 "windows": {}}
        for skip in skip_years:
            use = t >= t[0] + skip
            w = {"from": float(t[use][0]), "to": float(t[use][-1]), "months": int(use.sum()), "age": {}, "ratio": {}}
            for h, a in r["age"].items():
                keep = supported_months(t[use], a[use])
                tk, v = t[use][keep], a[use][keep]
                w["age"][str(h)] = {"mean_years": float(np.nanmean(v)), "ci95": block_ci(tk, v, rng, boot),
                                    "trend_years_per_decade": slope_per_decade(tk, v),
                                    "trend_ci95": trend_ci(tk, v, rng, boot),
                                    "months_used": int(keep.sum()), "last_month_used": float(tk[-1])}
            for h, q in r["ratio"].items():
                keep = supported_months(t[use], q[use])
                tk, v = t[use][keep], q[use][keep]
                w["ratio"][str(h)] = {"mean": float(np.nanmean(v)), "ci95": block_ci(tk, v, rng, boot),
                                      "months_used": int(keep.sum())}
            f = np.where(use, r["f107a"], np.nan)
            w["co2_growth"] = {"surface": growth(t, np.where(use, r["surface"], np.nan), f),
                               **{str(h): growth(t, np.where(use, c, np.nan), f) for h, c in r["co2"].items()}}
            entry["windows"][str(skip)] = w
        result["runs"][label] = entry
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    return result, runs


def plot(result, runs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]  # validated categorical slots 1-3 (light surface)
    markers = ["o", "s", "^"]
    ink, muted, grid = "#1f1f1e", "#6b6a63", "#e6e5df"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": muted, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    fig.patch.set_facecolor("#fcfcfb")
    for ax, h in zip(axes[:2], (40, 70)):
        ax.set_facecolor("#fcfcfb")
        ax.grid(color=grid, linewidth=0.8)
        ax.axhline(0, color=muted, linewidth=0.8)
        for k, (label, entry) in enumerate(result["runs"].items()):
            y = np.array([np.nan if v is None else v for v in entry["age_by_year_since_start"][str(h)]])
            x = np.arange(len(y)) + 0.5
            ax.plot(x, y, color=colors[k], linewidth=2, marker=markers[k], markersize=4,
                    markeredgecolor="#fcfcfb", markeredgewidth=1, label=label.split(" ")[0])
        ax.axvspan(0, 8, color=grid, alpha=0.6, linewidth=0)
        ax.text(2.6, 0.04, "excluded:\nrestart relaxation", transform=ax.get_xaxis_transform(),
                color=muted, fontsize=7.5, va="bottom")
        ax.set_title("Tracer age at %d km by year since restart" % h, fontsize=9.5, color=ink, loc="left")
        ax.set_xlabel("years since run start")
        ax.set_ylabel("tracer age (years)")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=ink, loc="lower right")
    ax = axes[2]
    ax.set_facecolor("#fcfcfb")
    ax.grid(color=grid, linewidth=0.8)
    for k, (label, entry) in enumerate(result["runs"].items()):
        w = entry["windows"]["8"]["age"]
        hs = [h for h in AGE_HEIGHTS if h <= 70]
        m = np.array([w[str(h)]["mean_years"] for h in hs])
        lo = m - np.array([w[str(h)]["ci95"][0] for h in hs])
        hi = np.array([w[str(h)]["ci95"][1] for h in hs]) - m
        ax.errorbar(m, np.array(hs) + (k - 1) * 0.8, xerr=[lo, hi], color=colors[k], marker=markers[k],
                    markersize=5, linewidth=2, capsize=0, markeredgecolor="#fcfcfb", markeredgewidth=1,
                    label="%s, %d-%d" % (label.split(" ")[0], int(entry["windows"]["8"]["from"]),
                                         int(entry["windows"]["8"]["age"]["40"]["last_month_used"])))
    ax.set_title("Equilibrated tracer age (window from year 8)", fontsize=9.5, color=ink, loc="left")
    ax.set_xlabel("tracer age (years)")
    ax.set_ylabel("geopotential height (km)")
    ax.legend(frameon=False, fontsize=8, labelcolor=ink, loc="lower right")
    fig.text(0.01, 0.01, "WACCM-X v2.1 regional subset (16-25N, 200-210E), 25-cell mean; deseasonalized monthly CO2 "
             "inverted against the prescribed surface co2vmr. Bars: 24-month block bootstrap 95%.",
             color=muted, fontsize=7)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    fig.savefig(path.with_suffix(".pdf"), facecolor=fig.get_facecolor())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/derived/waccmx_regional_daily_v1"))
    p.add_argument("--out", type=Path, default=Path("data/derived/waccmx_co2_lag_v1/lag.json"))
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--figure", type=Path, default=Path("data/derived/waccmx_co2_lag_v1/tracer_age.png"))
    a = p.parse_args()
    result, runs = analyse(a.data, a.out, boot=a.boot)
    plot(result, runs, a.figure)


if __name__ == "__main__":
    main()
