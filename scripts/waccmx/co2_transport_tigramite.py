"""How long surface CO2 signals take to reach the upper atmosphere in WACCM-X, using tigramite.

Signal: interannual anomalies of the 12-month log growth of CO2,
``g(t) = ln c(t) - ln c(t - 12 months)``, which removes the seasonal cycle, then
linearly detrended within each dataset. The prescribed surface ``co2vmr`` carries
year-to-year growth variations (e.g. the early-1990s slowdown); their arrival
time at each height is the transport delay of growth anomalies (closer to the
modal than to the mean transit time of a broad age spectrum).

Datasets: settled windows only, never joined (tigramite ``analysis_mode="multiple"``):
run 2 1982-01..1997-12 and run 3 2006-01..2021-12 (see the 11 Oct lag receipt).
Variables: surface, CO2 at fixed geopotential heights, and the 12-month change of
F10.7a. Background knowledge: surface CO2 and F10.7a are prescribed forcings,
so no link may point into them.

1. Lagged dependence (tigramite ParCorr, unconditional) of each height on the
   surface, lags 0..96 months on identical samples (cut_off = tau_max), raw and
   after removing the contemporaneous F10.7a term.
2. PCMCI+ (ParCorr) on the vertical chain, tau_max 36 months.
3. Synthetic recovery check with known delays and a gamma age spectrum.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import co2_lag_analysis as base  # noqa: E402

HEIGHTS = (20, 40, 60, 70, 80, 90, 100, 110, 120, 140)
WINDOWS = {19730101: ("run2", np.datetime64("1982-01"), np.datetime64("1997-12")),
           19970101: ("run3", np.datetime64("2006-01"), np.datetime64("2021-12"))}
VAR_NAMES = ["surface"] + ["%d km" % h for h in HEIGHTS] + ["dF10.7a"]


def growth12(keys, x):
    """ln x(t) - ln x(t-12), only where the month 12 earlier exists."""
    pos = {int(k): i for i, k in enumerate(keys.astype(int))}
    out = np.full(len(x), np.nan)
    for i, k in enumerate(keys.astype(int)):
        j = pos.get(k - 12)
        if j is not None:
            out[i] = np.log(x[i]) - np.log(x[j])
    return out


def diff12(keys, x):
    pos = {int(k): i for i, k in enumerate(keys.astype(int))}
    out = np.full(len(x), np.nan)
    for i, k in enumerate(keys.astype(int)):
        j = pos.get(k - 12)
        if j is not None:
            out[i] = x[i] - x[j]
    return out


def standardize(y):
    t = np.arange(len(y))
    ok = np.isfinite(y)
    y = y - np.polyval(np.polyfit(t[ok], y[ok], 1), t)
    return (y - np.nanmean(y)) / np.nanstd(y)


def build(directory):
    meta, d = base.load(directory)
    dates = d["dates"].astype("datetime64[D]")
    H = meta["heights_km_geopotential"]
    j = meta["height_vars"].index("CO2")
    fi = meta["drivers"].index("f107a")
    datasets, info = {}, {}
    for n, (b, (label, first, last)) in enumerate(WINDOWS.items()):
        sel = d["base_date"] == b
        keys, t, co2 = base.monthly(dates[sel], d["fixed_heights"][sel, j, :, 1])
        _, _, surf = base.monthly(dates[sel], d["drivers"][sel, 0])
        _, _, f107a = base.monthly(dates[sel], d["drivers"][sel, fi])
        cols = [growth12(keys, surf)] + [growth12(keys, co2[:, H.index(h)]) for h in HEIGHTS] + [diff12(keys, f107a)]
        X = np.column_stack(cols)
        use = (keys >= first) & (keys <= last)
        X = X[use]
        if np.isnan(X).any():
            raise ValueError("missing values inside %s window" % label)
        datasets[n] = np.column_stack([standardize(X[:, c]) for c in range(X.shape[1])])
        info[label] = {"first": str(keys[use][0]), "last": str(keys[use][-1]), "months": int(use.sum())}
    return datasets, info


def lag_profiles(datasets, tau_max=96, residualize_solar=False):
    from tigramite import data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    data = {}
    for k, X in datasets.items():
        X = X.copy()
        if residualize_solar:  # remove the contemporaneous solar term from every height series
            f = X[:, -1]
            for c in range(1, X.shape[1] - 1):
                X[:, c] -= (f @ X[:, c]) / (f @ f) * f
        data[k] = X
    df = pp.DataFrame(data, var_names=VAR_NAMES, analysis_mode="multiple")
    test = ParCorr(significance="analytic")
    test.set_dataframe(df)
    # Same samples for every lag: cut only tau_max months per dataset (PCMCI's default cuts 2 x tau_max).
    N = len(VAR_NAMES)
    val, p = np.full((N, N, tau_max + 1), np.nan), np.full((N, N, tau_max + 1), np.nan)
    for j in range(1, len(HEIGHTS) + 1):
        for tau in range(tau_max + 1):
            val[0, j, tau], p[0, j, tau] = test.run_test(X=[(0, -tau)], Y=[(j, 0)], Z=[], tau_max=tau_max,
                                                            cut_off="max_lag_or_tau_max")
    out = {}
    for j, h in enumerate(HEIGHTS, start=1):
        r = val[0, j, :]
        peak = int(np.nanargmax(r))
        half = np.flatnonzero(r >= r[peak] / 2) if r[peak] > 0 else np.array([], int)
        out[str(h)] = {"peak_lag_months": peak, "peak_r": float(r[peak]), "peak_p": float(p[0, j, peak]),
                       "half_max_lags_months": [int(half.min()), int(half.max())] if len(half) else None,
                       "r_by_lag": [None if not np.isfinite(v) else round(float(v), 4) for v in r]}
    return out


def pcmciplus(datasets, tau_max=36, pc_alpha=0.01):
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr
    df = pp.DataFrame(datasets, var_names=VAR_NAMES, analysis_mode="multiple")
    N = len(VAR_NAMES)
    exogenous = (0, N - 1)
    assumptions = {j: {} for j in range(N)}
    for j in range(N):
        if j in exogenous:
            continue  # prescribed forcings: no parents
        for i in range(N):
            for tau in range(0, tau_max + 1):
                if i == j and tau == 0:
                    continue
                if tau == 0 and i not in exogenous:
                    assumptions[j][(i, 0)] = "o?o"
                elif tau == 0:
                    assumptions[j][(i, 0)] = "-?>"
                    assumptions[i][(j, 0)] = "<?-"
                else:
                    assumptions[j][(i, -tau)] = "-?>"
    pc = PCMCI(dataframe=df, cond_ind_test=ParCorr(significance="analytic"), verbosity=0)
    res = pc.run_pcmciplus(link_assumptions=assumptions, tau_max=tau_max, pc_alpha=pc_alpha)
    graph, val, p = res["graph"], res["val_matrix"], res["p_matrix"]
    links = []
    for i in range(N):
        for j in range(N):
            for tau in range(tau_max + 1):
                if graph[i, j, tau] not in ("", None):
                    if tau == 0 and i > j and graph[i, j, 0] in ("<--",):
                        continue
                    links.append({"from": VAR_NAMES[i], "to": VAR_NAMES[j], "lag_months": tau,
                                  "type": str(graph[i, j, tau]), "val": float(val[i, j, tau]),
                                  "p": float(p[i, j, tau])})
    return links


def lag_r(datasets, tau_max=96, residualize_solar=False, surface=None):
    """Pearson r of height(t) on surface(t - tau) on the same samples ParCorr uses (cut tau_max per dataset)."""
    out = np.zeros((len(HEIGHTS), tau_max + 1))
    xs, ys = [], []
    for k, X in datasets.items():
        X = X.copy()
        if residualize_solar:
            f = X[:, -1]
            for c in range(1, X.shape[1] - 1):
                X[:, c] -= (f @ X[:, c]) / (f @ f) * f
        s = X[:, 0] if surface is None else surface[k]
        T = len(X)
        xs.append(np.stack([s[tau_max - tau:T - tau] for tau in range(tau_max + 1)]))
        ys.append(X[tau_max:, 1:1 + len(HEIGHTS)])
    x, y = np.concatenate(xs, axis=1), np.concatenate(ys, axis=0)
    x = (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)
    y = (y - y.mean(0)) / y.std(0)
    return (y.T @ x.T) / y.shape[0]


def phase_surrogate(s, rng):
    f = np.fft.rfft(s - s.mean())
    phases = rng.uniform(0, 2 * np.pi, len(f))
    phases[0] = 0
    if len(s) % 2 == 0:
        phases[-1] = 0
    return np.fft.irfft(np.abs(f) * np.exp(1j * phases), n=len(s)) + s.mean()


def surrogate_test(datasets, rng, n=1000, tau_max=96, residualize_solar=False):
    """Null: surface signal with the same power spectrum but random timing; statistic = max r over lags."""
    obs = lag_r(datasets, tau_max, residualize_solar).max(axis=1)
    null = np.array([lag_r(datasets, tau_max, residualize_solar,
                           surface={k: phase_surrogate(X[:, 0], rng) for k, X in datasets.items()}).max(axis=1)
                     for _ in range(n)])
    return {str(h): {"max_r": float(obs[i]), "p_surrogate": float((1 + (null[:, i] >= obs[i]).sum()) / (n + 1)),
                     "null_95th": float(np.percentile(null[:, i], 95))} for i, h in enumerate(HEIGHTS)}


def synthetic(datasets, rng, noise=0.5):
    """Replace the height series by delayed copies of the observed surface signal; recover the delays."""
    truths = {}
    syn = {}
    for k, X in datasets.items():
        s = X[:, 0]
        Y = X.copy()
        for c, (kind, a, b) in enumerate([("shift", 12, 0), ("shift", 36, 0), ("gamma", 36, 2.0),
                                          ("gamma", 50, 2.0), ("shift", 50, 0)], start=1):
            if kind == "shift":
                kernel = np.zeros(a + 1); kernel[a] = 1
            else:  # gamma age spectrum with mean a months and shape b (mode a*(b-1)/b)
                lags = np.arange(0, 4 * a)
                kernel = lags ** (b - 1) * np.exp(-lags * b / a); kernel /= kernel.sum()
            y = np.convolve(np.r_[np.repeat(s[0], len(kernel)), s], kernel)[len(kernel):len(kernel) + len(s)]
            y = (y - y.mean()) / y.std() + noise * rng.standard_normal(len(y))
            Y[:, c] = y
            truths[str(HEIGHTS[c - 1])] = {"kind": kind, "mean_lag_months": a,
                                           "mode_lag_months": a if kind == "shift" else a * (b - 1) / b}
        syn[k] = Y
    rec = lag_profiles(syn, tau_max=96)
    return {h: {**truths[h], "recovered_peak_months": rec[h]["peak_lag_months"], "peak_r": rec[h]["peak_r"]}
            for h in truths}


def plot(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blue, orange = "#2a78d6", "#eb6834"  # validated categorical slots 1-2
    ink, muted, grid, surface = "#1f1f1e", "#6b6a63", "#e6e5df", "#fcfcfb"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": muted, "xtick.color": muted, "ytick.color": muted,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True)
    fig.patch.set_facecolor(surface)
    for ax, h in zip(axes, ("20", "40", "60", "110")):
        ax.set_facecolor(surface)
        ax.grid(color=grid, linewidth=0.8)
        ax.axhline(0, color=muted, linewidth=0.8)
        for key, color, label, surr in (("lag_dependence", blue, "raw", "raw"),
                                        ("lag_dependence_solar_removed", orange, "solar removed", "solar_removed")):
            r = np.array(result[key][h]["r_by_lag"], float)
            ax.plot(np.arange(len(r)) / 12, r, color=color, linewidth=2, label=label)
            ax.axhline(result["surrogate_test"][surr][h]["null_95th"], color=color, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_title("%s km" % h, fontsize=9.5, color=ink, loc="left")
        ax.set_xlabel("lag after surface (years)")
    axes[0].set_ylabel("correlation of CO2 growth anomalies")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=ink, loc="upper right", bbox_to_anchor=(1.0, 0.86))
    fig.text(0.01, 0.01, "WACCM-X runs 2 (1982-1997) and 3 (2006-2021); tigramite ParCorr, lags 0-8 years. Dashed: "
             "95th percentile of the maximum over lags under 1000 phase-randomized surface surrogates.",
             color=muted, fontsize=7)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=180, facecolor=surface)
    fig.savefig(path.with_suffix(".pdf"), facecolor=surface)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/derived/waccmx_regional_daily_v1"))
    p.add_argument("--out", type=Path, default=Path("data/derived/waccmx_co2_tigramite_v1/result.json"))
    p.add_argument("--tau-max-pcmci", type=int, default=36)
    a = p.parse_args()
    started = time.time()
    datasets, info = build(a.data)
    rng = np.random.default_rng(20261011)
    result = {"version": "waccmx-co2-tigramite-v1", "method": __doc__.strip(), "datasets": info,
              "variables": VAR_NAMES}
    result["synthetic_check"] = synthetic(datasets, rng)
    print(json.dumps({"event": "synthetic", "seconds": round(time.time() - started)}), flush=True)
    result["lag_dependence"] = lag_profiles(datasets)
    result["lag_dependence_solar_removed"] = lag_profiles(datasets, residualize_solar=True)
    result["surrogate_test"] = {"note": "1000 phase-randomized surrogates of the surface series per dataset; "
                                        "statistic max r over lags 0..96 (controls autocorrelation and lag search)",
                                "raw": surrogate_test(datasets, rng),
                                "solar_removed": surrogate_test(datasets, rng, residualize_solar=True)}
    print(json.dumps({"event": "lags", "seconds": round(time.time() - started)}), flush=True)
    result["pcmciplus"] = {"tau_max": a.tau_max_pcmci, "pc_alpha": 0.01,
                           "links": pcmciplus(datasets, tau_max=a.tau_max_pcmci)}
    result["seconds"] = round(time.time() - started)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1) + "\n")
    plot(result, a.out.with_name("lag_correlation.png"))
    print(json.dumps({"event": "done", "seconds": result["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
