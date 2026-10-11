"""Cheap calendar/CO2 control study on the through-2025 Karman rows (design 2026-10-10).

prep:   gather identical train/validation/test rows once into a float32 cache.
fit:    one (cell, model, seed) fit; writes per-day x category error sums only.
report: paired cell contrasts with the paired-month bootstrap used for full25.

Read-only on the campaign and prepared arrays; writes only under --out.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

SOURCE = Path("/home/sergio/karman-phoenix/data/prepared-full25")
CONFIRM = Path("/home/sergio/karman-phoenix/bounded-20261006-full25/data/confirm")
STATIC = ["all__year__[y]", "tudelft_thermo__altitude__[m]", "tudelft_thermo__latitude__[deg]",
          "noaa_gml__co2_previous_day_observed__[ppm]", "noaa_gml__co2_trailing_365_day_mean__[ppm]"]
YEAR, ALT, CO2 = 0, 1, [3, 4]
F10, F10B, DTC = 5, 6, 13
BASE = [i for i in range(28) if i not in (YEAR, *CO2)]
ORIGINAL_MONTHS = ["2004-03", "2005-05", "2006-07", "2007-09", "2008-11", "2009-01", "2010-03",
                   "2011-05", "2012-07", "2013-09", "2014-11", "2015-01", "2016-03", "2017-05",
                   "2018-07", "2019-09"]
CATEGORIES = ["Overall"] + ["{}-{} km".format(a, a + 50) for a in range(200, 550, 50)] + [
    "Quiet (Ap 0-15)", "Mild (Ap 15-30)", "Minor (Ap 30-50)", "Major (Ap 50+)"]
CELLS = {
    "N0": "base",
    "N1": "base + CO2",
    "A": "base + integer year",
    "B": "base + integer year + CO2",
    "C": "base + natural cubic spline in decimal time (11 df)",
    "D": "C + CO2",
    "P": "base + integer year + smooth CO2 placebo (training-only fit on time)",
    "P2": "base + integer year + continuous decimal time (pre-declared placebo sensitivity)",
}
CONTRASTS = [  # (name, without, with): gain = MAPE(without) - MAPE(with)
    ("D-C: CO2 beyond flexible calendar", "C", "D"),
    ("B-A: CO2 beyond integer year", "A", "B"),
    ("B-P: CO2 vs smooth placebo", "P", "B"),
    ("B-P2: CO2 vs decimal time", "P2", "B"),
    ("P-A: placebo beyond integer year", "A", "P"),
    ("P2-A: decimal time beyond integer year", "A", "P2"),
    ("N1-N0: CO2 as the only clock", "N0", "N1"),
    ("A-N0: integer year alone", "N0", "A"),
]
HGB = dict(learning_rate=0.05, max_iter=2000, max_leaf_nodes=63, min_samples_leaf=200,
           l2_regularization=1.0, max_features=0.8, early_stopping=True, n_iter_no_change=50,
           scoring="loss")
RIDGE_ALPHA = 1e-3
SPLINE_KNOTS = np.arange(0.0, 22.0 + 1e-9, 2.0)  # decimal years since 2004-01-01: 12 knots, 11 df


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    with os.fdopen(fd, "w") as f:
        json.dump(value, f, indent=1, sort_keys=True, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def decimal_years(dates):
    """Years since 2004-01-01 as float64 (continuous time, not the integer year feature)."""
    return (dates.astype("datetime64[s]").astype(np.int64) - np.datetime64("2004-01-01", "s").astype(np.int64)) / (365.25 * 86400)


def natural_spline(t, knots=SPLINE_KNOTS):
    """Natural cubic spline basis without intercept (ESL eq. 5.4-5.5): len(knots)-1 columns."""
    k = len(knots)
    cube = lambda j: np.maximum(t - knots[j], 0) ** 3
    d = lambda j: (cube(j) - cube(k - 1)) / (knots[k - 1] - knots[j])
    last = d(k - 2)
    return np.column_stack([t] + [d(j) - last for j in range(k - 2)])


def prep(out):
    meta = json.loads((SOURCE / "metadata.json").read_text())
    if meta["arrays"]["jb08_static"]["features"][:5] != STATIC:
        raise ValueError("unexpected jb08_static column order")
    load = lambda name: np.load(SOURCE / (name + ".npy"), mmap_mode="r")
    splits = {"train": np.load(CONFIRM / "train_indices.npy"),
              "validation": np.load(CONFIRM / "validation_indices.npy"),
              "test": np.load(SOURCE / "test_indices.npy")}
    smoke = int(os.environ.get("CHEAP_SMOKE_STRIDE", "0"))
    if smoke:  # strided subsample for pipeline tests only; never used for results
        splits = {k: v[::smoke] for k, v in splits.items()}
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    record = {"source_metadata_sha256": sha(SOURCE / "metadata.json"), "splits": {}, "smoke_stride": smoke}
    for split, idx in splits.items():
        started = time.time()
        obs = np.asarray(load("observed_density")[idx], np.float64)
        jb = np.asarray(load("jb08_density")[idx], np.float64)
        if not (np.isfinite(obs).all() and np.isfinite(jb).all() and (obs > 0).all() and (jb > 0).all()):
            raise ValueError("invalid observed/JB08 density in " + split)
        dates = np.asarray(load("dates")[idx])
        arrays = dict(static=np.asarray(load("jb08_static")[idx], np.float32),
                      t=decimal_years(dates), r=np.log(obs) - np.log(jb), obs=obs, jb08=jb,
                      altitude=np.asarray(load("altitude_km")[idx], np.float32),
                      ap=np.asarray(load("ap_daily")[idx], np.float32),
                      day=dates.astype("datetime64[D]"))
        if not np.isfinite(arrays["static"]).all():
            raise ValueError("non-finite static features in " + split)
        np.savez(cache / (split + ".npz"), **arrays)
        record["splits"][split] = {"rows": int(len(idx)), "indices_sha256": hashlib.sha256(idx.tobytes()).hexdigest(),
                                   "seconds": time.time() - started}
        print(json.dumps({"event": "prep", "split": split, **record["splits"][split]}), flush=True)
    # Placebo: training-only least squares of each scaled CO2 column on smooth functions of time.
    train = np.load(cache / "train.npz")
    coef = np.linalg.lstsq(placebo_design(train["t"]), train["static"][:, CO2].astype(np.float64), rcond=None)[0]
    record["placebo_coefficients"] = coef.tolist()
    record["placebo_design"] = "[1, t, t^2, sin(2 pi t), cos(2 pi t)], t = decimal years since 2004-01-01"
    atomic_json(out / "prep.json", record)


def placebo_design(t):
    return np.column_stack([np.ones_like(t), t, t * t, np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)])


def features(cell, data, record):
    s, t = data["static"], data["t"]
    cols = [s[:, BASE]]
    if cell in ("A", "B", "P", "P2"):
        cols.append(s[:, [YEAR]])
    if cell in ("C", "D"):
        cols.append(natural_spline(t))
    if cell == "P":
        cols.append(placebo_design(t) @ np.asarray(record["placebo_coefficients"]))
    if cell == "P2":
        cols.append(t[:, None])
    if cell in ("N1", "B", "D"):
        cols.append(s[:, CO2])
    return np.column_stack(cols).astype(np.float64)


def ridge_terms(X, s):
    inter = np.column_stack([s[:, ALT] * s[:, j] for j in (F10, F10B, DTC)])
    return np.column_stack([X, inter])


def category_masks(altitude, ap):
    masks = [np.ones(len(altitude), bool)]
    masks += [(altitude >= a) & (altitude < a + 50) for a in range(200, 550, 50)]
    masks += [(ap >= 0) & (ap < 15), (ap >= 15) & (ap < 30), (ap >= 30) & (ap < 50), ap >= 50]
    return masks


def daily_sums(data, r_hat):
    pred = data["jb08"] * np.exp(r_hat)
    obs = data["obs"]
    if not (np.isfinite(pred).all() and (pred > 0).all()):
        raise ValueError("invalid predicted density")
    days, day_ix = np.unique(data["day"], return_inverse=True)
    per_row = {"ape": np.abs(pred - obs) / obs, "signed": (pred - obs) / obs, "sqlog": (r_hat - data["r"]) ** 2}
    sums = {k: np.zeros((len(CATEGORIES), len(days))) for k in per_row}
    counts = np.zeros((len(CATEGORIES), len(days)), np.int64)
    for c, mask in enumerate(category_masks(data["altitude"], data["ap"])):
        counts[c] = np.bincount(day_ix[mask], minlength=len(days))
        for k, v in per_row.items():
            sums[k][c] = np.bincount(day_ix[mask], weights=v[mask], minlength=len(days))
    return [str(d) for d in days], sums, counts


def fit(out, cell, model, seed, threads):
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor
    record = json.loads((out / "prep.json").read_text())
    data = {s: dict(np.load(out / "cache" / (s + ".npz"))) for s in ("train", "validation", "test")}
    X = {s: features(cell, d, record) for s, d in data.items()}
    started = time.time()
    info = {"cell": cell, "description": CELLS[cell], "model": model, "seed": seed, "n_features": int(X["train"].shape[1])}
    if model == "ridge":
        if seed != 0:
            raise ValueError("ridge is deterministic; seed must be 0")
        Z = {s: ridge_terms(X[s], data[s]["static"]) for s in X}
        mu, sd = Z["train"].mean(0), Z["train"].std(0)
        sd[sd == 0] = 1
        est = Ridge(alpha=RIDGE_ALPHA).fit((Z["train"] - mu) / sd, data["train"]["r"])
        predict = lambda s: est.predict((Z[s] - mu) / sd)
        info["ridge_alpha"] = RIDGE_ALPHA
    elif model == "hgb":
        est = HistGradientBoostingRegressor(random_state=seed, **HGB)
        est.fit(X["train"], data["train"]["r"], X_val=X["validation"], y_val=data["validation"]["r"])
        predict = lambda s: est.predict(X[s])
        info.update(hgb_params=HGB, n_iter=int(est.n_iter_))
    else:
        raise ValueError("model must be ridge or hgb")
    info["fit_seconds"] = time.time() - started
    for split in ("train", "validation"):
        info[split + "_log_mse"] = float(np.mean((predict(split) - data[split]["r"]) ** 2))
    days, sums, counts = daily_sums(data["test"], predict("test"))
    info.update(days=days, counts=counts.tolist(), **{k + "_sums": v.tolist() for k, v in sums.items()},
                threads=threads, total_seconds=time.time() - started, prep_sha256=sha(out / "prep.json"))
    atomic_json(out / "fits" / "{}__{}__seed_{}.json".format(cell, model, seed), info)
    print(json.dumps({k: info[k] for k in ("cell", "model", "seed", "n_features", "fit_seconds", "validation_log_mse")}), flush=True)


def runs():
    return [(c, "ridge", 0) for c in CELLS] + [(c, "hgb", s) for c in CELLS for s in range(3)]


def report(out, b=2000, rng_seed=20261005, min_clusters=4, min_valid_fraction=0.95):
    if json.loads((out / "prep.json").read_text()).get("smoke_stride", 0):
        raise ValueError("refusing to report results from a smoke-test cache")
    fits = {}
    for cell, model, seed in runs():
        fits[cell, model, seed] = json.loads((out / "fits" / "{}__{}__seed_{}.json".format(cell, model, seed)).read_text())
    ref = fits["N0", "ridge", 0]
    days, counts = ref["days"], np.asarray(ref["counts"])
    for f in fits.values():
        if f["days"] != days or f["counts"] != ref["counts"] or f["prep_sha256"] != ref["prep_sha256"]:
            raise ValueError("fits do not share identical test rows")
    months = sorted({d[:7] for d in days})
    month_of_day = np.array([months.index(d[:7]) for d in days])
    to_month = lambda a: np.stack([np.bincount(month_of_day, weights=row, minlength=len(months)) for row in a])
    mc = to_month(counts)
    periods = {"all_months": months, "original_period": [m for m in months if m in ORIGINAL_MONTHS],
               "added_years": [m for m in months if m not in ORIGINAL_MONTHS]}
    if periods["original_period"] != ORIGINAL_MONTHS or len(months) != 21:
        raise ValueError("unexpected held-out months")
    seeds = {"ridge": [0], "hgb": [0, 1, 2]}
    ape = {k: to_month(np.asarray(f["ape_sums"])) for k, f in fits.items()}
    result = {"cells": CELLS, "contrasts": [], "cell_metrics": [],
              "bootstrap": {"unit": "held-out calendar year-month, resampled within each period", "replicates": b,
                            "rng_seed": rng_seed, "min_contributing_clusters": min_clusters,
                            "min_valid_fraction": min_valid_fraction}}
    for period, pm in periods.items():
        cols = [months.index(m) for m in pm]
        weights = np.random.default_rng(rng_seed).multinomial(len(cols), [1 / len(cols)] * len(cols), size=b)
        for model, ss in seeds.items():
            for cell in CELLS:
                f = [fits[cell, model, s] for s in ss]
                n = mc[0, cols].sum()
                result["cell_metrics"].append({
                    "period": period, "model": model, "cell": cell, "rows": int(n),
                    "mape_percent_mean": float(np.mean([100 * ape[cell, model, s][0, cols].sum() / n for s in ss])),
                    "signed_bias_percent_mean": float(np.mean([100 * to_month(np.asarray(x["signed_sums"]))[0, cols].sum() / n for x in f])),
                    "test_log_mse_mean": float(np.mean([to_month(np.asarray(x["sqlog_sums"]))[0, cols].sum() / n for x in f])),
                    "validation_log_mse_mean": float(np.mean([x["validation_log_mse"] for x in f])),
                    "hgb_iterations": [x.get("n_iter") for x in f]})
            for name, without, with_ in CONTRASTS:
                for c, cat in enumerate(CATEGORIES):
                    cnt = mc[c, cols]
                    n = cnt.sum()
                    if not n:
                        result["contrasts"].append({"period": period, "model": model, "contrast": name, "category": cat, "rows": 0})
                        continue
                    sw = np.stack([ape[without, model, s][c, cols] for s in ss])
                    sv = np.stack([ape[with_, model, s][c, cols] for s in ss])
                    gains = 100 * (sw.sum(1) - sv.sum(1)) / n
                    rep_n = weights @ cnt
                    valid = rep_n > 0
                    ci = None
                    if np.count_nonzero(cnt) >= min_clusters and valid.mean() >= min_valid_fraction:
                        draws = (100 * (weights[valid] @ (sw - sv).T) / rep_n[valid, None]).mean(1)
                        ci = list(map(float, np.percentile(draws, [2.5, 97.5])))
                    result["contrasts"].append({
                        "period": period, "model": model, "contrast": name, "category": cat, "rows": int(n),
                        "contributing_clusters": int(np.count_nonzero(cnt)),
                        "gain_pp_mean": float(gains.mean()), "per_seed_gain_pp": gains.tolist(),
                        "seed_sd": float(gains.std(ddof=1)) if len(ss) > 1 else None,
                        "pointwise_95_percentile_interval": ci})
    result["interpretation"] = ("Exploratory: these test months were already scored by the neural campaign. "
                                "Gains are predictive relative to the stated calendar basis, not causal. "
                                "Month intervals hold fitted models fixed; seed SD (hgb) is reported separately. "
                                "added_years has 5 clusters and is descriptive only.")
    atomic_json(out / "report.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prep", "fit", "report"))
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--task", type=int, help="fit: 1-based index into runs()")
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    if a.action == "prep":
        prep(a.out)
    elif a.action == "fit":
        cell, model, seed = runs()[a.task - 1]
        fit(a.out, cell, model, seed, a.threads)
    else:
        report(a.out)


if __name__ == "__main__":
    sys.exit(main())
