# Cheap-control study results — 2026-10-10

Results for the [cheap-control design](cheap-control-study-design-2026-10-10.md),
approved and run on 10 Oct. SGE jobs 1574 (prep), 1575 (32-task fit array) and
1576 (report), all exit 0, from 15:50 to 16:21 CEST on `all.q@auriga`. Root:
`karman-phoenix/cheap-controls-20261010/` (`tools/cheap_controls.py`,
`outputs/report.json`). The test months had already been scored by the neural
campaign, so this study is **exploratory**.

## Setup as run

- Identical rows in every cell:
  - full25 confirmation training subset (5,000,000 rows);
  - its validation subset (1,000,000 rows), used only for boosting early stopping;
  - all 12,395,677 canonical test rows.

  Target: JB08 log residual. MAPE is computed on `JB08·exp(r̂)`.
- Ridge: deterministic. Boosting: seeds 0–2, stopping after 295–844 iterations.
- Placebo P: each scaled CO₂ column fitted on training rows to
  `[1, t, t², sin 2πt, cos 2πt]` in decimal years. P2: plain decimal time.
- A strided smoke test of the full pipeline ran first, and its cache is barred
  from reporting.

## Cell metrics (all 21 months; boosting = three-seed mean)

| Cell | Ridge MAPE % | Ridge bias % | Boosted MAPE % | Boosted bias % | Boosted val. log-MSE |
|---|---|---|---|---|---|
| N0 (no clock) | 18.44 | +3.06 | 15.33 | +2.20 | 0.0442 |
| N1 (+CO₂) | 18.48 | +3.14 | 15.27 | +2.85 | 0.0422 |
| A (+integer year) | 18.49 | +3.11 | 15.18 | +2.36 | 0.0423 |
| B (A+CO₂) | 18.26 | +3.32 | 15.22 | +2.81 | 0.0420 |
| C (+11-df time spline) | **17.99** | +2.70 | 15.20 | +2.92 | 0.0419 |
| D (C+CO₂) | 18.05 | +2.71 | 15.23 | +2.84 | 0.0418 |
| P (A+smooth placebo) | 18.23 | +3.17 | **15.15** | +2.52 | 0.0421 |
| P2 (A+decimal time) | 18.49 | +3.09 | 15.26 | +2.83 | 0.0423 |

These models have no OMNI history. The full25 neural OMNI models reach about
12.7% MAPE on the same rows, so the cheap models are a much weaker but matched
comparator.

## Contrasts (gain = MAPE without − with, pp; month-cluster 95% interval)

| Contrast | Period | Ridge | Boosted (per-seed) |
|---|---|---|---|
| **D − C** (primary) | all | −0.057 [−0.142, +0.036] | −0.025 [−0.159, +0.074] (+0.05, −0.03, −0.09) |
| | original 16 | −0.084 [−0.175, +0.011] | −0.054 [−0.221, +0.057] |
| | added 5 | +0.058 [−0.160, +0.258] | +0.103 [+0.044, +0.193] (+0.42, −0.15, +0.03) |
| B − A | all | +0.235 [−0.119, +0.556] | −0.035 [−0.254, +0.124] |
| B − P | all | −0.026 [−0.218, +0.151] | −0.068 [−0.286, +0.113] |
| | added 5 | −0.079 [−0.435, +0.293] | −0.349 [−0.924, +0.111] |
| B − P2 | all | +0.235 [−0.129, +0.560] | +0.039 [−0.127, +0.165] |
| P − A | all | +0.261 [−0.179, +0.681] | +0.033 [−0.105, +0.194] |
| P2 − A | all | −0.000 [−0.015, +0.014] | −0.074 [−0.171, +0.024] |
| N1 − N0 | all | −0.036 [−0.349, +0.278] | +0.063 [−0.343, +0.544] |
| A − N0 | all | −0.048 [−0.357, +0.246] | +0.146 [−0.183, +0.524] |

## Reading, against the pre-set rules

- **The stopping rule is met.** D − C is within its interval of 0 for both
  models in the original period and over all months, and B − P is ≈ 0 for
  both. In this setting, measured CO₂ adds no predictive information beyond a
  smooth time coordinate. **No neural follow-up is proposed.**
- For ridge, CO₂ (B − A, +0.24) and the smooth placebo (P − A, +0.26) help by
  the same amount. Plain decimal time adds nothing (P2 − A = 0), because a
  linear model already gets a linear trend from integer year. The 11-df spline
  helps most (C is the best ridge cell). Whatever CO₂ adds here is **trend
  curvature**, which a time-only fit reproduces.
- For boosting, every cell lies within about 0.2 pp. Integer year alone is
  worth about +0.15 pp, and nothing adds reliably on top of it. This matches
  the seminar observation that year is already a near-perfect secular proxy.
- Added years (5 clusters, 400–550 km only, descriptive): the boosted D − C
  interval excludes 0, but it rests on one seed (+0.42; the others are −0.15
  and +0.03). The smooth placebo beats measured CO₂ there (B − P = −0.35).
- Strata: in the original period, 7 of 72 stratum intervals exclude 0 (D − C,
  B − A, B − P and B − P2 × two models × nine strata with intervals; about 3.6
  expected by chance at 5%, uncorrected). They are small and mixed in sign:
  ridge D − C is negative at 350–400 and 500–550 km, and ridge B − A is
  positive at 350–400 km. The one boosted D − C hit (250–300 km) includes a
  −0.40 pp seed. No consistent altitude or activity pattern appears, so they
  are not interpreted.
- Predictive and relative to these calendar bases only. A null here does not
  refute a slow physical CO₂ effect carried by the trend itself (protocol gate 4).

## Housekeeping

The float32 row cache (`outputs/cache/`, 2.8 GB on shared `/home`) can be
deleted; `prep` regenerates it in about a minute.
