# Cheap-control study design (proposal, not run) — 2026-10-10

Status: **approved 2026-10-10** ("as complete as possible") and launched the
same day (SGE jobs 1574–1576, root `karman-phoenix/cheap-controls-20261010/`).
Results go in a separate receipt; this design is not edited after results.
Two changes were fixed before any fit, at approval:

- the decimal-time placebo became its own cell, **P2**, rather than an optional sensitivity;
- gradient boosting uses `max_features=0.8`. With an external validation set
  and no subsampling, scikit-learn's boosting is almost deterministic, so the
  three seeds would otherwise be near-copies. It implements
step 3 of the [protocol](../co2-paper-protocol.md) ("cheap calendar controls
before neural retraining") and the A–D table of the
[triangulation plan](co2-density-triangulation-plan-2026-10-05.md#the-next-necessary-comparison-is-not-another-architecture).

## Why

The through-2025 Karman campaign adds CO₂ to a model that already receives
`all__year__[y]`. That feature is an **integer step** (22 values, 2004–2025),
which correlates 0.997 with the trailing-365-day CO₂ mean. After a quadratic in
year, the CO₂ residual SD is about 5.5% of its total SD. CO₂ therefore mostly
adds a **continuous within-year clock** and a slightly different trend shape. A
flexible learner can partly rebuild that clock from year plus day-of-year;
a linear one cannot. The question is whether any CO₂ gain survives once the
model gets an equally smooth time coordinate that contains no CO₂ measurement.

## Data, rows and target (identical across cells)

- Source: `karman-phoenix/data/prepared-full25` (read-only).
- Training rows: the campaign's confirmation subset
  `bounded-20261006-full25/data/confirm/train_indices.npy` (5,000,000 rows).
  Validation: its 1,000,000-row subset, used **only** for gradient-boosting
  early stopping.
- Test rows: all 12,395,677 canonical test rows (21 held-out months). Every
  result is also reported for the 16 original-period months and the 5 added-year
  months separately.
- Target: JB08 log residual `r = ln(observed) − ln(JB08)`. Prediction
  `JB08·exp(r̂)` gives MAPE on the same scale as the neural campaign.
- Base features (all cells): `jb08_static` without year and CO₂. That covers
  geometry (altitude, latitude, longitude and local solar time sin/cos), season
  and time of day (day-of-year and seconds-in-day sin/cos), Sun and sidereal
  angles, SET F10/S10/M10/Y10 (observed and 81-day), and JB08 DTC. OMNI history
  is excluded so the comparison stays cheap; this is stated as a limitation.

## Cells

| Cell | Time coordinate | CO₂ | Role |
|---|---|---|---|
| N0 | none | no | floor |
| N1 | none | yes | CO₂ as the only clock |
| A | integer year (as in Karman) | no | neural no-CO₂ analogue |
| B | integer year | yes | neural with-CO₂ analogue |
| C | flexible calendar: natural cubic spline in continuous decimal time, **fixed 11 df** (knots every 2 years, set now) | no | protocol C |
| D | C + CO₂ | yes | protocol D |
| P | integer year + **placebo** | no (placebo) | is CO₂ better than a smooth trend? |
| P2 | integer year + continuous decimal time | no | placebo sensitivity: plain clock |

**Placebo (fixed now):** two columns, one per CO₂ feature. Each is a
training-rows-only least-squares fit of that CO₂ feature on a quadratic in
continuous decimal time plus one annual harmonic. The placebo is a deterministic
function of the date: it has CO₂'s smooth shape and seasonal phase but none of
its measured inter-annual variability. P2 replaces it with plain decimal time.

CO₂ features are the campaign's two: previous-day observed and the
trailing-365-day mean.

## Models (fixed hyperparameters, set before fitting)

1. **Linear:** ridge (α = 1e-3 on standardized features) on the base features,
   plus altitude × {F10, F10B, DTC} interactions, plus the cell's time/CO₂
   columns. Deterministic.
2. **Gradient boosting:** scikit-learn `HistGradientBoostingRegressor`,
   learning rate 0.05, max 2,000 iterations, 63 leaves, min 200 samples per
   leaf, L2 1.0, 80% of features per split, early stopping on the validation
   subset (patience 50). Three seeds, to separate seed noise from cell differences.

That gives 8 cells × (1 + 3) = 32 fast fits. No neural fits.

## Estimands and uncertainty

`gain = MAPE_without − MAPE_with` in percentage points, on identical rows:

- **Primary:** D − C (CO₂ beyond a flexible calendar; predictive, not causal).
- B − A (mirrors the neural ±CO₂ contrast).
- B − P (measured CO₂ against a smooth placebo trend, both with year); B − P2.
- P − A and P2 − A (how much a smooth clock alone adds to integer year).
- N1 − N0 (how much a clock alone is worth).

Report both log-MSE and MAPE, signed bias, and the paired-month bootstrap used
for the neural campaign (2,000 replicates, seed 20261005, held-out month
clusters, resampled within each period). The added-year period has only 5
clusters, so it is descriptive only.

## Interpretation rules (fixed before results)

- If |D − C| is within its interval of 0, **or** B − P is within its interval of
  0: no CO₂-specific predictive information beyond smooth time in this
  setting. No neural follow-up.
- Neural follow-up (≤8 pilot fits, one architecture, separately approved) is
  proposed only if, in the original period, D − C > 0 **and** B − P > 0 with
  both intervals excluding 0, **and** the linear and boosted models agree in sign.
- The test months were already scored for the neural campaign, so the test set
  is not fresh. The study is exploratory, and its claims are limited to
  "relative to this calendar basis".

## Execution (after approval)

One SGE script on `all.q@auriga` (8 slots, ~6 GB RAM for the gathered float32
matrices), outputs in a new root under `karman-phoenix/`, receipts in
`docs/research/`. Expected wall time: under 1 h.
