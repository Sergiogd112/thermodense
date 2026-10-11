# CO₂–density paper: results summary — 2026-10-11

Consolidates the evidence gathered under the [protocol](../co2-paper-protocol.md)
up to this date. Each item links its dated receipt; the numbers are copied from
those receipts and their outputs, not recomputed. This is a working summary for
drafting, not manuscript text.

## The question and where each stream stands

**Does CO₂ add density information beyond secular time proxies, and are the
observed density and cooling patterns compatible with model-conditioned CO₂
responses?**

| Stream | Status | One-line answer |
|---|---|---|
| Prediction (Karman ± CO₂, cheap controls) | Closed | No reliable CO₂ gain detected beyond a smooth time coordinate |
| Observed trends | Done, one input pending | About −4 to −7 %/decade at 250–575 km, solar-adjusted |
| Model physics (WACCM-X) | Partial | Historical total trend matches observations; CO₂-only share unknown |
| Without CO₂ | First pass done | Removing the published WACCM-X CO₂ response leaves no detectable trend above ~275 km; a small decline remains at 200–250 km |

## 1. Prediction: no CO₂-specific information

- **Original Karman study** (16 held-out months): paired gain +0.072 pp
  [−0.108, +0.271]. ([6 Oct receipt](co2-paper-execution-2026-10-06.md))
- **Through-2025 campaign** ([closeout](full25-closeout-2026-10-10.md)), as
  five-seed means. The month intervals hold the trained models fixed:
  - paper_jb08_omni: +0.122 [+0.028, +0.209];
  - bounded_omni_depth_01: −0.037 [−0.214, +0.144];
  - residual_jb08_depth_04: −0.180 [−0.290, −0.065].

  Seed variation is as large as the month intervals, single seeds drive the
  nominally significant results, and the signs disagree across architectures.
  This is not a reliable gain.
- **Cheap controls** ([results](cheap-control-results-2026-10-10.md)): CO₂
  beyond a flexible time spline (D − C) is −0.057 [−0.142, +0.036] for ridge and
  −0.025 [−0.159, +0.074] for boosting. Measured CO₂ against a smooth
  time-only placebo (B − P) is ≈ 0 for both. The pre-set rule stops any neural
  follow-up.
- **Claim:** within these models and splits, no reliable incremental gain from
  CO₂ was detected beyond calendar time; the intervals still allow small gains
  (up to about +0.15 pp), so this is a null result, not demonstrated absence. Karman's integer-year input already correlates 0.997
  with the CO₂ trend. This says nothing for or against a physical CO₂ effect.

## 2. Observed density trends (solar-adjusted, calendar time)

Estimator: OLS `log10 ρ ~ time + F10.7₈₁ + F10.7₈₁²`, 27-day HAC 95% intervals,
from `scripts/recreate_brown_density_trend_current.py`. Observed and baseline
rows come from `outputs/figures/results/current_density_trends/current_density_trends_by_dataset_altitude.csv`
(19 Aug 2026 run; SHA-256 `71111bb2…5b35f65`, local because its inputs are
partly access-conditional). WACCM-X rows come from `scripts/waccmx_density_trends.py`
(PR #68), which writes `data/derived/waccmx_density_trends_v1/waccmx_density_trends.csv`.
TU Delft's single 450 km estimate is the pooled mission-offset fit in the same
19 Aug table. Values at 400 km (WACCM-X at 398.5 km geometric; TU Delft at 450 km):

| Dataset | Period | %/decade [95%] | 250–575 km range |
|---|---|---|---|
| Orbit-derived global mean | 1967–2019 | −4.94 [−5.99, −3.88] | −6.2 to −4.1 |
| HASDM (Mauna Loa subset) | 2000–2025 | −5.76 [−8.33, −3.11] | −6.8 to −4.9 |
| TU Delft (450 km) | 2000–2025 | −3.16 [−14.59, +9.81] | short, wide |
| NRLMSISE-00 / 2.0 / 2.1 baselines | 1967–2026 | ≈ −1.4 [−2.1, −0.7] | −1.8 to −0.7 |
| JB2006 baseline | 1997–2026 | −1.02 [−2.90, +0.89] | −1.3 to −0.4 |
| JB2008 baseline | 1997–2026 | −9.77 [−12.77, −6.67] | −13.2 to −4.7 |

- The observational products (global mean, HASDM) sit at the negative end of
  the Brown et al. (2024) synthesis at 400 km.
- The empirical baselines are not observations. NRLMSIS has no explicit secular
  term, so its ≈ −1.4 %/decade reflects its drivers and sampling. JB2008's
  strong negative trend is not explained here; its solar-index inputs should be
  checked before it is interpreted.
- These are adjusted calendar-time associations, not CO₂ attributions.
- Figure: overlay of the current trends on the Brown synthesis in gray, with
  WACCM-X (PR #68, `density_trend_figure6_overlay`). The global-mean input now
  rebuilds on Phoenix and reproduces the 19 Aug trends exactly. The TU Delft
  input still needs the analysed per-mission files, which are blocked on storage.

## 3. Model physics: WACCM-X v2.1 historical, regional 16–25°N, 200–210°E

Receipts: [download](waccm-x-regional-download-2026-10-10.md),
[CO₂ lag](waccm-x-co2-transport-lag-2026-10-11.md),
[tigramite](waccm-x-co2-transport-tigramite-2026-10-11.md).

- **Data:** 27,026 verified daily files covering 1950–2024. 2022 was not
  delivered (it was simulated), and 2023-01-31 and 2023-02-03 are missing.
- **Run structure:** three runs (1949, 1973 and 1997 bases). Run 3's 2023–24
  output is a later continuation of the same run.
- **Restart artifacts:** each run starts from year-2000 CO₂ rescaled to its start
  year. CO₂ tracer ages are negative at first and need about 6–8 years to
  settle; run 1 never fully does. At 40–70 km CO₂ jumps about 7–10 ppm at 1998.
  Usable windows are 1982–1997 and 2006–2024 (for interval-matched tracer-age
  estimates, 2006–mid-2021).
- **Transport times** (tracer age): about 1.1 years to 20 km, about 3.0 years to
  40–60 km and about 4.1–4.2 years to 70 km; runs 2 and 3 agree.
  - Above about 75 km, CO₂ is depleted, so its age cannot be read. No transit
    time to the thermosphere is measured. The ~4-year age at 70 km is a
    CO₂-equivalent, mean-like delay; it bounds neither first arrival nor the
    thermospheric response time.
  - A lagged tigramite (ParCorr/PCMCI+) analysis cannot time it. No lag
    survives a surrogate test, and an apparent ~5.7-year thermospheric lag is
    solar aliasing.
- **CO₂ in the cooling region:** at 100–140 km, CO₂ grows at about 55–75% of
  the surface rate after a solar adjustment. Lower-thermosphere CO₂ falls 5–9%
  per 100 sfu. Above about 200 km CO₂ is a trace gas that mainly tracks thermal
  expansion.
- **Model density trend** (same estimator as section 2, settled windows):
  - about −1 to −2 %/decade in the stratosphere and mesosphere;
  - about −3.5 to −5 at 100–200 km;
  - −5.9 (run 3) and −6.6 (run 2) %/decade near 400 km.

  This matches the observed global-mean and HASDM trends. It is the model's
  **total** historical trend, from all forcings together. Coverage limits it to
  ≤ 427 km.

## 4. Density trends without CO₂

[Receipt](co2-only-expectation-2026-10-11.md). The Brown et al. (2024) WACCM-X
v1.2.2 `CO2_only` response (fixed CO₂ and F10.7, Kp = 0, CC BY 4.0) was removed
from each series day by day, on its own days, CO₂ and F10.7. The series was then
refitted with the same estimator, which recomputes the 95% interval from the
new residuals. WACCM-X uses its own prescribed CO₂.

- **Above about 275 km, no net trend remains.**
  - Global mean, 275–575 km: without-CO₂ trends of −0.56 to +0.47 %/decade, all
    intervals containing zero. At 400 km: observed −4.94, without CO₂ +0.47
    [−0.60, +1.55].
  - HASDM, 225–825 km: the same, with wider intervals.
- **At the lowest levels a significant decline remains:** −0.89 [−1.55, −0.22]
  at 250 km (global mean) and −1.70 [−2.99, −0.39] at 200 km (HASDM).
- WACCM-X v2.1's own historical windows, compared with their own prescribed CO₂,
  leave residuals of about −2 (1982–1997; significant only at 206 km) and −0.7
  to +2.3 %/decade (2006–2024; none significant), which indicates window and
  model dependence.
- **Claim:** given this model's CO₂ response, the observed trends above about
  275 km are consistent with CO₂ alone (compatibility, not sufficiency). The
  intervals bound only the net residual and still allow sizeable values;
  individual other causes are neither excluded nor bounded. The interval omits uncertainty in the CO₂ response:
  a 30% error would shift the without-CO₂ trend by about ±1.5 %/decade at 400 km.

## 5. What is not established

- **Uncertainty in the CO₂ response.** Section 4 rests on one model's published
  grid, and its intervals cover observational noise only. Run-level controlled output (requested from the
  authors) or several models' CO₂ responses are needed to bound it.
- What drives the residual decline at 200–250 km.
- Why the empirical-model baselines trend at all (MSIS near −1.4, JB2008 near
  −10 %/decade at 400 km): driver drift, sampling or construction.

## 6. Open items

- Controlled-run reply (requested by the user). Then: rerun section 4 with
  run-level output to attach intervals and reach below 200 km.
- UKSSDC questions (2022, missing days, definitions, tracers, terms): draft
  prepared outside the repository.
- `/qnap` space to rebuild TU Delft inputs on Phoenix.
- Optional: SABER 100–139 km cooling vs WACCM-X; per-column trend spread.
- Seven open PRs (#62–#68) to review and merge.
