# WACCM-X CO₂ transport time with tigramite — 2026-10-11

The user asked to try tigramite to measure how long CO₂ takes to reach the
thermosphere. Same model data and caveats as the
[tracer-age receipt](waccm-x-co2-transport-lag-2026-10-11.md): one free-running
WACCM-X v2.1 realisation, regional 25-cell mean, settled windows only.

## Setup (`scripts/waccmx/co2_transport_tigramite.py`, SGE 1586)

- **Signal:** 12-month log growth of CO₂, `ln c(t) − ln c(t − 12)`, linearly
  detrended and standardised per dataset. This removes the seasonal cycle and
  the long-term trend, leaving interannual growth swings. The prescribed
  surface CO₂ has such swings, for example the early-1990s slowdown.
- **Datasets:** run 2 (1982–1997) and run 3 (2006–2021), 192 months each,
  passed separately (tigramite `analysis_mode="multiple"`).
- **Variables:** surface; CO₂ at 20, 40, 60, 70, 80, 90, 100, 110, 120 and 140 km;
  the 12-month change of F10.7a. Surface CO₂ and F10.7a are prescribed
  forcings, so no link may point into them.
- **Lagged dependence:** tigramite ParCorr of each height on the surface, lags
  0–96 months. Every lag uses identical samples (`cut_off="max_lag_or_tau_max"`;
  PCMCI's default 2 × tau_max cut would leave no data at 96 months). It is run
  raw and after removing the contemporaneous F10.7a term.
- **Calibrated significance:** 1,000 phase-randomised surrogates of the surface
  series. They keep its spectrum and autocorrelation but scramble its timing.
  The statistic is the maximum r over all lags, which accounts for both
  autocorrelation and the lag search.
- **PCMCI+:** ParCorr, tau_max 36 months, pc_alpha 0.01.
- **Synthetic check:** the real surface signal, delayed by known amounts and
  with noise added, is recovered exactly for pure delays (12 → 11, 36 → 36 and
  50 → 50 months). For a gamma age spectrum, the peak falls between the mode and
  the mean: mean 36 / mode 18 gives 30; mean 50 / mode 25 gives 34. The peak
  lag therefore **underestimates the mean age**.

## Results

| Height | Peak lag, raw (months) | Raw p (surrogate) | Peak lag, solar removed | Solar-removed p (surrogate) |
|---|---|---|---|---|
| 20 km | 11 (r 0.47) | 0.058 | 13 | 0.086 |
| 40 km | 39 (r 0.36) | 0.20 | 41 | 0.18 |
| 60 km | 40 (r 0.19) | 0.67 | 37 | 0.64 |
| 70–90 km | 69–96 (r ≤ 0.15) | 0.39–0.88 | — | 0.72–0.86 |
| 100–140 km | 68–70 (r 0.10–0.18) | 0.37–0.83 | none (peak at 0, r ≤ 0.05) | 0.95–1.00 |

Figure: `data/derived/waccmx_co2_tigramite_v1/lag_correlation.png` (local only).

- **No lag is significant at 5% once autocorrelation and the lag search are
  accounted for.** The nominal analytic p-values (down to 10⁻¹²) are
  misleading. Two 16-year windows hold only a handful of independent growth
  swings, and random-timing surrogates reach r ≈ 0.4–0.5 at 20–40 km by chance.
- 20 km (~1 year) and 40 km (~3.3 years) peak where the tracer ages put them
  (1.1 and 3.0 years), so they are consistent but not independently established.
- **The apparent ~5.7-year thermospheric lag is a solar artifact.** In these
  windows, surface CO₂ growth correlates +0.47 with the F10.7a change (a
  coincidence of timing), and thermospheric CO₂ growth is driven directly by
  solar activity (r = −0.44 at 110 km, lag 0). Removing the solar term removes
  the lag entirely, and 5.7 years is about half the solar cycle.
- **PCMCI+** recovers a plausible lower chain: surface → 20 km (lag 0),
  20 → 40 km (12 months), 20 → 60 km (26 months). Above 70 km it finds only
  same-month coupling between thermospheric levels.

  It also drops the F10.7a → 110 km link, which is unconditionally strong. In
  the pruning step, p = 0.034 is above pc_alpha 0.01 once the series' own past
  is conditioned on. Smooth 12-month growth series give PCMCI+ too little
  independent information to time multi-year lags, so its graph is not used to
  answer the question.

## Answer to "how long does CO₂ take to reach the thermosphere?"

- **Tigramite cannot measure it in this dataset.** The surface signal is too
  weak and smooth over the settled windows. In the thermosphere it is
  swamped by solar variability.
- The best available estimate remains the tracer age (previous receipt):
  - about 1.1 years to 20 km;
  - about 3 years to 40–60 km;
  - about 4.1–4.2 years to 70 km.
- Above about 75 km, photolysis and diffusive separation deplete CO₂, so its
  age cannot be read from CO₂ alone. Reaching the thermosphere then takes about
  as long as reaching the upper mesosphere: roughly **4–5 years**. Vertical
  diffusion above the turbopause is much faster than the years of mesospheric
  transport, but that last step is physical reasoning, not a measurement here.
- The long-term increase arrives attenuated: about 55–75% of the surface growth
  rate at 100–140 km after a solar adjustment.

## What could measure the upper part

- An inert, non-depleted tracer: the regional files carry no age-of-air or
  SF₆-like tracer. CO with known chemistry would need extra modelling.
- A longer settled record without restarts. Run 3 from 2006 has 16 usable
  years; delivering 2022 would add one.
- A controlled step-change CO₂ experiment, the cleanest measure of the
  thermospheric response time.
