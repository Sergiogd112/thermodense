# The 200–250 km gap: bracketing the CO₂ response and SABER cooling — 2026-10-12

Follow-up to [density trends without CO₂](co2-only-expectation-2026-10-11.md),
which removed one model's CO₂ response (Brown et al. 2024, WACCM-X v1.2.2) and
found no net trend above ~275 km but a significant decline at 200–250 km
(−0.89 [−1.55, −0.22] %/decade at 250 km in the global mean; −1.70 [−2.99, −0.39]
at 200 km in HASDM). Two questions:

1. Would a different published CO₂ response close the gap?
2. Does SABER's lower-thermosphere CO₂ cooling trend differ from WACCM-X's?

Short answer:
- **In the global mean, no single strength of the CO₂ response fits both 250 and
  400 km.** The gap is in the altitude *shape* of the trend, not its size. HASDM
  suggests the same pattern but does not exclude a common scaling.
- Weaker published CO₂-only responses leave a decline at every height.
- SABER cannot discriminate. Its regional cooling-trend point estimate is close
  to the model's, but both intervals are broad.

## 1. Bracketing the CO₂ response (`scripts/co2_response_bracketing.py`)

Brown's response is scaled by a factor *k* at fixed altitude shape. The
without-CO₂ trend of `obs − k·s` is linear in *k*, and its HAC variance is
quadratic in *k*. Refits at k = 0, 1, 2 recover both exactly. Check: k = 1
reproduces the 11 Oct rows (−0.89 [−1.56, −0.22] at 250 km, +0.47 [−0.60,
+1.55] at 400 km; the third decimal is rounding).

### Published model responses relative to Brown 2024 at 400 km

Each 400-km trend in Brown et al. (2024, Table 2) is set against Brown's grid
under the same conditions. Two conversions are used.

- **Exact, from documented endpoints.** k is the published end/start density
  log-ratio divided by Brown's response to the same CO₂ change at the same F10.7.
  - Solomon 2015: prescribed 363 and 386 ppm. The 12-year endpoint change was
    scaled by 10/12 to a linear %/decade, so the endpoint ratio is
    1 + 1.2 × trend/100.
  - Solomon 2018 and 2019: Table 1, 330 and 375 ppm. At 400 km the density goes
    0.584 → 0.518 ng m⁻³ (70 sfu) and 7.23 → 6.87 ng m⁻³ (200 sfu).
  - Both extrapolate Brown's constant %/ppm below its 369-ppm floor, as for the
    observations before ~1999.
- **Approximate, as per-decade log-slopes** (Qian 2006, Cnossen 2020, Brown 2021).
  - Fixed F10.7: the grid at that F10.7 with the period's linear Mauna Loa CO₂
    growth.
  - All solar activity: the grid applied to the period's daily F10.7₈₁ and CO₂,
    fitted with the solar-adjusted estimator. That estimator differs from the
    studies' own (Qian used a ratio regression between varying- and fixed-CO₂
    runs).
  - Cnossen 2020: the forcing is the WACCM-X/CMIP6 prescribed `f107a` and
    `co2vmr` over the full 1950–2015. Observed F10.7 starts in 1957-10; every
    day is checked present.
  - The published "%/decade" is read as compounded. Reading it as a linear
    change over the period (`k_linear_reading`) raises k by ≤ 0.06 (Brown 2021:
    0.98; Cnossen 2020: 0.69).
  - These k values are normalisations, not matched estimands.
- **Unmatched conditions remain caveats:** the studies' geomagnetic and seasonal
  conditions (for example Solomon 2015's solar-case Kp, against Brown's Kp = 0)
  and model versions.

Forcing classes, as described in each study:
- **CO₂ only:** Qian 2006, Solomon 2015 and Brown 2021.
- **Anthropogenic:** Solomon 2018/2019 also changed CH₄, CFCs, ozone and the
  lower boundary (Codex/GPT-6.1 review of PR #70, citing the 2019 paper).
- **All forcings:** Cnossen 2020.

Only CO₂-only entries enter the shaded range in the figure.

A literature check of the open copies on 2026-10-12 found:
- **Solomon et al. 2018:** CO₂ 330→375 ppmv plus CH₄ and CFCs at perpetual solar
  minimum (Table 1: −3.9 %/decade at 400 km).
- **Qian et al. 2006:** −2.2 and −2.9 %/decade at 350 and 450 km at 70 sfu
  (average −2.55); −1.7 % is their all-activity regression.
- **Brown et al. 2021:** CO₂ only, with the magnetic field fixed (−5.8 %/decade,
  1975–2005).
- **Solomon et al. 2015:** only the abstract was reachable by the literature
  search. The round-2 Codex review read the full paper and confirmed the CO₂-only
  forcing and the 363/386-ppm endpoints.
- **Altitude shape:** no accessible paper reports values at 200–300 km. Other
  models' altitude shapes therefore cannot be compared here; they appear only in
  figures (for example Brown 2021, Fig. 5).

| Study (400 km) | Forcing | F10.7 | Period | Published %/dec | Conversion | *k* |
|---|---|---|---|---|---|---|
| Qian et al. 2006 (TIME-GCM 1D) | CO₂ | 70 | 1970–2000 | −2.5 | approx. | 0.41 |
| Qian et al. 2006 | CO₂ | all | 1970–2000 | −1.7 | approx. | 0.37 |
| Qian et al. 2006 | CO₂ | 210 | 1970–2000 | −0.75 | approx. | 0.31 |
| Solomon et al. 2015 (TIME-GCM), kq 1.5e-12 | CO₂ | 70 | 1996–2008 | −4.9 | endpoints | 0.64 |
| Solomon et al. 2015, kq 3.0e-12 | CO₂ | 70 | 1996–2008 | −6.8 | endpoints | 0.90 |
| Solomon et al. 2015, kq 1.5e-12 | CO₂ | 200 | 1996–2008 | −1.8 | endpoints | 0.59 |
| Solomon et al. 2015, kq 3.0e-12 | CO₂ | 200 | 1996–2008 | −2.1 | endpoints | 0.69 |
| Solomon et al. 2018 (WACCM-X) | anthropogenic | 70 | 1974–2003 | −3.9 | endpoints | 0.65 |
| Solomon et al. 2019 (WACCM-X) | anthropogenic | 200 | 1974–2003 | −1.7 | endpoints | 0.71 |
| Cnossen 2020 (WACCM-X 2.0) | all | all | 1950–2015 | −2.8 | approx. | 0.64 |
| Brown et al. 2021 (WACCM-X) | CO₂ | 70 | 1975–2005 | −5.8 | approx. | 0.92 |

**Brown 2024 is the strongest response in the compilation.** The other CO₂-only
responses are 0.31–0.92 times as strong at 400 km (up to 0.98 under the linear
reading). The anthropogenic and all-forcing runs fall at 0.64–0.71. Brown 2021,
the same model family, comes out at 0.92, which supports the conversion.
Table 2 values are global means at the studies' own conditions.

### Scaling the observations would need

The *k* that leaves no net trend (k\*), and the *k* range whose 95% residual
interval contains zero (`required_scaling.csv`):

| Series | 200 km | 250 km | 275 km | 325 km | 400 km | 575 km |
|---|---|---|---|---|---|---|
| Global mean, 1967–2019 | — | 1.27 [1.07, 1.47] | 1.15 [0.95, 1.35] | 1.01 [0.81, 1.20] | 0.92 [0.72, 1.11] | 1.06 [0.84, 1.28] |
| HASDM (Mauna Loa), 2000–2025 | 1.56 [1.13, 2.00] | 1.27 [0.83, 1.71] | 1.16 [0.73, 1.59] | 1.02 [0.59, 1.45] | 0.93 [0.50, 1.37] | 0.98 [0.49, 1.47] |

- Above ~300 km the observations want roughly Brown's response (k ≈ 0.9–1.06).
  In the global mean, only Brown 2021 (0.92) and Solomon 2015 with the doubled
  quenching rate at 70 sfu (0.90) fall inside [0.72, 1.11].
- The low altitudes want **more** than Brown's response (k ≥ 1.07 at 250 km in
  the global mean, ≥ 1.13 at 200 km in HASDM). That is stronger than every
  published response in the compilation, if their altitude shape is Brown's.
- The k sets are solved analytically from the quadratic inequality. Every set
  here is a bounded interval; an unbounded set would raise an error.
- Figure: `outputs/figures/results/co2_response_bracketing/co2_response_bracketing.{png,pdf}`.

### What the published responses leave behind (`without_co2_by_published_response.csv`)

| Response used | Global mean 250 km | Global mean 400 km | HASDM 200 km | HASDM 400 km |
|---|---|---|---|---|
| Brown 2024 (k = 1) | −0.89 [−1.56, −0.22] | +0.47 [−0.60, +1.55] | −1.70 [−2.99, −0.39] | +0.45 [−2.29, +3.27] |
| Brown 2021 (0.92) | −1.17 [−1.83, −0.50] | +0.00 [−1.07, +1.08] | −1.95 [−3.24, −0.65] | −0.09 [−2.82, +2.72] |
| Solomon 2015, kq 3e-12, 70 sfu (0.90) | −1.22 [−1.89, −0.55] | −0.09 [−1.16, +0.99] | −2.00 [−3.29, −0.70] | −0.19 [−2.92, +2.61] |
| Solomon 2018, anthropogenic (0.65) | −2.04 [−2.70, −1.37] | −1.47 [−2.53, −0.40] | −2.75 [−4.03, −1.45] | −1.78 [−4.47, +0.98] |
| Qian 2006, all (0.37) | −2.92 [−3.59, −2.26] | −2.95 [−4.01, −1.89] | −3.56 [−4.83, −2.27] | −3.48 [−6.12, −0.77] |

With any response weaker than about 0.72× Brown 2024, a significant decline
remains at 400 km in the global mean too. The "consistent with CO₂ alone above
275 km" result therefore holds only for the strongest modelled CO₂ responses
(Brown 2021 and 2024, and Solomon 2015 with the doubled quenching rate).
Weaker responses leave about 1.5–3 %/decade unexplained at every height.
This echoes the long-standing finding that models with the standard CO₂–O
quenching rate underpredict observed trends (Solomon et al. 2015).

### Altitude shape (`altitude_shape_test.csv`)

The trend of log10(ρ_low/ρ_400), the difference of two altitudes on common
days, against Brown's predicted change in that ratio:

| Series | Observed ratio trend | Brown 2024 predicts | Residual at k = 1 | k fitting the shape |
|---|---|---|---|---|
| Global mean 250/400 km | +0.85 [+0.40, +1.31] | +2.25 | −1.37 [−1.81, −0.93] | 0.38 [0.18, 0.58] |
| HASDM 200/400 km | +1.17 [−0.42, +2.79] | +3.39 | −2.14 [−3.71, −0.55] | 0.35 [−0.13, 0.83] |
| HASDM 250/400 km | +0.91 [−0.23, +2.05] | +2.45 | −1.51 [−2.63, −0.38] | 0.37 [−0.09, 0.84] |

- In the global mean, the *k* that fits the 250/400 shape (0.18–0.58) does not
  overlap the *k* that fits the 400-km magnitude (0.72–1.11). Under this model
  and estimator, no rescaling of Brown's profile fits both. **Between 250 and
  400 km, the modelled response steepens with altitude about 2.6 times faster
  than the observed trends.**
- HASDM points the same way, but its shape and magnitude ranges overlap
  (shape up to 0.83–0.84; magnitude from 0.50), so it does not exclude a common
  scaling.
- The ratio statistic still depends on *k*. What it adds is that it accounts for
  the covariance between the two altitudes on the same days.
- A response that falls off steeply below 300 km is what a cooler, smaller
  scale-height upper thermosphere produces. A more altitude-uniform decline
  points to a larger share of change at or below the base of the thermosphere.
  Candidates, which these data cannot separate:
  - stronger lower-thermosphere or mesospheric cooling than modelled;
  - other drivers acting on the lower boundary (ozone, other greenhouse gases,
    eddy diffusion, gravity waves);
  - errors in the products' low-altitude trends;
  - model differences (other models' altitude shapes are not available here).

## 2. SABER against WACCM-X cooling (`scripts/saber_waccmx_cooling_trends.py`)

Data:
- **SABER:** the official 100–139 km CO₂ cooling flux, from the verified
  regional extraction (13,902 scans, 16–25°N, 200–210°E).
- **WACCM-X:** a run-3 cooling proxy `∫ n_CO₂ n_O exp(−960/T) dz` over
  100–140 km geopotential, from regional daily means. This is the optically
  thin, O-excited limit; constants cancel in a relative trend.

Estimator: the density estimator (time + F10.7₈₁ + F10.7₈₁², 27-day HAC). For
SABER, a per-scan variant adds two harmonics each of solar local time and day of
year, because the ~60-day yaw cycle aliases local time.

| Series | Window | Trend %/decade [95%] |
|---|---|---|
| SABER, daily means | 2002-01-25 – 2019-12-01 | −2.04 [−5.45, +1.49] |
| SABER, scans + SLT/season harmonics | same | −1.73 [−4.65, +1.29] |
| SABER, daily means | 2006-01-01 – 2019-12-01 | −2.22 [−7.16, +2.98] |
| SABER, scans + harmonics | same | −1.87 [−5.86, +2.28] |
| WACCM-X run 3 proxy | 2006-01-01 – 2019-12-01 | −1.25 [−5.75, +3.46] |
| WACCM-X run 3 proxy | 2006–2024 | −4.00 [−7.30, −0.58] |

- **The point estimates are close, but the comparison is not informative.**
  Both are about −1 to −2 %/decade at fixed altitude over 2006–2019.
  - The individual intervals are ±4–5 %/decade. Their difference was not
    estimated; assuming independence it would be roughly ±6 %/decade, and no
    equivalence margin was set.
  - The comparison also spans different altitude bounds, sampled scans against
    daily model means, and a proxy against a retrieved flux.
  - The model's own trend depends on the window: −1.25 to 2019, −4.0 to 2024.
- At fixed altitude the cooling falls despite rising CO₂. Over 2006–2019, the
  model's log trend splits into cooling-weighted factor diagnostics
  (trapezoid-weighted), whose log-slopes sum exactly to the proxy's −1.25.
  The weights vary in time, so each diagnostic also mixes in shifts in where
  cooling occurs; these are not physical contributions:
  - CO₂ mixing ratio +3.87 [+0.54, +7.30];
  - O mixing ratio +2.72 [−0.21, +5.74];
  - number density squared −7.09 [−12.87, −0.93];
  - temperature term −0.60 [−3.35, +2.23];
  - remainder (redistribution between heights) +0.22 [−0.51, +0.95].

  The negative density term is consistent with contraction at fixed height,
  which would make a fixed-height cooling trend mix the forcing with its own
  response, but this decomposition does not identify that. Pressure levels would reduce the sensitivity to vertical
  displacement but would not isolate CO₂; that needs controlled runs.
- Limits:
  - regional, 18 years, about 2–84 scans a month;
  - the proxy ignores radiative transfer below ~110 km, N₂/O₂ excitation and
    diurnal covariance;
  - the model is total-forcing.

  A global SABER record (the 8,701 profile files are on Phoenix) would tighten
  the observed trend, but WACCM-X here is regional only.

## Implications for the paper

- State the without-CO₂ result as conditional on the strongest published CO₂
  responses. With weaker ones a residual decline of about 1.5–3 %/decade
  remains at all heights.
- Report the altitude-shape mismatch as a separate result for the global mean,
  with HASDM as supporting but not exclusive. It is the most specific
  constraint these data put on models.
- SABER: report as a consistency check with no discriminating power.

## Outputs (local, `outputs/figures/results/`)

- `co2_response_bracketing/`: `published_scalings.csv`, `required_scaling.csv`,
  `without_co2_by_published_response.csv`, `altitude_shape_test.csv`, and the
  figure.
- `saber_waccmx_cooling/cooling_trends.csv`.
