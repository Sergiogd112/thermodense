# The 200–250 km gap: bracketing the CO₂ response and SABER cooling — 2026-10-12

Follow-up to [density trends without CO₂](co2-only-expectation-2026-10-11.md),
which removed one model's CO₂ response (Brown et al. 2024, WACCM-X v1.2.2) and
found no net trend above ~275 km but a significant decline at 200–250 km
(−0.89 [−1.55, −0.22] %/decade at 250 km in the global mean; −1.70 [−2.99, −0.39]
at 200 km in HASDM). Two questions:

1. Would a different published CO₂ response close the gap?
2. Does SABER's lower-thermosphere CO₂ cooling trend differ from WACCM-X's?

Short answer: **no single strength of the CO₂ response fits both altitudes.** The
gap is in the altitude *shape* of the trend, not its size. Weaker published
responses leave a decline at every height. SABER cannot discriminate: its
regional cooling trend matches the model's within ±4–5 %/decade.

## 1. Bracketing the CO₂ response (`scripts/co2_response_bracketing.py`)

Brown's response is scaled by a factor *k* at fixed altitude shape. The
without-CO₂ trend of `obs − k·s` is linear in *k*, and its HAC variance is
quadratic in *k*. Refits at k = 0, 1, 2 recover both exactly. Check: k = 1
reproduces the 11 Oct rows (−0.89 [−1.56, −0.22] at 250 km, +0.47 [−0.60,
+1.55] at 400 km; the third decimal is rounding).

### Published model responses relative to Brown 2024 at 400 km

Each 400-km trend in Brown et al. (2024, Table 2) is divided by Brown's grid
under the same conditions. For a fixed F10.7, the grid is evaluated at that
F10.7 and the period's linear Mauna Loa CO₂ growth. For "all" solar activity,
the grid is applied to the period's daily F10.7₈₁ and CO₂ and fitted with the
same estimator; before 1959 the CO₂ is the WACCM-X/CMIP6 prescribed value. *k*
compares log-slopes.

| Study (400 km) | F10.7 | Period | Published %/dec | Brown 2024 equivalent | *k* |
|---|---|---|---|---|---|
| Qian et al. 2006 (TIME-GCM 1D) | 70 | 1970–2000 | −2.5 | −5.95 | 0.41 |
| Qian et al. 2006 | all | 1970–2000 | −1.7 | −4.48 | 0.37 |
| Qian et al. 2006 | 210 | 1970–2000 | −0.75 | −2.37 | 0.31 |
| Solomon et al. 2015 (TIME-GCM), kq 1.5e-12 | 70 | 1996–2008 | −4.9 | −7.72 | 0.63 |
| Solomon et al. 2015, kq 3.0e-12 | 70 | 1996–2008 | −6.8 | −7.72 | 0.88 |
| Solomon et al. 2015, kq 1.5e-12 | 200 | 1996–2008 | −1.8 | −3.09 | 0.58 |
| Solomon et al. 2015, kq 3.0e-12 | 200 | 1996–2008 | −2.1 | −3.09 | 0.68 |
| Solomon et al. 2018 (WACCM-X) | 70 | 1974–2003 | −3.9 | −6.19 | 0.62 |
| Solomon et al. 2019 (WACCM-X) | 200 | 1974–2003 | −1.7 | −2.46 | 0.69 |
| Cnossen 2020 (WACCM-X 2.0; **all forcings**) | all | 1950–2015 | −2.8 | −4.64 | 0.60 |
| Brown et al. 2021 (WACCM-X) | 70 | 1975–2005 | −5.8 | −6.31 | 0.92 |

**Brown 2024 is the strongest response in the compilation.** The other CO₂-only
responses are 0.31–0.92 times as strong at 400 km. Brown 2021, the same model
family, comes out at 0.92, which supports the conversion. Caveats:
- Table 2 values are global means at the studies' own conditions. The
  conversion assumes each period's CO₂ growth was the forcing.
- Some studies may include other changes. Cnossen (2020) is an all-forcing
  trend and is excluded from the shaded range in the figure.

### Scaling the observations would need

The *k* that leaves no net trend (k\*), and the *k* range whose 95% residual
interval contains zero (`required_scaling.csv`):

| Series | 200 km | 250 km | 275 km | 325 km | 400 km | 575 km |
|---|---|---|---|---|---|---|
| Global mean, 1967–2019 | — | 1.27 [1.07, 1.47] | 1.15 [0.95, 1.35] | 1.01 [0.81, 1.20] | 0.92 [0.72, 1.11] | 1.06 [0.84, 1.28] |
| HASDM (Mauna Loa), 2000–2025 | 1.56 [1.13, 2.00] | 1.27 [0.83, 1.71] | 1.16 [0.73, 1.59] | 1.02 [0.59, 1.45] | 0.93 [0.50, 1.37] | 0.98 [0.49, 1.47] |

- Above ~300 km the observations want roughly Brown's response (k ≈ 0.9–1.06).
  In the global mean, only Brown 2021 (0.92) and Solomon 2015 with the doubled
  quenching rate (0.88) fall inside [0.72, 1.11].
- The low altitudes want **more** than Brown's response (k ≥ 1.07 at 250 km in
  the global mean, ≥ 1.13 at 200 km in HASDM). That is stronger than every
  published response in the compilation.
- Figure: `outputs/figures/results/co2_response_bracketing/co2_response_bracketing.{png,pdf}`.

### What the published responses leave behind (`without_co2_by_published_response.csv`)

| Response used | Global mean 250 km | Global mean 400 km | HASDM 200 km | HASDM 400 km |
|---|---|---|---|---|
| Brown 2024 (k = 1) | −0.89 [−1.56, −0.22] | +0.47 [−0.60, +1.55] | −1.70 [−2.99, −0.39] | +0.45 [−2.29, +3.27] |
| Brown 2021 (0.92) | −1.17 [−1.83, −0.50] | +0.00 [−1.07, +1.08] | −1.95 [−3.24, −0.65] | — |
| Solomon 2015, kq 3e-12 (0.88) | −1.30 [−1.96, −0.63] | −0.22 [−1.28, +0.86] | −2.07 [−3.36, −0.77] | −0.34 [−3.06, +2.46] |
| Solomon 2018 (0.62) | −2.12 [−2.78, −1.45] | −1.61 [−2.67, −0.54] | −2.83 [−4.11, −1.53] | −1.94 [−4.62, +0.82] |
| Qian 2006, all (0.37) | −2.92 [−3.59, −2.26] | −2.95 [−4.01, −1.89] | −3.56 [−4.83, −2.27] | −3.48 [−6.12, −0.77] |

With any response weaker than about 0.72× Brown 2024, a significant decline
remains at 400 km in the global mean too. The "consistent with CO₂ alone above
275 km" result therefore holds only for the strongest modelled responses.
Typical modelled responses leave ~1.5–3 %/decade unexplained at every height.
This echoes the long-standing finding that models with the standard CO₂–O
quenching rate underpredict observed trends (Solomon et al. 2015).

### Altitude shape, independent of magnitude (`altitude_shape_test.csv`)

The trend of log10(ρ_low/ρ_400), the difference of two altitudes on common
days, against Brown's predicted change in that ratio:

| Series | Observed ratio trend | Brown 2024 predicts | Residual at k = 1 | k fitting the shape |
|---|---|---|---|---|
| Global mean 250/400 km | +0.85 [+0.40, +1.31] | +2.25 | −1.37 [−1.81, −0.93] | 0.38 [0.18, 0.58] |
| HASDM 200/400 km | +1.17 [−0.42, +2.79] | +3.39 | −2.14 [−3.71, −0.55] | 0.35 [−0.13, 0.83] |
| HASDM 250/400 km | +0.91 [−0.23, +2.05] | +2.45 | −1.51 [−2.63, −0.38] | 0.37 [−0.09, 0.84] |

- In the global mean, the *k* that fits the 250/400 shape (0.18–0.58) does not
  overlap the *k* that fits the 400-km magnitude (0.72–1.11). No rescaling of
  Brown's profile fits both. **The model's density response grows with altitude
  about 2.6 times faster between 250 and 400 km than the observed trends do.**
  In HASDM the ranges overlap marginally.
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

- **Observed and modelled cooling trends agree but are not informative here.**
  Both are about −1 to −2 %/decade at fixed altitude over 2006–2019, with
  intervals of ±4–5 %/decade. Those intervals are wider than any difference
  that could settle the density-shape question. The model's own trend depends on
  the window (−1.25 to 2019, −4.0 to 2024).
- At fixed altitude the cooling falls despite rising CO₂. Decomposed over
  2006–2019, the model's factors are:
  - CO₂ mixing ratio +3.98 [+0.55, +7.53];
  - O mixing ratio +2.78 [−0.53, +6.19];
  - number density squared −6.96 [−13.16, −0.32] (contraction at fixed height);
  - temperature term −0.87 [−3.81, +2.17].

  A fixed-height cooling trend therefore mixes the forcing with the contraction
  it causes. A pressure-level comparison would isolate the forcing.
- Limits:
  - regional, 18 years, about 2–84 scans a month;
  - the proxy ignores radiative transfer below ~110 km, N₂/O₂ excitation and
    diurnal covariance;
  - the model is total-forcing.

  A global SABER record (the 8,701 profile files are on Phoenix) would tighten
  the observed trend, but WACCM-X here is regional only.

## Implications for the paper

- State the without-CO₂ result as conditional on the strongest published CO₂
  response. With typical responses a residual decline of ~1.5–3 %/decade
  remains at all heights.
- Report the altitude-shape mismatch as a separate, magnitude-independent
  result. It is the most specific constraint these data put on models.
- SABER: report as a consistency check with no discriminating power.

## Outputs (local, `outputs/figures/results/`)

- `co2_response_bracketing/`: `published_scalings.csv`, `required_scaling.csv`,
  `without_co2_by_published_response.csv`, `altitude_shape_test.csv`, and the
  figure.
- `saber_waccmx_cooling/cooling_trends.csv`.
