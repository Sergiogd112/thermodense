# Density trends without CO₂ — 2026-10-11

Is any density trend left once the CO₂ response from published controlled WACCM-X
experiments (Brown et al., 2024) is removed from each series? This is a
model-conditioned test of whether CO₂ is sufficient to explain the observed
trends, not an attribution from the observations alone.

## Source

- **Brown et al. (2024)**, *JGR Space Physics* 129, e2024JA032659: WACCM-X v1.2.2
  with fixed surface CO₂ and fixed F10.7, Kp = 0, global-mean annual means.
- **Their published grid** `CO2_only` (doi:10.25500/edata.bham.00001075, CC BY 4.0;
  file `DEN-CO2scaling2000-2500_v2.nc`, MD5 `9373291f…967a`, checksum verified)
  gives density relative to the year 2000 for:
  - altitudes 200–1000 km;
  - CO₂ from 368.9 to 890 ppm;
  - F10.7 of 70, 100, 135, 170 and 200 sfu.

  Kept locally in `data/original/brown2024_co2_scaling/`.
- **Check against the paper:** at 400 km the grid gives −0.384 %/ppm (70 sfu)
  and −0.145 %/ppm (200 sfu) over 369–402 ppm, against −0.402 and −0.157 in
  their Table 3 for 2000–2020. The sensitivity falls below 300 km (about −0.07
  to −0.26 %/ppm at 200–300 km).

## Method (`scripts/co2_only_expected_trends.py`, PR #68)

For every series and altitude in the Figure 6 trend set (global mean, HASDM;
`current_density_trends_by_dataset_altitude.csv`) and the WACCM-X settled windows:

1. **CO₂-only signal.** On exactly the days used in that series' fit, compute
   `log10 R(CO₂(t), F10.7₈₁(t), z)`. For observational series, CO₂ is NOAA
   Mauna Loa deseasonalized monthly data shifted to the NOAA global mean (offset
   0.94 ppm) and interpolated to days. For WACCM-X, it is the run's own
   prescribed daily `co2vmr`, so the model is compared with the CO₂ history it
   actually saw (changed after review; the NOAA proxy grows faster than the
   model forcing in 2006–2024 and biased the earlier WACCM-X rows). ln R is linear in CO₂ (constant %/ppm, Brown et al.'s Table 3
   convention), with the slope fitted per altitude and F10.7 over the grid's
   first four CO₂ levels (369–431 ppm, covering the historical record).
   - Piecewise interpolation was rejected: at 200–300 km and 70 sfu, single
     grid segments are noisy (−0.60 then +0.11 %/ppm).
   - A three-level fit (369–402 ppm) changes results by ≤ 0.5 %/decade.
   - F10.7 is clamped to 70–200 sfu; altitudes outside 200–1000 km are skipped.
2. **Without CO₂.** Subtract that signal from the series day by day and refit
   with the same solar-adjusted estimator (time + F10.7₈₁ + F10.7₈₁², 27-day HAC).
   The interval is **recomputed** from the refitted residuals rather than
   copied from the observed fit. The CO₂ response varies with time and F10.7,
   so it need not lie in the regression's design space, and the residual
   covariance can change. Here the widths stay within 0–7% of the observed
   ones (for example 2.16 against 2.11 %/decade at 400 km in the global mean).
   An interval containing zero means no net trend is detected once this
   model's CO₂ response is removed. This assumes additivity in log density and
   treats the response as exact.
3. **For context**, the CO₂-only signal alone is fitted the same way; it is the
   trend CO₂ would produce by itself.

## Results (%/decade)

| Series | Altitude | Observed / modelled total | CO₂-only | **Without CO₂** [95%] |
|---|---|---|---|---|
| Orbit-derived global mean, 1967–2019 | 250 km | −4.11 | −3.25 | **−0.89 [−1.55, −0.22]** |
| | 275 km | −4.21 | −3.66 | −0.56 [−1.29, +0.18] |
| | 325 km | −4.44 | −4.40 | −0.03 [−0.91, +0.85] |
| | 400 km | −4.94 | −5.38 | +0.47 [−0.60, +1.55] |
| | 575 km | −6.17 | −5.84 | −0.34 [−1.65, +0.97] |
| HASDM Mauna Loa, 2000–2025 | 200 km | −4.65 | −2.99 | **−1.70 [−2.99, −0.39]** |
| | 225 km | −4.83 | −3.43 | −1.44 [−2.94, +0.08] |
| | 300 km | −5.07 | −4.74 | −0.34 [−2.39, +1.75] |
| | 400 km | −5.75 | −6.17 | +0.45 [−2.29, +3.27] |
| | 575 km | −6.78 | −6.92 | +0.16 [−3.26, +3.70] |
| | 800 km | −6.35 | −7.60 | +1.35 [−2.26, +5.09] |
| WACCM-X v2.1 regional, 2006–2024 (model CO₂) | 206 km | −3.90 | −3.21 | −0.72 [−2.33, +0.92] |
| | 398 km | −5.94 | −7.26 | +1.43 [−2.20, +5.19] |
| WACCM-X v2.1 regional, 1982–1997 (model CO₂) | 206 km | −4.27 | −2.24 | **−2.07 [−4.00, −0.10]** |
| | 398 km | −6.61 | −4.91 | −1.79 [−6.18, +2.80] |

Figure: `outputs/figures/results/co2_only_expected_trends/co2_only_expected_trends.png`
(local; the without-CO₂ trend and its interval against zero, with the observed and
CO₂-only trends as context). Full table: `co2_only_expected_trends.csv`.

## Reading

- **Above about 275 km (global mean) and 225 km (HASDM), no trend is left once
  CO₂ is removed.**
  - Global mean, 275–575 km: every without-CO₂ trend is between −0.56 and
    +0.47 %/decade, and every interval contains zero.
  - HASDM, 225–825 km: the same holds, with wider intervals (±1.5 to ±3.7).

  Against observed declines of −4 to −7 %/decade, this means no net residual
  trend is statistically detected at these heights once this model's CO₂
  response is removed. The intervals bound the **net** residual, the sum of
  all other influences plus any error in the modelled CO₂ response. They do
  not bound individual other causes, which could be larger and partly cancel.
- **At the lowest levels a significant decline remains:** −0.89 [−1.55, −0.22]
  at 250 km in the global mean and −1.70 [−2.99, −0.39] at 200 km in HASDM. The
  model's CO₂ response weakens sharply below 300 km while the observed trends do
  not. Candidates include other forcings (ozone, other greenhouse gases,
  geomagnetic or lower-atmosphere changes), the products' low-altitude behaviour,
  or a too-weak modelled CO₂ response. These data cannot separate them.
- **WACCM-X's own historical run**, compared with its own prescribed CO₂:
  - 1982–1997: residuals of about −2 %/decade at every height, significant
    only at 206 km;
  - 2006–2024: residuals of −0.7 to +2.3, none significant.

  The windows differ in CO₂ growth and solar activity, the model version
  (v2.1) differs from the CO₂-response grid (v1.2.2), and the runs are
  regional and not held at Kp = 0. This indicates window and model dependence,
  not a stable residual.
- **Claim** (protocol "fingerprint consistency"): given the WACCM-X CO₂ response
  of Brown et al. (2024), the observed density trends above about 275 km are
  consistent with CO₂ alone, with no net residual trend detectable. CO₂ is
  therefore a sufficient explanation there. That does not show other causes
  are absent or small individually, and the intervals do not include
  uncertainty in the CO₂ response.

## Limits

- **One model and version (v1.2.2) for the CO₂ response.** A 30% error in that
  response would shift the without-CO₂ trend by about ±1.5 %/decade at 400 km.
  The CO₂–O quenching rate alone changes modelled trends by up to about 40%
  (Solomon et al. 2015, in Brown's Table 2).
- **Scope.** The response is a global-mean annual mean at Kp = 0, applied to the
  global-mean product, regional HASDM and regional WACCM-X alike.
- **Linearity.** A constant %/ppm assumes linearity across 321–431 ppm; for
  1967–1999 this extrapolates below the grid's 369 ppm floor, as in Brown et
  al.'s own conversion. Additivity in log density is also assumed.
- **Inherited caveats.** The comparison inherits the observed trends'
  estimator, sampling and product caveats.

## Next

- If the controlled runs the user requested arrive, use run-level output to put
  an interval on the CO₂ response, propagate it into the without-CO₂ trend, and
  reach below 200 km.
- Meanwhile, bracket the CO₂ response with the other published model estimates
  (Solomon 2015, 2018, 2019; Cnossen 2020) to show how model-dependent the
  conclusion is.
