# CO₂-only expectation for the density trends — 2026-10-11

How much of each observed (and WACCM-X historical) density trend would CO₂ alone
produce, according to the published controlled WACCM-X experiments of Brown et
al. (2024)? This is a model-conditioned consistency check, not an attribution
from the observations.

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

1. **Synthetic series.** On exactly the days used in that series' fit, build
   `log10 R(CO₂(t), F10.7₈₁(t), z)`. CO₂ is NOAA Mauna Loa deseasonalized
   monthly data shifted to the NOAA global mean (offset 0.94 ppm) and
   interpolated to days.
2. **Same estimator.** Fit it with the solar-adjusted method used for the
   observations (time + F10.7₈₁ + F10.7₈₁², 27-day HAC), so the expectation and
   the observation share their days, solar weighting and definition.
3. **Constant %/ppm.** ln R is linear in CO₂, with the slope fitted per altitude
   and F10.7 over the grid's first four CO₂ levels (369–431 ppm, covering the
   historical record); this follows Brown et al.'s Table 3 convention.
   Piecewise interpolation was tried first and rejected: at 200–300 km and
   70 sfu, single grid segments are noisy (−0.60 then +0.11 %/ppm). A
   three-level fit (369–402 ppm) changes the expectations by ≤ 0.5 %/decade.
4. **Clamping and range.** F10.7 is clamped to 70–200 sfu, and altitudes outside
   200–1000 km are skipped.

## Results (%/decade; observed 95% HAC interval in brackets)

| Series | Altitude | Observed / modelled total | CO₂-only expectation | Share |
|---|---|---|---|---|
| Orbit-derived global mean, 1967–2019 | 250 km | −4.12 [−4.78, −3.45] | −3.25 | 0.79 |
| | 400 km | −4.94 [−5.99, −3.88] | −5.38 | 1.09 |
| | 575 km | −6.18 [−7.44, −4.90] | −5.85 | 0.95 |
| HASDM Mauna Loa, 2000–2025 | 200 km | −4.65 [−5.91, −3.37] | −3.00 | 0.64 |
| | 300 km | −5.07 [−7.03, −3.08] | −4.75 | 0.94 |
| | 400 km | −5.76 [−8.33, −3.11] | −6.18 | 1.07 |
| | 575 km | −6.78 [−9.95, −3.50] | −6.93 | 1.02 |
| | 800 km | −6.36 [−9.64, −2.95] | −7.60 | 1.20 |
| WACCM-X v2.1 regional, 2006–2024 | 206 km | −3.90 [−5.45, −2.33] | −3.70 | 0.95 |
| | 398 km | −5.94 [−9.27, −2.49] | −8.33 | 1.40 |
| WACCM-X v2.1 regional, 1982–1997 | 206 km | −4.27 [−6.15, −2.34] | −2.24 | 0.52 |
| | 398 km | −6.61 [−10.80, −2.22] | −4.89 | 0.74 |

Figure: `outputs/figures/results/co2_only_expected_trends/co2_only_expected_trends.png`
(local); full table `co2_only_expected_trends.csv`.

## Reading

- **Above about 300 km, CO₂ alone accounts for the observed trends in this model.**
  - Global mean at 325–575 km: shares 0.95–1.09.
  - HASDM at 300–600 km: shares 0.94–1.07.

  Every expectation above 300 km lies inside the observed 95% interval. Above
  600 km the CO₂-only expectation is larger than the HASDM trend (shares
  1.09–1.21), though still inside its wide interval. A purely
  CO₂-driven contraction of the size WACCM-X v1.2.2 simulates is therefore
  sufficient to explain the observed 1967–2019 and 2000–2025 trends there.
- **Below about 300 km the observations decline faster than CO₂ alone predicts.**
  The shares are 0.64–0.87 at 200–275 km. In the grid, the CO₂ response weakens
  sharply below 300 km, while the observed trends stay near −4 to −5 %/decade.
  Possible contributors include other forcings (ozone, other greenhouse gases,
  geomagnetic and lower-atmosphere changes), the products' low-altitude
  calibration, or a weaker model response. These data cannot separate them.
- **WACCM-X's own historical run does not follow the CO₂-only expectation
  consistently.** For 1982–1997 its total trend is stronger than the CO₂-only
  value (shares 0.52–0.74); for 2006–2024 it is weaker above 250 km (shares
  1.19–1.60). The two windows differ in CO₂ growth rate, solar activity and
  model version, the runs are regional rather than global and do not hold Kp
  at 0, and the run-3 window spans the deep 2008 and 2019 minima. This
  comparison is consistent with model-version and window dependence, not with
  one stable CO₂ share.
- **What can be claimed** (protocol "fingerprint consistency"): the sign,
  magnitude and altitude growth of the observed trends above about 300 km match
  a CO₂-only contraction as modelled by WACCM-X. That supports CO₂ as a
  sufficient explanation, not as the unique one. A single model of the CO₂
  response, with no intervals on the expectation, does not identify the CO₂
  share.

## Limits

- One model and version (v1.2.2) for the CO₂ response. The CO₂–O quenching
  rate alone changes modelled trends by up to about 40% (Solomon et al. 2015,
  in Brown's Table 2). No interval is attached to the expectation; its
  uncertainty is dominated by model choice, not by the fit.
- Global-mean annual means at Kp = 0, applied to the global-mean product,
  regional HASDM and regional WACCM-X alike.
- A constant %/ppm assumes linearity across 321–431 ppm. For 1967–1999 this
  extrapolates below the grid's 369 ppm floor, as in Brown et al.'s own
  conversion.
- The comparison inherits the observed trends' estimator, sampling and product
  caveats.

## Next

- If the controlled runs the user requested arrive, replace the published grid
  with run-level output to put intervals on the expectation and to reach below
  200 km.
- Possible additions: the Solomon (2015, 2018, 2019) and Cnossen (2020) modelled
  trends as extra CO₂-response estimates, to show how model-dependent the
  expectation is.
