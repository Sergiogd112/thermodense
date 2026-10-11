# WACCM-X surface-to-upper-atmosphere CO₂ lag — first results, 2026-10-11

Exploratory model diagnostic, started at the user's request to "start with what
we have" after the [regional download](waccm-x-regional-download-2026-10-10.md).
Single free-running WACCM-X v2.1 historical realisation, regional 5 × 5 subset
(16–25°N, 200–210°E), latitude-weighted 25-cell mean. This is not an
observation, and not a controlled CO₂ experiment.

## Run structure (from the provider README and file metadata)

| Run | Spin-up base | Stored | Written |
|---|---|---|---|
| 1 | 1949-01-01 | 1950-01-02 → 1973-12-31 | Mar–May 2023 |
| 2 | 1973-01-01 | 1974 → 1997 | Mar–Jun 2023 |
| 3 | 1997-01-01 | 1998 → 2021, then 2023 → 2024 | 2023; continuation Dec 2024–Feb 2025 |

- **Run 3's 2023–2024 output is the same run resumed.** It has the same base
  date, and the model step counter at 2023-01-02 equals the 2021-12-30 value
  plus exactly 368 days × 288 steps. 2022 was simulated but not delivered.
- **Restart initialisation.** The README says each run started from the
  default year-2000 state with CO₂ and CO scaled by the start year's surface
  ratio, with a one-year spin-up.

## Pipeline

1. `scripts/waccmx/extract_regional_daily.py` (SGE array 1580, 74 tasks, all
   exit 0) re-checks every member's CRC and writes daily means to
   `data/derived/waccmx_regional_daily_v1` (228 MB, gitignored):
   - forcing series;
   - 126-level profiles for the central column and the regional mean;
   - fixed geopotential heights (20–500 km), interpolated per hour and cell
     before averaging (log-linear for density), and NaN above the column top.
2. `scripts/waccmx/co2_lag_analysis.py` (SGE 1582) produces
   `data/derived/waccmx_co2_lag_v1/lag.json` and `tracer_age.png/.pdf`. Its steps:
   - take monthly means and remove the seasonal cycle with a centred 2×12-month
     running mean that never bridges the 2022 gap;
   - compute tracer age `t − S⁻¹(c_z(t))` against the prescribed surface
     `co2vmr` (monotonic after deseasonalizing);
   - compute intervals with a 24-month moving-block bootstrap (1,000 draws,
     seed 20261011) whose blocks never cross the missing 2022 or running-mean
     gaps (corrected 2026-10-11 after review: means unchanged; mean-age
     intervals moved by ≤ 0.03 years, trend intervals by ≤ 0.14 years/decade).

   On synthetic accelerating series with a seasonal cycle and the 2022 gap, the
   inversion recovers known lags of 1, 3 and 4.2 years to within 0.001 years.
3. A joint fit `c_z = a·S(t − τ)` was tried first and abandoned. With
   near-linear growth, the lag kept hitting its 0 and 10-year bounds, because
   a and τ cannot be separated.

## Findings

**1. Restart artifacts last 6–8+ years, not one.** Every run starts with
negative tracer ages at 20–70 km (−7 years in run 1, about −2 in run 2, about
−0.6 in run 3). That is physically impossible and comes from the scaled
year-2000 initial state. Ages reach a plateau after about 6–8 years in runs 2
and 3.

Run 1 never fully settles. In its year-8 window (1958–1973), ages at 40–70 km
still rise by +0.4 to +0.9 years per decade, with intervals excluding 0, and its
60 km age falls below its 40 km age. Treat run 1 as unusable for transport.
Through 1981 and 2005, the first ~8 years of runs 2 and 3 should also be
excluded from any upper-atmosphere CO₂ or cooling analysis.

**2. Equilibrated tracer age** (window from year 8; mean [95% block bootstrap]):

| Height | Run 2 (1982–1997) | Run 3 (2006–mid-2021) |
|---|---|---|
| 20 km | 1.11 [1.04, 1.20] y | 1.09 [1.03, 1.13] y |
| 40 km | 2.92 [2.87, 3.02] y | 3.00 [2.94, 3.07] y |
| 60 km | 3.01 [2.93, 3.13] y | 3.29 [3.23, 3.36] y |
| 70 km | 4.10 [4.05, 4.23] y | 4.16 [4.06, 4.31] y |

Run 3 uses only months inside a complete 24-month block: its 2023–24
continuation leaves 12 smoothed months, too short for any block, so it is
excluded from both the means and their intervals (aligned 2026-10-11 after
review).

- Runs 2 and 3 agree at 20, 40 and 70 km. Starting the window at year 6 or 10
  changes the means by ≤0.08 years.
- Run 3 shows small positive within-run trends (60 km: +0.23 [+0.09, +0.34]
  years per decade). That could be circulation change or the known bias of
  tracer age when growth accelerates; it is unresolved.
- At 80 km and above, CO₂ is already depleted (CO₂/surface ratio 0.94–0.95),
  so an inverted "age" there (11–18 years) is not a transport age and is not
  reported as one.

**3. Depletion above 80 km** (CO₂ / current surface value, run 3 year-8
window): 0.944 at 80 km, 0.736 at 90 km, 0.388 at 100 km, 0.170 at 110 km and
0.084 at 120 km. It declines slowly from run to run (100 km: 0.413 → 0.395 →
0.388).

**4. CO₂ growth in the cooling region tracks the surface only partially.**
Trend of ln CO₂ in % per decade, raw / adjusted for a linear F10.7a term:

| Height | Run 2 (1982–1997) | Run 3 (2006–2024) |
|---|---|---|
| surface | 4.18 / 4.22 | 5.12 / 5.12 |
| 20–80 km | 4.2–4.4 / 4.2–4.4 | 5.1–5.2 / 5.1–5.2 |
| 100 km | 3.59 / 2.60 | 3.03 / 3.89 |
| 110 km | 4.25 / 2.59 | 1.93 / 3.21 |
| 120 km | 4.58 / 2.76 | 1.76 / 2.94 |
| 140 km | 4.52 / 2.70 | 1.99 / 2.82 |

- Lower-thermosphere CO₂ is strongly solar-sensitive: about −5 to −9% per
  100 sfu at 100–140 km.
- Raw trends at 100–140 km are aliased by the solar cycle. Run 2's window
  starts near a maximum; run 3's runs from the 2008 minimum to the 2024 maximum.
- After the adjustment, the two runs agree: CO₂ at 100–140 km grows at roughly
  **2.6–3.9% per decade, about 55–75% of the surface rate**. Below 80 km it
  tracks the surface.
- These growth rates are descriptive point estimates without intervals; the
  residuals are autocorrelated and the solar term is a single linear F10.7a
  regressor.

## Implications and limits

- For the paper's model-conditioned fingerprint, a surface CO₂ ratio does not
  carry over unchanged to the 100–139 km cooling region. In this model, the CO₂
  increase there is about half to three-quarters of the surface increase, and
  it is strongly modulated by solar activity.
- Any WACCM-X trend analysis should drop the first ~8 years of each run and
  should not use run 1 above the stratosphere.
- Limits:
  - one realisation, one model version and one region;
  - the 25 cells are not independent;
  - the intervals describe month-to-month variability, not model or forcing uncertainty;
  - heights are geopotential;
  - the density and helium caveats in the
    [inventory](waccm-x-regional-inventory-2026-10-05.md) still apply.
- Next candidates, not started:
  - the 300/400 km density trend and solar dependence in runs 2 and 3 (now cheap);
  - a provider query about the 2022 output;
  - a comparison with SABER's 100–139 km cooling flux over the same years.
