# Karman bounded-study closeout

Date: 2026-10-05. **The benchmark is complete; temporal/data uncertainty is not
yet quantified.** No new model fitting, checkpoint deserialization or inference
was performed for this closeout. Frozen scientific files were read, not changed.

## Bottom line

Explicit CO₂ yields small positive **mean** overall MAPE changes, but the paired
direction is inconsistent across seeds and strata. These results do not establish
a robust predictive contribution beyond calendar time, a causal effect, or absence
of a physical CO₂ effect. **Do not launch another architecture sweep.**

## Verified completion and identities

- Pilot: 16/16; confirmation training: **30/30**; canonical evaluation: **30/30**;
  ratio sidecar: 36/36. The controller/monitor reports `finished`, integrity true.
- Three validation-selected finalists; five seeds × CO₂/no-CO₂ each. Confirmation
  uses 5M training / 1M validation rows; unchanged canonical test indices.
- Read-only attestation checked **all 30 evaluation-file hashes against the final
  report**, their run/manifest and bounded-input contracts, and **all 30 best-
  checkpoint hashes against their evaluation records**. No discrepancies.
- A separate read-only check matched **10 frozen implementation-file hashes**
  and both sampled train/validation index hashes against their stored expectations.
  This is not a new end-to-end training/evaluation reproduction or a full rehash
  of every original feature/target array.
- Independent arithmetic reconstructed **all 36 architecture/category report
  rows** from the per-seed JSONs using Python `statistics`, rather than invoking
  the campaign report function. Means, paired means, sample SD and counts agreed
  to absolute tolerance **1e-12**; all pairs match count and execution backend.

| Object | SHA-256 |
|---|---|
| Final `comparison.json` | `7247931f74fbcee8e904f11d7775289228041bf6a3b278e2038df188c9d3069c` |
| Frozen `selection.json` | `315f8e5ebeacf40a16f5c2ff52982cc06b3a98a6561a779a8790f4d01ac1dbb2` |
| Confirmation manifest | `ade07c7c8c66d9ee94f1c3db3eb12472258d5facbcd3a348a80793b84b07cf39` |
| Confirmation metadata | `fd115bea28229aae93da03e604cfd6d25bb238471bb9d714e276b498729f7f29` |
| Local attestation receipt | `0859bb2a0cff36c4659f0b3aa579691d46ee7cf4ef4c985898bbd2dcf1c88aa0` |
| Independent arithmetic output | `e6ebe946b979246a3d8d183dd2dcbbfbee377b5cd231608c2f7bb450ea8adcb3` |

Receipts are under ignored, local-only
`data/products/co2-paper-preparation-20261005/`: `karman-input-attestation.json`
and `karman-closeout-recomputed.json`. Canonical outputs/checkpoints stay in the
frozen deployed study. Local Git HEAD alone does not identify that deployment.
Test-date counts, frozen-code/index checks and the original ratio-support
summary are retained in `karman-test-support-attestation.json` (SHA-256
`7b24aa667b33bc8d18b0b5da76d916604c3eb9e08c569d16bd0cd95e6ddba6c7`).
This makes the 434-date/16-month and 9,581,190-row statements inspectable without
relying on a chat summary.

## Overall paired results

Define `gain_pp = MAPE_without_CO₂ − MAPE_with_CO₂`; positive is improvement.
MAPE is already expressed in percent, so gains below are **percentage points**.
Each mean uses five models evaluated on **10,058,908 identical test rows**.

| Architecture | Without CO₂ MAPE (%) | With CO₂ MAPE (%) | Mean gain (pp) | Paired seed SD (pp, n−1) | Positive seed pairs |
|---|---:|---:|---:|---:|---:|
| Paper static, two hidden layers | 14.708834 | 14.641917 | 0.066917 | 0.138652 | 3/5 |
| Paper OMNI | 12.571912 | 12.499765 | 0.072147 | 0.176470 | 3/5 |
| Residual OMNI depth 2 | 12.464114 | 12.449846 | 0.014268 | 0.238998 | 2/5 |

The 0.072147 pp paper-OMNI gain is approximately a **0.574% relative reduction
in MAPE**, not a 7.2% density change or a 1.48% test-MAPE gain. The earlier
~1.48/1.59% numbers refer to **validation log-MSE**, a different metric/sample.

### Individual overall paired gains (pp)

| Seed | Training backend | Static | Paper OMNI | Residual OMNI-2 |
|---|---|---:|---:|---:|
| 0 | GPU AMP | 0.244909 | 0.254312 | −0.054978 |
| 1 | GPU AMP | −0.048227 | −0.107212 | 0.257668 |
| 2 | CPU float32 | 0.072025 | −0.115092 | −0.270631 |
| 3 | CPU float32 | 0.154641 | 0.104584 | 0.266035 |
| 4 | CPU float32 | −0.088763 | 0.224143 | −0.126753 |

CO₂ controls use the same backend/precision by seed. Seed variability mixes
initialization/sampling and the fixed execution policy; it is not five independent
measurement datasets. No seed-only t interval or p-value is presented as a
data-level confidence statement.

## Stratum direction checks—not new selected claims

| Predefined test category | Rows | Static gain (pp) | Paper OMNI gain (pp) | Residual OMNI-2 gain (pp) |
|---|---:|---:|---:|---:|
| 200–250 km | 174,125 | 1.174927 | 0.261797 | −0.760250 |
| 250–300 km | 792,205 | 0.254571 | 0.057903 | −0.130710 |
| 300–350 km | 947,011 | 0.068345 | −0.143576 | −0.051816 |
| 350–400 km | 1,089,340 | 0.187166 | −0.156748 | −0.261907 |
| 400–450 km | 1,521,726 | 0.040496 | −0.226899 | −0.105989 |
| 450–500 km | 4,517,183 | −0.018106 | 0.245169 | 0.190133 |
| 500–550 km | 1,017,318 | 0.018095 | 0.175743 | 0.015970 |
| Quiet (Ap 0–15) | 8,835,584 | 0.066382 | 0.115426 | 0.010965 |
| Mild (Ap 15–30) | 952,176 | 0.092469 | −0.175020 | −0.035547 |
| Minor (Ap 30–50) | 193,388 | 0.014945 | −0.396463 | 0.128018 |
| Major (Ap 50+) | 77,760 | −0.055917 | −0.653527 | 0.716677 |

The overall result is observation-weighted, with many rows at 450–500 km and
quiet conditions. Altitude categories do not cover every overall row; activity
rows are another partition, not additional independent observations. Rare-event
row counts are not independent storm counts. No favourable stratum is promoted
after seeing these values.

## Test sample and uncertainty boundary

Reading the frozen test indices and dates—without model inference—found:

- **434 unique calendar dates**, 2004-03-01 through 2019-09-30.
- **16 held-out calendar months**, one per year in the rotating monthly split:
  March/May/July/September/November/January repeat across years.
- Gaps within months are retained; there are long gaps between held-out episodes.
  Ten million orbital rows are not ten million independent temporal replicates.
- This remains a same-period monthly holdout, **not** a future-year or unseen-
  mission test. The currently prepared arrays do not retain mission identity.

Canonical MAPE uses the full 10,058,908 neural test rows. Ratio diagnostics use
the shared empirical-baseline-QC subset **9,581,190 rows**, excluding 477,718
rows, mainly because of missing/invalid NRLMSISE-00 reference values. A ratio
histogram's full-tail population SD is not a confidence interval, and combining
per-seed histograms is a **mixture, not a per-row ensemble predictor**.

**The missing quantity:** neither canonical evaluator nor ratio sidecar retains
per-date/per-observation losses. Their sums/moments/histograms cannot reconstruct
a paired temporal bootstrap. Claiming a temporal CI from these artifacts would
be unsupported.

### Small next pass, if approved

Use frozen paper-OMNI checkpoints first (five matched CO₂ pairs), same canonical
test rows, no fitting/selection changes. In a read-only, isolated CPU float32
inference pass, retain per-date `sum_absolute_relative_error,count` and optionally
log-error moments, binding input/index/implementation/checkpoint hashes.

- **Do not apply the ratio support mask** to a canonical-MAPE uncertainty pass;
  that changes the target sample. Validate all row identities and counts, not
  only matching totals. Do not independently select rows for each CO₂ variant.
- Reconstruct each original MAPE within a declared device/rounding tolerance;
  compare before bootstrap. Outputs go to a new independent sidecar, never over
  canonical files or controller manifests.
- Pair month-cluster resampling across every seed/CO₂ vector, preserving all
  included row weights; report the limited **16-cluster** support and assumptions.
  Keep calendar gaps explicit. A within-episode day-block sensitivity must not
  join unrelated dates as a regular daily series. Raw-row resampling is invalid
  for this intended temporal inference.
- This is a post-result uncertainty analysis of the frozen comparison, not a
  fresh confirmatory selection exercise, a future-year forecast test or causal
  CO₂ attribution. Fix the uncertainty specification before generating it.
- Checkpoint loading uses trusted-model serialization in the current evaluator;
  inspect and isolate the pass before execution. No such pass was launched here.

## Paper wording and next decision

Safe now: “Adding explicit CO₂ produced small, seed- and stratum-dependent
changes in the frozen monthly-held-out density-estimation benchmark. A robust
data-level improvement and a CO₂-specific physical contribution were not
established by this aggregate comparison.”

Do **not** say “CO₂ has no effect,” “CO₂ causes the observed gain,” or “the
residual architecture proves the best scientific model.” Next priority is the
time-proxy comparison and paired uncertainty, not deeper neural heads.

Verification coverage: artifact hashes/contracts checked; aggregate arithmetic
independently reconciled; test-date support inspected; original fitting and
inference not rerun; temporal/statistical and causal coverage not established.
