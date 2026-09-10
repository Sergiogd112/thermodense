# Global-mean activity-grouped ParCorr and polynomial benchmark

## Agreed scope (8 September 2026)

- All ten existing global-mean density channels, **not an average across altitudes**:
  250, 275, 325, 375, 400, 425, 475, 525, 550, 575 km.
- Daily raw `F10.7_OBS`, daily `AP_AVG`, daily `KP_SUM`, and MLO `CO2_ppm`
  are the four additional graph nodes/predictors. `KP_SUM` is a daily sum,
  not an instantaneous Kp or the native 3-hour Kp used by the HASDM runner.
- **Raw F10.7 for fitting; `F10.7_OBS_CENTER81` for classification only.**
  Activity bins: low `<100`, moderate `[100,160)`, high `>=160` sfu.
- Full daily calendar, explicit missingness, no interpolation, detrending,
  deseasonalizing, or resampling of the analysis input. Legacy density values
  below -200 and negative MLO CO₂ sentinels become missing.
- ParCorr: an ungrouped reference and three separate activity analyses, with a
  0–180-day window. The activity mask selects **Y/target dates only**; preceding
  X/Z histories can cross regime boundaries. Selected dates are never concatenated.
- Polynomial: a **pooled predictive fit** on the same daily input, with held-out
  performance stratified by solar activity. It is not three regime-specific fits.

Centered activity labels include future dates relative to their central date.
The grouping is therefore retrospective, not an operational real-time forecast
classification. No causal-discovery qualification is claimed for these raw
grouped analyses. Differences in significant edges between groups do not by
themselves constitute significant between-group differences.

## Input assembly and preflight

```console
uv run python -m thermodense.benchmarks.global_activity_input \
  --density-parquet data/decoded/orbit_derived_global_mean/orbit-density-ds03-density-values.parquet \
  --space-weather-csv data/original/space_weather/SW-All.csv \
  --co2-csv data/original/co2/co2_daily_mlo.csv \
  --output data/products/global-activity-daily-v1
```

The immutable product consists of `manifest.json` and `input.npz`. The loader
checks the payload checksum, node schema, calendar, availability masks, and
raw/classification declarations. Source paths/hashes are retained from assembly;
the remote payload can be verified without re-reading the local source files.
Data remain outside tracked Git content.

Verified real input: **19,328 × 14**, daily UTC from **1967-01-01 to 2019-12-01**.
Density availability ranges from 19,290 to 19,328 days; weather variables are
present on all dates and MLO CO₂ on 13,905. Missing CO₂ is not filled.

| Group | Target dates | Dates after 360-day warm-up | Representative CI samples |
|---|---:|---:|---:|
| All | 19,328 | 18,968 | 13,882 |
| Low | 8,264 | 8,264 | 6,429 |
| Moderate | 6,852 | 6,518 | 4,105 |
| High | 4,212 | 4,186 | 3,348 |

The representative check used F10.7 at lag 1 → density at 250 km, conditioning on
lag-1 density and MLO CO₂ with `tau_max=180`. Its statistics were finite in each
group, locally and remotely. These counts **are not support guarantees for the
remaining tests**, which can have different conditioning sets.

## ParCorr execution and outputs

```console
python -m thermodense.benchmarks.global_activity_pcmci \
  --manifest /run/input/manifest.json --output /run/outputs/parcorr-low --group low
```

Tigramite PCMCI+ uses analytic ParCorr, `pc_alpha=0.05`, `mask_type='y'`, an explicit
missing flag, and `remove_missing_upto_maxlag=False`. Eligible candidates are
driver→density and density↔density; density→driver is forbidden. One BH family
of **25,285 hypotheses per group** covers 140 × 180 lagged links, 40 driver/density
contemporaneous links and 45 unique density pairs. This is **per-group correction**,
not a correction over all four analyses or a group-difference hypothesis test.

Each output directory is reserved before fitting. Outputs:

- `status.json`: running/failed/complete state, identities, versions, settings,
  interpretation limits, final hashes and actual CI-call count.
- `ci_test_trace.jsonl`: incremental X/Y/Z, nonconstant coordinates, actual sample
  count/conditioning dimension, cutoff, alpha and returned statistics/errors.
  The trace is per CI call; it is not mislabeled as one support value per edge.
- `pcmci_raw.npz`: raw graph/p/r matrices, saved atomically **before** CSV export.
- `eligible_links.csv`: full eligible BH family, q values, normalized orientation
  and original matrix coordinates. Adjacency and direction are distinct.

Failed fits do not become completed evidence. Raw matrices survive a later
export failure, but interrupted PCMCI fitting is not resumable from its CI trace.

## Polynomial benchmark

```console
python -m thermodense.benchmarks.global_activity_polynomial \
  --manifest /run/input/manifest.json --output /run/outputs/polynomial \
  --lag 180 --epochs 15 --threads 2 --batch-size 64 \
  --lambdas 0.0001,0.001,0.01 --seed 0
```

Ten density outputs use all fourteen predictors at lags 1–180 days. Degrees
1, 2, and 3 contain coordinatewise powers, **not cross-variable interaction
products**. Linear availability indicators handle missing predictor entries;
missing targets do not contribute to the loss. Numeric zero replacement after
training-only scaling is an explicit modeling convention, not interpolation.

The target dates are split chronologically 70/15/15. Training-only standardization
and feature scaling are permitted for this separate predictive sensitivity
benchmark, not inserted into the ParCorr input. Each degree's ridge/epoch winner
is selected using validation only; each winner is then evaluated on test once.
Earlier observed history remains available for one-step predictions across split
boundaries. This is not recursive multi-step forecasting or rolling-origin CV.

The optimization uses bounded minibatch Adam with a ridge penalty. Completing
15 epochs is reported as exhausting the epoch budget, **not proof of convergence
or the optimal ridge solution**. Performance is compared with persistence and a
training-mean baseline, including identical persistence-comparable support and
per-altitude/per-activity metrics. Unknown activity dates are counted separately.

Epoch checkpoints bind the input/code/config identity, degree, exact ridge, and
feature scales. Deterministic per-epoch permutations allow reproducible resume.
Completed artifacts are checksum-checked before reuse. The output includes
`trials.json`, `metrics.json`, `models.npz`, and `provenance.json`. The NPZ retains
both the overall validation-selected W1/W2/W3 and each degree winner's matrices
(`degree_1_W1` through `degree_3_W3`). W1 has value + availability channels;
W2/W3 contain value channels only. These matrices are **not causal effects**.

## Initial Phoenix submission

Run root: `~/.local/share/thermodense/runs/global-activity-daily-v1`.
Only the three new modules and package initializers were copied into its own
`src/` snapshot. Job 1139's source, inputs, environment and scheduler settings
were not modified. The existing project Python environment supplies dependencies.

Submitted on **8 September 2026 at 19:44 CEST** to `all.q@auriga`, using
`scripts/run_global_activity_sge.sh`, **two slots per job**, `h_rt=24:00:00`,
`h_vmem=4G`, and `RUN_ROOT` pointing at this isolated directory.

| Job | Analysis | State checked at 19:45 CEST |
|---|---|---|
| 1141 | ParCorr, all dates | Running; no stderr at initial check |
| 1142 | ParCorr, low activity | Queued |
| 1143 | ParCorr, moderate activity | Queued |
| 1144 | ParCorr, high activity | Queued |
| 1145 | Polynomial comparison | Queued |

The scheduler reports the per-user rule `10_core_limit/1`. With eight cores used
by job 1139, only one two-core job can start. Unused machine cores are not
available beyond this quota. Queued jobs have not failed; their wall-time limits
start at execution, not submission. No reliable completion ETA is available yet.

Example submission (the receipt above records jobs already submitted; do not
submit duplicates):

```bash
qsub -S /bin/bash -q all.q@auriga -pe smp 2 -l h_rt=24:00:00,h_vmem=4G \
  -N gm_pc_low -v RUN_ROOT="$root" \
  -o "$root/logs/parcorr-low.stdout" -e "$root/logs/parcorr-low.stderr" \
  "$root/run_global_activity_sge.sh" parcorr low
```

Local and remote hashes matched before submission:

| Artifact | SHA-256 |
|---|---|
| input.npz | `30587e76f72cebd7687cf9e544ac2f1fab15f090f5bec7923a2f37a9f32d1393` |
| manifest.json | `05a8876e6dfeebd36946c8503c1e07b80009147c8f29bcbfb3a2b998a318441f` |
| global_activity_input.py | `46146aa0e474ad21a2444bbe30076bd9288c677cad973112ca5a9bef4135677a` |
| global_activity_pcmci.py | `05d6ea39b01b0e9735bbbb3368019f6de8027457912e20d6c94c5f98f9c0d9af` |
| global_activity_polynomial.py | `39a1463b0d163976fa5263dfe4a7a6ee9978a5682d7232d714d45d6c5e40410f` |

Validation before deployment: **397 tests passed**; targeted BasedPyright at
error level and Ruff passed; shell syntax and `git diff --check` passed.
Independent scientific review blockers were resolved before submission. These
checks validate execution contracts, not scientific model adequacy.
