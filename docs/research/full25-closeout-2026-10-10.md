# Through-2025 campaign closeout: paired-month bootstrap and ratio rerun — 2026-10-10

Closes the two open items from the [9 Oct status](phoenix-status-and-cleanup-2026-10-09.md)
for `bounded-20261006-full25`. Inference only: no training, no changes to the
frozen campaign. Paths are relative to the account's `karman-phoenix/` directory.

## Migration verification (Phoenix, 10 Oct)

- `uv run pytest` on `feature/vires-density-source`: **508 passed**. The VirES
  branch adds tests on top of the 500 counted at migration.
- Load spot check passed. One mid-archive file per TU Delft v02 mission (CHAMP,
  GOCE, GRACE, GRACE-FO, Swarm) and one HASDM archive decode with the package
  decoders. A SABER CO₂ cooling file opens with the module's reader, and 8,701
  files are present. The CO₂ and space-weather loaders work, and every
  `data/prepared-full25` array opens (135,009,471 rows).
- Correction to the 9 Oct receipt: TU Delft v02 was **copied from the laptop**,
  not re-downloaded, because the provider server was down. Only SABER CO₂
  cooling was re-downloaded on Phoenix.

## Setup

New sidecar root `full25-sidecars-20261010/`, outside the campaign. Every job
ran in a `bwrap` sandbox with the campaign, source arrays, `full25-repo` and
the venv mounted read-only. SGE `all.q@auriga`, 4 slots per task, CPU float32.
The `phoenix` and `phoenix02` queues deny this account (see the 8 Oct amendment receipt).

- **Paired-month tool:** `paired-month/tools/paired_date_uncertainty.py`,
  adapted from `paired-date-20261006`. Changes:
  - anchors to the full25 `comparison.json` (SHA-256 `e41df29a…`);
  - accepts the 8 Oct CPU-evaluation amendment's extra execution settings and
    binds the amendment receipt by hash;
  - expects 12,395,677 rows, 570 dates and 21 months;
  - bootstraps each period separately.

  Unchanged: per-day APE caches, canonical MAPE/count parity at 1e-4 pp for
  every category, 2,000 replicates, seed 20261005, ≥4 contributing clusters.
  On synthetic records, its 16-month original-period output matches the
  original tool to 1e-12, with identical intervals.
- **Ratio tool:** `ratio-analysis/tools/ratio_distributions.py`, copied from
  `full25-repo`. The only change is that results go to an `--output-dir` outside
  the campaign, with single-run/report modes for SGE arrays. The rebuilt support
  mask is byte-identical to the blocked root's (SHA-256 `e6f76e9f…`, 11,529,769
  accepted rows; 865,908 rows excluded for invalid NRLMSISE-00).
- Jobs: pilots 1565–1566; ratio support 1567; paired array 1568 (30 tasks);
  paired reports 1569–1571; ratio array 1572 (36 tasks); ratio report 1573.

## Held-out support

The full25 test split has **21 held-out months**. The 16 original-study months
are kept exactly, with **10,058,908 rows**, the same count as the original
study. The other 5 are added years: 2020-11, 2021-01, 2022-03, 2023-05 and
2025-09 (2,336,769 rows). The rotation slot **2024-07 has no test rows**, and
training holds only 44k (2024) and 33k (2025) of its 5M rows in those years.
Added-year rows are all at 400–550 km (no 200–400 km support), with no Ap ≥ 50.

## Paired-month bootstrap (gain = MAPE without CO₂ − with CO₂, pp)

Five-seed mean; month-cluster 95% percentile interval; seed SD; positive seeds.

| Architecture | All 21 months | Original period (16) | Added years (5) |
|---|---|---|---|
| paper_jb08_omni | +0.122 [+0.028, +0.209], SD 0.200, 4/5 | +0.123 [+0.028, +0.211], SD 0.147, 4/5 | +0.116 [−0.176, +0.358], SD 0.608, 2/5 |
| bounded_omni_depth_01 | −0.037 [−0.214, +0.144], SD 0.042, 0/5 | +0.005 [−0.189, +0.213], SD 0.047, 3/5 | −0.219 [−0.494, +0.031], SD 0.127, 0/5 |
| residual_jb08_depth_04 | −0.180 [−0.290, −0.065], SD 0.332, 2/5 | −0.170 [−0.300, −0.041], SD 0.337, 2/5 | −0.223 [−0.332, −0.105], SD 0.387, 2/5 |

The all-month means reproduce the campaign's canonical gains exactly. Every
pass also passed per-category parity. For reference, the original study's
paper_jb08_omni gain on the same 16 months was +0.072 [−0.108, +0.271]. That
model was trained on different data with different normalization, so the two
results are not a replication pair.

Per-seed overall gains (all months):

- paper_jb08_omni: +0.009, +0.036, −0.043, +0.457, +0.151;
- bounded_omni_depth_01: −0.013, −0.059, −0.100, −0.005, −0.008;
- residual_jb08_depth_04: −0.656, +0.083, +0.165, −0.157, −0.335.

In the added years, paper_jb08_omni's +0.116 comes from one seed (+1.173;
the other four range from −0.307 to +0.041).

## Reading

- **No reliable CO₂ gain.** The month intervals hold the five trained models
  fixed. The training-seed standard error of the five-seed mean is
  0.089 pp (paper) and 0.148 pp (residual), comparable to or larger than the
  month-interval half-widths (about 0.09 and 0.11). The paper interval
  excluding 0 and the residual interval below 0 both rest on one or two
  seeds. Combining the two sources roughly, as independent, gives intervals
  that include 0 for every architecture.
- Signs disagree across architectures (+0.12, −0.04, −0.18). This is
  architecture- and seed-dependent prediction, not a consistent CO₂ effect.
- Added years: 5 clusters only, no 200–400 km support, and seed SDs up to
  0.6 pp, so they are descriptive at best. They are also not future-year
  extrapolation: the split rotates months and training includes 2020–2025.
- 3 architectures × 11 strata × 3 periods are uncorrected exploratory
  descriptions. The test months were already scored by the canonical
  evaluation, so these are post-result intervals.
- This is consistent with the seminar finding: `all__year__[y]` is already a
  near-perfect secular proxy. The [cheap-control design](cheap-control-study-design-2026-10-10.md)
  is the next step to test that directly.

## Ratio analysis (fresh root)

36/36 runs evaluated, no errors: 30 confirmation runs plus 6 pilot previews,
which are never pooled with confirmation runs. Ratio = observed / model density on the 11,529,769-row shared
support. Statistics are an equal-weight mixture of per-seed predictions, not
an ensemble.

| Model | Mean | SD | Mode | Ratios > 2 |
|---|---|---|---|---|
| JB08 | 0.927 | 0.339 | 0.853 | — |
| NRLMSISE-00 | 0.776 | 0.344 | 0.792 | — |
| paper_jb08_omni, with / without CO₂ | 1.021 / 1.022 | 0.337 / 0.338 | 0.992 / 0.988 | 71,099 / 70,043 |
| bounded_omni_depth_01, with / without | 1.018 / 1.030 | 0.335 / 0.351 | 0.988 / 0.992 | 62,082 / 74,991 |
| residual_jb08_depth_04, with / without | 1.015 / 1.021 | 0.336 / 0.344 | 0.972 / 0.978 | 192,451 / 198,218 |

- All 32 per-run results that also exist in the blocked root are
  **bit-identical** (mean, SD and histogram counts). The amendment changed
  provenance hashes only. The new root adds the four missing
  `bounded_omni_depth_01` no-CO₂ seeds (01–04), so all three cohorts are complete.
- Mean observed/model ratios are about 1.02 for the neural models, 0.927 for
  JB08 and 0.776 for NRLMSISE-00 on this support. So observations average
  about 7% (JB08) and 22% (NRLMSISE-00) below the models' densities, measured
  relative to model density; the reciprocal is not the same quantity. The
  ±CO₂ distributions differ by ≤0.012 in mean and ≤0.016 in SD, which does not
  change the MAPE reading above. (Wording corrected 2026-10-11 after review;
  the numbers are unchanged.)
- Outputs: `full25-sidecars-20261010/ratio-analysis/outputs/report.json`.

## Remaining

- Sidecar outputs stay on Phoenix (`full25-sidecars-20261010/*/outputs`,
  about 4 MB). They are not copied into this public repository.
- The old blocked `ratio-analysis/` inside the campaign was left untouched.
