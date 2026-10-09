# Separate through-2025 Karman campaign: assessment and resource adjustment

Date: **2026-10-06**, operational inspection around 22:31–22:58 CEST. The user
explicitly authorized checking work launched in another chat and stopping or
continuing it at discretion. This is a **preliminary artifact/configuration
assessment**, not independent qualification of every new input or result.

## Identified work

- New root: `/home/sergio/karman-phoenix/bounded-20261006-full25`.
- Separate implementation: `/home/sergio/karman-phoenix/full25-repo`.
- Separate data: `/home/sergio/karman-phoenix/data/prepared-full25`.
- The original `bounded-20261004` and completed paired-month sidecar are untouched.
- Data metadata: **135,009,471 rows**, 2004-02-01 through 2025-10-31, TU Delft v02
  inputs; CO₂ previous-day and trailing-365-day features; training-split static
  scaling and training-month OMNI quantile transformation.
- The preparation script documents the October 2025 cutoff as a trailing-365-day
  observed-CO₂ coverage gate, not an arbitrary retrospective outcome cutoff.
- Pilot: **16 fits**, two seeds, seven architecture names plus two block-order
  diagnostics, all with CO₂; 500k training / 100k validation rows, maximum 20
  adaptive epochs. It reuses the bounded study machinery, not the paused 125-run
  legacy scheduler. SGE initially submitted GPU job 1455 and CPU jobs 1456–1461.
- At the 22:52 inspection: pilot 6/16 complete, GPU job 1455 running, no selection
  or confirmation manifest yet. No canonical test result was inspected.

## Decision: continue, with narrower interpretation

Extending the population through 2025 can be useful as a **new reduced-training
predictive benchmark**. I did not kill its fits or change models, manifests,
learning rate, seeds, training precision, source arrays or scientific splits.
Its warning already says it is not exact full-data paper replication.

The rotating held-out-month split is retained, and training already includes
2020–2025 observations. This is **not future-year extrapolation**. Older held-out
calendar periods overlap periods already examined in the preceding campaign;
the new combined score cannot be presented as entirely outcome-uninspected evidence.
Before scientific interpretation, report older vs added-year support separately
and qualify mission/version/sampling changes and the new normalization.

The pilot subset coverage metadata contains no validation rows for 2024/2025,
despite training coverage there. This deserves a full split/support check; it is
not by itself proof that those years have no validation rows in the full source.
Calendar controls remain necessary for a CO₂-specific interpretation; ±CO₂ errors
alone are predictive comparisons, not causal identification. This continuation
does not automatically change the original paper protocol or authorize an
additional broad architecture sweep.

## Scheduling issue found and corrected before freeze

Three CPU jobs requested **8 slots each**, filling the **24-slot Auriga host**.
GPU job 1455 requested one slot and waited from about 22:28 until 22:48, when CPU
slots became available. This is observed scheduling contention; no separate claim
about GPU-memory availability or all queue-delay causes is made.

Under the controller lock, while state was still `pilot` and **before** any
`selection.json` froze execution settings, I changed **only `cpu_jobs: 6 → 2`**.
CPU slots/threads per job remain **8**, GPU concurrency remains **1**. This leaves
headroom for the GPU worker rather than routinely filling all 24 host slots.
Existing jobs were not canceled or restarted. The controller explicitly freezes
execution settings at confirmation selection, so no frozen policy was hot-patched.

The prior plan was backed up under `full25-logs/plan-before-cpu-reserve-20261006.json`.
The operation record is `full25-logs/cpu-concurrency-adjustment-20261006.json`, with
an ignored local copy beside the SABER/Karman receipts. It verifies that every
other plan field was unchanged:

- Before plan SHA-256:
  `51b77f6efd073a110d7c8c3f9f7d882f85b631c9026082bf217b1fa4e95c22ea`.
- After plan SHA-256:
  `e87e654d6e9eef1e2a84d913de154ecf9174c93a5345b3eff8cf76891fa1093c`.

## Remaining checks, not completion claims

Confirm array/split integrity, actual added-year and mission support, train-only
preprocessing, matched ablation execution policy and validation-only selection.
Do not combine old/new model scores as if evaluated on identical populations;
do not label a rotating-month holdout as future-year, causal CO₂, or fresh evidence
simply because its maximum date is 2025. The campaign can continue operationally;
these scientific claim gates remain open. No provider contact or SABER policy
change followed from this operational review.
