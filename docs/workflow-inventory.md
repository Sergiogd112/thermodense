# Workflow inventory and cleanup guardrails

This inventory separates the accepted reproduction interface, frozen scientific
experiments, and current isolated analysis modules. A filename containing
`prototype` or an output field saying `production` is not an adoption decision.

## Preserved baseline

Before cleanup on 9 September 2026, 164 local source/configuration/documentation
files were copied and checksum-verified into the ignored, source-only snapshot:

`data/products/source-snapshots/pre-cleanup-20260909T180928Z/`

Snapshot manifest SHA-256:
`e56432ea2c19061da522af2ff676357c1c88c6fda3770f610726dc234d540750`.
Git HEAD was `17e244c05397a89182d245e3fac447c7d20f804b`. The snapshot includes
uncommitted source work; it is not a Git commit. It excludes runtime/model data,
Git internals, account configuration, symlinks and external provider software.
Existing uncommitted user changes remain untouched.

Completed global-activity artifacts and the failed raw-36 run retain their remote
isolated source directories. Those directories must not be hot-patched as part
of cleanup. Result/input hashes and archived source identities remain historical
facts, even if a newer implementation is introduced later.

## Inventory

| Workflow/module | Status | Scientific contract | Recoverability |
|---|---|---|---|
| `cli.py`, `workflows.py`, `engine.py`, `checkpoints.py`, `configs/thesis/` | Accepted reproduction foundation under ADR 0001; unavailable stages still fail explicitly | Frozen Results-section workflows; no claim that every analysis is migrated | Provenance-checked stage reuse; not a universal solver checkpoint |
| `benchmarks/pcmci_methods.py` | Frozen synthetic benchmark | Published synthetic cases/settings | Preserve recorded identities and results |
| `benchmarks/pcmci_real.py` and its gates/cache | Frozen daily scientific reference | Registered preprocessing/F10.7 sensitivity matrix; method-specific lag/resource gates | Existing cache/stage capabilities only; verify the specific case |
| `benchmarks/raw_3hour_input.py`, `raw_3hour_pcmci.py` | Isolated raw-36 experiment; evidence not qualified | Raw 36 nodes, native 3-hour cadence, 1,440 lag steps; no interpolation | Job 1139 hit its 72-hour limit with no final matrices; in-memory fitting is **not resumable** |
| `benchmarks/global_activity_input.py`, `global_activity_pcmci.py` | Completed isolated retrospective association experiment | Ten density altitudes + four drivers; raw F10.7 fit, centered-81 target-date bins, daily 180-lag analysis | CI trace is diagnostic, not a resume cache; saved raw matrices can support re-export |
| `benchmarks/global_activity_polynomial.py` | Completed bounded predictive benchmark | Same daily input; degrees 1/2/3; train-only scaling, validation selection, fixed test | Trial/epoch checkpoints support compatible resume; checkpoint existence alone does not prove compatibility |
| `benchmarks/global_activity_comparison.py` | Verified derived-result replay | Hash-bound models and input; paired same-support predictive comparison | Cheap deterministic replay; no fitting or remote job required |
| `scripts/prototype_polynomial_var*.py` and direct-density/SABER prototypes | Exploratory/frozen experiment implementations | Preserve each named input/lag/scaling policy, including legacy spline-F10.7 differences | Varies by prototype; never infer resumability from a checkpoint directory name |

## Cleanup order

1. Preserve source and experiment identities; classify before moving/deleting.
2. Reuse the existing streaming checksum foundation for shared artifact integrity
   and immutable derived-output finalization. Distinguish verified outputs from
   solver state and from unverified checkpoint candidates.
3. Expose named, immutable scientific policy descriptions and validate against
   them. Do not replace different experiments with one universal default.
4. Consolidate read-only execution observation: job state, quota/deadline,
   accounting, artifact integrity and recovery capability. Submission, cancellation,
   retries and cloud deployment remain separate explicit actions.

The first integrations must not change the hashed global-activity fitting/input
modules or frozen daily/legacy implementations. Shared validation can initially
serve the comparison replay and read-only monitor, two real consumers. Existing
artifact layouts remain readable and tampered artifacts must fail closed.

## Adoption and retirement

Require numerical parity, relevant artifact-schema tests, and recorded scientific
scope before promoting or replacing a workflow. For reproduced figures, require
visual parity too. Retire an entry only after identifying its replacement and
preserving the historical source/results; bulk renames and deletions are not a
first cleanup step.

ADR 0001 is **Accepted**. ADR 0002's broader StudyPlanner/ArtifactRepository/
ExecutionController design is **Proposed** and is not automatically adopted by
this cleanup. Cloud capability, cost and restricted-data mobility are separate
execution decisions, not consequences of having student credits.
