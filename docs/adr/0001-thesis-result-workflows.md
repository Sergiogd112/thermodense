# ADR 0001: Thesis-result workflows

## Status

Accepted.

## Decision

The public reproduction interface is organised by Results-section workflow:
`global-mean`, `tudelft-density`, `hasdm-saber`, `maunaloa-msis-baselines`,
`model-errors`, `synthesis`, and `all`. Every workflow uses the fixed ordered
chain `acquire`, `prepare`, `derive`, `analyse`, and `publish`.

Configurations are frozen, checked-in TOML files. `reproduce` uses the frozen
`thesis` run identity; `refresh` always creates a new sortable UTC run identity
with microseconds and a random suffix. Ignored checkpoint JSON under
`runs/<workflow>/<run-id>` records configuration, implementation, input/output,
environment provenance, timestamps, and status. A success is reused only when
all recorded provenance still matches. This supports access-conditional,
code-only reproduction without distributing restricted or large data.

Artifacts are separated into ignored `data/sources`, `data/prepared`, and
`data/products`; checkpoints are ignored under `runs`. The canonical committed
publication destination is `thesis/figures/results`. Figures are outputs, never
upstream inputs.

Local execution is the reference adapter. An optional key-authenticated SSH
adapter reads profiles only from a user-local or explicit TOML file, synchronises
to an isolated remote run directory, executes without shell interpolation, and
returns run outputs/checkpoints. No profile or secret is committed.

Migration is incremental: unavailable stages fail explicitly rather than
silently doing nothing or claiming reproduction. A migrated workflow must meet
numerical and visual parity with its prior thesis result before becoming
available.

The global mean thermospheric density and HASDM analyses may use causal
discovery where the thesis specifies it. TU Delft density analysis and TU Delft
model log-density-ratio error do not use causal discovery because their sampling
and mission structure require a dedicated graph.

## Consequences

The initial foundation exposes planning and provenance before it exposes any
migrated thesis computation. Users must still obtain restricted source data via
the documented access process. SSH artifact selection and remote data sync are
activated only as stages migrate; this foundation does not claim current remote
data execution.

## Public-release addendum (2026-07-30)

The original decision remains historical. Its consequence that
`thesis/figures/results` is a committed publication destination is superseded
for the public code-first release: thesis sources, presentation materials, and
generated outputs are privately archived and are no longer tracked. Runtime
data, prepared data, products, runs, and generated outputs remain ignored.
The package-level publication path now resolves to the ignored
`outputs/figures/results` tree.

This addendum does not choose the future pipeline interface or alter the
incremental migration decision. Planning and status remain the supported public
interface until individual stages are migrated and parity-checked.
