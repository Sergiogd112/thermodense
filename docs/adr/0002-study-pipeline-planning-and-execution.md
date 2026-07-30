# ADR 0002: Study-pipeline planning and execution

## Status

Proposed.

## Context

ADR 0001's thesis-result reproduction is a historical and compatibility input. It
defines a fixed workflow interface; this ADR proposes the evolving study Interface
and would supersede that fixed workflow interface only when accepted and
implemented. No live jobs are enabled by this ADR.

The study planner prototype demonstrates the intended planning shape without
science or execution: `docs/prototypes/study-planner.md` currently expands 416
logical cases, 444 deduplicated artifacts, and 416 jobs. Nonlinear methods are
gated in that prototype.

## Decision

Science TOML and execution profiles are separate. Science TOML contains a readable
`[study].id`, shared defaults, and repeated `[[targets]]` with semantic source IDs.
It declares an explicit `density`, `model_density`, or `model_error` target kind;
products; either explicit analyses or one preset (`dependence`, `direct_trend`, or
`pcmci`); cadence, solar-proxy, and geomagnetic-driver variants; and all
full-core, overlap-core control, and extension cases. PCMCI declarations include
explicit altitude groups, primary and robustness profiles, physical lag durations
converted to steps only after cadence selection, and explicit `ParCorr`, `CMIknn`,
`GPDC`, and `GPDCtorch` variants.

One target family is selected per run/case. Every Cartesian dimension remains an
explicit logical case, including every PCMCI method and full-core, overlap-core,
and extension cases. UX may summarize and filter cases, but must not collapse
execution identity.

The `StudyPlanner` is pure and performs no computation. It validates
source/target/analysis capabilities, expands cases, creates an immutable,
deduplicated artifact DAG, assigns job requirements, and returns a complete dry
plan. Readable study and execution IDs coexist with deterministic semantic
fingerprints.

The typed target DAG has these forms:

- `density`: source acquisition/preparation to density diagnostic.
- `model_density`: model evaluation to model-density product.
- `model_error`: model evaluation plus its reference sampling frame to the
  model-error product.

An Analysis produces one canonical machine-readable result artifact. Rendering is
strictly downstream of that artifact.

Artifacts have a mobility policy of `portable`, `restricted-to-approved-pools`, or
`site-local`. A derived artifact inherits the most restrictive input policy unless
an explicit reviewed declassification/export rule applies. The
`ArtifactRepository` syncs only required content-addressed artifacts, verifies
checksums, and commits artifacts atomically.

The `ExecutionController` dispatches dynamically from a durable queue. It selects
compatible executors to optimize projected completion time using capabilities,
benchmark speed, startup and staging cost, data locality, mobility, per-pool
capacity, and shared quota. It starts on the initiating machine and can resume and
reconcile. Controller role does not imply executor role: local execution is an
explicit opt-in.

Only classified transient infrastructure failures are retried, for at most three
total attempts, preferably on another compatible executor. Scientific and
configuration failures stop. Dispatch is at least once; deterministic IDs and
atomic artifact finalization make it safe.

Each scientific job references a software-environment fingerprint, which its
executor Adapter materializes and verifies. The Adapter also records the probed
hardware and accelerator identity. A stable logical case may have result
realizations fingerprinted by both the logical case and the verified software and
hardware environment. A manifest explicitly nominates the preferred realization
for downstream rendering; realizations are never silently overwritten or reused
across environments.

The `ExecutorAdapter` Seam is real rather than an execution detail hidden in the
controller. Its Adapter implementations are local process, direct SSH,
SGE-over-SSH, and Kaggle notebook batch. The minimal Interface provides
`probe`, `submit`, `poll`, `cancel`, and `retrieve` (or equivalent operations),
without leaking scheduler details into the controller.

Initially, SGE submits one scheduler job per explicit case. Its CPU and GPU pools
share a 10-CPU-slot quota; GPU CPU cost remains provisional until probe. Archmini
is an optional direct-SSH executor reached through Tailscale, with user-local
endpoint and authentication and probe-driven capabilities. Kaggle pools are
optional: private notebooks and datasets only, internet disabled for analysis,
portable coarse-grained analysis jobs only, and session-by-session quota and
capability probes. Startup and staging are scheduling inputs. Restricted or
site-local data must not transfer to Kaggle.

`CMIknn`, `GPDC`, and `GPDCtorch` are unavailable until a frozen four-method
benchmark passes; `ParCorr` remains the initial method. Secrets, endpoints, and
execution-profile paths are user-local and outside Science TOML and the repository.

The CLI Interface is:

```console
thermodense study plan <study.toml>
thermodense study run <study.toml> --profile <name> [--execution-id <id>]
thermodense study status <study-id>/<execution-id>
thermodense study resume <study-id>/<execution-id>
thermodense study render <study-id>/<execution-id> [--realization <id>]
```

## Interface sketch

```toml
[study]
id = "hasdm-drivers"

[defaults]
cadences = ["daily"]
solar_proxies = ["f107", "f30"]
geomagnetic_drivers = ["ap", "kp"]
solar_variant_rows = "exact-common-quality"
pcmci_preprocessing_profiles = ["primary-detrended-anomaly", "robustness-seasonal-anomaly"]

[[targets]]
id = "hasdm-density"
kind = "density"                    # density | model_density | model_error
source_id = "hasdm-mauna-loa"
products = ["daily-mean-log-density", "daily-log-density-range"]
analyses = ["dependence", "direct_trend"]
extension = { name = "saber-co2", source_id = "saber-co2-cooling", overlap_window = "2002-01-01..2020-12-31" }

[[targets]]
id = "hasdm-model-error"
kind = "model_error"
source_id = "hasdm-mauna-loa-frame"
products = ["nrlmsise-00", "nrlmsis-2.1"]
preset = "pcmci"
independence_tests = ["parcorr", "cmiknn", "gpdc", "gpdctorch"]
pcmci_altitude_groups = [[325, 500], [825]]
min_lag = "0d"
max_lag = "180d"
```

The sketch is illustrative: source IDs are semantic rather than path-like, and
execution profile selection is supplied separately from this file.

## Migration inventory

The current frozen workflow Interface is implemented by
`src/thermodense/{cli,workflows,engine,ssh}.py` and `configs/thesis/`. Existing
source and derivation logic is split between `src/thermodense/downloader/`,
`decoding.py`, `msis.py`, `tle_density.py`, and the `scripts/decode_*` and
`scripts/generate_*` programs. It should migrate behind Source and Target Adapter
Seams rather than be wrapped one script at a time.

Existing analysis implementations are concentrated in:

- `global_mean.py`, `maunaloa_global_figures.py`,
  `tudelft_density_analysis.py`, and the Figure 19 programs for descriptive
  density analysis;
- `hasdm_msis_density_baseline_analysis.py`,
  `hasdm_msis_model_error_analysis.py`,
  `hasdm_msis_residual_saber_analysis.py`, and
  `tudelft_model_error_analysis.py` for model-density and model-error analysis;
- `tigramite_causal_global_mean.py`, `causal_hasdm_saber_maunaloa.py`, and the
  causal input/composite programs for PCMCI and its rendering; and
- `stats_utils.py`, `pgf_config.py`, and
  `recreate_brown_density_trend_current.py` for shared statistics, rendering,
  and direct-trend logic.

The planned Source and Target Adapter inventory covers Mauna Loa CO2, canonical
CelesTrak F10.7/F30 and Ap/Kp, global mean thermospheric density, HASDM, the TU
Delft mission files, SABER cooling/emission products, GIRO hmF2/NmF2, and
precomputed WACCM-X. Planned model Adapters cover NRLMSISE-00, NRLMSIS 2.0,
NRLMSIS 2.1, JB2006, and JB2008; WACCM-X remains a precomputed model Source rather
than an executable model Adapter. Planned Analysis Adapters cover time-series and
scatter products, frequency analysis, correlation and linear slope, activity-bin
statistics, direct trend, stationarity diagnostics, and PCMCI. Renderer
implementations consume those typed results separately.

## Deep module map

- **StudyPlanner Module** — one plan Interface hides validation, Cartesian case
  expansion, fingerprinting, target DAG construction, artifact deduplication, and
  job requirement derivation. Its Implementation is pure.
- **ArtifactRepository Module** — one artifact Interface resolves immutable
  content-addressed artifacts, applies mobility policy, performs required sync,
  and atomically commits verified results.
- **ExecutionController Module** — one control Interface owns durable dispatch,
  retry classification, reconciliation, and preferred-realization nomination.
- **ExecutorAdapter Seam** — a small execution Interface with local-process,
  direct-SSH, SGE-over-SSH, and Kaggle-notebook-batch Adapter implementations.
- **Scientific Seams** — Source, Target, and Analysis Adapter Seams declare
  capability and typed-artifact contracts; their Implementations contain source
  access and scientific computation.
- **Renderer Module** — consumes nominated machine-readable result artifacts and
  produces presentation artifacts, never upstream scientific inputs.

This package map is provisional. Prefer these deep Modules to many shallow
pass-through Modules.

```text
thermodense/studies/       StudyPlanner Interface and internal specification/catalog logic
thermodense/artifacts/     ArtifactRepository Interface and immutable storage Implementation
thermodense/execution/     ExecutionController Interface and internal routing/state logic
thermodense/execution/adapters/{local,ssh,sge,kaggle}.py
thermodense/sources/       Source and Target Adapter implementations
thermodense/analyses/      Analysis Adapter implementations and result schemas
thermodense/rendering/     Renderer Module
```

## Consequences

Planning becomes inspectable and safe before computation: `plan` can display the
complete logical case set, artifact DAG, requirements, capability rejections, and
candidate executors. Immutable fingerprints enable sharing without using a
readable path as proof of equivalence. Mobility and environment realization rules
make cross-pool reuse explicit, while a preferred-realization manifest makes
rendering deterministic.

Execution profiles can vary by user and machine without changing scientific
meaning or committing credentials. Explicit cases increase plan size and may
increase scheduler submissions, but preserve provenance and make controls and
robustness comparisons visible. The nonlinear gate deliberately limits initial
PCMCI execution to `ParCorr`.

## Deferred/open validation

- Actual cluster, Tailscale, and Kaggle capability probes.
- Frozen four-method benchmark values, resource classes, and the GPU CPU cost.
- Three-hour alignment, aggregation, and low-frequency-driver rules.
- Durable-queue and artifact persistence technology.
- Exact artifact transfer Implementation.
- Numerical cross-environment equivalence criteria.
- Production migration and parity with thesis-result workflows.

Until those validations and acceptance occur, this ADR authorizes neither live
jobs nor a production migration.
