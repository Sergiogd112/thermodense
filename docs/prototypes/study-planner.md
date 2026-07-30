# Throwaway study planner prototype

This prototype asks whether a TOML study interface can make target-family expansion,
immutable shared artifacts, analysis cases, resource-class jobs, and controller-led
dynamic multi-executor dispatch understandable before science or execution exists. It
is explicitly throwaway: it is not production workflow/CLI wiring, performs no
science, writes no durable state, and never submits a job.

From the repository root, run:

```console
uv run python -m thermodense.prototypes.study_planner
```

Use `uv run python -m thermodense.prototypes.study_planner --dump` for exhaustive,
deterministic inspection and CI validation.

`configs/prototypes/study-planner.toml` holds shared defaults, repeated semantic
`[[targets]]`, target kinds, either a singular preset or explicit analyses, proxy and
cadence variants, windows, and PCMCI groups/lags. PCMCI lag strings (`h`/`d`) retain
their physical form and are converted after cadence selection; its primary and
robustness preprocessing profiles are independent cases. PCMCI expands its explicit
independence tests: `parcorr`, `cmiknn`, `gpdc`, and `gpdctorch`; their resource class
and required capability derive from the test rather than from peer analysis names.
Extension cases carry a semantic extension source ID, while full and overlap core
cases do not consume it.
`[study].id` is the readable study grouping; `--execution-id` supplies a separate
conceptual execution grouping for displayed `runs/<study>/<execution>` and
`outputs/<study>/<execution>` paths. Neither changes deterministic artifact
fingerprints. `solar_variant_rows = "exact-common-quality"` makes the F10.7/F30
matched-row policy visible in cases and driver identities.
`configs/prototypes/executor-pools.toml` separately models generic local CPU, SGE CPU,
and SGE GPU pools, capabilities, non-secret relative tick multipliers, and shared
quotas. The SGE CPU and GPU pools share a provisional 10-CPU-slot quota; GPU jobs use
an illustrative one CPU slot until capability probing determines their real cost. The
controller and executor roles are independent: a pool must be both explicitly enabled
and currently available to be a candidate. The local pool is an explicit opt-in; setting
`enabled = false` makes the initiating host controller-only, with no implicit local science
execution. Pool state includes adapter kind (`local`, `sge`, `ssh`, or `kaggle`).
Archmini is an optional disabled/unavailable SSH template reached through Tailscale, then
probed for SSH, environment, and hardware; it advertises no unverified capabilities.
Kaggle CPU/GPU templates are optional and private-only by design: they accept only
portable coarse analyses, keep generated notebooks/datasets private by default, run
analysis with internet off, and probe quotas/capabilities each session. Their minimum
job ticks exclude small preparation jobs; startup and staging overheads are provisional
illustrative values. No nonlinear capability is advertised before the frozen gate passes.
Profile endpoints/authentication remain user-local; templates contain no credentials or
hostnames/addresses/users/tokens/paths. The interactive app clears and redraws summary,
cases, artifacts, jobs, and executors
views; `n`/`p` paginate long views and tick advances only a simulated controller.

Look for readable case expansion across F10.7/F30 and Ap/Kp, full-core/overlap-core/
extension cases, shared fingerprinted artifacts, explicit PCMCI altitude nodes and
lag steps, resource requirements, and compatible executor candidates. Also check that
job IDs/statuses/attempts are shown as serializable and reconcilable controller state.
Each case terminates in an immutable `analysis-result` artifact; rendering is
deliberately downstream and outside this prototype. Candidate `job@estimate` values
mean projected completion ticks: current executor load divided by slots plus the
job's estimated duration adjusted by that pool's multiplier; resource class,
locality, availability, per-pool CPU slots, shared quota, and required capabilities
remain hard constraints. Job specs show illustrative CPU slot cost and GPU requirement.
Target-kind DAGs preserve shared acquisition/preparation while making product terminals
explicit: density diagnostics, model evaluations, and model-evaluation-derived
log-density-ratio errors feed their corresponding analysis result.

## Provisional verdict

The explicit expansion is intentional and validated for the evolving Interface:
all 416 logical cases remain independently identifiable, schedulable, retryable, and
rerunnable. Production views should filter, group, and summarize this matrix without
collapsing PCMCI-method variants or full-core/overlap-core/extension identities.
