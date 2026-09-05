# J-PCMCI+ all-density-altitudes decision note

Date: 2026-08-09

## Purpose

This note records the recommended issue #18 decision for extending causal discovery across
the available fixed-altitude density products. It separates facts established by
the cited J-PCMCI+ literature and Tigramite implementation from project
decisions. It does not establish thermospheric validity of J-PCMCI+, an
intervention effect, or a causal design for the TU Delft satellite record.

## Data inventory and current contract

### Source-established / locally verified inventory

Local inspection of
`data/decoded/orbit_derived_global_mean/orbit-density-ds03-density-values.parquet`
shows a *wide* table: 19,328 timestamps and
ten aligned density columns at 250, 275, 325, 375, 400, 425, 475, 525, 550, and
575 km. Thus each altitude starts with the same calendar-row axis, subject to
its own finite-value and analysis masks. The record does **not** contain
19,328/10 observations per altitude.

All 26 annual files matching `data/decoded/hasdm/HASDM_*_merged.parquet` have
the same fixed grid of 27 altitude levels every 25 km from 175 through 825 km.
Existing preparation scripts pivot all levels, although the legacy
causal analysis subsequently selects five altitude quantiles. At daily cadence
it has approximately 9,333 dates before the common-support decision described
below. These are project-local inventory facts, not claims about the external
products' construction.

The files under `data/decoded/tudelft/` are materially different: they have
continuously time-varying orbital
altitude, mission-specific epochs and sampling, and tens of millions of
samples. Local inspection gives these altitude ranges (km): GOCE 224.5–294.6;
CHAMP 248.0–506.0; GRACE-A 305.7–537.5; GRACE-B 332.7–538.3; GRACE-FO
450.7–537.9; Swarm A/C about 422.7–527.6; and Swarm B 484.9–545.6. A mission
label alone cannot turn these trajectories into aligned altitude domains.

### Project decision / recommendation

`CONTEXT.md` remains the controlling contract: a configured altitude group
expands to a separate PCMCI+ graph, density channels remain separate nodes and
are never averaged, and TU Delft density and model-error targets are causally
ineligible until a mission/sampling graph is resolved. The primary
all-altitude design is therefore overlapping groups plus a companion
driver+density graph for every individual altitude. J-PCMCI+ is a prototype,
not a replacement for that design. TU Delft remains ineligible.

| Product | Primary graph set | Companion graphs | Status |
| --- | --- | --- | --- |
| Global mean | Five overlapping adjacent-inventory groups | 10, one per altitude | primary coverage |
| HASDM | Nine overlapping contiguous groups | 27, one per altitude | primary coverage |
| J-PCMCI+ | Small fixed-altitude diagnostic only after simulation | compare with both sets | prototype |
| TU Delft | none | none | causally ineligible |

“Adjacent-inventory groups” is deliberate terminology for the global product:
its available altitude spacing is irregular, so those groups are not physically
contiguous layers. The exact global groups are `[250,275,325]`,
`[325,375,400]`, `[400,425,475]`, `[475,525,550]`, and `[550,575]` km. The
exact HASDM contiguous groups are `[175,200,225,250,275]`,
`[250,275,300,325,350]`, `[325,350,375,400,425]`,
`[400,425,450,475,500]`, `[475,500,525,550,575]`,
`[550,575,600,625,650]`, `[625,650,675,700,725]`,
`[700,725,750,775,800]`, and `[775,800,825]` km. Each listed channel is a
separate density node; no group produces an altitude average.

The overlaps are useful anchors for diagnosing changes in selected links as
conditioning sets and altitude coverage change. They are not independent
replications: dates, drivers, shared density variability, and model shocks
overlap. Results across altitude and group should consequently be reported as
dependent evidence, including explicit ambiguity or conflict marks.

## What the two graph concepts do—and do not—mean

### Project decision / recommendation

Vertical-coupling adjacency means placing multiple altitude density channels as
separate system nodes *in the same graph*. Only this construction can represent
candidate density-to-density links such as `density_325 → density_375`, subject
to the causal-discovery assumptions. It is the reason for the grouped PCMCI+
graphs.

An invariance-constrained J-PCMCI+ graph has a different estimand. One
altitude-defined dataset/domain has a single common density system variable,
and altitude is its `space_context`. It contains no separate `density_a` and
`density_b` nodes and hence no density_a→density_b vertical edges. Its result
must be described as a **graph constrained to be shared across
altitude-defined domains**, not proof of invariance and not a vertical-coupling
result. Neither graph type identifies intervention effects merely from a
directed edge or a missing edge.

## J-PCMCI+ basis and limits

### Source-established

Günther, Ninad, and Runge introduce J-PCMCI+ for causal discovery from multiple
time-series datasets with contexts in the UAI 2023 PMLR paper, pp. 766–776,
including its main paper and supplement.[^gunther-main][^gunther-pdf][^gunther-supp]
Their Eq. (1), Definitions 1–5, Algorithm 2, and Theorem 2 define the
context-augmented structural causal model, target graph and consistency result.
The authors explicitly give spatial context as an example (including altitude).

The stated assumptions 0–4 matter here. Context variables are exogenous to the
system; the method assumes no latent context confounding between observed
context and system variables and no latent system confounding. A spatial
context is constant within a dataset and has only contemporaneous links to
system variables; a time context has the same realization across datasets.
The finite-sample discussion also warns that the asymptotic result requires both
the number of datasets and their length to increase. Treating a fixed, small
time series as if a growing number of altitude domains supplied ordinary
independent evidence can inflate false positives. The earlier PCMCI+ paper
establishes the single-dataset conditional time-series discovery framework; it
does not remove these J-PCMCI+ multi-context assumptions.[^runge20]

The official Tigramite implementation exposes this design through a
`DataFrame` with `analysis_mode='multiple'`, a dictionary of arrays with fixed
node count `N`, and optional `time_offsets`.[^dataframe-source] `JPCMCIplus`
is constructed around the DataFrame/PCMCI configuration, a conditional
independence test, and `node_classification`, then executed with
`run_jpcmciplus`.[^jpcmci-source] The documented roles are `system`,
`time_context`, `space_context`, `time_dummy`, and `space_dummy`. The official
tutorial also requires dummy-aware `ParCorrMult` handling when dummies are
included.[^jpcmci-tutorial]

### Project decision / recommendation

Altitude may be supplied as `space_context` only in the narrowly defined
one-altitude-per-domain prototype. Do not automatically label all drivers
`time_context`. A driver may receive that role only after an accepted causal
graph defends that it is shared and aligned across domains and defends the
exogeneity and context-confounding assumptions. Otherwise it remains a system
variable in a different design, or is excluded from the J-PCMCI+ prototype.
No cited source validates pooling thermospheric altitudes, trajectories, or
missions under these assumptions.

## Support, preprocessing, and inference requirements

### Project decision / recommendation

Prepare every graph by recording its exact calendar, per-node masks, and
complete-support decision. Perform stationarity qualification separately for
each registered preprocessing profile and graph; retain the context-defined
calendar rather than silently aligning through cross-altitude interpolation.
No cross-altitude interpolation is allowed. Report finite spans, missingness,
imputation masks if bounded gap interpolation is used, and rejected rows.

For a grouped graph, common support is a property of that group and its chosen
drivers, not a reason to rewrite another group's sample. For a per-altitude
graph, preserve the altitude's own accepted calendar while retaining sufficient
metadata to compare it with the corresponding group. A link that changes when
the calendar or conditioning set changes is an ambiguity to report, not a
result to reconcile by selecting the preferred run. In particular, a shared
wide-table row axis does not imply identical effective samples after masks,
driver availability, preprocessing, and stationarity spans are applied. Keep
the raw source calendar, analysis calendar, and exclusion reasons in result
provenance so that overlap-induced dependence and coverage changes remain
auditable.

A huge all-altitude graph may be run only as a robustness diagnostic after the
grouped design. High dimension and collinearity make it an especially weak
primary discovery design, and null edges do not establish causal absence.
Evaluate negative controls, calendar/time blocks, anchor stability across
overlapping groups, and marked conflicts before interpreting a link.

Define the inferential family before execution, not from the links that survive
PCMCI+ selection. The narrow recommended choice is one primary family per
product: all grouped-ParCorr links under the primary timing/preprocessing
profile, spanning group membership, source, target altitude, and physical lag.
Per-altitude companion graphs, alternate preprocessing/timing profiles,
nonlinear methods, and huge-graph runs remain descriptive sensitivity evidence.
Any confirmatory claim combining products, graph types, profiles, or methods
must instead expand the family explicitly across product, graph/group, method,
profile, source, target, and lag. Ordinary BH or BY adjustment of
post-selection p-values does not by itself solve this global-inference problem.

For simultaneous statements, use a project-level calendar-block resampling
wrapper that preserves the complete altitude vector, domain boundaries, masks,
and driver rows for each resampled date, reconstructs the Tigramite `DataFrame`,
and reruns the full selection procedure for every replicate. The current
Tigramite `DataFrame` bootstrap path does not support this multiple-dataset use,
so this is not an ordinary API switch.[^dataframe-source] Use a predeclared max
statistic for simultaneous control, or label selection frequencies explicitly
as descriptive stability frequencies. Report sensitivity to block length.

Operational resource gates (for example, a time or memory budget) decide
whether an execution proceeds. They are not scientific-validity gates and must
not convert an otherwise unqualified run into evidence.

## Dummy-variable memory and implementation risk

### Source-established

The official tutorial constructs a `T × T` one-hot time dummy and horizontally
stacks it into each dataset to model latent time contexts, while using
dummy-aware conditional-independence handling.[^jpcmci-tutorial] Omitting that
dummy changes the latent-context protection; it is not an interchangeable
performance optimisation.

### Project decision / recommendation

For global `T=19,328`, one dense float64 `T × T` dummy is approximately
2.99 GB (`19,328² × 8` bytes). Naïvely copied for `M=10` domains it is about
29.9 GB before density, drivers, masks, Python objects, or library copies. For
HASDM, the daily `T≈9,333` estimate is about 0.70 GB per dummy and 18.8 GB
across 27 domains—but only after common-support verification. These are
lower-bound implementation arithmetic, not measured peak RAM. Profile the
actual array layout, copying behavior, `vector_vars`, dummies, and conditional
independence test; do not infer feasibility from the arithmetic alone.

## Staged implementation plan and stop conditions

### Project decision / recommendation

1. Implement the grouped and per-altitude PCMCI+ expansion first, preserving
   exact calendars, masks, channel identities, provenance, and the registered
   preprocessing/lag variants.
2. Simulate data matched to candidate J-PCMCI+ `M`, `T`, missingness, and
   common-support pattern, with a known graph. Include both truly shared
   mechanisms and altitude-varying mechanisms. Test recovery, false positives,
   the declared multiplicity procedure, API inputs, `vector_vars`, dummies, and
   measured memory.
3. Only if that simulation passes, run the smallest useful fixed-altitude
   diagnostic (three to five domains), with one altitude per dataset, then
   compare it to the corresponding grouped and per-altitude results.
4. Expand only after the diagnostic is stable across blocks and explicitly
   reported assumptions remain defensible.

Stop the J-PCMCI+ prototype if the installed API cannot represent the required
arrays/roles/dummies; dummy-aware testing or memory profiling fails; simulated
known links are not recovered at the planned support; false positives or
altitude-varying mechanisms are spuriously presented as shared; common support
or driver alignment is inadequate; or the required context causal graph cannot
defend exogeneity and absence of the specified latent confounding. A successful
resource gate alone does not override any stop condition.

TU Delft may be reconsidered only after a mission/sampling causal graph records
mission, instrument and processing, altitude trajectory, local time, cadence,
epoch, selection arrows, common variables and support, and a measurement
model. It must show why pooled observations meet the relevant assumptions;
mission label alone is not a valid fix.

## Sources

[^gunther-main]: Günther, W., Ninad, U., and Runge, J. (2023), “Causal discovery for time series from multiple datasets with latent contexts,” *Proceedings of the 39th Conference on Uncertainty in Artificial Intelligence*, PMLR 216, 766–776, [landing page](https://proceedings.mlr.press/v216/gunther23a.html).
[^gunther-pdf]: Günther, Ninad, and Runge (2023), [main paper PDF](https://proceedings.mlr.press/v216/gunther23a/gunther23a.pdf), especially assumptions 0–4, Eq. (1), Definitions 1–5, Algorithm 2, Theorem 2, and the finite-sample discussion.
[^gunther-supp]: Günther, Ninad, and Runge (2023), [supplement](https://proceedings.mlr.press/v216/gunther23a/gunther23a-supp.pdf).
[^runge20]: Runge, J. (2020), “Discovering contemporaneous and lagged causal relations in autocorrelated nonlinear time series datasets,” *Proceedings of the 36th Conference on Uncertainty in Artificial Intelligence*, PMLR 124, [paper](https://proceedings.mlr.press/v124/runge20a.html).
[^jpcmci-source]: Tigramite, [`jpcmciplus` implementation source](https://jakobrunge.github.io/tigramite/_modules/tigramite/jpcmciplus.html).
[^jpcmci-tutorial]: Tigramite, [official J-PCMCI+ tutorial notebook](https://github.com/jakobrunge/tigramite/blob/master/tutorials/causal_discovery/tigramite_tutorial_jpcmciplus.ipynb).
[^dataframe-source]: Tigramite, [`DataFrame` implementation source](https://jakobrunge.github.io/tigramite/_modules/tigramite/data_processing.html).
