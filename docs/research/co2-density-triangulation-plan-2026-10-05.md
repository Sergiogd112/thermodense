# A bounded CO₂–thermospheric-density triangulation study

Date: 2026-10-05. **Research plan, not authorization to run new experiments or
bulk transfers.** This responds to the clarified goal of complementary evidence
about CO₂, not just repackaging the thesis descriptively. It updates the earlier
scope-reset review's provisional advice to park WACCM-X: the provided regional
archives have now been located and merit a **bounded qualification step**.
Historical plans, frozen runs and GitHub decisions remain unchanged.

**Preparation update:** the Karman benchmark is now complete and its small,
inconsistent paired gains are documented in the
[verified closeout](karman-bounded-closeout-2026-10-05.md). The
[data-qualification pass](co2-paper-data-qualification-2026-10-05.md) found a
material unbounded-nearest-scan issue in the legacy SABER local product; its
flux column is not a ready fixed-region diagnostic. Use the concise
[protocol](../co2-paper-protocol.md) as the current working view. The dated
progress observations below remain historical snapshots, not current status.

## Executive recommendation

Pursue one question:

> **How much evidence for a CO₂-related density contribution survives realistic
> solar/geomagnetic, calendar-time and sampling controls, and is its altitude/
> solar dependence compatible with independently designed physical perturbation
> experiments?**

Use **three evidence pillars**, not a collection of interchangeable causality
tests:

1. **Observational constraints:** strengthen the existing long-term trend and
   product/model-error analysis, with explicit time-confounding and uncertainty.
2. **Predictive contribution:** complete the existing Karman ±CO₂ experiment,
   then, only if justified, make one time-proxy-controlled comparison using a
   fixed architecture and a separately frozen evaluation protocol.
3. **Physical consistency/attribution:** use the new regional WACCM-X history
   for multidecadal/vertical fingerprints; use existing controlled-CO₂ model
   experiments as the intervention anchor where their designs and outputs match
   the question. A historical single run is **not** its own fixed-CO₂ control.

One small conditional-dependence graph may be a supplementary diagnostic
inside pillar 3. It is not a fourth major programme. New architectures, new
nonlinear CI algorithms, full-volume graphs and new GCM runs are not the next
step.

The physically plausible existence of CO₂-driven cooling/contraction already
has substantial literature support [S3–S5]. A publishable contribution here is
**which evidence is informative beyond secular proxies, what conditional
magnitudes/uncertainties are defensible, and why methods or products disagree**—
not merely rediscovering a negative correlation.

## 1. Current evidence and the most valuable thesis limitations

The July 13 thesis already has a coherent multi-product argument: strongest
solar dependence, secondary disturbed-time geomagnetic structure, negative
product-specific adjusted trends, and conditional anomaly graphs. Its Overview,
§§3.6–3.9, §5.6 and §6.1 correctly distinguish long-term trends from detrended
short-/medium-lag dependence. Those parts should be retained, not rebuilt.
The existing six-figure paper draft is a useful starting artifact, not a final
proof of the quantitative claims.

Highest-value limitations to address:

| Limitation | Why it matters | Bounded remedy |
|---|---|---|
| CO₂ and historical time are strongly covarying | The same density history can fit CO₂ or another secular proxy; algorithms cannot manufacture exposure overlap | Compare physical CO₂ with matched clock/proxy controls and report identification failure honestly |
| Solar-control specification and interval dependence | Inadequate solar removal or optimistic daily intervals can create a convincing-looking secular signal | Predeclare a small solar/geomagnetic control sensitivity, date-block uncertainty and sample windows |
| Mission/altitude/local-time sampling | Changes in satellite coverage can mimic changes in density | Preserve mission strata, recover mission identifiers before new ML/mission-holdout claims, distinguish within-product from cross-product results |
| Conditional graphs and secular attribution have different estimands | Three-year detrending removes much of the slow CO₂ signal; null graph edges do not refute it | Keep existing graphs timescale-specific; do not demand longer graphs as the default “fix” |
| Reference/model independence and provenance | HASDM is assimilative and models share fitting/source assumptions | Retain the calibration/source-dependence ledger and figure-to-artifact trace |

The checked thesis trends near 400 km are −4.94%/decade for the global-mean
record and −5.76%/decade for the Mauna Loa HASDM subset, with their reported HAC
intervals below zero. Their saved scalar conversions and rounded thesis prose
reconcile in the preceding review. They remain **solar-adjusted calendar-time
diagnostics**, not estimates of a CO₂-only effect or independent replicated
climate measurements. This plan does not reclassify them.

## 2. New dataset: located, useful, but not automatically attribution-ready

See the detailed [verified inventory](waccm-x-regional-inventory-2026-10-05.md).

### What was actually found

The provider is **UKSSDC / SERENE at the University of Birmingham**, distributing
**WACCM-X 2.1 / CESM 2.1.3** long-history simulation output. The recovered account
directory is accessible by **FTP**; SFTP was not the protocol verified.

- **74 regional yearly archives**, total **152.17 GB**, nominally 1950–2024,
  with **2022 absent**. 1950 starts January 2; 2023 packaging/coverage requires
  separate inspection. Endpoint/member-size checks are not full timestamp/QC
  verification.
- A representative **2014-01-01** member opened successfully; its local-header
  CRC and SHA-256 were verified. It has **56 variables**, including bookkeeping,
  **24 hourly timestamps, 126 midpoint levels, 127 interfaces, 5×5 grid points**.
- Actual latitudes are approximately **16.105–23.684°N**, with **1.895° spacing**;
  longitudes **200–210°E**, **2.5° spacing**. The nearest earlier Mauna Loa column
  is 19.8947°N, 205°E. This is a small regional neighbourhood, not global spatial
  coverage or 25 independent experiments.
- The representative top varies from **559.6 to 681.2 km geopotential height**.
  Pressure/hybrid levels move in height; full-history fixed-altitude support is
  not established. Do not assume every timestamp reaches 500, 700 or 825 km.
- Key fields: density `DEN`, neutral temperature `T`, local `CO2`, boundary
  `co2vmr`, `NO`, `O/O2/N2/H/N/He/Op`, `QJOULE`, winds `U/V`, height `Z3`,
  electron/ion state, solar and geomagnetic drivers, and other boundary GHGs.
  **No direct species-resolved CO₂ IR cooling tendency is in the inspected
  regional schema.** Concentration is not radiative cooling.

### Why it is valuable

Its long record and joint vertical state can test whether density, temperature,
pressure-surface heights and composition show **compatible climate fingerprints**
under the same simulated forcing history. Neighbouring cells support spatial-
sampling sensitivity. They do not supply independent greenhouse-gas trajectories.
Hourly data allow explicit aggregation and storm/local-time diagnostics rather
than forcing all questions into daily mean/range summaries.

### Why it is not a clean observational or intervention dataset

1. It is a **free-running historical simulation without specified dynamics**:
   calendar-date weather need not match real weather. Compare statistical/
   climatological behaviour; do not interpret daily mismatch as model accuracy
   against local observations without an appropriate matching design.
2. CO₂, other GHGs, solar/geomagnetic histories, aerosols and lower-boundary
   conditions change together. No matched fixed-CO₂ control was found.
3. The three runs begin in 1949, 1973 and 1997 with one spin-up year. Inspect
   the stored segment transitions around **1974/1998**; compare trends within
   segments before treating a joined history as homogeneous. Pre-1963 particle
   forcing is partly reconstructed by repeating a later solar cycle.
4. Provider says **helium and density were added post-run**, using scaled
   NRLMSISE-00 helium and model constituents. This creates an MSIS-related
   dependence, especially relevant aloft. WACCM-X `DEN` is not wholly independent
   evidence against MSIS.
5. Important metadata need clarification: boundary `co2vmr` is ~0.0003963 in
   the sample despite the website's ppm label; `EDens` is labelled `cm^3` rather
   than inverse volume; `He` exceeds 1 despite a mol/mol label. The provider's
   displayed constituent-mass conversion also has a dimensional inconsistency.
   None proves the numerical dataset is wrong, but no units/definitions should
   be “fixed” by assumption.
6. `DEN`'s dimension order differs from most state fields. Use named dimensions.
   Resolve time-bound/averaging and geopotential-versus-geometric conventions,
   pressure reconstruction and native-top coverage before computing trends.

**Verdict:** high value for a bounded climatological/physical-consistency branch;
conditional usefulness for causal discovery; **insufficient by itself for isolated
historical CO₂ attribution or complete radiative mediation**.

## 3. Exactly what each method can claim

| Evidence type | Recommended implementation | Estimand / legitimate statement | Not established |
|---|---|---|---|
| Association | Existing density/driver and model-error summaries | Product-specific relationships and their sampling dependence | Physical causal direction |
| Long-term conditional association | Compact predeclared trend model with solar/geomagnetic controls, seasonality, sampling and restart checks | Adjusted calendar-time trend; conditional CO₂ coefficient under the chosen specification | A deconfounded CO₂ effect merely because controls are present |
| Prediction | Karman feature ablation with matched data/training and calendar controls | Held-out loss change from making CO₂ features available | Effect of changing atmospheric CO₂ |
| Temporal predictive causality | Restricted forecast comparison, only if explicitly adopted | Past information improves future prediction under stated information availability | Intervention/physical attribution; Karman density estimation is not automatically such a forecast |
| Conditional dependence / causal discovery | At most one small physically motivated PCMCI+ graph | Conditional lagged/anomaly links, with assumptions, stationarity/support and multiple-testing limits | Secular cooling after removing secular variation; hidden-confounder-free causation |
| Physical attribution within a model | Verified matched CO₂ perturbation experiments with other forcings held fixed | Model-conditional density contrast under an explicit intervention | Exact real-world magnitude, independence from model assumptions |
| Mechanism / mediation | Controlled runs plus cooling/temperature/height/composition diagnostics | Mechanistic consistency; stronger pathway evidence when direct budgets and identification exist | Causal mediated fraction from a historical regression of endogenous states |

Do not count ParCorr, CMIknn, another neural architecture and another solar
proxy as four independent proofs. They may be useful robustness checks, but
answer related questions on shared inputs.

## 4. Karman: much of the proposed comparison already exists

### Published methodology and the local adaptation

Acciarini et al. [S2], §§2.2.1–2.2.2 and Appendix A, train feed-forward density
models using empirical-model drivers, geometry and time features; the OMNI
variant adds **48 hours at hourly resolution**. Target is
`ln(rho * 1e12)` with MSE, Adam, learning rate `5e-5`, approximately `1.37e-8`
weight decay and five seed repeats. Main evaluation uses rotating held-out
months; Appendix A separately tests an unseen satellite. Mean MAPE across
seeds and the paper's model-ensemble discussion are not equivalent to treating
five prediction histograms as one per-row ensemble prediction.

Crucially, **calendar year is already an input** in the published architecture
description and local `BASE_LINEAR`. Absence of explicit CO₂ does not mean
absence of secular information.

The current adaptation already compares paper-style static/OMNI and a shallow
residual OMNI model with/without two CO₂ features: **previous-day observed CO₂
and trailing-365-day CO₂ mean**. Only those columns are removed in the control.
Rows, normalization, optimizer, seeds and backend assignment are matched;
finalists were selected on pilot validation before canonical test evaluation.
This is an architecture/method adaptation with different prepared data and a
bounded training budget—not a numerical reproduction of the paper's scores.

### Completed-fit validation results—not final test claims

| Architecture | No CO₂ mean best log-MSE | With CO₂ | Relative reduction from adding CO₂ |
|---|---:|---:|---:|
| Paper static, two hidden layers | 0.03877018 | 0.03891824 | **−0.38%** (slightly worse) |
| Paper OMNI | 0.02902324 | 0.02859468 | **1.48%** |
| Residual OMNI depth 2 | 0.02862425 | 0.02816980 | **1.59%** |

Each row is five completed seeds; these are averages of validation-selected
checkpoint losses. Gains are modest and not uniform across individual seeds.
Canonical evaluation was **22/30 complete** at this snapshot; its final results
must be frozen before closing the current study. No significance or causal
statement follows from this table alone.
At the final read-only plan review, evaluation had advanced to **28/30** with
collector integrity reported true and no warnings; the completed validation
table above was unchanged. These dated progress observations are not completion
attestations or a substitute for the final canonical result artifact.

### The next necessary comparison is not another architecture

Freeze one architecture, preferably the paper-style OMNI baseline for the
paper-method question; retain residual-2 as a secondary existing result.
On one new protocol, compare:

| Cell | Features |
|---|---|
| A | Physical drivers + geometry/seasonality + existing calendar representation |
| B | A + measured CO₂ features |
| C | Physical drivers + geometry/seasonality + **predeclared flexible calendar representation** |
| D | C + measured CO₂ features |

The key quantity is **D versus C**, not simply B versus A. C tests whether a
finite-capacity learner benefits from a better secular coordinate even without
CO₂. Match input/parameter budget using time-derived controls where material;
do not let tuning budgets differ by cell.
Even a positive D–C result establishes incremental prediction only **relative to
that chosen calendar basis and evaluation distribution**. It may still reflect
unmeasured secular structure, finite-model approximation or sampling—not a
uniquely identified physical CO₂ contribution.

- Calendar basis complexity is fixed before the new evaluation; do not search
  splines until a desired CO₂ result appears.
- As a cheap diagnostic, compare measured CO₂ to a **training-only predicted
  secular/seasonal CO₂ proxy** from time. Similar benefit weakens claims of
  CO₂-specific explanatory information, although it cannot disprove a true
  physical effect carried by that trend.
- Examine residual/seasonal CO₂ information separately from long-term growth.
  A null anomaly test does not refute a secular effect.
- Do not globally shuffle trending CO₂ or use arbitrary future CO₂ as a
  supposedly valid null. Time-preserving placebo/surrogate checks require their
  own null assumptions; otherwise label them diagnostics, not calibrated tests.
- Hold architecture, sampling and precision policy fixed. Report both log-MSE
  and original-paper MAPE, signed bias, full-tail spread and predeclared strata.

**Protocol/selection gate:** the current held-out test has already been observed
and is not a fresh selection set. For prospective confirmation, predeclare
chronological validation/test periods, fit every transformation using training
data only, and buffer temporal dependence/history at split boundaries. A later
period, such as eligible post-2020 mission data not used for this model tuning,
is a candidate—not presumed sealed or analysis-ready. Recover mission identity
before claiming mission-specific replication or holding a mission out. If no
new evaluation support is available, label the extension exploratory.

Before calling that comparison confirmatory, record the evaluation dataset's
identity and prior use, show that its model-comparison outcomes did not inform
feature/calendar/hyperparameter selection, lock the protocol, and inspect its
scores only after fitting/selection is complete. General source-data exploration
is not the same as an untouched model-comparison outcome. Check feature/mission/
altitude/solar/CO₂ support separately: chronological evaluation may deliberately
test extrapolation. Report that as such rather than assuming on-support
generalization or attributing extrapolation failure to absent physical CO₂ effects.

First use a cheap structured regression/GAM comparator to assess whether the
calendar-control question is worth neural retraining. If it is, the ceiling is
**four cells × two pilot seeds = eight new pilot fits**, with one architecture
and fixed budget. Confirm at five seeds only for a prespecified scientific
contrast that survives. This is a proposed ceiling, not a launched sweep.

## 5. Observational and empirical-model branch: improve identification discipline

Choose **one primary long-term observational product**—the orbit-derived
global-mean record after exact source/version verification. Use HASDM and
mission-stratified TU Delft as differently limited comparisons, not replicate
measurements of that global quantity. Keep the existing paired JB2006/JB2008
and MSIS error diagnostics as model/reference context.

Use the existing adjusted trend as a baseline. Predeclare only a few sensitivities:

1. Add a geomagnetic term and seasonal controls to the quadratic-solar model;
   permit a small prespecified CO₂-by-solar interaction if physically motivated.
2. Compare linear versus one fixed flexible calendar specification and matched
   sample periods; inspect rank/conditioning and stability of a joint CO₂/time
   model. An unstable coefficient is an identification result, not a reason to
   drop time until significance appears.
3. Use dependence-aware intervals and one block/bandwidth sensitivity, plus
   mission-offset/sampling checks where applicable. HAC corrects covariance
   under its assumptions; it does not remove confounding or model misspecification.

Report %/decade for historical trends. Report % per specified CO₂ contrast only
when its conditional assumptions and exposure range are explicit. Conditioning
on flexible time may absorb a genuine slow CO₂ effect; conversely, omitting
time may attribute unrelated drift to CO₂. **If CO₂ is almost a deterministic
function of time on this record, a unique secular coefficient is not robustly
identified without additional structure or intervention evidence.**

Do not make 11-year span, one small p-value or a narrow nominal Fisher interval
the whole reliability criterion. Count independent dates/events, actual coverage,
solar-cycle representation and instrument/mission transitions.

## 6. WACCM-X branch: two tasks, one optional graph

### 6.1 Historical fingerprints, not invented counterfactuals

After data qualification, begin with **one nearest-grid column**, daily climate
summaries and fixed-height densities at **300/400 km**, adding 500 km only if
native support and helium sensitivity permit. Retain hourly/native-level data
for QC; verify height/time conventions before interpolating. Do not extrapolate
above the model top or fill the missing 2022 year.

Start with a single post-restart, pre-gap window such as **2000–2021** if its
timestamp/QC manifest qualifies. This overlaps HASDM and avoids mixing the
1974/1998 restart boundaries into the first pilot. It still contains evolving
forcings and is not a controlled experiment. Broader historical segments are
sensitivity expansions only after the core answers its question.

Test a small physical fingerprint: negative fixed-altitude density change,
compatible neutral-temperature/pressure-surface-height changes, and solar/altitude
dependence. Compare conditioned climatology and trend uncertainty, not exact
day-by-day weather with observations. Use neighbouring cells as one spatial
sensitivity, not extra independent n.

Avoid a tautological validation: density is constructed from pressure,
temperature and composition. Near-perfect prediction from these constituents
does not establish why CO₂ changed them. Assess the documented MSIS helium
contribution separately, including an explicitly labelled helium-excluded
diagnostic if the postprocessor/units permit—not a silent replacement of total
density.

### 6.2 Physical intervention anchor: use existing controlled outputs first

Brown et al. [S3], §§2–5, describe WACCM-X/CESM **v1.2.2** experiments with
different fixed CO₂ levels and solar conditions, Kp held at zero, chemically
consistent CO₂/CO initialization, spin-up and repeated annual realizations.
These are different from the UKSSDC historical **v2.1** run. They support
model-conditional physical contrasts; verify other forcing settings in the
selected archived cases before calling a raw-file contrast CO₂-only.
This is a distinct experimental **design**, not an independent model-family
replication: the cases still share WACCM-X physics/parameterization assumptions.

Useful existing products:

- **Brown scaling surface [S4]:** openly licensed ~2 MB NetCDF plus metadata,
  global annual mean density relative to year 2000, varying CO₂ and F10.7;
  much of the surface is interpolated between simulated cases. It is the
  lowest-cost starting reference, not fresh measurements or independent
  validation of historical local output.
- **Brown controlled raw runs [S5]:** public CEDA record (~801 GB total),
  repeated-year cases with different CO₂ concentrations, daily/monthly output,
  raw thermal/composition variables. Inspect/select only the necessary existing
  cases or request a small derived subset; do not download the whole archive.
- **Solomon solar-min/max archive [S6]:** another available model-design contrast,
  but lower-boundary CO₂, CH₄, CFCs/SST can differ together. Do not automatically
  relabel it a pure CO₂ perturbation or independent model-family replication.

Define a matched contrast at height z and forcing state s:

`theta(z,s;C1,C0) = E[ln rho(z) | prescribed C1,s] - E[ln rho(z) | prescribed C0,s]`.

Its multiplicative effect is `100*(exp(theta)-1)`. State the intervention,
baseline concentration, equilibrated duration, solar/geomagnetic conditions,
local/global averaging and model version. A repeated-year simulation's variability
interval is not the real-world structural-uncertainty interval. Prefer a native-
altitude, directly simulated contrast; high-altitude extrapolation/helium and
interpolated forcing combinations must be labelled.
Keep any scaling-surface query inside its documented solar/concentration domain
(Brown describes roughly 70–200 sfu and 370–890 ppm); do not extend it silently
to high-solar episodes outside that grid. Select a concentration contrast relevant
to the historical exposure range where possible. A fitted %/decade trend is not
a universal %/ppm law, and a future large-CO₂ model scenario is not a matched
replication of a small historical concentration change.

Ask the UK group whether **existing matched fixed-CO₂ historical runs** and
species-resolved IR cooling budgets can be supplied. If not, do not commission
new GCM runs by default. The historical archive becomes consistency evidence,
with controlled published designs as a separate model-conditional anchor.

### 6.3 Optional conditional-dependence/mechanism diagnostic

Only after the causal question and data eligibility are explicit, use a small
graph at one altitude with one solar and one geomagnetic driver, one CO₂
definition, and at most the relevant mediator(s). Do not include all 56 variables
or all 126 levels. Retain one established CI test; nonlinear sensitivity is
worth adding only if it changes a predeclared conclusion under a calibrated,
feasible protocol. PCMCI+ consistency relies on assumptions including causal
sufficiency [S8]; more variables/tests do not ensure those assumptions.

Local CO₂, temperature, winds and composition are endogenous state variables.
For a **total CO₂ effect**, adjusting away temperature/composition can block
the very pathway of interest. Mechanism questions require a different graph/
estimand, with feedback, confounding and lag resolution considered explicitly.
Without direct cooling budgets and mediation identification, do not report a
percentage “mediated by cooling.” Detrended anomaly graphs retain their existing
timescale limitation; raw trending graphs do not become causally qualified by
being larger or completing successfully.

## 7. Confounding and uncertainty: make the checks match the claim

### Predictive estimand

Define paired held-out loss differences on **the same rows**; aggregate them
by calendar day/orbit/mission as declared. Use paired date-block resampling
preserving the model/CO₂ vector, plus seed variation. Avoid pseudoreplication
from millions of nearby orbital rows. Stratify only by predeclared coverage-rich
altitude/activity groups and show sparse-event counts.

Predeclare one primary metric and one practically relevant improvement margin
before evaluating a new confirmatory contrast. Report both the mean effect and
dependence-aware uncertainty; require stable direction across evaluation windows
and no important bias/tail regression. Do not invent a universal 1%/3% threshold
from the already viewed results. Account for selection/multiplicity among formal
contrasts; two pilot seeds cannot establish an all-regime improvement.

### Trend/attribution estimand

Separate sampling uncertainty, serial dependence, solar/seasonal specification,
measurement/calibration errors and physical-model structural assumptions. Use
block/bandwidth and sample-window sensitivities, not only a single HAC band.
Sample/pressure-height uncertainty and model restarts belong in the GCM branch;
mission/drag-coefficient calibration belongs in the observational branch.

Do not infer a physical magnitude from SHAP, a neural partial derivative, an
off-manifold CO₂ swap or error reduction. A surrogate trained only on the
historical trajectory has not learned the response to changing CO₂ while
keeping time and all other forcings fixed. Model counterfactuals need supported
perturbation training or explicit identification, not just a flexible predictor.

### Synthesis across evidence pillars

Build a claim/evidence table with estimand, intervention/adjustment, exposure
range, altitude/averaging, source identity, uncertainty and dependency. Compare:

1. direction;
2. conditional magnitude where estimands really match;
3. altitude/solar fingerprint;
4. robustness to time/sampling/specification controls;
5. plausible reasons for disagreement.

**Do not combine p-values or count agreeing algorithms as independent votes.**
Karman and TU Delft reuse density information; HASDM contains an empirical/
assimilation chain; WACCM-X `DEN` includes MSIS-linked helium; published WACCM
studies share model assumptions. A future hierarchical/Bayesian synthesis could
model those dependencies and biases, but it is not the first-paper prerequisite
and cannot identify an effect that the data do not identify.

Use a graded conclusion rather than a manufactured “probability of causality”:
strong/limited/inconclusive evidence for **physical plausibility**, **predictive
increment beyond the chosen clock controls**, **historical trend compatibility**,
and **conditional magnitude** separately. Agreement strengthens an argument only
when distinct failure modes have genuinely been challenged. Prediction-only
nulls or anomaly-only disagreements do not negate established slow physics.

## 8. Practical sequence, budgets and stop rules

| Phase | Deliverable and gate | Scope/budget |
|---|---|---|
| 0 — Close existing work | Frozen Karman final metrics; confirm which planned comparison is already answered | No new architecture or training; existing evaluation only |
| 1 — Qualify WACCM | Provider clarification, archive/date/schema/units manifest, one reproducible nearest-column sample | Inspection completed partly here; next **one** approved representative-year ingest (~2 GB) before scaling |
| 2 — Freeze common questions | Estimand/DAG, one altitude set, sample windows, calendar controls, uncertainty and held-out selection policy | One protocol document; no new tool/platform programme |
| 3 — Low-cost evidence pilot | Existing controlled scaling/selected cases; strengthened observational trend; cheap structured calendar-control predictor | Cached/small products; not 801 GB controlled archives or 152 GB regional bulk |
| 4 — Regional history branch | Qualified one-column 2000–2021 fingerprints, with explicit gaps/top coverage | Roughly 45 GB of listed regional archives if whole-year retrieval is needed; seek provider column subset first; transfer/storage authorization required |
| 5 — Necessary neural check only | A–D calendar/CO₂ contrast on one fixed architecture | At most 8 pilot fits; five-seed confirmation only for a retained prespecified contrast; stop if clock controls explain the gain |
| 6 — Paper | Integrated evidence ledger, 4–6 justified main figures and coherent limitations | Preserve useful six-figure draft; regroup rather than append every new result |

Phases 3–5 are **gated branches**, not an instruction to execute every row. If
data units/provenance are unresolved, the regional branch stops. If the structured
predictor shows no promising incremental signal beyond clock controls, report
that finding and skip the neural expansion. If controlled model outputs cannot
provide the required contrast, weaken the attribution claim rather than invent
a historical counterfactual. A smaller graph is never a covert substitute for a
previously preregistered full graph; it needs its own declared exploratory scope.

A useful first milestone is a **one-week protocol/data-qualification package**
conditional on provider clarification, small-subset availability or approved
retrieval, storage and metadata verification: manifest, primary column/altitudes,
selected controls and one claim table. An indicative subsequent 2–4-week analysis block
is reasonable only after access, resource and runtime probes pass; it is not an
ETA for unimplemented fits. Do not make a first paper depend on a new global
model or a completely migrated reproduction framework.

## 9. Paper value and defensible conclusions

High-value potential contributions:

- A correctly time-controlled extension of the Karman ±CO₂ question, including
  a credible null or small conditional gain rather than significance hunting.
- A consistent observational/assimilative/GCM comparison of trend and vertical/
  solar fingerprints with explicit shared-source and measurement limitations.
- An explanation of disagreement: proxy time information, mission sampling,
  solar removal, restart seams, MSIS helium or genuinely different estimands.
- A clear distinction between measured/simulated CO₂ concentration, cooling
  diagnostics and physical perturbation results.

Low-value scope additions now: another depth sweep; TCN/Transformer/TabM purely
for a better aggregate score; many CI algorithms on the same graph; all-species/
all-altitude causal matrices; ionospheric extension merely because `EDens` is
available; whole-GCM replication; and a universal software platform.

Possible final statements, depending on results:

- **Prediction:** “CO₂ improves held-out density estimation beyond the specified
  drivers and calendar controls on these periods/strata,” or “its apparent gain
  is explained by a secular proxy / remains statistically inconclusive.”
- **History:** “Adjusted density/temperature/height patterns are compatible with
  the modelled CO₂ response, but historical forcings are not independently
  identifiable on this trajectory.”
- **Physics:** “Verified controlled WACCM experiments imply a conditional density
  response for the stated CO₂/solar/geomagnetic/altitude contrast.”
- **Synthesis:** “The evidence is compatible across distinct designs under stated
  limitations,” not “N agreeing methods prove CO₂ caused the observed trend.”

## 10. Sources, inspected objects and verification limits

### Primary sources

- **[S1] UKSSDC** [model/forcing history](https://www.ukssdc.ac.uk/waccm-x/)
  and [format/variables](https://www.ukssdc.ac.uk/waccm-x/data.html). Read directly;
  regional sample metadata and bounded FTP headers checked separately.
- **[S2] Acciarini et al. (2024)**
  [DOI 10.1029/2023SW003652](https://doi.org/10.1029/2023SW003652),
  [NERC published version](https://nora.nerc.ac.uk/id/eprint/536861/).
  Published PDF §§2.2.1–2.2.2, Results and Appendix A inspected; note explicit
  calendar year, the monthly split and separate unseen-satellite experiment.
- **[S3] Brown et al. (2024)**
  [DOI 10.1029/2024JA032659](https://doi.org/10.1029/2024JA032659).
  Local published PDF §§2–6 inspected: controlled setup, model version, helium/
  extrapolation assumptions and interpolated scaling surface.
- **[S4] Brown et al. scaling product**
  [DOI 10.25500/edata.bham.00001075](https://doi.org/10.25500/edata.bham.00001075),
  [metadata](https://edata.bham.ac.uk/1075/2/metadata.yaml). Landing page/metadata
  inspected, not the NetCDF values; ~2 MB, CC BY 4.0, global annual means and
  WACCM-X v1.2.2 lineage. Surface values are not new observed effects.
- **[S5] Brown controlled raw output**
  [DOI 10.5285/09198c58032d4b8197fd7c6748b92785](https://doi.org/10.5285/09198c58032d4b8197fd7c6748b92785).
  CEDA metadata inspected: public/OGL, ~801 GB, select repeated-year cases,
  spin-up included, raw daily/monthly fields. Detailed selected-case forcing
  identity and data values must be verified before reuse.
- **[S6] Solomon climate-change archive**
  [DOI 10.26024/YPNZ-D857](https://doi.org/10.26024/ypnz-d857).
  NCAR metadata inspected: monthly free-running solar-min/max cases, changing
  lower boundaries, ~391.2 GB. Not assumed to isolate CO₂ alone.
- **[S7] Cnossen (2020)**
  [DOI 10.1029/2020JA028623](https://doi.org/10.1029/2020JA028623).
  Publisher-deposited abstract checked: transient 1950–2015 history, improved
  solar removal with quadratic F10.7, magnetic-field contributions and indirect-
  effect limitations. Full paper not re-audited here; it is not the same UKSSDC
  regional case or a substitute for a verified perturbation contrast.
- **[S8] Runge (2020), PCMCI+**
  [primary PMLR abstract](https://proceedings.mlr.press/v124/runge20a.html).
  Causal sufficiency/oracle-consistency scope checked; project-specific type-I
  error/power and mediation identification were not established by that source.

### Project evidence and limits

Thesis SHA-256 and project/worktree identities are pinned in
`project-review-and-scope-reset-2026-10-05.md`. Current relevant code:
`karman/experiments.py:13–17,48–58`,
`scripts/input_data_prep/prepare_co2_experiment_data.py:26–39`, bounded-study
docs, and the retained Thermodense trend/statistical implementations. The
Karman result table comes from fresh read-only telemetry with integrity reported
true; it is not newly recomputed from checkpoints in this research pass.

The regional inventory records sample CRC/SHA-256 and the difference between
directory/endpoint checks and unverified intermediate records. No original
scientific analyses were rerun, no graphs/fits simulated, no bulk transfer
authorized, and no new dataset adapter or model implemented. The standard
research-scout failed free-tier access; a bounded user-authorized GPT-6 Luna
source review was used instead, with primary verification correcting its
“5°×5°” wording to **5×5 grid points**.

Independent GPT-6 Luna critique checked the plan against primary provider,
Karman and physical-model sources. Material clarifications incorporated:
calendar-controlled gains remain predictive, new evaluation support needs a
prior-use/on-support gate, related WACCM versions are not independent model
families, complete regional timestamps/units/native heights must qualify before
trend analysis, and retrieval/milestone budgets remain conditional. No critique
or metadata inspection validates a new model result or authorizes bulk ingestion.
