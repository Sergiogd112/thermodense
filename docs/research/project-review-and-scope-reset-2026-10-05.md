# Project review and proposed scope reset

Reviewed 2026-10-05. **Advisory scope/evidence review, not a full scientific
reproducibility certification.** No scientific implementation, running campaign,
roadmap issue, manuscript or frozen result was changed by this review.
Paths below are relative to the Thermodense checkout root unless absolute.

**Subsequent clarification:** the user explicitly wants a bounded attribution/
evidence-triangulation study, not only a descriptive-paper consolidation. The
provided regional WACCM-X archives were subsequently located and sampled.
[The updated scientific plan](co2-density-triangulation-plan-2026-10-05.md) and
[verified inventory](waccm-x-regional-inventory-2026-10-05.md) supersede this
review's provisional advice to park WACCM-X, without changing frozen studies or
silently revising existing tracker decisions. The organizational findings and
review limits below remain a historical audit.

## Read this first

**You have a research programme, but your next deliverable should be one
supervisor-reviewable paper packet.** Keep the thesis's broad question about
density variability, the forcing hierarchy and product/model dependence. Do not
require a new causal-discovery programme, a new predictive architecture and a
general software platform to finish that packet.

The packet is not a blank-page exercise: an existing **13-page, six-figure draft**
is at `../thermodense-paper/manuscript/thermodense-first-paper.pdf`, with source
at `../thermodense-paper/manuscript/main.tex`. It is a provisional starting point,
not independently validated final evidence.

### One active deliverable; three immediate actions

1. **Finish and preserve Karman's already specified evaluation.** Training is
   complete. Make a bounded closeout summary; do not automatically launch the
   architecture shortlist. Karman is a separate predictive study, not a condition
   for the Thermodense paper.
2. **Make one claim–figure–artifact checklist for the existing paper.** Start with
   Figure 6's trend synthesis, then retain only supporting results that pass
   their stated evidence checks. Keep all six existing candidates visible;
   “already rendered” is not the same as “must retain.”
3. **Get one explicit scope decision from the supervisors:** can the first
   paper proceed with a dependence/trend/model-comparison core, with unresolved
   causal work labelled exploratory or deferred? If not, choose one bounded
   causal question and an explicit stop rule rather than completing the whole
   expanded programme.

This would revise existing decisions; it does **not** silently replace them.
In particular, issue #55 superseded the daily-first causal allocation with a
3-hour design. Do not label the retired daily core accepted again without a new
decision. Do not launch a replacement experiment just because this review
recommends a smaller paper.

## 1. What belongs to which project

| Workstream | Existing state | Recommendation now |
|---|---|---|
| Bachelor's thesis | 189-page July 13 PDF; substantial methods/results and explicit caveats | Preserve as historical baseline; do not reopen every thesis figure |
| Thermodense first paper | Six rendered figures, LaTeX draft, summary/caption artifacts in a separate worktree; evidence audit still open | **Primary next deliverable:** qualify and trim the existing draft for supervisor review |
| Karman ML comparison | Bounded pilot/confirmation; 30 confirmation fits completed; canonical evaluation underway at review time | Close out the frozen comparison; no new architecture sweep |
| Thesis reproduction package | Honest workflow foundation; configured stages unavailable; scientific scripts retained | Repair only a concrete retained-result reproduction path; park general migration |
| Raw 36-node 3-hour PCMCI | Failed long fit produced no final evidence; current runner forbids causal wording | Park unless specifically selected as the next scientific question |
| Daily nonlinear/global-activity/polynomial benchmarks | Separate experiments with different inputs, estimands and recovery properties | Preserve results; do not substitute their output for the failed 3-hour graph |
| New SABER species, F30, hmF2/NmF2, WACCM-X, new architectures, cloud capacity | Mixture of explored, proposed, partly implemented and gated work | Parking lot, not a shared paper prerequisite |

Keep this table as the working view. An issue catalogue, source directory or
large configuration matrix is useful inventory, but it is not today's task list.

## 2. What the thesis already establishes—and does not

The thesis's overview, Methods §§3.6–3.9, synthesis §5.6 and Conclusions §6.1
already separate three questions:

- **Resolved variability:** solar dependence is strongest; geomagnetic structure
  contributes especially during disturbed conditions. Descriptive consistency
  across products does not make those products independent replicates.
- **Long-term change:** solar-adjusted calendar-time slopes are product-specific
  diagnostics consistent with contraction, not isolated CO₂ effects.
- **Conditional anomaly structure:** the centred approximately three-year
  background removal suppresses secular variation. Absence of a surface-CO₂
  PCMCI link does not test or refute long-term CO₂-driven contraction.

These are strengths to retain, not missing discoveries that must all be repaired
before writing. The model-error definition is also explicit: natural-log
`ln(rho_model/rho_reference)`, whereas density diagnostics use `log10(rho)`.
One Luna first-pass concern suggested possible logarithm ambiguity; inspection
of the paper's equation (1), domain definitions and trend implementation resolves
that concern. It is not a verified formula defect.

### A checked numerical bridge

The supplied thesis PDF, printed pp. 113–115 / Figure 5.70, reports near 400 km:

| Product | % per decade | Reported 95% HAC interval |
|---|---:|---:|
| Global mean | −4.9 | −6.0 to −3.9 |
| Mauna Loa HASDM subset | −5.8 | −8.3 to −3.1 |

The current trend CSV gives −4.94019 [−5.98775, −3.88096] and −5.75614
[−8.33461, −3.10513]. Independently applying
`100 * (10**(10 * beta_year) - 1)` to its saved log10 slopes and endpoints
agrees. This checks the conversion and rounded statements, **not** the fit from
original observations or its full uncertainty budget.

`scripts/recreate_brown_density_trend_current.py:199–254,302–399` implements
quadratic F10.7 adjustment and calendar-day score aggregation for 27-day HAC,
including missing/duplicate dates. That code reading agrees with the stated
estimator; approved isolated reruns and independent covariance tests were not
performed in this review.

## 3. High-impact findings

### R1 — Major: competing scope contracts are the main organizational blocker

**Evidence:** original July supervisor correspondence asks for something simple
and promising, JB2008 and later unsmoothed F10.7; the July roadmap ranks optional
extensions. The August paper chat explicitly requests the existing six figures
and consultation before expensive missing work. Issue #14 sets core/pilot gates;
issue #38 carries the draft to author review; September issue #55 supersedes the
daily core with a native 3-hour graph. Root `CONTEXT.md` still describes the
daily sensitivity design extensively.

**Impact:** an old accepted plan, a newer scope reset and a provisional draft can
all look like the current binding task list. More work does not resolve which
claim is necessary.

**Action:** record one supersession decision covering paper scope, primary causal
status and deferred extensions. Then reconcile #38/#55 and the domain/figure
documents. Preserve history; do not silently close or rewrite issues. This review
has not made those tracker changes. **Confidence high; chat/document verified.**

### R2 — Major: the paper already exists, but its evidence is not frozen

**Evidence:** the separate paper worktree contains the six figures and 13-page
PDF. `manuscript/main.tex:40–65,149–183,194–210` already reports quantitative
relationships, nominal Fisher intervals, median/circular-bootstrap diagnostics
and HAC trends. Its Sept 5 baseline audit explicitly marks source restoration,
calibration overlap and artifact lineage incomplete. Main-checkout figure-plan
blockers do not mean no draft renderer or rendered figure exists elsewhere.

**Impact:** treating the draft as absent duplicates work; treating a clean PDF or
past passing tests as validated science skips the important checks. The older
audit's “12-page” count is not the current delivered PDF's 13 pages.

**Action:** preserve the existing prose and rendered candidates. Make one row
per retained claim with exact source artifact, date/row support, method, code
identity, figure and verification status. Rebuild only what that ledger needs.
**Confidence high; source/artifact verified, not reproduced.**

### R3 — Major if used for a causal claim: the large raw graph is not paper evidence

**Evidence:** the historical raw-36 job 1139 reached its 72-hour limit without
final matrices (`docs/workflow-inventory.md:32–34` and the next-step chat).
The current contract has 36 nodes, 1,440 lag steps and 1,400,274 eligible
link/lag tests. `docs/raw-36-node-3hour-pcmci.md:62–89` explicitly says pairwise
support is not conditional-test support, raw diagnostics do not qualify causal
wording, and fitting has no solver checkpoint.

**Impact:** even completing an unchanged expensive run would not automatically
qualify atmospheric causal attribution. This is a scientific-design and
recoverability gate, not simply a need for a faster GPU.

**Action:** do not rerun unchanged or substitute a polynomial/predictive result.
Either defer it from the first paper, or choose a separately bounded design with
explicit eligibility, uncertainty, support and recovery criteria. Existing raw
runner fixes do not retroactively produce the failed job's evidence.
**Confidence high; documented contract verified; remote accounting not rerun.**

### R4 — Major before formal inference: uncertainty and comparison contracts diverge

**Evidence:** thesis §§3.2–3.3 and paper §2.2 honestly qualify independent-pair
Fisher/classical OLS intervals. `scripts/stats_utils.py:9–75` implements those
nominal intervals. Trend intervals instead use HAC. The paper uses signed
medians/median absolute errors and a 250-replicate circular bootstrap; root
`CONTEXT.md` specifies means, stationary date bootstrap, 54/81-day sensitivities
and simultaneous coverage for seven formal contrasts. Issue #14's later
resolution specifies median diagnostics, so the records themselves need
adjudication—not an assumption that one word choice is scientifically invalid.

**Impact:** intervals are not interchangeable. A caveat permits descriptive
figures but does not establish autocorrelation-aware coverage or formal model
ranking. Likewise a small validation/test MSE change is not a causal CO₂ effect.

**Action:** choose the intended estimands and claim level once. For retained
inferential results, use dependence-aware, paired uncertainty and documented
sample support; do not rebuild all historical descriptive plots merely to make
every chart identical. Keep centred F10.7 as descriptive/retrospective context,
not an unqualified operational timing or causal-lag variable.
**Confidence high; method/code/document comparison verified, coverage untested.**

### R5 — Major before publication: figure provenance is partly strong, not complete

**Evidence:** the current Figure 6 CSV and Brown digitization hashes match their
provenance. The trend conversion and selected 400-km prose reconcile. The
literature panel is explicitly digitized figure geometry, not original study
data. The paper identifies different epochs and reference/calibration domains.
JB2008's current saved trend is −9.77%/decade at 400 km, versus −1.02% for JB2006:
these model-generated slopes are not observations of climate forcing.

**Impact:** that contrast deserves a forcing/proxy/method explanation rather than
being read as independent evidence of fast contraction. Hashes attest byte
identity, not estimator validity. The private thesis archive Git object and
current thesis PDF are different review objects; source checkout contents have
also been removed, so exact rendered-version reconstruction is not established.

**Action:** attest only retained inputs/results/code and document access routes,
calibration overlap, exclusions and versions. Prioritize a bounded trend-control /
bandwidth / common-period sensitivity if the organizing claim depends on its
magnitude. Select that sensitivity after scope freeze; this review did not run it.
**Confidence high for checked artifacts; upstream generation partly verified.**

### R6 — Major as a scope trap, not a code defect: infrastructure is not the paper

**Evidence:** `README.md:8–12`, `workflows.py:36–45` and thesis TOML configs
explicitly leave workflow stages unavailable. ADR 0002 is **Proposed**, with a
prototype expansion of 416 cases/444 artifacts; it enables no live execution.
Frozen benchmarks and standalone scripts have separate scientific contracts.

**Impact:** a generalized planner/migration can become a prerequisite for all
research even though the needed results already have local producers. It creates
more interfaces, environments and completion definitions to keep in mind.

**Action:** retain the honest boundaries and immutable artifacts. Fix a broken
path only when it blocks a chosen claim. Park broad platform adoption, cloud
orchestration, complete thesis migration and alternate architectures until a
named next deliverable actually needs them. **Confidence high; code/document verified.**

## 4. Minimum claim-to-code ledger for the next packet

Do not mark a whole pipeline “verified” because one number checks out.

| Claim/object | Method/code | Saved evidence inspected | Implementation | Provenance | Reproduction | Independent check | Statistical validity |
|---|---|---|---|---|---|---|---|
| 400-km adjusted trends and %/decade conversion | Trend script, log10 transformation, quadratic solar control, HAC | Thesis pp.113–115, current CSV, Figure 6 provenance | Logic inspected; not independently tested end-to-end | CSV checksum matches; original generation not re-attested | Not rerun | Scalar conversion and rounded values agree | Product-specific; control/bandwidth sensitivity not established here |
| Density correlation / binned slopes | `pearsonr_ci`, `ols_slope_ci` | Thesis Methods; paper §§2.2/3; selected rendered plots | Formula inspected | Full source lineage not re-attested | Not rerun | No independent full fit | Nominal independent-pair/classical intervals; not exact time-series coverage |
| PCMCI anomaly interpretation | Centred seasonal/background transformations; ParCorr/CMIknn | Thesis §§3.6–3.9 and §6.1; sampled graphs | Selected contracts read; no full graph-code trace | Historical artifacts not fully reconciled | Not rerun | No independent significance calibration | Exploratory/conditional; not a secular CO₂ test |
| Paired five-model errors | Natural-log ratio; draft median/circular-bootstrap method | Paper §2.3, Figure 5 and summary; scope contracts | Definition checked; inference not fully traced | Exact common-sample assertion not independently recomputed | Not rerun | No independent seven-contrast calculation | Formal contrast/uncertainty contract unresolved |
| Karman predictive comparison | Frozen data, validation-selected finalists, paired CO₂ controls | Live read-only telemetry; bounded-study docs | Prior work reviewed; no new implementation changes | Collector reports integrity pass | No new fit/rerun | Final closeout still needed | Predictive ablation; not CO₂ attribution |

## 5. Proposed parking lot and finish line

**Park by default:** WACCM-X ingestion/new GCMs; full-volume or TU Delft causal
graphs; broad 3-hour/nonlinear CI expansion; NO/F30/F2 extensions not needed by
the frozen claim; general scheduler/platform migration; paid/cloud deployment;
new Karman TCN/Fixup/TabM/probabilistic-head training.

Parking preserves work and ranked future options. It is not a statement that the
ideas are bad or that user-approved historical decisions never existed. An
unavailable extension can be explicitly deferred; it must not become fabricated
evidence or an indefinite blocker by inertia.

**First-packet finish line:** one agreed scientific question; fixed retained
figures; inspectable numbers and source paths; appropriate uncertainty for each
claimed inference; explicit sampling/access/calibration limits; and one compiled
draft with a short supervisor note listing unresolved choices. Target-journal
formatting and a universal reproduction platform are not required at that stage.

**Work-in-progress limit:** one substantive research deliverable, one small
supporting repair at a time, and maintenance of already-running jobs. A proposed
new experiment must name the current claim it can change, the exact artifact it
will produce, its budget and stopping rule. Otherwise it goes in the parking lot.

## 6. Review coverage, identities and limits

- Thermodense checkout: `bd5f6da17ac69079bdea575eda229384f643d5bd`, on
  `checkpoint/pre-paper-20260905`; pre-existing local changes to the SABER stream
  prototype and figure-review data were preserved.
- Paper worktree: `a05c3eebbe697e7145a31f0f5cf18b95528f8a6c`,
  `paper/first-paper-draft`; uncommitted manuscript/authorship changes and local
  review notes were preserved. Current PDF SHA-256:
  `b666e0466322195205e2fcd09920aa393684d21c9661c7f9cb8677f9583d5a2d`.
- Thesis archive HEAD: `c38ec6688b29289cbf9214f52c3490e3ad018b6b`; current
  189-page PDF SHA-256:
  `05d5574a31f48b004823ca5eb47e58004f480ef861a2f5550a1a9ccd2fe1d9ea`.
  Extracted archived source was used as separately identified historical context,
  not assumed to generate the current PDF.
- Karman checkout: `2adeadb6bc1d593a35826f94cc9c46f156e097a2`, plus substantial
  uncommitted/untracked study work; live deployed frozen campaign is a separate
  reviewed identity, not simply local HEAD.
- Read-only extraction covered project-specific primary-text records from local
  OpenCode and T3 stores; the OpenCode extraction starts with the July 29
  post-thesis roadmap. Exact cross-store duplicate text was marked/omitted.
  Key roadmap, six-figure, scope-reset and ML sessions were sampled using full
  nearby context. **Not every historical message, attachment or subagent tool
  trace was audited.** The original two supervisor email screenshots were read
  by the primary reviewer. Transcript exports remain private under
  `/tmp/opencode/project-review-20261005`; no raw chats/accounts/cookies or
  private email screenshots are added to the repository.
- Muse Spark 1.3 Contributor Free was tested but rejected with HTTP 403.
  The user explicitly authorized available **GPT-6 Luna**, not an assumed 6.1
  alias. Three read-only first-pass reviews used `openai/gpt-6-luna` through
  OpenCode; initial read-permission failures were corrected with no persistent
  global configuration change. First-pass outputs are leads, not authority:
  limited thesis access/reading and stale main-checkout figure status were
  corrected by primary checks against the PDF, paper worktree and artifacts.
- Selected thesis methods/synthesis/conclusions, paper source and extracted PDF,
  workflow/statistics/trend code, provenance, GitHub issues and private chat
  decisions were read. Visual spot checks cover thesis printed pp.111 and114,
  paper pp.6 and8, and the current two-panel Figure 6. **This is not an audit of
  every page/figure of the 189-page thesis or a source-by-source bibliography
  verification.** Such a completion gate is not claimed.
- Brown's NERC publication/abstract and PCMCI+'s primary PMLR abstract were
  checked for the limited physical/causal-sufficiency statements used here;
  local published papers and the whole bibliography were not re-audited.
- No author scientific code, tests, fits, downloads or compilation was executed.
  Read-only Git/database inspection, text extraction, bounded PDF raster previews
  and independent scalar arithmetic were used. Full isolated reproducibility
  checks remain unverified and require a separately approved execution plan.

**Recommendation:** major revision/evidence qualification before publication;
not a rejection of the thesis or a demand for an expanded programme. Its cautious
core can support a finishable first-paper direction, provided scope is explicitly
reset and retained claims—not every possible experiment—are verified.

### Primary references consulted

- [First-paper map #38](https://github.com/Sergiogd112/thermodense/issues/38)
- [Scope reset #55](https://github.com/Sergiogd112/thermodense/issues/55)
- [Core/extension gates #14](https://github.com/Sergiogd112/thermodense/issues/14)
- [Evidence-freeze decision #46](https://github.com/Sergiogd112/thermodense/issues/46)
- `CONTEXT.md`, `docs/first-paper-figure-plan.md`, `docs/workflow-inventory.md`,
  `docs/raw-36-node-3hour-pcmci.md`, ADRs 0001/0002.
- Paper worktree: `manuscript/main.tex`, `manuscript/figures/first-paper-figures-summary.json`,
  `docs/research/first-paper-manuscript-baseline-audit-2026-09-05.md` (historical audit,
  not a new reproduction).
- [Brown et al. (2024), NERC record](https://nora.nerc.ac.uk/id/eprint/537943/)
- [Runge (2020), PCMCI+ primary abstract](https://proceedings.mlr.press/v124/runge20a.html)
