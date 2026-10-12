# CO₂–density paper: one-page protocol v0.1

Date: 2026-10-05. Target: **JGR: Space Physics**. Status: scope/design draft
for supervisor agreement, **not a retrospective preregistration**. Existing
Karman test results have been seen; frozen experiments and tracker decisions
are unchanged. No new fitting, bulk transfer or Overleaf edit/push is authorized.

**Execution amendment, 2026-10-06:** the user subsequently authorized implementing
and proceeding through the bounded qualification gates: local fixed-region SABER
extraction and an independent inference-only Karman temporal-uncertainty sidecar.
This does not authorize neural retraining, archive-wide remote acquisition,
frozen-campaign changes or Overleaf edits. Record actual pilot/parity outcomes
before escalating; a descriptive extraction does not establish climate eligibility.

## Question and contribution

**Does CO₂ add density information beyond secular time proxies, and are observed
density/cooling patterns compatible with model-conditioned CO₂ responses?**
Separate predictive information, historical trends and physical attribution;
explain disagreement without claiming independent proofs of an observed CO₂ effect.

## Fixed initial scope

| Evidence stream | Primary comparison | Permitted claim |
|---|---|---|
| Observations | Global-mean trends; HASDM/TU Delft sampling checks; solar/geomagnetic/seasonal controls | Adjusted association, not CO₂-only attribution |
| Prediction | Frozen Karman ±CO₂; cheap calendar controls before neural retraining | Prediction relative to chosen controls/test |
| Physics/cooling | WACCM-X density/T/height; SABER **100–139 km flux**; controlled-model reference | Fingerprint consistency; intervention only in verified controlled runs |

- Density targets: **300/400 km**, subject to native support; no extrapolation.
  Pressure-surface height is distinct from fixed-altitude density.
- Region: **16–25°N, 200–210°E**; WACCM column 19.894737°N/205°E.
  Neighbouring cells are sensitivity, not independent n.
- Candidate window: **2002-01-25–2019-12-01**, subject to masks/coverage.
  Preserve longer trends and overlap controls; global means are regional context.
- SABER: official per-scan flux (W/m²), geography referenced at **119 km**;
  retain UTC rollover, orbit, scan, local time, solar zenith angle, QC and counts.
  Prefer sampling-qualified climatology; do not fabricate daily completeness.

## Estimands, controls and uncertainty

- Prediction: `gain = MAPE_without_CO₂ − MAPE_with_CO₂`, **percentage points**,
  on identical rows, seed-matched. Current full canonical support: 10,058,908
  rows / 434 dates / **16 held-out months**; ratio-QC support is a different sample.
- Calendar comparison: A=drivers+calendar, B=A+CO₂, C=drivers+fixed flexible
  calendar, D=C+CO₂. **D−C is predictive, not causal identification.** Lock basis/
  tuning budgets and a new outcome-uninspected evaluation protocol before scoring.
- History: %/decade; limited control sensitivities. Unstable CO₂/time coefficients
  remain unresolved, not evidence of an isolated effect.
- Physics: % density change for an explicit CO₂ contrast and fixed forcing state,
  model version, altitude and averaging; no historical CO₂ swap from a single-run emulator.
- Preserve paired model vectors, observation weights and calendar gaps in temporal
  uncertainty. Seed SD is **not** a data CI; 16 month clusters limit inference.
- Cooling can be a mediator; do not adjust it away when targeting a total effect.
  No causal mediated fraction or combined “probability of causality” from these data.

## Gates and stopping rules

1. Qualify provenance/units/geography/heights/sampling; record exclusions. No
   effect-driven region widening or gap filling.
2. Only an approved, qualified representative-year/column ingest. No default bulk download.
3. Neural extension only after cheap controls: **one architecture, ≤8 pilot fits**,
   separately approved; no depth sweep.
4. A small dependence graph is optional. Anomaly/prediction nulls do not refute slow physics.
5. Finish with traceable claims, uncertainty and limits. Park new GCM runs, CI
   algorithms, species/F2 and general migration unless evidence-changing.

**Boundary:** analysis/provenance here; manuscript/template in standalone Overleaf,
untouched. No raw/restricted data, credentials or private manuscript imports.

Details: [research plan](research/co2-density-triangulation-plan-2026-10-05.md),
[Karman closeout](research/karman-bounded-closeout-2026-10-05.md),
[qualification gates](research/co2-paper-data-qualification-2026-10-05.md).
