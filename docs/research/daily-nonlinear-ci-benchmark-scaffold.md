# Daily nonlinear conditional-independence benchmark scaffold

Issue #61 records a plan only.  It does not read, copy, or publish the
access-conditional daily five-node input, and it does not change the frozen
`pcmci_real` runner.

`thermodense.benchmarks.daily_nonlinear_ci.plan_document()` emits the candidate
availability, common-input four-cell daily matrix, and promotion rules.  Its
`result_template(candidate_name)` emits the machine-readable envelope a later
executor must fill.  Both are deliberately JSON-serializable without optional
CI implementations or real data.

CMIknn and GPDCtorch remain the existing candidates.  GPDCtorch retains its
existing CUDA gate.  RCoT/RCIT is included as the selected external candidate,
but explicitly has an absent adapter until a Tigramite integration and its
time-series/null-calibration contract are implemented and validated.

Promotion requires identical accepted-quality input at `tau_max=10`, calibrated
type-I error, representative power, stability, provenance, and a predeclared
runtime criterion.  Runtime is recorded, but speed alone cannot promote an
external method.  The scaffold makes no speed claim for RCoT/RCIT.  A nonlinear
`tau_max=180` decision is expressly outside this benchmark.  Comparisons with
the 3-hour ParCorr core remain bounded daily method sensitivity only.
