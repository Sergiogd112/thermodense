# Global-mean linear versus nonlinear matrix comparison

Verified 9 September 2026 from the completed global-activity benchmark, SGE job
1145. The job exited 0 after 94 seconds on two allocated cores. No training was
repeated to produce this comparison; exported matrices were replayed locally.

## Matched held-out comparison

- Input: all ten global-mean density altitudes, daily raw F10.7, daily Ap and Kp
  sum, and daily MLO CO₂. Centered-81-day F10.7 labels activity only.
- One-step predictions from 180 prior daily lags; chronological 70/15/15 split.
- Each degree's ridge/epoch choice uses validation only, with 15 training epochs
  per trial. The overall validation-selected winner is **degree 1**.
- Test period: **2012-01-20–2019-12-01**, 2,873 dates and **28,730 identical
  observed density-target cells**, including persistence-comparable support.
- Errors are standardized using the archived training-only scales. They are not
  relative physical-density errors or errors in original density units.

| Model | Pooled standardized test MSE | Change versus linear MSE |
|---|---:|---:|
| Linear matrix / degree 1 | 0.02518825 | Reference |
| Additive quadratic / degree 2 | 0.03174399 | **26.03% higher** |
| Additive cubic / degree 3 | 0.18276544 | **625.60% higher** |
| Persistence | **0.00928264** | Lower than all three fitted models |

There is **no nonlinear improvement in this tested configuration**. All three
activity-bin point estimates also favor linear over quadratic/cubic. However,
the test set contains only **23 high-activity dates**; this subset cannot support
a general claim about high-solar-activity forecasting.

## Paired temporal uncertainty check

Improvement is defined as `100 * (1 - nonlinear_MSE / linear_MSE)`; negative
values mean the nonlinear model is worse. A paired moving-block bootstrap samples
calendar-date blocks jointly across all ten altitudes, preserving their paired
errors and using the same persistence-comparable support as the reported scores.
There are 2,000 replicates for each predetermined 30-, 90-, and 180-day block
length; seed 20260909. Activity-group results remain point estimates.

| Block length | Quadratic improvement, 95% interval | Cubic improvement, 95% interval |
|---|---:|---:|
| 30 days | −33.76% to −18.80% | −731.47% to −529.43% |
| 90 days | −34.26% to −17.97% | −754.81% to −496.81% |
| 180 days | −34.66% to −18.36% | −728.28% to −475.82% |

These descriptive intervals condition on the selected models and this one test
period. They do not account for all training/selection uncertainty, establish
causality, or provide multiple-comparison-adjusted group-difference inference.
None supports a nonlinear improvement here. Fifteen bounded optimization epochs
do not establish convergence: this is not a verdict against all nonlinear models.

## Integrity and reproduction

The comparison checks the archived input/result identities and checksums, local
fitting-code hashes, test indices, node order, lag order, and scaling arrays. It
reconstructs validation and test predictions and matches the stored pooled and
per-altitude scores/support before reporting the comparison. Historical source
paths are not resolved as part of replay; the bound input NPZ is verified.

```console
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 uv run python \
  -m thermodense.benchmarks.global_activity_comparison \
  --manifest data/products/global-activity-daily-v1/manifest.json \
  --results data/products/global-activity-results-v1 \
  --output data/products/global-activity-results-v1/comparison/comparison-verified.json
```

The output is immutable: choose a new output filename rather than overwriting an
existing comparison. Raw data and model artifacts remain ignored by Git. The
derived comparison JSON includes settings, source/result hashes and the comparison
module hash. Local replay/bootstrap takes seconds, not another training run.

**Recommendation:** retain persistence as the performance baseline and the linear
matrix as the simpler fitted benchmark. Before increasing polynomial complexity,
investigate optimization convergence and validation-only tuning, then evaluate
on a fresh temporal holdout or rolling-origin splits. Do not tune on this test
period and subsequently describe it as untouched test evidence.

The separate original 36-node native-3-hour job (1139) is not part of these
numbers. It was killed at its 72-hour limit on 9 September at 14:48:18 CEST,
with exit 137 and no final output directory. A later wall-time extension attempt
was refused because that job had already ended; no automatic rerun was submitted.
