# PCMCI method synthetic benchmark — results

- `benchmark_version`: `pcmci-methods-synthetic-1`
- `spec_digest`: `8f75a972d453e554781b50efdda4bd48a0e6b6dace4a50bdd5c23ee5084e9563`
- Harness commit: `ea29b7e`
- Grid: 5-node seeded VAR (seed `20260731`), methods parcorr → cmiknn → gpdc, levels small (`512`, `tau_max 7`), medium (`2048`, `30`), representative (`8400`, `180`); one CPU thread per case; 1800 s per-case timeout with progressive per-method stop.
- Executors: Auriga (SGE, full locked env incl. torch/gpytorch, 1 slot) and Spacehopper (CPU-only conda env), three concurrent per-method processes per host.

## Outcome

| case | Auriga | Spacehopper | cross-host digest equal |
|---|---:|---:|:---:|
| parcorr / small | succeeded, 19.5 s | succeeded, 4.7 s | no (BLAS) |
| parcorr / medium | succeeded, 6.6 s | succeeded, 7.6 s | no (BLAS) |
| parcorr / representative | succeeded, 102.1 s | succeeded, 212.5 s | no (BLAS) |
| cmiknn / small | succeeded, 119.3 s | succeeded, 148.7 s | yes |
| cmiknn / medium | timeout at 1800 s | timeout at 1800 s | — |
| cmiknn / representative | skipped | skipped | — |
| gpdc / small | succeeded, 109.0 s | succeeded, 170.8 s | yes |
| gpdc / medium | timeout at 1800 s | timeout at 1800 s | — |
| gpdc / representative | skipped | skipped | — |

Per host: 5 succeeded, 2 timeout, 2 skipped. Peak RSS ~0.5 GiB (Auriga) / ~0.3 GiB (Spacehopper).

The 1800-second grid established the timeout boundary, after which deliberately
budgeted long runs and a recorded CMIknn worker probe resolved the true wall
times:

| case | Auriga | Spacehopper | cross-host digest equal |
|---|---:|---:|:---:|
| cmiknn / medium, workers=1 | 5425.3 s (1.51 h) | 7304.2 s (2.03 h) | yes |
| gpdc / medium, workers=1 | 12026.4 s (3.34 h) | 26228.5 s (7.29 h) | yes |
| cmiknn / medium, workers=24 probe | — | 848.8 s (14.1 min) | same digest as workers=1 |
| cmiknn / representative, workers=24 probe | — | 76164.3 s (21.16 h) | not mirrored |

## Findings

1. **ParCorr is tractable at all three scales** at one CPU thread: representative completes in ~102 s (Auriga) / ~212 s (Spacehopper).
2. **CMIknn medium is tractable but hours-long at workers=1**. Raising its explicit SciPy/sklearn worker setting to 24 gives an 8.6× medium speedup without changing the result digest, and makes representative scale feasible in 21.16 h.
3. **GPDC medium is tractable with a long budget**, but representative projects to roughly three weeks at one CPU thread and remains GPU-deferred.
4. **Cross-host bit-reproducibility**: cmiknn and gpdc result digests are identical across hosts at small and medium scales; parcorr digests differ by BLAS float rounding (conda OpenBLAS on Spacehopper vs manylinux wheels on Auriga/local). This is expected and does not affect the timing conclusions.
5. **Intra-host reproducibility holds**: repeated parcorr/small runs produced identical digests on Spacehopper, and Auriga's digest matches the local arch workstation run.
6. Auriga per-child startup adds ~15 s (torch/gpytorch import in the locked venv); negligible on the Spacehopper conda env.
7. Every timeout was a clean process-group kill at 1800 s; no leaked child processes on either host.

## Implication for the paper claim (issue #8)

- Feasibility boundary: ParCorr is straightforward at representative scale; CMIknn is feasible there with its explicit worker setting raised to 24; GPDC representative remains GPU-deferred. The separate real-data stage uses PCMCI+ on primary-profile daily mean `log10` HASDM density at 325/825 km with CO2, F10.7, and Ap.
- This benchmark is a computational comparison only; it is not evidence about real-data scientific results or conditional-independence validity.
