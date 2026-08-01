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

## Findings

1. **ParCorr is the only method tractable at all three scales** at one CPU thread: representative completes in ~102 s (Auriga) / ~212 s (Spacehopper).
2. **CMIknn and GPDC exceed 30 minutes at medium** (`2048`, `tau_max 30`) on both hosts; representative was skipped by the progressive design. On these CPU executors they are practical only near the small scale.
3. **Cross-host bit-reproducibility**: cmiknn and gpdc result digests are identical across hosts (numba-kernel and sklearn-GP paths); parcorr digests differ by BLAS float rounding (conda OpenBLAS on Spacehopper vs manylinux wheels on Auriga/local). This is expected, not a correctness issue, and does not affect the timing/feasibility conclusions.
4. **Intra-host reproducibility holds**: repeated parcorr/small runs produced identical digests on Spacehopper (verify + grid rows), and Auriga's digest matches the local arch workstation run.
5. Auriga per-child startup adds ~15 s (torch/gpytorch import in the locked venv); negligible on the Spacehopper conda env.
6. Every timeout was a clean process-group kill at 1800 s; no leaked child processes on either host.

## Implication for the paper claim (issue #8)

- Feasibility boundary at 1 CPU thread: representative scale (`tau_max 180`, `n=8400`) is only ParCorr-tractable on these executors. CMIknn/GPDC need smaller samples/`tau_max`, more threads, or GPU acceleration for the real-data second stage (primary-profile daily mean `log10` HASDM density at 325/500 km with CO2, F10.7, Ap).
- This benchmark is a computational comparison only; it is not evidence about real-data scientific results or conditional-independence validity.
