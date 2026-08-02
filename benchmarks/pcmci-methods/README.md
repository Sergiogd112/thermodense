# PCMCI method CPU benchmark

This is the frozen synthetic phase for comparing Tigramite PCMCI methods. It runs a deterministic, standardized five-node stable autoregressive process (seed `20260731`) at paired progressive scales: small (`512`, `tau_max=7`), medium (`2048`, `30`), and representative (`8400`, `180`). It measures prepared-array PCMCI computation only: no scientific preprocessing or plotting is included.

Runnable methods are `parcorr`, `cmiknn`, and `gpdc`; `gpdctorch` is explicitly deferred. The verified Auriga driver is incompatible with the locked Torch CUDA 13 environment. This benchmark is a computational comparison, not evidence about the real-data scientific result or conditional-independence validity.

Run locally or on a remote executor from a repo checkout:

```sh
uv run python -m thermodense.benchmarks.pcmci_methods plan --format json
uv run python -m thermodense.benchmarks.pcmci_methods run --output results.jsonl \
  --host-label auriga --environment-label locked-cuda13
```

Verified remote interpreters:

- Phoenix/Auriga (SGE, full locked environment with torch/gpytorch):
  `~/.local/share/thermodense/source/.venv/bin/python -m thermodense.benchmarks.pcmci_methods ...`
  inside a job submitted with `qsub -S /bin/bash -cwd -pe smp 1`.
- Spacehopper (CPU-only conda environment):
  `~/.local/share/thermodense/envs/cpu-parcorr-conda/bin/python -m thermodense.benchmarks.pcmci_methods ...`

Each case runs in its own child process (own process group) with a 30-minute (`1800` second) timeout by default; on timeout the entire process group is terminated so BLAS/numba workers do not leak. Pass `--timeout` to override (final long runs use e.g. `--timeout 43200` for a 12-hour budget); the value actually used is recorded as `timeout_seconds` in every row. A timeout, kill/OOM, or execution failure stops only later scales for that method; successful methods continue. Thread-related OpenMP and common BLAS environment variables are forced to one unless `--threads` explicitly chooses another value. JSONL rows are appended and fsynced after every completed case; an existing output is refused unless `--overwrite` is supplied. Hostnames are never inferred: provide only a deliberate `--host-label`.

Every row carries the `git_commit` (HEAD of the enclosing checkout) and `spec_digest` (SHA-256 of this `spec.toml`); the plan document carries the same `spec_digest`. Parity between `spec.toml` and the module constants is enforced by `tests/test_pcmci_methods_benchmark.py::test_spec_toml_parity_with_module_constants`. The synthetic data generation and the CMIknn shuffle-test RNGs are seeded per `(method, level)` from the frozen seed, so a case is bit-reproducible across hosts.

This benchmark is a computational comparison, not evidence about the real-data scientific result or conditional-independence validity.

The separately reviewed second-stage real analysis is the primary-profile daily
mean `log10` HASDM density at 325 and 825 km with NOAA GML Mauna Loa daily CO2
(including its documented Maunakea substitution), F10.7, and Ap. It is
deliberately outside this frozen synthetic harness and uses PCMCI+ rather than
PCMCI:

```sh
uv run python -m thermodense.benchmarks.real_data
uv run python -m thermodense.benchmarks.pcmci_real run \
  --output results-real.jsonl --methods parcorr cmiknn \
  --tau-max 180 --cmiknn-workers 24
```

The real runner preserves the consecutive daily calendar, retains unavailable
CO2 days as missing observations, and writes both provenance JSONL and canonical
compressed NPZ matrices. GPDC is deferred for this stage because the synthetic
CPU results project a multi-week representative run.

The separate ParCorr-only IAAFT and circular-shift control robustness runner
does not modify either primary runner. It preprocesses the physical nodes once,
then appends one source-missing-mask-preserving IAAFT and one
missing-mask-preserving circular-shift
control per node without preprocessing the controls again:

```sh
uv run python -m thermodense.benchmarks.pcmci_real_controls run \
  --output results-real-controls.jsonl --tau-max 180
```

Its JSONL records deterministic seeds, temporary IAAFT fill and spectral-error
diagnostics, and shift offsets. The accompanying NPZ includes augmented node
names; the selected-link JSONL includes control family/source classifications.
