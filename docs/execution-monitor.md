# Execution monitor

`python -m thermodense.execution_monitor --kind global-polynomial --run-dir PATH --input-manifest PATH`
emits one read-only JSON observation. It separately reports declared completion,
streaming SHA-256 artifact verification, verified input bytes and identity binding,
scientific-contract matching, and the workflow's descriptive recovery boundary.
`contract_matches` does not mean stationarity or causal qualification. A checkpoint
or result file existing is not resume approval.

The known local layouts are polynomial (`provenance.json`, `models.npz`,
`metrics.json`), global ParCorr (`status.json`), raw-36 (`provenance.json`), and
a derived `comparison.json`. Local manifests are preferred; an embedded manifest
is used only when no local manifest is supplied. Embedded metadata is
`declared_only`: without local input bytes it cannot produce `verified_complete`.
Missing, invalid, unrelated, or byte-mismatched local inputs fail closed. Comparison
JSON currently reports artifact integrity as unverified because its `result_hashes`
identify upstream model files, not the derived comparison file itself.
For raw-36 runs, verification covers both the core and SABER NPZ bytes; checking
the core alone is insufficient. `verified_complete` additionally requires the
canonical input identity and exact 2000-01-01T00:00 through 2025-07-20T00:00
three-hour contract; fixture manifests remain unverified. This verifies declared
contract and artifact bytes only: it does not replay or recompute the 1.4-million-
row BH export and does not establish paper readiness.

Optional SGE observation is `--ssh-host phoenix --job-id 1139
--scheduler-timezone Europe/Madrid`. It uses only fixed `qstat`, `qacct`, and
`qquota`-compatible read queries over SSH; it never submits, cancels, retries,
syncs, loads checkpoints, or changes local/remote state. Pending time is not used
for a deadline: a deadline requires SGE's actual start time and `h_rt`.
Scheduler accounting is accepted only from one complete record whose `jobnumber`
matches the requested job and which declares both `failed` and `exit_status`.
