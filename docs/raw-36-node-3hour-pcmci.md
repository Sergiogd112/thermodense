# Raw 36-node native-3-hour PCMCI+ core

`thermodense.benchmarks.raw_3hour_pcmci` is a separate canonical runner; it
does not change the frozen daily `pcmci_real` runner or consume exploratory
polynomial/40×60/50×70 streams.  It accepts a JSON manifest with
`input_contract: "raw-36-node-native-3hour-v1"`, `input_kind` (`canonical` or
`fixture`), relative, distinct `core_npz` and `saber_npz` paths with required
SHA-256 digests, raw daily-held F10.7, and an all-`none` transformations
declaration. Canonical inputs are exactly 74,657 UTC slots from
2000-01-01T00:00 through 2025-07-20T00:00 at a three-hour cadence; the manifest
records and the loader verifies that calendar. Fixtures may use a shorter declared
payload calendar, but cannot qualify as a verified completed canonical result.

The core NPZ supplies consecutive UTC 3-hour `timestamps`, the 27 fixed
`density_altitudes_km`, `density_log10_values`/`density_masks`, and
`f107`, `ap`, `kp`, and `mlo_co2` `*_values`/`*_masks`.  The SABER NPZ supplies
matching `timestamps`, exact channels `CO2cool`, `NOcool`, `O2_1delta_ver`,
`OH_16_ver`, `OH_20_ver`, and 5×T `values`/`masks`.  A canonical manifest
requires T=74,657; short `fixture` manifests are intentionally supported for
local tests.  No raw/provider data belongs in Git.

## Canonical-input assembly

Use `python -m thermodense.benchmarks.raw_3hour_input` with local,
already-authorized inputs only:

```console
uv run python -m thermodense.benchmarks.raw_3hour_input \
  --legacy-prepared-npz /path/to/prepared_input.npz \
  --space-weather-csv /path/to/SW-All.csv \
  --saber-50x70-npz /path/to/final_arrays.npz \
  --output /path/to/raw-36-canonical
```

The legacy input must be named `prepared_input.npz` and have its fixed
36-channel order. Only its 27 log-density columns, native Ap/Kp, and already
UTC-daily-held MLO CO2 are retained. Its `F10.7_raw` column is validated as a
known legacy node and discarded because it is spline-derived. `DATE` and
`F10.7_OBS` with `F10.7_DATA_TYPE` exactly `OBS` from the CelesTrak compatibility
CSV provide the replacement raw F10.7 through UTC-date lookup repeated over each
3-hour slot. PRD/PRM/non-OBS rows remain missing; no interpolation is performed.
Canonical assembly also requires adjacent `prepared_input.provenance.json` binding
the prepared NPZ digest, producer version, channel order, shape/cadence, four
declared upstream source hashes, and legacy CO2 missing-value policy. This verifies
the prepared artifact/provenance binding; upstream Ap/Kp and MLO semantics remain
declarations by that frozen legacy producer rather than independently re-parsed
source evidence. Fixtures explicitly record unverified legacy semantics.
The SABER NPZ must have exact matching timestamps and the five
listed channels in order. The assembler rejects nonmatching calendars, invalid
daily holds, wrong source-node order, incomplete canonical length, and an
existing output directory. It copies the validated SABER arrays into the output
as `saber.npz`, then writes `core.npz`, `manifest.json`, and `provenance.json`
with source and output SHA-256 hashes and raw/no-transform declarations. The
loader rejects absolute/traversing artifact paths and byte mismatches. Use
`--fixture` only for short test data.

The manifest classifies the copied arrays as restricted provider-derived input:
they are for authorized local/site storage only, and any cloud upload requires
explicit approval. The assembler performs no upload; the operator remains
responsible for choosing an authorized, ignored destination.

The graph has density nodes 175..825 km and nine drivers.  It permits
driver→density and density↔density candidates, forbids density→driver, uses
`tau_max=1440` (0--180 days), explicit missing flag `-999999`, and BH across
every declared eligible link/lag. Results retain raw p, q, partial-r,
normalized orientation, physical lag and explicitly labelled pairwise support.
Pairwise support is not conditional-test support; it excludes conditioning
variables and Tigramite's cut-off. Raw ADF/KPSS diagnostics are reporting-only,
and this unregistered raw preprocessing profile is always labelled
association/sensitivity-only with causal wording forbidden.
KPSS warnings stating that the actual p-value is below the returned table bound
make node-level qualification indeterminate; those nodes are never labelled
qualified from the bounded value.

Completed provenance records the exact node order, machine-readable link
topology, PCMCI+ options, input availability counts, full BH family size
(1,400,274), runner digest, and scientific runtime identities. Fitted matrices are
first retained, then their dimensions and finite eligible p/r values are checked
before an export may be declared complete. Invalid retained matrices receive
`fit_complete_validation_failed`, not completed/export-ready status.

The output directory is reserved before fitting. Before fitting, the runner
requires the pinned Tigramite VCS commit recorded by its installed
`direct_url.json`, and retains its commit plus SciPy and statsmodels versions.
After fitting, raw matrices are
atomically persisted before BH/export work, so a post-fit export failure retains
the expensive result. Tigramite fitting itself still has no solver checkpoint:
process or scheduler interruption during the fit requires a restart and must not
be described as resumable.
