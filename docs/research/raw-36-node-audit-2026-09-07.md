# Raw 36-node runner audit — 7 September 2026

## Scope and disposition

Read-only audit of the current, untracked `raw_3hour_pcmci.py` and
`raw_3hour_input.py`, their focused tests, and the reused stationarity diagnostic
in `pcmci_real.py`. Repository HEAD was
`17e244c05397a89182d245e3fac447c7d20f804b`; these additions are not included in
that commit. Requirements: [#59](https://github.com/Sergiogd112/thermodense/issues/59)
and [#60](https://github.com/Sergiogd112/thermodense/issues/60).

Independent standards/provenance and scientific-contract reviews were checked
against local synthetic probes and remote input metadata. No running-job source,
input, environment, scheduler setting, or result file was changed. Job 1139 was
still running on `auriga` at 18:22 CEST on 7 September. This is not a completed-run
validation or evidence of a completion percentage.

**Disposition:** retain association/sensitivity-only status. Do not promote the
eventual CSV/provenance to paper-ready evidence without resolving support,
orientation, and stationarity-reporting limitations below. No input mismatch was
found; this audit alone does not justify an automatic restart.

## Follow-up implementation status — 10 September 2026

The replacement runner now addresses the prospective execution defects from this
audit: it requires and verifies core/SABER hashes, rejects escaping paths and
non-midnight calendars, reserves output before fitting, persists fit matrices
atomically before export, normalizes reverse contemporaneous orientation, labels
the availability count as pairwise-only, and unconditionally forbids causal
wording. These corrections apply to future runs only. They do not retroactively
complete, repair, or qualify failed job 1139, and no rerun has been launched.
Follow-up hardening also implements canonical calendar identity, OBS-only F10.7,
bound legacy-prepared provenance, pinned Tigramite VCS identity, and completed
metadata for the missing-data/BH/stationarity contract. The read-only monitor
checks those declared canonical identities and bytes, but does not replay the
1.4-million-row BH export or establish paper readiness.

The legacy prepared NPZ is now byte-bound to its adjacent producer provenance,
including the declared four-source hash set and Ap/Kp/MLO policies. This verifies
the prepared artifact binding, not the authority of the frozen producer's
upstream semantic declarations; those still require source-level review before
paper use.

## Standards and operational integrity

1. **High — hashes are recorded but not enforced by the runner.**
   `raw_3hour_pcmci.py:73-118` computes current core/SABER hashes without comparing
   them with the assembler's expected hashes. `raw_3hour_input.py:143-150` records
   those expected identities. This does not implement #60's fail-closed integrity
   check. The separate audit found that current source and output bytes match
   the assembly records; the vulnerability is not evidence of actual corruption.
   Persist checked identities and a source/environment fingerprint for future
   runs. Relative source paths also need an explicit base: in this run they
   resolve against the run root, not the manifest directory or SSH working directory.

2. **High — long-running results are not durably saved before postprocessing.**
   `raw_3hour_pcmci.py:210-227` calls `result_rows()` before writing the NPZ.
   Nonfinite p-values cause BH to raise (`145-147`), potentially losing an otherwise
   completed fit. Output-directory collisions are also checked only after fitting.
   NPZ/CSV writes are not atomic, and interrupted writes have no recovery protocol.
   This falls short of ADR 0002's atomic/durable artifact intent and #60's integrity
   requirement. A NaN failure has **not** been observed in job 1139. Future runs
   should reserve their output and persist raw results atomically before export.

3. **Medium — direct manifests bypass the UTC-midnight check.**
   The assembler enforces midnight (`raw_3hour_input.py:72-82`), but the loader
   checks only three-hour spacing (`raw_3hour_pcmci.py:89-92`). Its daily-hold check
   groups every eight rows from the first row (`64-70`), not actual UTC dates.
   This is a latent direct-manifest contract gap; the audited calendar starts at
   midnight and is unaffected.

## Spec and scientific reporting

1. **High — exported effective support is only pairwise availability.**
   `raw_3hour_pcmci.py:155-168` counts available X/Y pairs. Tigramite's CI arrays
   also exclude the lag warm-up and missing conditioning variables. A 16-row
   synthetic self-lag example reports 13 samples in the exporter, but Tigramite
   uses 12 after its `2*tau_max` cut-off and X/Y missingness, or 11 with a missing
   conditioning value. The full input length is not per-test sample size either.
   Rename the existing measure `pairwise_support`; actual CI support and
   conditioning dimension require test-level instrumentation or a validated replay.
   The saved graph/p/r matrices alone do not retain that trace. This remains an
   evidence-qualification blocker even if the computation finishes successfully.

2. **High — reverse contemporaneous density edges are marked undetected.**
   The BH family contains one lag-zero representative per density pair. For a
   significant `graph[0,1,0] == '<--'`, `result_rows()` emits node 0/node 1 with
   `detected=False`, because detection checks `mark.endswith('>')`
   (`raw_3hour_pcmci.py:161-168`). It emits no swapped node 1→node 0 row. The raw
   graph mark is preserved, so orientation is recoverable if the NPZ is saved.
   Normalize edge direction in export, retaining the single undirected-pair
   hypothesis in the BH family and distinguishing adjacency from orientation.

3. **High — “30 qualified” is not a defensible stationarity summary.**
   All 30 nodes labeled qualified in the stored diagnostic have KPSS returned
   p=0.01 and a warning that the actual p-value is **smaller** than that table
   bound. The diagnostic treats it as exact, yielding Holm-adjusted p=0.36 across
   36 nodes (`pcmci_real.py:888-905,962-1005`). The reported Holm arithmetic is
   not itself wrong, but this upper-bound calculation cannot establish that the
   true p-values would survive correction. Non-rejection does not establish
   stationarity in any case. Report **qualification indeterminate due to bounded
   KPSS p-values**, not 30 stationary nodes and not 30 proven nonstationary nodes.
   The overall causal-wording gate remains false because six other nodes fail
   the raw diagnostic's contiguous-span requirement.

## Six short-span nodes

The diagnostic's `sample_count` means **longest contiguous finite run**, not total
observations (`pcmci_real.py:923-945`). The raw wrapper requires 2,881 contiguous
slots (`raw_3hour_pcmci.py:178-190`), corresponding to 360 days between the first
and last observation. Counts below were recomputed from the canonical masks:

| Node | Available slots overall | Finite runs | Longest run (3-hour slots) |
|---|---:|---:|---:|
| MLO CO₂ | 65,537 | 754 | 672 |
| CO2cool | 40,092 | 12,293 | 7 |
| NOcool | 40,161 | 12,276 | 10 |
| O2_1delta_ver | 40,149 | 12,270 | 10 |
| OH_16_ver | 40,149 | 12,270 | 10 |
| OH_20_ver | 40,149 | 12,270 | 10 |

All six exceed 2,881 observations in aggregate; none has an uninterrupted run
that long. MLO's longest run covers 6 September–28 November 2010 (84 UTC days).
Daily-held slots are not independent daily measurements. SABER CO2cool's longest
run spans 00:00–18:00 UTC on 14 April 2013; the other four span 18:00 UTC on
8 November–21:00 UTC on 9 November 2015.

MLO was tested by the reused daily diagnostic because 672 exceeds its original
361-slot threshold. Its ADF p≈0.895 does not reject a unit root. The raw wrapper
then overwrites the node outcome with the stricter short-span label. SABER's
7–10-slot runs are too short even for the reused threshold, so no ADF/KPSS tests
were performed for them. The six short-span labels are **not six test rejections**.

This is an inability to qualify stationarity by the current contiguous-span
protocol, not proof that no lagged CI samples exist. Tigramite can use nonadjacent
valid observations with the configured missing-data policy. Do not interpolate,
compress the time axis, shorten the canonical lag window, or lower the gate
merely to obtain a pass. Actual conditional support still needs checking.

## Verified input and source identity

- Calendar: 74,657 × 36, from 2000-01-01 00:00 to 2025-07-20 00:00 UTC.
- Current core/manifest checksums match assembly provenance.
- Legacy prepared input, CelesTrak CSV, and exact 50°×70° SABER checksums all
  match their recorded source identities; the actually loaded SABER path matches too.
- Reconstructing retained density/Ap/Kp/MLO columns, the historical UTC-held
  `F10.7_OBS` values, and masked SABER channels reproduced **all values, masks,
  and timestamps** exactly, comparing missing values as equal. That pre-fix input
  did not enforce `F10.7_DATA_TYPE == OBS`, so this identity check is not evidence
  that it satisfies the replacement canonical profile's OBS-only policy. No
  provider observations were printed or copied into the repository.
- The three reviewed local Python files match their current remote source hashes.
  This checks present files, not a historical attestation of in-memory job code.

| Artifact | SHA-256 |
|---|---|
| core.npz | `2c5d2855ed04491c5c51917aff5d994856865d52dc405e9cccd0da980ca1d765` |
| manifest.json | `1c26d87febd701dd1e920582063b367d8921097a093545f20f5c779c4730cacf` |
| legacy prepared NPZ | `9887cac146434797a50b5c2cd39ca7cdb799a175d1fa764a88a46af5702807a4` |
| CelesTrak CSV | `860b2549709b85d0a1fa4bedc23ef96170c949f4b61f36affd5bd9e20605528f` |
| SABER final arrays | `5d6a579da237f90acd6dc879363ca8a58a1331b2a44ddcac84f37cec1155163a` |
| raw_3hour_pcmci.py | `fa3c6f730fe535695755f9e769904fccaf9244c6403a904e4c4f596640cfa045` |
| raw_3hour_input.py | `a6fa3fdf98402be646c0aa64a5b1125e7231fb174885e721436d0734bb2e6c40` |
| pcmci_real.py | `f78410c36611bef4d3a5193c6263e037dce29d6b69786e3d4361a7332c3f1d68` |

Remote versions observed: NumPy 2.4.4, SciPy 1.17.1, statsmodels 0.14.6,
Tigramite package version 5.2.10.1. A package version alone does not identify a
Tigramite fork commit.

## Validation and follow-up

Focused existing tests: **6 passed** (`tests/test_raw_3hour_input.py` and
`tests/test_raw_3hour_pcmci.py`). Separate synthetic probes reproduced the support
and reverse-orientation defects. The declared BH family has **1,400,274** members:
972 × 1,440 lagged candidates plus 243 driver/density and 351 unique density-pair
contemporaneous candidates. A single full-family correction is implemented;
no issue was found with the basic BH arithmetic, node ordering, or intended
candidate restrictions.

Recommended next work, without hot-patching job 1139:

1. Implement and regression-test corrected export and bounded-p stationarity
   reporting locally; preserve the frozen daily runner's behavior unless a
   separate change is explicitly agreed.
2. Add fail-closed input hash checks, output reservation, atomic raw-result saves,
   and exact source/environment provenance for future runs.
3. Design CI-test support capture/replay before promising paper-ready support
   tables. Determine whether additional computation is necessary after inspecting
   the actual completed artifacts; do not silently use pairwise overlap instead.
4. Validate job exit status, files and checksums at termination. Repair exports
   from retained matrices where possible, while keeping unqualified outputs out
   of causal-discovery claims.

Review summary: **3 standards/operational findings** (high: integrity/durability)
and **3 scientific-spec findings** (high: support, orientation, qualification).

## External method and provider references

Retrieved 10 September 2026. These sources support provider/method semantics;
the local SHA-256 values above establish only identity of the audited local bytes.

- CelesTrak, [Space Weather Data](https://celestrak.org/SpaceData/): provider
  format/source context for observed and predicted space-weather records.
- statsmodels,
  [`statsmodels.tsa.stattools.kpss`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html):
  KPSS hypotheses, returned table bounds, and boundary warnings.
- Tigramite,
  [documentation](https://jakobrunge.github.io/tigramite/) and the
  [pinned fork revision](https://github.com/Sergiogd112/tigramite/tree/7c7b177cfbff77e11d805ab04fc2647301da1951):
  PCMCI+, graph marks, lag/missing-data behavior, and implementation identity.
- GATS SABER, [instrument overview](https://saber.gats-inc.com/overview.php) and
  [data documentation](https://saber.gats-inc.com/documentation.php): provider
  context for the five derived SABER channels; local gridding remains a project
  transformation rather than a provider claim.
