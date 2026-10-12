# Bounded execution receipt — 2026-10-06

This is an execution amendment to the [working protocol](../co2-paper-protocol.md),
not preregistration or a change to the frozen campaign. Raw data, historical
products, unrelated dirty work and the standalone Overleaf clone remain untouched.
No training or bulk remote archive transfer was started.

## SABER: corrected pilot passes, full window blocked

**Later same-day resolution:** the internal audit found both copied-orbit anomalies
outside the prespecified region. The user explicitly approved region-scoped
validation, bound to the complete source-hashed audit; the
[full regional descriptive extraction](saber-regional-extraction-completion-2026-10-06.md)
is now complete (13,902 scans / 216 monthly rows), without shifting timestamps or
contacting providers. The historical stop and initial investigation below are
preserved as provenance, not a current provider-contact prerequisite.

The offline, read-only-input pilot (2002-01-25–2002-02-28) read 35 local archives /
42,539 archive scan records. It retained **59** fixed-region physical scans on
**23** UTC dates: 12 January scans / 5 dates; 47 February scans / 18 dates.
All 59 had finite official flux and available SLT/SZA; 51 had the full native
profile inside the region. The missing padding filename was 2002-01-24. No daily
completeness, representativeness or climate eligibility is implied. An independent
output audit checked hashes, fixed region, physical IDs, UTC ordering, monthly
means/medians/counts and SLT/day-night accounting.

Two real-data stops were investigated, not silently bypassed:

1. `SABER_CO2_PROFILE_FLUX_2003235_V1.0.nc`, scan 409, orbit 9239,
   event **0**, reference latitude −11.2022°, longitude 288.3752°. The source
   description explicitly says event is **1-based**. The extractor now records
   invalid nonregional/unplaceable IDs in quarantine provenance; invalid
   **regional** IDs still fail. This is independent of flux/outcome values.
2. Orbit **71879**, event **2**, appears in `2015073_V1.0` scan 1278 and
   `2015074_V1.0` scan 0. Both store the same reference clock **86,445,936 ms**
   (float32), geometry (2.25361°N, 6.97460°E), flux and point-rate/SLT/SZA.
   Adding each file's stated day assigns timestamps **one day apart**. This is
   not float32 rounding (half-spacing only 4 ms). The full candidate window
   **2002-01-25–2019-12-01** fails the physical-identity/UTC gate and publishes
   **no completed output**. Do not repair this by subtracting a day, dropping
   the conflict or changing the scientific window without a documented decision.

Provider clarification is needed: which daily archive supplies the authoritative
scan date, whether copied orbit-boundary records retain the previous file's
millisecond origin, and whether a corrected cooling-profile archive exists.
Requalify the processing lineage and all conflicts before a full monthly product
or climate/cooling trend claim. Existing contact-update V1.1 metadata is not itself
proof of a scientific correction.

The official [cooling-product README](https://data.gats-inc.com/saber/Version2_0/SABER_cooling/Documentation/README_SABER_CoolingRateProfiles_Dataset.pdf),
sections 3–4 and the example header, defines milliseconds relative to midnight
of the **day in which the orbit started** and says each daily file contains orbits
beginning that day. It does not resolve this duplicated orbit's inconsistent
file-day assignment. Section 1.3 asks for early instrument-team contact, normally
offering coauthorship and appropriate acknowledgment before publication. No
provider message has been sent.

Local ignored receipts/products live under
`data/products/co2-paper-execution-20261006/`. Pilot v1 predates the corrective unit
and date-boundary checks; v2 has those checks and identical scan/month file hashes:
`12c7cc0b9216433bf84807ed0c654b4758b0c92977bd98f7be2802a66b8dba30`
and `6020bf9ecdde68d770d11afe2d0ee8f809e8b7f4b0a5b67b3fe1787804cf20fb`.

## Karman: all ten passes and paired-month report completed and verified

An independent snapshot is staged under Phoenix
`/home/sergio/karman-phoenix/paired-date-20261006/`, not in the frozen root/repo.
SGE uses four CPU slots on `all.q@auriga`, CPU float32, batch 256, zero loader
workers, no GPU/autocast. Bubblewrap disables networking, clears the environment,
mounts code/source/checkpoints read-only and exposes only a separate writable
output directory. The old scheduler/paused study was not restarted.

- Pilot job **1449** failed closed before inference: PyTorch 2.5.1 rejects the
  tuple-form safe-global alias API introduced in 2.6. Compatibility now uses
  only real audited NumPy reconstruction callables on older PyTorch; unsupported
  pickle names still fail. `weights_only=True` is mandatory throughout.
- Pilot job **1450** completed successfully: checkpoint
  `confirm__paper_jb08_omni__co2__seed_00`, **65,536 rows / 8 prefix dates**,
  **5.3421 s** sidecar elapsed (12 s scheduler wall time), PyTorch **2.5.1+cu121**,
  NumPy **2.4.6**. The cache is explicitly `pilot=true`, `complete=false`; this
  does not establish full-sample canonical MAPE parity.
- Full gate job **1451** passed: **10,058,908 rows / 434 dates / 16 months**,
  793.3182 s sidecar elapsed (800 s scheduler wall), unchanged frozen identities.
  Independent JSON arithmetic reconciled all twelve category counts exactly;
  maximum absolute MAPE discrepancy **5.39699e-7 percentage points**, well below
  the unchanged **1e-4 pp** tolerance. Overall MAPE for this one CO₂/seed-0 model
  is 12.4847272873%, not the five-seed architecture mean. Cache SHA-256:
  `1e7b2591c4c60fb709b04b7423e347a3dfb9139521294ea04dd8da06a193ddee`.
- Remaining-nine array job **1452**, tasks 1–9, was submitted only after that gate
  passed. At most **three concurrent tasks × four CPU slots**; each has a
  30-minute scheduler/outer wall limit, 1,740-second shared inference deadline,
  6 GB per-slot virtual-memory request, unchanged CPU/batch/isolation recipe.
  Planned compute wall time was roughly 40–50 minutes (three waves); worst allocated
  inference budget **18 CPU-slot hours**, excluding prior pilot/full gate. No
  automatic retry or tolerance relaxation. Task mapping covers CO₂ seeds 1–4
  and no-CO₂ seeds 0–4, retaining the accepted CO₂ seed-0 cache. All nine tasks
  exited successfully (`failed=0`, `exit_status=0`), approximately 796–813 s each;
  the last ended at **12:28:04 CEST**. Muse monitored completion read-only.
- Report job **1453** was held on array **1452**, with one slot / five minutes.
  Scheduler dependency alone is not a scientific success gate: the report checks
  all ten complete/parity-accepted caches and current hashes before writing the
  2,000-draw fixed-five-seed paired-month bootstrap. Missing or invalid passes
  abort publication. It completed at **12:28:19 CEST**, `failed=0`,
  `exit_status=0`, after two seconds of scheduler wall time.

The final comparison anchor remains
`7247931f74fbcee8e904f11d7775289228041bf6a3b278e2038df188c9d3069c`.
The report is a fixed-five-seed mean of paired gains with shared year-month
resampling, not an ensemble, causal effect, seed-as-data uncertainty or future-year
test. Its SHA-256 is
`ceba851dfe3d4ed5dcfacd1774b0cdeecc68224d3b6070b40bf289b3ee6545e4`.

### Independent closeout and result

Primary verification checked all ten complete, non-pilot caches, matched five
seeds × two CO₂ variants, identical physical/date/category support, all **120**
canonical category counts and MAPEs, current checkpoint hashes (without loading),
frozen code/manifests/selection/subset-index hashes and report/cache SHA bindings.
Maximum absolute canonical MAPE discrepancy was **1.201924e-6 percentage points**,
below the unchanged **1e-4 pp** tolerance. Source-array size/mtime and source
metadata hashes matched the frozen plan; this is **not a full source-array rehash
or reproduction of training**. Ten inference passes still represent only
**10,058,908 distinct test rows**, not ten independent observational samples.

An independent local reconstruction used the algebraically equivalent mean paired
APE differences per month divided by common resampled counts, rather than
per-model MAPE resampling. NumPy PCG64 generated the prescribed 2,000 shared draws
(seed 20261005); manually sorted/interpolated quantiles and Python seed statistics
agreed with every reported category to **2.77556e-15 pp**.

**Primary overall gain:** **+0.072146852 pp**, separate paired-seed SD
**0.176470354 pp**, descriptive pointwise 95% paired-month percentile interval
**[−0.108074009, +0.270671739] pp**; all 16 months contribute, 2,000/2,000 draws
are valid. The interval includes zero: this analysis does **not establish a
reliably positive overall predictive increment** under its assumptions. It is not
proof of no CO₂ effect, equivalence, or a causal test. Existing canonical point
results remain unchanged within numerical parity.

| Category | Rows | Months | Valid draws / 2,000 | Gain (pp) | Pointwise 95% interval (pp) |
|---|---:|---:|---:|---:|---|
| Overall | 10,058,908 | 16 | 2,000 | +0.072147 | [−0.108074, +0.270672] |
| 200–250 km | 174,125 | 1 | 1,282 | +0.261797 | unavailable |
| 250–300 km | 792,205 | 4 | 1,985 | +0.057903 | [−0.388052, +0.287319] |
| 300–350 km | 947,011 | 5 | 1,995 | −0.143576 | [−0.448309, +0.171416] |
| 350–400 km | 1,089,340 | 8 | 2,000 | −0.156748 | [−0.537485, +0.129042] |
| 400–450 km | 1,521,726 | 9 | 2,000 | −0.226899 | [−0.378289, −0.022700] |
| 450–500 km | 4,517,183 | 16 | 2,000 | +0.245169 | [+0.006093, +0.547341] |
| 500–550 km | 1,017,318 | 12 | 2,000 | +0.175743 | [−0.181729, +0.498722] |
| Quiet (Ap 0–15) | 8,835,584 | 16 | 2,000 | +0.115426 | [−0.083563, +0.324130] |
| Mild (Ap 15–30) | 952,176 | 12 | 2,000 | −0.175019 | [−0.446280, +0.048386] |
| Minor (Ap 30–50) | 193,388 | 9 | 2,000 | −0.396463 | [−0.820744, −0.033236] |
| Major (Ap 50+) | 77,760 | 2 | 1,762 | −0.653526 | unavailable |

Unavailable intervals satisfy neither the required ≥4 contributing months nor
≥95% valid draws. The eleven strata are exploratory and multiplicity-uncorrected;
their intervals excluding zero are not confirmatory discoveries. Month resampling
assumes conditional exchangeability of only 16 nonconsecutive, unequally sampled
clusters; dependence across months, calendar drift and lost mission identity are
not resolved. The mixed training backend policy and the five fixed seeds are
preserved, not resampled into a combined uncertainty claim.

Ignored local evidence: `karman-all-pass-audit.json`,
`karman-bootstrap-reconstruction.json`, the ten `completed-karman-runs/` caches
and the original `paper_jb08_omni-paired-month-bootstrap.json`. These contain
processed aggregate losses/provenance, not downloaded raw arrays or checkpoints.

## Validation and remaining gates

The new synthetic seams currently pass **20 SABER / 19 Karman** tests, including
red→green regression checks for real-data failure patterns and a literal two-row
public inference fixture. Focused Ruff checks and SABER error-level typechecking
pass. After the real-data compatibility repairs, full sandbox suites passed
**466 Thermodense + 267 Karman** tests (one Thermodense skip). Subsequent identity
audit/regional-mode additions passed **499 Thermodense tests** with one skip;
the Karman source was unchanged. Initial broader
Thermodense sandbox failures were
missing public fixtures, minimal Git metadata, TeX runtime and a cleared USER,
not changes to scientific application code. Independent repair review found no
remaining code blockers; scientific source/parity gates are separate.

WACCM unit/postprocessor/coverage/licensing questions and the outcome-uninspected
calendar-control design remain open. The original 5 October supervisor PDF is
preserved; its separately dated three-page execution addendum includes the
completed, independently verified uncertainty result, all-strata vector plot and
explicit limitations. The combined supervisor packet has eighteen pages.
