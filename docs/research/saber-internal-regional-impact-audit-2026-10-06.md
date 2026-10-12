# SABER internal audit: regional impact, not a timestamp repair

Date: **2026-10-06**. User chose an internal investigation rather than provider
outreach. **No message sent, timestamp shifted, region widened, scan removed from
a regional product, or production extraction policy changed.**

## Question and method

Does the global orbit/event validation failure actually affect the prespecified
**16–25°N, 200–210°E** region at native **119 km**, in the candidate inclusive UTC
window **2002-01-25–2019-12-01**?

`scripts/audit_saber_scan_identity.py` indexes every valid positive orbit/event
pair in temporary SQLite, including nonadjacent reuse, instead of stopping at the
first anomaly. It reads the highest numeric filename version per archive day,
including one previous-day padding file when available. It distinguishes raw
clock, conditional file-origin UTC, numeric measurements, metadata and region/
window membership. Only repeated identities trigger full-profile comparisons.
Invalid IDs/clocks/geometry remain diagnostic records, not silently dropped.

Execution: network-disabled Bubblewrap, source/code read-only, one thread,
4 GiB address-space limit, 30-minute outer wall/CPU cap, 1,740-second audit deadline,
2 GiB SQLite disk guard and explicit duplicate/invalid-detail caps. New output
published atomically only on completion; SQLite was closed and removed. The
audit is not a flux extraction, climatological analysis or authoritative UTC rule.

## Complete local-window results

| Quantity | Count |
|---|---:|
| Local daily files processed | **6,482** |
| Archive scan records | **8,624,986** |
| Positive-ID records | 8,624,985 |
| Unique positive orbit/event pairs | 8,624,809 |
| Repeated-ID groups | **176** |
| Records belonging to repeated groups | 352 |
| Region-selected records in the interpreted UTC window | **13,902** |
| Unique selected positive orbit/event pairs | **13,902** |
| Repeated-ID groups with any regional member | **0** |
| Regional invalid-ID / invalid-clock records | **0 / 0** |
| Invalid-ID records globally | **1**, outside the region |
| Missing local filename days, including start padding | **39** |

All 176 groups comprise two identical raw scan payloads with different interpreted
UTCs. Numeric fields, native altitude grid and relevant unit/fill/range metadata
matched. Every difference is **86,400 seconds** under the file-header-origin rule.
There were no observed different-measurement or regional clock-reuse groups.

The groups belong to just **two copied orbits**:

- Orbit **71879**: 91 event pairs across `2015073_V1.0` / `2015074_V1.0`.
- Orbit **76369**: 85 event pairs across `2016010_V1.0` / `2016011_V1.0`.

The lone invalid event-0 record is the previously identified `2003235_V1.0`,
orbit 9239, scan index 409, outside the region. These are measured diagnostic
patterns; no claim is made that a boundary copy was intentional or which global
timestamp should replace either occurrence.

## Independent verification

A separate NumPy-vectorized reader, without importing the production helpers or
SQLite audit, reread all **6,482** files. It rechecked every source SHA-256,
independently applied declared numeric fills/ranges and the fixed geographic mask,
confirmed 13,902 unique regional positive IDs with valid clocks, and compared all
six stored profile vectors plus scalar flux for every repeated group. It confirmed
the two-orbit counts, zero regional reuse, the one nonregional invalid ID and the
one-day conditional UTC differences. An independent review found no remaining
code blocker to this narrowly worded regional-impact finding.

The complete report SHA-256 is
`21fa31e144393d672ad840884c51166bd33db29ef3d0dee1d4c794cb56b2c247`.
Audit implementation SHA-256 at execution:
`ef77f8a3db993ad4c2f1983c442333d195fc6dc2789ca92fb4499de4a2aa7727`.
The production extractor SHA-256 remained
`8162118d653a61abc979af0fc8c8c365b1d635ef0f964fb5d06ddb43cc098160`.

Ignored local evidence under `data/products/co2-paper-execution-20261006/`:
`saber-identity-full-2002-2019-v1/{report.json,provenance.json,COMPLETE}`,
`saber-identity-regional-verification.json`, and
`verify_saber_regional_audit.py`. Synthetic audit tests: **17 passed**; together
with the unchanged extractor tests: **37 passed**. Focused Ruff passed.
The full Thermodense sandbox suite subsequently passed **483 tests**, with one
skip; no production-policy change was made.

## What this changes—and what it does not

The current extractor still fails because its fatal repeated-ID clock check covers
**all** global scans. That was a conservative implementation choice, not evidence
that a selected regional scan has conflicting copies. The audit demonstrates
that these observed failures have **no selected regional member** in the local
files inspected. It does not prove absolute clock accuracy, remote completeness,
quality of missing days, or climatological representativeness. Conditional UTC
remains the documented file day plus reference-level milliseconds.

### Recommended handling for user review—not implemented

1. Keep the current clock interpretation; **do not repair or choose a global
   duplicate timestamp**.
2. For a **regional descriptive extraction only**, retain nonregional anomalies
   as explicit source diagnostics/provenance rather than fatal regional gates.
3. Continue to fail on invalid/conflicting regional identities or clocks. A reuse
   group with any regional member or changed geographic eligibility must not be
   quietly dismissed as an excluded global record.
4. Bind the approved extraction to the audit/source hashes, disclose the 39 missing
   local filename days, and keep sampling/local-time/day-night qualification open.

This spatially scoped validation is a reasonable way to proceed without making
provider contact a prerequisite for the fixed-region descriptive table. It is
**not approval to alter the existing policy yet**, nor a global-data correction.
Global cooling products or climate/attribution claims may still require additional
time-origin and processing-lineage qualification.

**Subsequent user decision, same date:** the user explicitly approved applying
regional gates. The [completed regional extraction](saber-regional-extraction-completion-2026-10-06.md)
now supplies 13,902 scans and 216 monthly summaries, with unchanged timestamps and
retained global diagnostics. The recommendation above records the earlier decision
stage; it is no longer an unresolved approval or provider-contact blocker.
