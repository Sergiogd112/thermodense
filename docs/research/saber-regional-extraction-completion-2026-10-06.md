# SABER regional descriptive extraction — approved and completed

Date: **2026-10-06**. This closes the regional extraction blocker, not the global
clock-origin questions or climate/attribution qualification. No outreach occurred.

## Explicit decision and unchanged science

After the complete [internal regional-impact audit](saber-internal-regional-impact-audit-2026-10-06.md),
the user explicitly selected **“Apply regional gates”**. The approved behavior:
nonregional anomalies remain source diagnostics, while invalid/conflicting
regional identities/clocks remain fatal. The prespecified region, reference level,
UTC interpretation, requested dates and flux/QC rules are unchanged. No scan's
timestamp was corrected, no region widened, no effect-driven exclusion introduced.

The API/CLI retains **global validation as the default** for backward compatibility.
The completed run explicitly requested `--validation-scope regional` and bound
`--identity-audit-dir` to the independently verified full-window report SHA-256
`21fa31e144393d672ad840884c51166bd33db29ef3d0dee1d4c794cb56b2c247`.
Source filenames/hashes, audited producer version, complete status, dates and zero
regional issue counters must match; source/audit drift stops publication. This
also covers distant reuse missed by bounded excluded-scan bookkeeping.

## Completed output and independent checks

Ignored output:
`data/products/co2-paper-execution-20261006/saber-fixed-region-2002-2019-regional-v1/`.
Execution was offline with read-only inputs/code, a separate writable output,
4 GiB address-space limit and 30-minute outer CPU/wall cap. Publication was atomic
and no-replace. Existing historical products and earlier audit/PDF snapshots were
not overwritten as scientific inputs.

| Quantity | Result |
|---|---:|
| Chosen local source files | 6,482 |
| Source archive records | 8,624,986 |
| Unique selected regional scans | **13,902** |
| Distinct interpreted UTC dates | **5,463** |
| Calendar monthly rows | **216**, none empty |
| Monthly selected scans, range | 2–84 |
| Monthly sampled dates, range / median | 1–31 / **25** |
| Months below 10 sampled dates | 2 (descriptive count, not an eligibility threshold) |
| Selected scans with UTC date differing from archive day | **186** |
| Selected scans with entire stored profile inside the region | **12,197** |
| Invalid selected flux / unknown SLT / unknown SZA | **0 / 0 / 0** |
| Day / night selected scans (SZA <90° / ≥90°) | 5,010 / 8,892 |
| Negative finite native 119-km cooling values retained | 1 |
| Global duplicated IDs retained diagnostically | 176 groups / 352 members, all nonregional |
| Nonregional invalid-ID records quarantined | 1 |
| Missing local filename days, including start padding | 39 |

The two endpoint months are partial by the requested scientific window
(2002-01-25 start; 2019-12-01 end). All monthly statistics remain scan-weighted,
with no gaps filled. Overall six four-hour SLT bins contain respectively
**3,126; 2,379; 1,346; 1,256; 2,389; 3,406** scans. These unequal support counts
are a reason to examine local-time and seasonal sampling before any trend fit.
The official flux ranges from 9.3029637e-5 to 0.0052129454 W/m²; this is observed
sample support, not an accuracy claim or CO₂-response estimate.

Independent output checks recomputed every monthly mean, median, count, distinct
date/orbit total, SLT/day-night accounting and output hash. A second reader compared
**every selected row** against its raw source scan across 5,470 source files:
source SHA, orbit/event, conditional UTC, reference geometry, official flux,
native cooling, SLT/SZA and validity flags all matched. It did not import the
production scalar helpers. No new density alignment, trend fit, model training
or causal analysis was performed.

## Identity and reproducibility

- `scans.parquet`: **13,902 rows**, SHA-256
  `8bc998e1d526288909e8d112b42c0601b8d0a904cbaf0d490fbbae76397d36e5`.
- `monthly.parquet`: **216 rows**, SHA-256
  `c43ed174a559c7b2abd9f1a284cc56734c26b488c4fac6078626cb06178060c4`.
- Output provenance SHA-256:
  `cd9e3b183fc421aa66d6c29e122c5091fca047f5a26a4d4cb5b25a234af31642`.
- Extractor code SHA-256 at execution:
  `2f55ea25f78391e59c43289cee98ad34f5628ef1fa4c1a01ea9a90989d305d3c`.
- Local checks: `saber-regional-full-extraction-audit.json`,
  `saber-raw-to-regional-values-verification.json` and
  `verify_saber_extracted_values.py` beside the ignored receipts.

Validation: **53** focused extractor/audit tests passed in the offline sandbox;
full Thermodense suite **499 passed, one skipped**; focused Ruff and extractor
error-level typechecking passed. Independent code review found no concrete blocker
to the explicit regional mode. Unrelated dirty work and the frozen Karman campaign
remain untouched; the separate through-2025 campaign is assessed independently.

## Remaining gates

This is a **descriptive regional scan/month product** under the documented
file-origin clock convention—not proof of absolute UTC accuracy, remote archive
completeness, sampling representativeness or a climate/CO₂ effect. The 39 missing
local filename days do not mean exactly 39 missing regional observing days.
Study month-by-month season/SLT/day-night support and native-footprint sensitivity
before choosing climatological eligibility or fitting any secular trend. Preserve
the global diagnostics; do not use this scoped mode to claim a repaired global
cooling product. Provider contact remains optional, not a prerequisite imposed
by the resolved nonregional anomalies.
