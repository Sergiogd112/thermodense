# SABER fixed-region CO₂ official flux (descriptive)

From an **already local** CO₂ cooling-profile NetCDF3 directory, run in an
appropriate isolated environment:

```sh
python -m thermodense.saber_fixed_region \
  --input-dir data/original/saber/co2_cooling_profiles \
  --output-dir data/products/saber-fixed-region-2002-2019 \
  --start 2002-01-25 --end 2019-12-01
```

This is the backward-compatible **global** validation default: any valid
physical pair with different file-origin UTC clocks or conflicting payloads
fails, even outside the region; any missing UTC clock fails. To apply the
approved **regional** gate instead, use the same arguments plus:

```sh
  --validation-scope regional \
  --identity-audit-dir data/products/co2-paper-execution-20261006/saber-identity-full-2002-2019-v1
```

Regional mode requires a complete, independently produced identity audit of
the *exact* chosen highest-version source filenames, SHA-256 hashes, start/end
dates and padded archive window. It requires `COMPLETE`, matching report and
provenance bindings, the approved audit producer code SHA-256
`ef77f8a3db993ad4c2f1983c442333d195fc6dc2789ca92fb4499de4a2aa7727`
(the audited version of the fixed 16–25°N, 200–210°E, nearest 119 ±0.5 km
contract), and zero in-window regional issue groups, regional unknown-clock
groups, invalid regional IDs and invalid regional clocks. The default operator
anchor is the approved full-audit **report** SHA-256
`21fa31e144393d672ad840884c51166bd33db29ef3d0dee1d4c794cb56b2c247`.
For a separately reviewed audit (including synthetic tests), explicitly pass
`--expected-identity-audit-sha256 <64-hex-report-digest>`; this is an operator
trust decision, **not** a way to bypass the audit's source/code/completion
checks. The audit's `production_code_sha256` records the *pre-change*
extractor and is retained as lineage, not compared to this changed extractor.
The audited producer version is the geometry contract; the report does not
serialize ROI constants. Do not substitute a self-declared `complete` JSON
without its operator-approved report hash and provenance/COMPLETE binding.

Python 3.14 with NumPy, SciPy (`scipy.io.netcdf_file`) and Polars is required.
The output **parent must exist**, the output directory must not exist, and the
input directory is never modified. Choose a new directory for each run; do not
point this at the legacy daily or HASDM-matched 3-hour output paths. No network,
download or density alignment is performed. Linux `renameat2` is required for
atomic no-replace publication. A complete output contains `scans.parquet`,
`monthly.parquet`, `provenance.json` and `COMPLETE`; failed writes remove the
private staging directory. Provenance includes both output SHA-256 digests and
row counts. Input hashes are checked during reading and again before publication.

Public API: `prepare_fixed_region(input_dir=Path(...), output_dir=Path(...),
start=date(...), end=date(...), validation_scope="global",
identity_audit_dir=None, expected_identity_audit_sha256=None)
-> (scan_frame, monthly_frame)`. Inclusive dates
filter on the **119-km UTC timestamp interpreted under the recorded file-origin
rule**, rather than filename date alone. This is not independent proof of absolute
physical clock accuracy. One previous archive day is read to
retain midnight spillover at the beginning of the window. Only the highest
numeric filename `Vmajor.minor` for each archive day is read (as in the cooling
downloader inventory). Filename, `File_Name`, `Dataset_Version`, `Dataset_Name`,
year/day headers and calendar DOY must agree. UTC `time` and `slt` must declare
`msec`: UTC is file midnight plus 119-km milliseconds (allowed [0, 48h)); SLT
is milliseconds / 3,600,000 (allowed [0, 24h)). Missing/invalid UTC clock
fails the extraction rather than silently assigning a month for selected
regional scans (and for all scans in global mode). In regional mode, invalid
nonregional clocks are quarantined without a UTC month, with raw source IDs,
raw clock and reference geometry in source provenance.
Timestamps have an explicit UTC timezone and microsecond precision; fractional
milliseconds are retained at that precision. Scan/month schemas remain fixed
even when no regional scans or valid flux values are present.

`scans.parquet` is one row per physical scan geographically inside the **closed**
16–25°N, 200–210°E region at the nearest native 119-km level (within 0.5 km).
Latitude/longitude are validated against declared fills/ranges and physical
coordinate ranges *before* converting longitude to east [0, 360). Scan-level
`official_flux_w_m2` is the provided 100–139-km integrated flux; it is **not**
recomputed from cooling-rate levels. Declared altitude, official-flux and
point-rate units are checked before extraction. `co2cool_119km_w_m3` is the
nullable native *point* rate; the source variable is required. Finite negative
rates and fluxes remain valid. Missing
SLT/SZA/flux are flagged in `qc_flags`, with nullable fields and `valid_flux`;
they do not remove geographically selected rows. SZA <90° is the descriptive
day category, >=90° night, missing unknown. `full_profile_in_region` checks
every stored altitude's geometry as a labelled sensitivity flag, never as the
primary selection. Geographically missing/outside scans are counted separately
in provenance, not mislabelled as observed regional rows.

Duplicate physical identity is `(orbit, event, 119-km UTC timestamp)`, **not**
source filename. Identical duplicates retain the first selected source and
`duplicate_sources`/`duplicate_records` count; conflicting reference geometry,
QC, flux, SLT/SZA, native rate or footprint fail before output publication.
Orbit/event identifiers must be positive integers (the product describes event
as 1-based). Invalid **regional** identities fail; nonregional/unplaceable
identities are quarantined in source provenance with scan index, raw IDs,
reference geometry and raw time, and counted separately rather than silently
accepted as zero-based. This rule followed a real out-of-region event=0 record
in `SABER_CO2_PROFILE_FLUX_2003235_V1.0.nc`; it is not a flux-dependent exclusion.
A repeated valid pair with a
different timestamp is ambiguous and fails, including across the UTC-window
boundary. Selected identities are retained; excluded-scan bookkeeping is bounded
to the possible adjacent-day overlap. Full footprints are calculated only for
reference-level-selected scans. Out-of-window counts describe archive records,
not a deduplicated regional population.
Under regional validation, repeated pairs with differing file-origin UTC or
payload remain diagnostics **only if every member is nonregional**. The full
audit catches distant/outside-first reuse that bounded overlap bookkeeping
cannot detect. Any regional member with a conflicting identity is fatal;
identical same-UTC duplicates retain the existing physical dedup policy.
The audit's nonregional duplicate groups (including source/scan identities),
global counts, four zero regional issue counters, source hash bindings and
report/provenance digests are preserved in output provenance. Conditional
file-origin UTC counts for nonregional/out-of-window scans are **not corrected
physical timestamps**. Source and audit hashes are rechecked before atomic
publication. Neither mode repairs clocks, changes date selection, applies
flux/SLT/SZA sign filters, or requests provider outreach.
`monthly.parquet` keeps every calendar month, null flux statistics for empty or
invalid-flux months, scan-weighted mean/median of valid official flux, geographic
and valid-flux counts, distinct UTC dates, distinct orbit numbers, day/night/
unknown-SZA counts, six 4-hour SLT bins and unknown-SLT count, missing-flux and
full-profile counts. It never fills gaps. These are **sampling summaries**, not
an eligibility threshold, precision guarantee, independent daily observations,
or climate inference.

`provenance.json` records region, reference/time/fill/dedup rules, actual chosen
source names, SHA-256, versions and scan counts, local input manifest/inventory
hashes if present, code SHA-256 and runtime package versions, missing **local
filename days** (including the start padding day), out-of-window and rejected
geometry counts. A missing filename is not proof of an officially missing remote
day; even present files do not guarantee a regional observation. The cooling
V1.1 header documents a **contact update**, not an established radiometric
break. Product-team processing lineage and sampling qualification remain open
before any climatological/CO₂ attribution claim.

The historical nearest-per-altitude daily product has official flux but no
bounded geographic radius and can mix scans across levels. The current HASDM
3-hour product lacks official flux and SLT and uses a moving reference cell.
Neither is an input or is overwritten here.
