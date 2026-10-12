# CO₂ paper data qualification: first bounded pass

Date: 2026-10-05. **Partial qualification, not analysis-ready clearance.**
Existing local/remote samples and metadata were read. No raw dataset was changed,
bulk transfer started, new scientific fit run or Overleaf content edited.

## Status in one table

| Product | Verified now | Gate before new paper use |
|---|---|---|
| Karman frozen benchmark | 30 evaluation/checkpoint hash bindings, independent aggregate replay, 434 test dates/16 months | Per-date paired loss extraction before temporal confidence intervals |
| WACCM-X regional provision | 74 archive listing, bounded endpoints for 73, one regional NetCDF CRC/SHA/56-variable schema | Units/He/DEN and averaging clarification; full date/QC manifest; native-height support |
| SABER raw CO₂ cooling | 8,537 local files; five metadata/sample checks; official per-scan flux and local-time fields present | Processing lineage, fixed-region/raw-scan extraction, sampling/QC eligibility |
| SABER legacy nearest-scan daily product | Flux column exists, 119-km geometry/date audit performed | **Not a fixed-region local measurement**; cannot silently reuse as the new primary diagnostic |
| SABER 3-hour HASDM-matched product | Schema/row count inspected; point rates/counts retained | Flux/local-time metadata absent; geometry differs from WACCM fixed box |

## 1. WACCM-X: keep the existing qualified facts, no new bulk ingestion

The [regional inventory](waccm-x-regional-inventory-2026-10-05.md) identifies
UKSSDC/SERENE WACCM-X 2.1/CESM 2.1.3, 74 provided yearly archives (~152.17 GB),
a 5×5 grid-point region, 126 levels and hourly sample spacing. Missing 2022,
1950's January 2 start, nonstandard 2023 packaging, absent ZIP central-directory
records and sample-only CRC verification remain explicit limitations.

No additional archive was ingested in this preparation pass. Key unresolved
issues remain fractional boundary CO₂ versus the provider ppm label, `EDens`
inverse-volume units, `He` values above 1 despite a mol/mol label, `DEN`/helium
post-processing involving MSIS, time-averaging and hybrid-height conventions,
and model segment joins. The 56-variable schema has **no direct CO₂ IR cooling
budget field**. Historical model states are not a controlled intervention.

Candidate small pilot: one nearest grid column at 19.894737°N/205°E; 300/400 km
only after native-height checks; a fixed 16–25°N/200–210°E comparison region.
Define actual masks before choosing a common window, provisionally
2002-01-25–2019-12-01. Do not extrapolate, fill 2022 or widen the domain because
an effect is weak. Provider clarification and storage/transfer approval precede
any representative-year or longer ingest.

## 2. SABER raw holdings and direct metadata checks

Files are already local under ignored
`data/original/saber/co2_cooling_profiles/`. Filename inventory finds **8,537
files**, 2002-01-25 through 2025-07-20, with **41 missing archive-day names**
within that span. This is not a complete payload/QC manifest or a measure of
regional observation coverage.

Five existing files were read and hashed: 2002-01-25, 2014-01-01, 2019-01-01,
2020-01-01 and 2025-01-01. Each has 40 stored altitude points, **100–139 km at
1-km spacing**, scan-level `flux` (W/m²), `CO2cool` (W/m³), orbit/event, per-
altitude geolocation, time, solar local time and solar zenith angle. Grid spacing
is not an assertion of independent 1-km physical resolving power.

- UTC `time` and `slt` are labelled **milliseconds**. Sample SLT ranges are
  consistent with a 0–24-hour clock after division by 3,600,000; validate fills
  and ranges explicitly. Do not treat these values as hours unconverted.
- Files group **orbits beginning in a named day**, not necessarily all scans
  observed that day. The 2014/2019/2020/2025 samples contain 24/42/12/52 scans
  whose 119-km UTC time exceeds 24 hours. Preserve `file_day + milliseconds`
  rollover and bin by actual timestamp.
- Samples include **negative point cooling rates**. Do not classify all negative
  values as missing by assumption; distinguish declared fills/QC from net-rate
  interpretation with the product team. Provided integrated flux is a different
  quantity; no positivity rule is invented here.
- At a 119-km geographic reference, only **2–3 scans per sampled file** fall in
  the proposed fixed region. These are finite/geometry checks, not final QC or
  a general lifetime coverage claim. Sampling-aware monthly/climatological
  aggregation is worth evaluating before claiming daily causal resolution.

The [official README](https://data.gats-inc.com/saber/Version2_0/SABER_cooling/Documentation/README_SABER_CoolingRateProfiles_Dataset.pdf)
describes profiles and integrated CO₂ flux over 100–139 km, derived from Level 2
products. Its native-method information must not be confused with stored grid
spacing or product update cadence.

### Version lineage—not a guessed processing break

The selected files mix cooling-product **V1.0/V1.1**; the V1.1 global attribute
explicitly says **“updated dataset contact.”** Both identify underlying SABER
data only as **“2.0.”** This is not enough to identify 2.07/2.08 processing or
establish a radiometric change at a V1.1 label.

Current [SABER notices](https://saber.gats-inc.com/data.php) describe reprocessing
after the 2019-12-15 assumed-CO₂ change and a 2026 altitude-registration source
change. Trace whether/how these apply to the cooling files used, rather than
assuming they do. The Level2C temperature/CO₂ concentration withdrawal is **not**
a blanket cooling-product withdrawal. Contact the product team for appropriate
use, processing history and publication expectations before a strong climate claim.

## 3. A material legacy-product problem: “nearest” is not bounded local sampling

`data/decoded/saber/saber_co2_cooling_maunaloa_daily.parquet` has 316,640 rows,
40 levels and a 2002-01-25–2023-11-10 file-date span. Contrary to the first
3-hour-pipeline-only map, **this older product retains `flux_w_m2`**, plus
source file, timestamp, scan index, nearest geolocation and distance.

However, `scripts/decode_saber.py:49–89` selects **one nearest finite scan
separately at each altitude with no maximum distance**. It keeps the file date
even when the timestamp rolls into the next UTC day. It does not retain orbit,
SLT/SZA or all candidate scans. Flux may be repeated across altitude rows and
the selected scan can differ by altitude—do not sum those flux rows as a column
integration or assume all levels form one observed profile.

Direct audit of the **7,916 119-km rows**:

| Check | Finding |
|---|---:|
| Median existing distance metric | 3.050334 angular degrees |
| Maximum existing distance metric | 170.106256 angular degrees |
| Distance metric greater than 5° | **1,444 rows** (~18.2%) |
| Geolocation inside fixed 16–25°N/200–210°E region | **6,413 rows** (~81.0%) |
| Stored `date` differs from UTC date of `timestamp` | **107 rows** |

This metric is the decoder's cosine-weighted angular-distance approximation,
not a verified geodesic distance. A worst example is a 2003-07-18 selected
119-km point at −25.67° latitude/30.42°E; it is not a Mauna Loa-region sample.

**Impact:** the legacy artifact cannot be described as a bounded local cooling
record. This does not invalidate every association or the raw SABER measurements,
but local-mechanism wording and current figure provenance require correction
before reuse. Do not erase the historical artifact or silently re-render old
claims with a new sample frame.

## 4. The current 3-hour product has a different limitation

`saber_hasdm_maunaloa_3hour.parquet` has **74,657 rows**, 2000-01-01 through
2025-07-20. It retains 15 point-rate/proxy channels at 100/119/139 km and their
counts. It does **not** retain integrated flux, per-scan orbit, SLT/SZA or full
profile/geolocation information.

Its matching frame is the HASDM latitude band 15–25°N and a **per-slot moving
longitude-centred cell ±7.5°**, not the WACCM fixed 200–210°E region. A fixed-
region cooling diagnostic cannot be recovered by changing the column label or
integrating three stored point rates. Re-extract the required fields from the
already local raw archives at a newly declared seam; no new downloader is needed.

## 5. Minimal corrective seam, not a pipeline rewrite

The next separately specified extractor should emit one row per selected scan:

`source_file/hash/product version, orbit/event, actual UTC timestamp,
lat/lon at declared 119-km reference, SLT hours and SZA/fill/QC,
official flux W/m², sampling-frame identity`.

Use the same selected scan/profile once, retain rejected/missing geometry and
coverage counts, preserve rollover, and deduplicate explicitly. Monthly products
must retain distinct observed dates, orbit counts and local-time support, not
only an average. Decide eligibility/standardization on sampling information
before viewing density/cooling effect estimates. A stricter full-profile
geographic footprint can be a labelled sensitivity, not an unannounced sample
change. No extractor or new climatological analysis was implemented here.

## Receipts and next actions

Ignored local receipts: `data/products/co2-paper-preparation-20261005/`:

- `saber-qualification-samples.json`: five raw hashes/full attributes, filename
  inventory and prepared schemas. SHA-256:
  `b9ad519422a18113a0ea14dc32912a94d6120d0a36840d8c2f3b0b0f874d6544`.
- `saber-legacy-geometry-audit.json`: bounded 119-km Parquet audit retaining
  source hash, region definition, distance/date statistics and their limitations.
  SHA-256: `b1e794ba5607997e6f605864ed06e98f9726d0d13c993ffc0d089c48cb57eb1d`.
- `karman-test-support-attestation.json`: test-date/month counts and original
  ratio-support summary, with frozen code/index checks. SHA-256:
  `7b24aa667b33bc8d18b0b5da76d916604c3eb9e08c569d16bd0cd95e6ddba6c7`.
- Raw download-manifest hash:
  `1c24fdf24bb3db2a15cde2cc0e34c8f388ae7a1dad3e963b88fa9a268f5778b4`.
- Raw remote-inventory hash:
  `7b239ff52d2aebac6c57243964a80a16d430d08320c8da0de48320f64ee3f4de`.
- Legacy daily Parquet hash:
  `86cbcc148d9656184e46da83e833603f73ff80b975e61daa21ee135a0a04e30a`.
- HASDM-matched 3-hour Parquet hash:
  `6eb44a05140ae9330e6f47f217f32271b77610c89f28835d03a956b1830dec2a`.

**Concrete priorities:** fix/declaration of SABER selection and sampling; preserve
the completed Karman point-result closeout; approve a bounded per-date inference
pass if temporal uncertainty is needed; ask WACCM/SABER providers the unresolved
units/lineage questions. Nothing here justifies new architecture fitting, broad
causal matrices or a bulk archive transfer.
