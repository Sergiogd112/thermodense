# Verified WACCM-X regional inventory

Inspection date: 2026-10-05. Metadata/sample audit only; no bulk dataset download,
production extraction, new simulation or scientific fit. Account-specific paths
and connection identifiers are intentionally omitted.

## Identity and access

This is **UK Solar System Data Centre (UKSSDC) / SERENE, University of
Birmingham WACCM-X long-history output**, associated with Matthew K. Brown,
Sean Elvidge, Oliver Allanson and David R. Themens—not a verified WMO dataset.
Provider documentation identifies WACCM-X 2.1 within CESM 2.1.3, a transient
1950–2024 simulation **without specified dynamics**. One inspected regional
NetCDF identifies case `longHist_c`, CF-1.0 and the SERENE research group.

Existing project chats and a local archive README supplied the access route.
The provider's account directory was successfully listed over **anonymous FTP**;
that is the protocol verified here, not SFTP. Credentials and the user-specific
directory must stay out of the public repository.

Sources: [UKSSDC overview](https://www.ukssdc.ac.uk/waccm-x/),
[data description](https://www.ukssdc.ac.uk/waccm-x/data.html), local reference
archive README, provider directory/file headers, and the inspected NetCDF.

## Three different holdings—do not confuse them

| Holding | Verified state |
|---|---|
| Old local global test ZIP | `~/Downloads/WACCM-X_20241224-20241231_2026072811_03_23.zip`; 18,216,558,879 bytes, eight daily global NetCDF members plus README; listing verified, not all payloads re-extracted |
| Old Phoenix point samples | Three existing 2024-12-24/25/26 files at the previously selected 19.8947°N, 205°E grid point; 24 hourly samples and 126 levels each; these are **not** a completed 75-year download |
| New provider regional subset | **74 yearly archives**, labelled 16–25°N / 200–210°E, spanning 1950–2024 except **2022**; total listed bytes **152,167,302,713** (~152.17 GB decimal); this is the substantial provided subset |

The old downloader's pending block remains 1950-01-01–03 and its recorded PID
was not alive at inspection. It was not restarted or modified. The new regional
archives do not require resuming the old global seven-/three-day request scheme.

### Coverage qualifications

- The 1950 regional archive's first member is **1950-01-02**, not January 1.
- First/last local member headers and constant-size accounting were checked for
  **73** regional archives. They indicate daily files covering the named year's
  start/end, with 364/365/366 inferred records as applicable.
- **2022 has no regional archive in the inspected directory.** This is not a
  statement that the parent simulation lacks 2022.
- The **2023 archive exists**, but does not satisfy the same fixed-record-size
  shortcut and needs separate indexing. Its full internal coverage is unverified.
- Intermediate timestamps, every member CRC and all historical schemas were
  **not** audited. File names and endpoint checks are not a complete date/QC
  manifest. A full ingest must enumerate actual records and retain gaps.

## Archive packaging

Bounded tail reads found no classic ZIP end-central-directory record in the
regional archives. The standard-size cases consist of uncompressed local ZIP
records; first/last headers can be read directly with FTP restart offsets.
This explains why a conventional ZIP-index reader can fail without proving
that each embedded NetCDF is damaged.

One representative 2014 member was read using its local header, verified against
the embedded CRC, and opened successfully as NetCDF. Do not promote this to a
whole-archive integrity claim. Obtain a provider manifest/repackaging guidance
and validate each member before scientific ingestion.

The successful regional audit read **5,649,185 bytes** on Phoenix, including
one ~5.63 MB member and small endpoint headers. The initial unsuccessful
central-directory probe read ~9.70 MB of archive tails. Only metadata were
returned locally; the 152 GB archive set was not transferred.

## Representative provided file: actual metadata

- Archive: `WACCM-X_2014-01-01_2014-12-31_16-25_200-210.zip`.
- Member: `WACCM-X_2014-01-01_16-25_200-210.nc`.
- Uncompressed size: **5,630,339 bytes**; CRC verified.
- SHA-256: `d702f7829e927024a8aefc11736dd748d410cdab34b3b53411b5ee9b7ae978fd`.
- **56 NetCDF variables**, including coordinates and bookkeeping—not 56
  independent physical predictors.
- Dimensions: `time=24`, `lev=126`, `ilev=127`, `lat=5`, `lon=5`, `nbnd=2`,
  `chars=8`.
- Time: 2014-01-01 00:00 through 23:00, **3,600-second spacing**, Gregorian;
  `days since 1997-01-01`. Decode time separately for each simulation segment.
  Averaging versus instantaneous semantics still require `time_bnds` /
  `cell_methods` checks; hourly spacing alone does not settle them.
- Latitudes: **16.105263, 18.000000, 19.894737, 21.789474, 23.684211°N**.
- Longitudes: **200, 202.5, 205, 207.5, 210°E**.
- Horizontal spacing: about **1.894737° latitude × 2.5° longitude**. This is
  a **5×5 grid-point subset**, not 5°×5° resolution or a global dataset.
- `lev` is labelled hybrid pressure-like midpoints in hPa,
  `1000*(A+B)`; not altitude. Pressure reconstruction must use
  `hyam*P0 + hybm*PS` as appropriate, rather than treating nominal `lev` as
  actual pressure everywhere.
- Sample `Z3` is geopotential height in metres. Lowest-level heights span
  **0.065–0.123 km**; top-level heights span **559.639–681.185 km**, depending
  on hour/cell. All inspected profiles are strictly descending in stored order.
  The old December 2024 point sample reaches ~846.7 km. Neither range is a
  verified full-history coverage bound. Resolve geopotential/geometric altitude
  conventions before comparison with satellite densities; do not extrapolate
  a fixed-altitude target beyond a profile's available top.

### Complete 56-variable inventory, grouped

| Group | Variables | Meaning/units |
|---|---|---|
| Coordinates (5) | `lat, lon, lev, ilev, time` | degrees; pressure-like hybrid levels; decoded time |
| Grid/pressure coefficients (6) | `gw, hyam, hybm, P0, hyai, hybi` | latitude weights, hybrid coefficients, reference pressure (Pa) |
| Time/bookkeeping (13) | `date, datesec, time_bnds, date_written, time_written, ndbase, nsbase, nbdate, nbsec, mdt, ndcur, nscur, nsteph` | calendar/output/timestep metadata, not scientific forcing nodes |
| Boundary GHG series (5) | `co2vmr, ch4vmr, n2ovmr, f11vmr, f12vmr` | file units missing; do not apply provider ppm label blindly |
| Solar drivers (4) | `sol_tsi, f107, f107a, f107p` | W/m², F10.7/sfu family, centred-81-day and previous-day proxies |
| Geomagnetic drivers (2) | `kp, ap` | dimensionless index series; inspect actual forcing interpolation |
| Electrodynamic bookkeeping (2) | `colat_crit1, colat_crit2` | critical electro-potential co-latitudes (degrees) |
| Species (9) | `CO2, H, N, N2_vmr, NO, O, O2, Op, He` | labelled mol/mol; `He` explicitly scaled from MSIS |
| Electron density (1) | `EDens` | number-density long name, but file unit is `cm^3`; unit needs correction/confirmation |
| Surface pressure (1) | `PS` | Pa |
| Heating (1) | `QJOULE` | Joule heating (K/s), **not** CO₂ radiative cooling |
| Temperature (3) | `T, TElec, TIon` | K; neutral, electron, ion |
| Winds (2) | `U, V` | m/s; zonal/meridional |
| Height (1) | `Z3` | geopotential metres above sea level |
| Density (1) | `DEN` | kg/m³, postprocessed neutral density |

Profile variables generally use `(time,lev,lat,lon)`, but **`DEN` uses
`(lev,time,lat,lon)`**. Align by named dimensions; do not flatten or reuse a
positional axis convention. `PS` uses `(time,lat,lon)`.

## Scientific and metadata gates

1. **Single historical forcing trajectory.** Solar/geomagnetic activity, GHGs,
   aerosols and other boundary conditions evolve together. The regional archive
   contains no verified fixed-CO₂ counterfactual or factorial experiment.
2. **Free-running dynamics.** Weather at a calendar date need not match observed
   weather. Compare long-term/statistical structure, not exact daily event
   agreement or 25 independent cell “replications.”
3. **Segment restarts.** Provider runs start 1949, 1973 and 1997, each with one
   spin-up year; the stored transitions around 1974/1998 need discontinuity checks.
   Before 1963, a repeated solar-cycle ion-pair-production history is prescribed.
   IGRF-13 and other historical forcing decisions also matter to interpretation.
4. **Model-derived density is partly MSIS-linked.** Provider says `He` and
   `DEN` were added post-run using scaled NRLMSISE-00 helium and constituent
   densities. It is not an independent ground-truth test of MSIS. Get the exact
   postprocessor and quantify helium dependence within the retained altitude band.
5. **Unit/definition inconsistencies.** Sample `co2vmr` is ~0.0003963, which is
   consistent with a fractional mixing ratio near 396 ppm, although the web table
   labels it ppm and the file has no unit attribute. `EDens` is labelled `cm^3`
   rather than an inverse-volume number-density unit. Sample `He` exceeds 1
   despite its mol/mol label (~6.09); verify whether it is a background-relative
   ratio rather than a normalized full-mixture mole fraction. Do not renormalize
   or “repair” these quantities by assumption.
6. **Direct cooling is absent from this schema.** There is no inspected
   species-resolved CO₂ IR cooling tendency or radiative-energy budget. `CO2`
   is a concentration and `NO` is a concentration—not a cooling-rate measurement.
   Rich state variables support consistency diagnostics, not a complete direct
   radiative mediation test.
7. **Derived formula caution.** The provider's displayed constituent-mass formula
   multiplies number density by molar mass and Avogadro's constant. With the
   stated molar-mass convention the conversion should use `M/N_A`, not `M*N_A`.
   This is a documentation/dimensional-consistency issue; it does not establish
   that the actual `DEN` implementation uses that formula. Obtain source/clarification.
8. **Terms.** The bundled README asks for author acknowledgment and notification
   to UKSSDC; a clear redistribution licence for this bespoke subset was not
   established. Keep source data access-conditional until confirmed. Licences of
   separate Brown/NCAR archives cannot be transferred to this one by analogy.

## What to ask the provider—one short request

- Confirm 2022 delivery, 1950-01-01 availability, the 2023 record/index discrepancy
  and recommended handling of the no-central-directory ZIP streams; supply hashes
  or a complete date/member manifest.
- Confirm forcing/case/version and segment joins, hourly averaging conventions,
  `co2vmr`, `EDens`, `He` units/definitions, and the `DEN` postprocessor.
- Are there already available matched **fixed-CO₂ / varying-CO₂** runs under the
  same historical forcings? Are species-resolved CO₂ cooling tendencies available?
- Can one representative column or a small selected-variable subset be delivered,
  and what research/derived-product redistribution terms apply?

These are requests for existing clarification/products, not a commitment to
download all archives or commission a new global simulation.
