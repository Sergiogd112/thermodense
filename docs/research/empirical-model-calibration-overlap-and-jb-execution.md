# Empirical-model calibration overlap and JB execution

Date: 2026-08-03

## Purpose

This note audits how the documented fitting data for NRLMSISE-00,
NRLMSIS 2.0, NRLMSIS 2.1, JB2006, and JB2008 overlap the two reference
families planned for the first paper:

- the TU Delft CHAMP, GOCE, GRACE-A/B, GRACE-FO, and Swarm-A/B/C
  accelerometer-derived density products; and
- the Mauna Loa subset of the SET HASDM density database.

It also records the currently defensible route for executing JB2006 and
JB2008. The audit follows the **Calibration-domain representativeness**
classes in `CONTEXT.md`. It distinguishes a named mission or process overlap
from proof that the exact processed rows were reused.

## Conclusions

1. **JB2008 has the strongest documented calibration-domain overlap.** Its development data
   explicitly included Air Force HASDM densities from 2001--2005, CHAMP
   accelerometer densities from 2001--2005, and GRACE accelerometer densities
   from 2002--2005.[^jb2008-paper] Independence from those mission/source and
   epoch domains must not be assumed. Identity with the TU Delft products,
   the SET HASDM1 release, or the evaluated rows is not established.
2. **JB2006 is partially related to HASDM.** Its fitted equations used daily
   density values derived from satellite drag analysis and temperature
   corrections applied in the Air Force HASDM-modified Jacchia-70 modelling
   chain over 1978--2004.[^jb2006-paper] The relationship to the later SET
   HASDM1 release or its rows is not established.
3. **NRLMSIS 2.0 kept CHAMP and GOCE out of fitting.** It fitted upper-
   thermospheric mass-density behaviour to global orbit-derived daily means
   from 1986--2005, while CHAMP and GOCE accelerometer densities were listed
   as independent comparison data.[^msis20-paper] This supports an
   out-of-fitting-sample interpretation for the Figure 19 recreation, subject
   to documenting the exact TU Delft product version used here.
4. **NRLMSIS 2.1 does not refit the base mass-density model.** Its upgrade to
   2.0 consists solely of adding an empirical NO model from six other
   space-based instruments; none is a TU Delft density mission or HASDM.[^msis21-paper]
5. **The JB license creates a permissions risk gate.** SET permits use and
   distribution without charge but prohibits modifying, adapting, or
   translating the software and its driving data products.[^set-license]
   The terms do not expressly address a separate thin wrapper that invokes
   unmodified routines, so provider clarification is required before
   Thermodense distributes such a wrapper or a translated implementation.

## Classification rules

| Class | Meaning in this audit |
| --- | --- |
| Documented direct overlap | The model source names the same mission, instrument, or reference product and an overlapping epoch as fitting/development data. Exact processed-row identity is stated separately. |
| Related-source or partial overlap | The model used the same modelling/assimilation process or a related density source, but exact reference-product or row reuse is not established. |
| No documented overlap | The reviewed primary model description names no matching mission, product, or fitting source. This is not proof that no indirect relationship exists. |
| Unknown | The available primary source is insufficient to classify the relationship. |

## Model audit

### NRLMSISE-00

Picone et al. document orbit-determination drag data from 1961--1973;
accelerometer data from Atmosphere Explorer, SETA, CACTUS, and San Marco 5;
Millstone Hill and Arecibo incoherent-scatter-radar data through 1997; and
Solar Maximum Mission O2 occultation data.[^msise00-paper] The paper does not
name CHAMP, GOCE, GRACE, GRACE-FO, Swarm, or HASDM as fitting data.

| Evaluation reference | Classification | Reason |
| --- | --- | --- |
| TU Delft missions | No documented overlap | The named calibration missions and epochs precede the TU Delft mission set. |
| SET HASDM | No documented overlap | The model used older Jacchia/Barlier drag data, not a documented HASDM product or process. |

This is a historically disjoint comparison, but the old calibration domain
also means that later mission, solar-cycle, and instrument regimes are model
extrapolations rather than a modern independent-validation design.

### NRLMSIS 2.0

NRLMSIS 2.0 used global-average, orbit-derived daily mass densities at
400--575 km from 1986--2005 for fitting. The complete orbit-derived series
spans 1967--2013 and comes from two-line elements for approximately 5,000
objects.[^msis20-paper] The paper explicitly says that CHAMP and GOCE
accelerometer densities were not used to estimate the model parameters. It
uses CHAMP observations from January 2001--September 2010 and GOCE version
2.0 observations from November 2009--October 2013 for independent comparison.

| Evaluation reference | Classification | Reason |
| --- | --- | --- |
| TU Delft CHAMP and GOCE | No direct calibration overlap; source-aligned independent comparison | The missions are explicit independent comparison sets, not fitting sets. The paper and this project both use TU Delft-family accelerometer products, but the exact processing-version relationship must be reported. |
| TU Delft GRACE/GRACE-FO/Swarm | No documented overlap | These missions are not named in the model's thermospheric mass-density fitting set. |
| SET HASDM | No documented overlap | HASDM is not named as a fitting source. |

The Figure 19 caption excludes CHAMP years 2006--2009, while the paper's data
section says 2005--2009.[^msis20-paper] Thermodense should retain and label
both variants rather than silently resolving the discrepancy.

### NRLMSIS 2.1

Emmert et al. describe NRLMSIS 2.1 as a 2.0 upgrade consisting solely of an
added empirical NO model. The added observations are UARS/HALOE, SNOE,
Envisat/MIPAS, ACE/FTS, Odin/SMR, and AIM/SOFIE.[^msis21-paper] The base
temperature and species-density calibration-domain conclusions therefore
follow NRLMSIS 2.0, subject to verifying implementation equivalence; the
added NO data introduce no documented TU Delft or HASDM overlap.

| Evaluation reference | Classification | Reason |
| --- | --- | --- |
| TU Delft missions | Same as NRLMSIS 2.0 for base mass density | The added NO instruments are different missions. |
| SET HASDM | Same as NRLMSIS 2.0 for base mass density | No HASDM fitting source is introduced by the NO upgrade. |

### JB2006

Bowman et al. state that the density data used to develop JB2006 were accurate
daily values from drag analysis of satellites with perigees from 175--1100 km.
Those density values and temperature corrections were produced in the Air
Force HASDM-modified Jacchia-70 modelling chain over 1978--2004; the paper
does not establish reuse of rows from the later SET release. The diurnal
correction used 79 calibration satellites over 1994--2003 plus 35 satellites
for the 1989 solar maximum.[^jb2006-paper]

| Evaluation reference | Classification | Reason |
| --- | --- | --- |
| TU Delft missions | No documented direct overlap | The paper does not name TU Delft, CHAMP, GOCE, GRACE, GRACE-FO, or Swarm accelerometer densities as development data. Temporal overlap alone does not establish source overlap. |
| SET HASDM | Related-source or partial overlap | JB2006 used the Air Force HASDM-modified J70 process and corrections through 2004. Exact reuse of the released SET grid or Mauna Loa rows is not documented. |

### JB2008

The JB2008 paper names four development sources: Air Force daily density
values from 1997--2007, Air Force HASDM densities from 2001--2005, CHAMP
accelerometer densities from 2001--2005, and GRACE accelerometer densities
from 2002--2005.[^jb2008-paper] It later states that the CHAMP series was
scaled by 1.17 and GRACE by 0.74 to adjust their average levels to HASDM.

| Evaluation reference | Classification | Reason |
| --- | --- | --- |
| TU Delft CHAMP | Documented mission/epoch overlap; product identity unknown | CHAMP accelerometer densities from 2001--2005 helped develop JB2008. The paper acknowledges a CNES-provided series; identity with TU Delft version 02 is not established. |
| TU Delft GRACE product(s) | Documented GRACE mission/epoch overlap; spacecraft and product identity unknown | GRACE accelerometer densities from 2002--2005 helped develop JB2008. The paper acknowledges a University of Texas-provided series but does not establish identity with TU Delft GRACE-A/GRACE-B files or version-02 processing. |
| TU Delft GOCE, GRACE-FO, Swarm | No documented overlap | These missions/products are not named as JB2008 development sources. |
| SET HASDM | Related-source or partial overlap | Air Force HASDM densities from 2001--2005 were development data. The sources do not establish identity with the current HASDM1 grid release or the evaluated Mauna Loa rows. |

The current SET database page identifies the released 2000--2019 and
2020--2025 grids as HASDM1, based on J70 with drag-derived temperature
corrections.[^hasdm-set] The JB2008 paper separately reports Air Force HASDM
development densities from 2001--2005. The sources do not establish that
these were the same operational HASDM1 product, grid release, or rows. The
defensible classification is HASDM-related development overlap, not exact
SET-HASDM1 row reuse or a simple JB2008-background relationship.

## Consequences for the first paper

1. Add a calibration-overlap annotation to every model/reference stratum.
2. Do not call the full five-model comparison an independent validation.
   Independence differs by model, mission, epoch, and reference.
3. Separate JB2008 results for CHAMP 2001--2005 and GRACE 2002--2005 from
   mission/epoch strata with no documented overlap. At minimum, report a
   sensitivity excluding those periods.
4. Treat JB2006-versus-HASDM and JB2008-versus-HASDM as HASDM-related
   evaluations. They can measure disagreement with the released reference,
   but not performance against a wholly external observational truth.
5. Keep the paired-sample and paired-bootstrap rules from `CONTEXT.md`.
   Calibration overlap is an interpretation stratum, not a reason to evaluate
   models on different rows.
6. Preserve the primary claim as conditional performance. These provenance
   differences make a universal model ranking less defensible, not more.

## JB execution and licensing audit

### JB2008

SET currently publishes:

- Y2K-compliant Fortran model and driver files;[^set-code]
- the currently linked `jb2008.zip`, containing Fortran model/driver files, a
  Python orchestration script, and a README. The supplied script is grid/driver
  oriented; the official page does not document a general arbitrary-point
  Python API;[^set-modern-jb]
- a separate validation archive with expected output grids;[^set-modern-jb]
- live SOLFSMY, Ap, Dst, and dTc index links, with a documented 45-day lag.[^set-indices]

The archives inspected on 2026-08-03 had SHA-256
`0b310c9e259b5ca9285d570610d52afa5964d82506ea4fb43e995a8332535d6a`
for `jb2008.zip` and
`4d18c286b148fc5b2271667dab71dd6a702204bb5176bc8a54200aa17e3fff98`
for `jb2008validate.zip`. These are retrieval records, not permanent provider
versions. The live index files are also revised products. SET documents S10
recalibration and historical corrections, so every run must record retrieval
time, source URL, byte checksum, and any version or header information actually
present in each downloaded file.[^set-indices]

### JB2006

The live JB2006 site is unavailable, but its 2016 Wayback capture preserves
the complete Fortran source, calling convention, and official test case.[^jb2006-code]
The test case expects density `0.4066D-11 kg m-3` at its stated date, geometry,
and indices. JB2006 uses one-day-lag F10/S10, five-day-lag M10, and Ap at a
6.7-hour lag; these conventions differ from JB2008.[^jb2006-code]

### Local execution record

An ignored local wrapper now calls the unmodified provider routines for both
models at 19.5362° N, 204.4237° E and 125–825 km in 25-km steps. The JB2006
official test reproduces `0.4066D-11 kg m-3`; the JB2008 source compiles and
reproduces the validation grid structure, although revised live indices no
longer reproduce the archive's stale September-2023 values byte-for-byte. The
wrapper converts the WGS-84 geodetic site latitude to each model's required
geocentric latitude separately at every altitude.

The live `SOLRESAP.TXT` retrieved for this run contained only 33 records ending
on 1997 day 6, so it could not support the selected record. JB2006 therefore
uses the repository's CelesTrak three-hour Ap observations, mechanically written
in the fixed format consumed by SET's unmodified `SOLFLUX` routine. SET
`SOLFSMY.TXT` supplies F10/S10/M10/Y10, and SET `DTCFILE.TXT` supplies JB2008
dTc. The paired output covers 1997-01-06 through 2026-04-19; CelesTrak's current
contiguous observed record fixes the endpoint. Both models omit 1998-05-17
through 1999-03-15 because required SET solar proxies are below the official
driver's `<40` validity threshold. Source, index, compiler, transformation, and
output checksums are recorded in the ignored output provenance. This local
execution record does not resolve permission to distribute the wrapper.

### License decision

The SET license applies to both JB2006 and JB2008. It permits use without
charge and permits commercial distribution with acknowledgement and provider
links, but states that users may not modify, adapt, translate, reverse
engineer, decompile, or disassemble the software or driving data products.[^set-license]
It does not expressly state whether a separately written thin wrapper that
invokes unmodified routines is permitted.

Therefore:

1. Do not commit SET Fortran or index files to this repository.
2. Do not implement or adopt a translated port until its compatibility with
   the SET terms has been confirmed.
3. Ask SET whether Thermodense may write and distribute a thin driver that
   calls the unmodified subroutines at arbitrary reference sample points.
4. If permission is granted, keep provider files under ignored `data/` or a
   user cache, verify JB2006's test case and JB2008's validation archive, and
   record compiler, source checksums, index checksums, and coordinate
   conversions.
5. Until that clarification, Orekit is a candidate rather than a selected JB
   implementation, despite its Apache-2.0 project license.

This is a technical provenance assessment, not legal advice.

## Sources

[^msise00-paper]: Picone, J. M., Hedin, A. E., Drob, D. P., and Aikin, A. C. (2002), “NRLMSISE-00 empirical model of the atmosphere: Statistical comparisons and scientific issues,” *Journal of Geophysical Research: Space Physics*, [doi:10.1029/2002JA009430](https://doi.org/10.1029/2002JA009430), especially section 2.2.
[^msis20-paper]: Emmert, J. T. et al. (2021), “NRLMSIS 2.0: A Whole-Atmosphere Empirical Model of Temperature and Neutral Species Densities,” *Earth and Space Science*, [doi:10.1029/2020EA001321](https://doi.org/10.1029/2020EA001321), especially Table 1 and sections 3.1, 4.7, and 6.5.
[^msis21-paper]: Emmert, J. T. et al. (2022), “NRLMSIS 2.1: An Empirical Model of Nitric Oxide Incorporated Into MSIS,” *Journal of Geophysical Research: Space Physics*, [doi:10.1029/2022JA030896](https://doi.org/10.1029/2022JA030896); see also NASA CCMC's [NRLMSIS 2.1 model catalogue](https://ccmc.gsfc.nasa.gov/models/NRLMSIS~2.1/).
[^jb2006-paper]: Bowman, B. R., Tobiska, W. K., Marcos, F. A., and Valladares, C. (2008), “The JB2006 empirical thermospheric density model,” *Journal of Atmospheric and Solar-Terrestrial Physics*, [doi:10.1016/j.jastp.2007.10.002](https://doi.org/10.1016/j.jastp.2007.10.002), especially sections 2, 4, and 5.
[^jb2008-paper]: Bowman, B. R. et al. (2008), “A New Empirical Thermospheric Density Model JB2008 Using New Solar and Geomagnetic Indices,” AIAA 2008-6438, [official SET PDF](https://sol.spacenvironment.net/JB2008/pubs/AIAA_2008-6438_JB2008_Model.pdf), especially section II.
[^hasdm-set]: Space Environment Technologies, [SET HASDM Thermospheric Density Database](https://spacewx.com/hasdm/).
[^set-license]: Space Environment Technologies, [JB2006 and JB2008 Software License and Warranty Agreement](https://sol.spacenvironment.net/JB2008/License.html).
[^set-code]: Space Environment Technologies, [JB2008 software downloads](https://sol.spacenvironment.net/JB2008/code.html).
[^set-modern-jb]: Space Environment Technologies, [JB2008 resources and validation](https://spacewx.com/jb2008/).
[^set-indices]: Space Environment Technologies, [JB2008 indices](https://sol.spacenvironment.net/JB2008/indices.html).
[^jb2006-code]: Space Environment Technologies, [archived JB2006 Fortran source and test case](http://web.archive.org/web/20160821193912/http://sol.spacenvironment.net/jb2006/code.html).
