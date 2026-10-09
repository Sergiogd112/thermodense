# SABER clock clarification: official-source check and unsent query

Accessed **2026-10-06**. Scope: the CO₂ per-scan cooling-profile/flux archive,
not the Level2C daytime CO₂ concentration product. **No message sent, no extraction
rule changed, no authoritative timestamp correction adopted.**

## Confirmed evidence

The [official cooling-product README](https://data.gats-inc.com/saber/Version2_0/SABER_cooling/Documentation/README_SABER_CoolingRateProfiles_Dataset.pdf)
(Documentation index last-modified 2018-04-12) says daily files contain scans from
orbits **beginning that day** (§3). Its example header defines `time` as
“SABER standard msec since midnight of the day in which the orbit started” and
orbit/event numbering as **1-based**. The example is an NO file; the shared
format text covers both species, and the local CO₂ files independently contain
the same `time` description. This supports the ordinary origin convention,
not a rule for duplicate records assigned to two different header days.

The [official IDL reader](https://data.gats-inc.com/saber/Version2_0/SABER_cooling/ReadSoftware/read_profiles.pro)
explicitly supports NO and CO₂. Its comment says that an orbit extending past
midnight stays in the file for the day it originates. It reads scalar `year/day`,
converts DOY to month/day, then calculates:

```idl
doy_to_mdy,day,year,mm,dd
jdates = REFORM(JULDAY(mm,dd,year,0,0,time[15,*]*1.e-3))
```

The selected altitude index in this example is not our 119-km reference index.
Nevertheless, the **origin convention** is the same: header day plus clock. The
reader does not select an authoritative copy, recover a separate orbit-start
date, or document adjacent-file duplicate handling. It was inspected, **not run**.

### Minimal local two-file reproduction

An independent, offline SciPy reader opened only these already-local files, checked
their SHA-256 before/after reading, and selected orbit **71879**, event **2**:

| File | Zero-based scan index | Header year/DOY | Native 119-km clock (ms) | UTC **if header day is the origin** |
|---|---:|---|---:|---|
| `SABER_CO2_PROFILE_FLUX_2015073_V1.0.nc` | 1278 | 2015 / 73 | 86,445,936 | 2015-03-15 00:00:45.936Z |
| `SABER_CO2_PROFILE_FLUX_2015074_V1.0.nc` | 0 | 2015 / 74 | 86,445,936 | 2015-03-16 00:00:45.936Z |

The **full stored scan vectors**, not just 119-km values, are equal for `time`,
`tplatitude`, `tplongitude`, `CO2cool`, `slt`, `sza`; scalar `flux` is equal too.
The reference location is 2.253606°N / 6.974603°E, outside the fixed region.
The timestamp difference above is conditional on the documented origin
interpretation; neither value is asserted as the authoritative physical UTC.
It is one day, not float32 clock rounding (half-spacing only 4 ms).

Source identities:

- Day 073: 1,331,476 bytes;
  `9eade6aa34d967c9d8ac3d4eb4188fe0af8de6049da2db4dcd79ec4446751ebb`.
- Day 074: 1,326,624 bytes;
  `c7a105fd866d5a4104bf3b7f93672922b35aed484ca9b963f70de02d200cdfc5`.

Code and structured receipt are in ignored
`data/products/co2-paper-execution-20261006/`:
`reproduce_saber_provider_case.py`, `saber-provider-minimal-reproducer.json`.
Reproducer code SHA-256:
`1098ccc9e83bb06813b2b01ed90cff220d645b60905c075eb23227ad448b5c54`.
No raw vectors or provider code are copied into this note.

## Current listing and correction notices

The bounded [targeted live listing](https://data.gats-inc.com/saber/Version2_0/SABER_cooling/CO2_CoolingRate_Profiles/?P=%2A201507%5B34%5D%2A)
shows only the two V1.0 filenames, both last-modified **2018-03-27 15:14**; no
alternate named version for those two days is advertised by this filtered view.
This is **not a remote payload checksum or proof that no unlisted correction
exists**. No remote raw NetCDF was downloaded. The unfiltered listing exceeded
the explicit 3 MB limit; it was stopped and retained with `.partial.html` naming,
not treated as a complete inventory. The successful filtered page is retained as
`saber-targeted-directory-2015.html` in the ignored receipt directory.

The [Data Services notices](https://saber.gats-inc.com/data.php) distinguish:

- Temporary withdrawal of the combined daytime T/CO₂ **Level2C** product because
  of unknown per-profile CO₂ vertical resolution. No stated extension to the
  per-scan `SABER_cooling` date/clock problem was found.
- v2.08 reprocessing from **2019-12-15 onward** for a WACCM CO₂ input change.
  This does not establish a correction to the 2003/2015 cooling files.
- NCEP replacing NCAR for altitude registration from **2026-03-18 onward**.
  Again, no stated correction to these historical files.

The [Documentation listing](https://data.gats-inc.com/saber/Version2_0/SABER_cooling/Documentation/)
contains the cooling/global-power/radiated-flux READMEs and method papers, but no
listed date-boundary erratum. The inspected README §2.1 documents original V1.0;
it contains no V1.1 scientific-change entry. Previously observed V1.1 contact
updates do not by themselves establish a radiometric or timing correction.
Absence of a notice is not proof of absence of downstream processing changes.

## Contact and publication guidance

README §1.2 lists **Linda Hunt, Linda.A.Hunt@nasa.gov** for data questions and
**Marty Mlynczak, m.g.mlynczak@nasa.gov** as Associate PI. These are listed
contacts, not independently verified deliverable/current mailboxes. The scout's
fetches of [Team](https://saber.gats-inc.com/contacts.php) returned 403; no newer
preferred channel was confirmed. README §1.3 asks for early PI/designee contact,
normally offering coauthorship (may be declined), appropriate acknowledgment,
reporting version numbers and heeding caveats. This is process guidance, not a
new license/legal conclusion or a commitment on authorship.

## Unsent email draft — confirm recipient/text before use

**Proposed To:** Linda.A.Hunt@nasa.gov  
**Optional Cc:** m.g.mlynczak@nasa.gov; supervisor, at the user's discretion  
**Subject:** SABER CO₂ cooling-profile UTC origin: duplicated orbit 71879/event 2

Dear Linda Hunt,

We are preparing an analysis of SABER profile-integrated CO₂ cooling flux and
thermospheric density variability. Before assigning UTC dates or interpreting
monthly results, could you advise on an adjacent-file duplicate in the V1.0
cooling-profile archive?

Orbit 71879/event 2 appears at zero-based scan 1278 in
`SABER_CO2_PROFILE_FLUX_2015073_V1.0.nc` and scan 0 in
`SABER_CO2_PROFILE_FLUX_2015074_V1.0.nc`. The compared full scan vectors (`time`,
`tplatitude`, `tplongitude`, `CO2cool`, `slt`, `sza`) and scalar `flux` are identical,
including a 119-km `time` of 86,445,936 ms, but the scalar year/day
headers are 2015/73 and 2015/74. Adding those header days to the clock produces
2015-03-15 and 2015-03-16 at 00:00:45.936 UTC respectively. We have not assumed
which timestamp is correct or silently discarded/shifted the record.

1. Is the second occurrence an intended boundary copy, and which orbit-start
   date/UTC is authoritative?
2. What is the recommended time-origin and deduplication rule when the same
   orbit/event appears in adjacent files with different year/day headers?
3. Are corrected cooling-profile files available for these days, and does V1.1
   change timing/scientific content or only contact metadata?
4. Separately, `SABER_CO2_PROFILE_FLUX_2003235_V1.0.nc`, zero-based scan 409,
   has orbit 9239/event 0 despite the documented 1-based event numbering. What
   QC treatment is recommended for such records?

A two-file metadata reproducer, byte sizes and SHA-256 identities are available
if useful. We would also appreciate guidance on the appropriate science-team
contact, limitations and collaboration/acknowledgment expectations for this work.

Thank you,  
[Name / affiliation / supervisor, to be filled in by the user]

## Safe restart gate

**Later internal-audit update, same date:** the
[complete regional-impact audit](saber-internal-regional-impact-audit-2026-10-06.md)
found 176 duplicated identities from two copied orbits, all outside the fixed
region, and no ID/clock-reuse anomaly among 13,902 selected regional records.
Provider contact is therefore **not a necessary prerequisite** to a regional
descriptive product if the user approves spatially scoped fatal validation with
retained global diagnostics. The global-origin questions and unsent draft remain
available; no outreach is planned. The recommendations below apply to changing a
global timestamp/dedup rule, not a requirement to send this email.

Obtain an authoritative, version-specific origin/dedup/QC rule or corrected
product identity. Record the answer and its scope; add regression cases for the
two-file boundary and event-0 record before changing any interpretation. Then
requalify **all** physical-identity conflicts in the fixed scientific window,
source lineage, sampling and units. Do not widen the region, infer a day shift,
drop a conflicting record or restart full climate analysis merely to obtain an
output. Manuscript/Overleaf and the completed Karman campaign remain untouched.
