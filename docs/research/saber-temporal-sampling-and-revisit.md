# SABER temporal sampling and geographic revisit

Date: 2026-09-01

## Question

What is SABER's temporal resolution, and how long does SABER take to observe the
same area again? How do those mission characteristics relate to the sparse
3-hour SABER channels used by thermodense?

## Short answer

SABER does not observe a global image at a fixed time step. It is a limb sounder
that produces vertical profiles sequentially along the TIMED orbit. In one
official 2020 daily product, adjacent archived profiles within an orbit are
separated by a median of about **66.7 seconds**, with about **91 profiles per
orbit**. The same archive gives a median orbit-to-orbit start interval of about **96.4
minutes**, or roughly 15 orbits per day. These are profile cadence and orbital
period, not exact-location revisit times.

There is no single fixed revisit interval for an exact geographic area. The
ground track shifts in longitude between orbits, SABER observes the limb rather
than nadir pixels, the latitude range changes after spacecraft yaw maneuvers,
and local solar time precesses. For a **10° latitude by 15° longitude** box
centered near Mauna Loa, an empirical 2020 calculation from the official CO2
profile archive found 498 distinct 119-km tangent-point encounters, or **1.36
encounters/day**. The median gap was **15.6 hours**; the interquartile range was
about **9.0–24.2 hours**, the 95th percentile was 24.23 hours, and the maximum
that year was 46.8 hours.
This describes that box and year, not a mission guarantee.

The thermodense product is sparser still and does not represent a fixed
geographic box. It bins observations into 3-hour slots inside 15–25°N and
within ±7.5° of a longitude that moves with the selected HASDM cell. For the
119 km CO2 channel, 11,704 of 74,657 slots are nonempty (15.7%). Among nonempty
slots, the median gap is 15 hours, the 75th percentile is 24 hours, the 95th is
27 hours, and the 99th is 39 hours. The maximum observed interval is 759
hours; the archive audit below attributes that maximum to the moving-cell
spatial match rather than missing daily source files. Therefore, **3 hours is
the analysis grid, not SABER's native temporal resolution or guaranteed
revisit**.

## Distinct time scales

| Quantity | Approximate value | Meaning |
|---|---:|---|
| Altitude samples within one retrieved profile | span about 4.75 seconds in a representative 2020 file | Time represented by the selected 40-level portion of one profile |
| Adjacent profiles within an orbit | median 66.7 seconds in a representative 2020 file | Archived along-track profile separation, not an instrument requirement |
| TIMED orbit | official 625 km circular, 74.1° inclination; observed median 96.4 minutes | Time to circle Earth, not time to revisit the same longitude |
| Orbits per day | about 15 | Global sampling frequency along different tracks |
| Orbit-plane precession relative to the Sun | about 3°/day | Official solar-relative rate; distinct from the overview's nodal-regression rate |
| Idealized local-time drift derived from that rate | about 0.2 hour/day, or 120 days per 24 hours | Geometric conversion only; actual SABER tangent-point local times were not evaluated here |
| Spacecraft yaw | about every 60 days | Reverses spacecraft orientation and changes which pole SABER can reach |

The values measured from distributed files are descriptive statistics, not
instrument requirements. The profile intervals vary because the observing
sequence includes different scan/event modes and occasional gaps.

## Geographic coverage

The official SABER overview gives a 625 km circular orbit, 74.1° inclination,
and 720°/year nodal regression. That Earth-referenced nodal rate is not the same
quantity as the operations page's approximately 3°/day rate relative to the
Sun; local-time reasoning below uses only the explicitly solar-relative rate.
The official instrument description identifies
SABER as a limb-viewing radiometer with a 2 km vertical instantaneous field of
view and a scan mirror that moves the tangent height from Earth to 400 km. The
overview reports a 0.4 km limb vertical sampling interval. These vertical
figures are not horizontal footprint size or revisit time.

Official daily profiles show the familiar alternating yaw geometry:

- one orientation reaches approximately 52°S–83°N;
- the other reaches approximately 83°S–52°N.

The TIMED operations page states that the orbit precesses about 3° per day
relative to the Sun and that a 180° yaw is performed after about 60 days. A
low-latitude box such as 15–25°N remains inside both latitude modes, so yaw does
not exclude it by latitude alone. Other operational effects can still produce
gaps. Locations poleward of about 52° are available only in the orientation
that looks toward their pole.

Converting 3°/day to local solar time gives an idealized drift of about 0.2
hour/day and about 120 days for one orbital node to move through 24 local-time
hours. This is a geometric interpretation, not a source-reported SABER coverage
guarantee; actual tangent-point local-time coverage requires analysis of the
profile timestamps and geometry.

An exact point does not have a useful deterministic "revisit" in the way a
wide-swath imager might. SABER samples discrete tangent profiles. Revisit must
therefore be defined using a latitude/longitude tolerance, altitude, product,
quality policy, and whether several adjacent profiles count as one pass.

## Empirical methods

### Native cadence and orbit period

The representative file
`SABER_CO2_PROFILE_FLUX_2020001_V1.0.nc` contains 1,362 profiles across 15
orbits. At the first altitude level:

- median profile-to-profile interval within an orbit: 66.72 seconds;
- median profiles per orbit: 91;
- median first-profile spacing between consecutive orbits: 96.40 minutes;
- median time span across the file's 40 altitude timestamps for one profile:
  4.75 seconds.

Files from 2002 and 2010 give similar median profile intervals of about
67.9 and 66.8 seconds, respectively.

### Fixed Mauna Loa tangent-point box

The 2020 calculation used the official daily CO2 cooling profile files at the
native altitude nearest 119 km. It selected tangent points within 15–25°N and
within ±7.5° longitude of 204.4237°E. Consecutive selected profiles separated
by no more than 30 minutes were grouped into one encounter. This is a
tangent-point re-encounter calculation, not a claim that the instrument imaged
the whole box. Results:

- 1,324 profiles in 498 encounters over 366 days;
- median three profiles per encounter;
- 1.36 encounters/day;
- encounter-start gap quantiles: 25% 9.02 h, median 15.60 h, 75% 24.20 h,
  95% 24.23 h, 99% 37.77 h, maximum 46.80 h.

Changing box size, altitude, year, product validity, or encounter-grouping
threshold will change these values.

### Thermodense moving HASDM cell

`thermodense.saber._bin()` floors profile times to three-hour UTC slots. The
decoder retains profiles within ±5° latitude of the 20°N HASDM grid center and
within ±7.5° longitude of the per-slot HASDM longitude. Multiple matches are
averaged and empty slots remain null. The longitude is not fixed: the completed
product contains 769 distinct decoded longitude-center values over its lifetime.

For `saber_co2cool_119km_w_m3_observations`:

- calendar slots: 74,657;
- nonempty slots: 11,704 (15.68%);
- gaps between nonempty slots: minimum 3 h, 25% 9 h, median 15 h, 75% 24 h,
  95% 27 h, 99% 39 h, maximum 759 h.

These observed gaps are the relevant effective cadence for the current PCMCI
input. The combined mask from orbital geometry, moving-cell matching, finite
values, source availability, and binning can sharply reduce complete-case rows
when several lagged SABER channels are conditioned jointly. Except for the
longest interval audited below, this note does not apportion missingness among
those mechanisms.

### Archive completeness and longest gaps

A live comparison with the official GATS directory indexes on 2026-09-01 found
8,760 latest-version daily files in each cooling archive, covering available
days from 2002-01-25 through 2026-02-28. The local CO2 and NO directories each
contain 8,537 files through the analysis endpoint, 2025-07-20. No official
latest-version cooling file through that endpoint is missing locally. The 223
remote files absent locally in each archive are all later than the analysis
endpoint, from 2025-07-21 through 2026-02-28.

The official CO2 and NO indexes have identical day coverage. Across their
published span, 41 calendar days are absent in 16 runs. The longest official
daily-file gap is seven days, 2008-01-09 through 2008-01-15. The corresponding
official Level2A directories are also empty. Across both sets of 8,537 local
CO2 and NO daily files, the longest interval between any two archived profiles
is **190.97
hours** (about 7 days 22 hours 58 minutes), from 2008-01-08 09:05:12.496 to
2008-01-16 08:03:39.464. This is an archive-wide observation gap, not a
geographic revisit interval.

The much longer gap in the decoded moving-cell product is not a skipped daily
file. Every one of the 15 channels has consecutive populated slot timestamps
separated by **759 hours** (31 days 15 hours), from 2024-04-30 18:00 to
2024-06-01 09:00. There are 252 empty three-hour slots between them. Both local
cooling directories and the official indexes contain all 31 May 2024 daily
files. Those CO2 files contain 40,104 profiles during May, with a maximum
archive-wide profile gap of 10.45 hours, but none of their finite
geolocation/time footprints intersects the per-slot moving HASDM cell. The
759-hour result is therefore caused by the spatial-time matching geometry, not
by omitted May source files.

Level2A is intentionally not mirrored in full. Recomputing the decoder's
cooling-footprint selection produced 12,461 required day/orbit pairs through
2025-07-20, and all 12,461 have exactly one nonempty, manifested local file at
the selected official version. The official Level2A archive contains additional
orbits and extends beyond the cooling products: the latest nonempty directory
checked was 2026 day 240 (2026-08-28). Those extra files are outside the current
HASDM endpoint or outside the moving-cell orbit selection.

The website also publishes daily global CO2 and NO cooling-power text products.
They are not downloaded here because they are global daily aggregates and do
not replace the tangent-point profiles needed for local geographic matching.

## Interpretation for analysis

1. Describe the representative archived sampling as **approximately one limb
   profile per minute along track**, not as a three-hour instrument or a fixed
   instrument requirement.
2. Describe the orbit as about **96–97 minutes**, but do not equate one orbit
   with geographic revisit.
3. In the stated 2020 calculation for a 10°×15° box near 20°N, tangent-point
   encounter gaps were typically **half a day to one day**, with occasional
   longer gaps. Use the measured product mask for statistical design rather
   than assuming daily coverage or generalizing one year to the mission.
4. Treat the roughly 60-day yaw scale as an orientation and latitude-coverage
   cycle, not the revisit time for a low-latitude area. Verify local-time
   coverage separately from the actual tangent-point data.
5. State the spatial tolerance and encounter definition whenever reporting
   revisit.

## Primary sources

- GATS SABER, [Overview](https://saber.gats-inc.com/overview.php): official
  orbit, inclination, nodal regression, and vertical sampling characteristics.
- GATS SABER, [Instrument](https://saber.gats-inc.com/instrument.php): official
  limb-viewing and scan-mirror description.
- Johns Hopkins APL TIMED Science Data System,
  [Spacecraft yaw maneuvers](https://timedsds.jhuapl.edu/Mission/yaws/yaws.html):
  official 3°/day solar-relative precession and approximately 60-day yaw cycle.
- Esplin et al. (2023),
  [SABER instrument and science measurement description](https://doi.org/10.1029/2023EA002999):
  mission-team instrument paper confirming the 625 km, 74.1° orbit and routine
  profile data collection.
- GATS SABER, [Data description documents](https://saber.gats-inc.com/documentation.php):
  official Level 1B, Level 2, Level 2A, and Level 2B product schemas.
- Official GATS daily files under
  `data/original/saber/co2_cooling_profiles/`: primary distributed measurements
  used for the empirical cadence, latitude-range, and fixed-box calculations.
- GATS SABER, [CO2 cooling profile archive](https://data.gats-inc.com/saber/Version2_0/SABER_cooling/CO2_CoolingRate_Profiles/)
  and [NO cooling profile archive](https://data.gats-inc.com/saber/Version2_0/SABER_cooling/NO_CoolingRate_Profiles/):
  official daily-file indexes used for the local/remote completeness audit.
- GATS SABER, [Level2A archive](https://data.gats-inc.com/saber/Version2_0/Level2A/):
  official per-day, per-orbit profile archive.
- GATS SABER, [daily global cooling power](https://data.gats-inc.com/saber/Version2_0/SABER_cooling/Daily_Global_Power/):
  official global aggregate product not used by the moving-cell decoder.

## Repository implementation sources

- `src/thermodense/saber.py`: timestamp binning and geographic matching.
- `docs/saber-hasdm-ingestion.md`: output contract and missing-value policy.
- `data/decoded/saber/saber_hasdm_maunaloa_3hour.provenance.json`: decoded grid
  bounds and moving-longitude policy.
- `data/decoded/saber/saber_hasdm_maunaloa_3hour.parquet`: observed effective
  three-hour-slot coverage.
