# SABER HASDM 3-hour ingestion

`python -m thermodense.saber` creates the HASDM-local SABER extension table. It
keeps direct `CO2cool` and `NOcool` products separate from the Level2A
`O2_1delta_ver`, `OH_16_ver`, and `OH_20_ver` emission proxies. The latter are
converted from `ergs/cm3/sec` to `W/m^3` by multiplying by 0.1; they are not
cooling rates.

Download CO2 and NO (resumable, each selecting the latest daily `V1.x` file).
Each downloader atomically writes `remote_inventory.json`, an auditable complete
latest-per-day remote listing used by the decoder to distinguish officially absent
days from incomplete local downloads:

```sh
uv run python -c 'from datetime import date; from thermodense.downloader.saber import download_saber_co2_cooling; download_saber_co2_cooling(start_date=date(2002, 1, 25), end_date=date(2025, 7, 20))'
uv run python -c 'from datetime import date; from thermodense.downloader.saber import download_saber_no_cooling; download_saber_no_cooling(start_date=date(2002, 1, 25), end_date=date(2025, 7, 20))'
```

Select and download only the required Level2A orbit files after both cooling
archives are present:

```sh
uv run python -c 'from datetime import date; from pathlib import Path; from thermodense.saber import CO2_FILE_RE, NO_FILE_RE, _coverage, derive_hasdm_grid, hasdm_longitudes_by_timestamp, required_l2a_orbits; from thermodense.downloader.saber import download_saber_l2a; source=Path("data/decoded/hasdm/HASDM_2000_merged.parquet"); samples=Path("outputs/figures/results/hasdm_msis_model_errors/data/hasdm_msis_errors_nearest_timestamp_grid_samples.parquet"); start, end=date(2002, 1, 25), date(2025, 7, 20); lat, lat_step, lon_step=derive_hasdm_grid(source); events=required_l2a_orbits(_coverage(Path("data/original/saber/co2_cooling_profiles"), CO2_FILE_RE, start, end, "CO2 cooling"), _coverage(Path("data/original/saber/no_cooling_profiles"), NO_FILE_RE, start, end, "NO cooling"), hasdm_longitudes_by_timestamp(samples), lat, lat_step, lon_step); download_saber_l2a(events)'
```

Decode the full calendar product:

```sh
uv run python -m thermodense.saber
```

By default the calendar ends exactly at `2025-07-20 00:00`; use `--end` with a
timezone-naive 3-hour datetime for another exact endpoint. A date-only `--end`
retains inclusive full-day semantics.

The output is `data/decoded/saber/saber_hasdm_maunaloa_3hour.parquet`, with one
UTC 3-hour row per calendar slot, 15 `saber_*_<alt>km_w_m3` value columns, a
matching `*_observations` count column for each, and the actual per-slot HASDM
longitude. Empty bins remain null. Its adjacent `.provenance.json` records
decoded HASDM grid spacing/bounds, native altitudes used, conversion, and
missing-value policy. The decoder fails if post-mission CO2, NO, or required
Level2A source coverage is incomplete, or if the requested calendar extends
beyond the available HASDM longitude time bounds. Internal HASDM longitude
gaps remain null. It requires the remote inventories to
reach the requested end date and every officially listed requested cooling file
to have a successful or skipped manifest entry and a nonempty local file; old
manifests without inventories must be refreshed by rerunning the downloader.
For Level2A it requires exactly one manifested official `02.07` file before
2019-12-15 and `02.08` on or after that date, rejecting stale, duplicate,
failed, missing, and unmanifested files for required day/orbit pairs. If an
orbit is absent from its cooling day directory, the downloader resolves it from
exactly one adjacent (previous or next) Level2A day directory; validation
accepts that manifested adjacent-day filename for the original cooling day/orbit.

Required Level2A orbits are the union of CO2 and NO daily-archive scan
footprints with finite geolocation and time that intersect the HASDM spatial
sample. This is the deliberate cooling-supported/common spatial sample:
cooling values themselves need not be finite, and it does not request every
possible proxy-only Level2A scan. The exact policy is recorded in output
provenance.

The direct-density PCMCI runner consumes these 15 sparse channels only in its
3-hour HASDM cases (`hasdm_all-*` and `hasdm_selected-*`), applying its normal
slot-aware seasonal/trend transform without filling missing SABER values. Daily
global-mean cases do not include SABER.
