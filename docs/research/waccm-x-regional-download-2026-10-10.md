# WACCM-X regional archive download — 2026-10-10

The user asked to download everything in the provider's FTP directory. This
receipt records what was transferred, how it is stored and what is missing. It
complements the [5 Oct regional inventory](waccm-x-regional-inventory-2026-10-05.md),
whose scientific and metadata gates still apply. No analysis has been run.

## Source and transfer

- UKSSDC FTP directory for this project's bespoke request (anonymous access).
  The listing was unchanged since the 5 Oct inventory. It holds 74 yearly
  regional archives (16–25°N, 200–210°E; 152,167,302,713 bytes) and the 4.7 GB
  global 1950-01-01..03 test archive.
- All 74 regional archives were streamed once by SGE array 1579 on
  `all.q@auriga`, 4 at a time, between 22:30 and 23:27 CEST. All tasks exited 0.
  For every archive, the bytes read equal the listed size and no bytes follow
  the last member.
- The global test archive was already on Phoenix from the old downloader. Its
  size and last 1 MB match the FTP copy, so it was moved rather than downloaded again.

## Storage (`data/original/waccm_x_regional/`, gitignored, not redistributable)

- One directory per yearly archive. Each daily member's original NetCDF4 bytes
  are stored **unchanged** in an `.xz` container (LZMA preset 6). This keeps
  every variable, attribute and value exactly. LZMA (3.4 MB per day)
  compressed better than an HDF5 gzip repack (4.2 MB per day).
- Every member's provider CRC-32 was verified before storing. `manifest.json`
  records each member's name, archive offset, size, CRC-32, SHA-256 and stored
  size. `COMPLETE.json` records the archive-level checks.
- Totals: **27,026 daily members**, 91.7 GB stored (from 152.2 GB raw).
- `.tools/fetch_archive.py` is the downloader. `.tools/waccmx_day.py`
  (`open_day("YYYY-MM-DD")`) re-checks the CRC and returns an `h5py.File`.
  h5py is not a project dependency; use `uv run --no-project --with h5py`.

## Coverage

- **1950-01-02 through 2024-12-31**, no duplicate days. Each day has 24 hourly
  records (per the 5 Oct inspection) on a 5 × 5 grid with 126 levels.
- Missing: **all of 2022** (no archive was delivered), and **2023-01-31 and
  2023-02-03** (absent from the provider's 2023 archive).
- Three 2023 members (06-14 to 06-16) are 1,403 bytes larger than all the
  others. They open normally; the reason is unknown.
- Spot reads: `co2vmr` is 310.6 ppm on 1950-01-02, 415.1 ppm on 2023-06-15 and
  418.1 ppm on 2024-12-31. The 2014-06-15 profile gives 400 ppm CO₂ at the
  surface, 384 ppm near 80 km and 33 ppm near 120 km.

## Cluster housekeeping (user-approved, same day)

- Removed the dead cron downloader `~/.local/share/waccm-mauna-loa`: its venv,
  log, partial extraction and an older duplicate of the global test archive.
- The `saber-50x70-v1` run's SABER CO₂ cooling copy (8,537 files) was a strict
  subset of the 9 Oct re-download (sizes equal, sampled bytes identical). It
  was replaced with a symlink so that run still resolves.
- Cleared the uv and pip caches.
- `/home` now has about 50 GB free (94% used). The 41 GB SABER NO-cooling raw
  data, used only by parked prototypes, is still on Phoenix under that run's
  directory. The 9 Oct migration notes listed it as missing.

## Still open (from the inventory)

Ask the provider about 2022 and the two missing 2023 days, the larger June
2023 members, hourly-mean versus instantaneous conventions, the `DEN`
postprocessor and the redistribution terms. Analysis needs its own approved design.
