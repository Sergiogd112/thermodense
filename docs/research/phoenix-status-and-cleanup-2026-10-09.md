# Phoenix status and operational cleanup — 2026-10-09

Read-only inspection around 19:19 CEST, followed by user-authorized operational
cleanup. No campaign data, results, models or source arrays were changed or deleted.
Paths are relative to the account's `karman-phoenix/` directory.

## Campaign state

- SGE queue empty on all hosts (auriga, phoenix, phoenix02).
- `bounded-20261006-full25` (through-2025 campaign): `phase=finished`,
  30/30 confirmation evaluations complete at **2026-10-08 17:46 CEST**,
  12,395,677 test rows. Overall paired gains (with − without CO₂ MAPE reduction,
  five-seed mean, seed SD in parentheses):

  | Architecture | Gain (pp) |
  |---|---|
  | paper_jb08_omni | +0.122 (0.200) |
  | bounded_omni_depth_01 | −0.037 (0.042) |
  | residual_jb08_depth_04 | −0.180 (0.331) |

  These are seed summaries, not data uncertainty. No paired-month bootstrap has
  been run for this campaign, and the claim gates in the
  [6 Oct assessment](parallel-full25-campaign-assessment-2026-10-06.md) remain open.
- Its ratio-analysis sidecar is **blocked** after three worker failures from
  2026-10-08 12:00 CEST: `ratio-analysis input/implementation changed; use a new
  analysis root`. The identity guard tripped because the 8 Oct CPU-evaluation
  execution amendment changed `selection.json` after `ratio-analysis/support.json`
  was written on 7 Oct. Rerunning requires a fresh analysis root; the existing
  partial `report.json` (8 Oct 10:25) predates the amendment.
- `bounded-20261004` (original study) remains `phase=finished`.

## Cleanup performed

- Crontab backed up to `~/crontab-backup-20261009.txt`, then all five entries
  removed: per-minute controller ticks and five-minute ratio dispatchers for both
  finished campaigns, and the 15-minute WACCM Mauna Loa downloader.
- The downloader had failed about 10,200 times, repeatedly extracting the same
  4.7 GB 1950-01-01–03 global archive (7-Zip exit 2). The
  [regional inventory](waccm-x-regional-inventory-2026-10-05.md) already records
  that the provided regional archives do not need this request scheme. Its
  `work/` files (about 9 GB) were left in place.
- The two dashboard servers for the finished campaigns (ports 8090/8091) were stopped.

`karman-phoenix/` occupies 216 GB on a shared `/home` that is 81% full; pruning
intermediate data needs a separate decision.

## Migration to Phoenix as the primary machine

Later the same day the working environment moved to Phoenix (`~/thermodense`,
paper worktree `~/thermodense-paper`). Large public inputs are re-downloaded
from providers on Phoenix instead of uploaded from the laptop:

- Copied: `data/products`, `data/derived`, `data/external`,
  `data/analyzed_downsampled`, `outputs/`, `papers/`, `presentation/`,
  `thesis/`, small raw inputs and access-conditional HASDM archives.
- Re-downloaded: TU Delft v02 and SABER CO₂ cooling profiles.
- Not yet on Phoenix: derived `decoded/`, `analyzed/`, `msis/`, `mmsis/`
  (~80 GB), SABER Level2A and NO cooling (~116 GB, parked prototypes only).
  Regenerating derived products is a new run with new hashes, not a reproduction.

The shared `/home` is the constraint (~150 GB free). Bulk data should move to a
per-user `/qnap` directory once an administrator creates one.
On Phoenix's ext4, same-size rewrites within one timestamp tick keep the same
mtime; one test relied on finer timestamps and was made portable.
