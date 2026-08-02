# ParCorr structural-control diagnostics

- Runner: `pcmci-real-controls-1`
- Family-separated commit: `2e91a764eb3b909405e4710560e7191c4df6a569`
- Input SHA-256: `a7844c1dda119114f0d5a228cead3fac6bdbbcf17908f62b67b9ed069d2f1aab`
- Physical nodes: F10.7 center81, Ap, NOAA GML Mauna Loa/Maunakea CO2,
  and HASDM daily mean log-density at 325/825 km
- PCMCI+: ParCorr, `tau_max=180`, `pc_alpha=0.05`, no FDR correction
- Control seed: `20260601`

## Runs

The family-separated runs are the basis for attribution. The combined run is
retained as a secondary stress test because its two control families can compete
with one another during joint parent selection.

| controls added | wall time | peak RSS | control links | control → physical | physical entries retained |
|---|---:|---:|---:|---:|---:|
| IAAFT only | 137.13 s | 253 MB | 64 | 7 | 30/42 (71.4%) |
| circular shift only | 109.62 s | 242 MB | 55 | 4 | 33/42 (78.6%) |
| both families | 287.35 s | 344 MB | 110 | 10 | 31/42 (73.8%) |

All recorded NPZ and canonical control-link JSONL checksums were independently
recomputed and match their result manifests.

## IAAFT controls

Each IAAFT control preserves the exact finite observed marginal and the source
missingness mask. Spectral relative L2 error is 0.0032 for F10.7, 0.0173 for Ap,
0.0235 and 0.0267 for the two density series, and 0.2190 for CO2. The larger CO2
error is a caveat: its 1,140 missing observations make simultaneous spectrum,
marginal, and mask matching substantially harder.

The 64 canonical selected control links comprise 43 control-control, 14
physical-to-control, and 7 control-to-physical links. The latter target Ap (4),
325-km density (2), and CO2 (1), with median `|partial r|=0.0437` and maximum
`0.1772`. Three have `p <= 0.001`. The maximum is the source-matched IAAFT CO2
control to observed CO2 at lag 125; it is not a control-to-density link.

Relative to the primary physical graph, 12 entries are removed and 2 added.
Most removals are CO2 self-lags; the others are weak or long-lag self/cross
links. All four canonical primary Ap-density links are retained: contemporaneous
Ap-325/825 density and lagged Ap → density at 2/1 days.

## Circular-shift controls

Offsets are deterministic and exceed `tau_max`: 704 days for Ap, 3,672 for CO2,
8,288 for F10.7, 1,967 for 325-km density, and 1,235 for 825-km density. They
preserve each observed series and missingness pattern while breaking its original
calendar alignment.

The 55 canonical selected control links comprise 41 control-control, 10
physical-to-control, and 4 control-to-physical links. The latter target 325-km
density (2), Ap (1), and CO2 (1), with median `|partial r|=0.0403` and maximum
`0.1703`. Two have `p <= 0.001`. The maximum is shifted CO2 to observed CO2 at
lag 127; it is not a control-to-density link.

Relative to the primary physical graph, 9 entries are removed and 4 added.
Changes again concentrate in CO2 self-lags and weak physical self/cross links.
All four canonical primary Ap-density links are retained.

## Interpretation

The structural controls support the specific geomagnetic-density result: neither
family displaces the primary contemporaneous Ap-density associations or the
lagged Ap → density links. They do not support treating the entire 42-entry graph
as stable. Only 71-79% of physical entries survive family-separated augmentation,
with instability concentrated among long-lag CO2 structure and weak physical
self/cross links.

Control links are not expected to be identically zero when hundreds of lags are
screened at `pc_alpha=0.05` without multiplicity correction. Source-matched
control-to-source links, especially for strongly autocorrelated CO2, also show
that preserving temporal structure creates a stricter null than white noise.
Accordingly, robust interpretation should emphasize the retained Ap-density
topology and treat weak, isolated, and long-lag links as exploratory. The primary
graph contains no selected F10.7 → density or CO2 → density lagged links, so these
runs provide no positive robustness claim for such effects.
