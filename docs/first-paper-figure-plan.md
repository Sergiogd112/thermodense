# First-paper figure plan

Based on supervisor feedback, the user selected a concise, figures-first paper:
six ordered candidate figures are planned, with five or six retained in the main
text after artifact review. The prior PCMCI figure-review set is supplementary,
not this paper set.

## Narrative order and contracts

1. **Coverage and provenance** — a two-panel time-altitude map contrasts
   regular/gridded products with TU Delft mission coverage. It establishes what
   is covered and where it came from; it does not show a scientific effect.
2. **F10.7 response by altitude** — daily mean log density and daily
   log-density range across applicable Global mean, HASDM, TU Delft,
   NRLMSIS-family, and paired JB2006/JB2008 products. Show slopes/correlation,
   95% uncertainty, and record-length markers. WACCM-X is out of this figure.
3. **HASDM scatter composite** — an exact 4x4: rows are mean log density at
   175 and 825 km, then daily log-density range at 175 and 825 km; columns are
   centered 81-day F10.7, Ap, tropospheric CO2, and SABER CO2 cooling at 139 km.
   It is descriptive points only, with no regression lines.
4. **Altitude relationship summary** — two panels for mean/range relationships
   with CO2 and SABER cooling, plus F10.7 and Ap/Kp context. Use 95% intervals,
   effect-size/reference bands, and record-length markers. Exact activity
   stratification remains a pre-render decision when the current renderer cannot
   express it cleanly.
5. **Paired empirical-model residual diagnostic** — HASDM-referenced time
   series plus residual FFT for NRLMSISE-00, NRLMSIS 2.0, NRLMSIS 2.1, JB2006,
   and JB2008. It may move to the appendix if the main set is crowded.
6. **Secular density trend comparison** — the organizing result: Panel A is a
   vector reconstruction of the Brown et al. 2024 Figure 2 literature altitude
   profile with attribution; Panel B shows Thermodense's updated solar-adjusted,
   altitude-resolved estimates in percent per decade with HAC 95% intervals and
   record-length markers. Brown is not a 400-km-only profile: do not infer its
   altitude curves from its 400-km tables.

## Resolved decisions

JB2006 and JB2008 are always paired. WACCM-X is excluded from Figure 2.
Figures come before an expansive inventory of analyses, and the paper remains
concise. The machine-readable source is
[`configs/paper/first-paper-figures-v1.json`](../configs/paper/first-paper-figures-v1.json).

## Current implementation map and blockers

`scripts/maunaloa_global_figures.py` now defines the exact Figure 3 scatter
layout and a selected-driver Figure 4 candidate, while preserving its existing
all-driver altitude-correlation output. These are renderers, not committed
figure artifacts; running them needs external prepared data. The remaining
cross-product composites, paired JB residual composition, and Figure 1 common
coverage renderer remain blockers.
Activity stratification for Figure 4 is deliberately unresolved rather than
represented as implemented.

Figure 6 now has a fail-closed final local render with paired external JB2006 and
JB2008 outputs. The repository also generates full-history 1967–2026 hourly
NRLMSISE-00, NRLMSIS 2.0, and NRLMSIS 2.1 baselines with
`scripts/generate_maunaloa_msis_full_history.py`. A no-JB render remains
available as a draft. Panel A remains composed from the bundled
presentation-derived digitized CSV:

```bash
.venv/bin/python -m scripts.compose_density_trend_figure6
```

`data/derived/literature/brown_2024_figure2_digitized.csv` has repository SHA-256
`1fafa2718250adcd01677d4c9257cef4f72d3e7d654a7a14accc2c8cdc216583` and contains
427 plot-precision values vector-extracted from Brown Figure 2 for 16 studies. The
presentation-source CSV had identical rows with CRLF packaging and SHA-256
`1bd91d049f801edba688aabf49952cf8a7a553a5e4b9c47c5ba59909d6a5a7e2`.
They are explicitly **not replacements for the original study data**. The
compositor vector-renders these values at Brown's published x=-7..1
percent-per-decade limits with a study legend, then shares a 0–850 km altitude
axis with Panel B. It writes
`density_trend_figure6.png`, `.pdf`, `_caption.txt`, `_alt_text.txt`, and
`_provenance.json` under `outputs/figures/results/density_trend_figure6/`.
Both panels remain vector in the PDF. The external publisher PDF remains
provenance/reference only and is not copied into the repository; its checksum is
`ac2f2097d3ee28b85bce2e7d082af7e4203459c87e16408480fbdfefa9c392ea`; do not
copy that source PDF into the repository. Record its CC BY 4.0 license, source
URL, CSV checksum/extraction disclaimer, shared limits, and disclosure that Panel
A is digitized third-party figure geometry while Panel B is project data.

## Relationship to the figure-review workbench

The workbench reviews immutable rendered figure sets with provenance. Planned
paper candidates enter it only after their rendered artifacts and provenance are
available. Its currently committed PCMCI material is a sample/supplementary set.

## Figure 6 paired JB finalization seam

The final Figure 6 contract is fail-closed and requires the pair, never a
singleton:

```bash
.venv/bin/python -m scripts.recreate_brown_density_trend_current --require-jb --jb-path PATH
.venv/bin/python -m scripts.compose_density_trend_figure6 --require-jb
```

`PATH` is an externally generated provider-model daily-wide Parquet (the default
is `outputs/figures/results/maunaloa_jb_density_baselines/data/maunaloa_jb_density_baselines_daily_wide.parquet`).
It must contain `date`, `F10.7_OBS_CENTER81`, and both
`jb2006_log10rho_daily_mean_<alt>km` and
`jb2008_log10rho_daily_mean_<alt>km` columns for identical altitude sets. The
validated local product covers 1997-01-06 through 2026-04-19 at 125–825 km in
25-km steps, excluding 1998-05-17 through 1999-03-15 for both models because the
required SET proxy values fail the official driver's validity threshold. The
repository does not distribute JB values or contain SET software or indices: a
local ignored wrapper executes the unmodified provider routines, and provider-
wrapper distribution permission remains unresolved per
`docs/research/empirical-model-calibration-overlap-and-jb-execution.md`.
