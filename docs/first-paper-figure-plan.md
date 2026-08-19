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
6. **Secular density trend comparison** — the organizing result: original
   thesis/literature framing beside updated solar-adjusted, altitude-resolved
   estimates and Brown et al. 2024 context, in percent per decade with HAC 95%
   intervals and record-length markers.

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
cross-product composites, paired JB residual composition, Figure 1 common
coverage renderer, and Figure 6 comparison/provenance audit remain blockers.
Activity stratification for Figure 4 is deliberately unresolved rather than
represented as implemented.

## Relationship to the figure-review workbench

The workbench reviews immutable rendered figure sets with provenance. Planned
paper candidates enter it only after their rendered artifacts and provenance are
available. Its currently committed PCMCI material is a sample/supplementary set.
