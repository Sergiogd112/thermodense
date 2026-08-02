# ParCorr surrogate robustness diagnostics

- Runner: `pcmci-real-surrogates-1`
- Commit: `1870158191b3e1a883659aed3fd1b166cb81ab73`
- Input SHA-256: `a7844c1dda119114f0d5a228cead3fac6bdbbcf17908f62b67b9ed069d2f1aab`
- Physical nodes: F10.7 center81, Ap, NOAA GML Mauna Loa/Maunakea CO2,
  and HASDM daily mean log-density at 325/825 km
- Controls: 5 Gaussian white-noise, 3 six-month sine-plus-noise, and 3
  11.4-year sine-plus-noise nodes (seed `20260601`)
- PCMCI+: ParCorr, `tau_max=180`, `pc_alpha=0.05`, no FDR correction
- Wall time: 3863.95 s (64.4 min); peak RSS: 808 MB

## Selected control links

The canonical selected-link table contains 633 links involving at least one
surrogate node:

| relation | selected links |
|---|---:|
| physical → surrogate | 36 |
| surrogate → physical | 30 |
| surrogate ↔ surrogate | 567 |

621 are lagged and 12 contemporaneous. Surrogate-to-physical links target Ap
(23), 325-km density (4), and 825-km density (3). No surrogate may cause F10.7
because the pre-registered F10.7-exogenous assumptions apply to the augmented
graph.

The control links are weak: surrogate-to-physical median `|partial r|` is
0.0258 and the maximum is 0.0385. Across every surrogate-involving link, the
maximum is 0.0542. By comparison, the primary five-node selected links have
median `|partial r|` 0.0774 and maximum 0.7987; 28/42 primary matrix entries
have `p <= 0.001`, versus 2/30 surrogate-to-physical links.

## Physical-topology stability

Adding all 11 controls retains 40 of the 42 selected physical-only matrix
entries (95.2%). Three entries change:

- removed: Ap → Ap at 27 days;
- removed: CO2 → Ap at 142 days;
- added: Ap → 825-km density at 38 days.

The primary and augmented physical subgraphs therefore remain highly similar,
but the changed weak/long-lag links should not be treated as robust claims.

## Interpretation

The large number of weak control links is expected when thousands of
autocorrelated candidate links are screened at `pc_alpha=0.05` with
`fdr_method="none"`. The surrogate run is not a zero-link pass/fail test. It
shows that strong primary links are well separated in effect size while weak,
isolated, or long-lag primary links need multiplicity and surrogate-sensitivity
caveats.

Useful follow-up nulls are phase-randomized or IAAFT surrogates, which preserve
the observed spectrum/autocorrelation, and circular time shifts, which preserve
each series and its missingness while breaking alignment. Those should remain
separate robustness runs rather than additional primary nodes.
