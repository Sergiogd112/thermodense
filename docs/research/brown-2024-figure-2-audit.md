# Brown 2024 Figure 2 audit

## Source and license

Brown, M. K., H. G. Lewis, A. J. Kavanagh, I. Cnossen, and S. Elvidge (2024),
“Future Climate Change in the Thermosphere Under Varying Solar Activity
Conditions,” *Journal of Geophysical Research: Space Physics*, 129(9),
e2024JA032659, https://doi.org/10.1029/2024JA032659. The article is CC BY 4.0.

- Official DOI: <https://doi.org/10.1029/2024JA032659>
- NERC accepted manuscript:
  <https://nora.nerc.ac.uk/id/eprint/537943/1/JGR%20Space%20Physics%20-%202024%20-%20Brown%20-%20Future%20Climate%20Change%20in%20the%20Thermosphere%20Under%20Varying%20Solar%20Activity%20Conditions.pdf>

The verified official publisher PDF used as Figure 6 provenance/reference has SHA-256
`ac2f2097d3ee28b85bce2e7d082af7e4203459c87e16408480fbdfefa9c392ea`. The
source PDF remains external to this repository. Record its source URL, CC BY 4.0
license, and checksum in figure artifact provenance.

## Verified Figure 2 scope

Figure 2 plots reported density trend (%/decade) against altitude. It combines
observed/derived and modeled literature profiles, with error bars where those
are available. Tables 1 and 2 summarize values at 400 km, and the Figure 2
caption calls this a 400-km literature comparison; the plotted curves
themselves span altitude. Figure 2 is an updated version of the corresponding
Emmert (2008) and Solomon et al. (2015) figures.

## Digitized reconstruction and Figure 6 contract

Figure 6 Panel A is vector-rendered from
`data/derived/literature/brown_2024_figure2_digitized.csv`, whose repository
SHA-256 is
`1fafa2718250adcd01677d4c9257cef4f72d3e7d654a7a14accc2c8cdc216583`. The
presentation-source CSV had identical rows with CRLF packaging and SHA-256
`1bd91d049f801edba688aabf49952cf8a7a553a5e4b9c47c5ba59909d6a5a7e2`. It
contains 427 plot-precision values vector-extracted from Brown et al. (2024)
Figure 2 for 16 studies. These values are explicitly **not replacements for the
original study data**. Panel A uses the literature altitude profile rather than
profiles inferred from the 400-km Tables 1 and 2; Panel B is Thermodense's
updated solar-adjusted trends.

The compositor maps the digitized series directly to Matplotlib vectors at
Brown's published x=-7..1 percent-per-decade limits and a shared 0–850 km
altitude axis. It records the CSV path and checksum, extraction basis/disclaimer,
16-study/427-row count, external PDF provenance, shared limits, and a disclosure
that Panel A is generated from digitized third-party figure geometry under CC BY
4.0 while Panel B is generated from project data. Render with:

```bash
python scripts/compose_density_trend_figure6.py
```
