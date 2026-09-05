# Tigramite decision for the exploratory 3-hour all-altitude run

Date: 2026-08-29

## Decision

Use **PCMCI+ with analytic ParCorr**, with one graph per empirical-model family
and geomagnetic-driver variant. Each graph contains the 27 altitude error
channels as separate target nodes, spline-interpolated raw F10.7, and either
native 3-hour Ap or native 3-hour Kp. The six-month window is represented as
183 days, or 1,464 three-hour lag steps.

The graph is deliberately restricted to driver history, driver-to-target links,
and each target's own history. Cross-altitude target links are not tested in this
exploratory Figure 5 diagnostic. This restriction makes “all altitudes in one
run” an inferential-family and execution choice, not a claim that vertical
coupling was estimated.

## Why not another Tigramite algorithm?

- **PCMCI** is designed primarily for time-lagged discovery and can also test
  unoriented lag-zero associations. PCMCI+ extends it to a full lagged and
  contemporaneous equivalence-class graph. PCMCI+ is retained here for
  continuity with the daily diagnostic and explicit lag-zero handling, not
  because PCMCI is incapable of returning a lag-zero association. No
  same-bin physical ordering is inferred from the observational timestamps.[^pcmci][^pcmciplus]
- **LPCMCI** is the relevant choice when latent confounding is part of the
  stated model. It estimates lagged and contemporaneous relations under latent
  confounding, but Tigramite still labels the implementation experimental. It
  is not a computational shortcut for a large lag window.[^lpcmci-paper][^lpcmci-docs]
- **J-PCMCI+** pools multiple datasets with observed and latent contexts. An
  altitude-as-spatial-context design would estimate a mechanism shared across
  altitude-defined domains. It would not retain 27 altitude channels as 27
  system nodes and therefore answers a different question from an all-altitude
  graph.[^jpcmci]
- **RPCMCI** estimates persistent causal regimes by repeatedly invoking PCMCI.
  Tigramite restricts it to a single scalar dataset; it is neither an
  all-altitude pooling method nor a cheaper large-lag PCMCI variant.[^rpcmci]

No Tigramite algorithm identified in the official documentation or primary
papers is specifically optimized to turn 27 simultaneous altitude channels and
1,464 lags into one unrestricted graph. LPCMCI and J-PCMCI+ change assumptions
and estimands rather than solving that scaling problem. For an unrestricted
vertical-coupling study, the existing project recommendation remains overlapping
altitude groups plus per-altitude companion graphs; a huge graph is at most a
robustness diagnostic.

## Scope and cautions

Omitting cross-altitude links in Tigramite's `link_assumptions` structurally
declares those links absent; it is not only a performance optimization. Shared
vertical dynamics can therefore remain an omitted-common-cause risk. The result
is conditional on that restricted graph and requires overlapping-group or
neighbor-link sensitivity work before any vertical-coupling interpretation.

PCMCI+'s consistency result assumes causal sufficiency, faithfulness, the
causal Markov condition, causal stationarity, and an acyclic contemporaneous
graph.[^pcmciplus] The current 25.55-year calendar contains 74,657 three-hour
timestamps, so a 1,464-step lag leaves far more than one lag window of support;
the six months is a maximum lag, not the sample duration. Runtime and
conditioning-set stability at this large lag still require empirical checks.

The current run is labeled exploratory because spline interpolation,
preprocessing, stationarity, linear/Gaussian ParCorr adequacy, model-error
construction, and post-selection inference all require separate scientific
qualification. The spline preserves daily knots and forbids extrapolation, but
no linear/step-hold interpolation sensitivity has yet been run. BH correction
is applied over exactly 2 drivers × 27 targets × 1,465 lags within each graph.
The ten model/geomagnetic graphs are not jointly adjusted, so this does not make
algorithm-selected p-values globally confirmatory.

[^pcmci]: Runge et al. (2019), “Detecting and quantifying causal associations in large nonlinear time series datasets,” *Science Advances* 5, eaau4996, <https://doi.org/10.1126/sciadv.aau4996>.
[^pcmciplus]: Runge (2020), “Discovering contemporaneous and lagged causal relations in autocorrelated nonlinear time series datasets,” *PMLR* 124, <https://proceedings.mlr.press/v124/runge20a.html>.
[^lpcmci-paper]: Gerhardus and Runge (2020), “High-recall causal discovery for autocorrelated time series with latent confounders,” *NeurIPS 33*, <https://proceedings.neurips.cc/paper_files/paper/2020/hash/94e70705efae423efda1088614128d0b-Abstract.html>.
[^lpcmci-docs]: Tigramite LPCMCI implementation documentation, <https://jakobrunge.github.io/tigramite/_modules/tigramite/lpcmci.html>.
[^jpcmci]: Günther, Ninad, and Runge (2023), “Causal Discovery for time series from multiple datasets with latent contexts,” *PMLR* 216, <https://proceedings.mlr.press/v216/gunther23a.html>.
[^rpcmci]: Tigramite RPCMCI implementation documentation and Saggioro et al. (2020), <https://jakobrunge.github.io/tigramite/_modules/tigramite/rpcmci.html> and <https://doi.org/10.1063/5.0020538>.
