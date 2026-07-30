# Thermodense

Thermodense studies long-term thermospheric density variability and its relationship to solar, geomagnetic, and carbon-dioxide-related forcing.

## Language

**NRLMSISE-00**:
The legacy MSIS-family empirical neutral-atmosphere model used as the first baseline in model-performance comparisons.
_Avoid_: MSIS 1.0, MSISE-00 unless quoting a source

**NRLMSIS 2.0**:
The updated MSIS-family empirical neutral-atmosphere model introduced by Emmert et al. for whole-atmosphere density, temperature, and composition.
_Avoid_: MSISE 2.0

**NRLMSIS 2.1**:
The MSIS-family empirical neutral-atmosphere model extending NRLMSIS 2.0 with nitric oxide.
_Avoid_: MSISE 2.1

**JB2006 and JB2008 density baselines**:
The paired Jacchia--Bowman empirical total-density model outputs evaluated on the same timestamps, locations, altitudes, and reference samples as the MSIS-family baselines, so both JB versions can be compared consistently with each other and with the existing model log-density-ratio errors.
_Avoid_: adding JB2008 alone when the post-thesis empirical-model comparison calls for both JB2006 and JB2008

**TU Delft satellite density dataset**:
The accelerometer-derived thermospheric density dataset formed from the available TU Delft CHAMP, GOCE, GRACE-A, GRACE-B, GRACE-FO, Swarm-A, Swarm-B, and Swarm-C mission files.
_Avoid_: all satellites unless the included missions are specified

**Global mean thermospheric density**:
The daily altitude-resolved orbit-derived global mean density record used as the broadest density product in the thesis, described cautiously through Emmert-style satellite orbit/drag analyses when exact source-file construction is not documented in the local data file.
_Avoid_: implying local Mauna Loa sampling or unsupported details about the exact averaging procedure

**TU Delft density analysis**:
The observation-only dependence analysis of measured TU Delft satellite densities and their relationship to solar, geomagnetic, carbon-dioxide-related, altitude, and mission-sampling variables, excluding causal discovery.
_Avoid_: TU Delft model-error analysis when no MSIS residuals are being analysed

**Model log-density-ratio error**:
The model-performance error defined as \(\ln(\rho_m / \rho_\mathrm{ref})\), where zero means perfect agreement, positive values mean model overestimation, and negative values mean model underestimation.
_Avoid_: relative error unless explicitly using \((\rho_\mathrm{ref}-\rho_m)/\rho_m\)

**Mauna Loa MSIS density baselines**:
The standalone density time series from **NRLMSISE-00**, **NRLMSIS 2.0**, and **NRLMSIS 2.1** evaluated at the **Mauna Loa HASDM subset** timestamps, nearest available HASDM longitudes, and HASDM altitudes before computing model error.
_Avoid_: MSIS results when the analysis specifically concerns density baselines rather than residuals

**Mauna Loa HASDM subset**:
The HASDM density time series sampled at the nearest available HASDM latitude to Mauna Loa, with the nearest available longitude selected for each timestamp and altitude, without spatial interpolation.
_Avoid_: bilinearly interpolated HASDM at Mauna Loa

**Daily log-density range**:
The within-day density variability diagnostic computed as daily maximum minus daily minimum in \(\log_{10}\rho\). It may contain carbon-dioxide-cooling-related structure, but it is interpreted cautiously because it is also sensitive to local-time sampling, geomagnetic disturbances, altitude, and daily sample coverage.
_Avoid_: treating density range as a direct CO2 cooling effect without sampling and driver controls

**Daily mean log-density notation**:
The plot notation \(\bar{\ell}_\rho\) denotes daily mean \(\log_{10}\rho\). Use this symbol for average log-density axes, legends, and titles instead of verbose labels such as average log10 density when the plotted quantity is the daily mean log-density diagnostic.
_Avoid_: calling this quantity global mean density when the intended label is the plotted metric rather than the dataset

**Daily log-density range notation**:
The plot notation \(\Delta\ell_\rho\) denotes the **Daily log-density range** diagnostic. Use this symbol for range axes, legends, and titles, while keeping the interpretation cautious because the diagnostic is sensitive to sampling, local-time structure, geomagnetic disturbances, altitude, and daily coverage.
_Avoid_: using density range labels without distinguishing them from mean-density diagnostics

**Dependence analysis and causal discovery**:
The thesis term for the combined workflow where correlation, frequency-domain analysis, and binning are exploratory dependence analyses, while PCMCI\(^+\) is the causal-discovery method used to infer conditional time-series links.
_Avoid_: causality detection when referring to the full workflow

**Dataset provenance and uncertainty**:
The qualitative thesis background for each analysed data product, describing its measurement or model chain, whether it is direct, derived, assimilated, or model-generated, and the main limitations that affect interpretation.
_Avoid_: unsupported numeric accuracy claims unless a cited source provides them clearly

**Activity-sigma bin**:
A driver regime defined by grouping values of a solar or geomagnetic driver by standard deviations from that driver's mean, used to compare fitted density or model-error relationships across low, near-average, elevated, and sparse extreme activity conditions.
_Avoid_: treating sigma bins as equally reliable when sample counts differ strongly

**Subplot-level Results interpretation**:
A Results-writing convention where each visible analytical subplot receives its own interpretation, while sample-count-only support panels may be folded into the corresponding slope or fit interpretation.
_Avoid_: figure-level-only interpretation for multi-panel diagnostic figures

**Binned-fit summary interpretation**:
A Results-writing convention for heatmaps and activity-binned diagnostics that summarises the important slope, correlation, and sample-count patterns instead of mechanically describing every displayed fit statistic.
_Avoid_: treating zero crossings or auxiliary metrics as important when they do not affect the scientific interpretation

**Direct calendar-time density trend**:
The descriptive slope obtained by fitting density against calendar time for a selected data product and altitude, reported in \% per decade for comparison with literature compilations. It preserves the product's sampling interval, solar-cycle history, altitude coverage, and model/reference character, so it is a broad trend diagnostic rather than a deconfounded secular \(\mathrm{CO_2}\) cooling estimate.
_Avoid_: presenting it as a directly comparable causal \(\mathrm{CO_2}\)-only trend unless solar, geomagnetic, sampling, and method differences are controlled.

**Access-conditional thesis reproduction**:
The reproducibility contract in which a code-only, checkpointed workflow rebuilds thesis results from original sources, while clearly requiring credentials or manually supplied files for sources that are not anonymously obtainable.
_Avoid_: claiming that every reader can regenerate every result without the documented external data access

**Thesis result workflow**:
An executable provenance chain corresponding to a thesis Results section, consuming named prepared or derived artifacts to regenerate that section's figures, tables, and result summaries.
_Avoid_: treating every external source or intermediate artifact as a peer reader-facing workflow

**Ionospheric F2-layer peak state**:
The paired observables hmF2, the altitude of the F2-layer electron-density maximum, and NmF2, the electron density at that maximum, used to distinguish a change in ionospheric position from a change in peak intensity.
_Avoid_: using total electron content alone to claim that the ionosphere moved vertically

## Relationships

- **NRLMSISE-00**, **NRLMSIS 2.0**, and **NRLMSIS 2.1** are compared as empirical model baselines against observed or assimilated thermospheric density.
- **JB2006 and JB2008 density baselines** extend the empirical-model comparison as a pair and use the same reference samples and **Model log-density-ratio error** convention as the MSIS-family baselines.
- **Global mean thermospheric density** provides the broadest density reference for time-scale, correlation, binning, and selected causal-discovery analysis before the more sampling-specific TU Delft and HASDM analyses.
- The **TU Delft satellite density dataset** supports both the observation-only **TU Delft density analysis** and the **Model log-density-ratio error** comparison against **NRLMSISE-00**, **NRLMSIS 2.0**, and **NRLMSIS 2.1**.
- The **Mauna Loa MSIS density baselines** are evaluated on the same samples as the **Mauna Loa HASDM subset** before being transformed into **Model log-density-ratio error**.
- **Model log-density-ratio error** is computed consistently for satellite-derived and HASDM reference densities.
- The **Mauna Loa HASDM subset** is used for local density and model-error analysis near the Mauna Loa \\(\mathrm{CO_2}\\) record.
- **Daily log-density range** complements daily mean density in HASDM-based analyses, but it remains an exploratory variability diagnostic rather than a standalone causal estimate of CO2 cooling.
- **Daily mean log-density notation** and **Daily log-density range notation** keep Results figures consistent by labelling daily mean \(\log_{10}\rho\) as \(\bar{\ell}_\rho\) and within-day log-density range as \(\Delta\ell_\rho\).
- **Dataset provenance and uncertainty** frames the datasets used in the thesis before the Results chapter, so that density, cooling, driver, and model-baseline products are not treated as equally direct observations.
- **Activity-sigma bins** support conditional and binned dependence analysis by separating fitted \\(\mathrm{CO_2}\\)-related slopes across driver regimes while preserving sample-count caveats.
- **Subplot-level Results interpretation** makes multi-panel Results figures explicit, while the scatter-plot grouping exception does not apply to **TU Delft density analysis** because mission and sampling differences are part of the result.
- Sample-count-only panels support **Activity-sigma bin** interpretation and may be discussed together with the corresponding fitted-slope or fit-diagnostic panel rather than receiving a standalone paragraph.
- **Binned-fit summary interpretation** keeps conditional heatmap prose focused on the scientifically relevant fitted-slope, correlation, and robustness patterns rather than repeating every annotation in each cell.
- **Direct calendar-time density trend** provides a synthesis-level comparison with historical trend compilations, but it remains method-dependent and should be interpreted together with dependence analyses rather than as a standalone causal cooling estimate.
- **Access-conditional thesis reproduction** persists local products between stages so an interrupted run or expensive analysis can resume without repeating valid upstream work.
- A **Thesis result workflow** is the reader-facing unit of **Access-conditional thesis reproduction**, while source-specific acquisition and preparation steps remain independently runnable provenance operations.
- **Ionospheric F2-layer peak state** complements neutral-density and cooling diagnostics when studying thermosphere--ionosphere coupling; hmF2 represents peak position while NmF2 prevents a height change from being conflated with a change in peak electron density.

## Example dialogue

> **Dev:** "Should the thesis compare MSIS 1.0, 2.0, and 2.1?"
> **Domain expert:** "Use NRLMSISE-00 for the legacy baseline, then NRLMSIS 2.0 and NRLMSIS 2.1 for the newer baselines."

> **Dev:** "Does all TU Delft satellite data include only CHAMP and GOCE?"
> **Domain expert:** "No — use CHAMP, GOCE, GRACE-A, GRACE-B, GRACE-FO, Swarm-A, Swarm-B, and Swarm-C."

> **Dev:** "Should the TU Delft density section include MSIS residual plots?"
> **Domain expert:** "No — reserve MSIS residuals for the model-error section; the TU Delft density section analyses observed density only."

> **Dev:** "Should the TU Delft density section run PCMCI+?"
> **Domain expert:** "No — keep it as dependence analysis because the mission sampling, altitude, local time, and cadence structure would need a dedicated causal graph."

> **Dev:** "Should TU Delft model-error residuals run PCMCI+?"
> **Domain expert:** "No — omit causal discovery for both **TU Delft density analysis** and TU Delft-based **Model log-density-ratio error**, because the residual graph would need to model mission identity, altitude, local-time sampling, cadence, and multiple related model-error series."

> **Dev:** "Should HASDM model errors use a different formula from TU Delft?"
> **Domain expert:** "No — use the same model log-density-ratio error, replacing the TU Delft observed density with the HASDM reference density."

> **Dev:** "Should HASDM density be bilinearly interpolated to exact Mauna Loa?"
> **Domain expert:** "No — use the **Mauna Loa HASDM subset**, and evaluate model errors on the same nearest available samples."

> **Dev:** "Should the MSIS section show residuals only?"
> **Domain expert:** "No — first show the **Mauna Loa MSIS density baselines** as standalone model-density outputs, then analyse residuals under **Model log-density-ratio error**."

> **Dev:** "Should raw Mauna Loa MSIS density baselines use causal discovery?"
> **Domain expert:** "No — treat them as standalone model-density dependence analysis, because causal discovery on prescribed model outputs would mainly reflect model-input structure rather than atmospheric causal links."

> **Dev:** "Should HASDM density range be interpreted as a direct increase caused by CO2 cooling?"
> **Domain expert:** "No — use **Daily log-density range** as an equally visible but cautious variability diagnostic, because range can also reflect sampling, local-time structure, geomagnetic activity, and altitude."

> **Dev:** "When the binned analysis uses low, medium, and high sigmas, should those bins be analysed directly?"
> **Domain expert:** "Yes — treat them as **Activity-sigma bins** and discuss where fitted slopes are strongest, weakest, or sample-limited."

> **Dev:** "Can multi-panel Results figures be interpreted only at figure level?"
> **Domain expert:** "No — use **Subplot-level Results interpretation**, except sample-count-only support panels may be folded into slope interpretation, and selected correlation scatter plots may be grouped by driver when altitude differences add little new interpretation; the scatter exception does not apply to TU Delft figures."

> **Dev:** "Should every number printed in a binned heatmap be discussed in the Results?"
> **Domain expert:** "No — use **Binned-fit summary interpretation** and focus on the important correlation, slope, and sample-count trends; ignore zero crossings unless they materially affect interpretation."

> **Dev:** "Can every reader regenerate every thesis figure without external data credentials?"
> **Domain expert:** "No — **Access-conditional thesis reproduction** provides the complete resumable workflow, but restricted sources must be obtained using the documented access process."

> **Dev:** "Should a reader run separate dataset commands and work out which figures they produce?"
> **Domain expert:** "No — run the **Thesis result workflow** matching the Results section; source-specific downloaders are used only when rebuilding its prepared artifacts."

## Flagged ambiguities

- "MSIS 1.0" was used to mean **NRLMSISE-00**; resolved: thesis and code should use **NRLMSISE-00** as the canonical term.
- "All satellites available from TU Delft" means the eight-mission **TU Delft satellite density dataset**, not only the CHAMP and GOCE missions used in the original Figure 19 comparison.
- "Global mean density" should mean **Global mean thermospheric density**, not a Mauna Loa local or satellite-mission-specific density product.
- "TU Delft analysis" can mean either observed-density analysis or MSIS model-error analysis; resolved: **TU Delft density analysis** is observation-only, while MSIS residual plots belong under **Model log-density-ratio error**.
- "TU Delft causal discovery" was considered for observed-density results; resolved: do not apply causal discovery to **TU Delft density analysis** in this thesis chapter.
- "TU Delft model-error causal discovery" was considered for MSIS residuals; resolved: do not apply causal discovery to TU Delft-based **Model log-density-ratio error** in this thesis chapter.
- "Nearest HASDM grid point" was used ambiguously; resolved: the **Mauna Loa HASDM subset** uses nearest latitude and timestamp-/altitude-specific nearest longitude, not spatial interpolation to a fixed latitude-longitude point.
- "MSIS section" can mean raw model densities or residuals; resolved: use **Mauna Loa MSIS density baselines** for raw model densities and **Model log-density-ratio error** for residuals.
- "MSIS causal discovery" can refer to raw model densities or residual model errors; resolved: do not apply causal discovery to **Mauna Loa MSIS density baselines**, but keep HASDM-based **Model log-density-ratio error** causal discovery as a residual diagnostic.
- "Density range" can sound like a direct CO2 cooling response; resolved: use **Daily log-density range** as a cautious dependence diagnostic, not as a direct causal effect without additional controls.
- "Average log10 density" and "density range" in figure labels were visually inconsistent; resolved: use **Daily mean log-density notation** \(\bar{\ell}_\rho\) and **Daily log-density range notation** \(\Delta\ell_\rho\) in Results figures.
- "Dataset accuracy" should not imply a complete numeric uncertainty budget; resolved: use **Dataset provenance and uncertainty** as qualitative, cited context unless specific uncertainty values are available from the literature.
- "High/low sigmas" in binned plots means **Activity-sigma bins** of the conditioning driver, not uncertainty sigma or statistical significance.
- "TUDelft" and "TuDelft" are shorthand/capitalization slips; resolved: use **TU Delft** consistently when referring to the institution or dataset.
- "Each subplot" means **Subplot-level Results interpretation**, with only the stated sample-count support-panel and correlation-scatter grouping exceptions.
- "Analyse the heatmap" does not mean describing every annotated metric; resolved: use **Binned-fit summary interpretation** and discuss only scientifically relevant trends.
- "Density trend" in the synthesis figure means **Direct calendar-time density trend** for the current thesis products, not a fully deconfounded literature-equivalent secular \(\mathrm{CO_2}\) trend.
- "Replicate the results" was ambiguous about distributing prepared data; resolved: use **Access-conditional thesis reproduction**, keep large datasets out of the repository, and persist ignored local checkpoints between pipeline stages.
- The proposed "dataset axis" mixed external sources, model-generated baselines, and model-error products; resolved: reader-facing execution mirrors **Thesis result workflows**, while acquisition remains split by external source dataset.
- "CO2" as an acronym entry was considered; resolved: CO₂ is a chemical formula, not an acronym — remove it from the acronym list and use a `\coo` macro (\ensuremath{\mathrm{CO_2}}) for consistent formatting instead of `\ac{CO2}`.
- Acronyms in formulas, subscripts, or section titles were considered; resolved: no `\ac{}` calls inside math environments or in `\section`/`\subsection`/`\subsubsection` titles — spell out or use the short form directly without the acronym macro.
- "SABER" as an acronym entry was considered; resolved: SABER is an instrument name, not a generic term — remove it from the acronym list, spell out "Sounding of the Atmosphere using Broadband Emission Radiometry (SABER)" once at first mention in the dataset and \(\mathrm{CO_2}\) cooling-rate explanation, then use plain "SABER" text onwards.
- "Electron content" was proposed to track how the ionosphere's position changes; resolved: use **Ionospheric F2-layer peak state** (hmF2 together with NmF2), because total electron content is column-integrated and does not locate the layer by itself.
