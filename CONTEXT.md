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

**Thesis-continuation paper**:
The first paper preserves the thesis's main goal, specific objectives, overall structure, and combined dependence-analysis, empirical-model, and causal-discovery scope, while strengthening the evidence with refreshed sources, more complete analysis, and the paired JB2006/JB2008 baselines alongside the MSIS family. The thesis-defense presentation supplies the initial main-text result sequence and evidence hierarchy, while the thesis supplies the complete methods, results, and caveats. The paper's organizing claim is the forcing hierarchy: solar forcing dominates resolved density variability, geomagnetic forcing supplies secondary disturbed-time structure, and CO2-related evidence is physically consistent but conditional. The empirical-model comparison tests how faithfully model errors preserve that hierarchy rather than serving as a standalone matched-error benchmark.
_Avoid_: narrowing the first paper to only empirical-model benchmarking; presenting all results as an unranked inventory; replacing the thesis question with a new question merely because the paper is more complete

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

**Density target diagnostic**:
A named, source-independent density quantity exposed as an **Analysis target**. The initial diagnostics are daily mean \(\log_{10}\rho\) and **Daily log-density range**; model-error targets separately use **Model log-density-ratio error**. Source adapters map their native columns into these definitions rather than exposing arbitrary numeric columns.
_Avoid_: raw source-column selection as scientific configuration; silently mixing raw density, log10 density, and natural-log error

**Dependence analysis and causal discovery**:
The thesis term for the combined workflow where correlation, frequency-domain analysis, and binning are exploratory dependence analyses, while PCMCI\(^+\) is the causal-discovery method used to infer conditional time-series links.
_Avoid_: causality detection when referring to the full workflow

**Dataset provenance and uncertainty**:
The qualitative thesis background for each analysed data product, describing its measurement or model chain, whether it is direct, derived, assimilated, or model-generated, and the main limitations that affect interpretation.
_Avoid_: unsupported numeric accuracy claims unless a cited source provides them clearly

**Activity-sigma bin**:
A driver regime defined by grouping values of a solar or geomagnetic driver by standard deviations from that driver's mean, used to compare fitted density or model-error relationships across low, near-average, elevated, and sparse extreme activity conditions.
_Avoid_: treating sigma bins as equally reliable when sample counts differ strongly

**Activity-binned product**:
A Pearson/Fisher or fitted-slope result produced by binning one selected solar or geomagnetic driver at a time, then estimating the target's relationship with each other selected driver in a separate plot. The binning driver is not also used as the fitted predictor. Correlation or slope with uncertainty is plotted against altitude, with one consistently coloured series per activity bin; a companion sample-count histogram reuses the same bin colours. A core run may emit separate solar-binned and geomagnetic-binned products from the same prepared table without expanding them into separate pipeline runs.
_Avoid_: fitting the target against the same driver used to define the bins; combining several fitted drivers in one plot; changing bin colours between estimate and sample-count panels

**Driver scatter product**:
A descriptive figure for one analysis-target altitude with one subplot for every selected core or extension driver node. Each selected altitude channel of an altitude-resolved driver such as a SABER product remains a separate subplot; large selections are paginated rather than vertically averaged or matched to the target altitude. Subplots show points only by default; configuration may enable a fit for all or named drivers, using the same linear-fit artifact and uncertainty estimator as the slope analysis.
_Avoid_: omitting extension drivers silently; collapsing selected SABER altitude channels into one scatter variable; fitting an independent renderer-only trendline

**Subplot-level Results interpretation**:
A Results-writing convention where each visible analytical subplot receives its own interpretation, while sample-count-only support panels may be folded into the corresponding slope or fit interpretation.
_Avoid_: figure-level-only interpretation for multi-panel diagnostic figures

**Binned-fit summary interpretation**:
A Results-writing convention for heatmaps and activity-binned diagnostics that summarises the important slope, correlation, and sample-count patterns instead of mechanically describing every displayed fit statistic.
_Avoid_: treating zero crossings or auxiliary metrics as important when they do not affect the scientific interpretation

**Direct calendar-time density trend**:
The descriptive slope obtained by fitting density against calendar time for a selected data product and altitude, reported in \% per decade for comparison with literature compilations. It preserves the product's sampling interval, solar-cycle history, altitude coverage, and model/reference character, so it is a broad trend diagnostic rather than a deconfounded secular \(\mathrm{CO_2}\) cooling estimate.
_Avoid_: presenting it as a directly comparable causal \(\mathrm{CO_2}\)-only trend unless solar, geomagnetic, sampling, and method differences are controlled.

**Solar-adjusted direct trend estimator**:
The dedicated trend module that estimates a **Direct calendar-time density trend** while fitting configured solar-control terms. It returns trend estimates and uncertainty rather than a solar-adjusted input series for frequency analysis, binning, or causal discovery.
_Avoid_: remove solar effects as a shared preprocessing stage

**PCMCI preprocessing profile**:
One of two pre-registered input transformations for causal discovery: the primary detrended-anomaly profile uses seasonal anomalies with the 3-year rolling mean removed, while the robustness profile uses seasonal anomalies without that long-term removal. A seasonal-anomaly robustness run is still computed when its pre-registered stationarity checks fail, but it then provides sensitivity evidence only and cannot support a causal claim. Raw-standardized inputs support stationarity diagnostics only, and the former CO2-preserved anomaly is not a default PCMCI\(^+\) claim.
_Avoid_: treating all historical preprocessing variants as peer primary analyses; composing arbitrary unlabelled transformations

**PCMCI stationarity qualification**:
The interpretation gate applied separately to every **PCMCI preprocessing profile**. Each node is tested on its longest contiguous finite span using ADF and KPSS at a familywise 0.05 level with Holm correction across the graph's nodes; qualification requires ADF rejection of a unit root and no KPSS rejection of level stationarity for every node. Companion 365-day rolling mean and variance plots expose practical drift but do not independently pass or fail the profile. A failed profile is still computed as sensitivity evidence but cannot support a causal claim.
_Avoid_: dropping a failed robustness run; treating one uncorrected p-value or a visual rolling-window judgment as sufficient stationarity evidence

**PCMCI F10.7 timing variant**:
The matched choice of solar driver for PCMCI\(^+\): raw observed daily F10.7 is the primary variant because it preserves the timing of daily forcing, while the centered 81-day mean is a robustness variant because its symmetric window includes future observations. It is crossed with both registered **PCMCI preprocessing profiles** in a full 2×2 sensitivity matrix: raw plus detrended anomaly is primary, the two one-factor changes are robustness runs, and centered plus seasonal anomaly is the interaction diagnostic. All four cells use the same accepted-quality rows and otherwise identical graph, lag, and inference settings.
_Avoid_: treating the centered 81-day mean as a primary causal driver; interpreting its lagged links as daily forcing times without accounting for the centered window

**PCMCI sensitivity agreement**:
The tiered interpretation of links within each full 2×2 **PCMCI F10.7 timing variant** by **PCMCI preprocessing profile** matrix. The production ParCorr matrix uses the full 0--180-day physical lag window; a separate CMIknn matrix uses only 0--10 days under the current resource cap and is a nonlinear method sensitivity, not a substitute for untested longer lags. GPDCtorch is a gated third full-row 0--10-day matrix: an eligible GPU must first complete a full-row `tau_max=1` capability probe within five hours, then the raw-plus-detrended primary `tau_max=10` cell within 24 hours before the other three cells may run. It is never replaced by a shorter prefix of the record. Within any matrix, a main-text robust link appears in the primary graph and survives both single-factor checks: detrending agreement requires the same lag and sign for the same F10.7 series, while raw-versus-centered agreement requires only the same source, target, and sign and never implies an equivalent delay. Primary-only links remain visible as factor-sensitive results, and links found only outside the primary cell are exploratory. CMIknn or GPDCtorch agreement within 0--10 days strengthens a ParCorr result; disagreement qualifies but does not automatically veto it because the tests have different power and estimands. The centered-plus-seasonal interaction cell diagnoses combined sensitivity rather than defining the primary intersection.
_Avoid_: hiding disagreements; promoting alternate-only links; matching raw and centered F10.7 links by a broad lag tolerance

**Frequency-analysis method**:
The sampling-driven choice between FFT for regularly aligned series and Lomb--Scargle for irregularly sampled series. The frequency-analysis module selects the method from the prepared target's sampling contract and records that choice in provenance rather than requiring a routine user setting.
_Avoid_: running FFT on irregular samples; treating FFT and Lomb--Scargle as interchangeable configuration variants

**Analysis capability**:
A target or source adapter's declared support for analysis methods, cadences, altitude structure, and geographic matching. The planner rejects requests outside those capabilities. TU Delft density and TU Delft model-error targets remain ineligible for PCMCI\(^+\) until their mission and sampling graph is resolved; raw executable empirical-model densities from MSIS or JB are also ineligible because their links primarily reflect prescribed model inputs. Native-grid WACCM-X density may use PCMCI\(^+\), as may otherwise eligible observed/reference and model-error targets.
_Avoid_: warning and continuing with an unsupported method; exposing every analysis for every target

**Analysis result artifact**:
A canonical machine-readable analysis result with its provenance, such as a spectrum table, activity-binned statistics, PCMCI\(^+\) matrices and links, or direct-trend estimates. Figures are rendered from these artifacts and are never the only retained result or an upstream input.
_Avoid_: treating a plot file as the analysis result; rerunning expensive analysis solely to change presentation

**Study artifact graph**:
The planned provenance graph in which acquisition, parsing, reference sampling, and shared driver preparation produce reusable immutable artifacts consumed by independently expanded analysis runs. Compatible cases share upstream artifacts and checkpoints rather than repeating the same work.
_Avoid_: a self-contained download-to-plot pipeline for every target and driver variant; manually wiring shared cache paths between runs

**Study execution identity**:
The hybrid identity in which a human-readable execution ID groups one invocation of a study, while each prepared artifact and analysis case has a deterministic fingerprint derived from normalized configuration, input manifests, and implementation version. Reuse requires the fingerprint to match.
_Avoid_: timestamp-only cache keys; reusing a checkpoint because its path exists without validating provenance

**Analysis selection and preset**:
The explicit list of analysis modules requested for a target family, optionally supplied through a named reusable preset. The planner expands the preset and validates every requested analysis against the target's **Analysis capabilities**; adapters do not silently choose analyses and the pipeline does not run every method by default.
_Avoid_: hidden per-target analysis defaults; automatic run-everything behavior

**Access-conditional thesis reproduction**:
The reproducibility contract in which a code-only, checkpointed workflow rebuilds thesis results from original sources, while clearly requiring credentials or manually supplied files for sources that are not anonymously obtainable.
_Avoid_: claiming that every reader can regenerate every result without the documented external data access

**Thesis result workflow**:
An executable provenance chain corresponding to a thesis Results section, consuming named prepared or derived artifacts to regenerate that section's figures, tables, and result summaries.
_Avoid_: treating every external source or intermediate artifact as a peer reader-facing workflow

**Ionospheric F2-layer peak state**:
The paired observables hmF2, the altitude of the F2-layer electron-density maximum, and NmF2, the electron density at that maximum, used to distinguish a change in ionospheric position from a change in peak intensity.
_Avoid_: using total electron content alone to claim that the ionosphere moved vertically

**Analysis target**:
The one physical density diagnostic or **Model log-density-ratio error** family analysed by a pipeline run. Descriptive analyses may retain that target's full altitude axis, but selecting different density products, model outputs, or error definitions expands into separate runs rather than combining them as peer targets.
_Avoid_: study variable; treating Global mean thermospheric density and a HASDM-derived target as one target

**PCMCI altitude group**:
A configured set of altitude channels represented as separate target nodes in one multivariate PCMCI\(^+\) graph. Selecting several groups expands into independent PCMCI\(^+\) runs; for example, `[325, 500], 825` means one graph with distinct 325 km and 500 km nodes and another graph with only 825 km.
_Avoid_: averaging grouped altitude channels into one density series

**Physical PCMCI lag window**:
The configured minimum and maximum causal-discovery lag expressed as physical durations, such as `0d` through `180d`. The planner converts those durations to integer lag steps only after resolving the run's **Cadence variant**, rejects durations that do not fall on cadence boundaries, and records both forms in provenance.
_Avoid_: exposing one raw `tau_max` step count whose physical meaning changes between daily and 3-hour runs; choosing lag limits silently inside the PCMCI adapter

**Reference sampling frame**:
The exact timestamps, locations, altitudes, and sample identities extracted from the **TU Delft satellite density dataset** or HASDM and used unchanged to evaluate empirical-model density baselines and **Model log-density-ratio error**.
_Avoid_: locations alone; inventing a model-comparison sampling frame for **Global mean thermospheric density** when its spatial construction is not documented

**Paired model comparison sample**:
The common complete reference sample set frozen before evaluating a selected model set. Every model-density and model-error run in the comparison uses exactly those rows, and cross-model renderers combine separate result artifacts only after this pairing is enforced.
_Avoid_: comparing models evaluated on different available rows; intersecting samples only at plotting time

**Conditional empirical-model performance claim**:
The minimum model-comparison claim that, on a **Paired model comparison sample**, empirical-model errors differ by reference product, altitude, activity regime, **Calibration-domain representativeness**, and error metric. Its minimum evidence set extends the Emmert et al. (2020) Figure 19 recreation against TU Delft to JB2006 and JB2008, then compares HASDM-referenced error time series and residual FFT structure across the same empirical-model set. Each reference uses its full quality-controlled all-model common sample window; the Figure 19 recreation alone retains its published CHAMP 2006--2009 exclusion, and every mission and HASDM endpoint is reported. Direct model-to-model performance differences report signed mean log-ratio error for bias and mean absolute log error for accuracy, with median absolute log error as an outlier-robust sensitivity. Their paired intervals use a stationary bootstrap over calendar dates with a 27-day mean block, preserving the complete model vector within every resampled date and keeping missions and pre-specified strata separate; 54- and 81-day mean blocks are sensitivity checks. The seven formal contrasts shown within one comparison figure receive joint 95% simultaneous coverage from that bootstrap rather than separate nominal intervals. TU Delft mission uncertainty bands remain separate reference-attribution guides rather than confidence intervals or strict error bounds. FFT structure is a residual-timescale diagnostic rather than an accuracy ranking. Other model-error diagnostics are optional extensions rather than prerequisites for this claim. Improvements may be reported for those pre-specified conditions, but the study does not collapse them into one universal model winner.
_Avoid_: claiming that one JB or MSIS-family model is best overall from a pooled score; describing a conditional improvement without naming its reference, altitude or activity context, and metric

**Calibration-domain representativeness**:
The documented relationship between an empirical model's fitting data and an evaluation stratum. The first-paper audit classifies each model/reference stratum as documented direct overlap, related-source or partial overlap, no documented overlap, or unknown using cited missions, instruments, epochs, altitude ranges, and activity coverage. A quantitative state-space support distance is optional and may be used only if comparable fitting-data distributions are available for every compared model.
_Avoid_: unexpected atmospheric state without a measurable definition; inferring poor training representation from a large residual; calling evaluation independent when its mission, source, or epoch overlaps documented fitting data

**Chronological empirical-model contrast set**:
The seven pre-specified model pairs used for direct better/worse claims: all three internal MSIS-family pairs (NRLMSISE-00 versus NRLMSIS 2.0, NRLMSISE-00 versus NRLMSIS 2.1, and NRLMSIS 2.0 versus NRLMSIS 2.1), the JB2006-versus-JB2008 internal pair, the chronological cross-family neighbours NRLMSISE-00 versus JB2006 and JB2008 versus NRLMSIS 2.0, and the latest cross-family comparison JB2008 versus NRLMSIS 2.1.
_Avoid_: treating every one of the ten possible pairs as a confirmatory contrast; omitting established within-family context; comparing historical cross-family versions without a declared chronological reason

**Model density source**:
Either an executable empirical-model adapter, such as the MSIS-family or JB models, or a precomputed physics-based model dataset, such as WACCM-X, that produces modelled thermospheric density. WACCM-X may be analysed on its native grid as a standalone model-density target, but it must be explicitly aligned to a **Reference sampling frame** before computing **Model log-density-ratio error**.
_Avoid_: treating WACCM-X as an observational density dataset or as an executable empirical-model adapter

**Solar-proxy variant**:
A matched analysis run using exactly one solar proxy, either F10.7 or F30, while holding the analysis target, cadence, geomagnetic driver, preprocessing choices, and accepted-quality sample rows fixed. Selecting both proxies expands into separate runs on their exact common-quality rows rather than placing both correlated proxies in one input matrix.
_Avoid_: including F10.7 and F30 together as independent drivers in one analysis case

**Geomagnetic-driver variant**:
A matched analysis run using exactly one geomagnetic driver, either Ap or Kp, while holding the analysis target, cadence, solar proxy, preprocessing choices, and sample rows fixed. Selecting both drivers expands into separate runs rather than placing both related indices in one input matrix.
_Avoid_: including Ap and Kp together as independent drivers in one analysis case

**Core driver run and extension run**:
A core driver run analyses one target with Mauna Loa tropospheric CO2, one **Solar-proxy variant**, and one **Geomagnetic-driver variant**. Selecting narrower-coverage SABER or **Ionospheric F2-layer peak state** inputs creates explicit extension runs on documented common-coverage windows while preserving the corresponding core run.
_Avoid_: silently shortening the core run when an optional driver is selected; imputing optional products across unsupported periods

**Extension overlap control**:
A core-driver-only run restricted to the exact rows used by a SABER or ionospheric extension run. Every extension comparison retains the full-window core run, this overlap-window control, and the extension run so changes caused by the coverage window can be distinguished from changes caused by adding nodes.
_Avoid_: comparing an extension only with a longer core run; discarding the full-window core result

**SABER extension selection**:
An arbitrary selected combination of SABER CO2 and NO direct-cooling products and OH and O2 emission-rate proxy products. Every product remains a separate typed node with its own role and units even when several coexist in one extension run.
_Avoid_: calling OH or O2 a cooling rate; averaging cooling rates and emission proxies into one SABER variable

**Ionospheric extension run**:
An extension run that adds paired hmF2 and NmF2 nodes from one named ionospheric source and requires that source's geographic support to be compatible with the analysis target. The initial GIRO Lualualei source is restricted to Hawaii/Mauna Loa-compatible targets; COSMIC remains a separate future source rather than being averaged with GIRO.
_Avoid_: attaching Lualualei as a global contextual driver; including hmF2 without NmF2

**Cadence variant**:
A matched analysis run using exactly one explicit temporal cadence. Selecting daily and 3-hour cadences expands into separate runs; daily is the initial supported cadence, while 3-hour runs remain unavailable until their alignment, aggregation, and low-frequency-driver rules are resolved.
_Avoid_: mixed-cadence input matrices; silently upsampling or downsampling inputs

**Bounded gap interpolation**:
The explicit per-cadence policy for linearly filling only target gaps no longer than a configured `max_gap_steps`, bounded by real samples on both sides. Extrapolation is forbidden, longer gaps remain missing, and an imputation mask is retained in result provenance.
_Avoid_: filling every internal gap; source-specific silent interpolation

## Relationships

- **NRLMSISE-00**, **NRLMSIS 2.0**, and **NRLMSIS 2.1** are compared as empirical model baselines against observed or assimilated thermospheric density.
- **JB2006 and JB2008 density baselines** extend the empirical-model comparison as a pair and use the same reference samples and **Model log-density-ratio error** convention as the MSIS-family baselines.
- The **Thesis-continuation paper** retains the thesis's broad density-variability and forcing question; its expanded JB/MSIS comparison supports that question without becoming the paper's sole organizing claim.
- **Global mean thermospheric density** provides the broadest density reference for time-scale, correlation, binning, and selected causal-discovery analysis before the more sampling-specific TU Delft and HASDM analyses.
- The **TU Delft satellite density dataset** supports both the observation-only **TU Delft density analysis** and the **Model log-density-ratio error** comparison against **NRLMSISE-00**, **NRLMSIS 2.0**, and **NRLMSIS 2.1**.
- The **Mauna Loa MSIS density baselines** are evaluated on the same samples as the **Mauna Loa HASDM subset** before being transformed into **Model log-density-ratio error**.
- **Model log-density-ratio error** is computed consistently for satellite-derived and HASDM reference densities.
- The **Mauna Loa HASDM subset** is used for local density and model-error analysis near the Mauna Loa \\(\mathrm{CO_2}\\) record.
- **Daily log-density range** complements daily mean density in HASDM-based analyses, but it remains an exploratory variability diagnostic rather than a standalone causal estimate of CO2 cooling.
- **Daily mean log-density notation** and **Daily log-density range notation** keep Results figures consistent by labelling daily mean \(\log_{10}\rho\) as \(\bar{\ell}_\rho\) and within-day log-density range as \(\Delta\ell_\rho\).
- **Density target diagnostics** give observational and model sources the same target vocabulary while preserving the separate natural-log convention for model error.
- **Dataset provenance and uncertainty** frames the datasets used in the thesis before the Results chapter, so that density, cooling, driver, and model-baseline products are not treated as equally direct observations.
- **Activity-sigma bins** support conditional and binned dependence analysis by separating fitted \\(\mathrm{CO_2}\\)-related slopes across driver regimes while preserving sample-count caveats.
- Each configured binning driver produces its own **Activity-binned product** within the same core run, preserving a direct relationship to the shared prepared inputs.
- **Driver scatter products** expose the target's relationship with every selected input node at each target altitude while preserving altitude-resolved driver channels.
- **Subplot-level Results interpretation** makes multi-panel Results figures explicit, while the scatter-plot grouping exception does not apply to **TU Delft density analysis** because mission and sampling differences are part of the result.
- Sample-count-only panels support **Activity-sigma bin** interpretation and may be discussed together with the corresponding fitted-slope or fit-diagnostic panel rather than receiving a standalone paragraph.
- **Binned-fit summary interpretation** keeps conditional heatmap prose focused on the scientifically relevant fitted-slope, correlation, and robustness patterns rather than repeating every annotation in each cell.
- **Direct calendar-time density trend** provides a synthesis-level comparison with historical trend compilations, but it remains method-dependent and should be interpreted together with dependence analyses rather than as a standalone causal cooling estimate.
- The **Solar-adjusted direct trend estimator** owns solar adjustment for the trend output; it does not mutate the shared analysis input used by other methods.
- PCMCI\(^+\) runs compare the primary and robustness **PCMCI preprocessing profiles** explicitly; raw inputs remain diagnostic rather than a third peer causal claim.
- Raw observed and centered-81-day F10.7 form matched **PCMCI F10.7 timing variants** crossed with both **PCMCI preprocessing profiles**; centered F10.7 never defines a primary daily causal lag.
- **PCMCI stationarity qualification** controls whether a preprocessing profile can support causal interpretation without suppressing failed robustness computations.
- **PCMCI sensitivity agreement** separates within-method factor robustness from cross-method support across the full-window ParCorr and bounded nonlinear matrices.
- The **Frequency-analysis method** follows the target's sampling contract, allowing one interface to return comparable spectrum products without hiding which estimator was used.
- **Analysis capabilities** make invalid target, cadence, extension, and method combinations planning errors instead of runtime surprises.
- **Analysis result artifacts** separate scientific calculation from rendering, so figures and composites can be regenerated and tested without repeating upstream analysis.
- The **Study artifact graph** gives expanded runs shared, provenance-checked upstream work while keeping each analysis target and variant independently rerunnable.
- **Study execution identity** separates human navigation from deterministic checkpoint validity, allowing equivalent cases to reuse artifacts across executions without hiding provenance changes.
- An **Analysis selection and preset** makes computational and scientific scope visible while still allowing standard analysis bundles to be reused.
- **Access-conditional thesis reproduction** persists local products between stages so an interrupted run or expensive analysis can resume without repeating valid upstream work.
- A **Thesis result workflow** is the reader-facing unit of **Access-conditional thesis reproduction**, while source-specific acquisition and preparation steps remain independently runnable provenance operations.
- **Ionospheric F2-layer peak state** complements neutral-density and cooling diagnostics when studying thermosphere--ionosphere coupling; hmF2 represents peak position while NmF2 prevents a height change from being conflated with a change in peak electron density.
- Each pipeline run has exactly one **Analysis target**; selecting multiple density products or model-error families produces independent runs that may share the same driver and preprocessing selections. Descriptive outputs may retain all available altitudes, while causal-discovery runs use explicit **PCMCI altitude groups**.
- Empirical-model density and error targets are evaluated on a **Reference sampling frame** from TU Delft or HASDM. **Global mean thermospheric density** remains observation-only unless its sampling geometry is recovered.
- Multi-model studies derive a **Paired model comparison sample** from the reference frame before model evaluation, preserving fair JB2006/JB2008 and MSIS-family comparisons.
- The **Conditional empirical-model performance claim** uses those paired samples to qualify every reported model improvement by reference product, altitude, activity regime, and metric rather than producing a universal ranking.
- **Calibration-domain representativeness** distinguishes documented fitting-data overlap from genuine out-of-source evaluation before differences among empirical models are interpreted.
- The **Chronological empirical-model contrast set** retains within-family development context while limiting cross-family claims to adjacent generations and the latest JB/MSIS comparison.
- A **Model density source** may be executable or precomputed. Native-grid WACCM-X analysis does not require an observational reference, while WACCM-X model-error comparison does require alignment to a **Reference sampling frame**.
- F10.7 and F30 are alternative **Solar-proxy variants**, not simultaneous drivers; their matched runs support a direct sensitivity comparison.
- Ap and Kp are alternative **Geomagnetic-driver variants**, not simultaneous drivers; their matched runs support a direct sensitivity comparison.
- SABER and ionospheric inputs produce explicit extension runs matched to a **Core driver run**, so narrower product availability is visible rather than silently changing the core sample.
- Every optional extension has an **Extension overlap control** on identical rows as well as the full-window core run.
- A **SABER extension selection** may contain any species combination, but it preserves direct-cooling and emission-proxy products as distinct typed nodes.
- An **Ionospheric extension run** keeps hmF2 and NmF2 paired and rejects target/source combinations without compatible geographic support.
- Daily and 3-hour analyses are separate **Cadence variants**; unsupported cadence/source combinations fail during planning rather than being silently resampled.
- **Bounded gap interpolation** makes any filled target values visible and reproducible while preventing long gaps or edge gaps from being invented.
- A **Physical PCMCI lag window** preserves the same scientific lag meaning across **Cadence variants**, while each expanded run retains the exact converted step indices needed by the PCMCI implementation.

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
- Joint solar-and-geomagnetic binning was considered; deferred as low-priority future work. If revisited, use a two-dimensional solar-bin by geomagnetic-bin product with an explicit, accessible colour encoding rather than adding it to the initial pipeline interface.
- "Density trend" in the synthesis figure means **Direct calendar-time density trend** for the current thesis products, not a fully deconfounded literature-equivalent secular \(\mathrm{CO_2}\) trend.
- "Replicate the results" was ambiguous about distributing prepared data; resolved: use **Access-conditional thesis reproduction**, keep large datasets out of the repository, and persist ignored local checkpoints between pipeline stages.
- The proposed "dataset axis" mixed external sources, model-generated baselines, and model-error products; resolved: reader-facing execution mirrors **Thesis result workflows**, while acquisition remains split by external source dataset.
- "CO2" as an acronym entry was considered; resolved: CO₂ is a chemical formula, not an acronym — remove it from the acronym list and use a `\coo` macro (\ensuremath{\mathrm{CO_2}}) for consistent formatting instead of `\ac{CO2}`.
- Acronyms in formulas, subscripts, or section titles were considered; resolved: no `\ac{}` calls inside math environments or in `\section`/`\subsection`/`\subsubsection` titles — spell out or use the short form directly without the acronym macro.
- "SABER" as an acronym entry was considered; resolved: SABER is an instrument name, not a generic term — remove it from the acronym list, spell out "Sounding of the Atmosphere using Broadband Emission Radiometry (SABER)" once at first mention in the dataset and \(\mathrm{CO_2}\) cooling-rate explanation, then use plain "SABER" text onwards.
- "Electron content" was proposed to track how the ionosphere's position changes; resolved: use **Ionospheric F2-layer peak state** (hmF2 together with NmF2), because total electron content is column-integrated and does not locate the layer by itself.
