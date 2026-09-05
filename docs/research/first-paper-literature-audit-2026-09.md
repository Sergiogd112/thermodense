# First-paper literature audit through September 2026

Research date: 2026-09-05

## Scope and cutoff

This audit supports the journal-neutral **Thesis-continuation paper**. It reviews
the current manuscript and bibliography against primary peer-reviewed papers and
first-party model or data documentation for:

- thermospheric density variability and secular contraction;
- solar, geomagnetic, tropospheric-CO2, and SABER cooling relationships;
- the Global mean, HASDM, TU Delft, NRLMSISE-00, NRLMSIS 2.0/2.1,
  JB2006, and JB2008 products;
- paired empirical-model comparison and calibration overlap;
- autocorrelation-aware trend and uncertainty methods; and
- PCMCI+ assumptions and causal-claim limits.

The requested cutoff is 30 September 2026, but that date is after the research
date. This note is current through 5 September 2026. The literature ticket must
receive a final delta search at the end of September if the first draft is still
open then.

## Immediate manuscript blockers

1. The bibliography is too sparse for the current Introduction and Discussion.
   It omits several foundational trend studies, solar-activity sensitivities,
   SABER sampling limitations, and recent upper-atmosphere climate syntheses.
2. The `emmert2015` BibTeX title is wrong. The verified title is
   *Altitude and solar activity dependence of 1967–2005 thermospheric density
   trends derived from orbital drag*; the paper also examines 1967–2013 trends
   but that is not its title
   ([doi:10.1002/2015JA021047](https://doi.org/10.1002/2015JA021047)).
3. Exact project dates, row counts, altitude grids, common samples, correlations,
   errors, and trends require result-artifact provenance. Literature citations
   can establish a parent product or method, but cannot validate those local
   quantities.
4. The manuscript needs direct sources for its calibration-overlap statements,
   particularly the NRLMSIS 2.0 CHAMP/GOCE comparison role and JB2008 use of
   HASDM-, CHAMP-, and GRACE-domain densities.
5. The paper must distinguish CO2 concentration, radiative cooling rate, and
   outgoing radiative power. They are related but not interchangeable.
6. The 27-day HAC bandwidth and bootstrap block length are preregistered project
   choices, not values proven uniquely correct by Newey--West or Künsch.
7. The exact TU Delft version-02 citation and the exact processed SABER product
   endpoint remain unresolved provider-metadata gaps.

## Citation-ready source inventory

### Thermospheric contraction and trend context

| Source | What it supports | Required qualification | Action |
| --- | --- | --- | --- |
| [Emmert et al. (2004), doi:10.1029/2003JA010176](https://doi.org/10.1029/2003JA010176) | Orbit-derived evidence of secular density decrease over 200--700 km. | Limited object set and historical interval; not one universal trend magnitude. | Add |
| [Emmert (2009), doi:10.1029/2009JA014102](https://doi.org/10.1029/2009JA014102) | Method and provenance for the parent globally averaged orbit-derived density product. | Supports the published 1967--2007 product, not automatically the local 1967--2019 artifact. | Keep and qualify |
| [Emmert and Picone (2011), doi:10.1029/2010JA016382](https://doi.org/10.1029/2010JA016382) | Domain-specific statistical uncertainty of orbit-derived density trends. | Does not validate the project's exact HAC bandwidth. | Add |
| [Emmert (2015), doi:10.1002/2015JA021047](https://doi.org/10.1002/2015JA021047) | A 1967--2005 trend of -2.0 +/- 0.5% per decade at 400 km and increasingly negative trends from 250 to 575 km; solar-flux dependence is weak relative to uncertainty. | Monotonic driver changes complicate separation from CO2 increase. | Correct title and keep |
| [Keating et al. (2000), doi:10.1029/2000GL003771](https://doi.org/10.1029/2000GL003771) | Early satellite-drag evidence of a comparatively large density decline. | Strong method and attribution assumptions; useful counter-evidence, not a consensus magnitude. | Add |
| [Marcos et al. (2005), doi:10.1029/2004GL021269](https://doi.org/10.1029/2004GL021269) | An alternative estimate near 400 km over 1970--2000. | Different model correction and period. | Add |
| [Saunders et al. (2011), doi:10.1029/2010JA016358](https://doi.org/10.1029/2010JA016358) | Independent ballistic-coefficient method for long-term density change. | Methodological differences must remain visible. | Add |
| [Oliver et al. (2014), doi:10.1002/2014JA020311](https://doi.org/10.1002/2014JA020311) | Local radar evidence of altitude-dependent temperature and density behavior. | Local site and different observables; counters a simple altitude-independent narrative. | Add as counter-evidence |
| [Brown et al. (2021), doi:10.1029/2021JD034589](https://doi.org/10.1029/2021JD034589) | WACCM-X historical and scenario-based density changes under controlled forcing. | Model/scenario evidence, not an observed mixed-forcing record. | Add |
| [Brown et al. (2024), doi:10.1029/2024JA032659](https://doi.org/10.1029/2024JA032659) | Solar-condition dependence and the literature-profile source used in Figure 6. | Digitized Figure 2 geometry is not replacement study data; do not infer altitude curves from 400-km tables. | Keep and state source role |
| [Weng et al. (2020), doi:10.1029/2020GL087140](https://doi.org/10.1029/2020GL087140) | A machine-learning route to long-term thermospheric density trends. | A distinct estimator, useful for method-dependent comparison. | Add |
| [Lastovicka et al. (2026), doi:10.1371/journal.pclm.0000836](https://doi.org/10.1371/journal.pclm.0000836) | Current whole-system synthesis: CO2 is the main broad driver, with ozone, solar variability, and secular magnetic-field change also relevant; it summarizes density decline near -2% per decade. | The publisher labels it **Opinion**; use as current synthesis, not primary quantitative evidence. | Add as context |
| [Cnossen et al. (2024), doi:10.1016/j.asr.2023.09.043](https://doi.org/10.1016/j.asr.2023.09.043) | Recent review of global long-term middle- and upper-atmosphere changes for empirical-model context. | Review source; trace central quantitative claims to original studies. | Add as synthesis |

### Solar, geomagnetic, CO2, and SABER evidence

| Source | What it supports | Required qualification | Action |
| --- | --- | --- | --- |
| [Roble and Dickinson (1989), doi:10.1029/GL016i012p01441](https://doi.org/10.1029/GL016i012p01441) | Modeled mechanism by which increasing greenhouse gases cool and alter the upper atmosphere. | Theoretical/model evidence, not direct attribution of the project trends. | Keep and narrow wording |
| [Qian et al. (2006), doi:10.1029/2006GL027185](https://doi.org/10.1029/2006GL027185) | Solar-condition dependence of modeled and observed thermospheric climate change. | Different interval and model design from the current study. | Add |
| [Qian et al. (2008), doi:10.1016/j.asr.2007.10.019](https://doi.org/10.1016/j.asr.2007.10.019) | Thermospheric neutral-density response to solar forcing. | Supports the forcing hierarchy, not the project's exact correlations. | Add |
| [Xu et al. (2015), doi:10.1002/2014JA020830](https://doi.org/10.1002/2014JA020830) | Multiday density oscillations associated with solar and geomagnetic variability. | Context for timescales and daily range, not CO2 attribution. | Add |
| [Solomon et al. (2015), doi:10.1002/2014JA020886](https://doi.org/10.1002/2014JA020886) | Three-dimensional simulations with trend magnitude dependent on solar activity and NO cooling. | Model result, not direct observation. | Add |
| [Solomon et al. (2019), doi:10.1029/2019JA026678](https://doi.org/10.1029/2019JA026678) | Whole-atmosphere climate-change response depends on solar conditions; NO cooling modifies high-activity response. | Ensemble/model limitations apply. | Add |
| [Mlynczak et al. (2010), doi:10.1029/2009JA014713](https://doi.org/10.1029/2009JA014713) | SABER observations of infrared cooling on daily to multiyear timescales. | Does not establish the local processed endpoint or a causal HASDM relationship. | Keep; complete authors |
| [Rezac et al. (2018), doi:10.1029/2018JA025892](https://doi.org/10.1029/2018JA025892) | Time-varying gaps and nonuniform local-time sampling can bias SABER CO2 trends; 60-day averaging reduced relative-trend bias in their synthetic study. | Concerns CO2 trend estimation; apply cautiously to the paper's cooling product. | Add |
| [Mlynczak et al. (2024), doi:10.1029/2024GL109757](https://doi.org/10.1029/2024GL109757) | Over 20 years, trends in exiting longwave radiation from 65--105 km were not significantly different from zero at 95% or 99% confidence despite cooling and contraction. | Radiative power is not cooling rate or CO2 concentration. | Add as counter-evidence |
| [Cnossen (2020), doi:10.1029/2020JA028623](https://doi.org/10.1029/2020JA028623) | A transient WACCM-X study found standard regression insufficiently removed solar-cycle effects; adding squared averaged F10.7 improved results. It also found CO2 dominant globally with magnetic-field effects at high magnetic latitudes. | Model-based attribution; supports the quadratic-control rationale, not the exact local trend. | Add |

### Density products and empirical models

| Source | What it supports | Required qualification | Action |
| --- | --- | --- | --- |
| [Picone et al. (2002), doi:10.1029/2002JA009430](https://doi.org/10.1029/2002JA009430) | NRLMSISE-00 formulation, calibration sources, and comparison context. | Use the canonical model name. | Keep |
| [Emmert et al. (2021), doi:10.1029/2020EA001321](https://doi.org/10.1029/2020EA001321) | NRLMSIS 2.0 whole-atmosphere formulation and mass-density model. | Cite directly for fit/comparison-domain statements. | Keep; complete authors |
| [NASA CCMC NRLMSIS 2.0](https://ccmc.gsfc.nasa.gov/models/NRLMSIS~2.0/) | First-party inputs, outputs, 0--1000 km domain, and model-generated status. | The page update date is not the model publication date. | Add as provider documentation |
| [Emmert et al. (2022), doi:10.1029/2022JA030896](https://doi.org/10.1029/2022JA030896) | NRLMSIS 2.1 NO extension. | Do not describe it as an independently refitted base mass-density model. | Keep; complete authors |
| [NASA CCMC NRLMSIS 2.1](https://ccmc.gsfc.nasa.gov/models/NRLMSIS~2.1/) | First-party documentation states that the 2.1 upgrade consists solely of adding NO and lists its six NO instruments. | Provider page, not a replacement for the model paper. | Add |
| [Bowman et al. (2008), JB2006, doi:10.1016/j.jastp.2007.10.002](https://doi.org/10.1016/j.jastp.2007.10.002) | JB2006 formulation and provenance. | Does not alone prove independence from HASDM. | Keep |
| [Bowman et al. (2008), JB2008, doi:10.2514/6.2008-6438](https://doi.org/10.2514/6.2008-6438) | JB2008 and its solar and geomagnetic proxy system. | Conference/event metadata vary; record the AIAA paper number and DOI. | Keep |
| [Storz et al. (2005), doi:10.1016/j.asr.2004.02.020](https://doi.org/10.1016/j.asr.2004.02.020) | HASDM drag-assimilation/model provenance. | Not documentation for the exact current database release or local extraction. | Keep |
| [Licata et al. (2022), doi:10.1029/2021SW002915](https://doi.org/10.1029/2021SW002915) | HASDM-derived modeling and uncertainty context. | Does not provide uncertainty for this exact subset. | Add |
| [Guo et al. (2024), doi:10.1016/j.asr.2024.05.063](https://doi.org/10.1016/j.asr.2024.05.063) | Recent NRLMSIS 2.1 validation against GRACE-A and Swarm-C. | Mission-, altitude-, and epoch-specific; no universal ranking. | Add |
| [Siemes et al. (2023), doi:10.1051/swsc/2023014](https://doi.org/10.1051/swsc/2023014) | Related accelerometer-derived CHAMP/GRACE/GRACE-FO density lineage. | Not automatically the same as the eight-mission TU Delft version-02 product. | Add only as related provenance |
| [van den IJssel et al. (2020), doi:10.1016/j.asr.2020.01.004](https://doi.org/10.1016/j.asr.2020.01.004) | Swarm-derived density estimation method. | Does not document all TU Delft missions. | Add if Swarm methods are discussed |

The Space Environment Technologies HASDM page URL recorded in earlier research
returned HTTP 404 on 2026-09-05. Its grid facts must be recovered from a current
first-party page or archived provider documentation before citation.

### Statistical and causal-discovery methods

| Source | What it supports | Required qualification | Action |
| --- | --- | --- | --- |
| [Newey and West (1987), doi:10.2307/1913610](https://doi.org/10.2307/1913610) | HAC covariance estimation. | Does not select a 27-day bandwidth for this study. | Keep |
| [Künsch (1989), doi:10.1214/aos/1176347265](https://doi.org/10.1214/aos/1176347265) | Bootstrap methods for dependent stationary observations. | Does not by itself validate the exact circular-calendar-day implementation or simultaneous contrasts. | Keep |
| [Benjamini and Hochberg (1995), doi:10.1111/j.2517-6161.1995.tb02031.x](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x) | False-discovery-rate control. | State the exact family over which correction is applied. | Keep |
| [Runge (2020)](https://proceedings.mlr.press/v124/runge20a.html) | PCMCI+ for lagged and contemporaneous causal discovery in autocorrelated time series. | Reliability claims require assumptions including causal sufficiency; software output does not itself license causal language. | Keep and expand limitations |
| [Runge et al. (2019), doi:10.5194/esd-10-843-2019](https://doi.org/10.5194/esd-10-843-2019) | Causal-inference assumptions and constraints in Earth-system time series. | Use to frame, not erase, project-specific graph and sampling failures. | Add |
| [Runge et al. (2019), doi:10.1126/sciadv.aau4996](https://doi.org/10.1126/sciadv.aau4996) | PCMCI framework for high-dimensional nonlinear time series. | Does not resolve altitude, mission, local-time, selection, or prescribed-model-input confounding. | Add |
| [Tigramite upstream](https://github.com/jakobrunge/tigramite) | Software implementation and version provenance. | Not evidence that the scientific graph is causally valid. | Add to software statement |

## Current bibliography audit

| Key | Finding | Action |
| --- | --- | --- |
| `benjamini1995` | Core metadata consistent with the DOI. | Keep |
| `bowman2008jb2006` | Core metadata consistent. | Keep |
| `bowman2008jb2008` | Core metadata consistent; clarify AIAA event/publication metadata. | Keep/correct |
| `brown2024` | Core metadata consistent. | Keep |
| `emmert2009` | Core metadata consistent. | Keep |
| `emmert2015` | Title incorrectly says 1967--2013. | Correct to 1967--2005 |
| `emmert2021` | Core metadata consistent; `and others` hides named authors. | Complete authors or adopt a consistent bibliography policy |
| `emmert2022` | Core metadata consistent; `and others` hides named authors. | Complete authors or adopt a consistent bibliography policy |
| `kunsch1989` | Core metadata consistent. | Keep |
| `matzka2021` | Core metadata consistent. | Keep and add the provider for the exact Ap series |
| `mlynczak2010` | Core metadata consistent; author list is truncated manually. | Complete authors |
| `neweywest1987` | Core metadata consistent. | Keep |
| `noaaco2` | `2026` is an access year, not a conventional publication year. | Use online/no-date metadata plus exact access date and data-version notes |
| `picone2002` | Core metadata consistent. | Keep |
| `roble1989` | Core metadata consistent. | Keep and narrow attribution wording |
| `runge2020` | PMLR metadata and canonical URL are appropriate. | Keep |
| `storz2005` | Core metadata consistent. | Keep; standardize author names |
| `tapping2013` | Core metadata consistent. | Keep |
| `tudelftdata` | Exact formal metadata and publication date were not recovered. | Verify before final bibliography |

## Manuscript assertions needing direct support or rewriting

- Cite a density/drag source for the operational opening statement.
- Cite primary solar and geomagnetic response studies for the forcing hierarchy.
- Support the global-mean loss of geographic/local-time detail from its method,
  not from assertion alone.
- Support mission, altitude, local-time, and epoch mixing in trajectory-derived
  products with the exact TU Delft processing lineage.
- Keep the local global-mean extension, SABER endpoint, HASDM extraction, common
  sample, and all numerical ranges tied to machine-readable project provenance.
- Cite the exact Ap source and aggregation, not Kp literature alone.
- State that Fisher intervals are nominal under serial dependence and cite an
  appropriate time-series treatment; do not imply daily rows are independent.
- Cite direct fitting/comparison documentation for NRLMSIS and JB calibration
  overlap. Treat unknown identity between development data and evaluated rows as
  unknown, not as established overlap or independence.
- Label the 27-day HAC and bootstrap settings as preregistered analysis choices;
  retain 54- and 81-day sensitivities required by the inherited contract.
- Expand PCMCI+ limitations to causal sufficiency, stationarity qualification,
  graph specification, missingness, selection, multiplicity, and method
  sensitivity.
- Cite HASDM's assimilative nature where the Discussion explains why it is not
  ground truth.
- Qualify “broadly consistent with historical literature” by showing the wide
  method-, period-, altitude-, and solar-condition-dependent range.

## Counter-evidence that must remain visible

- Historical trend magnitudes differ materially across Keating, Emmert, Marcos,
  Saunders, Weng, Brown, and the local estimates.
- Qian, Solomon, Cnossen, and Brown show that solar condition and solar-control
  design can materially change estimated long-term response.
- Rezac et al. show that SABER's nonuniform sampling can bias trends.
- Oliver et al. show local and altitude-dependent behavior that resists a simple
  global contraction narrative.
- Mlynczak et al. (2024) distinguish a statistically null long-term trend in
  exiting radiative power from cooling and contraction.
- Cnossen and Lastovicka et al. identify geomagnetic-field, ozone, and other
  contributions alongside CO2.
- HASDM is assimilative, and empirical-model fitting domains and prescribed
  proxies differ; paired samples improve comparability but do not create an
  independent truth reference.
- PCMCI+ conclusions remain conditional on the graph and method assumptions;
  nonlinear tests do not repair sampling or causal-identification failures.

## Unresolved searches

- Exact citation and formal metadata for the eight-mission TU Delft version-02
  landing product.
- Current first-party SET HASDM and JB documentation URLs replacing the observed
  404 page.
- A first-party source for the exact processed SABER cooling product version and
  2002--2023 endpoint.
- A source-specific justification, if one exists, for 27-day uncertainty blocks;
  none was found, so the project must own and sensitivity-test this choice.
- Direct textual evidence for every calibration-overlap category in the final
  model/reference table.
- Any papers published from 6--30 September 2026; run a final delta search if the
  ticket remains open at month end.

## Recommended bibliography workflow

1. Correct existing records before adding new prose.
2. Add foundational trend and solar-sensitivity sources, then the recent
   synthesis and counter-evidence.
3. Add first-party model/data/software documentation separately from
   peer-reviewed method papers.
4. Build a claim-to-citation ledger that distinguishes external literature facts
   from local artifact facts.
5. Verify each added BibTeX record against its DOI or publisher page and compile
   the manuscript without warnings.
6. Repeat the September 2026 delta search at the actual cutoff date when needed.
