# Paper-writing and figure-review guidance

Research date: 2026-08-02

## Purpose and scope

This note derives a reusable Thermodense writing and figure-review baseline from
official publisher and university guidance. It is not a substitute for the
instructions of the journal ultimately selected. Word limits, section order,
file formats, and submission assets must remain journal profiles layered on top
of this common baseline.

AGU/Wiley and Copernicus Publications are the most plausible first publisher
profiles for an atmospheric or space-science paper. Nature Portfolio and
Elsevier provide useful additional reproducibility, integrity, and production
checks. MIT, UNC, Wisconsin, Manchester, and Harvard provide writing-process
guidance rather than submission rules.

## Common writing baseline

### Begin with the claim and evidence, not prose

Before drafting, state the paper's main message in one or two sentences and map
each part of it to an analysis artifact, figure, table, or cited source. MIT's
Communication Lab treats a clear main message, results that answer the stated
question, and figures that can largely convey that message independently as
criteria for success [1]. UNC emphasizes that readers must be able to follow the
logic from evidence to conclusion and that conclusions must be defended by the
reported evidence [2].

For Thermodense this means maintaining a claim-evidence ledger. Every central
claim should identify:

- the supporting result artifact and figure/table;
- the population, location, altitude, cadence, and time window to which it
  applies;
- whether it is descriptive, predictive, or causal-discovery language;
- uncertainty, robustness checks, and known counter-evidence;
- the exact limitation that prevents a broader claim.

### Work figures-first

A practical order is: freeze the question and analysis contract, produce and
review the key figures/tables, write Results and Methods, then Introduction and
Discussion, and finish the abstract and title. MIT explicitly recommends
preliminary figures before writing and notes that many readers primarily inspect
the abstract, figures, and conclusions [1]. Harvard physics guidance likewise
recommends planning figure and panel arrangements before drafting the body [6].

This is a workflow recommendation, not a required manuscript order. A rough
abstract may still be useful early, but the submission abstract should be
rewritten after the evidence and conclusions are stable.

### Give sections distinct jobs

UNC's scientific-report guidance is written substantially for course and lab
reports. Its clear methods, evidence-linked results, and explicit limitations are
transferable writing principles, but not journal formatting requirements. The
following are Thermodense drafting conventions, subject to the target journal's
structure.

- **Introduction:** establish the broad problem, narrow to the unresolved gap,
  and state the question and scope. The Manchester Academic Phrasebank organizes
  introductions around the established territory, the niche, and occupation of
  that niche [5].
- **Methods:** explain enough of the data lineage, choices, preprocessing,
  algorithms, controls, and uncertainty treatment for an informed researcher to
  evaluate and reproduce the work. Rationale matters as well as procedure [1,2].
- **Results:** report the selected observations and quantitative patterns without
  silently broadening their scope. Do not duplicate all values already visible
  in a figure or table [2].
- **Discussion:** connect findings to the question, prior work, limitations,
  plausible mechanisms, and implications. Explicitly explain how the evidence
  warrants each conclusion and account for conflicting or anomalous evidence
  [2].

This separation is a useful Thermodense drafting rule, not a universal publisher
format. Some journals permit or require combined Results and Discussion sections.

### Make the paper skimmable at three resolutions

The story should remain coherent when a reader sees only:

1. the title and abstract;
2. the figures and captions;
3. the first sentence of each paragraph.

MIT recommends that the abstract stand alone, the figures be self-explanatory,
and each paragraph begin with its message [1]. Titles should be concise,
informative, searchable, and free of avoidable abbreviations. Abstracts should
state the purpose, principal evidence, conclusion, and significance without
depending on figure references or undefined abbreviations; the target journal's
length limit still controls.

### Revise as a skeptical reader

Review in small units—claim map, figure set, outline, section, then manuscript—so
co-author feedback arrives before prose becomes expensive to reorganize [1]. For
each claim ask:

- Is this what the analysis actually estimated?
- Is the direction, magnitude, uncertainty, and sample support visible?
- Does a robustness result qualify or contradict it?
- Would the wording still be true if the figure were viewed without its preferred
  interpretation?
- Are words such as *causes*, *significant*, *global*, or *long-term* justified by
  the design and statistics?

## Thermodense figure-review baseline

This section combines publisher requirements, widely recommended practices, and
internal Thermodense controls. Only a selected journal profile can determine
submission acceptance; immutable artifacts and review metadata are project-level
controls rather than publisher mandates.

### Scientific content

- One defensible message per figure; multi-panel figures may support that message
  with distinct evidence.
- Every visible analytical subplot receives its own interpretation, except the
  documented sample-count support-panel exception and the limited
  correlation-scatter grouping exception defined in `CONTEXT.md`; the latter
  does not apply to TU Delft figures.
- Captions disclose the information needed to interpret the particular result,
  such as axes, units, transformations, sample basis, uncertainty, missingness,
  and quality filters where relevant.
- Comparisons use matched samples or clearly disclose differing coverage.
- Negative, null, and robustness results are not hidden when they constrain the
  main claim.
- The figure and caption can be understood without searching the main text for
  symbol or acronym definitions [1].

### Provenance and integrity

- The figure points to immutable analysis-result artifacts; it is never the only
  retained result or an upstream input.
- Input manifests, analysis fingerprint, renderer/configuration version, code
  commit, environment, and random seed where relevant are recorded.
- The data needed to validate a figure and the code used to generate it have a
  documented availability path. Nature Portfolio requires data-availability
  statements for original research and a code-availability section when custom
  code is central to the conclusions [7]. Copernicus encourages FAIR repository
  deposition and DOI citation for data, code, and interactive environments [8].
  AGU guidance likewise requires availability and citation of research data and
  software needed to evaluate the work, including figure-generating analysis
  code [9].
- For image-based figures, apply the selected publisher's integrity policy.
  Elsevier prohibits introducing, removing, moving, or obscuring image features;
  Nature requires minimal, disclosed processing [7,10]. Analytical
  transformations, filtering, and plotting choices are permitted only when
  scientifically justified, reproducible, and disclosed. Preserve source data
  and editable/vector figure inputs.
- Reused or adapted material records its source, licence, permission status, and
  required caption attribution [8].

### Accessibility and visual validity

- Colour is not the only carrier of meaning; lines also differ by symbol, pattern,
  or direct label where appropriate.
- Avoid rainbow scales and red-green contrasts; run a colour-vision-deficiency
  simulation and contrast check. Copernicus explicitly requires interpretable
  colour schemes for readers with colour-vision deficiencies [8].
- Text remains legible at final publication size; panel labels, symbols, and units
  are consistent.
- Provide concise alt text or an equivalent structured description for each
  figure, even if the eventual journal does not require it.
- Verify that rasterization, compression, and down-scaling do not alter the
  scientific reading of the figure.

### Caption and manuscript integration

- Store the caption separately from the image. Begin with a concise statement of
  what the figure shows, then define panels, encodings, uncertainty, sample basis,
  and abbreviations.
- Keep scientific interpretation in the Results/Discussion prose when the target
  journal expects a descriptive caption; support a journal profile if a different
  caption convention applies.
- The prose states the relevant result rather than merely saying that a figure
  exists. Figure order and placement follow the target-journal profile.
- Multi-panel labels are unique, ordered, and consistent between image, caption,
  and prose.

### Production profile

Technical requirements must be validated against the selected journal at the
time of submission. Common checks include accepted vector/raster formats,
effective resolution at final size, embedded fonts, physical width, file size,
RGB/CMYK mode, consolidated multi-panel files, and separate caption files.
Copernicus currently requests 300 dpi, at least 8 cm width, one file per composite
figure, embedded fonts for vectors, and captions in the text rather than figure
files [8]. These values must not be treated as universal.

## Figure-review workbench implications

The proposed workbench should be a local-first review layer over immutable
**Analysis result artifacts** and deterministic renderings. It must not edit
analysis outputs, infer scientific approval from technical checks, or make review
annotations inputs to later analyses.

### Proposed Thermodense workflow

The workflow and metadata below are project-level engineering and review
requirements derived from the repository's artifact model. They are not universal
publisher mandates.

1. Browse a compact figures-and-captions view, with filters for study, target,
   analysis, variant, and review state.
2. Compare selected renderings side-by-side while exposing differences in input
   coverage, configuration, and provenance.
3. Mark each figure `undecided`, `interesting`, `include`, `supplement`, `revise`,
   or `exclude`, with a required rationale for publication decisions.
4. Add comments at figure or panel level, explicitly typed as:
   `scientific-interpretation`, `claim-limitation`, `presentation-change`,
   `caption-change`, `integrity`, or `question`.
5. Review a claim-evidence card linking the proposed prose claim to the figure,
   underlying result, uncertainty, controls, and limitations.
6. Run mechanical checks for missing captions, undefined labels, incomplete
   provenance, colour accessibility, dimensions/resolution, permissions, and
   target-journal profile compliance.
7. Export a versioned machine-readable review manifest and human-readable report
   for the publication workflow.

### Recommended Thermodense review-manifest fields

- schema version, study execution ID, figure ID, and rendering fingerprint;
- source result-artifact IDs/checksums and input-manifest references;
- renderer commit, configuration fingerprint, environment, and seed;
- figure and panel labels, captions, alt text, and panel-level comments;
- candidate claim, claim class, scope, supporting evidence, uncertainty,
  robustness evidence, and limitations;
- analysis method and capability decision; descriptive, dependence-analysis, or
  causal-discovery status; sampling frame and coverage; preprocessing profile and
  physical lag window; controls/confounders; and explicit causal-eligibility or
  causal-exclusion rationale;
- review status, publication location, rationale, reviewer, and timestamps;
- integrity, accessibility, technical-production, and permission check states;
- target publisher/journal profile and profile version;
- append-only decision history, so changed judgements remain auditable.

Automatic checks should produce warnings, not scientific approval. Inclusion and
claim decisions remain human judgements.

## Thermodense recommendation

Use this common baseline immediately for figure review and manuscript planning.
Do not freeze publisher-specific dimensions or word limits until a target journal
is chosen. Given the subject, create an AGU/Wiley profile first, followed by a
Copernicus profile; add Nature Portfolio or Elsevier profiles only if a concrete
target journal makes them relevant.

The first prototype should test one high-value loop: compare related figures,
inspect their artifact provenance, make an inclusion decision, add separate
scientific and presentation comments, and export the review manifest. Manuscript
editing, collaborative real-time review, and automated prose generation should
remain outside that prototype.

## Official sources

1. MIT School of Engineering Communication Lab, [Journal Article](https://mitcommlab.mit.edu/cee/commkit/journal-article/) and [Introduction to Figure Design](https://mitcommlab.mit.edu/cee/commkit/figure-design/).
2. University of North Carolina Writing Center, [Scientific Reports](https://writingcenter.unc.edu/tips-and-tools/scientific-reports/) and [Sciences](https://writingcenter.unc.edu/tips-and-tools/sciences/).
3. University of Wisconsin–Madison Writing Center, [Formatting Science Reports](https://wric.wisc.edu/handbook/sciencereport/).
4. Harvard Catalyst Writing and Communication Center, [Getting Started](https://writingcenter.catalyst.harvard.edu/getting-started) and [Developing Your Manuscript](https://writingcenter.catalyst.harvard.edu/developing-your-manuscript).
5. University of Manchester, [Academic Phrasebank](https://www.phrasebank.manchester.ac.uk/).
6. Harvard University Department of Physics, [How to Write a Scientific Paper](https://hoffman.physics.harvard.edu/Hoffman-Example-Paper.pdf).
7. Nature Portfolio, [Reporting standards and availability of data, materials, code and protocols](https://www.nature.com/nature-portfolio/editorial-policies/reporting-standards), [Image integrity and standards](https://www.nature.com/nature-portfolio/editorial-policies/image-integrity), and [Nature research figure guide](https://research-figure-guide.nature.com/).
8. Copernicus Publications, [Manuscript preparation](https://publications.copernicus.org/for_authors/manuscript_preparation.html) and [Data policy](https://publications.copernicus.org/services/data_policy.html).
9. AGU Publications, [Text & Graphics Requirements](https://www.agu.org/publications/authors/journals/text-graphics-requirements), [Submission Checklists](https://www.agu.org/publications/authors/journals/submission-checklists), and [Data and Software for Authors](https://www.agu.org/publications/authors/journals/data-software-for-authors). AGU also preserves its data/software sharing guidance at [Zenodo](https://doi.org/10.5281/zenodo.5124741).
10. Elsevier, [Artwork and media instructions](https://www.elsevier.com/about/policies-and-standards/author/artwork-and-media-instructions), [Publishing ethics](https://www.elsevier.com/about/policies-and-standards/publishing-ethics), and [Research data](https://www.elsevier.com/about/policies-and-standards/research-data).

Access caveat: some AGU/Wiley pages returned HTTP 403 to automated retrieval on
2026-08-02. Their inclusion above is based on official indexed content and the
stable AGU Zenodo guidance; exact journal requirements must be rechecked manually
before submission.
