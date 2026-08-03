# Figure-review workbench prototype (issue #15)

Status: **prototype for decision**, captured on branch
`prototype/issue-15-figure-review`.

Current layout decision: **B — Board**, selected as the working direction while
the prototype remains open to refinement.

## Question

See
[issue #15](https://github.com/Sergiogd112/thermodense/issues/15):
what minimal local-first figure-review interface should let one researcher
rapidly browse and compare rendered figures, inspect provenance and underlying
analysis artifacts, mark outputs as included/supplemented/revised/excluded,
attach figure- or panel-level comments separately as scientific interpretation,
limitation, caption, or presentation requests, and export a versioned
machine-readable review manifest that publication workflows can consume without
rerunning analyses?

## Shape chosen

Web UI prototype with three structurally different variants (A ledger, B board,
C claim-first), because the core of the question is *browsing and comparing
rendered figures* — a visual judgment a terminal TUI cannot surface. The repo
has no web stack, so the prototype is a self-contained static bundle served by
one command:

```sh
PYTHONPATH=src python -m thermodense.prototypes.figure_review
```

## What the prototype covers

- Browse / compare / inspect: figure set derived from the committed real ParCorr
  results (`benchmarks/pcmci-methods/results/real-1/local/`), provenance shown
  per figure (source artifacts + sha256), in-browser SHA-256 verification of
  both the PNG preview and vector PDF publication artifact.
- Publication output: paired PNG browser previews and LaTeX-friendly vector PDF
  figures with embedded TrueType fonts; PDFs are downloadable from figure detail.
- Appearance: follows the operating-system preference by default, with explicit
  Light and Dark overrides shared by all three variants.
- Mobile board: one swipe-snapped decision column at a time, explicit card
  decision selectors as the touch alternative to drag-and-drop, and full-screen
  detail/compare surfaces.
- Tailnet delivery: the localhost-only server runs as an enabled user service
  and is proxied tailnet-only at `/figure-review/` on the workstation's existing
  Tailscale HTTPS hostname; the root route remains assigned to t3code.
- Physical print-size view: imposes each figure at its intended LaTeX width on a
  complete, physically scaled A4 sheet (21 × 29.7 cm) with visible paper margins.
  A ruler-based 10 cm calibration compensates
  for monitor density, operating-system scaling, and browser zoom; calibration
  stays device-local while the intended `printWidthCm` travels in the manifest.
- Decide: `include` / `supplement` / `revise` / `exclude` per figure.
- Comment: figure-level and panel-level, typed `scientific` / `limitation` /
  `caption` / `presentation`.
- Claim-evidence cards with verdicts.
- Export / import of a versioned JSON review manifest.
- Journal profiles (AGU/Wiley, Copernicus) with advisory limit warnings.
- Advisory-only framing: checks warn but never imply scientific approval.

## Open questions for the user

1. Which variant (or which mix, e.g. "header from B, list from A") feels right?
2. Is the manifest schema close to what a publication workflow should consume?
3. What should the real (non-throwaway) workbench keep vs drop?

## Capture

The winning layout folds into the real workbench; the losing variants and the
switcher move to the throwaway branch with this prototype (see
`src/thermodense/prototypes/figure_review/README.md`).
