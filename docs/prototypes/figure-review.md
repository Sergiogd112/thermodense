# Figure-review workbench prototype (issue #15)

Status: **prototype for decision**, captured on branch
`prototype/issue-15-figure-review`.

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
  per figure (source artifacts + sha256), in-browser SHA-256 verification.
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
