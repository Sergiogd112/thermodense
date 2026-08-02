# Figure-review workbench — PROTOTYPE (issue #15)

**Question being answered:** *What minimal local-first figure-review interface
should let one researcher rapidly browse and compare rendered figures, inspect
provenance and underlying analysis artifacts, mark outputs as
included/supplemented/revised/excluded, attach figure- or panel-level comments
separately as scientific interpretation, limitation, caption, or presentation
requests, and export a versioned machine-readable review manifest that
publication workflows can consume without rerunning analyses?*

This is **throwaway code**. It is not the workbench; it exists to settle the
interface question. Captured on branch `prototype/issue-15-figure-review`.

## Run

```sh
PYTHONPATH=src python -m thermodense.prototypes.figure_review
```

Opens `http://127.0.0.1:8124/?variant=A`. Regenerate the sample figures from
the committed real results (optional):

```sh
PYTHONPATH=src python -m thermodense.prototypes.figure_review.make_figures
```

## The three variants

Switch with the floating bottom bar or `?variant=A|B|C`:

- **A — Ledger (keyboard triage):** dense figure list + master detail. Keys
  `j`/`k` navigate, `i`/`s`/`r`/`e` decide, `x` clears, `n` jumps to the next
  unreviewed figure. Provenance and advisory checks inline.
- **B — Board (drag & drop):** kanban columns by decision; drag cards between
  columns; tick *compare* on two figures for a side-by-side modal.
- **C — Claim-first (manifest audit):** claim-evidence cards drive the review;
  live JSON manifest preview in the right rail; export-centric.

All variants share: figure/panel-level comments typed as
`scientific` / `limitation` / `caption` / `presentation`, the
`include` / `supplement` / `revise` / `exclude` decision set, claim-evidence
cards, SHA-256 integrity verification of each rendered figure against its
artifact, and AGU/Wiley and Copernicus journal profiles whose warnings are
**advisory only — never scientific approval**.

## Manifest

Export produces `review-manifest-<figureSetVersion>-<timestamp>.json`
(`manifestVersion 0.1-prototype`), import rehydrates it. Fields: figures
(decision, comments, claimCardIds, contentSha256), claims (verdict), profile.

## Deliberately out of scope for this prototype

Manuscript editing, generated prose, real-time collaboration, and production
infrastructure. Integrity/accessibility/limits checks warn; they do not decide.

## Files

- `make_figures.py` — throwaway generator; renders 3 figures from the committed
  real ParCorr results and writes `data.json` + `figures/*.png`.
- `index.html`, `review.css`, `review.js` — the three variants + switcher.
- `data.json`, `figures/` — sample review data (real-data-derived).
- `__main__.py` — localhost server (one command to run).
