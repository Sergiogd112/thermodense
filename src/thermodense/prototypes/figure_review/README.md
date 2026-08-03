# Figure-review workbench — PROTOTYPE (issue #15)

**Question being answered:** *What minimal local-first figure-review interface
should let one researcher rapidly browse and compare rendered figures, inspect
provenance and underlying analysis artifacts, mark outputs as
included/appendix/revised/excluded, attach figure- or panel-level comments
separately as scientific interpretation, limitation, caption, or presentation
requests, and export a versioned machine-readable review manifest that
publication workflows can consume without rerunning analyses?*

This is **throwaway code**. It is not the workbench; it exists to settle the
interface question. Captured on branch `prototype/issue-15-figure-review`.

## Run

```sh
PYTHONPATH=src python -m thermodense.prototypes.figure_review
```

Opens `http://127.0.0.1:8124/?variant=B`. Regenerate the sample figures from
the committed real results (optional):

```sh
PYTHONPATH=src python -m thermodense.prototypes.figure_review.make_figures
```

## Tailnet access

This workstation serves the prototype tailnet-only at:

`https://omarchy.taileb3aa0.ts.net/figure-review/`

The node's root URL remains reserved for t3code. Tailscale Serve provides one
MagicDNS hostname for the node, so a scoped `/figure-review/` route avoids
overwriting `/` without requiring unsupported ad-hoc subdomains. This is not a
public Funnel; normal tailnet identity and ACLs apply.

The localhost process runs through the tracked user-systemd unit:

```sh
systemctl --user link "$PWD/scripts/systemd/thermodense-figure-review.service"
systemctl --user enable --now thermodense-figure-review.service
tailscale serve --bg --yes --set-path /figure-review http://127.0.0.1:8124
```

The server remains bound to `127.0.0.1`; Tailscale terminates HTTPS and proxies
the scoped route. `THERMODENSE_FIGURE_REVIEW_NO_OPEN=1` prevents background
service restarts from opening a browser.

## Mobile board

At phone widths, variant B presents one decision column at a time. Swipe left or
right between columns and use each card's **Move to…** selector instead of drag
and drop. Cards use touch-sized controls, comparison/detail opens full-screen,
and the calibrated A4 preview remains physically sized and scrollable rather
than shrinking to the phone.

## The three variants

Switch with the floating bottom bar or `?variant=A|B|C`:

**Current choice: B — Board.** It is now the default while A and C remain in the
throwaway prototype for comparison.

- **A — Ledger (keyboard triage):** dense figure list + master detail. Keys
  `j`/`k` navigate, `i`/`s`/`r`/`e` decide, `x` clears, `n` jumps to the next
  unreviewed figure. Provenance and advisory checks inline.
- **B — Board (drag & drop):** kanban columns by decision; drag cards between
  columns; tick *compare* on two figures for a side-by-side modal.
- **C — Claim-first (manifest audit):** claim-evidence cards drive the review;
  live JSON manifest preview in the right rail; export-centric.

All variants share: figure/panel-level comments typed as
`scientific` / `limitation` / `caption` / `presentation`, the
`include` / `appendix` / `revise` / `exclude` decision set, claim-evidence
cards, SHA-256 integrity verification of each preview and publication artifact,
a System / Light / Dark appearance control, and AGU/Wiley and Copernicus journal
profiles whose warnings are **advisory only — never scientific approval**.

## Physical print-size preview

Choose **View: Physical print size**, select a figure, and enter the same width
used by LaTeX (for example `5 cm`). The preview shows a complete calibrated A4
sheet (21 × 29.7 cm) with 1.6 cm side margins, 2 cm top/bottom margins, and the
centred figure plus a 10 pt caption at that imposed physical size. Width presets
are provided for 5, 8.5, and 17.8 cm; these are conveniences, not journal rules.

CSS centimetres are not reliably physical across monitors. Open **Calibrate
physical scale**, hold a ruler against the 10 cm line, and adjust until its end
marks are exactly 10 cm apart. Calibration is stored locally with the current
browser device-pixel ratio and is not part of the scientific manifest. The old
scale is disabled when browser zoom or display scale changes; moving to another
monitor may still require manual recalibration.

## LaTeX figure output

The generator writes two representations from the same Matplotlib figure:

- `figures/*.pdf` — the publication artifact for `\includegraphics`; vector
  graphics with embedded TrueType fonts (`pdf.fonttype = 42`).
- `figures/*.png` — a 150 dpi browser preview only, not the submission artifact.

The workbench offers the PDF as a download and verifies the SHA-256 of both
representations independently. Example:

```tex
\usepackage{graphicx}
\includegraphics[width=5cm]{figures/figure-01-graph.pdf}
```

## Manifest

Export produces `review-manifest-<figureSetVersion>-<timestamp>.json`
(`manifestVersion 0.4-prototype`), import rehydrates it. Legacy `supplement`
decisions import as `appendix`. Fields: figures
(decision, comments, claimCardIds, preview contentSha256, publication PDF path /
format / sha256, intended `printWidthCm`), claims (verdict), profile.

## Deliberately out of scope for this prototype

Manuscript editing, generated prose, real-time collaboration, and production
infrastructure. Integrity/accessibility/limits checks warn; they do not decide.

## Files

- `make_figures.py` — throwaway generator; renders 3 figures from the committed
  real ParCorr results and writes `data.json` + paired `figures/*.png` previews
  and `figures/*.pdf` LaTeX publication artifacts.
- `index.html`, `review.css`, `review.js` — the three variants + switcher.
- `data.json`, `figures/` — sample review data (real-data-derived).
- `__main__.py` — no-cache localhost server (one command to run).
- `scripts/systemd/thermodense-figure-review.service` — persistent user service.
