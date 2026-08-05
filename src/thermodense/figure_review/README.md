# Figure-review workbench

The maintained local-first Board for reviewing versioned Thermodense figure
sets. Each figure is in exactly one decision column: `unreviewed`, `include`,
`appendix`, `revise`, or `exclude`.

## Run

```sh
uv run python -m thermodense.figure_review
```

The server binds only to `127.0.0.1:8124` and opens
`http://127.0.0.1:8124/`. Set `THERMODENSE_FIGURE_REVIEW_NO_OPEN=1` for a
background service. `THERMODENSE_FIGURE_REVIEW_PORT` selects a different local
port for testing. Relative assets support a scoped reverse-proxy route such as
`/figure-review/`; do not expose the server publicly.

Regenerate the committed sample figure set with:

```sh
uv run python -m thermodense.figure_review.make_figures
```

## Review behavior

- Desktop: drag cards or use the explicit decision selector.
- Mobile: swipe-snapped columns and touch-sized decision selectors.
- Detail: typed figure- and panel-level comments, claim verdicts, immutable
  provenance, independently verified preview/PDF hashes, and vector-PDF access.
- Compare: select exactly two cards for side-by-side desktop or stacked mobile
  review.
- Physical view: an unshrunk 21 × 29.7 cm A4 sheet with configurable LaTeX
  figure width. Device calibration is stored separately and invalidated when
  browser/display scale changes.
- Appearance: System, Light, and Dark.

Every scientific review mutation is saved in browser local storage under the
exact `figureSetVersion`. A different figure set never silently restores that
draft.

## Manifest contract

Exports use `manifestVersion: "1.0"` and retain the figure-set version, journal
profile, all decisions, typed comments, claim verdicts, print widths, preview
hashes, and publication PDF identity. Import rejects a different figure-set
version or artifact identity. Legacy `supplement` decisions migrate to
`appendix`; new exports emit only `appendix`.

Display calibration is device state and is deliberately excluded from the
scientific manifest. Journal and integrity checks are advisory and never imply
scientific approval.

## Tailnet service

```sh
systemctl --user link "$PWD/scripts/systemd/thermodense-figure-review.service"
systemctl --user enable --now thermodense-figure-review.service
tailscale serve --bg --yes --set-path /figure-review http://127.0.0.1:8124
```

The service remains localhost-only; Tailscale terminates HTTPS and applies
tailnet identity and ACLs.

## Tests

```sh
node --test tests/js/figure_review_core.test.mjs
cd tests/js && npm install
FIGURE_REVIEW_URL=http://127.0.0.1:8124/ npm run test:smoke
```

The Playwright smoke suite uses the system Chromium by default; override it with
`CHROMIUM_PATH` when needed.

## Out of scope

Real-time collaboration, manuscript editing, generated prose, public internet
exposure, server-side review storage, and causal-analysis reruns.
