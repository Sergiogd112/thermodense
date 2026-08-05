# Figure-review workbench

Status: maintained local-first implementation for issue #17. The accepted Board
direction comes from prototype decision #15; Ledger and Claim-first remain
historical design evidence on `prototype/issue-15-figure-review` and are not
production navigation.

## Product contract

The workbench reviews immutable, versioned figure sets without rerunning an
analysis. The Board has exactly five mutually exclusive decisions:
`unreviewed`, `include`, `appendix`, `revise`, and `exclude`.

Review state includes typed figure/panel comments, claim verdicts, intended
LaTeX print widths, journal profile, and publication artifact identity. It is
saved locally after every mutation and restored only for the exact matching
figure-set version. Versioned JSON export/import is the portable scientific
record; device calibration remains separate.

Preview PNG and publication PDF integrity are checked independently against the
figure-set SHA-256 declarations. Those checks and journal profiles are advisory:
neither constitutes scientific approval.

## Delivery

Run `uv run python -m thermodense.figure_review`. The no-cache static server
binds to `127.0.0.1:8124`. All browser assets are relative so the application
works both at localhost root and below a scoped reverse-proxy path. The tracked
user-systemd unit suppresses browser opening and remains suitable for tailnet-
only Tailscale Serve access.

## Verification

Pure JavaScript contract tests cover legacy manifest migration, round-trip
publication identity, figure-set and hash guards, physical-size calculations,
draft keys, and route-safe asset resolution. Browser smoke checks cover desktop
and phone Board movement, detail, comparison, and physical preview.
