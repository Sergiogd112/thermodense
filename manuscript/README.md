# First-paper manuscript

The manuscript is built around the six provenance-tracked figures in
`figures/`. From this directory, run:

```bash
make
```

This writes `thermodense-first-paper.pdf`. Figure regeneration requires the
prepared research artifacts and is intentionally separate from manuscript
compilation:

```bash
uv run python scripts/build_first_paper_figures.py \
  --research-root /path/to/prepared/thermodense \
  --output-dir manuscript/figures
```

Before submission, confirm the author affiliation, corresponding-author email,
target journal template, data/code archive DOI, funding statement, and final
provider access dates.
