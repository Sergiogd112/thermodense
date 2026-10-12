# Thermodense

Thermodense is an evolving, code-first research project for benchmarking
thermospheric density observations and empirical atmosphere models. Its scientific
priorities are long-term density variability, model-observation error, solar and
geomagnetic drivers, and the possible influence of increasing CO₂.

The public workflow interface is intentionally preparatory: workflow stages are
currently unavailable while the thesis implementations are migrated and checked
for numerical and visual parity. Use planning and status commands rather than
treating an unavailable workflow as a reproduced result. The migration roadmap
is maintained in the [Wayfinder map](https://github.com/Sergiogd112/thermodense/issues/2).

## Post-thesis priorities

In order: a code-first public release; paired JB2006/JB2008 baselines with
updated sources; raw F10.7; then, subject to access and migration gates, full
SABER coverage, F30, 3-hour sampling, and hmF2+NmF2. ParCorr is the only
conditional-independence test used before cluster access. Omitting 3-year
detrending is a robustness check, not a replacement analysis.

The selected first-paper work is a concise, figures-first set of six candidates
(five or six after artifact review); its secular density trend comparison
(Figure 6) is the organizing result. See
[`docs/first-paper-figure-plan.md`](docs/first-paper-figure-plan.md).

The current bounded paper-scope draft is the
[one-page CO₂ paper protocol](docs/co2-paper-protocol.md). Its linked Karman
closeout and data-qualification report distinguish completed results from
unresolved uncertainty/sampling gates. This draft does not silently supersede
registered roadmap decisions or authorize new runs. The manuscript/Overleaf
project remains in a separate repository.
The [6 October execution receipt](docs/research/co2-paper-execution-2026-10-06.md)
records the qualified pilot, actual source-data stop and independent inference gates.
The [SABER provider query](docs/research/saber-clock-provider-clarification-2026-10-06.md)
records the official-source check and an unsent clarification draft.
The [internal regional-impact audit](docs/research/saber-internal-regional-impact-audit-2026-10-06.md)
shows that the observed duplicate-clock anomalies are outside the fixed region;
the user subsequently approved region-scoped validation. The
[regional extraction closeout](docs/research/saber-regional-extraction-completion-2026-10-06.md)
records the completed descriptive output and remaining sampling gates.

## Install and inspect

```console
uv sync
uv run pytest
uv run thermodense plan global-mean
uv run thermodense status
```

The retained scripts in [`scripts/`](scripts/README.md) are transitional,
checkout-relative scientific implementations. They are not a claim that a full
current workflow or its generated results is committed to this repository.

## Data and access

Data are external and, in some cases, access-conditional. Downloads and derived
products belong in ignored `data/`; for example, the space-weather downloader
prepares `data/original/space_weather/SW-All.csv` from CelesTrak's `SW-All.txt`.
Obtain source data under each provider's terms before running an analysis. See
[data-access.md](docs/data-access.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

No third-party raw datasets or publisher PDFs are tracked here. The exact
Bachelor's thesis snapshot, including its generated outputs and presentation
materials, is privately archived pending a public UPC deposit link.

## Repository layout

```text
src/thermodense/       Package, CLI, downloaders, and migration foundations
scripts/               Transitional scientific analysis modules
tests/                 Automated tests
configs/               Checked-in workflow configuration
docs/                  Architecture decisions and access documentation
data/                  Ignored external inputs and generated products
```

`configs/thesis/` contains first-party frozen migration configurations, not the
removed thesis document or generated thesis artifacts. Its interface remains
unchanged until the pipeline decision in issue #7.

## License

[MIT](LICENSE.txt) applies to first-party code and documentation only. It does
not license third-party software, data, provider APIs, or publications; consult
the notices and source terms above.
