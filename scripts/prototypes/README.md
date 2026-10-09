# Prototypes

Exploratory, frozen or throwaway experiment implementations, grouped here on
2026-10-09 (previously `scripts/prototype_*.py`). Being here is not an adoption
decision; see [`docs/workflow-inventory.md`](../../docs/workflow-inventory.md).
Each preserves its own input/lag/scaling policy and output directory under
`outputs/prototypes/`; never infer resumability from a checkpoint directory name.
Historical provenance records may still cite the old `scripts/prototype_*` paths.

Run from the repository root, for example:

```console
uv run python -m scripts.prototypes.polynomial_var --help
```

## Direct-density PCMCI

- `density_pcmci_3hour_and_daily` — portable direct-density PCMCI+/analytic-ParCorr
  runner; staged by the Archmini/Kaggle orchestrators.

## Empirical-model Figure 5

- `empirical_model_figure5` — Figure 5 empirical-model comparison layouts.
- `empirical_model_figure5_3hour` — resumable 3-hour residual-dependence pipeline.

## Polynomial VAR

- `polynomial_var` — matrix-free polynomial VAR feasibility prototype.
- `polynomial_var_heatmaps` — lag-aggregated heatmaps for its matrices.
- `polynomial_var_a4_heatmaps` — A4 heatmap-layout variants.

## SABER

- `saber_vertical_totals` — vertically integrated products in the HASDM cell.
- `saber_vertical_totals_expanded_cells` — coverage feasibility in expanded cells.
- `saber_vertical_totals_40x60_stream` — exact, resumable configurable-cell stream.
- `saber_altitude_spatial_binning` — native CO₂ altitude strata and wider cells.
- `saber_pcmci_completion_pilot` — isolated PCMCI+ pilot for five-channel totals.
