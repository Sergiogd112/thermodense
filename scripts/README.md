# Scientific script catalog

These checkout-relative modules are the existing scientific implementations
being migrated incrementally behind the `thermodense` workflow interface. Run
them from the repository root with:

```console
uv run python -m scripts.<module> [options]
```

## Acquisition and preparation

- `decode_hasdm` — decode HASDM archives into merged Parquet products.
- `decode_saber` — prepare daily SABER cooling-rate profiles near Mauna Loa.
- `generate_causal_input_csvs` — export prepared inputs for causal-discovery runs.

## Retained analyses

- `global_mean` — global mean thermospheric density dependence figures.
- `tudelft_density_analysis` — observation-only TU Delft density analysis.
- `tudelft_model_error_analysis` — TU Delft model log-density-ratio error analysis.
- `causal_hasdm_saber_maunaloa` — Mauna Loa HASDM and SABER daily products and dependence diagnostics.
- `maunaloa_global_figures` — Mauna Loa HASDM and SABER result figures.
- `hasdm_msis_density_baseline_analysis` — Mauna Loa MSIS density baselines.
- `hasdm_msis_model_error_analysis` — HASDM model log-density-ratio error products and diagnostics.
- `hasdm_msis_residual_saber_analysis` — HASDM model-error and SABER validation diagnostics.
- `recreate_brown_density_trend_current` — direct calendar-time density trend synthesis.
- `generate_figure19_champ_goce` and `figure19_champ_goce_stats` — focused Figure 19 reproduction and statistics.

## Causal discovery and publication

- `tigramite_causal_global_mean` — PCMCI\(^+\) runs for the configured global mean, HASDM, or HASDM model-error targets.
- `plot_causal_composites` — publication composites for causal-discovery runs.
- `plot_hasdm_msis_residual_causal_composites` — HASDM model-error causal-result composites.

`pgf_config` and `stats_utils` are shared implementation modules, not standalone
commands. Unmigrated scripts may still combine more than one canonical pipeline
stage; their stage separation happens only after numerical and visual parity
checks are in place.
