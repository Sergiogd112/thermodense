"""Create throwaway A4 heatmap-layout prototypes for the cubic polynomial VAR."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.colors import LogNorm, Normalize, SymLogNorm  # noqa: E402

if __package__:
    from scripts.plot_polynomial_var_heatmaps import (
        DEFAULT_MODEL,
        aggregate,
        display_name,
        mask_names,
        sha256,
    )
else:
    from plot_polynomial_var_heatmaps import (
        DEFAULT_MODEL,
        aggregate,
        display_name,
        mask_names,
        sha256,
    )

import matplotlib.pyplot as plt  # noqa: E402


A4_LANDSCAPE = (11.69, 8.27)
PNG_DPI = 600
DEFAULT_OUTPUT = DEFAULT_MODEL.parent / "heatmaps_a4_prototypes"
MATRIX_NAMES = ("W1", "W2", "W3")
LAG_TICKS = [0.0, 3.0 / 24.0, 1.0, 10.0, 30.5, 61.0, 91.5, 100.0, 183.0]
LAG_TICK_LABELS = [
    "0h",
    "3h",
    "1 day",
    "10 days",
    "1 month",
    "2 months",
    "3 months",
    "100 days",
    "6 months",
]


@dataclass(frozen=True)
class Metric:
    key: str
    title: str
    definition: str
    colorbar_label: str
    kind: str


METRICS = (
    Metric(
        "signed",
        "Signed coefficient sum",
        "sum of coefficients over the 1,464-lag axis; sign retained",
        "Σ lag coefficients (signed; symmetric log scale)",
        "signed",
    ),
    Metric(
        "rms",
        "Coefficient RMS",
        "sqrt(mean(coefficient² over the 1,464-lag axis))",
        "Lag-RMS coefficient magnitude (log scale)",
        "rms",
    ),
    Metric(
        "average_lag",
        "Magnitude-weighted average lag",
        "sum(abs(coefficient) × lag) / sum(abs(coefficient)), in days",
        "Magnitude-weighted average lag (piecewise linear)",
        "lag",
    ),
    Metric(
        "peak_signed",
        "Signed coefficient at peak magnitude",
        "signed coefficient at argmax(abs(coefficient)) over lag",
        "Coefficient at maximum |coefficient| (symmetric log scale)",
        "peak_signed",
    ),
    Metric(
        "peak_lag",
        "Lag of peak magnitude",
        "lag at argmax(abs(coefficient)), in days",
        "Lag of maximum |coefficient| (piecewise linear)",
        "lag",
    ),
)


def _load(model_path: Path) -> tuple[list[np.ndarray], list[str], list[str], np.ndarray]:
    with np.load(model_path, allow_pickle=False) as model:
        matrices = [model[name].astype(np.float64) for name in MATRIX_NAMES]
        outputs = model["channel_names"].astype(str)
        augmented = model["augmented_channel_names"].astype(str)
        mapping = model["mask_mapping"].astype(int)
        mask_count = int(model["mask_channel_count"])
        lag_order = model["lag_order"].astype(int)

    expected = (len(outputs), len(augmented), len(lag_order))
    if any(matrix.shape != expected for matrix in matrices):
        raise ValueError(f"coefficient shape does not match {expected}")
    if not all(np.isfinite(matrix).all() for matrix in matrices):
        raise FloatingPointError("coefficient matrices contain non-finite values")
    if not np.array_equal(lag_order, np.arange(1, len(lag_order) + 1)):
        raise ValueError("lag order is not contiguous lag-1 first")

    inputs = _concise_predictor_labels(outputs, mapping, mask_count)
    if len(inputs) != len(augmented):
        raise ValueError("derived input labels do not match augmented channels")
    return matrices, [display_name(name) for name in outputs], inputs, lag_order


def _concise_channel_name(name: str) -> str:
    if name.startswith("log10rho_"):
        return f"logρ {name.removeprefix('log10rho_').removesuffix('km')}"
    return {
        "F10.7_raw": "F10.7",
        "co2_mlo_ppm": "CO₂ MLO",
        "CO2cool": "CO₂ cool",
        "NOcool": "NO cool",
        "O2_1delta_ver": "O₂ 1Δ",
        "OH_16_ver": "OH 1.6",
        "OH_20_ver": "OH 2.0",
    }.get(name, display_name(name))


def _concise_mask_name(members: list[str]) -> str:
    member_set = set(members)
    if any(name.startswith("log10rho_") for name in members):
        return "avail density"
    if "co2_mlo_ppm" in member_set:
        return "avail CO₂ MLO"
    if "CO2cool" in member_set:
        return "avail CO₂ cool"
    if "NOcool" in member_set:
        return "avail NO cool"
    if any(name.startswith(("O2_", "OH_")) for name in members):
        return "avail O₂/OH"
    return "avail " + "/".join(_concise_channel_name(name) for name in members[:2])


def _concise_predictor_labels(
    channel_names: np.ndarray, mask_mapping: np.ndarray, mask_count: int
) -> list[str]:
    """Return print-safe predictor labels while deriving masks from their members."""
    # Retain the existing helper as the authoritative source for full mask labels.
    full_masks = mask_names(channel_names, mask_mapping, mask_count)
    concise_masks = []
    for mask_id, _ in enumerate(full_masks):
        members = [
            str(name)
            for name, mapped in zip(channel_names, mask_mapping, strict=True)
            if int(mapped) == mask_id
        ]
        concise_masks.append(_concise_mask_name(members))
    return [_concise_channel_name(str(name)) for name in channel_names] + concise_masks


def _norms(values: dict[str, tuple[np.ndarray, ...]], lag_order: np.ndarray) -> dict[str, object]:
    def positive(key: str) -> np.ndarray:
        data = np.concatenate([np.abs(item).ravel() for item in values[key]])
        return data[np.isfinite(data) & (data > 0)]

    signed_positive = positive("signed")
    rms_positive = positive("rms")
    peak_positive = positive("peak_signed")
    if not len(signed_positive) or not len(rms_positive) or not len(peak_positive):
        raise ValueError("coefficient matrices contain no non-zero values")
    signed_limit = float(np.percentile(signed_positive, 99.5))
    peak_limit = float(np.percentile(peak_positive, 99.5))
    return {
        "signed": SymLogNorm(
            linthresh=max(float(np.percentile(signed_positive, 10)), signed_limit * 1e-4),
            linscale=1.0,
            vmin=-signed_limit,
            vmax=signed_limit,
            base=10,
        ),
        "rms": LogNorm(
            vmin=float(np.percentile(rms_positive, 1)),
            vmax=float(np.percentile(rms_positive, 99.5)),
        ),
        "peak_signed": SymLogNorm(
            linthresh=max(float(np.percentile(peak_positive, 10)), peak_limit * 1e-4),
            linscale=1.0,
            vmin=-peak_limit,
            vmax=peak_limit,
            base=10,
        ),
        "short_lag": Normalize(vmin=0.0, vmax=1.0),
        "long_lag": Normalize(vmin=1.0, vmax=float(lag_order.max()) / 8.0),
    }


def _draw_metric(axis, metric: Metric, data: np.ndarray, norms: dict[str, object]):
    if metric.kind == "lag":
        short = axis.imshow(
            np.ma.masked_where(~np.isfinite(data) | (data > 1.0), data),
            aspect="auto",
            cmap="Blues",
            norm=norms["short_lag"],
        )
        long = axis.imshow(
            np.ma.masked_where(~np.isfinite(data) | (data <= 1.0), data),
            aspect="auto",
            cmap="magma",
            norm=norms["long_lag"],
        )
        return short, long
    return (
        axis.imshow(
            np.ma.masked_equal(data, 0) if metric.kind == "rms" else np.ma.masked_invalid(data),
            aspect="auto",
            cmap="viridis" if metric.kind == "rms" else "RdBu_r",
            norm=norms[metric.kind],
        ),
    )


def _decorate(
    axis,
    output_labels: list[str],
    input_labels: list[str],
    labels: bool,
    *,
    group_labels: bool = False,
) -> None:
    for boundary in (26.5, 30.5, 35.5):
        axis.axvline(boundary, color="white", linewidth=0.8, alpha=0.95)
    for boundary in (26.5, 30.5):
        axis.axhline(boundary, color="white", linewidth=0.8, alpha=0.95)
    if labels:
        axis.set_xticks(np.arange(len(input_labels)), input_labels, rotation=90, fontsize=5)
        axis.set_yticks(np.arange(len(output_labels)), output_labels, fontsize=5.5)
    elif group_labels:
        axis.set_xticks([13, 28.5, 33, 38], ["density", "drivers", "SABER", "masks"])
        axis.set_yticks([13, 28.5, 33], ["density", "drivers", "SABER"])
        axis.tick_params(axis="y", labelsize=6, length=2)
        axis.tick_params(axis="x", labelsize=5, labelrotation=35, length=2)
        for label in axis.get_xticklabels():
            label.set_horizontalalignment("right")
    else:
        axis.set_xticks([])
        axis.set_yticks([])


def _add_colorbar(figure, images, metric: Metric, *, cax, label: bool = True) -> None:
    if metric.kind != "lag":
        bar = figure.colorbar(images[0], cax=cax)
        if label:
            bar.set_label(metric.colorbar_label, fontsize=7)
        bar.ax.tick_params(labelsize=6)
        return
    short_ax, long_ax = cax
    short_bar = figure.colorbar(images[0], cax=short_ax, ticklocation="left")
    short_bar.set_ticks(LAG_TICKS[:3], labels=LAG_TICK_LABELS[:3])
    long_bar = figure.colorbar(images[1], cax=long_ax)
    long_bar.set_ticks(LAG_TICKS[3:], labels=LAG_TICK_LABELS[3:])
    if label:
        long_bar.set_label(metric.colorbar_label, fontsize=7)
    for bar in (short_bar, long_bar):
        bar.ax.tick_params(labelsize=6)


def _page_title(figure, title: str, subtitle: str) -> None:
    figure.suptitle(title, fontsize=14, fontweight="bold", y=0.975)
    figure.text(0.5, 0.935, subtitle, ha="center", va="top", fontsize=7)


def _group_note() -> str:
    return "Predictors: density profiles (27) | drivers (4) | SABER emissions (5) | availability masks (5)"


def _save_page(pdf: PdfPages, figure, preview: Path | None) -> None:
    pdf.savefig(figure)
    if preview is not None:
        figure.savefig(preview, dpi=PNG_DPI)
    plt.close(figure)


def _metric_booklet(
    output: Path,
    values: dict[str, tuple[np.ndarray, ...]],
    output_labels: list[str],
    input_labels: list[str],
    norms: dict[str, object],
) -> tuple[Path, Path]:
    pdf_path = output / "metric_booklet.pdf"
    preview = output / "metric_booklet_preview.png"
    with PdfPages(pdf_path) as pdf:
        for metric in METRICS:
            figure, axes = plt.subplots(
                1, 3, figsize=A4_LANDSCAPE, sharex=True, sharey=True
            )
            images = None
            for index, axis in enumerate(axes):
                images = _draw_metric(axis, metric, values[metric.key][index], norms)
                _decorate(axis, output_labels, input_labels, labels=True)
                axis.set_title(f"W{index + 1}: degree {index + 1}", fontsize=10)
                if index:
                    axis.tick_params(axis="y", labelleft=False)
            axes[0].set_ylabel("Predicted variable", fontsize=8)
            assert images is not None
            _page_title(figure, metric.title, f"{metric.definition}.  {_group_note()}")
            figure.subplots_adjust(
                left=0.12, right=0.87, bottom=0.22, top=0.88, wspace=0.10
            )
            if metric.kind == "lag":
                _add_colorbar(
                    figure,
                    images,
                    metric,
                    cax=(
                        figure.add_axes((0.89, 0.34, 0.012, 0.42)),
                        figure.add_axes((0.925, 0.34, 0.018, 0.42)),
                    ),
                )
            else:
                _add_colorbar(
                    figure,
                    images,
                    metric,
                    cax=figure.add_axes((0.90, 0.34, 0.018, 0.42)),
                )
            _save_page(pdf, figure, preview if metric.key == "average_lag" else None)
    return pdf_path, preview


def _matrix_booklet(
    output: Path,
    values: dict[str, tuple[np.ndarray, ...]],
    output_labels: list[str],
    input_labels: list[str],
    norms: dict[str, object],
) -> tuple[Path, Path]:
    pdf_path = output / "matrix_booklet.pdf"
    preview = output / "matrix_booklet_preview.png"
    with PdfPages(pdf_path) as pdf:
        for matrix_index, matrix_name in enumerate(MATRIX_NAMES):
            figure, axes = plt.subplots(2, 3, figsize=A4_LANDSCAPE)
            for metric, axis in zip(METRICS, axes.flat, strict=False):
                images = _draw_metric(axis, metric, values[metric.key][matrix_index], norms)
                _decorate(
                    axis,
                    output_labels,
                    input_labels,
                    labels=False,
                    group_labels=True,
                )
                axis.set_title(metric.title, fontsize=9)
                if metric.kind == "lag":
                    _add_colorbar(
                        figure,
                        images,
                        metric,
                        cax=(
                            axis.inset_axes((1.02, 0.08, 0.025, 0.84)),
                            axis.inset_axes((1.09, 0.08, 0.04, 0.84)),
                        ),
                        label=False,
                    )
                else:
                    _add_colorbar(
                        figure,
                        images,
                        metric,
                        cax=axis.inset_axes((1.02, 0.08, 0.04, 0.84)),
                        label=False,
                    )
            note_axis = axes.flat[5]
            note_axis.axis("off")
            note_axis.text(
                0.0, 0.95, "Reading guide", fontsize=11, fontweight="bold", va="top"
            )
            note_axis.text(
                0.0, 0.82,
                "Rows: predicted variables\nColumns: lagged predictors\n\n"
                "White separators\n27 density | 4 drivers | 5 SABER | 5 masks\n\n"
                "Signed metrics: RdBu_r, symmetric log\nRMS: viridis, log\nLag: Blues 0–1 day; magma >1 day\n\n"
                "Lag colorbars: 0h, 3h, 1 day, then\n10 days to 6 months.",
                fontsize=8,
                va="top",
                linespacing=1.5,
            )
            _page_title(figure, f"{matrix_name}: degree {matrix_index + 1}", _group_note())
            figure.subplots_adjust(
                left=0.08, right=0.91, bottom=0.10, top=0.88, hspace=0.38, wspace=0.48
            )
            _save_page(pdf, figure, preview if matrix_index == 0 else None)
    return pdf_path, preview


def _single_panel_atlas(
    output: Path,
    values: dict[str, tuple[np.ndarray, ...]],
    output_labels: list[str],
    input_labels: list[str],
    norms: dict[str, object],
) -> tuple[Path, Path]:
    pdf_path = output / "single_panel_atlas.pdf"
    preview = output / "single_panel_atlas_preview.png"
    with PdfPages(pdf_path) as pdf:
        for metric in METRICS:
            for matrix_index, matrix_name in enumerate(MATRIX_NAMES):
                figure, axis = plt.subplots(figsize=A4_LANDSCAPE)
                images = _draw_metric(axis, metric, values[metric.key][matrix_index], norms)
                _decorate(axis, output_labels, input_labels, labels=True)
                axis.set_xlabel("Lagged predictor", fontsize=9)
                axis.set_ylabel("Predicted variable", fontsize=9)
                _page_title(
                    figure,
                    f"{metric.title} — {matrix_name}: degree {matrix_index + 1}",
                    f"{metric.definition}.  {_group_note()}",
                )
                figure.subplots_adjust(left=0.16, right=0.84, bottom=0.25, top=0.88)
                if metric.kind == "lag":
                    _add_colorbar(
                        figure,
                        images,
                        metric,
                        cax=(
                            figure.add_axes((0.87, 0.30, 0.012, 0.48)),
                            figure.add_axes((0.91, 0.30, 0.018, 0.48)),
                        ),
                    )
                else:
                    _add_colorbar(
                        figure,
                        images,
                        metric,
                        cax=figure.add_axes((0.88, 0.30, 0.018, 0.48)),
                    )
                _save_page(
                    pdf,
                    figure,
                    preview
                    if metric.key == "average_lag" and matrix_index == 0
                    else None,
                )
    return pdf_path, preview


def _plot(model_path: Path, output: Path) -> dict[str, object]:
    """Write all three A4 layout prototypes and return their manifest."""
    matrices, output_labels, input_labels, lag_order = _load(model_path)
    aggregated = tuple(aggregate(matrix, lag_order) for matrix in matrices)
    values = {
        metric.key: tuple(item[index] for item in aggregated)
        for index, metric in enumerate(METRICS)
    }
    norms = _norms(values, lag_order)
    output.mkdir(parents=True, exist_ok=True)
    metric_pdf, metric_preview = _metric_booklet(output, values, output_labels, input_labels, norms)
    matrix_pdf, matrix_preview = _matrix_booklet(output, values, output_labels, input_labels, norms)
    atlas_pdf, atlas_preview = _single_panel_atlas(output, values, output_labels, input_labels, norms)
    manifest_path = output / "heatmaps_a4_prototypes.json"
    manifest = {
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "a4_landscape_inches": list(A4_LANDSCAPE),
        "png_dpi": PNG_DPI,
        "metric_definitions": {metric.key: metric.definition for metric in METRICS},
        "color_semantics": {
            "signed_and_peak_signed": "RdBu_r with global symmetric-log normalization",
            "rms": "viridis with global log normalization",
            "lag_0_to_1_day": "Blues with linear 0–1 day normalization",
            "lag_over_1_day": "magma with linear 1 day–six month normalization",
            "lag_ticks_days": LAG_TICKS,
            "lag_tick_labels": LAG_TICK_LABELS,
        },
        "outputs": {
            "metric_booklet": {
                "pdf": str(metric_pdf),
                "preview_png": str(metric_preview),
                "pages": 5,
                "intended_use": "Compare each metric across W1, W2, and W3; trades detailed labels for side-by-side comparison.",
            },
            "matrix_booklet": {
                "pdf": str(matrix_pdf),
                "preview_png": str(matrix_preview),
                "pages": 3,
                "intended_use": "Compact visual overview of all five metrics for one matrix; uses sparse positional labels.",
            },
            "single_panel_atlas": {
                "pdf": str(atlas_pdf),
                "preview_png": str(atlas_preview),
                "pages": 15,
                "intended_use": "Detailed single-metric, single-matrix lookup atlas with maximal labels; trades compactness for 15 pages.",
            },
            "manifest": str(manifest_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def plot(model_path: Path, output: Path) -> dict[str, object]:
    """Render without inheriting LaTeX text settings from other figure code."""
    with plt.rc_context({"text.usetex": False}):
        return _plot(model_path, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(plot(args.model, args.output), indent=2))


if __name__ == "__main__":
    main()
