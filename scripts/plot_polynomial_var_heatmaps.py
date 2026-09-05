"""Plot lag-aggregated heatmaps for the throwaway cubic polynomial VAR matrices."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.colors import LogNorm, Normalize, SymLogNorm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_MODEL = Path(
    "outputs/prototypes/polynomial_var_throwaway/phoenix_full_lag_qc_v1/final_model.npz"
)
DEFAULT_OUTPUT = DEFAULT_MODEL.parent / "heatmaps"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_name(name: str) -> str:
    if name.startswith("log10rho_"):
        return f"log₁₀ρ {name.removeprefix('log10rho_').replace('km', ' km')}"
    return {
        "F10.7_raw": "F10.7",
        "co2_mlo_ppm": "CO₂ MLO",
        "CO2cool": "CO₂ cooling",
        "NOcool": "NO cooling",
        "O2_1delta_ver": "O₂ 1Δ",
        "OH_16_ver": "OH 1.6",
        "OH_20_ver": "OH 2.0",
    }.get(name, name)


def mask_names(
    channel_names: np.ndarray, mask_mapping: np.ndarray, mask_count: int
) -> list[str]:
    labels = []
    for mask_id in range(mask_count):
        members = [
            display_name(str(name))
            for name, mapped in zip(channel_names, mask_mapping, strict=True)
            if int(mapped) == mask_id
        ]
        if len(members) > 3:
            label = f"{members[0]}…{members[-1]}"
        else:
            label = "/".join(members)
        labels.append(f"available: {label}")
    return labels


def aggregate(
    weights: np.ndarray, lag_order: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return signed sum, RMS, mean lag, signed peak, and peak lag."""
    magnitude = np.abs(weights)
    magnitude_sum = magnitude.sum(axis=2)
    average_lag_days = (
        np.divide(
            np.sum(magnitude * lag_order[None, None, :], axis=2),
            magnitude_sum,
            out=np.full(magnitude_sum.shape, np.nan),
            where=magnitude_sum > 0,
        )
        / 8.0
    )
    peak_indices = np.argmax(magnitude, axis=2)
    peak_signed = np.take_along_axis(weights, peak_indices[:, :, None], axis=2)[:, :, 0]
    peak_lag_days = lag_order[peak_indices] / 8.0
    peak_signed = np.where(magnitude_sum > 0, peak_signed, np.nan)
    peak_lag_days = np.where(magnitude_sum > 0, peak_lag_days, np.nan)
    return (
        weights.sum(axis=2),
        np.sqrt(np.mean(np.square(weights), axis=2)),
        average_lag_days,
        peak_signed,
        peak_lag_days,
    )


def _plot(model_path: Path, output: Path) -> dict[str, object]:
    with np.load(model_path, allow_pickle=False) as model:
        matrices = [model[name].astype(np.float64) for name in ("W1", "W2", "W3")]
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

    input_labels = [display_name(name) for name in outputs] + mask_names(
        outputs, mapping, mask_count
    )
    if len(input_labels) != len(augmented):
        raise ValueError("derived input labels do not match augmented channels")
    output_labels = [display_name(name) for name in outputs]
    signed, rms, average_lag_days, peak_signed, peak_lag_days = zip(
        *(aggregate(matrix, lag_order) for matrix in matrices), strict=True
    )

    signed_values = np.concatenate([np.abs(item).ravel() for item in signed])
    signed_positive = signed_values[signed_values > 0]
    rms_values = np.concatenate([item.ravel() for item in rms])
    rms_positive = rms_values[rms_values > 0]
    peak_values = np.concatenate([np.abs(item).ravel() for item in peak_signed])
    peak_positive = peak_values[np.isfinite(peak_values) & (peak_values > 0)]
    if not len(signed_positive) or not len(rms_positive) or not len(peak_positive):
        raise ValueError("coefficient matrices contain no non-zero values")

    signed_limit = float(np.percentile(signed_positive, 99.5))
    signed_linear = max(float(np.percentile(signed_positive, 10)), signed_limit * 1e-4)
    rms_min = float(np.percentile(rms_positive, 1))
    rms_max = float(np.percentile(rms_positive, 99.5))
    peak_limit = float(np.percentile(peak_positive, 99.5))
    peak_linear = max(float(np.percentile(peak_positive, 10)), peak_limit * 1e-4)
    signed_norm = SymLogNorm(
        linthresh=signed_linear,
        linscale=1.0,
        vmin=-signed_limit,
        vmax=signed_limit,
        base=10,
    )
    rms_norm = LogNorm(vmin=rms_min, vmax=rms_max)
    peak_norm = SymLogNorm(
        linthresh=peak_linear,
        linscale=1.0,
        vmin=-peak_limit,
        vmax=peak_limit,
        base=10,
    )
    short_lag_norm = Normalize(vmin=0.0, vmax=1.0)
    long_lag_norm = Normalize(vmin=1.0, vmax=float(lag_order.max()) / 8.0)

    figure, axes = plt.subplots(5, 3, figsize=(30, 45), sharex=True, sharey=True)
    signed_image = rms_image = peak_image = None
    average_short_image = average_long_image = None
    peak_short_image = peak_long_image = None
    for column, order in enumerate((1, 2, 3)):
        signed_image = axes[0, column].imshow(
            signed[column], aspect="auto", cmap="RdBu_r", norm=signed_norm
        )
        rms_image = axes[1, column].imshow(
            np.ma.masked_equal(rms[column], 0),
            aspect="auto",
            cmap="viridis",
            norm=rms_norm,
        )
        average_short_image = axes[2, column].imshow(
            np.ma.masked_where(
                ~np.isfinite(average_lag_days[column])
                | (average_lag_days[column] > 1.0),
                average_lag_days[column],
            ),
            aspect="auto",
            cmap="Blues",
            norm=short_lag_norm,
        )
        average_long_image = axes[2, column].imshow(
            np.ma.masked_where(
                ~np.isfinite(average_lag_days[column])
                | (average_lag_days[column] <= 1.0),
                average_lag_days[column],
            ),
            aspect="auto",
            cmap="magma",
            norm=long_lag_norm,
        )
        peak_image = axes[3, column].imshow(
            np.ma.masked_invalid(peak_signed[column]),
            aspect="auto",
            cmap="RdBu_r",
            norm=peak_norm,
        )
        peak_short_image = axes[4, column].imshow(
            np.ma.masked_where(
                ~np.isfinite(peak_lag_days[column]) | (peak_lag_days[column] > 1.0),
                peak_lag_days[column],
            ),
            aspect="auto",
            cmap="Blues",
            norm=short_lag_norm,
        )
        peak_long_image = axes[4, column].imshow(
            np.ma.masked_where(
                ~np.isfinite(peak_lag_days[column]) | (peak_lag_days[column] <= 1.0),
                peak_lag_days[column],
            ),
            aspect="auto",
            cmap="magma",
            norm=long_lag_norm,
        )
        axes[0, column].set_title(f"W{order}: degree {order}", fontsize=15)
        axes[4, column].set_xlabel("Lagged predictor", fontsize=12)
        for row in range(5):
            axis = axes[row, column]
            axis.set_xticks(np.arange(len(input_labels)), input_labels)
            axis.tick_params(axis="x", labelrotation=90, labelsize=6)
            axis.tick_params(axis="y", labelsize=6)
            for boundary in (26.5, 30.5, 35.5):
                axis.axvline(boundary, color="white", linewidth=0.7, alpha=0.9)
            for boundary in (26.5, 30.5):
                axis.axhline(boundary, color="white", linewidth=0.7, alpha=0.9)
    axes[0, 0].set_ylabel("Predicted variable\nSigned coefficient sum over 1,464 lags")
    axes[1, 0].set_ylabel("Predicted variable\nCoefficient RMS over 1,464 lags")
    axes[2, 0].set_ylabel("Predicted variable\nMagnitude-weighted average lag (days)")
    axes[3, 0].set_ylabel(
        "Predicted variable\nSigned coefficient at maximum |coefficient|"
    )
    axes[4, 0].set_ylabel("Predicted variable\nLag of maximum |coefficient| (days)")
    for row in range(5):
        axes[row, 0].set_yticks(np.arange(len(output_labels)), output_labels)
    for row in range(4):
        for axis in axes[row]:
            axis.tick_params(axis="x", labelbottom=False)

    assert (
        signed_image is not None
        and rms_image is not None
        and average_short_image is not None
        and average_long_image is not None
        and peak_image is not None
        and peak_short_image is not None
        and peak_long_image is not None
    )
    figure.subplots_adjust(
        left=0.11, right=0.90, bottom=0.11, top=0.95, hspace=0.10, wspace=0.08
    )
    signed_bar = figure.colorbar(
        signed_image, cax=figure.add_axes((0.92, 0.80, 0.012, 0.14))
    )
    signed_bar.set_label("Σ lag coefficients (signed; symmetric log scale)")
    rms_bar = figure.colorbar(rms_image, cax=figure.add_axes((0.92, 0.63, 0.012, 0.14)))
    rms_bar.set_label("Lag-RMS coefficient magnitude (log scale)")

    def add_lag_colorbars(short_image, long_image, y: float, label: str) -> None:
        short_bar = figure.colorbar(
            short_image,
            cax=figure.add_axes((0.912, y, 0.008, 0.13)),
            ticklocation="left",
        )
        short_bar.set_ticks([0.0, 3.0 / 24.0, 1.0], labels=["0h", "3h", "1day"])
        long_bar = figure.colorbar(
            long_image, cax=figure.add_axes((0.945, y, 0.012, 0.13))
        )
        long_bar.set_ticks(
            [10.0, 30.5, 61.0, 91.5, 100.0, 183.0],
            labels=[
                "10 days",
                "1 month",
                "2 months",
                "3 months",
                "100 days",
                "6 months",
            ],
        )
        long_bar.set_label(label)

    add_lag_colorbars(
        average_short_image,
        average_long_image,
        0.46,
        "Magnitude-weighted average lag (piecewise linear)",
    )
    peak_bar = figure.colorbar(
        peak_image, cax=figure.add_axes((0.92, 0.29, 0.012, 0.13))
    )
    peak_bar.set_label("Coefficient at maximum |coefficient| (symmetric log scale)")
    add_lag_colorbars(
        peak_short_image,
        peak_long_image,
        0.12,
        "Lag of maximum |coefficient| (piecewise linear)",
    )
    figure.suptitle(
        "Six-month cubic polynomial VAR coefficient matrices\n"
        "36 jointly predicted variables; availability masks are linear-only",
        fontsize=18,
    )
    output.mkdir(parents=True, exist_ok=True)
    png = output / "polynomial_var_matrix_heatmaps.png"
    pdf = output / "polynomial_var_matrix_heatmaps.pdf"
    figure.savefig(png, dpi=220, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)

    report = {
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "shape_per_matrix": list(expected),
        "lag_aggregation": {
            "row_1": "sum of coefficients over lag axis; sign retained",
            "row_2": "sqrt(mean(coefficient^2 over lag axis))",
            "row_3": "magnitude-weighted average lag = sum(abs(coefficient) * lag) / sum(abs(coefficient)), converted from 3-hour lag steps to days",
            "row_4": "signed coefficient at argmax(abs(coefficient)) over lag",
            "row_5": "lag at argmax(abs(coefficient)), converted from 3-hour lag steps to days",
        },
        "color_limits": {
            "signed_absolute_99_5_percentile": signed_limit,
            "signed_linear_threshold": signed_linear,
            "rms_1_percentile": rms_min,
            "rms_99_5_percentile": rms_max,
            "peak_absolute_99_5_percentile": peak_limit,
            "peak_linear_threshold": peak_linear,
            "average_lag_days_min": 0.0,
            "average_lag_days_max": float(lag_order.max()) / 8.0,
        },
        "average_lag_colormaps": {
            "0_to_1_day": "Blues",
            "over_1_day": "magma",
        },
        "average_lag_color_scale": "separate linear scales for 0--1 day and 1--183 days",
        "average_lag_ticks": {
            "days": [0.0, 0.125, 1.0, 10.0, 30.5, 61.0, 91.5, 100.0, 183.0],
            "labels": [
                "0h",
                "3h",
                "1day",
                "10 days",
                "1 month",
                "2 months",
                "3 months",
                "100 days",
                "6 months",
            ],
        },
        "matrix_ranges": {
            f"W{index}": {
                "signed_min": float(signed[index - 1].min()),
                "signed_max": float(signed[index - 1].max()),
                "rms_min": float(rms[index - 1].min()),
                "rms_max": float(rms[index - 1].max()),
                "average_lag_days_min": float(np.nanmin(average_lag_days[index - 1])),
                "average_lag_days_max": float(np.nanmax(average_lag_days[index - 1])),
                "peak_signed_min": float(np.nanmin(peak_signed[index - 1])),
                "peak_signed_max": float(np.nanmax(peak_signed[index - 1])),
                "peak_lag_days_min": float(np.nanmin(peak_lag_days[index - 1])),
                "peak_lag_days_max": float(np.nanmax(peak_lag_days[index - 1])),
            }
            for index in range(1, 4)
        },
        "png": str(png),
        "pdf": str(pdf),
    }
    report_path = output / "polynomial_var_matrix_heatmaps.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


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
