import json

import matplotlib
import numpy as np

from scripts.plot_polynomial_var_heatmaps import aggregate, mask_names, plot

matplotlib.use("Agg")


def test_aggregate_and_mask_labels():
    weights = np.array([[[1.0, -3.0]]])
    signed, rms, average_lag_days, peak_signed, peak_lag_days = aggregate(
        weights, np.array([1, 2])
    )
    assert signed.item() == -2.0
    assert rms.item() == np.sqrt(5.0)
    assert average_lag_days.item() == 1.75 / 8.0
    assert peak_signed.item() == -3.0
    assert peak_lag_days.item() == 2.0 / 8.0
    assert mask_names(np.array(["CO2cool", "OH_16_ver"]), np.array([0, 0]), 1) == [
        "available: CO₂ cooling/OH 1.6"
    ]


def test_plot_writes_heatmap_and_provenance(tmp_path):
    channels = np.array([f"channel_{index}" for index in range(36)])
    augmented = np.array([*channels, "availability_pattern_0"])
    shape = (36, 37, 2)
    rng = np.random.default_rng(4)
    matrices = [rng.normal(size=shape).astype(np.float32) for _ in range(3)]
    matrices[1][:, -1] = 0
    matrices[2][:, -1] = 0
    model = tmp_path / "model.npz"
    np.savez_compressed(
        model,
        W1=matrices[0],
        W2=matrices[1],
        W3=matrices[2],
        channel_names=channels,
        augmented_channel_names=augmented,
        mask_mapping=np.array([0, *([-1] * 35)]),
        mask_channel_count=np.array(1),
        lag_order=np.array([1, 2]),
    )

    report = plot(model, tmp_path / "plots")

    assert report["shape_per_matrix"] == [36, 37, 2]
    assert "average lag" in report["lag_aggregation"]["row_3"]
    assert "argmax" in report["lag_aggregation"]["row_4"]
    assert "argmax" in report["lag_aggregation"]["row_5"]
    assert report["average_lag_colormaps"] == {
        "0_to_1_day": "Blues",
        "over_1_day": "magma",
    }
    assert report["average_lag_color_scale"].startswith("separate linear")
    assert report["average_lag_ticks"]["labels"] == [
        "0h",
        "3h",
        "1day",
        "10 days",
        "1 month",
        "2 months",
        "3 months",
        "100 days",
        "6 months",
    ]
    assert (tmp_path / "plots/polynomial_var_matrix_heatmaps.png").stat().st_size > 0
    assert (tmp_path / "plots/polynomial_var_matrix_heatmaps.pdf").stat().st_size > 0
    saved = json.loads(
        (tmp_path / "plots/polynomial_var_matrix_heatmaps.json").read_text()
    )
    assert saved["model_sha256"] == report["model_sha256"]
