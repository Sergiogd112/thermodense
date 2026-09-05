"""Focused contracts for the isolated throwaway polynomial VAR prototype."""

import argparse
import json

import numpy as np
import pytest
import torch

import scripts.prototype_polynomial_var as runner


def test_lag_ordering_and_no_lookahead():
    values = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    masks = torch.ones(10, 1)
    lagged, _ = runner.batch_lags(values, masks, torch.tensor([5]), 3)
    assert lagged.tolist() == [[[8.0, 6.0, 4.0], [9.0, 7.0, 5.0]]]


def test_polynomial_formula_and_masked_loss():
    model = runner.PolynomialVAR(1, 1, 1, 2)
    with torch.no_grad():
        model.w1_values[:] = torch.tensor([[[2.0, 3.0]]])
        model.w2_values[:] = torch.tensor([[[5.0, 7.0]]])
        model.w3_values[:] = torch.tensor([[[11.0, 13.0]]])
        model.w1_masks[:] = torch.tensor([[[17.0, 19.0]]])
    assert (
        model(torch.tensor([[[2.0, 3.0]]]), torch.tensor([[[1.0, 0.0]]])).item()
        == 2 * 2 + 3 * 3 + 5 * 4 + 7 * 9 + 11 * 8 + 13 * 27 + 17
    )
    assert (
        runner.masked_mse(
            torch.tensor([[2.0, 100.0]]),
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[1.0, 0.0]]),
        ).item()
        == 1.0
    )


def test_internal_feature_scaling_exports_exact_cubic_coefficients():
    scales = torch.tensor([[[2.0]], [[4.0]], [[8.0]]])
    mask_scales = torch.tensor([[10.0]])
    model = runner.PolynomialVAR(1, 1, 1, 1, scales, mask_scales)
    with torch.no_grad():
        model.w1_values.fill_(6.0)
        model.w2_values.fill_(20.0)
        model.w3_values.fill_(56.0)
        model.w1_masks.fill_(20.0)
    value_lags = torch.tensor([[[2.0]]])
    mask_lags = torch.tensor([[[1.0]]])
    prediction = model(value_lags, mask_lags)
    w1, w2, w3 = model.exported_value_coefficients()
    exported_prediction = (
        w1 * value_lags + w2 * value_lags.square() + w3 * value_lags.pow(3)
    ).sum() + (model.exported_mask_coefficients() * mask_lags).sum()
    assert prediction.item() == exported_prediction.item() == 3 * 2 + 5 * 4 + 7 * 8 + 2


def test_feature_scales_include_global_dimension_normalization():
    values = np.ones((6, 1), dtype=np.float32)
    masks = np.ones((6, 1), dtype=np.float32)
    value_scales, mask_scales = runner.feature_scales(
        values, masks, np.array([2, 3, 4, 5]), lag=2
    )
    expected = np.sqrt(2 * (3 + 1))
    assert np.allclose(value_scales, expected)
    assert np.allclose(mask_scales, expected)


def test_mask_deduplication_and_co2_daily_hold(tmp_path):
    availability = np.array([[1, 1, 0], [0, 0, 0], [1, 1, 0], [1, 1, 0]], dtype=bool)
    matrix = runner.mask_matrix(
        availability,
        runner.scaling_and_masks(np.where(availability, 2.0, np.nan), 1)[4],
    )
    assert matrix.shape == (4, 1) and matrix[:, 0].tolist() == [1.0, 0.0, 1.0, 1.0]
    source = tmp_path / "co2.csv"
    source.write_text("# comment\n2020,1,1,2020.0,400\n2020,1,2,2020.1,-99\n")
    slots = np.array(
        ["2020-01-01T00", "2020-01-01T21", "2020-01-02T00"], dtype="datetime64[ns]"
    )
    assert np.isnan(
        runner.load_co2_daily_hold(slots, source)[2]
    ) and runner.load_co2_daily_hold(slots, source)[:2].tolist() == [400.0, 400.0]


def test_limits_and_checkpoint_compatibility(tmp_path):
    with pytest.raises(ValueError, match="1464"):
        runner.validate_limits(1465, 1)
    with pytest.raises(ValueError, match="10"):
        runner.validate_limits(1, 11)
    invalid = argparse.Namespace(
        batch_size=0, epochs=1, learning_rate=1e-3, lambdas="0.001"
    )
    with pytest.raises(ValueError, match="batch-size"):
        runner.validate_fit_arguments(invalid)
    path = tmp_path / "state.pt"
    runner.atomic_torch({"fingerprint": "a"}, path)
    assert runner.checkpoint_compatible(path, "a") and not runner.checkpoint_compatible(
        path, "b"
    )


def test_tiny_fit_exports_zero_mask_polynomials(tmp_path):
    raw = (
        np.tile(np.arange(36, dtype=float), (12, 1))
        + np.arange(12, dtype=float)[:, None]
    )
    raw[2, 0] = np.nan
    prepared = tmp_path / "input.npz"
    runner.atomic_npz(
        prepared,
        time_ns=np.arange(12),
        values=raw,
        channel_names=np.asarray(runner.channel_names()),
    )
    args = argparse.Namespace(
        input=prepared,
        output=tmp_path / "fit",
        lag=2,
        batch_size=1,
        threads=1,
        epochs=1,
        lambdas="0.0",
        learning_rate=1e-3,
        seed=2,
        reset=False,
    )
    torch.manual_seed(2)
    np.random.seed(2)
    final = runner.fit(args)
    with np.load(final) as exported:
        masks = (exported["mask_mapping"] >= 0).sum()
        assert np.array_equal(
            exported["W2"][:, -masks:], np.zeros_like(exported["W2"][:, -masks:])
        )
        assert np.array_equal(
            exported["W3"][:, -masks:], np.zeros_like(exported["W3"][:, -masks:])
        )
    assert (
        json.loads((tmp_path / "fit" / "fit.provenance.json").read_text())["status"]
        == "complete"
    )
    provenance = json.loads((tmp_path / "fit" / "fit.provenance.json").read_text())
    assert np.isfinite(provenance["zero_validation_mse"])
