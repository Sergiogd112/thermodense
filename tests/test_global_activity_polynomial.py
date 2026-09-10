"""Contracts for the pooled global-activity polynomial sensitivity benchmark."""

import argparse
import json
import shutil
import sys
import types
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from thermodense.benchmarks import global_activity_polynomial as benchmark


@dataclass
class TinyInput:
    timestamps: np.ndarray
    values: np.ndarray
    available: np.ndarray
    activity_f107_center81: np.ndarray
    node_names: list[str]
    manifest: dict[str, str]


def tiny_input(rows: int = 40) -> TinyInput:
    values = np.arange(rows * 14, dtype=float).reshape(rows, 14)
    values[:, :10] /= 100.0
    values[8, 2] = np.nan
    timestamps = np.datetime64("2000-01-01") + np.arange(rows)
    return TinyInput(
        timestamps,
        values,
        np.isfinite(values),
        np.array([90.0, 100.0, 159.0, 160.0] * (rows // 4), dtype=float),
        [*[f"log10rho_{alt}" for alt in (250, 275, 325, 375, 400, 425, 475, 525, 550, 575)], "F10.7_OBS", "AP_AVG", "KP_SUM", "CO2_ppm"],
        {"fixture": "tiny"},
    )


def fake_group_mask(activity: np.ndarray, group: str) -> np.ndarray:
    if group == "all":
        return np.ones(len(activity), dtype=bool)
    if group == "low":
        return activity < 100
    if group == "moderate":
        return (activity >= 100) & (activity < 160)
    if group == "high":
        return activity >= 160
    raise ValueError(group)


def install_input(monkeypatch, source: TinyInput) -> None:
    module = types.ModuleType("thermodense.benchmarks.global_activity_input")
    module.load_manifest = lambda path: source
    module.group_mask = fake_group_mask
    monkeypatch.setitem(sys.modules, module.__name__, module)


def test_train_scaling_and_lags_cannot_see_future_values():
    source = tiny_input()
    early = benchmark.prepare(source.values, source.available, lag=3)
    changed = source.values.copy()
    changed[31:, :] += 1_000_000
    late = benchmark.prepare(changed, np.isfinite(changed), lag=3)
    assert np.array_equal(early.means, late.means)
    assert np.array_equal(early.stds, late.stds)
    values = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    lags, _ = benchmark.batch_lags(values, torch.ones_like(values), torch.tensor([6]), 3)
    assert lags.tolist() == [[[10.0, 8.0, 6.0], [11.0, 9.0, 7.0]]]


def test_degree_one_has_no_nonlinear_weights_and_missing_is_indicated():
    scales = torch.ones((1, 14, 2))
    model = benchmark.PolynomialVAR(1, 2, scales, torch.ones((14, 2)))
    assert len(model.value_weights) == 1
    with torch.no_grad():
        model.mask_weights[0, 0, 0] = 3
    masks = torch.zeros((1, 14, 2))
    masks[0, 0, 0] = 1
    prediction = model(torch.zeros((1, 14, 2)), masks)
    assert prediction[0, 0].item() == 3
    _, w2, w3 = model.exported()
    assert not w2.any() and not w3.any()


def test_groups_and_persistence_common_support_report_counts():
    standardized = np.zeros((5, 14), dtype=np.float32)
    standardized[:, :10] = np.arange(5)[:, None]
    available = np.ones((5, 14), dtype=bool)
    available[2, 0] = False  # Test target 2 lacks a target; target 3 lacks persistence.
    indices = np.array([2, 3, 4])
    report = benchmark.evaluate_models(np.zeros((3, 10)), standardized, available, indices, np.array([90.0, 100.0, 160.0]), np.zeros(10), fake_group_mask)
    assert report["all_available"]["polynomial"]["low"]["dates"] == 1
    assert report["all_available"]["polynomial"]["moderate"]["dates"] == 1
    assert report["all_available"]["polynomial"]["high"]["dates"] == 1
    assert report["all_available"]["polynomial"]["all"]["support"][0] == 2
    assert report["same_support_with_persistence"]["polynomial"]["all"]["support"][0] == 1
    assert report["same_support_with_persistence"]["persistence"]["all"]["support"] == report["same_support_with_persistence"]["polynomial"]["all"]["support"]
    assert report["all_available"]["polynomial"]["low"]["per_altitude_standardized_mse"][0] is None


def test_tiny_end_to_end_writes_reproducible_artifacts(monkeypatch, tmp_path):
    source = tiny_input()
    install_input(monkeypatch, source)
    args = argparse.Namespace(manifest=tmp_path / "manifest.json", output=tmp_path / "result", lag=3, epochs=1, threads=1, batch_size=8, lambdas="0.001", seed=4)
    first = benchmark.run(args)
    second = benchmark.run(args)
    assert first == second and first.exists()
    with np.load(first) as model:
        assert model["W1"].shape == (10, 28, 3)
        assert model["W2"].shape == model["W3"].shape == (10, 14, 3)
        assert model["degree_3_W3"].shape == (10, 14, 3)
        assert not model["degree_1_W3"].any()
        assert model["lag_days"].tolist() == [1, 2, 3]
    provenance = json.loads((args.output / "provenance.json").read_text())
    metrics = json.loads((args.output / "metrics.json").read_text())
    assert provenance["status"] == "complete" and provenance["fingerprint"]
    assert provenance["result_hashes"]["models.npz"]
    assert metrics["test"]["selected_test"]["same_support_with_persistence"]["polynomial"]["all"]["support"]
    assert set(metrics["test"]["validation_by_degree"]) == {"1", "2", "3"}


def test_interrupted_resume_matches_uninterrupted_trajectory(monkeypatch, tmp_path):
    source = tiny_input()
    install_input(monkeypatch, source)
    args = argparse.Namespace(manifest=tmp_path / "manifest.json", output=tmp_path / "resumed", lag=3, epochs=2, threads=1, batch_size=8, lambdas="0.001", seed=9)
    original = benchmark._atomic_torch
    interrupted = False

    def checkpoint_then_interrupt(value, path):
        nonlocal interrupted
        original(value, path)
        if not interrupted:
            interrupted = True
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(benchmark, "_atomic_torch", checkpoint_then_interrupt)
    with pytest.raises(RuntimeError, match="simulated"):
        benchmark.run(args)
    monkeypatch.setattr(benchmark, "_atomic_torch", original)
    resumed = benchmark.run(args)
    clean_args = argparse.Namespace(**{**vars(args), "output": tmp_path / "clean"})
    clean = benchmark.run(clean_args)
    with np.load(resumed) as resumed_model, np.load(clean) as clean_model:
        for name in ("W1", "W2", "W3"):
            assert np.array_equal(resumed_model[name], clean_model[name])


def test_resume_rejects_checkpoint_copied_from_another_lambda(monkeypatch, tmp_path):
    install_input(monkeypatch, tiny_input())
    args = argparse.Namespace(manifest=tmp_path / "manifest.json", output=tmp_path / "result", lag=3, epochs=1, threads=1, batch_size=8, lambdas="0.001,0.002", seed=3)
    benchmark.run(args)
    trials = json.loads((args.output / "trials.json").read_text())["trials"]
    first = next(item for item in trials if item["degree"] == 1 and item["lambda"] == 0.001)
    second = next(item for item in trials if item["degree"] == 1 and item["lambda"] == 0.002)
    shutil.copyfile(first["checkpoint"], second["checkpoint"])
    provenance_path = args.output / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["status"] = "running"
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(FileExistsError, match="incompatible trial checkpoint"):
        benchmark.run(args)
