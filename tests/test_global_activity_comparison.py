"""Contracts for read-only global activity linear/nonlinear comparison."""

import json
from pathlib import Path

import numpy as np
import pytest

from thermodense.benchmarks import global_activity_comparison as comparison


def test_reconstruction_uses_zero_for_missing_values_and_availability_weights():
    standardized = np.array([[2.0] * 14, [3.0] * 14, [4.0] * 14])
    available = np.ones_like(standardized, dtype=bool)
    available[1, 0] = False
    matrices = {"lag_days": np.array([1]), "degree_1_W1": np.zeros((10, 28, 1))}
    matrices["degree_1_W1"][0, 0, 0] = 5.0
    matrices["degree_1_W1"][0, 14, 0] = 7.0
    prediction = comparison.reconstruct_predictions(standardized, available, np.array([2]), matrices, 1)
    assert prediction[0, 0] == 0.0  # unavailable value becomes zero and its indicator is zero


def test_score_uses_one_identical_support_for_every_model():
    target = np.zeros((2, 10))
    support = np.ones((2, 10), dtype=bool)
    support[0, 0] = False
    activity = np.array([90.0, 170.0])
    linear = comparison._score(np.ones((2, 10)), target, support, activity)
    nonlinear = comparison._score(np.full((2, 10), 2.0), target, support, activity)
    assert linear["all"]["support"] == nonlinear["all"]["support"]
    assert linear["low"]["pooled_support"] == nonlinear["low"]["pooled_support"] == 9


def test_joint_block_bootstrap_is_deterministic_and_reports_missing_support():
    linear = np.full((12, 10), 4.0)
    nonlinear = np.full((12, 10), 1.0)
    support = np.ones((12, 10), dtype=bool)
    first = comparison.bootstrap_improvement(linear, nonlinear, support, 3, seed=4, replicates=100)
    second = comparison.bootstrap_improvement(linear, nonlinear, support, 3, seed=4, replicates=100)
    assert first == second
    assert first["interval_95_percent"] == [75.0, 75.0]
    absent = comparison.bootstrap_improvement(linear, nonlinear, np.zeros_like(support), 3, replicates=10)
    assert absent == {"interval_95_percent": None, "unsupported_replicates": 10}


def test_verify_artifacts_rejects_tampered_result_hash(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    results = tmp_path / "results"
    results.mkdir()
    (results / "models.npz").write_bytes(b"not a model")
    (results / "metrics.json").write_text(json.dumps({"fingerprint": "x"}))
    (results / "trials.json").write_text("{}")
    (results / "provenance.json").write_text(json.dumps({
        "status": "complete",
        "result_hashes": {},
        "config": {"lag": 180},
        "interpretation": "predictive sensitivity benchmark; coefficients are not causal",
        "optimization_status": "bounded epoch budget; neither convergence nor globally optimal ridge is claimed",
        "evaluation_activity": "centered81 only; low <100, moderate 100<=F<160, high >=160; unknown excluded from grouped summaries",
        "manifest_identity": {
            "input_contract": "global-activity-daily-v1",
            "calendar": {"cadence": "daily UTC"},
            "transformations": {"detrending": "none", "deseasonalizing": "none", "standardizing": "none", "interpolation": "none"},
            "f107_fit": "F10.7_OBS raw",
            "activity_classification": "F10.7_OBS_CENTER81: low <100, moderate 100<=F<160, high >=160",
        },
    }))
    monkeypatch.setattr(comparison, "load_manifest", lambda path: object())
    with pytest.raises(ValueError, match="result hashes"):
        comparison.verify_artifacts(manifest, results)


def test_verify_artifacts_accepts_known_completed_metrics_and_matrices():
    root = Path(__file__).parents[1]
    manifest = root / "data/products/global-activity-daily-v1/manifest.json"
    results = root / "data/products/global-activity-results-v1"
    if not (manifest.exists() and results.exists()):
        pytest.skip("completed local global-activity artifacts are unavailable")
    source, provenance, metrics = comparison.verify_artifacts(manifest, results)
    assert source.node_names[0] == "log10rho_250"
    assert provenance["result_hashes"] == {
        "metrics.json": "ab4da70bd4f1b3f8aff82b7707568aa89ed7c9d424ef05d9e4f31d30ba87aa28",
        "models.npz": "31b9c3874f2b8503c171570342d4436cb574d18900756a977d6348e7fb910a5c",
    }
    assert metrics["fingerprint"] == provenance["fingerprint"]
