"""Read-only linear versus additive-polynomial global-activity comparison.

This reconstructs predictions from the completed benchmark's exported matrices;
it never fits or modifies the fitting workflow.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from thermodense.benchmarks.global_activity_input import group_mask, load_manifest, sha256
from thermodense.benchmarks.global_activity_polynomial import DENSITY_COUNT, split_bounds
from thermodense.artifacts import ArtifactIntegrityError, publish_json_new, verify_files
from thermodense.workflow_policies import validate_policy

BLOCK_LENGTHS = (30, 90, 180)
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260909


def _without_historical_paths(value: Any) -> Any:
    """Compare recorded manifests without resolving their historical source paths."""
    if isinstance(value, dict):
        return {key: _without_historical_paths(item) for key, item in value.items() if key != "path"}
    if isinstance(value, list):
        return [_without_historical_paths(item) for item in value]
    return value


def verify_artifacts(manifest_path: Path, results_path: Path) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Load only a completed, hash-verified result bound to this immutable input."""
    source = load_manifest(manifest_path)
    provenance_path = results_path / "provenance.json"
    metrics_path = results_path / "metrics.json"
    models_path = results_path / "models.npz"
    trials_path = results_path / "trials.json"
    if not all(path.is_file() for path in (provenance_path, metrics_path, models_path, trials_path)):
        raise FileNotFoundError("completed result requires provenance, metrics, models, and trials")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if provenance.get("status") != "complete":
        raise ValueError("result provenance is not complete")
    validate_policy("global-polynomial", provenance, provenance.get("manifest_identity"))
    expected_hashes = provenance.get("result_hashes")
    if not isinstance(expected_hashes, dict) or set(expected_hashes) != {"models.npz", "metrics.json"}:
        raise ValueError("result hashes must declare exactly models.npz and metrics.json")
    try:
        verify_files(results_path, expected_hashes, required=("models.npz", "metrics.json"))
    except ArtifactIntegrityError as error:
        raise ValueError("result hashes do not match provenance") from error
    if _without_historical_paths(provenance.get("manifest_identity")) != _without_historical_paths(source.manifest):
        raise ValueError("result manifest identity does not match local immutable input")
    expected_code = provenance.get("code_hashes", {})
    for filename, digest in expected_code.items():
        local = Path(__file__).with_name(filename)
        if not local.is_file() or sha256(local) != digest:
            raise ValueError(f"local {filename} does not match completed result provenance")
    if provenance.get("fingerprint") != metrics.get("fingerprint"):
        raise ValueError("metrics fingerprint does not match provenance")
    return source, provenance, metrics


def reconstruct_predictions(
    standardized: np.ndarray, available: np.ndarray, indices: np.ndarray, matrices: dict[str, np.ndarray], degree: int
) -> np.ndarray:
    """Apply exported matrices to one-step histories (missing values stay zero)."""
    lag = matrices["lag_days"]
    if not np.array_equal(lag, np.arange(1, len(lag) + 1)):
        raise ValueError("lag_days must be ordered one through lag")
    positions = indices[:, None] - lag[None, :]
    values = standardized[positions].transpose(0, 2, 1)
    masks = available[positions].astype(float).transpose(0, 2, 1)
    values = np.where(masks.astype(bool), values, 0.0)
    w1 = matrices[f"degree_{degree}_W1"]
    prediction = np.einsum("ovl,bvl->bo", w1[:, :14], values, optimize=True)
    prediction += np.einsum("ovl,bvl->bo", w1[:, 14:], masks, optimize=True)
    for power in range(2, degree + 1):
        prediction += np.einsum("ovl,bvl->bo", matrices[f"degree_{degree}_W{power}"], values**power, optimize=True)
    if not np.isfinite(prediction).all():
        raise FloatingPointError("reconstructed predictions are non-finite")
    return prediction


def _score(prediction: np.ndarray, target: np.ndarray, support: np.ndarray, activity: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("all", "low", "moderate", "high"):
        date_mask = group_mask(activity, name)
        selected = support & date_mask[:, None]
        counts = selected.sum(axis=0)
        squared = np.where(selected, (prediction - target) ** 2, 0.0).sum(axis=0)
        per = np.divide(squared, counts, out=np.zeros(DENSITY_COUNT), where=counts > 0)
        total = int(counts.sum())
        mse = float(squared.sum() / total) if total else None
        result[name] = {
            "dates": int(date_mask.sum()), "support": counts.tolist(), "pooled_support": total,
            "pooled_standardized_mse": mse,
            "per_altitude_standardized_mse": [float(value) if count else None for value, count in zip(per, counts, strict=True)],
        }
    return result


def _assert_stored_scores(calculated: dict[str, Any], stored: dict[str, Any], label: str) -> None:
    for support_name in ("all_available", "same_support_with_persistence"):
        expected = stored[support_name]["polynomial"]
        observed = calculated[support_name]
        for group in ("all", "low", "moderate", "high"):
            if observed[group]["support"] != expected[group]["support"]:
                raise ValueError(f"{label} {support_name} {group} support differs from stored metrics")
            for key in ("pooled_standardized_mse", "per_altitude_standardized_mse"):
                left = np.asarray(observed[group][key], dtype=float)
                right = np.asarray(expected[group][key], dtype=float)
                if not np.allclose(left, right, rtol=1e-4, atol=1e-7, equal_nan=True):
                    raise ValueError(f"{label} {support_name} {group} {key} differs from stored metrics")


def bootstrap_improvement(
    linear_errors: np.ndarray, nonlinear_errors: np.ndarray, support: np.ndarray, block_length: int,
    *, seed: int = BOOTSTRAP_SEED, replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Joint-date moving-block bootstrap from per-date error sums and counts."""
    if block_length < 1 or replicates < 1:
        raise ValueError("block length and replicates must be positive")
    if linear_errors.shape != nonlinear_errors.shape or support.shape != linear_errors.shape:
        raise ValueError("errors and support must have equal date-by-altitude shapes")
    dates = len(support)
    if dates == 0:
        return {"interval_95_percent": None, "unsupported_replicates": replicates}
    linear_sums = np.where(support, linear_errors, 0.0).sum(axis=1)
    nonlinear_sums = np.where(support, nonlinear_errors, 0.0).sum(axis=1)
    counts = support.sum(axis=1)
    rng = np.random.default_rng(seed + block_length)
    draws_per_replicate = (dates + block_length - 1) // block_length
    starts = rng.integers(0, dates - block_length + 1 if block_length <= dates else 1, size=(replicates, draws_per_replicate))
    offsets = np.arange(block_length)
    sampled = (starts[:, :, None] + offsets).reshape(replicates, -1)[:, :dates]
    sampled = np.minimum(sampled, dates - 1)  # permits the declared block length when it exceeds the window
    denominator = counts[sampled].sum(axis=1)
    linear_mse = np.divide(linear_sums[sampled].sum(axis=1), denominator, out=np.full(replicates, np.nan), where=denominator > 0)
    nonlinear_mse = np.divide(nonlinear_sums[sampled].sum(axis=1), denominator, out=np.full(replicates, np.nan), where=denominator > 0)
    improvement = 100.0 * (1.0 - nonlinear_mse / linear_mse)
    valid = np.isfinite(improvement)
    return {
        "interval_95_percent": [float(item) for item in np.percentile(improvement[valid], [2.5, 97.5])] if valid.any() else None,
        "unsupported_replicates": int((~valid).sum()),
    }


def run(manifest_path: Path, results_path: Path, output_path: Path) -> Path:
    """Verify, reconstruct, compare, and write a new derived-only JSON artifact."""
    started = time.monotonic()
    source, provenance, metrics = verify_artifacts(manifest_path, results_path)
    lag = int(provenance["config"]["lag"])
    target_rows = len(source.values) - lag
    bounds = split_bounds(target_rows)
    indices = {name: np.arange(lag + begin, lag + end) for name, (begin, end) in bounds.items()}
    with np.load(results_path / "models.npz", allow_pickle=False) as saved:
        matrices = {name: np.asarray(saved[name]) for name in saved.files}
    if matrices["W1"].shape != (10, 28, lag) or matrices["W2"].shape != (10, 14, lag) or matrices["W3"].shape != (10, 14, lag):
        raise ValueError("exported matrix shapes do not match the 10-output, 14-predictor contract")
    if [str(item) for item in matrices["node_names"].tolist()] != source.node_names or not np.array_equal(matrices["target_indices_test"], indices["test"]):
        raise ValueError("saved node names or test indices do not match immutable input")
    means, stds = matrices["scaling_means"], matrices["scaling_stds"]
    if means.shape != (14,) or stds.shape != (14,) or not np.isfinite(means).all() or not np.isfinite(stds).all() or (stds <= 0).any():
        raise ValueError("invalid archived train-only scaling")
    standardized = np.where(source.available, (source.values - means) / stds, 0.0)
    predictions = {degree: reconstruct_predictions(standardized, source.available, indices["test"], matrices, degree) for degree in (1, 2, 3)}
    for degree in (1, 2, 3):
        validation_prediction = reconstruct_predictions(standardized, source.available, indices["validation"], matrices, degree)
        for split, prediction in (("validation", validation_prediction), ("test", predictions[degree])):
            selected = indices[split]
            target, activity = standardized[selected, :10], source.activity_f107_center81[selected]
            available = source.available[selected, :10]
            persistence = available & source.available[selected - 1, :10]
            calculated = {"all_available": _score(prediction, target, available, activity), "same_support_with_persistence": _score(prediction, target, persistence, activity)}
            _assert_stored_scores(calculated, metrics["test"][f"{split}_by_degree"][str(degree)], f"degree {degree} {split}")
    target = standardized[indices["test"], :10]
    activity = source.activity_f107_center81[indices["test"]]
    common = source.available[indices["test"], :10] & np.logical_and.reduce([np.isfinite(prediction) for prediction in predictions.values()])
    persistence_support = common & source.available[indices["test"] - 1, :10]
    errors = {degree: (prediction - target) ** 2 for degree, prediction in predictions.items()}
    report: dict[str, Any] = {"schema_version": 1, "input_manifest_sha256": sha256(manifest_path), "input_core_sha256": source.manifest["core_sha256"], "result_hashes": provenance["result_hashes"], "settings": {"lag_days": lag, "bootstrap": {"scope": "pooled only; grouped results are point estimates", "method": "paired calendar-date moving-block bootstrap, sampled jointly across ten altitudes", "block_lengths_days": list(BLOCK_LENGTHS), "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES}}, "test_window": {"start": str(source.timestamps[indices["test"][0]]), "end": str(source.timestamps[indices["test"][-1]]), "dates": len(indices["test"])}, "support": {"common_model_cells": int(common.sum()), "persistence_comparable_cells": int(persistence_support.sum())}}
    report["models"] = {str(degree): _score(predictions[degree], target, persistence_support, activity) for degree in (1, 2, 3)}
    report["persistence"] = _score(standardized[indices["test"] - 1, :10], target, persistence_support, activity)
    report["improvement_percent_vs_degree_1"] = {str(degree): {group: None if report["models"]["1"][group]["pooled_standardized_mse"] in (None, 0) or report["models"][str(degree)][group]["pooled_standardized_mse"] is None else 100.0 * (1.0 - report["models"][str(degree)][group]["pooled_standardized_mse"] / report["models"]["1"][group]["pooled_standardized_mse"]) for group in ("all", "low", "moderate", "high")} for degree in (2, 3)}
    report["bootstrap_95_percent_intervals"] = {str(degree): {str(block): bootstrap_improvement(errors[1], errors[degree], persistence_support, block) for block in BLOCK_LENGTHS} for degree in (2, 3)}
    report["comparison_code_sha256"] = sha256(Path(__file__))
    report["interpretation"] = "Descriptive, conditional on the validation-selected models and this one test period; intervals are not formal cross-regime or multiple-comparison-adjusted inference. Predictive coefficients and differences are not causal."
    report["runtime_seconds"] = time.monotonic() - started
    publish_json_new(output_path, report)
    return output_path


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--results", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    return command


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    print(run(args.manifest, args.results, args.output))


if __name__ == "__main__":
    main()
