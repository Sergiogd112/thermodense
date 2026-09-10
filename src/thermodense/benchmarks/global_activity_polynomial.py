"""Resumable pooled polynomial lag benchmark for global-mean density.

This is a predictive sensitivity benchmark.  Its fitted weights are not causal
coefficients: all ten density targets are pooled into one multi-output fit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from filelock import FileLock, Timeout

VERSION = "global-activity-polynomial-v1"
DENSITY_COUNT = 10
PREDICTOR_COUNT = 14
DEFAULT_LAMBDAS = "0.0001,0.001,0.01"


def _atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **cast(dict[str, Any], arrays))
    temporary.replace(path)


def _atomic_torch(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_hash(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _trial_epoch_seed(seed: int, degree: int, ridge_index: int, epoch: int) -> int:
    """Stable independent shuffle seed, so resumed epochs match uninterrupted ones."""
    material = f"{seed}:{degree}:{ridge_index}:{epoch}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little")


def _trial_fingerprint(
    benchmark_fingerprint: str,
    degree: int,
    ridge: float,
    value_scales: np.ndarray,
    mask_scales: np.ndarray,
) -> str:
    """Bind a checkpoint to its exact trial and training-derived feature scales."""
    material = {
        "benchmark_fingerprint": benchmark_fingerprint,
        "degree": degree,
        "lambda": ridge,
        "value_scales_sha256": _sha256_bytes(value_scales.tobytes()),
        "mask_scales_sha256": _sha256_bytes(mask_scales.tobytes()),
    }
    return _sha256_bytes(json.dumps(material, sort_keys=True).encode())


def split_bounds(target_rows: int) -> dict[str, tuple[int, int]]:
    """Return chronological 70/15/15 bounds over *target dates*."""
    train_end, validation_end = int(target_rows * 0.70), int(target_rows * 0.85)
    if train_end < 1 or validation_end <= train_end or validation_end >= target_rows:
        raise ValueError("too few target dates for chronological 70/15/15 splits")
    return {"train": (0, train_end), "validation": (train_end, validation_end), "test": (validation_end, target_rows)}


@dataclass(frozen=True)
class Prepared:
    standardized: np.ndarray
    available: np.ndarray
    means: np.ndarray
    stds: np.ndarray
    target_indices: dict[str, np.ndarray]


def prepare(values: np.ndarray, available: np.ndarray, lag: int) -> Prepared:
    """Scale from the train timeline only and retain the full daily calendar."""
    values = np.asarray(values, dtype=float)
    available = np.asarray(available, dtype=bool) & np.isfinite(values)
    if values.ndim != 2 or values.shape[1] != PREDICTOR_COUNT or available.shape != values.shape:
        raise ValueError("expected daily values and availability shaped T x 14")
    target_rows = len(values) - lag
    bounds = split_bounds(target_rows)
    target_indices = {name: np.arange(lag + begin, lag + end) for name, (begin, end) in bounds.items()}
    # Every historic predictor used by a training target, but no validation target
    # date, may inform scaling.  Missing values become zero only after this step.
    training_timeline = values[: lag + bounds["train"][1]]
    observed = available[: lag + bounds["train"][1]]
    counts = observed.sum(axis=0)
    means = np.divide(np.where(observed, training_timeline, 0.0).sum(axis=0), counts, out=np.zeros(PREDICTOR_COUNT), where=counts > 0)
    centered = np.where(observed, training_timeline - means, 0.0)
    stds = np.sqrt(np.divide((centered * centered).sum(axis=0), counts, out=np.ones(PREDICTOR_COUNT), where=counts > 0))
    stds[~np.isfinite(stds) | (stds == 0)] = 1.0
    standardized = np.where(available, (values - means) / stds, 0.0).astype(np.float32)
    return Prepared(standardized, available, means, stds, target_indices)


def batch_lags(values: torch.Tensor, availability: torch.Tensor, target_indices: torch.Tensor, lag: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather strictly prior dates; gaps stay gaps rather than becoming adjacent."""
    offsets = torch.arange(1, lag + 1, device=target_indices.device)
    positions = target_indices[:, None] - offsets[None, :]
    return values[positions].transpose(1, 2), availability[positions].transpose(1, 2)


def feature_scales(values: np.ndarray, availability: np.ndarray, train_indices: np.ndarray, lag: int, degree: int) -> tuple[np.ndarray, np.ndarray]:
    """Training-only RMS scales used internally by Adam, not by exported weights."""
    dimension = np.sqrt(lag * (degree * PREDICTOR_COUNT + PREDICTOR_COUNT))
    value_scales = np.empty((degree, PREDICTOR_COUNT, lag), dtype=np.float32)
    for offset in range(1, lag + 1):
        x = values[train_indices - offset].astype(np.float64)
        for power in range(1, degree + 1):
            value_scales[power - 1, :, offset - 1] = np.maximum(np.sqrt(np.mean(x ** (2 * power), axis=0)), 1.0) * dimension
    return value_scales, np.full((PREDICTOR_COUNT, lag), dimension, dtype=np.float32)


class PolynomialVAR(torch.nn.Module):
    """Additive coordinatewise X, X², X³ predictor-history model; no interactions."""

    value_scales: torch.Tensor
    mask_scales: torch.Tensor

    def __init__(self, degree: int, lag: int, value_scales: torch.Tensor, mask_scales: torch.Tensor) -> None:
        super().__init__()
        self.degree = degree
        self.register_buffer("value_scales", value_scales)
        self.register_buffer("mask_scales", mask_scales)
        self.value_scales = value_scales
        self.mask_scales = mask_scales
        self.value_weights = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(DENSITY_COUNT, PREDICTOR_COUNT, lag)) for _ in range(degree)])
        self.mask_weights = torch.nn.Parameter(torch.zeros(DENSITY_COUNT, PREDICTOR_COUNT, lag))

    def forward(self, value_lags: torch.Tensor, mask_lags: torch.Tensor) -> torch.Tensor:
        result = torch.einsum("ovl,bvl->bo", self.mask_weights, mask_lags / self.mask_scales)
        for power, weights in enumerate(self.value_weights, start=1):
            result = result + torch.einsum("ovl,bvl->bo", weights, value_lags.pow(power) / self.value_scales[power - 1])
        return result

    def ridge_penalty(self) -> torch.Tensor:
        return (self.mask_weights / self.mask_scales).square().sum() + sum((weight / self.value_scales[power]).square().sum() for power, weight in enumerate(self.value_weights))

    def exported(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        w1 = np.concatenate(((self.value_weights[0] / self.value_scales[0]).detach().numpy(), (self.mask_weights / self.mask_scales).detach().numpy()), axis=1)
        zeros = np.zeros((DENSITY_COUNT, PREDICTOR_COUNT, self.mask_weights.shape[-1]), dtype=np.float32)
        w2 = (self.value_weights[1] / self.value_scales[1]).detach().numpy() if self.degree >= 2 else zeros
        w3 = (self.value_weights[2] / self.value_scales[2]).detach().numpy() if self.degree >= 3 else zeros
        return w1, w2, w3


def _finite(label: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"non-finite {label}")


def _predict(model: PolynomialVAR, values: torch.Tensor, availability: torch.Tensor, indices: np.ndarray, lag: int, batch_size: int) -> np.ndarray:
    output: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            selected = torch.as_tensor(indices[start : start + batch_size])
            x, masks = batch_lags(values, availability, selected, lag)
            prediction = model(x, masks)
            _finite("prediction", prediction)
            output.append(prediction.numpy())
    return np.concatenate(output, axis=0)


def _scores(prediction: np.ndarray, target: np.ndarray, target_available: np.ndarray, activity: np.ndarray, group_mask: Any, extra_support: np.ndarray | None = None) -> dict[str, Any]:
    support = target_available if extra_support is None else target_available & extra_support
    groups: dict[str, Any] = {}
    for group in ("all", "low", "moderate", "high"):
        dates = np.ones(len(target), dtype=bool) if group == "all" else np.asarray(group_mask(activity, group), dtype=bool)
        counts = (support & dates[:, None]).sum(axis=0)
        squared = np.where(support & dates[:, None], (prediction - target) ** 2, 0.0).sum(axis=0)
        per_mse = np.divide(squared, counts, out=np.zeros(DENSITY_COUNT), where=counts > 0)
        pooled_mse = float(squared.sum() / counts.sum()) if counts.sum() else None
        per_mse_json = [float(value) if count else None for value, count in zip(per_mse, counts, strict=True)]
        groups[group] = {"dates": int(dates.sum()), "support": counts.tolist(), "pooled_support": int(counts.sum()), "pooled_standardized_mse": pooled_mse, "pooled_standardized_rmse": float(np.sqrt(pooled_mse)) if pooled_mse is not None else None, "per_altitude_standardized_mse": per_mse_json, "per_altitude_standardized_rmse": [float(np.sqrt(value)) if value is not None else None for value in per_mse_json]}
    groups["unknown_activity_dates_excluded_from_grouped_summaries"] = int((~np.isfinite(activity)).sum())
    return groups


def evaluate_models(model_prediction: np.ndarray, standardized: np.ndarray, available: np.ndarray, indices: np.ndarray, activity: np.ndarray, train_target_mean: np.ndarray, group_mask: Any) -> dict[str, Any]:
    """Score all models and their identical persistence-comparable support."""
    target = standardized[indices, :DENSITY_COUNT]
    target_available = available[indices, :DENSITY_COUNT]
    persistence_available = target_available & available[indices - 1, :DENSITY_COUNT]
    persistence = standardized[indices - 1, :DENSITY_COUNT]
    mean_prediction = np.broadcast_to(train_target_mean, target.shape)
    common = persistence_available
    return {
        "all_available": {
            "polynomial": _scores(model_prediction, target, target_available, activity, group_mask),
            "persistence": _scores(persistence, target, persistence_available, activity, group_mask),
            "training_mean": _scores(mean_prediction, target, target_available, activity, group_mask),
        },
        "same_support_with_persistence": {
            "polynomial": _scores(model_prediction, target, target_available, activity, group_mask, common),
            "persistence": _scores(persistence, target, persistence_available, activity, group_mask, common),
            "training_mean": _scores(mean_prediction, target, target_available, activity, group_mask, common),
        },
    }


def _code_hashes() -> dict[str, str]:
    input_module = Path(__file__).with_name("global_activity_input.py")
    code_hashes = {"global_activity_polynomial.py": _file_hash(Path(__file__))}
    if input_module.exists():
        code_hashes[input_module.name] = _file_hash(input_module)
    return code_hashes


def _identity(manifest_path: Path, manifest: dict[str, Any], config: dict[str, Any]) -> str:
    material = {"version": VERSION, "manifest_path": str(manifest_path.resolve()), "manifest": manifest, "config": config, "code_hashes": _code_hashes(), "python": sys.version, "torch": torch.__version__, "numpy": np.__version__}
    return _sha256_bytes(json.dumps(material, sort_keys=True, default=str).encode())


def _load_input(manifest_path: Path) -> Any:
    from thermodense.benchmarks.global_activity_input import load_manifest

    return load_manifest(manifest_path)


def run(args: argparse.Namespace) -> Path:
    """Run under a non-blocking output lock to prevent concurrent writers."""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(args.output.parent / f".{args.output.name}.lock"))
    try:
        with lock.acquire(timeout=0):
            return _run_locked(args)
    except Timeout as error:
        raise RuntimeError("another global activity polynomial run holds this output") from error


def _run_locked(args: argparse.Namespace) -> Path:
    if args.lag < 1 or args.epochs < 1 or args.batch_size < 1 or args.threads < 1 or args.threads > 2:
        raise ValueError("lag, epochs, and batch-size must be positive; threads must be 1 or 2")
    lambdas = [float(value) for value in args.lambdas.split(",") if value.strip()]
    if not lambdas or any(not np.isfinite(value) or value < 0 for value in lambdas):
        raise ValueError("--lambdas must be finite non-negative comma-separated values")
    source = _load_input(args.manifest)
    if list(source.node_names)[DENSITY_COUNT:] != ["F10.7_OBS", "AP_AVG", "KP_SUM", "CO2_ppm"]:
        raise ValueError("input must retain raw F10.7_OBS as predictor 11")
    if len(source.timestamps) != len(source.values) or not np.array_equal(np.diff(source.timestamps.astype("datetime64[D]").astype(np.int64)), np.ones(len(source.timestamps) - 1, dtype=np.int64)):
        raise ValueError("input must contain the full consecutive daily calendar")
    config = {"lag": args.lag, "epochs": args.epochs, "batch_size": args.batch_size, "lambdas": lambdas, "seed": args.seed, "threads": args.threads, "optimizer": "Adam minibatch ridge with training-only RMS feature normalization"}
    fingerprint = _identity(args.manifest, source.manifest, config)
    output = args.output
    if output.exists() and not (output / "provenance.json").exists():
        if any(path.name != ".lock" for path in output.iterdir()):
            raise FileExistsError("refusing nonempty output without benchmark provenance")
    output.mkdir(parents=True, exist_ok=True)  # reserve the requested output before fitting
    provenance_path, models_path = output / "provenance.json", output / "models.npz"
    if provenance_path.exists():
        prior = json.loads(provenance_path.read_text())
        if prior.get("fingerprint") != fingerprint:
            raise FileExistsError("output belongs to an incompatible benchmark identity")
        if prior.get("status") == "complete":
            hashes = prior.get("result_hashes", {})
            if not models_path.is_file() or not (output / "metrics.json").is_file():
                raise FileExistsError("complete output lacks models or metrics")
            if hashes != {"models.npz": _file_hash(models_path), "metrics.json": _file_hash(output / "metrics.json")}:
                raise FileExistsError("complete output result hashes do not match")
            return models_path
    _atomic_json({"status": "running", "fingerprint": fingerprint, "config": config, "code_hashes": _code_hashes(), "optimization_status": "bounded epoch budget; neither convergence nor globally optimal ridge is claimed", "interpretation": "predictive sensitivity benchmark; coefficients are not causal", "formula": "pooled density targets = additive coordinatewise predictor-history powers X, X², X³ plus linear availability indicators; no interactions", "runtime_limit": "CPU threads capped at 2; minibatches avoid a dense lag design or normal-equations inversion"}, provenance_path)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    prepared = prepare(source.values, source.available, args.lag)
    values = torch.from_numpy(prepared.standardized)
    availability = torch.from_numpy(prepared.available.astype(np.float32))
    train_targets = prepared.target_indices["train"]
    train_target_counts = prepared.available[train_targets, :DENSITY_COUNT].sum(axis=0)
    if (train_target_counts == 0).any():
        missing = np.asarray(source.node_names[:DENSITY_COUNT])[train_target_counts == 0]
        raise ValueError(f"no training target observations for: {', '.join(missing)}")
    train_mean = np.divide((prepared.standardized[train_targets, :DENSITY_COUNT] * prepared.available[train_targets, :DENSITY_COUNT]).sum(axis=0), train_target_counts)
    trials: list[dict[str, Any]] = []
    for degree in (1, 2, 3):
        scales, mask_scales = feature_scales(prepared.standardized, prepared.available, train_targets, args.lag, degree)
        print(f"degree {degree}/3: starting", flush=True)
        for ridge_index, ridge in enumerate(lambdas):
            trial_fingerprint = _trial_fingerprint(fingerprint, degree, ridge, scales, mask_scales)
            key = f"degree_{degree}_lambda_{ridge_index}_{trial_fingerprint[:12]}"
            checkpoint = output / "checkpoints" / f"{key}.pt"
            model = PolynomialVAR(degree, args.lag, torch.from_numpy(scales), torch.from_numpy(mask_scales))
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            history: list[dict[str, float]] = []
            start_epoch = 0
            best_validation = float("inf")
            best_state: dict[str, torch.Tensor] | None = None
            if checkpoint.exists():
                saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                if saved.get("fingerprint") != fingerprint or saved.get("checkpoint_fingerprint") != trial_fingerprint:
                    raise FileExistsError("incompatible trial checkpoint")
                model.load_state_dict(saved["model"])
                optimizer.load_state_dict(saved["optimizer"])
                history, start_epoch = saved["history"], saved["epoch"]
                best_validation, best_state = saved["best_validation"], saved["best_model"]
            for epoch in range(start_epoch, args.epochs):
                model.train()
                order = np.random.default_rng(_trial_epoch_seed(args.seed, degree, ridge_index, epoch)).permutation(train_targets)
                losses: list[float] = []
                for start in range(0, len(order), args.batch_size):
                    selected = torch.as_tensor(order[start : start + args.batch_size])
                    x, masks = batch_lags(values, availability, selected, args.lag)
                    target_mask = availability[selected, :DENSITY_COUNT]
                    prediction = model(x, masks)
                    loss = ((prediction - values[selected, :DENSITY_COUNT]).square() * target_mask).sum() / target_mask.sum().clamp_min(1)
                    objective = loss + ridge * model.ridge_penalty()
                    _finite("objective", objective)
                    optimizer.zero_grad()
                    objective.backward()
                    for parameter in model.parameters():
                        gradient = parameter.grad
                        if gradient is None:
                            raise RuntimeError("missing gradient")
                        _finite("gradient", gradient)
                        _finite("parameter", parameter)
                    optimizer.step()
                    for parameter in model.parameters():
                        _finite("updated parameter", parameter)
                    losses.append(float(loss.detach()))
                validation_prediction = _predict(model, values, availability, prepared.target_indices["validation"], args.lag, args.batch_size)
                validation_metrics = evaluate_models(validation_prediction, prepared.standardized, prepared.available, prepared.target_indices["validation"], np.asarray(source.activity_f107_center81)[prepared.target_indices["validation"]], train_mean, _group_mask())
                validation_mse = validation_metrics["all_available"]["polynomial"]["all"]["pooled_standardized_mse"]
                if validation_mse is None or not np.isfinite(validation_mse):
                    raise FloatingPointError("non-finite validation MSE")
                history.append({"epoch": epoch + 1, "train_masked_mse": float(np.mean(losses)), "validation_mse": validation_mse})
                if validation_mse < best_validation:
                    best_validation = validation_mse
                    best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
                _atomic_torch({"fingerprint": fingerprint, "checkpoint_fingerprint": trial_fingerprint, "epoch": epoch + 1, "history": history, "model": model.state_dict(), "best_model": best_state, "best_validation": best_validation, "optimizer": optimizer.state_dict()}, checkpoint)
                print(f"degree {degree}/3 lambda {ridge:g} epoch {epoch + 1}/{args.epochs}: validation MSE {validation_mse:.6g}", flush=True)
            if best_state is None:
                raise RuntimeError("trial has no validation checkpoint")
            trial = {"degree": degree, "lambda": ridge, "trial_fingerprint": trial_fingerprint, "validation_mse": best_validation, "epochs_completed": len(history), "optimization_status": "epoch_budget_exhausted", "checkpoint": str(checkpoint), "history": history}
            trials.append(trial)
            _atomic_json({"fingerprint": fingerprint, "trials": trials}, output / "trials.json")
            print(f"degree {degree}/3 lambda {ridge:g}: retained best validation MSE {best_validation:.6g}", flush=True)
    test_indices = prepared.target_indices["test"]
    degree_winners = {degree: min((trial for trial in trials if trial["degree"] == degree), key=lambda item: float(item["validation_mse"])) for degree in (1, 2, 3)}
    test_by_degree: dict[str, Any] = {}
    validation_by_degree: dict[str, Any] = {}
    fitted_models: dict[int, PolynomialVAR] = {}
    for degree, degree_winner in degree_winners.items():
        saved = torch.load(Path(str(degree_winner["checkpoint"])), map_location="cpu", weights_only=False)
        if saved.get("fingerprint") != fingerprint or saved.get("checkpoint_fingerprint") != degree_winner["trial_fingerprint"]:
            raise FileExistsError("selected trial checkpoint is incompatible")
        scales, mask_scales = feature_scales(prepared.standardized, prepared.available, train_targets, args.lag, degree)
        model = PolynomialVAR(degree, args.lag, torch.from_numpy(scales), torch.from_numpy(mask_scales))
        model.load_state_dict(saved["best_model"])
        fitted_models[degree] = model
        validation_prediction = _predict(model, values, availability, prepared.target_indices["validation"], args.lag, args.batch_size)
        validation_by_degree[str(degree)] = evaluate_models(validation_prediction, prepared.standardized, prepared.available, prepared.target_indices["validation"], np.asarray(source.activity_f107_center81)[prepared.target_indices["validation"]], train_mean, _group_mask())
        prediction = _predict(model, values, availability, test_indices, args.lag, args.batch_size)
        test_by_degree[str(degree)] = evaluate_models(prediction, prepared.standardized, prepared.available, test_indices, np.asarray(source.activity_f107_center81)[test_indices], train_mean, _group_mask())
    winner = min(degree_winners.values(), key=lambda item: float(item["validation_mse"]))
    best_model = fitted_models[int(winner["degree"])]
    metrics = {"validation_by_degree": validation_by_degree, "test_by_degree": test_by_degree, "selected_degree": int(winner["degree"]), "selected_validation": validation_by_degree[str(winner["degree"])], "selected_test": test_by_degree[str(winner["degree"])]}
    w1, w2, w3 = best_model.exported()
    _atomic_json({"fingerprint": fingerprint, "winner": winner, "test": metrics}, output / "metrics.json")
    coefficients: dict[str, np.ndarray] = {"W1": w1, "W2": w2, "W3": w3}
    for degree, model in fitted_models.items():
        for power, weights in enumerate(model.exported(), start=1):
            coefficients[f"degree_{degree}_W{power}"] = weights
    _atomic_npz(models_path, **coefficients, node_names=np.asarray(source.node_names), density_target_names=np.asarray(source.node_names[:DENSITY_COUNT]), scaling_means=prepared.means, scaling_stds=prepared.stds, target_indices_test=test_indices, lag_days=np.arange(1, args.lag + 1))
    result_hashes = {"models.npz": _file_hash(models_path), "metrics.json": _file_hash(output / "metrics.json")}
    _atomic_json({"status": "complete", "fingerprint": fingerprint, "config": config, "code_hashes": _code_hashes(), "manifest_identity": source.manifest, "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__}, "winner": winner, "test_metrics": "metrics.json", "models": "models.npz", "result_hashes": result_hashes, "optimization_status": "bounded epoch budget; neither convergence nor globally optimal ridge is claimed", "evaluation_activity": "centered81 only; low <100, moderate 100<=F<160, high >=160; unknown excluded from grouped summaries", "interpretation": "predictive sensitivity benchmark; coefficients are not causal"}, provenance_path)
    return models_path


def _group_mask() -> Any:
    from thermodense.benchmarks.global_activity_input import group_mask

    return group_mask


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--lag", type=int, default=180)
    result.add_argument("--epochs", type=int, default=15)
    result.add_argument("--threads", type=int, default=2)
    result.add_argument("--batch-size", type=int, default=64)
    result.add_argument("--lambdas", default=DEFAULT_LAMBDAS)
    result.add_argument("--seed", type=int, default=0)
    return result


def main(argv: list[str] | None = None) -> None:
    print(run(parser().parse_args(argv)))


if __name__ == "__main__":
    main()
