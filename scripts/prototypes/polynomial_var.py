"""THROWAWAY resumable matrix-free polynomial VAR feasibility prototype.

This script is deliberately isolated from production modules, schedulers, PCMCI
scripts, and SABER artifacts.  It consumes the immutable exact inputs read-only.
It fits a ridge-penalized Adam approximation, not a closed-form ridge solution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from filelock import FileLock, Timeout

VERSION = "throwaway-polynomial-var-v3"
PREPARED_VERSION = "throwaway-polynomial-var-v1"
MAX_LAG = 183 * 8
MAX_THREADS = 10
DIVERGENCE_FACTOR = 100.0
ALTITUDES = tuple(range(175, 826, 25))
SABER_CHANNELS = ("CO2cool", "NOcool", "O2_1delta_ver", "OH_16_ver", "OH_20_ver")
BUNDLE = Path("outputs/prototypes/density_pcmci_3hour_and_daily/analysis_bundle.npz")
STREAM = Path("outputs/prototypes/saber_vertical_totals_40x60_stream/final_arrays.npz")
REPORT = Path("outputs/prototypes/saber_vertical_totals_40x60_stream/final_report.json")
CO2 = Path("data/original/co2/co2_daily_mlo.csv")
DEFAULT_OUTPUT = Path("outputs/prototypes/polynomial_var_throwaway")
DEFAULT_FIT_OUTPUT = DEFAULT_OUTPUT / "fit"
DEFAULT_BENCHMARK_OUTPUT = DEFAULT_OUTPUT / "benchmark"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def atomic_torch(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def channel_names() -> list[str]:
    return [
        *[f"log10rho_{altitude}km" for altitude in ALTITUDES],
        "F10.7_raw",
        "Ap",
        "Kp",
        "co2_mlo_ppm",
        *SABER_CHANNELS,
    ]


def load_co2_daily_hold(times: np.ndarray, path: Path = CO2) -> np.ndarray:
    """Return raw daily MLO values repeated over each UTC day's eight slots."""
    frame = pd.read_csv(
        path, comment="#", header=None, names=["year", "month", "day", "decimal", "ppm"]
    )
    dates = pd.to_datetime(frame[["year", "month", "day"]])
    daily = (
        pd.Series(frame["ppm"].where(frame["ppm"] >= 0).to_numpy(float), index=dates)
        .groupby(level=0)
        .last()
    )
    slots = pd.DatetimeIndex(times.astype("datetime64[ns]")).normalize()
    return daily.reindex(slots).to_numpy(float)


def load_exact_sources(
    bundle_path: Path = BUNDLE,
    stream_path: Path = STREAM,
    report_path: Path = REPORT,
    co2_path: Path = CO2,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, str]]:
    """Load only the declared exact HASDM/SABER inputs and reject mismatches."""
    report = json.loads(report_path.read_text())
    if (
        report.get("exact_full_cell") is not True
        or report.get("output_status") != "exact_complete"
    ):
        raise ValueError(
            "refusing SABER report not declared exact_full_cell/exact_complete"
        )
    with (
        np.load(bundle_path, allow_pickle=False) as bundle,
        np.load(stream_path, allow_pickle=False) as stream,
    ):
        required = {
            "hasdm_time_ns",
            "hasdm_targets",
            "hasdm_f107",
            "hasdm_ap",
            "hasdm_kp",
        }
        if required - set(bundle.files):
            raise ValueError("analysis bundle lacks required HASDM/F10.7/Ap/Kp inputs")
        if tuple(stream["channels"].tolist()) != SABER_CHANNELS:
            raise ValueError(
                "SABER channel names do not exactly match the five required channels"
            )
        times = np.asarray(bundle["hasdm_time_ns"], dtype=np.int64)
        saber_times = stream["timestamps"].astype("datetime64[ns]").astype(np.int64)
        if not np.array_equal(times, saber_times):
            raise ValueError(
                "SABER timestamps do not exactly match analysis bundle HASDM timestamps"
            )
        density = np.asarray(bundle["hasdm_targets"], dtype=float)
        saber = np.asarray(stream["values"], dtype=float)
        if density.shape != (len(times), 27) or saber.shape != (5, len(times)):
            raise ValueError(
                "exact source dimensions differ from the 27 HASDM / five SABER contract"
            )
        values = np.column_stack(
            (
                density,
                bundle["hasdm_f107"],
                bundle["hasdm_ap"],
                bundle["hasdm_kp"],
                load_co2_daily_hold(times.astype("datetime64[ns]"), co2_path),
                saber.T,
            )
        )
    if values.shape[1] != 36:
        raise AssertionError("36-variable state contract violated")
    return (
        times,
        values,
        channel_names(),
        {str(p): sha256(p) for p in (bundle_path, stream_path, report_path, co2_path)},
    )


def prepare(output: Path = DEFAULT_OUTPUT) -> Path:
    """Write a compact immutable input, provenance, and availability diagnostics."""
    times, values, names, hashes = load_exact_sources()
    input_path, provenance = (
        output / "prepared_input.npz",
        output / "prepared_input.provenance.json",
    )
    identity = {
        "version": PREPARED_VERSION,
        "source_sha256": hashes,
        "channels": names,
    }
    if input_path.exists() or provenance.exists():
        if (
            input_path.exists()
            and provenance.exists()
            and json.loads(provenance.read_text()).get("identity") == identity
            and json.loads(provenance.read_text()).get("prepared_sha256")
            == sha256(input_path)
        ):
            return input_path
        raise FileExistsError(
            "prepared input is immutable and incompatible; choose another output"
        )
    available = np.isfinite(values)
    atomic_npz(
        input_path, time_ns=times, values=values, channel_names=np.asarray(names)
    )
    atomic_json(
        {
            "prototype": "THROWAWAY; never production",
            "identity": identity,
            "prepared_sha256": sha256(input_path),
            "shape": list(values.shape),
            "cadence": "3-hour",
            "f107_policy": "hasdm_f107: documented 3-hour spline representation of raw daily observations",
            "co2_policy": "raw daily Mauna Loa ppm, negative values missing, repeated over UTC day slots without interpolation",
            "saber_policy": "exact_complete final stream only; exact timestamps and five channels verified",
            "availability": {
                name: int(available[:, i].sum()) for i, name in enumerate(names)
            },
        },
        provenance,
    )
    return input_path


def load_prepared(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    with np.load(path, allow_pickle=False) as data:
        times, values = (
            np.asarray(data["time_ns"], dtype=np.int64),
            np.asarray(data["values"], dtype=float),
        )
        names = [str(x) for x in data["channel_names"].tolist()]
    if (
        values.ndim != 2
        or values.shape[1] != 36
        or len(times) != len(values)
        or names != channel_names()
    ):
        raise ValueError(
            "prepared input is incompatible with the fixed 36-channel state contract"
        )
    return times, values, names, sha256(path)


def split_bounds(rows: int) -> dict[str, tuple[int, int]]:
    train_end, valid_end = int(rows * 0.70), int(rows * 0.85)
    return {
        "train": (0, train_end),
        "validation": (train_end, valid_end),
        "test": (valid_end, rows),
    }


def scaling_and_masks(
    values: np.ndarray, lag: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int | None]]:
    rows = len(values) - lag
    if rows < 3:
        raise ValueError("lag leaves fewer than three rows")
    train_end = split_bounds(rows)["train"][1]
    # Include every timestamp used either as a training target or as one of its
    # lagged predictors, while stopping before the validation target interval.
    training_timeline = values[: lag + train_end]
    observed = np.isfinite(training_timeline)
    channels = values.shape[1]
    means = np.divide(
        np.nansum(training_timeline, axis=0),
        observed.sum(axis=0),
        out=np.zeros(channels),
        where=observed.sum(axis=0) > 0,
    )
    centered = training_timeline - means
    stds = np.sqrt(
        np.divide(
            np.nansum(centered**2, axis=0),
            observed.sum(axis=0),
            out=np.ones(channels),
            where=observed.sum(axis=0) > 0,
        )
    )
    stds[~np.isfinite(stds) | (stds == 0)] = 1.0
    standardized = np.where(np.isfinite(values), (values - means) / stds, 0.0).astype(
        np.float32
    )
    availability = np.isfinite(values)
    patterns, inverse = np.unique(availability.T, axis=0, return_inverse=True)
    nonconstant = [
        i
        for i, pattern in enumerate(patterns)
        if not (pattern.all() or (~pattern).all())
    ]
    remap = {old: new for new, old in enumerate(nonconstant)}
    mapping = [remap.get(int(index)) for index in inverse]
    return standardized, availability, means, stds, mapping


def mask_matrix(availability: np.ndarray, mapping: list[int | None]) -> np.ndarray:
    count = max((x for x in mapping if x is not None), default=-1) + 1
    matrix = np.zeros((len(availability), count), dtype=np.float32)
    for channel, mask_id in enumerate(mapping):
        if mask_id is not None:
            matrix[:, mask_id] = availability[:, channel]
    return matrix


class PolynomialVAR(torch.nn.Module):
    """Matrix-free W1/W2/W3 model; nonlinear mask coefficients do not exist."""

    def __init__(
        self,
        outputs: int,
        values: int,
        masks: int,
        lag: int,
        value_scales: torch.Tensor | None = None,
        mask_scales: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.lag, self.values, self.masks = lag, values, masks
        if value_scales is None:
            value_scales = torch.ones(3, values, lag)
        if mask_scales is None:
            mask_scales = torch.ones(masks, lag)
        if (
            value_scales.shape != (3, values, lag)
            or not torch.isfinite(value_scales).all()
            or (value_scales <= 0).any()
        ):
            raise ValueError(
                "value feature scales must be finite and positive [3, values, lag]"
            )
        if (
            mask_scales.shape != (masks, lag)
            or not torch.isfinite(mask_scales).all()
            or (mask_scales <= 0).any()
        ):
            raise ValueError(
                "mask feature scales must be finite and positive [masks, lag]"
            )
        # Adam operates on RMS- and dimension-normalized feature columns. Dividing
        # learned parameters by these fixed training-only scales recovers exact
        # coefficients for standardized X, X², X³, and linear masks on export.
        self.register_buffer("value_scales", value_scales)
        self.register_buffer("mask_scales", mask_scales)
        self.w1_values = torch.nn.Parameter(torch.zeros(outputs, values, lag))
        self.w1_masks = torch.nn.Parameter(torch.zeros(outputs, masks, lag))
        self.w2_values = torch.nn.Parameter(torch.zeros(outputs, values, lag))
        self.w3_values = torch.nn.Parameter(torch.zeros(outputs, values, lag))

    def forward(
        self, value_lags: torch.Tensor, mask_lags: torch.Tensor
    ) -> torch.Tensor:
        return (
            torch.einsum(
                "ovl,bvl->bo", self.w1_values, value_lags / self.value_scales[0]
            )
            + torch.einsum(
                "ovl,bvl->bo",
                self.w2_values,
                value_lags.square() / self.value_scales[1],
            )
            + torch.einsum(
                "ovl,bvl->bo", self.w3_values, value_lags.pow(3) / self.value_scales[2]
            )
            + torch.einsum("oml,bml->bo", self.w1_masks, mask_lags / self.mask_scales)
        )

    def ridge_penalty(self) -> torch.Tensor:
        """Penalty in exported-coefficient space, not normalized optimizer space."""
        return (
            (self.w1_values / self.value_scales[0]).square().sum()
            + (self.w2_values / self.value_scales[1]).square().sum()
            + (self.w3_values / self.value_scales[2]).square().sum()
            + (self.w1_masks / self.mask_scales).square().sum()
        )

    def exported_value_coefficients(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.w1_values / self.value_scales[0],
            self.w2_values / self.value_scales[1],
            self.w3_values / self.value_scales[2],
        )

    def exported_mask_coefficients(self) -> torch.Tensor:
        return self.w1_masks / self.mask_scales


def batch_lags(
    values: torch.Tensor, masks: torch.Tensor, target_indices: torch.Tensor, lag: int
) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = torch.arange(1, lag + 1, device=target_indices.device)
    positions = target_indices[:, None] - offsets[None, :]
    return values[positions].transpose(1, 2), masks[positions].transpose(1, 2)


def masked_mse(
    prediction: torch.Tensor, target: torch.Tensor, available: torch.Tensor
) -> torch.Tensor:
    return (
        (prediction - target).square() * available
    ).sum() / available.sum().clamp_min(1)


def fingerprint(prepared_hash: str, config: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            {"version": VERSION, "prepared_sha256": prepared_hash, "config": config},
            sort_keys=True,
        ).encode()
    ).hexdigest()


def validate_limits(lag: int, threads: int) -> None:
    if lag > MAX_LAG:
        raise ValueError(f"--lag must not exceed {MAX_LAG}")
    if lag < 1:
        raise ValueError("--lag must be positive")
    if threads > MAX_THREADS:
        raise ValueError(f"--threads must not exceed hard cap {MAX_THREADS}")
    if threads < 1:
        raise ValueError("--threads must be positive")


def validate_fit_arguments(args: argparse.Namespace) -> list[float]:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("--learning-rate must be finite and positive")
    lambdas = [float(x) for x in args.lambdas.split(",") if x.strip()]
    if not lambdas or any(not np.isfinite(x) or x < 0 for x in lambdas):
        raise ValueError("--lambdas must contain finite non-negative values")
    return lambdas


def config_from_args(args: argparse.Namespace, prepared_hash: str) -> dict[str, object]:
    lambdas = validate_fit_arguments(args)
    return {
        "prepared_sha256": prepared_hash,
        "lag": args.lag,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lambdas": lambdas,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "threads": args.threads,
        "optimizer": "Adam on training-RMS- and dimension-normalized polynomial features; ridge penalty in exported coefficient space",
    }


def checkpoint_compatible(path: Path, identity: str) -> bool:
    return (
        path.is_file()
        and torch.load(path, map_location="cpu", weights_only=False).get("fingerprint")
        == identity
    )


def feature_scales(
    standardized: np.ndarray,
    masks: np.ndarray,
    train_indices: np.ndarray,
    lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return training-only RMS and dimensional scales for all predictors."""
    # Adam's first update is nearly invariant to an individual feature's scale.
    # The common sqrt(P) factor prevents P correlated O(lr) contributions from
    # making the first prediction O(P*lr), while preserving exact export semantics.
    dimensional_scale = np.sqrt(lag * (3 * standardized.shape[1] + masks.shape[1]))
    scales = np.empty((3, standardized.shape[1], lag), dtype=np.float32)
    for offset in range(1, lag + 1):
        predictors = standardized[train_indices - offset].astype(np.float64)
        for power in range(1, 4):
            rms = np.sqrt(np.mean(np.power(predictors, 2 * power), axis=0))
            scales[power - 1, :, offset - 1] = np.maximum(rms, 1.0) * dimensional_scale
    if not np.isfinite(scales).all():
        raise FloatingPointError("non-finite training feature scale")
    # Binary masks have RMS <= 1. Keeping their floor at one avoids amplifying
    # rare availability transitions relative to ordinary standardized predictors.
    mask_scales = np.full((masks.shape[1], lag), dimensional_scale, dtype=np.float32)
    return scales, mask_scales


def require_finite(label: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"non-finite {label}")


def evaluate(
    model: PolynomialVAR,
    values: torch.Tensor,
    masks: torch.Tensor,
    availability: torch.Tensor,
    indices: np.ndarray,
    lag: int,
    batch_size: int,
) -> tuple[float, list[float]]:
    sums, counts = torch.zeros(36), torch.zeros(36)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            target_i = torch.as_tensor(indices[start : start + batch_size])
            v, m = batch_lags(values, masks, target_i, lag)
            prediction = model(v, m)
            require_finite("evaluation prediction", prediction)
            residual = (prediction - values[target_i]).square() * availability[target_i]
            require_finite("evaluation residual", residual)
            sums += residual.sum(0).cpu()
            counts += availability[target_i].sum(0).cpu()
    per_variable = (sums / counts.clamp_min(1)).tolist()
    return float(sums.sum() / counts.sum().clamp_min(1)), [
        float(x) for x in per_variable
    ]


def evaluate_zero(
    values: torch.Tensor, availability: torch.Tensor, indices: np.ndarray
) -> tuple[float, list[float]]:
    target = values[torch.as_tensor(indices)]
    residual = target.square() * availability[torch.as_tensor(indices)]
    sums, counts = residual.sum(0), availability[torch.as_tensor(indices)].sum(0)
    return float(sums.sum() / counts.sum().clamp_min(1)), [
        float(x) for x in (sums / counts.clamp_min(1)).tolist()
    ]


def fit(args: argparse.Namespace) -> Path:
    validate_limits(args.lag, args.threads)
    torch.set_num_threads(args.threads)
    prepared = Path(args.input)
    _, raw, names, prepared_hash = load_prepared(prepared)
    config = config_from_args(args, prepared_hash)
    identity = fingerprint(prepared_hash, config)
    output = Path(args.output)
    if args.reset and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(output / ".fit.lock"))
    try:
        with lock.acquire(timeout=0):
            existing = output / "fit.provenance.json"
            final = output / "final_model.npz"
            if existing.exists():
                prior = json.loads(existing.read_text())
                if prior.get("fingerprint") != identity:
                    raise FileExistsError(
                        "incompatible fit output; use --reset to recompute"
                    )
                if final.exists():
                    return final
            atomic_json(
                {
                    "prototype": "THROWAWAY; never production",
                    "fingerprint": identity,
                    "config": config,
                    "formula": "standardized X_t = W1 A + W2(value(A)^2) + W3(value(A)^3); no intercept",
                    "optimizer": "Adam on training-RMS- and dimension-normalized features; exported coefficients retain the exact formula",
                    "status": "running",
                },
                existing,
            )
            standardized, available_np, means, stds, mapping = scaling_and_masks(
                raw, args.lag
            )
            masks_np = mask_matrix(available_np, mapping)
            values, masks, available = (
                torch.from_numpy(standardized),
                torch.from_numpy(masks_np),
                torch.from_numpy(available_np.astype(np.float32)),
            )
            splits = split_bounds(len(raw) - args.lag)
            target_sets = {
                key: np.arange(args.lag + begin, args.lag + end)
                for key, (begin, end) in splits.items()
            }
            scales_np, mask_scales_np = feature_scales(
                standardized, masks_np, target_sets["train"], args.lag
            )
            value_scales = torch.from_numpy(scales_np)
            mask_scales = torch.from_numpy(mask_scales_np)
            zero_validation, zero_validation_per = evaluate_zero(
                values, available, target_sets["validation"]
            )
            if not np.isfinite(zero_validation):
                raise FloatingPointError("non-finite zero-model validation MSE")
            trials: list[dict[str, object]] = []
            for trial, ridge in enumerate(config["lambdas"]):
                checkpoint = output / "checkpoints" / f"lambda_{trial}.pt"
                model = PolynomialVAR(
                    36,
                    36,
                    masks_np.shape[1],
                    args.lag,
                    value_scales,
                    mask_scales,
                )
                optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
                start_epoch, history = 0, []
                if checkpoint.exists():
                    if not checkpoint_compatible(checkpoint, identity):
                        raise FileExistsError(
                            "incompatible checkpoint; use --reset to recompute"
                        )
                    saved = torch.load(
                        checkpoint, map_location="cpu", weights_only=False
                    )
                    model.load_state_dict(saved["model"])
                    optimizer.load_state_dict(saved["optimizer"])
                    start_epoch, history = saved["epoch"], saved["history"]
                    torch.set_rng_state(saved["torch_rng"])
                    np.random.set_state(saved["numpy_rng"])
                    random.setstate(saved["python_rng"])
                for epoch in range(start_epoch, args.epochs):
                    model.train()
                    order = target_sets["train"].copy()
                    np.random.shuffle(order)
                    losses = []
                    for start in range(0, len(order), args.batch_size):
                        target_i = torch.as_tensor(
                            order[start : start + args.batch_size]
                        )
                        v, m = batch_lags(values, masks, target_i, args.lag)
                        loss = masked_mse(
                            model(v, m), values[target_i], available[target_i]
                        )
                        require_finite("training loss", loss)
                        # Standard ridge objective: mean prediction error plus
                        # lambda times the sum of squared exported coefficients.
                        penalty = model.ridge_penalty()
                        objective = loss + float(ridge) * penalty
                        require_finite("training objective", objective)
                        optimizer.zero_grad()
                        objective.backward()
                        for parameter in model.parameters():
                            if parameter.grad is None:
                                raise RuntimeError(
                                    "optimizer parameter has no gradient"
                                )
                            require_finite("training gradient", parameter.grad)
                        optimizer.step()
                        for parameter in model.parameters():
                            require_finite("model parameter", parameter)
                        losses.append(float(loss.detach()))
                    valid, _ = evaluate(
                        model,
                        values,
                        masks,
                        available,
                        target_sets["validation"],
                        args.lag,
                        args.batch_size,
                    )
                    if (
                        zero_validation > 0
                        and valid > zero_validation * DIVERGENCE_FACTOR
                    ):
                        raise FloatingPointError(
                            "validation MSE exceeded the zero-model baseline by "
                            f"{DIVERGENCE_FACTOR:g}x"
                        )
                    history.append(
                        {
                            "epoch": epoch + 1,
                            "train_masked_mse": float(np.mean(losses)),
                            "validation_masked_mse": valid,
                        }
                    )
                    atomic_torch(
                        {
                            "fingerprint": identity,
                            "epoch": epoch + 1,
                            "history": history,
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "torch_rng": torch.get_rng_state(),
                            "numpy_rng": np.random.get_state(),
                            "python_rng": random.getstate(),
                        },
                        checkpoint,
                    )
                valid, valid_per = evaluate(
                    model,
                    values,
                    masks,
                    available,
                    target_sets["validation"],
                    args.lag,
                    args.batch_size,
                )
                trials.append(
                    {
                        "trial": trial,
                        "lambda": ridge,
                        "validation_mse": valid,
                        "validation_per_variable": valid_per,
                        "zero_validation_mse": zero_validation,
                        "checkpoint": str(checkpoint),
                        "history": history,
                    }
                )
                atomic_json(
                    {"fingerprint": identity, "trials": trials}, output / "trials.json"
                )
            best = min(trials, key=lambda item: float(item["validation_mse"]))
            saved = torch.load(
                Path(str(best["checkpoint"])), map_location="cpu", weights_only=False
            )
            model = PolynomialVAR(
                36,
                36,
                masks_np.shape[1],
                args.lag,
                value_scales,
                mask_scales,
            )
            model.load_state_dict(saved["model"])
            test, test_per = evaluate(
                model,
                values,
                masks,
                available,
                target_sets["test"],
                args.lag,
                args.batch_size,
            )
            w1_values, w2_values, w3_values = model.exported_value_coefficients()
            w1 = np.concatenate(
                (
                    w1_values.detach().numpy(),
                    model.exported_mask_coefficients().detach().numpy(),
                ),
                axis=1,
            )
            zeros = np.zeros((36, masks_np.shape[1], args.lag), dtype=np.float32)
            atomic_npz(
                final,
                W1=w1,
                W2=np.concatenate((w2_values.detach().numpy(), zeros), axis=1),
                W3=np.concatenate((w3_values.detach().numpy(), zeros), axis=1),
                channel_names=np.asarray(names),
                augmented_channel_names=np.asarray(
                    [
                        *names,
                        *[
                            f"availability_pattern_{i}"
                            for i in range(masks_np.shape[1])
                        ],
                    ]
                ),
                mask_mapping=np.asarray([-1 if x is None else x for x in mapping]),
                mask_channel_count=np.asarray(masks_np.shape[1]),
                scaling_means=means,
                scaling_stds=stds,
                lag_order=np.arange(1, args.lag + 1),
                target_availability=available_np,
            )
            atomic_json(
                {
                    "prototype": "THROWAWAY; never production",
                    "fingerprint": identity,
                    "status": "complete",
                    "config": config,
                    "dimensions": {
                        "outputs": 36,
                        "value_channels": 36,
                        "mask_channels": masks_np.shape[1],
                        "augmented_channels": 36 + masks_np.shape[1],
                        "lag": args.lag,
                        "trained_parameters": sum(
                            p.numel() for p in model.parameters()
                        ),
                        "exported_parameters": int(
                            36 * (36 + masks_np.shape[1]) * args.lag * 3
                        ),
                    },
                    "coefficient_space": "coefficients predict standardized X; original units = prediction * scaling_stds + scaling_means",
                    "feature_scaling": "training-only predictor RMS and sqrt(feature-count) scaling is internal to Adam; W1/W2/W3 are exported in exact standardized-feature space",
                    "mask_mapping": {names[i]: mapping[i] for i in range(36)},
                    "structural_zeros": "W2/W3 availability-mask sections are exactly zero and were never allocated/trained",
                    "best": best,
                    "zero_validation_mse": zero_validation,
                    "zero_validation_per_variable": dict(
                        zip(names, zero_validation_per, strict=True)
                    ),
                    "test_masked_mse": test,
                    "test_per_variable": dict(zip(names, test_per, strict=True)),
                },
                existing,
            )
            return final
    except Timeout as error:
        raise RuntimeError("another fit holds this prototype output lock") from error


def benchmark(args: argparse.Namespace) -> dict[str, object]:
    validate_limits(args.lag, args.threads)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    torch.set_num_threads(args.threads)
    _, raw, _, _ = load_prepared(Path(args.input))
    standardized, available, _, _, mapping = scaling_and_masks(raw, args.lag)
    masks = mask_matrix(available, mapping)
    splits = split_bounds(len(raw) - args.lag)
    train_begin, train_end = splits["train"]
    train_indices = np.arange(args.lag + train_begin, args.lag + train_end)
    preprocessing_started = time.monotonic()
    value_scales_np, mask_scales_np = feature_scales(
        standardized, masks, train_indices, args.lag
    )
    preprocessing_seconds = time.monotonic() - preprocessing_started
    model = PolynomialVAR(
        36,
        36,
        masks.shape[1],
        args.lag,
        torch.from_numpy(value_scales_np),
        torch.from_numpy(mask_scales_np),
    )
    indices = torch.arange(args.lag, min(len(raw), args.lag + args.batch_size))
    values, mask_values = torch.from_numpy(standardized), torch.from_numpy(masks)
    target_availability = torch.from_numpy(available.astype(np.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    started = time.monotonic()
    v, m = batch_lags(values, mask_values, indices, args.lag)
    target = values[indices]
    loss = masked_mse(model(v, m), target, target_availability[indices])
    loss.backward()
    optimizer.step()
    seconds = time.monotonic() - started
    with torch.no_grad():
        post_update_loss = masked_mse(model(v, m), target, target_availability[indices])
    require_finite("benchmark post-update loss", post_update_loss)
    train_batches = int(np.ceil(len(train_indices) / len(indices)))
    result = {
        "batch": len(indices),
        "lag": args.lag,
        "augmented_channels": 36 + masks.shape[1],
        "seconds_forward_backward_adam": seconds,
        "seconds_feature_scaling": preprocessing_seconds,
        "initial_masked_mse": float(loss.detach()),
        "post_update_masked_mse": float(post_update_loss),
        "rows_per_second": len(indices) / seconds,
        "estimated_train_batches_per_epoch": train_batches,
        "estimated_optimizer_seconds_per_epoch": train_batches * seconds,
        "lag_tensor_bytes": int((v.numel() + m.numel()) * 4),
        "dense_design_never_materialized": True,
    }
    atomic_json(result, Path(args.output) / "benchmark.json")
    return result


def status(output: Path) -> dict[str, object]:
    note = output / "fit.provenance.json"
    return json.loads(note.read_text()) if note.exists() else {"status": "absent"}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    for name in ("benchmark", "fit"):
        command = commands.add_parser(name)
        command.add_argument(
            "--input", type=Path, default=DEFAULT_OUTPUT / "prepared_input.npz"
        )
        command.add_argument(
            "--output",
            type=Path,
            default=(
                DEFAULT_BENCHMARK_OUTPUT if name == "benchmark" else DEFAULT_FIT_OUTPUT
            ),
        )
        command.add_argument("--lag", type=int, default=MAX_LAG)
        command.add_argument("--batch-size", type=int, default=64)
        command.add_argument("--threads", type=int, default=10)
        if name == "fit":
            command.add_argument("--epochs", type=int, default=5)
            command.add_argument("--lambdas", default="0.0001,0.001,0.01")
            command.add_argument("--learning-rate", type=float, default=1e-3)
            command.add_argument("--seed", type=int, default=0)
            command.add_argument("--reset", action="store_true")
    status_parser = commands.add_parser("status")
    status_parser.add_argument("--output", type=Path, default=DEFAULT_FIT_OUTPUT)
    return result


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "prepare":
        print(prepare(args.output))
    elif args.command == "benchmark":
        print(json.dumps(benchmark(args), indent=2))
    elif args.command == "fit":
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(fit(args))
    else:
        print(json.dumps(status(args.output), indent=2))


if __name__ == "__main__":
    main()
