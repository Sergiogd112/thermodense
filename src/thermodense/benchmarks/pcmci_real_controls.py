"""IAAFT and circular-shift control robustness for the real PCMCI+ input.

The five physical series receive the registered primary preprocessing exactly
once.  Controls are then made from those preprocessed series and appended
unchanged, so this runner cannot alter the primary or synthetic runners.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import random
import resource
import shutil
import sys
import time
from typing import Any

import numpy as np
from thermodense.benchmarks import pcmci_control_graph, pcmci_real, runtime
from thermodense.benchmarks.real_data import (
    DEFAULT_OUTPUT as DEFAULT_INPUT,
    NODE_COLUMNS,
)

SCHEMA_VERSION = "1"
RUNNER_VERSION = "pcmci-real-controls-1"
METHOD = "parcorr"
DEFAULT_TAU_MAX = 180
SEED = 20260803
IAAFT_ITERATIONS = 100
ANNUAL_DAYS = 365
ANNUAL_EXCLUSION_DAYS = 7
MISSING_FLAG = pcmci_real.MISSING_FLAG


def _sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def interpolate_for_iaaft(values: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Fill only an ephemeral IAAFT copy by linear interpolation and edge hold."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("IAAFT requires at least one finite value")
    positions = np.arange(len(values))
    filled = np.interp(positions, positions[finite], values[finite])
    return filled, {
        "method": "deterministic linear interpolation with nearest finite edge fill",
        "missing_input_count": int((~finite).sum()),
        "filled_input_count": int((~finite).sum()),
    }


def iaaft_surrogate(
    values: np.ndarray, seed: int, iterations: int = IAAFT_ITERATIONS
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a deterministic source-mask-matched IAAFT control and diagnostics."""
    if iterations <= 0:
        raise ValueError("IAAFT iterations must be positive")
    values = np.asarray(values, dtype=float)
    filled, fill_metadata = interpolate_for_iaaft(values)
    rng = np.random.default_rng(seed)
    finite = np.isfinite(values)
    sorted_values = np.sort(filled)
    target_amplitude = np.abs(np.fft.rfft(filled))
    surrogate = rng.permutation(filled)
    for _ in range(iterations):
        phase_adjusted = np.fft.irfft(
            target_amplitude * np.exp(1j * np.angle(np.fft.rfft(surrogate))),
            n=len(filled),
        )
        ranks = np.argsort(np.argsort(phase_adjusted, kind="stable"), kind="stable")
        surrogate = sorted_values[ranks]
    output = np.full_like(values, np.nan)
    finite_ranks = np.argsort(
        np.argsort(surrogate[finite], kind="stable"), kind="stable"
    )
    output[finite] = np.sort(values[finite])[finite_ranks]
    spectral_output, spectral_output_fill = interpolate_for_iaaft(output)
    spectral_error = float(
        np.linalg.norm(np.abs(np.fft.rfft(spectral_output)) - target_amplitude)
        / max(np.linalg.norm(target_amplitude), np.finfo(float).eps)
    )
    return output, fill_metadata | {
        "seed": seed,
        "iterations": iterations,
        "missing_count": int((~finite).sum()),
        "missing_mask_sha256": _sha256(~finite),
        "missingness_matched": bool(np.array_equal(~np.isfinite(output), ~finite)),
        "finite_marginal_reference": "sorted finite observed source values",
        "finite_marginal_preserved": bool(
            np.array_equal(np.sort(output[finite]), np.sort(values[finite]))
        ),
        "spectral_reference": "temporary complete interpolation copy of source",
        "spectral_output_completion": spectral_output_fill["method"],
        "spectral_relative_l2_error": spectral_error,
        "output_sha256": _sha256(output),
    }


def valid_shift_offsets(length: int, tau_max: int) -> np.ndarray:
    """Return offsets strictly outside the lag window and annual aliases."""
    offsets = np.arange(tau_max + 1, length - tau_max, dtype=int)
    modulo = np.minimum(offsets % ANNUAL_DAYS, (-offsets) % ANNUAL_DAYS)
    return offsets[modulo > ANNUAL_EXCLUSION_DAYS]


def generate_controls(
    preprocessed: np.ndarray, node_names: list[str], tau_max: int, seed: int
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    """Generate one IAAFT and one circular-shift control for every physical node."""
    if preprocessed.ndim != 2 or preprocessed.shape[1] != len(node_names):
        raise ValueError("preprocessed data must have one column per physical node")
    offsets = valid_shift_offsets(len(preprocessed), tau_max)
    if not len(offsets):
        raise ValueError("input is too short for a valid circular-shift offset")
    seed_sequence = np.random.SeedSequence(seed)
    children = seed_sequence.spawn(2 * len(node_names))
    controls: dict[str, np.ndarray] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for index, source in enumerate(node_names):
        iaaft_seed = int(children[2 * index].generate_state(1)[0])
        shift_seed = int(children[2 * index + 1].generate_state(1)[0])
        shift_rng = np.random.default_rng(shift_seed)
        iaaft_name = f"control_iaaft_{source}"
        shift_name = f"control_circular_shift_{source}"
        iaaft, iaaft_metadata = iaaft_surrogate(preprocessed[:, index], iaaft_seed)
        offset = int(shift_rng.choice(offsets))
        shifted = np.roll(preprocessed[:, index], offset)
        controls[iaaft_name] = iaaft
        controls[shift_name] = shifted
        metadata[iaaft_name] = iaaft_metadata | {
            "family": "iaaft",
            "source_node": source,
            "source_sha256": _sha256(preprocessed[:, index]),
            "source_mask_matched": True,
        }
        metadata[shift_name] = {
            "family": "circular_shift",
            "source_node": source,
            "seed": shift_seed,
            "offset": offset,
            "annual_modulo_days": int(offset % ANNUAL_DAYS),
            "source_sha256": _sha256(preprocessed[:, index]),
            "output_sha256": _sha256(shifted),
            "missing_count": int((~np.isfinite(shifted)).sum()),
        }
    return controls, metadata


def selected_control_links(
    results: dict[str, np.ndarray],
    node_names: list[str],
    control_metadata: dict[str, dict[str, Any]],
) -> list[dict[str, str | int | float]]:
    """Summarize selected links with endpoint family and source classifications."""
    rows = pcmci_control_graph.selected_control_links(
        results, node_names, set(control_metadata)
    )
    for row in rows:
        for endpoint in ("cause", "target"):
            metadata = control_metadata.get(str(row[endpoint]))
            row[f"{endpoint}_family"] = metadata["family"] if metadata else "physical"
            row[f"{endpoint}_source"] = (
                metadata["source_node"] if metadata else str(row[endpoint])
            )
    return rows


def run_pcmciplus(
    input_data: pcmci_real.RealInput,
    tau_max: int,
    seed: int,
    artifact_path: Path | None = None,
    summary_path: Path | None = None,
) -> dict[str, Any]:
    """Run ParCorr PCMCI+ with controls appended after primary preprocessing."""
    from tigramite import data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI

    if len(input_data.dates) <= 2 * tau_max:
        raise ValueError(
            f"input has {len(input_data.dates)} rows; PCMCI+ requires more than 2*tau_max={2 * tau_max} rows"
        )
    physical_names = list(input_data.metadata.get("node_order", NODE_COLUMNS))
    preprocessed = pcmci_real.preprocess(input_data.values, input_data.dates)
    controls, control_metadata = generate_controls(
        preprocessed, physical_names, tau_max, seed
    )
    node_names = [*physical_names, *controls]
    values = np.column_stack([preprocessed, *(controls[name] for name in controls)])
    random.seed(seed)
    np.random.seed(seed)
    dataframe = pp.DataFrame(
        np.where(np.isfinite(values), values, MISSING_FLAG),
        datatime=np.arange(len(values)),
        var_names=node_names,
        missing_flag=MISSING_FLAG,
        remove_missing_upto_maxlag=False,
    )
    results = PCMCI(
        dataframe=dataframe, cond_ind_test=ParCorr(significance="analytic"), verbosity=0
    ).run_pcmciplus(
        link_assumptions=pcmci_control_graph.build_link_assumptions(
            node_names, tau_max
        ),
        tau_min=0,
        tau_max=tau_max,
        pc_alpha=0.05,
        contemp_collider_rule="majority",
        conflict_resolution=True,
        fdr_method="none",
    )
    matrices = {name: results[name] for name in ("val_matrix", "p_matrix", "graph")}
    payload: dict[str, Any] = {
        "seed": seed,
        "node_names": node_names,
        "control_metadata": control_metadata,
        "matrix_shapes": {
            name: list(np.asarray(value).shape) for name, value in matrices.items()
        },
        "result_digest": runtime.compact_result_digest(matrices),
        "process_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * 1024,
    }
    if artifact_path is not None:
        payload["artifact"] = runtime.write_npz_artifact(
            artifact_path, matrices, node_names=node_names
        )
    if summary_path is not None:
        payload["control_link_summary"] = runtime.write_jsonl_artifact(
            summary_path, selected_control_links(results, node_names, control_metadata)
        )
    return payload


def _child_main(args: argparse.Namespace) -> int:
    try:
        started = time.monotonic()
        payload = run_pcmciplus(
            pcmci_real.load_input(args.input, args.row_limit),
            args.tau_max,
            args.seed,
            args.artifact,
            args.summary,
        )
        payload.update(status="succeeded", wall_seconds=time.monotonic() - started)
    except Exception as error:
        payload = {
            "status": "failed",
            "failure_reason": f"{type(error).__name__}: {error}",
        }
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0


def _run_isolated_case(
    args: argparse.Namespace, threads: int, artifact: Path, summary: Path
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "thermodense.benchmarks.pcmci_real_controls",
        "case",
        "--input",
        str(args.input),
        "--tau-max",
        str(args.tau_max),
        "--seed",
        str(args.seed),
        "--artifact",
        str(artifact),
        "--summary",
        str(summary),
    ]
    if args.row_limit is not None:
        command.extend(["--row-limit", str(args.row_limit)])
    return runtime.run_isolated_process(command, args.timeout, threads)


def _base_row(
    args: argparse.Namespace, input_data: pcmci_real.RealInput, threads: int
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "runner_version": RUNNER_VERSION,
        "timestamp": datetime.now(UTC).isoformat(),
        "status": "pending",
        "synthetic": False,
        "host_label": args.host_label,
        "environment_label": args.environment_label,
        "environment_fingerprint": args.environment_fingerprint,
        "method": METHOD,
        "tau_max": args.tau_max,
        "timeout_seconds": args.timeout,
        "input": input_data.metadata,
        "controls": {
            "seed": args.seed,
            "one_iaaft_and_one_circular_shift_per_physical_node": True,
            "iaaft_iterations": IAAFT_ITERATIONS,
            "iaaft_fill_method": "deterministic linear interpolation with nearest finite edge fill",
            "annual_shift_exclusion_days": ANNUAL_EXCLUSION_DAYS,
        },
        "preprocessing": {
            "profile": "primary_detrended_anomaly",
            "computed_once_on_physical_nodes": True,
            "controls_appended_without_preprocessing": True,
        },
        "algorithm": {
            "name": "PCMCI+",
            "entry_point": "PCMCI.run_pcmciplus",
            "tau_min": 0,
            "pc_alpha": 0.05,
            "contemp_collider_rule": "majority",
            "conflict_resolution": True,
            "fdr_method": "none",
        },
        "link_assumptions": {
            "source": "dynamic F10.7-exogenous assumptions",
            "f10_7_node": "f10_7_center81",
        },
        "missing_data_policy": {
            "sentinel": MISSING_FLAG,
            "remove_missing_upto_maxlag": False,
            "physical_missing_values_preserved": True,
            "controls_source_mask_matched": True,
        },
        "settings": {"significance": "analytic", "threads": threads},
        "package_versions": runtime.package_versions(),
        "git_commit": runtime.git_commit(),
        "wall_seconds": None,
        "process_max_rss_bytes": None,
        "matrix_shapes": {},
        "result_digest": None,
        "artifact": None,
        "control_link_summary": None,
        "failure_reason": None,
    }


def run(args: argparse.Namespace) -> int:
    artifact_directory = args.output.parent / f"{args.output.stem}_artifacts"
    if (args.output.exists() or artifact_directory.exists()) and not args.overwrite:
        raise ValueError(
            f"Refusing to overwrite existing result or artifacts: {args.output}; use --overwrite."
        )
    input_data = pcmci_real.load_input(args.input, args.row_limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        args.output.write_text("")
        shutil.rmtree(artifact_directory, ignore_errors=True)
    artifact_directory.mkdir(parents=True, exist_ok=True)
    row = _base_row(args, input_data, args.threads or 1)
    row.update(
        _run_isolated_case(
            args,
            args.threads or 1,
            artifact_directory / "parcorr.npz",
            artifact_directory / "control_links.jsonl",
        )
    )
    runtime.append_jsonl(args.output, row)
    print(f"{args.output} {json.dumps({row['status']: 1}, sort_keys=True)}")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="python -m thermodense.benchmarks.pcmci_real_controls"
    )
    commands = result.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser(
        "run", help="run isolated IAAFT and circular-shift ParCorr controls"
    )
    run_parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--tau-max", type=int, default=DEFAULT_TAU_MAX)
    run_parser.add_argument(
        "--row-limit", type=int, help="calibration-only prefix row limit"
    )
    run_parser.add_argument("--seed", type=int, default=SEED)
    run_parser.add_argument("--threads", type=int)
    run_parser.add_argument("--timeout", type=float, default=1800.0)
    run_parser.add_argument("--host-label", default="unspecified")
    run_parser.add_argument("--environment-label", default="unspecified")
    run_parser.add_argument("--environment-fingerprint", default="unspecified")
    run_parser.add_argument("--overwrite", action="store_true")
    case = commands.add_parser("case", help=argparse.SUPPRESS)
    case.add_argument("--input", type=Path, required=True)
    case.add_argument("--tau-max", type=int, required=True)
    case.add_argument("--row-limit", type=int)
    case.add_argument("--seed", type=int, required=True)
    case.add_argument("--artifact", type=Path, required=True)
    case.add_argument("--summary", type=Path, required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "case":
        return _child_main(args)
    try:
        if (
            args.timeout <= 0
            or args.tau_max < 0
            or (args.threads is not None and args.threads <= 0)
        ):
            raise ValueError(
                "--timeout and --threads must be positive; --tau-max must be non-negative."
            )
        return run(args)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
