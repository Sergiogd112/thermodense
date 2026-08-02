"""Isolated ParCorr surrogate robustness runs for the real five-node product.

This runner deliberately does not alter :mod:`pcmci_real`: its five physical
nodes remain the primary real-data estimand.  Controls are appended to the raw
daily node matrix and then receive precisely the same registered preprocessing.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import shutil
import sys
import tempfile
import time
from typing import Any

import numpy as np
import polars as pl

from thermodense.benchmarks import pcmci_real, runtime
from thermodense.benchmarks.real_data import (
    DEFAULT_OUTPUT as DEFAULT_INPUT,
    NODE_COLUMNS,
)

SCHEMA_VERSION = "1"
RUNNER_VERSION = "pcmci-real-surrogates-1"
METHOD = "parcorr"
DEFAULT_TAU_MAX = 180
SEED = 20260601
SIX_MONTH_PERIOD_DAYS = 365.25 / 2
SOLAR_CYCLE_PERIOD_DAYS = 11.4 * 365.25
MISSING_FLAG = pcmci_real.MISSING_FLAG


def generate_surrogates(
    dates: np.ndarray,
    seed: int = SEED,
    white_count: int = 5,
    six_month_count: int = 3,
    solar_cycle_count: int = 3,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    """Generate deterministic raw daily controls using the established families."""
    days = np.asarray(dates).astype("datetime64[D]").astype("int64")
    if not len(days):
        raise ValueError("cannot generate surrogates without dates")
    days = days - days[0]
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {}
    settings: dict[str, dict[str, Any]] = {}

    for index in range(1, white_count + 1):
        name = f"surrogate_white_noise_{index}"
        data[name] = rng.normal(0.0, 1.0, size=len(days))
        settings[name] = {
            "family": "gaussian_white_noise",
            "mean": 0.0,
            "variance": 1.0,
        }

    for prefix, family, period_days, count in (
        (
            "surrogate_sine_6mo",
            "six_month_sine_plus_noise",
            SIX_MONTH_PERIOD_DAYS,
            six_month_count,
        ),
        (
            "surrogate_sine_11p4yr",
            "solar_cycle_sine_plus_noise",
            SOLAR_CYCLE_PERIOD_DAYS,
            solar_cycle_count,
        ),
    ):
        for index in range(1, count + 1):
            name = f"{prefix}_{index}"
            phase = float(rng.uniform(0.0, 2 * np.pi))
            sine = np.sin(2 * np.pi * days / period_days + phase)
            sine_variance = float(np.var(sine))
            noise_variance = 0.25 * sine_variance
            data[name] = sine + rng.normal(0.0, np.sqrt(noise_variance), size=len(days))
            settings[name] = {
                "family": family,
                "period_days": period_days,
                "phase_radians": phase,
                "sine_variance": sine_variance,
                "noise_variance": noise_variance,
                "noise_variance_fraction_of_sine": 0.25,
            }
    return data, settings


def build_link_assumptions(
    node_names: list[str], tau_max: int, f107_node: str = "f10_7_center81"
) -> dict[int, dict[tuple[int, int], str]]:
    """Apply the F10.7-exogenous policy to an arbitrary augmented node list."""
    f107_index = node_names.index(f107_node)
    assumptions: dict[int, dict[tuple[int, int], str]] = {
        target: {} for target in range(len(node_names))
    }
    for target in range(len(node_names)):
        for cause in range(len(node_names)):
            for lag in range(1, tau_max + 1):
                if target == f107_index and cause != f107_index:
                    continue
                assumptions[target][(cause, -lag)] = "-?>"
    for cause in range(len(node_names)):
        for target in range(cause + 1, len(node_names)):
            if cause == f107_index:
                assumptions[target][(f107_index, 0)] = "-?>"
            elif target == f107_index:
                assumptions[cause][(f107_index, 0)] = "-?>"
            else:
                assumptions[target][(cause, 0)] = "o?o"
    return assumptions


def selected_surrogate_links(
    results: dict[str, np.ndarray], node_names: list[str], surrogate_names: set[str]
) -> list[dict[str, str | int | float]]:
    """Return one canonical row per selected graph link involving a control."""
    rows: list[dict[str, str | int | float]] = []
    graph, p_matrix, val_matrix = (
        results["graph"],
        results["p_matrix"],
        results["val_matrix"],
    )
    for cause_index, cause in enumerate(node_names):
        for target_index, target in enumerate(node_names):
            if cause not in surrogate_names and target not in surrogate_names:
                continue
            for lag in range(graph.shape[2]):
                graph_mark = str(graph[cause_index, target_index, lag])
                if not graph_mark or (lag == 0 and graph_mark == "<--"):
                    continue
                if (
                    lag == 0
                    and graph_mark in {"o-o", "x-x"}
                    and cause_index > target_index
                ):
                    continue
                if lag > 0 and graph_mark != "-->":
                    continue
                if cause in surrogate_names and target in surrogate_names:
                    relation = "surrogate↔surrogate"
                elif cause in surrogate_names:
                    relation = "surrogate→physical"
                else:
                    relation = "physical→surrogate"
                rows.append(
                    {
                        "relation": relation,
                        "cause": cause,
                        "target": target,
                        "lag": lag,
                        "graph_mark": graph_mark,
                        "p_value": float(p_matrix[cause_index, target_index, lag]),
                        "val": float(val_matrix[cause_index, target_index, lag]),
                    }
                )
    return rows


def _write_csv_atomic(
    path: Path, rows: list[dict[str, str | int | float]]
) -> dict[str, Any]:
    schema = {
        "relation": pl.String,
        "cause": pl.String,
        "target": pl.String,
        "lag": pl.Int64,
        "graph_mark": pl.String,
        "p_value": pl.Float64,
        "val": pl.Float64,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".csv", delete=False
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        (
            pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)
        ).write_csv(temporary_path)
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return {
        "path": str(path),
        "name": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "format": "csv",
        "row_count": len(rows),
    }


def run_pcmciplus(
    input_data: pcmci_real.RealInput,
    tau_max: int,
    seed: int,
    white_count: int,
    six_month_count: int,
    solar_cycle_count: int,
    artifact_path: Path | None = None,
    summary_path: Path | None = None,
) -> dict[str, Any]:
    """Run PCMCI+ ParCorr on physical raw inputs plus raw surrogate controls."""
    from tigramite import data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI

    if len(input_data.dates) <= 2 * tau_max:
        raise ValueError(
            f"input has {len(input_data.dates)} rows; PCMCI+ requires more than 2*tau_max={2 * tau_max} rows"
        )
    surrogate_data, surrogate_settings = generate_surrogates(
        input_data.dates, seed, white_count, six_month_count, solar_cycle_count
    )
    node_names = [*input_data.metadata.get("node_order", NODE_COLUMNS), *surrogate_data]
    raw_values = np.column_stack(
        [input_data.values, *(surrogate_data[name] for name in surrogate_data)]
    )
    random.seed(seed)
    np.random.seed(seed)
    transformed = pcmci_real.preprocess(raw_values, input_data.dates)
    dataframe = pp.DataFrame(
        np.where(np.isfinite(transformed), transformed, MISSING_FLAG),
        datatime=np.arange(len(transformed)),
        var_names=node_names,
        missing_flag=MISSING_FLAG,
        remove_missing_upto_maxlag=False,
    )
    results = PCMCI(
        dataframe=dataframe, cond_ind_test=ParCorr(significance="analytic"), verbosity=0
    ).run_pcmciplus(
        link_assumptions=build_link_assumptions(node_names, tau_max),
        tau_min=0,
        tau_max=tau_max,
        pc_alpha=0.05,
        contemp_collider_rule="majority",
        conflict_resolution=True,
        fdr_method="none",
    )
    matrices = {name: results[name] for name in ("val_matrix", "p_matrix", "graph")}
    summary_rows = selected_surrogate_links(results, node_names, set(surrogate_data))
    payload: dict[str, Any] = {
        "seed": seed,
        "node_names": node_names,
        "surrogate_names": list(surrogate_data),
        "surrogate_settings": surrogate_settings,
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
        payload["surrogate_link_summary"] = _write_csv_atomic(
            summary_path, summary_rows
        )
    return payload


def _child_main(args: argparse.Namespace) -> int:
    try:
        started = time.monotonic()
        payload = run_pcmciplus(
            pcmci_real.load_input(args.input, args.row_limit),
            args.tau_max,
            args.surrogate_seed,
            args.white_surrogates,
            args.six_month_surrogates,
            args.solar_cycle_surrogates,
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
        "thermodense.benchmarks.pcmci_real_surrogates",
        "case",
        "--input",
        str(args.input),
        "--tau-max",
        str(args.tau_max),
        "--surrogate-seed",
        str(args.surrogate_seed),
        "--white-surrogates",
        str(args.white_surrogates),
        "--six-month-surrogates",
        str(args.six_month_surrogates),
        "--solar-cycle-surrogates",
        str(args.solar_cycle_surrogates),
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
    surrogate_names = [
        *(
            f"surrogate_white_noise_{index}"
            for index in range(1, args.white_surrogates + 1)
        ),
        *(
            f"surrogate_sine_6mo_{index}"
            for index in range(1, args.six_month_surrogates + 1)
        ),
        *(
            f"surrogate_sine_11p4yr_{index}"
            for index in range(1, args.solar_cycle_surrogates + 1)
        ),
    ]
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
        "surrogates": {
            "seed": args.surrogate_seed,
            "families": {
                "gaussian_white_noise": {
                    "count": args.white_surrogates,
                    "variance": 1.0,
                },
                "six_month_sine_plus_noise": {
                    "count": args.six_month_surrogates,
                    "period_days": SIX_MONTH_PERIOD_DAYS,
                    "noise_variance_fraction_of_sine": 0.25,
                },
                "solar_cycle_sine_plus_noise": {
                    "count": args.solar_cycle_surrogates,
                    "period_days": SOLAR_CYCLE_PERIOD_DAYS,
                    "noise_variance_fraction_of_sine": 0.25,
                },
            },
            "names": surrogate_names,
            "appended_before_preprocessing": True,
        },
        "preprocessing": {
            "profile": "primary_detrended_anomaly",
            "source": "thermodense.benchmarks.pcmci_real.preprocess",
            "missing_values_preserved": True,
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
            "other_nodes_cannot_cause_f10_7_at_lagged_or_contemporaneous_lags": True,
            "f10_7_self_lags_allowed": True,
        },
        "missing_data_policy": {
            "sentinel": MISSING_FLAG,
            "remove_missing_upto_maxlag": False,
            "drivers_interpolated": False,
            "rows_dropped": False,
        },
        "settings": {"significance": "analytic", "threads": threads},
        "package_versions": runtime.package_versions(),
        "git_commit": runtime.git_commit(),
        "wall_seconds": None,
        "process_max_rss_bytes": None,
        "matrix_shapes": {},
        "result_digest": None,
        "artifact": None,
        "surrogate_link_summary": None,
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
    threads = args.threads if args.threads is not None else 1
    row = _base_row(args, input_data, threads)
    row.update(
        _run_isolated_case(
            args,
            threads,
            artifact_directory / "parcorr.npz",
            artifact_directory / "surrogate_links.csv",
        )
    )
    runtime.append_jsonl(args.output, row)
    print(f"{args.output} {json.dumps({row['status']: 1}, sort_keys=True)}")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="python -m thermodense.benchmarks.pcmci_real_surrogates"
    )
    commands = result.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser(
        "run", help="run isolated real-data ParCorr surrogate robustness"
    )
    run_parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--tau-max", type=int, default=DEFAULT_TAU_MAX)
    run_parser.add_argument(
        "--row-limit", type=int, help="calibration-only prefix row limit"
    )
    run_parser.add_argument("--surrogate-seed", type=int, default=SEED)
    run_parser.add_argument("--white-surrogates", type=int, default=5)
    run_parser.add_argument("--six-month-surrogates", type=int, default=3)
    run_parser.add_argument("--solar-cycle-surrogates", type=int, default=3)
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
    case.add_argument("--surrogate-seed", type=int, required=True)
    case.add_argument("--white-surrogates", type=int, required=True)
    case.add_argument("--six-month-surrogates", type=int, required=True)
    case.add_argument("--solar-cycle-surrogates", type=int, required=True)
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
            or min(
                args.white_surrogates,
                args.six_month_surrogates,
                args.solar_cycle_surrogates,
            )
            < 0
        ):
            raise ValueError(
                "--timeout and --threads must be positive; --tau-max and surrogate counts must be non-negative."
            )
        return run(args)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
