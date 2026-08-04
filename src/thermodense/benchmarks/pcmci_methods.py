"""Reproducible CPU benchmark for selected Tigramite PCMCI methods.

This module intentionally benchmarks only synthetic arrays and PCMCI execution.
It does not prepare scientific workflow inputs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import functools
import hashlib
import json
from pathlib import Path
import random
import resource
import sys
import time
from typing import Any

import numpy as np

from thermodense.benchmarks import runtime

SPEC_RELATIVE_PATH = Path("benchmarks") / "pcmci-methods" / "spec.toml"

SCHEMA_VERSION = "1"
BENCHMARK_VERSION = "pcmci-methods-synthetic-2"
SEED = 20260731
NODES = 5
METHODS = ("parcorr", "cmiknn", "gpdc", "gpdctorch")
LEVELS = (
    ("small", 512, 7),
    ("medium", 2048, 30),
    ("representative", 8400, 180),
)
THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OPENBLAS_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


@dataclass(frozen=True)
class Case:
    method: str
    level: str
    samples: int
    tau_max: int


def benchmark_plan(
    methods: tuple[str, ...] = METHODS,
    levels: tuple[str, ...] = tuple(level[0] for level in LEVELS),
) -> list[Case]:
    """Return the frozen, method-major progressive benchmark plan."""
    level_specs = {name: (samples, tau_max) for name, samples, tau_max in LEVELS}
    return [
        Case(method, level, *level_specs[level])
        for method in methods
        for level in levels
        if not (
            method == "cmiknn"
            and level_specs[level][1] > runtime.CMIKNN_MAX_TAU_STEPS
        )
    ]


def plan_document() -> dict[str, Any]:
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "schema_version": SCHEMA_VERSION,
        "synthetic": True,
        "seed": SEED,
        "nodes": NODES,
        "methods": list(METHODS),
        "spec_digest": spec_digest(),
        "deferred_methods": {},
        "cases": [asdict(case) for case in benchmark_plan()],
    }


def generate_synthetic_data(samples: int, *, level_index: int) -> np.ndarray:
    """Generate one independently seeded, stable five-node VAR process.

    The fixed lag-one and lag-two coefficients encode known lagged links.  A
    level-specific derived seed makes each scale a fresh deterministic draw.
    """
    rng = np.random.default_rng(np.random.SeedSequence([SEED, level_index]))
    coefficients_1 = np.array(
        [
            [0.45, 0.00, 0.00, 0.00, 0.00],
            [0.30, 0.35, 0.00, 0.00, 0.00],
            [0.00, -0.25, 0.30, 0.00, 0.00],
            [0.00, 0.00, 0.28, 0.25, 0.00],
            [-0.20, 0.00, 0.00, 0.32, 0.20],
        ]
    )
    coefficients_2 = np.array(
        [
            [0.00, 0.00, 0.00, 0.00, 0.15],
            [0.00, 0.00, 0.00, 0.00, 0.00],
            [0.12, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.10, 0.00, 0.00, 0.00],
        ]
    )
    burn_in = 200
    values = np.zeros((samples + burn_in, NODES), dtype=np.float64)
    noise = rng.normal(size=values.shape)
    for index in range(2, len(values)):
        values[index] = (
            coefficients_1 @ values[index - 1]
            + coefficients_2 @ values[index - 2]
            + noise[index]
        )
    data = values[burn_in:]
    return (data - data.mean(axis=0)) / data.std(axis=0, ddof=0)


def method_settings(method: str, cmiknn_workers: int | None = None) -> dict[str, Any]:
    return runtime.method_settings(method, cmiknn_workers)


def compact_result_digest(results: dict[str, Any]) -> str:
    """Hash output matrices without writing their potentially large contents."""
    return runtime.compact_result_digest(results)


def _package_versions() -> dict[str, str]:
    return runtime.package_versions()


def _repo_root() -> Path | None:
    """Return the enclosing git checkout root, or None when unavailable."""
    for candidate in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
        if (candidate / ".git").exists():
            return candidate
    return None


@functools.lru_cache(maxsize=1)
def spec_digest() -> str | None:
    """SHA-256 of the frozen spec.toml, or None when the checkout is unavailable."""
    root = _repo_root()
    if root is None:
        return None
    spec_path = root / SPEC_RELATIVE_PATH
    if not spec_path.exists():
        return None
    return hashlib.sha256(spec_path.read_bytes()).hexdigest()


@functools.lru_cache(maxsize=1)
def git_commit() -> str | None:
    """HEAD commit of the enclosing checkout, or None when unavailable."""
    return runtime.git_commit()


def run_pcmci_case(case: Case, *, cmiknn_workers: int | None = None) -> dict[str, Any]:
    """Run one real PCMCI case; imported dependencies stay inside the child."""
    runtime.validate_cmiknn_tau(case.method, case.tau_max)
    from tigramite import data_processing as pp
    from tigramite.independence_tests.cmiknn import CMIknn
    from tigramite.independence_tests.gpdc import GPDC
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI

    level_index = next(
        index for index, level in enumerate(LEVELS) if level[0] == case.level
    )
    data = generate_synthetic_data(case.samples, level_index=level_index)
    settings = method_settings(case.method, cmiknn_workers)
    # CMIknn's shuffle test draws from the process-global RNGs; reseed them per
    # case so every run of a given (method, level) is bit-reproducible.
    derived = int(
        np.random.SeedSequence(
            [SEED, level_index, METHODS.index(case.method)]
        ).generate_state(1)[0]
    )
    random.seed(derived)
    np.random.seed(derived)
    if case.method == "parcorr":
        test = ParCorr(significance="analytic")
    elif case.method == "cmiknn":
        test = CMIknn(
            significance="shuffle_test",
            sig_samples=20,
            sig_blocklength=4,
            knn=0.1,
            shuffle_neighbors=5,
            workers=settings["workers"],
        )
    elif case.method == "gpdc":
        test = GPDC(significance="analytic")
    else:
        from tigramite.independence_tests.gpdc_torch import GPDCtorch

        test = GPDCtorch(significance="analytic")
    pcmci = PCMCI(dataframe=pp.DataFrame(data), cond_ind_test=test, verbosity=0)
    result = pcmci.run_pcmci(tau_max=case.tau_max, pc_alpha=0.05, alpha_level=0.05)
    matrices = {
        name: value
        for name, value in result.items()
        if name in {"val_matrix", "p_matrix", "graph"}
    }
    return {
        "settings": settings,
        "matrix_shapes": {
            name: list(np.asarray(value).shape) for name, value in matrices.items()
        },
        "result_digest": compact_result_digest(matrices),
        "process_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * 1024,
    }


def _child_main(args: argparse.Namespace) -> int:
    case = Case(args.method, args.level, args.samples, args.tau_max)
    try:
        started = time.monotonic()
        payload = run_pcmci_case(case, cmiknn_workers=args.cmiknn_workers)
        payload["status"] = "succeeded"
        payload["wall_seconds"] = time.monotonic() - started
    except Exception as error:  # Child exceptions must become durable parent rows.
        payload = {
            "status": "failed",
            "failure_reason": f"{type(error).__name__}: {error}",
        }
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0


def _thread_environment(threads: int) -> dict[str, str]:
    return {name: str(threads) for name in THREAD_ENVIRONMENT}


def _run_isolated_case(
    case: Case,
    timeout: float,
    threads: int,
    command: list[str] | None = None,
    cmiknn_workers: int | None = None,
) -> dict[str, Any]:
    child_command = command or [
        sys.executable,
        "-m",
        "thermodense.benchmarks.pcmci_methods",
        "case",
        "--method",
        case.method,
        "--level",
        case.level,
        "--samples",
        str(case.samples),
        "--tau-max",
        str(case.tau_max),
    ]
    if cmiknn_workers is not None:
        child_command += ["--cmiknn-workers", str(cmiknn_workers)]
    return runtime.run_isolated_process(child_command, timeout, threads)


def _base_row(case: Case, args: argparse.Namespace, threads: int) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_version": BENCHMARK_VERSION,
        "timestamp": datetime.now(UTC).isoformat(),
        "host_label": args.host_label,
        "environment_label": args.environment_label,
        "environment_fingerprint": args.environment_fingerprint,
        "git_commit": git_commit(),
        "spec_digest": spec_digest(),
        "timeout_seconds": args.timeout,
        "synthetic": True,
        "method": case.method,
        "level": case.level,
        "samples": case.samples,
        "nodes": NODES,
        "tau_max": case.tau_max,
        "seed": SEED,
        "settings": method_settings(case.method, args.cmiknn_workers)
        | {"threads": threads},
        "package_versions": _package_versions(),
        "status": "pending",
        "skip_reason": None,
        "failure_reason": None,
        "wall_seconds": None,
        "process_max_rss_bytes": None,
        "matrix_shapes": {},
        "result_digest": None,
    }


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    runtime.append_jsonl(path, row)


def run_benchmark(args: argparse.Namespace) -> int:
    output = args.output
    if output.exists() and not args.overwrite:
        raise ValueError(
            f"Refusing to overwrite existing result: {output}; use --overwrite."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        output.write_text("")
    requested_methods = set(args.methods or METHODS)
    requested_levels = set(args.levels or tuple(level[0] for level in LEVELS))
    methods = tuple(method for method in METHODS if method in requested_methods)
    levels = tuple(level[0] for level in LEVELS if level[0] in requested_levels)
    threads = args.threads if args.threads is not None else 1
    blocked: dict[str, str] = {}
    summary: dict[str, int] = {}
    for case in benchmark_plan(methods, levels):
        row = _base_row(case, args, threads)
        if case.method in blocked:
            row.update(
                status="skipped", skip_reason=blocked[case.method], wall_seconds=0.0
            )
        else:
            print(f"running {case.method}/{case.level}", file=sys.stderr, flush=True)
            row.update(
                _run_isolated_case(
                    case, args.timeout, threads, cmiknn_workers=args.cmiknn_workers
                )
            )
            if row["status"] != "succeeded":
                blocked[case.method] = (
                    f"previous {case.level} case {row['status']}: "
                    f"{row.get('failure_reason', 'unknown failure')}"
                )
        _append_jsonl(output, row)
        summary[row["status"]] = summary.get(row["status"], 0) + 1
    print(f"{output} {json.dumps(summary, sort_keys=True)}")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="python -m thermodense.benchmarks.pcmci_methods"
    )
    commands = result.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="show the frozen synthetic benchmark plan")
    plan.add_argument("--format", choices=("human", "json"), default="human")
    run = commands.add_parser(
        "run", help="run benchmark cases in isolated child processes"
    )
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--host-label", default="unspecified")
    run.add_argument("--environment-label", default="unspecified")
    run.add_argument("--environment-fingerprint", default="unspecified")
    run.add_argument("--timeout", type=float, default=1800.0)
    run.add_argument("--methods", choices=METHODS, nargs="+")
    run.add_argument("--levels", choices=tuple(level[0] for level in LEVELS), nargs="+")
    run.add_argument("--threads", type=int)
    run.add_argument("--overwrite", action="store_true")
    run.add_argument(
        "--cmiknn-workers",
        type=int,
        default=None,
        help="spec-divergent probe override: CMIknn scipy workers (frozen spec pins 1)",
    )
    case = commands.add_parser("case", help=argparse.SUPPRESS)
    case.add_argument("--method", choices=METHODS, required=True)
    case.add_argument(
        "--level", choices=tuple(level[0] for level in LEVELS), required=True
    )
    case.add_argument("--samples", type=int, required=True)
    case.add_argument("--tau-max", type=int, required=True)
    case.add_argument("--cmiknn-workers", type=int, default=None)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "plan":
        document = plan_document()
        if args.format == "json":
            print(json.dumps(document, sort_keys=True))
        else:
            print(f"{document['benchmark_version']}: {len(document['cases'])} cases")
            for case in document["cases"]:
                print(
                    f"  {case['method']} {case['level']}: n={case['samples']}, tau_max={case['tau_max']}"
                )
            print(
                "  CMIknn: current-compute resource cap is "
                f"tau_max <= {runtime.CMIKNN_MAX_TAU_STEPS}"
            )
        return 0
    if args.command == "case":
        return _child_main(args)
    try:
        if args.timeout <= 0 or (args.threads is not None and args.threads <= 0):
            raise ValueError("--timeout and --threads must be positive.")
        return run_benchmark(args)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
