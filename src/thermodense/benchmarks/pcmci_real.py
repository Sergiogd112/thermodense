"""Isolated PCMCI+ runs for the assembled real five-node daily product.

This is deliberately separate from :mod:`pcmci_methods`: that module is the
frozen synthetic harness and its plan and results must not change.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import random
import resource
import sys
import time
from typing import Any

import numpy as np
import polars as pl

from thermodense.benchmarks import pcmci_methods
from thermodense.benchmarks.real_data import (
    DATE_COLUMN,
    DEFAULT_OUTPUT as DEFAULT_INPUT,
    IMPUTATION_MASK_COLUMNS,
    NODE_COLUMNS,
)

SCHEMA_VERSION = "1"
RUNNER_VERSION = "pcmci-real-1"
METHODS = ("parcorr", "cmiknn")
DEFERRED_METHODS = {"gpdc": "explicitly deferred for real-data PCMCI+ runs"}
DEFAULT_TAU_MAX = 180
DEFAULT_CMIKNN_WORKERS = 24
MISSING_FLAG = -999999.0
SEED = 20260802
ROLLING_WINDOW = 1095


@dataclass(frozen=True)
class RealInput:
    dates: np.ndarray
    values: np.ndarray
    metadata: dict[str, Any]


def _day_of_year(dates: np.ndarray) -> np.ndarray:
    python_dates = dates.astype("datetime64[D]").astype(object)
    return np.array([date.timetuple().tm_yday for date in python_dates], dtype=int)


def rolling_nanmean(values: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling mean which ignores, but never fills, missing values."""
    values = np.asarray(values, dtype=float)
    window = max(1, min(window, len(values)))
    finite = np.isfinite(values)
    numerator = np.convolve(np.where(finite, values, 0.0), np.ones(window), mode="same")
    denominator = np.convolve(finite.astype(float), np.ones(window), mode="same")
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(values, np.nan),
        where=denominator > 0,
    )


def finite_standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return values.copy()
    mean = np.mean(values[finite])
    std = np.std(values[finite])
    result = values - mean if std == 0 else (values - mean) / std
    result[~finite] = np.nan
    return result


def preprocess(values: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """Apply the registered primary daily detrended-anomaly transformation."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must be a two-dimensional node matrix")
    doys = _day_of_year(dates)
    result = np.empty_like(values, dtype=float)
    for column in range(values.shape[1]):
        node = values[:, column]
        climatology = np.full(367, np.nan)
        for doy in range(1, 367):
            selected = node[doys == doy]
            finite = selected[np.isfinite(selected)]
            if finite.size:
                climatology[doy] = finite.mean()
        finite_node = node[np.isfinite(node)]
        fallback = finite_node.mean() if finite_node.size else np.nan
        climatology = np.where(np.isfinite(climatology), climatology, fallback)
        anomaly = node - climatology[doys]
        result[:, column] = finite_standardize(
            anomaly - rolling_nanmean(anomaly, ROLLING_WINDOW)
        )
    return result


def validate_daily_dates(dates: np.ndarray) -> None:
    days = np.asarray(dates).astype("datetime64[D]")
    if len(days) == 0:
        raise ValueError("input CSV contains no dated rows")
    if np.isnat(days).any():
        raise ValueError("input CSV contains an invalid date")
    differences = np.diff(days.astype("int64"))
    if np.any(differences == 0):
        raise ValueError("input CSV dates must be unique")
    if np.any(differences != 1):
        raise ValueError("input CSV dates must be consecutive daily dates")


def _node_counts(values: np.ndarray, imputed: np.ndarray) -> dict[str, dict[str, int]]:
    return {
        column: {
            "observed": int((np.isfinite(values[:, index]) & ~imputed[:, index]).sum()),
            "missing": int((~np.isfinite(values[:, index])).sum()),
            "imputed": int(imputed[:, index].sum()),
        }
        for index, column in enumerate(NODE_COLUMNS)
    }


def load_input(path: Path, row_limit: int | None = None) -> RealInput:
    required = [DATE_COLUMN, *NODE_COLUMNS]
    raw = pl.read_csv(path, null_values=["", "NaN", "nan"])
    missing_columns = set(required) - set(raw.columns)
    if missing_columns:
        raise ValueError(f"input CSV is missing columns: {sorted(missing_columns)}")
    frame = raw.select(
        pl.col(DATE_COLUMN)
        .cast(pl.String)
        .str.to_date("%Y-%m-%d", strict=False)
        .alias(DATE_COLUMN),
        *NODE_COLUMNS,
        *[column for column in IMPUTATION_MASK_COLUMNS if column in raw.columns],
    )
    dates = frame[DATE_COLUMN].to_numpy().astype("datetime64[D]")
    validate_daily_dates(dates)
    if row_limit is not None:
        if row_limit <= 0:
            raise ValueError("--row-limit must be positive")
        frame = frame.head(row_limit)
        dates = dates[: len(frame)]
    values = np.column_stack(
        [frame[column].to_numpy().astype(float) for column in NODE_COLUMNS]
    )
    imputed = np.column_stack(
        [
            frame[column].fill_null(False).cast(pl.Boolean).to_numpy()
            if column in frame.columns
            else np.zeros(len(frame), dtype=bool)
            for column in IMPUTATION_MASK_COLUMNS
        ]
    )
    metadata = {
        "path": str(path),
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "date_range": {"start": str(dates[0]), "end": str(dates[-1])},
        "row_count": len(frame),
        "node_order": NODE_COLUMNS,
        "node_counts": _node_counts(values, imputed),
        "row_limit": row_limit,
        "row_limit_calibration_only": row_limit is not None,
        "co2_source": (
            "NOAA GML Mauna Loa daily means; source includes Maunakea "
            "substitute observations from December 2022 through 2023-07-04"
        ),
    }
    return RealInput(dates, values, metadata)


def build_link_assumptions(tau_max: int) -> dict[int, dict[tuple[int, int], str]]:
    """Match the established exogenous-F10.7 assumptions in the analysis script."""
    f107_index = NODE_COLUMNS.index("f10_7_center81")
    assumptions: dict[int, dict[tuple[int, int], str]] = {
        target: {} for target in range(len(NODE_COLUMNS))
    }
    for target in range(len(NODE_COLUMNS)):
        for cause in range(len(NODE_COLUMNS)):
            for lag in range(1, tau_max + 1):
                if target == f107_index and cause != f107_index:
                    continue
                assumptions[target][(cause, -lag)] = "-?>"
    for cause in range(len(NODE_COLUMNS)):
        for target in range(cause + 1, len(NODE_COLUMNS)):
            if cause == f107_index:
                assumptions[target][(f107_index, 0)] = "-?>"
            elif target == f107_index:
                assumptions[cause][(f107_index, 0)] = "-?>"
            else:
                assumptions[target][(cause, 0)] = "o?o"
    return assumptions


def _link_assumption_metadata(tau_max: int) -> dict[str, Any]:
    return {
        "source": "scripts/tigramite_causal_global_mean.py:build_link_assumptions",
        "f10_7_node": "f10_7_center81",
        "other_nodes_cannot_cause_f10_7_at_lagged_or_contemporaneous_lags": True,
        "f10_7_self_lags_allowed": True,
        "lagged_link_mark": "-?>",
        "non_f10_7_contemporaneous_link_mark": "o?o",
        "tau_max": tau_max,
        "link_count": sum(
            len(links) for links in build_link_assumptions(tau_max).values()
        ),
    }


def _method_seed(method: str) -> int:
    return int(
        np.random.SeedSequence([SEED, METHODS.index(method)]).generate_state(1)[0]
    )


def real_method_settings(method: str, cmiknn_workers: int) -> dict[str, Any]:
    """Return only settings consumed by the real PCMCI+ execution."""
    settings = pcmci_methods.method_settings(method, cmiknn_workers)
    # The synthetic run_pcmci entry point consumes alpha_level; run_pcmciplus
    # does not expose that argument and instead uses pc_alpha.
    settings.pop("alpha_level")
    return settings


def run_pcmciplus(
    input_data: RealInput, method: str, tau_max: int, cmiknn_workers: int
) -> dict[str, Any]:
    """Run one real PCMCI+ method. Tigramite imports remain in the child process."""
    from tigramite import data_processing as pp
    from tigramite.independence_tests.cmiknn import CMIknn
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI

    if len(input_data.dates) <= 2 * tau_max:
        raise ValueError(
            f"input has {len(input_data.dates)} rows; PCMCI+ requires more than "
            f"2*tau_max={2 * tau_max} rows"
        )
    seed = _method_seed(method)
    random.seed(seed)
    np.random.seed(seed)
    settings = real_method_settings(method, cmiknn_workers)
    test = (
        ParCorr(significance="analytic")
        if method == "parcorr"
        else CMIknn(
            significance="shuffle_test",
            sig_samples=20,
            sig_blocklength=4,
            knn=0.1,
            shuffle_neighbors=5,
            workers=cmiknn_workers,
        )
    )
    transformed = preprocess(input_data.values, input_data.dates)
    dataframe = pp.DataFrame(
        np.where(np.isfinite(transformed), transformed, MISSING_FLAG),
        datatime=np.arange(len(transformed)),
        var_names=NODE_COLUMNS,
        missing_flag=MISSING_FLAG,
        remove_missing_upto_maxlag=False,
    )
    results = PCMCI(dataframe=dataframe, cond_ind_test=test, verbosity=0).run_pcmciplus(
        link_assumptions=build_link_assumptions(tau_max),
        tau_min=0,
        tau_max=tau_max,
        pc_alpha=0.05,
        contemp_collider_rule="majority",
        conflict_resolution=True,
        fdr_method="none",
    )
    matrices = {
        name: results[name]
        for name in ("val_matrix", "p_matrix", "graph")
        if name in results
    }
    return {
        "settings": settings,
        "seed": seed,
        "matrix_shapes": {
            name: list(np.asarray(value).shape) for name, value in matrices.items()
        },
        "result_digest": pcmci_methods.compact_result_digest(matrices),
        "process_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * 1024,
    }


def _child_main(args: argparse.Namespace) -> int:
    try:
        started = time.monotonic()
        payload = run_pcmciplus(
            load_input(args.input, args.row_limit),
            args.method,
            args.tau_max,
            args.cmiknn_workers,
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
    args: argparse.Namespace, method: str, threads: int
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "thermodense.benchmarks.pcmci_real",
        "case",
        "--input",
        str(args.input),
        "--method",
        method,
        "--tau-max",
        str(args.tau_max),
        "--cmiknn-workers",
        str(args.cmiknn_workers),
    ]
    if args.row_limit is not None:
        command.extend(["--row-limit", str(args.row_limit)])
    return pcmci_methods._run_isolated_case(  # noqa: SLF001 - shared tested isolation primitive
        pcmci_methods.Case(method, "real", 0, args.tau_max),
        args.timeout,
        threads,
        command=command,
    )


def _base_row(
    args: argparse.Namespace, method: str, input_data: RealInput, threads: int
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
        "method": method,
        "tau_max": args.tau_max,
        "timeout_seconds": args.timeout,
        "deferred_methods": DEFERRED_METHODS,
        "input": input_data.metadata,
        "preprocessing": {
            "profile": "primary_detrended_anomaly",
            "day_of_year_anomaly": True,
            "centered_rolling_nanmean_window": ROLLING_WINDOW,
            "finite_standardization": True,
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
        "link_assumptions": _link_assumption_metadata(args.tau_max),
        "missing_data_policy": {
            "sentinel": MISSING_FLAG,
            "remove_missing_upto_maxlag": False,
            "drivers_interpolated": False,
            "rows_dropped": False,
        },
        "settings": real_method_settings(method, args.cmiknn_workers)
        | {"threads": threads},
        "package_versions": pcmci_methods._package_versions(),  # noqa: SLF001
        "git_commit": pcmci_methods.git_commit(),
        "wall_seconds": None,
        "process_max_rss_bytes": None,
        "matrix_shapes": {},
        "result_digest": None,
        "failure_reason": None,
    }


def run(args: argparse.Namespace) -> int:
    if args.output.exists() and not args.overwrite:
        raise ValueError(
            f"Refusing to overwrite existing result: {args.output}; use --overwrite."
        )
    input_data = load_input(args.input, args.row_limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        args.output.write_text("")
    threads = args.threads if args.threads is not None else 1
    summary: dict[str, int] = {}
    for method in args.methods or METHODS:
        print(f"running {method}", file=sys.stderr, flush=True)
        row = _base_row(args, method, input_data, threads)
        row.update(_run_isolated_case(args, method, threads))
        pcmci_methods._append_jsonl(args.output, row)  # noqa: SLF001
        summary[row["status"]] = summary.get(row["status"], 0) + 1
    print(f"{args.output} {json.dumps(summary, sort_keys=True)}")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="python -m thermodense.benchmarks.pcmci_real")
    commands = result.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser(
        "run", help="run real PCMCI+ methods in isolated child processes"
    )
    run_parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--methods", choices=METHODS, nargs="+")
    run_parser.add_argument("--tau-max", type=int, default=DEFAULT_TAU_MAX)
    run_parser.add_argument(
        "--row-limit", type=int, help="calibration-only prefix row limit"
    )
    run_parser.add_argument(
        "--cmiknn-workers", type=int, default=DEFAULT_CMIKNN_WORKERS
    )
    run_parser.add_argument("--threads", type=int)
    run_parser.add_argument("--timeout", type=float, default=1800.0)
    run_parser.add_argument("--host-label", default="unspecified")
    run_parser.add_argument("--environment-label", default="unspecified")
    run_parser.add_argument("--environment-fingerprint", default="unspecified")
    run_parser.add_argument("--overwrite", action="store_true")
    case = commands.add_parser("case", help=argparse.SUPPRESS)
    case.add_argument("--input", type=Path, required=True)
    case.add_argument("--method", choices=METHODS, required=True)
    case.add_argument("--tau-max", type=int, required=True)
    case.add_argument("--row-limit", type=int)
    case.add_argument("--cmiknn-workers", type=int, default=DEFAULT_CMIKNN_WORKERS)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "case":
        return _child_main(args)
    try:
        if (
            args.timeout <= 0
            or args.tau_max < 0
            or args.cmiknn_workers <= 0
            or (args.threads is not None and args.threads <= 0)
        ):
            raise ValueError(
                "--timeout, --cmiknn-workers, and --threads must be positive; --tau-max must be non-negative."
            )
        return run(args)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
