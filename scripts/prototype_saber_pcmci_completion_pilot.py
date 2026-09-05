"""THROWAWAY isolated PCMCI+ pilot for completed five-channel SABER totals.

This deliberately consumes only the exact-complete stream artifact and the
read-only direct-density analysis bundle.  It is not a production runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "saber-pcmci-completion-pilot-v1"
MISSING_FLAG = -999999.0
ALPHA = 0.05
TAU_MAX = 61 * 8
TARGET_ALTITUDES = (175, 500, 825)
TARGET_INDICES = (0, 13, 26)
CHANNELS = ("CO2cool", "NOcool", "O2_1delta_ver", "OH_16_ver", "OH_20_ver")
STREAM = Path("outputs/prototypes/saber_vertical_totals_40x60_stream")
BUNDLE = Path("outputs/prototypes/density_pcmci_3hour_and_daily/analysis_bundle.npz")
OUTPUT = Path("outputs/prototypes/saber_pcmci_completion_pilot")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def cases() -> tuple[str, str]:
    return ("hasdm_selected-ap", "hasdm_selected-kp")


def preprocess(values: np.ndarray, times: pd.DatetimeIndex) -> np.ndarray:
    """Calendar month/day/slot anomalies, centered three-year mean, finite z-score."""
    series = pd.Series(np.asarray(values, dtype=float))
    key = [times.month, times.day, times.hour // 3]
    seasonal = series - series.groupby(key).transform("mean")
    detrended = (
        seasonal - seasonal.rolling(3 * 365 * 8, center=True, min_periods=1).mean()
    )
    scale = detrended.std(ddof=0, skipna=True)
    return (
        detrended / scale if np.isfinite(scale) and scale > 0 else detrended * np.nan
    ).to_numpy()


def link_assumptions(
    columns: list[str], targets: list[str], tau: int
) -> dict[int, dict[tuple[int, int], str]]:
    """No target-to-target links, including vertical links, by structural assumption."""
    result = {target_i: {} for target_i in range(len(columns))}
    target_set = set(targets)
    for target_i, target in enumerate(columns):
        for cause_i, cause in enumerate(columns):
            driver_pair = target not in target_set and cause not in target_set
            allowed_history = (
                driver_pair
                or (target in target_set and cause not in target_set)
                or cause == target
            )
            if allowed_history:
                for lag in range(1, tau + 1):
                    result[target_i][(cause_i, -lag)] = "-?>"
            if target in target_set and cause not in target_set:
                result[target_i][(cause_i, 0)] = "-?>"
            elif driver_pair and cause_i < target_i:
                result[target_i][(cause_i, 0)] = "o?o"
    return result


def policy() -> dict[str, object]:
    return {
        "prototype_output": "throwaway; never production",
        "targets_km": list(TARGET_ALTITUDES),
        "cadence": "3-hour",
        "tau_max": TAU_MAX,
        "physical_max_lag_days": 61,
        "preprocessing": "calendar month/day/3-hour-slot anomaly; centered 3-year mean detrending; finite z-score",
        "drivers": "f107 plus AP or KP plus five separate SABER vertical-total channels",
        "missing_flag": MISSING_FLAG,
        "remove_missing_upto_maxlag": False,
        "algorithm": "PCMCI+ / ParCorr analytic",
        "pc_alpha": ALPHA,
        "fdr_method": "none",
        "contemp_collider_rule": "majority",
        "conflict_resolution": True,
        "max_conds_dim": None,
        "max_conds_dim_policy": "null/unlimited; omitted from Tigramite call",
        "structural_assumption": "No target-to-target links are permitted, including vertical altitude links.",
    }


def _stream_paths(partial_input: Path | None) -> tuple[Path, Path, bool]:
    if partial_input is None:
        return STREAM / "final_arrays.npz", STREAM / "final_report.json", False
    return partial_input, partial_input.with_name("partial_report.json"), True


def load_sources(
    partial_input: Path | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    arrays_path, report_path, partial = _stream_paths(partial_input)
    if not arrays_path.is_file():
        kind = "explicit partial input" if partial else "exact final stream artifact"
        raise FileNotFoundError(f"{kind} is absent: {arrays_path}")
    if not report_path.is_file():
        raise FileNotFoundError(f"stream completion report is absent: {report_path}")
    report = json.loads(report_path.read_text())
    if not partial and (
        report.get("exact_full_cell") is not True
        or report.get("output_status") != "exact_complete"
    ):
        raise ValueError(
            "refusing stream output not declared exact_full_cell/exact_complete"
        )
    if partial and report.get("output_status") != "partial_not_for_pcmci":
        raise ValueError(
            "explicit partial input must be paired with partial_not_for_pcmci report"
        )
    if not BUNDLE.is_file():
        raise FileNotFoundError(f"read-only analysis bundle is absent: {BUNDLE}")
    with (
        np.load(arrays_path, allow_pickle=False) as stream,
        np.load(BUNDLE, allow_pickle=False) as bundle,
    ):
        required_stream = {
            "timestamps",
            "channels",
            "values",
            "counts",
            "masks",
            "joint_all_five_mask",
        }
        missing = required_stream - set(stream.files)
        if missing:
            raise ValueError(f"stream arrays missing keys: {sorted(missing)}")
        channel_names = tuple(str(value) for value in stream["channels"].tolist())
        if channel_names != CHANNELS:
            raise ValueError(
                f"stream channel names differ from expected {list(CHANNELS)}"
            )
        timestamps = stream["timestamps"].astype("datetime64[ns]")
        hasdm_time_ns = bundle["hasdm_time_ns"].astype(np.int64)
        if not np.array_equal(timestamps.astype(np.int64), hasdm_time_ns):
            raise ValueError("stream timestamps do not exactly match hasdm_time_ns")
        values = np.asarray(stream["values"], dtype=float)
        if values.shape != (len(CHANNELS), len(hasdm_time_ns)):
            raise ValueError(
                "stream values must be exactly five channels by HASDM timestamps"
            )
        targets = np.asarray(bundle["hasdm_targets"], dtype=float)
        if (
            targets.ndim != 2
            or targets.shape[0] != len(hasdm_time_ns)
            or targets.shape[1] < 27
        ):
            raise ValueError("HASDM targets have incompatible selected-altitude matrix")
        source = {
            "time_ns": hasdm_time_ns,
            "targets": targets[:, TARGET_INDICES],
            "f107": np.asarray(bundle["hasdm_f107"], dtype=float),
            "ap": np.asarray(bundle["hasdm_ap"], dtype=float),
            "kp": np.asarray(bundle["hasdm_kp"], dtype=float),
            "saber": values.T,
            "stream_counts": np.asarray(stream["counts"]),
            "stream_masks": np.asarray(stream["masks"]),
            "stream_joint_mask": np.asarray(stream["joint_all_five_mask"]),
        }
    return source, {
        "stream_arrays": str(arrays_path),
        "stream_report": str(report_path),
        "stream_arrays_sha256": sha256(arrays_path),
        "stream_report_sha256": sha256(report_path),
        "bundle": str(BUNDLE),
        "bundle_sha256": sha256(BUNDLE),
        "partial_input": partial,
    }


def diagnostics(values: np.ndarray, columns: list[str]) -> dict[str, object]:
    finite = np.isfinite(values)
    shifted_complete = [
        int((finite[lag:] & finite[: len(values) - lag]).all(axis=1).sum())
        for lag in range(TAU_MAX + 1)
    ]
    return {
        "finite_counts": {
            name: int(finite[:, index].sum()) for index, name in enumerate(columns)
        },
        "complete_rows": int(finite.all(axis=1).sum()),
        "shifted_complete_rows_lag_0_to_tau_max": shifted_complete,
        "joint_saber_finite_rows": int(finite[:, -len(CHANNELS) :].all(axis=1).sum()),
    }


def prepare(
    output: Path = OUTPUT,
    partial_input: Path | None = None,
    preflight_only: bool = False,
) -> Path | None:
    source, source_note = load_sources(partial_input)
    if preflight_only:
        if not source_note["partial_input"]:
            raise ValueError(
                "--preflight-only requires an explicit --partial-input; default input must be exact complete"
            )
        return None
    if source_note["partial_input"]:
        raise ValueError(
            "partial input is permitted only with --preflight-only and is never prepared for PCMCI"
        )
    input_path = output / "pilot_input.npz"
    provenance_path = output / "pilot_input.provenance.json"
    if input_path.exists() or provenance_path.exists():
        raise FileExistsError(
            "prepared pilot input is immutable; choose a new --output directory"
        )
    times = pd.DatetimeIndex(source["time_ns"].astype("datetime64[ns]"))
    target_names = [f"log10rho_{altitude}" for altitude in TARGET_ALTITUDES]
    columns = [*target_names, "f107", "ap", "kp", *CHANNELS]
    raw = np.column_stack(
        [source["targets"], source["f107"], source["ap"], source["kp"], source["saber"]]
    )
    transformed = np.column_stack(
        [preprocess(raw[:, index], times) for index in range(raw.shape[1])]
    )
    note = {
        "version": VERSION,
        "production": False,
        "policy": policy(),
        "sources": source_note,
        "columns": columns,
        "raw_diagnostics": diagnostics(raw, columns),
        "preprocessed_diagnostics": diagnostics(transformed, columns),
    }
    atomic_npz(
        input_path,
        time_ns=source["time_ns"],
        values=transformed,
        columns=np.asarray(columns),
        target_altitudes_km=np.asarray(TARGET_ALTITUDES),
    )
    note["pilot_input_sha256"] = sha256(input_path)
    atomic_json(note, provenance_path)
    return input_path


def load_prepared(output: Path) -> tuple[np.ndarray, list[str], str, dict[str, object]]:
    input_path, note_path = (
        output / "pilot_input.npz",
        output / "pilot_input.provenance.json",
    )
    if not input_path.is_file() or not note_path.is_file():
        raise FileNotFoundError(
            "run requires prepare-created pilot_input.npz and provenance"
        )
    note = json.loads(note_path.read_text())
    input_hash = sha256(input_path)
    if (
        note.get("pilot_input_sha256") != input_hash
        or note.get("production") is not False
    ):
        raise ValueError("prepared pilot input/provenance is altered or incompatible")
    with np.load(input_path, allow_pickle=False) as saved:
        return (
            np.asarray(saved["values"], dtype=float),
            [str(x) for x in saved["columns"].tolist()],
            input_hash,
            note,
        )


def fingerprint(case: str, input_hash: str) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "version": VERSION,
                "case": case,
                "pilot_input_sha256": input_hash,
                "policy": policy(),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()


def run_pcmci(
    values: np.ndarray, columns: list[str], targets: list[str]
) -> dict[str, np.ndarray]:
    from tigramite import data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI

    frame = pp.DataFrame(
        values,
        datatime=np.arange(len(values)),
        var_names=columns,
        missing_flag=MISSING_FLAG,
        remove_missing_upto_maxlag=False,
    )
    return PCMCI(
        frame, cond_ind_test=ParCorr(significance="analytic"), verbosity=0
    ).run_pcmciplus(
        link_assumptions=link_assumptions(columns, targets, TAU_MAX),
        tau_min=0,
        tau_max=TAU_MAX,
        pc_alpha=ALPHA,
        fdr_method="none",
        contemp_collider_rule="majority",
        conflict_resolution=True,
    )


def driver_target_rows(
    result: dict[str, np.ndarray], columns: list[str], targets: list[str], case: str
) -> pd.DataFrame:
    drivers = ["f107", case.rsplit("-", 1)[1], *CHANNELS]
    rows = []
    for driver in drivers:
        for target in targets:
            for lag in range(TAU_MAX + 1):
                location = (columns.index(driver), columns.index(target), lag)
                rows.append(
                    {
                        "source": driver,
                        "target": target,
                        "lag_steps": lag,
                        "lag_hours": lag * 3,
                        "partial_r": float(result["val_matrix"][location]),
                        "raw_p_value": float(result["p_matrix"][location]),
                        "graph_mark": str(result["graph"][location]),
                    }
                )
    return pd.DataFrame(rows)


def reusable(directory: Path, identity: str) -> bool:
    provenance, metadata, arrays, rows = (
        directory / "provenance.json",
        directory / "execution_metadata.json",
        directory / "pcmci_results.npz",
        directory / "driver_target_rows.csv",
    )
    if not all(path.is_file() for path in (provenance, metadata, arrays, rows)):
        return False
    try:
        record = json.loads(provenance.read_text())
        execution = json.loads(metadata.read_text())
        return (
            record.get("production") is False
            and record.get("status") == "completed"
            and record.get("fingerprint") == identity
            and execution.get("status") == "completed"
            and execution.get("fingerprint") == identity
            and record["result_files"]
            == {arrays.name: sha256(arrays), rows.name: sha256(rows)}
        )
    except OSError, ValueError, KeyError, TypeError:
        return False


def run(
    case: str,
    output: Path = OUTPUT,
    host_label: str | None = None,
    recompute: bool = False,
) -> Path:
    values, columns, input_hash, note = load_prepared(output)
    identity, directory = fingerprint(case, input_hash), output / "cases" / case
    if not recompute and reusable(directory, identity):
        return directory
    metadata = directory / "execution_metadata.json"
    started_epoch = time.time()
    atomic_json(
        {
            "production": False,
            "status": "running",
            "case": case,
            "fingerprint": identity,
            "host_label": host_label,
            "started_epoch": started_epoch,
        },
        metadata,
    )
    targets = [f"log10rho_{altitude}" for altitude in TARGET_ALTITUDES]
    selected_columns = [*targets, "f107", case.rsplit("-", 1)[1], *CHANNELS]
    selected = values[:, [columns.index(name) for name in selected_columns]].copy()
    selected[~np.isfinite(selected)] = MISSING_FLAG
    started = time.monotonic()
    try:
        result = run_pcmci(selected, selected_columns, targets)
        stage = "write_results"
        results_path, rows_path = (
            directory / "pcmci_results.npz",
            directory / "driver_target_rows.csv",
        )
        atomic_npz(
            results_path,
            graph=result["graph"],
            p_matrix=result["p_matrix"],
            val_matrix=result["val_matrix"],
            columns=np.asarray(selected_columns),
        )
        atomic_csv(
            driver_target_rows(result, selected_columns, targets, case), rows_path
        )
        atomic_json(
            {
                "version": VERSION,
                "production": False,
                "status": "completed",
                "case": case,
                "fingerprint": identity,
                "pilot_input_sha256": input_hash,
                "input_provenance": note["sources"],
                "host_label": host_label,
                "runtime_seconds": time.monotonic() - started,
                "policy": policy(),
                "finite_counts_after_preprocessing": {
                    name: int((selected[:, i] != MISSING_FLAG).sum())
                    for i, name in enumerate(selected_columns)
                },
                "result_files": {
                    results_path.name: sha256(results_path),
                    rows_path.name: sha256(rows_path),
                },
            },
            directory / "provenance.json",
        )
    except Exception as error:
        atomic_json(
            {
                "production": False,
                "status": "failed",
                "case": case,
                "fingerprint": identity,
                "host_label": host_label,
                "started_epoch": started_epoch,
                "finished_epoch": time.time(),
                "failure_stage": locals().get("stage", "pcmci"),
                "error": f"{type(error).__name__}: {error}",
            },
            metadata,
        )
        raise
    atomic_json(
        {
            "production": False,
            "status": "completed",
            "case": case,
            "fingerprint": identity,
            "host_label": host_label,
            "started_epoch": started_epoch,
            "finished_epoch": time.time(),
        },
        metadata,
    )
    return directory


def status(output: Path = OUTPUT) -> pd.DataFrame:
    rows = []
    for case in cases():
        path = output / "cases" / case / "execution_metadata.json"
        record = (
            json.loads(path.read_text())
            if path.is_file()
            else {"status": "not_started"}
        )
        rows.append(
            {
                "case": case,
                "status": record.get("status"),
                "host_label": record.get("host_label"),
                "error": record.get("error"),
            }
        )
    return pd.DataFrame(rows)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument(
        "--output",
        type=Path,
        default=OUTPUT,
        help="Isolated throwaway pilot output directory",
    )
    subcommands = command.add_subparsers(dest="command", required=True)
    prepare_parser = subcommands.add_parser("prepare")
    prepare_parser.add_argument(
        "--partial-input",
        type=Path,
        help="Explicit partial_arrays.npz, allowed only for --preflight-only",
    )
    prepare_parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate only an explicit partial input; never create PCMCI input",
    )
    run_parser = subcommands.add_parser("run")
    run_parser.add_argument("case", choices=cases())
    run_parser.add_argument(
        "--host-label", required=True, help="Explicit local/remote execution label"
    )
    run_parser.add_argument("--recompute", action="store_true")
    subcommands.add_parser("status")
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.output, args.partial_input, args.preflight_only)
        print("preflight passed" if result is None else result)
    elif args.command == "run":
        print(run(args.case, args.output, args.host_label, args.recompute))
    else:
        print(status(args.output).to_csv(index=False), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
