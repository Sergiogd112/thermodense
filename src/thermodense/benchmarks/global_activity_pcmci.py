"""Daily PCMCI+ retrospective association/sensitivity analysis by activity group."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path
from typing import Any, cast

import numpy as np

from thermodense.benchmarks import global_activity_input
from thermodense.benchmarks.global_activity_input import GlobalActivityInput, atomic_json, group_mask, load_manifest

VERSION = "global-activity-pcmci-v1"
MISSING_FLAG = -999999.0
ALPHA = 0.05
TAU_MAX = 180


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **cast(dict[str, Any], arrays))
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _package_identity(name: str) -> dict[str, Any]:
    distribution = importlib.metadata.distribution(name)
    direct_url = cast(Path, distribution.locate_file("direct_url.json"))
    result: dict[str, Any] = {"version": distribution.version}
    if direct_url.is_file():
        result["direct_url"] = json.loads(direct_url.read_text(encoding="utf-8"))
    return result


def link_assumptions(node_names: list[str], tau_max: int) -> dict[int, dict[tuple[int, int], str]]:
    """Permit density↔density and drivers→density, never density→driver."""
    densities = set(node_names[:10])
    drivers = set(node_names[10:])
    result = {target: {} for target in range(len(node_names))}
    for target_i, target in enumerate(node_names):
        if target not in densities:
            continue
        for source_i, source in enumerate(node_names):
            if source not in densities | drivers:
                continue
            for lag in range(1, tau_max + 1):
                result[target_i][(source_i, -lag)] = "-?>"
            if source in drivers:
                result[target_i][(source_i, 0)] = "-?>"
            elif source_i < target_i:
                result[target_i][(source_i, 0)] = "o?o"
    return result


def eligible_locations(node_names: list[str], tau_max: int) -> list[tuple[int, int, int]]:
    return sorted((source, target, -lag) for target, links in link_assumptions(node_names, tau_max).items() for (source, lag) in links)


def bh_qvalues(pvalues: np.ndarray) -> np.ndarray:
    if not np.isfinite(pvalues).all() or np.any((pvalues < 0) | (pvalues > 1)):
        raise ValueError("BH family requires finite p-values in [0, 1]")
    order = np.argsort(pvalues)
    sorted_values = pvalues[order] * len(pvalues) / np.arange(1, len(pvalues) + 1)
    adjusted = np.minimum.accumulate(sorted_values[::-1])[::-1].clip(0, 1)
    result = np.empty_like(pvalues)
    result[order] = adjusted
    return result


def _tracing_parcorr(trace_path: Path):
    from tigramite.independence_tests.parcorr import ParCorr

    class TracingParCorr(ParCorr):
        test_count = 0

        def run_test(self, X, Y, Z=None, tau_max=0, cut_off="2xtau_max", alpha_or_thres=None):  # type: ignore[no-untyped-def]
            array_result = cast(tuple[Any, Any, Any, Any, Any, Any, Any, Any], self._get_array(X=X, Y=Y, Z=Z, tau_max=tau_max, cut_off=cut_off, remove_constant_data=True, verbosity=self.verbosity))
            _array, _xyz, _XYZ, _type, nonzero_array, _nonzero_xyz, nonzero_XYZ, _nonzero_type = array_result
            self.test_count += 1
            def coordinates(values: Any) -> list[list[int]]:
                return [[int(node), int(lag)] for node, lag in values]

            record: dict[str, Any] = {"test_id": self.test_count, "X": coordinates(X), "Y": coordinates(Y), "Z": coordinates(Z or []), "sample_count": int(nonzero_array.shape[1]), "conditioning_dimension": len(nonzero_XYZ[2]), "tested": bool(nonzero_XYZ[0] and nonzero_XYZ[1]), "nonconstant_XYZ": [coordinates(nodes) for nodes in nonzero_XYZ], "tau_max": int(tau_max), "cut_off": cut_off, "alpha_or_thres": None if alpha_or_thres is None else float(alpha_or_thres)}
            try:
                outcome = cast(tuple[float, float] | tuple[float, float, bool], super().run_test(X, Y, Z, tau_max, cut_off, alpha_or_thres))
                record["partial_r"], record["p_value"] = (float(outcome[0]), float(outcome[1]))
                if len(outcome) == 3:
                    record["dependent"] = bool(outcome[2])
                return outcome
            except Exception as error:
                record["error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                with trace_path.open("a", encoding="utf-8") as trace:
                    trace.write(json.dumps(record) + "\n")
                    trace.flush()
                if self.test_count % 1000 == 0:
                    print(f"CI tests completed: {self.test_count}", flush=True)

    return TracingParCorr(significance="analytic", mask_type="y")


def run_pcmci(input_data: GlobalActivityInput, group: str, tau_max: int, trace_path: Path) -> dict[str, np.ndarray]:
    """Mask only Y/current target dates; lagged X/Z histories remain on-calendar."""
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI

    if tau_max < 0:
        raise ValueError("tau_max must be non-negative")
    values = input_data.values.copy()
    values[~input_data.available] = MISSING_FLAG
    selected = group_mask(input_data.activity_f107_center81, group)
    unsupported = [input_data.node_names[index] for index in range(10) if not np.any(selected[2 * tau_max:] & input_data.available[2 * tau_max:, index])]
    if unsupported:
        raise ValueError(f"group={group} has no eligible target support after tau_max={tau_max}: {', '.join(unsupported)}")
    # Tigramite's mask convention is True = exclude. mask_type='y' applies this
    # only to the contemporaneous target row, not its lagged predictors.
    mask = np.broadcast_to(~selected[:, None], values.shape).copy()
    frame = pp.DataFrame(values, mask=mask, missing_flag=MISSING_FLAG, datatime=np.arange(len(values)), var_names=input_data.node_names, remove_missing_upto_maxlag=False)
    return cast(dict[str, np.ndarray], PCMCI(frame, cond_ind_test=_tracing_parcorr(trace_path), verbosity=0).run_pcmciplus(link_assumptions=link_assumptions(input_data.node_names, tau_max), tau_min=0, tau_max=tau_max, pc_alpha=ALPHA, fdr_method="none", contemp_collider_rule="majority", conflict_resolution=True))


def result_rows(result: dict[str, np.ndarray], input_data: GlobalActivityInput, group: str) -> list[dict[str, Any]]:
    locations = eligible_locations(input_data.node_names, result["p_matrix"].shape[2] - 1)
    pvalues = np.array([result["p_matrix"][source, target, lag] for source, target, lag in locations], dtype=float)
    qvalues = bh_qvalues(pvalues)
    rows: list[dict[str, Any]] = []
    for (source, target, lag), qvalue in zip(locations, qvalues, strict=True):
        mark = str(result["graph"][source, target, lag])
        out_source, out_target = source, target
        direction, ambiguity, normalized_mark = "source_to_target", False, mark
        if lag == 0 and mark == "<--":
            out_source, out_target, direction, normalized_mark = target, source, "target_to_source", "-->"
        elif lag == 0 and mark not in {"-->", "<--"}:
            direction, ambiguity = "ambiguous", True
        rows.append({"group": group, "graph_source": input_data.node_names[source], "graph_target": input_data.node_names[target], "source": input_data.node_names[out_source], "target": input_data.node_names[out_target], "lag_days": lag, "partial_r": float(result["val_matrix"][source, target, lag]), "raw_p_value": float(pvalues[len(rows)]), "q_value": float(qvalue), "graph_mark_original": mark, "graph_mark_normalized": normalized_mark, "adjacency": bool(mark), "direction": direction, "ambiguous": ambiguity, "detected": bool(qvalue <= ALPHA and bool(mark))})
    return rows


def run(manifest_path: Path, output: Path, group: str = "all", tau_max: int = TAU_MAX, *, fixture: bool = False) -> Path:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output {output}")
    if tau_max != TAU_MAX and not fixture:
        raise ValueError("a non-default tau_max is allowed only with --fixture")
    output.mkdir(parents=True)
    started = time.monotonic()
    status: dict[str, Any] = {"version": VERSION, "status": "running", "group": group, "tau_max": tau_max, "interpretation": "retrospective association/sensitivity only; not causal qualification"}
    atomic_json(output / "status.json", status)
    try:
        input_data = load_manifest(manifest_path)
        trace_path = output / "ci_test_trace.jsonl"
        print("phase: PCMCI+ association analysis", flush=True)
        result = run_pcmci(input_data, group, tau_max, trace_path)
        print("phase: persist raw matrices", flush=True)
        raw_path = output / "pcmci_raw.npz"
        _atomic_npz(raw_path, graph=result["graph"], p_matrix=result["p_matrix"], val_matrix=result["val_matrix"], node_names=np.asarray(input_data.node_names))
        rows = result_rows(result, input_data, group)
        print("phase: export BH family", flush=True)
        csv_path = output / "eligible_links.csv"
        _atomic_csv(csv_path, rows)
        status |= {"status": "complete", "runtime_seconds": time.monotonic() - started, "args": {"manifest": str(manifest_path), "output": str(output), "group": group, "tau_max": tau_max, "fixture": fixture}, "input_identity": {"manifest": str(manifest_path), "manifest_sha256": _sha256(manifest_path), "core_sha256": input_data.manifest["core_sha256"]}, "algorithm": {"name": "PCMCI+", "ci_test": "analytic ParCorr", "mask_type": "y", "missing_flag": MISSING_FLAG, "remove_missing_upto_maxlag": False}, "bh_family": {"scope": f"unique eligible links for group={group} only", "test_count": len(rows), "method": "Benjamini-Hochberg"}, "code": {"pcmci_module_sha256": _sha256(Path(__file__)), "input_module_sha256": _sha256(Path(global_activity_input.__file__))}, "versions": {name: _package_identity(name) for name in ("numpy", "scipy", "tigramite")}, "result_files": {raw_path.name: _sha256(raw_path), csv_path.name: _sha256(csv_path), trace_path.name: _sha256(trace_path)}}
        with trace_path.open(encoding="utf-8") as trace:
            status["ci_call_count"] = sum(1 for _ in trace)
        status["algorithm"].update(pc_alpha=ALPHA, bh_alpha=ALPHA, fdr_method="none", contemp_collider_rule="majority", conflict_resolution=True, default_cut_off="2xtau_max")
        atomic_json(output / "status.json", status)
    except Exception as error:
        status |= {"status": "failed", "error": f"{type(error).__name__}: {error}", "runtime_seconds": time.monotonic() - started}
        atomic_json(output / "status.json", status)
        raise
    return output


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--group", choices=["all", "low", "moderate", "high"], default="all")
    command.add_argument("--tau-max", type=int, default=TAU_MAX)
    command.add_argument("--fixture", action="store_true", help="allow a smaller explicit tau_max for test fixtures")
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(run(args.manifest, args.output, args.group, args.tau_max, fixture=args.fixture))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
