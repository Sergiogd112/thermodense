from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tomllib

import numpy as np

from thermodense.benchmarks import pcmci_methods as benchmark

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO_ROOT / "benchmarks" / "pcmci-methods" / "spec.toml"


def test_synthetic_generation_is_deterministic_and_standardized() -> None:
    first = benchmark.generate_synthetic_data(32, level_index=0)
    second = benchmark.generate_synthetic_data(32, level_index=0)
    other_level = benchmark.generate_synthetic_data(32, level_index=1)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other_level)
    assert first.shape == (32, 5)
    assert np.allclose(first.mean(axis=0), 0.0)
    assert np.allclose(first.std(axis=0), 1.0)
    assert not np.isnan(first).any()


def test_plan_is_exact_and_excludes_deferred_gpdctorch() -> None:
    document = benchmark.plan_document()
    assert [(case.method, case.level) for case in benchmark.benchmark_plan()] == [
        (method, level)
        for method in ("parcorr", "cmiknn", "gpdc")
        for level in ("small", "medium", "representative")
    ]
    assert len(document["cases"]) == 9
    assert "gpdctorch" not in document["methods"]
    assert "gpdctorch" in document["deferred_methods"]


def test_spec_toml_parity_with_module_constants() -> None:
    with SPEC_PATH.open("rb") as handle:
        spec = tomllib.load(handle)
    assert spec["benchmark_version"] == benchmark.BENCHMARK_VERSION
    assert spec["schema_version"] == benchmark.SCHEMA_VERSION
    assert spec["seed"] == benchmark.SEED
    assert spec["nodes"] == benchmark.NODES
    assert spec["method_order"] == list(benchmark.METHODS)
    assert spec["default_timeout_seconds"] == 1800
    assert spec["default_threads"] == 1
    assert [
        (level["name"], level["samples"], level["tau_max"]) for level in spec["levels"]
    ] == [(name, samples, tau_max) for name, samples, tau_max in benchmark.LEVELS]
    for name, settings in spec["methods"].items():
        assert benchmark.method_settings(name) == {
            "pc_alpha": 0.05,
            "alpha_level": 0.05,
            **settings,
        }
    document = benchmark.plan_document()
    assert spec["deferred"]["gpdctorch"]["reason"] == document["deferred_methods"]["gpdctorch"]
    assert document["spec_digest"] is not None
    assert len(document["spec_digest"]) == 64


def test_progressive_stop_skips_only_the_failed_method(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_case(case, timeout, threads):
        calls.append((case.method, case.level))
        if case.method == "cmiknn" and case.level == "small":
            return {"status": "failed", "failure_reason": "fixture failure"}
        return {"status": "succeeded", "matrix_shapes": {}, "result_digest": "x"}

    monkeypatch.setattr(benchmark, "_run_isolated_case", fake_case)
    args = _run_args(tmp_path / "results.jsonl", methods=["cmiknn", "gpdc"])
    assert benchmark.run_benchmark(args) == 0
    rows = _rows(args.output)
    assert [row["status"] for row in rows] == [
        "failed",
        "skipped",
        "skipped",
        "succeeded",
        "succeeded",
        "succeeded",
    ]
    assert calls == [("cmiknn", "small"), ("gpdc", "small"), ("gpdc", "medium"), ("gpdc", "representative")]
    assert "previous small case failed" in rows[1]["skip_reason"]


def test_result_schema_and_digest_are_stable_for_mocked_case(
    tmp_path: Path, monkeypatch
) -> None:
    matrices = {
        "p_matrix": np.array([[[0.25]]]),
        "val_matrix": np.array([[[0.5]]]),
    }
    expected_digest = benchmark.compact_result_digest(matrices)

    def fake_case(case, timeout, threads):
        return {
            "status": "succeeded",
            "matrix_shapes": {name: list(value.shape) for name, value in matrices.items()},
            "result_digest": expected_digest,
            "process_max_rss_bytes": 1024,
        }

    monkeypatch.setattr(benchmark, "_run_isolated_case", fake_case)
    args = _run_args(tmp_path / "result.jsonl", methods=["parcorr"], levels=["small"])
    benchmark.run_benchmark(args)
    row = _rows(args.output)[0]
    assert row["schema_version"] == "1"
    assert row["synthetic"] is True
    assert row["matrix_shapes"] == {"p_matrix": [1, 1, 1], "val_matrix": [1, 1, 1]}
    assert row["result_digest"] == expected_digest
    assert benchmark.compact_result_digest(matrices) == expected_digest
    assert "git_commit" in row
    assert row["spec_digest"] == benchmark.spec_digest()


def test_isolated_case_timeout_uses_harmless_child() -> None:
    case = benchmark.Case("parcorr", "small", 512, 7)
    result = benchmark._run_isolated_case(
        case,
        timeout=0.05,
        threads=1,
        command=[sys.executable, "-c", "import time; time.sleep(10)"],
    )
    assert result["status"] == "timeout"
    assert "exceeded timeout" in result["failure_reason"]


def test_run_refuses_to_overwrite_existing_result(tmp_path: Path, capsys) -> None:
    output = tmp_path / "existing.jsonl"
    output.write_text("existing\n")
    assert benchmark.main(["run", "--output", str(output), "--methods", "parcorr"]) == 2
    assert output.read_text() == "existing\n"
    assert "Refusing to overwrite" in capsys.readouterr().err


def _run_args(
    output: Path, methods: list[str], levels: list[str] | None = None
) -> argparse.Namespace:
    return argparse.Namespace(
        output=output,
        overwrite=False,
        methods=methods,
        levels=levels,
        threads=None,
        timeout=1.0,
        host_label="test-host",
        environment_label="test-environment",
        environment_fingerprint="test-fingerprint",
    )


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]
