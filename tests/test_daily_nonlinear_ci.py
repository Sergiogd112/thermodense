from __future__ import annotations

import json

import pytest

from thermodense.benchmarks import daily_nonlinear_ci as benchmark


def test_plan_declares_candidates_and_absent_rcot_adapter_without_importing_it() -> None:
    plan = benchmark.plan_document()

    candidates = {item["name"]: item for item in plan["candidates"]}
    assert candidates["cmiknn"]["availability"] == "available"
    assert candidates["gpdctorch"]["availability"] == "available_with_cuda_gate"
    assert candidates["rcot_rcit"] == {
        "name": "rcot_rcit",
        "availability": "adapter_absent",
        "adapter": None,
        "availability_reason": "RCoT/RCIT has no Tigramite adapter or validated project time-series/null-calibration contract.",
    }
    assert json.loads(json.dumps(plan)) == plan


def test_plan_encodes_full_bounded_matrix_and_promotion_limits() -> None:
    plan = benchmark.plan_document()

    assert len(plan["matrix"]) == 4
    assert {case["tau_max"] for case in plan["matrix"]} == {10}
    assert plan["input_contract"]["shortened_record_forbidden"] is True
    assert plan["promotion_gates"]["all_required"] == list(benchmark.GATE_NAMES)
    assert plan["interpretation"]["tau_max_180"] == "outside_this_benchmark_requires_separate_decision"
    assert plan["interpretation"]["external_method_speed_claim"] == "not made"


def test_result_template_is_machine_readable_and_pending_all_required_gates() -> None:
    result = benchmark.result_template("rcot_rcit")

    assert result["state"] == "not_run"
    assert len(result["matrix_results"]) == 4
    assert all(case["status"] == "not_run" for case in result["matrix_results"])
    assert set(result["promotion"]["gates"]) == set(benchmark.GATE_NAMES)
    assert set(result["promotion"]["gates"].values()) == {"not_assessed"}
    assert json.loads(json.dumps(result)) == result


def test_unknown_candidate_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown daily nonlinear"):
        benchmark.result_template("fastkci")
