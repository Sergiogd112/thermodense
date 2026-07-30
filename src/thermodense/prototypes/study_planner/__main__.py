"""Terminal-only interface for the throwaway issue #7 planner prototype."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .core import Controller, Plan, load_pools, load_toml, plan_study

PAGE_SIZE = 16

def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Throwaway study planner prototype; never executes jobs.")
    result.add_argument("--study", type=Path, default=Path("configs/prototypes/study-planner.toml"))
    result.add_argument("--executors", type=Path, default=Path("configs/prototypes/executor-pools.toml"))
    result.add_argument("--dump", action="store_true", help="print deterministic state and exit")
    result.add_argument("--ticks", type=int, default=0, help="advance this many simulated dispatch ticks")
    result.add_argument("--execution-id", default="issue-7-preview", help="conceptual execution grouping only")
    return result


def render(plan: Plan, controller: Controller, execution_id: str, view: str = "summary") -> str:
    if view == "summary":
        statuses = {status: sum(job.status == status for job in controller.jobs) for status in ("queued", "running", "completed", "gated")}
        return "\n".join((
            "THROWAWAY study planner — no science or jobs are executed",
            f"study={plan.study_id} execution={execution_id} solar_rows={plan.solar_variant_rows}",
            f"runs/{plan.study_id}/{execution_id} outputs/{plan.study_id}/{execution_id} (conceptual only)",
            f"cases={len(plan.cases)} artifacts={len(plan.artifacts)} jobs={len(plan.jobs)} tick={controller.tick_number}",
            " ".join(f"{name}={count}" for name, count in statuses.items()),
            "durable state: job ID, status, attempt, executor, and remaining ticks are serializable/reconcilable",
        ))
    if view == "artifacts":
        return "\n".join(["ARTIFACT DAG (immutable deterministic fingerprints)"] + [
            f"{item.id} [{item.kind}: {item.label}] <- {', '.join(item.inputs) or 'source'}" for item in plan.artifacts
        ])
    if view == "cases":
        return "\n".join(["CASES (one target family per expanded run)"] + [
            f"{item.id}: {item.target_kind}/{item.target_value} {item.analysis}; "
            f"test={item.independence_test or '-'} cadence={item.cadence}; "
            f"{item.solar_proxy}+{item.geomagnetic_driver} rows={item.solar_variant_rows}; "
            f"{item.case_type}; {item.window}; "
            f"extension={item.extension or '-'}; "
            f"altitudes={item.altitude_group or 'all'} profile={item.preprocessing_profile or '-'}; "
            f"lags={item.physical_lags or '-'} -> steps={item.lag_steps or '-'}"
            for item in plan.cases
        ])
    if view == "jobs":
        return "\n".join(["DURABLE CONTROLLER QUEUE"] + [
            f"{job.id} {job.status} attempt={job.attempt} class={job.resource_class} "
            f"executor={job.executor_id or '-'} remaining={job.remaining_ticks or '-'} "
            f"prototype-spec=cpu_slots:{job.cpu_slots},gpu:{job.gpu_required}; "
            f"capabilities={','.join(job.required_capabilities) or '-'} case={job.case_id}"
            f"{' (' + job.blocked_reason + ')' if job.blocked_reason else ''}"
            for job in controller.jobs
        ])
    return "\n".join(["EXECUTOR POOLS (compatible candidates for queued jobs)"] + [
            f"{pool.id} ({pool.adapter_kind}): slots={pool.slots}, quota={pool.quota_group}/{pool.quota_limit}, "
            f"enabled={pool.enabled}, available={pool.available}, min_ticks={pool.minimum_job_ticks}, "
            f"overhead={pool.startup_ticks:g}+{pool.staging_ticks:g}, classes={','.join(pool.resource_classes)}, "
        f"speed={pool.tick_multipliers}, candidates="
        f"{','.join(f'{job.id}@{controller.projected_completion(job, pool):g}' for job in controller.jobs if job.status == 'queued' and pool in controller.candidates(job)) or '-'}"
        for pool in controller.pools
    ])


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    pools = load_pools(args.executors)
    plan = plan_study(load_toml(args.study), pools)
    controller = Controller(jobs=[replace(job) for job in plan.jobs], pools=pools)
    for _ in range(args.ticks):
        controller.tick()
    if args.dump:
        for view in ("summary", "artifacts", "cases", "jobs", "executors"):
            print(render(plan, controller, args.execution_id, view))
        reconciled = Controller.reconcile(controller.state(), pools)
        print(f"RECONCILE: tick={reconciled.tick_number} jobs={len(reconciled.jobs)}")
        return 0
    _interactive(plan, controller, args.execution_id)
    return 0


def _interactive(plan: Plan, controller: Controller, execution_id: str) -> None:
    view = "summary"
    page = 0
    while True:
        _clear_terminal()
        content = render(plan, controller, execution_id, view)
        print(_page(content, view, page))
        command = input("[s]ummary [c]ases [a]rtifacts [j]obs [e]xecutors [n]ext [p]revious [t]ick [f]ail [r]eset [q]uit: ").strip().lower()
        if command == "q":
            return
        if command == "t":
            controller.tick()
        elif command == "f":
            controller.fail_first_running()
        elif command == "r":
            controller.jobs = [replace(job) for job in plan.jobs]
            controller.tick_number = 0
        elif command in {"s", "c", "a", "j", "e"}:
            view = {"s": "summary", "c": "cases", "a": "artifacts", "j": "jobs", "e": "executors"}[command]
            page = 0
        elif command == "n":
            page = min(page + 1, _page_count(content) - 1)
        elif command == "p":
            page = max(0, page - 1)


def _clear_terminal() -> None:
    print("\033[2J\033[H", end="")


def _page(content: str, view: str, page: int) -> str:
    lines = content.splitlines()
    header, body = lines[:1], lines[1:]
    page_count = _page_count(content)
    page = min(page, page_count - 1)
    start = page * PAGE_SIZE
    return "\n".join([*header, *body[start : start + PAGE_SIZE], f"[{view} page {page + 1}/{page_count}]"])


def _page_count(content: str) -> int:
    return max(1, (len(content.splitlines()) - 1 + PAGE_SIZE - 1) // PAGE_SIZE)


if __name__ == "__main__":
    raise SystemExit(main())
