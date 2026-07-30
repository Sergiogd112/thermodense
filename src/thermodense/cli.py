"""Public command line interface for thesis-result workflows."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .engine import WorkflowEngine, fresh_run_id, latest_run_id
from .paths import repository_root
from .ssh import SSHExecutionAdapter, SSHProfileError, load_profile
from .workflows import STAGES, WorkflowError, list_workflows, load_workflow, stage_range


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="thermodense")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "workflows", help="list thesis-section workflows and migration state"
    )
    plan = commands.add_parser("plan", help="show a workflow's stage plan")
    plan.add_argument("workflow")
    reproduce = commands.add_parser(
        "reproduce", help="run frozen workflow configuration"
    )
    reproduce.add_argument("workflow")
    _run_options(reproduce)
    refresh = commands.add_parser(
        "refresh", help="refresh from the latest configuration"
    )
    refresh.add_argument("workflow")
    _run_options(refresh)
    stage = commands.add_parser("stage", help="run one stage")
    stage.add_argument("workflow")
    stage.add_argument("stage", choices=STAGES)
    stage.add_argument("--dry-run", action="store_true")
    status = commands.add_parser("status", help="show migration and checkpoint status")
    status.add_argument("workflow", nargs="?")
    return result


def _run_options(command: argparse.ArgumentParser) -> None:
    command.add_argument("--from-stage", choices=STAGES)
    command.add_argument("--to-stage", choices=STAGES)
    command.add_argument("--dry-run", action="store_true")
    command.add_argument("--force-stage", choices=STAGES, action="append", default=[])
    command.add_argument("--backend", choices=("local", "ssh"), default="local")
    command.add_argument("--ssh-profile")
    command.add_argument("--ssh-profile-path", type=Path, help=argparse.SUPPRESS)
    command.add_argument("--run-id", help=argparse.SUPPRESS)


def _print_workflow(workflow) -> None:
    available = ", ".join(workflow.available_stages) or "none"
    print(f"{workflow.name}: {workflow.migration} (available stages: {available})")


def main(argv: list[str] | None = None) -> int:
    try:
        args = parser().parse_args(argv)
        root = repository_root()
        if args.command == "workflows":
            for workflow in list_workflows(root):
                _print_workflow(workflow)
            return 0
        if args.command == "plan":
            workflow = load_workflow(args.workflow, root)
            _print_workflow(workflow)
            for stage in STAGES:
                state = (
                    "available" if stage in workflow.available_stages else "unavailable"
                )
                print(f"  {stage}: {state}")
            return 0
        if args.command == "status":
            workflows = (
                [load_workflow(args.workflow, root)]
                if args.workflow
                else list_workflows(root)
            )
            for workflow in workflows:
                _print_workflow(workflow)
                run = latest_run_id(root, workflow.name)
                print(f"  latest checkpoint run: {run or 'none'}")
            return 0
        workflow = load_workflow(args.workflow, root)
        if args.command == "stage":
            workflow.require_stage(args.stage)
            raise WorkflowError(
                f"No migrated implementation is registered for {workflow.name}:{args.stage}."
            )
        selected = stage_range(args.from_stage, args.to_stage)
        for name in selected:
            workflow.require_stage(name)
        if args.backend == "ssh":
            if not args.ssh_profile:
                raise WorkflowError("--ssh-profile is required with --backend ssh.")
            profile = load_profile(args.ssh_profile, args.ssh_profile_path)
            run_id = args.run_id or (
                "thesis" if args.command == "reproduce" else fresh_run_id()
            )
            adapter = SSHExecutionAdapter(profile)
            remote_args = [
                args.command,
                args.workflow,
                "--backend",
                "local",
                "--run-id",
                run_id,
            ]
            for option in ("from_stage", "to_stage"):
                if getattr(args, option):
                    remote_args.extend(
                        [f"--{option.replace('_', '-')}", getattr(args, option)]
                    )
            if args.dry_run:
                remote_args.append("--dry-run")
            for stage in args.force_stage:
                remote_args.extend(["--force-stage", stage])
            adapter.run(
                root,
                workflow.name,
                run_id,
                remote_args,
            )
            return 0
        run_id = args.run_id or (
            "thesis" if args.command == "reproduce" else fresh_run_id()
        )
        # No thesis stage is migrated in this foundation PR; checks above deliberately fail first.
        results = WorkflowEngine(root).run(
            workflow,
            {},
            mode=args.command,
            run_id=run_id,
            from_stage=args.from_stage,
            to_stage=args.to_stage,
            dry_run=args.dry_run,
            force_stages=args.force_stage,
        )
        print("\n".join(results))
        return 0
    except (WorkflowError, SSHProfileError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
