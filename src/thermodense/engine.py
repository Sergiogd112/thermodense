"""Local stage execution with checkpoint reuse."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import secrets
from typing import Callable

from . import checkpoints
from .paths import runs_root
from .workflows import Workflow, WorkflowError, stage_range

StageAction = Callable[[], None]


@dataclass(frozen=True)
class StageSpec:
    inputs: tuple[Path, ...] = ()
    outputs: tuple[Path, ...] = ()
    implementation: tuple[Path, ...] = ()
    action: StageAction | None = None


class LocalExecutionAdapter:
    """Reference adapter; stage actions stay in-process and use no shell."""

    def execute(self, action: StageAction) -> None:
        action()


def fresh_run_id() -> str:
    return f"{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}-{secrets.token_hex(4)}"


def latest_run_id(root: Path, workflow: str) -> str | None:
    directory = runs_root(root) / workflow
    candidates = (
        sorted(path.name for path in directory.iterdir() if path.is_dir())
        if directory.exists()
        else []
    )
    return candidates[-1] if candidates else None


class WorkflowEngine:
    def __init__(self, root: Path, adapter: LocalExecutionAdapter | None = None):
        self.root = root
        self.adapter = adapter or LocalExecutionAdapter()

    def run(
        self,
        workflow: Workflow,
        specs: dict[str, StageSpec],
        *,
        mode: str,
        run_id: str,
        from_stage: str | None = None,
        to_stage: str | None = None,
        dry_run: bool = False,
        force_stages: Iterable[str] = (),
    ) -> list[str]:
        force = set(force_stages)
        unknown = force - set(specs)
        if unknown:
            raise WorkflowError(
                f"No stage specification for: {', '.join(sorted(unknown))}"
            )
        results: list[str] = []
        run_dir = runs_root(self.root) / workflow.name / run_id
        config_hash = checkpoints.sha256_bytes(workflow.raw_config)
        for stage in stage_range(from_stage, to_stage):
            workflow.require_stage(stage)
            spec = specs.get(stage)
            if spec is None or spec.action is None:
                raise WorkflowError(
                    f"No migrated implementation is registered for {workflow.name}:{stage}."
                )
            checkpoint_file = checkpoints.checkpoint_path(run_dir, stage)
            previous = checkpoints.load(checkpoint_file)
            inputs = checkpoints.fingerprints(
                spec.inputs, previous.inputs if previous else ()
            )
            outputs = checkpoints.fingerprints(
                spec.outputs, previous.outputs if previous else ()
            )
            implementation = checkpoints.implementation_fingerprint(spec.implementation)
            if (
                stage not in force
                and previous
                and previous.matches(config_hash, implementation, inputs, outputs)
            ):
                results.append(f"{stage}: cached")
                continue
            if dry_run:
                results.append(f"{stage}: would run")
                continue
            started = checkpoints.now()
            running = checkpoints.Checkpoint(
                workflow.name,
                mode,
                stage,
                "running",
                config_hash,
                implementation,
                inputs,
                outputs,
                started,
                None,
                checkpoints.metadata(),
            )
            checkpoints.write(checkpoint_file, running)
            try:
                self.adapter.execute(spec.action)
                final_outputs = checkpoints.fingerprints(spec.outputs)
                if any(item["state"] == "missing" for item in final_outputs):
                    raise WorkflowError(
                        f"{workflow.name}:{stage} completed without declared outputs."
                    )
            except Exception as error:
                checkpoints.write(
                    checkpoint_file,
                    checkpoints.Checkpoint(
                        workflow.name,
                        mode,
                        stage,
                        "failed",
                        config_hash,
                        implementation,
                        inputs,
                        checkpoints.fingerprints(spec.outputs),
                        started,
                        checkpoints.now(),
                        checkpoints.metadata(),
                        f"{type(error).__name__}: {error}",
                    ),
                )
                raise
            checkpoints.write(
                checkpoint_file,
                checkpoints.Checkpoint(
                    workflow.name,
                    mode,
                    stage,
                    "success",
                    config_hash,
                    implementation,
                    inputs,
                    final_outputs,
                    started,
                    checkpoints.now(),
                    checkpoints.metadata(),
                ),
            )
            results.append(f"{stage}: ran")
        return results
