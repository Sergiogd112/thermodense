"""Workflow definitions loaded from frozen thesis TOML configurations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib

from .paths import WorkflowError, repository_root

STAGES = ("acquire", "prepare", "derive", "analyse", "publish")
THESIS_WORKFLOWS = (
    "global-mean",
    "tudelft-density",
    "hasdm-saber",
    "maunaloa-msis-baselines",
    "model-errors",
    "synthesis",
    "all",
)


class UnavailableStageError(WorkflowError):
    pass


@dataclass(frozen=True)
class Workflow:
    name: str
    description: str
    migration: str
    available_stages: tuple[str, ...]
    config_path: Path
    raw_config: bytes

    def require_stage(self, stage: str) -> None:
        if stage not in STAGES:
            raise WorkflowError(
                f"Unknown stage {stage!r}; expected one of {', '.join(STAGES)}."
            )
        if stage not in self.available_stages:
            raise UnavailableStageError(
                f"{self.name}:{stage} is unavailable: {self.migration}. "
                "This foundation PR does not claim to reproduce unmigrated stages."
            )


def config_directory(root: Path | None = None) -> Path:
    return (root or repository_root()) / "configs" / "thesis"


def load_workflow(name: str, root: Path | None = None) -> Workflow:
    if name not in THESIS_WORKFLOWS:
        raise WorkflowError(f"Unknown workflow {name!r}.")
    path = config_directory(root) / f"{name}.toml"
    try:
        raw = path.read_bytes()
        config = tomllib.loads(raw.decode("utf-8"))
    except FileNotFoundError as error:
        raise WorkflowError(f"Missing frozen workflow configuration: {path}") from error
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
        raise WorkflowError(
            f"Malformed workflow configuration: {path}: {error}"
        ) from error
    if not isinstance(config, dict):
        raise WorkflowError(
            f"Malformed workflow configuration: {path}: expected a TOML table."
        )
    required = ("workflow", "description", "migration", "available_stages")
    missing = [key for key in required if key not in config]
    if missing:
        raise WorkflowError(
            f"Malformed workflow configuration: {path}: missing {', '.join(missing)}."
        )
    if not all(isinstance(config[key], str) for key in required[:3]) or not isinstance(
        config["available_stages"], list
    ):
        raise WorkflowError(
            f"Malformed workflow configuration: {path}: invalid field types."
        )
    stages = tuple(config.get("available_stages", []))
    if not all(isinstance(stage, str) for stage in stages):
        raise WorkflowError(
            f"Malformed workflow configuration: {path}: stages must be strings."
        )
    invalid = set(stages) - set(STAGES)
    if invalid:
        raise WorkflowError(f"Invalid stages in {path}: {', '.join(sorted(invalid))}")
    if config["workflow"] != name:
        raise WorkflowError(
            f"Malformed workflow configuration: {path}: workflow must be {name!r}."
        )
    return Workflow(
        name=config["workflow"],
        description=config["description"],
        migration=config["migration"],
        available_stages=stages,
        config_path=path,
        raw_config=raw,
    )


def list_workflows(root: Path | None = None) -> list[Workflow]:
    return [load_workflow(name, root) for name in THESIS_WORKFLOWS]


def stage_range(start: str | None, end: str | None) -> tuple[str, ...]:
    start = start or STAGES[0]
    end = end or STAGES[-1]
    try:
        first, last = STAGES.index(start), STAGES.index(end)
    except ValueError as error:
        raise WorkflowError(f"Stages must be one of: {', '.join(STAGES)}.") from error
    if first > last:
        raise WorkflowError("--from-stage must not come after --to-stage.")
    return STAGES[first : last + 1]
