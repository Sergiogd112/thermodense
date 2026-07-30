"""Canonical repository paths for workflow artifacts."""

from pathlib import Path


class WorkflowError(RuntimeError):
    """A repository checkout or workflow configuration is unavailable."""


def repository_root(start: Path | None = None) -> Path:
    """Return the repository root for an explicit path, checkout, or source tree."""
    starts = (start,) if start is not None else (Path.cwd(), Path(__file__))
    for start_path in starts:
        current = start_path.resolve()
        for candidate in (current, *current.parents):
            if (candidate / "pyproject.toml").is_file():
                return candidate
    raise WorkflowError(
        "Run thermodense from a repository checkout (missing pyproject.toml)."
    )


def source_root(root: Path | None = None) -> Path:
    return (root or repository_root()) / "data" / "sources"


def prepared_root(root: Path | None = None) -> Path:
    return (root or repository_root()) / "data" / "prepared"


def product_root(root: Path | None = None) -> Path:
    return (root or repository_root()) / "data" / "products"


def runs_root(root: Path | None = None) -> Path:
    return (root or repository_root()) / "runs"


def publication_root(root: Path | None = None) -> Path:
    """Return the ignored publication-output destination for current workflows."""
    return (root or repository_root()) / "outputs" / "figures" / "results"
