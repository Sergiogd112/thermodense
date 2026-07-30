from pathlib import Path
import subprocess


def test_tracked_repository_root_uses_the_code_first_allowlist() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    allowed = {
        "AGENTS.md",
        "CONTEXT.md",
        "LICENSE.txt",
        "README.md",
        "THIRD_PARTY_NOTICES.md",
        "opencode.json",
        "pyproject.toml",
        "uv.lock",
    }
    tracked_files = subprocess.run(
        ["git", "ls-files"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    offenders = sorted(
        path
        for path in tracked_files
        if "/" not in path
        and path not in {".gitignore", ".python-version"}
        and path not in allowed
    )

    assert offenders == []


def test_tracked_tree_excludes_private_and_generated_artifacts() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    forbidden_prefixes = (
        "data/",
        "kaggle/",
        "outputs/",
        "papers/",
        "presentation/",
        "runs/",
        "thesis/",
    )
    forbidden_suffixes = (
        ".csv",
        ".html",
        ".jpg",
        ".jpeg",
        ".parquet",
        ".pdf",
        ".png",
        ".whl",
        ".zip",
    )
    tracked_files = subprocess.run(
        ["git", "ls-files"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    offenders = sorted(
        path
        for path in tracked_files
        if path.startswith(forbidden_prefixes)
        or path.lower().endswith(forbidden_suffixes)
        or "cookie" in Path(path).name.lower()
        or Path(path).name == ".env"
    )

    assert offenders == []


def test_retained_scripts_only_use_outputs_for_generated_figures() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    forbidden_references = (
        "thesis/figures/results",
        'Path("thesis")',
        "Path('thesis')",
    )

    offenders = sorted(
        path.relative_to(repository_root).as_posix()
        for path in (repository_root / "scripts").rglob("*.py")
        if any(
            reference in path.read_text(encoding="utf-8")
            for reference in forbidden_references
        )
    )

    assert offenders == []
