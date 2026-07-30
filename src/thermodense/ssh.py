"""Optional key-authenticated SSH sync-and-return execution adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import subprocess
import tomllib


class SSHProfileError(RuntimeError):
    pass


_HOST = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?")
_USER = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*")
_REMOTE_ROOT = re.compile(r"/[A-Za-z0-9._/-]*")
_WORKFLOW = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_RUN_ID = re.compile(r"(?:thesis|\d{8}T\d{12}Z-[0-9a-f]{8})")


def _validate_profile(profile: "SSHProfile") -> None:
    if not _HOST.fullmatch(profile.host):
        raise SSHProfileError("SSH profile host contains unsafe characters.")
    if profile.user and not _USER.fullmatch(profile.user):
        raise SSHProfileError("SSH profile user contains unsafe characters.")
    root = profile.remote_root
    if not _REMOTE_ROOT.fullmatch(root) or "//" in root or ".." in root.split("/"):
        raise SSHProfileError("SSH profile remote_root must be a safe absolute path.")


def _validate_run(workflow: str, run_id: str) -> None:
    if not _WORKFLOW.fullmatch(workflow):
        raise SSHProfileError("Workflow contains unsafe characters.")
    if not _RUN_ID.fullmatch(run_id):
        raise SSHProfileError("Run ID contains unsafe characters.")


@dataclass(frozen=True)
class SSHProfile:
    host: str
    remote_root: str
    user: str | None = None
    identity_file: str | None = None

    @property
    def destination(self) -> str:
        return f"{self.user}@{self.host}" if self.user else self.host


def load_profile(name: str, path: Path | None = None) -> SSHProfile:
    path = path or Path.home() / ".config" / "thermodense" / "ssh-profiles.toml"
    try:
        data = tomllib.loads(path.read_text())
        values = data["profiles"][name]
    except FileNotFoundError as error:
        raise SSHProfileError(f"SSH profile file not found: {path}") from error
    except KeyError as error:
        raise SSHProfileError(f"SSH profile {name!r} not found in {path}") from error
    except (tomllib.TOMLDecodeError, TypeError, UnicodeDecodeError) as error:
        raise SSHProfileError(f"Malformed SSH profile file: {path}: {error}") from error
    if not isinstance(values, dict):
        raise SSHProfileError(f"SSH profile {name!r} in {path} must be a table.")
    required = {"host", "remote_root"}
    missing = required - set(values)
    if missing:
        raise SSHProfileError(
            f"SSH profile {name!r} lacks: {', '.join(sorted(missing))}"
        )
    profile = SSHProfile(
        **{key: values[key] for key in SSHProfile.__dataclass_fields__ if key in values}
    )
    if not all(
        isinstance(value, str)
        for value in (
            profile.host,
            profile.remote_root,
            profile.user,
            profile.identity_file,
        )
        if value is not None
    ):
        raise SSHProfileError(f"SSH profile {name!r} contains non-string values.")
    _validate_profile(profile)
    return profile


class SSHExecutionAdapter:
    """Sync an isolated source tree, execute remotely, then return its run data."""

    def __init__(self, profile: SSHProfile, runner=subprocess.run):
        self.profile = profile
        self.runner = runner

    def commands(
        self, repository: Path, workflow: str, run_id: str
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        _validate_profile(self.profile)
        _validate_run(workflow, run_id)
        remote_dir = f"{self.profile.remote_root.rstrip('/')}/{workflow}/{run_id}"
        ssh = ["ssh", "-o", "BatchMode=yes"]
        if self.profile.identity_file:
            ssh.extend(["-i", self.profile.identity_file])
        mkdir = [*ssh, self.profile.destination, "mkdir", "-p", remote_dir]
        push = [
            "rsync",
            "--archive",
            "--delete",
            "--exclude",
            ".git",
            "--exclude",
            ".venv",
            "--exclude",
            "data",
            "--exclude",
            "runs",
            f"{repository}/",
            f"{self.profile.destination}:{remote_dir}/source/",
        ]
        # The bootstrap changes only the remote interpreter's working directory;
        # subprocess still receives an argv list and never invokes a local shell.
        bootstrap = (
            "import os, sys; "
            "root = sys.argv.pop(1); "
            "os.chdir(root); "
            "sys.path.insert(0, os.path.join(root, 'src')); "
            "from thermodense.cli import main; "
            "raise SystemExit(main(sys.argv[1:]))"
        )
        remote = [
            *ssh,
            self.profile.destination,
            "python",
            "-c",
            bootstrap,
            f"{remote_dir}/source",
        ]
        pull = [
            "rsync",
            "--archive",
            f"{self.profile.destination}:{remote_dir}/source/runs/{workflow}/{run_id}/",
            f"{repository}/runs/{workflow}/{run_id}/",
        ]
        return mkdir, push, remote, pull

    def run(
        self, repository: Path, workflow: str, run_id: str, arguments: list[str]
    ) -> None:
        missing = [tool for tool in ("ssh", "rsync") if shutil.which(tool) is None]
        if missing:
            raise SSHProfileError(
                f"SSH backend requires installed tools: {', '.join(missing)}"
            )
        mkdir, push, remote, pull = self.commands(repository, workflow, run_id)
        for command in (mkdir, push, [*remote, *arguments], pull):
            try:
                self.runner(command, check=True)
            except subprocess.CalledProcessError as error:
                raise SSHProfileError(
                    f"SSH backend command failed ({error.returncode}): {command[0]}"
                ) from error
            except OSError as error:
                raise SSHProfileError(
                    f"SSH backend could not execute {command[0]}: {error}"
                ) from error
