"""Durable gates for the bounded real-data GPDCtorch matrix.

The state file is deliberately an input to the private gated runner, not an
unlock flag for the ordinary PCMCI command.  A successful predecessor is
revalidated from its recorded identity before a later stage is considered.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import secrets
import subprocess
import hashlib
import socket
import tempfile
from typing import Any, Callable

from thermodense.benchmarks import runtime

CAPABILITY_TIMEOUT_SECONDS = 5 * 60 * 60
PRIMARY_TIMEOUT_SECONDS = 24 * 60 * 60
STATE_SCHEMA_VERSION = "1"
STAGES = ("capability", "primary", "raw_seasonal", "centered_detrended", "interaction")
_LEGACY_TIGRAMITE_PIN_ERRORS = (OSError, subprocess.SubprocessError)


class GateError(ValueError):
    """A gate cannot safely advance."""


@dataclass(frozen=True)
class Stage:
    name: str
    tau_max: int
    timing_variant: str
    preprocessing_profile: str
    role: str
    timeout_seconds: int


def stages() -> tuple[Stage, ...]:
    return (
        Stage(
            "capability",
            1,
            "raw_observed_daily",
            "detrended_anomaly",
            "primary",
            CAPABILITY_TIMEOUT_SECONDS,
        ),
        Stage(
            "primary",
            10,
            "raw_observed_daily",
            "detrended_anomaly",
            "primary",
            PRIMARY_TIMEOUT_SECONDS,
        ),
        Stage(
            "raw_seasonal",
            10,
            "raw_observed_daily",
            "seasonal_anomaly",
            "robustness",
            PRIMARY_TIMEOUT_SECONDS,
        ),
        Stage(
            "centered_detrended",
            10,
            "centered_81_day",
            "detrended_anomaly",
            "robustness",
            PRIMARY_TIMEOUT_SECONDS,
        ),
        Stage(
            "interaction",
            10,
            "centered_81_day",
            "seasonal_anomaly",
            "interaction_diagnostic",
            PRIMARY_TIMEOUT_SECONDS,
        ),
    )


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    """Replace a state document only after its complete bytes reach disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def gpu_hardware() -> dict[str, Any]:
    """Return enough CUDA identity to prevent cross-machine evidence reuse."""
    try:
        import torch

        available = bool(torch.cuda.is_available())
        if not available:
            return {
                "eligible": False,
                "reason": "torch.cuda.is_available() is false",
                "torch": torch.__version__,
            }
        properties = torch.cuda.get_device_properties(0)
        hardware = {
            "eligible": True,
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "device_count": torch.cuda.device_count(),
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
            "driver": None,
            "gpu_uuid": None,
            "pci_bus_id": None,
            "driver_probe_failure": None,
        }
        try:
            probe = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version,uuid,pci.bus_id",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            values = probe.stdout.strip().splitlines()
            fields = values[0].split(", ") if values else []
            hardware.update(
                driver=fields[0] if len(fields) == 3 else None,
                gpu_uuid=fields[1] if len(fields) == 3 else None,
                pci_bus_id=fields[2] if len(fields) == 3 else None,
            )
            hardware["driver_probe_failure"] = (
                None
                if probe.returncode == 0 and len(fields) == 3
                else probe.stderr.strip() or "unexpected nvidia-smi output"
            )
        except (OSError, subprocess.SubprocessError) as error:
            hardware["driver_probe_failure"] = f"{type(error).__name__}: {error}"
        return hardware
    except Exception as error:
        return {
            "eligible": False,
            "reason": f"CUDA inspection failed: {type(error).__name__}: {error}",
        }


def environment_identity() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "git_commit": runtime.git_commit(),
        "package_versions": runtime.package_versions(),
    }


def new_state(identity: dict[str, Any], hardware: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now(UTC).isoformat()
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "state": "pending",
        "identity": identity,
        "hardware": hardware,
        "environment": environment_identity(),
        "created_at": now,
        "updated_at": now,
        "transitions": [{"at": now, "from": None, "to": "pending"}],
        "stages": {
            stage.name: {
                "name": stage.name,
                "status": "pending",
                "attempts": [],
                "tau_max": stage.tau_max,
                "timing_variant": stage.timing_variant,
                "preprocessing_profile": stage.preprocessing_profile,
                "role": stage.role,
                "timeout_seconds": stage.timeout_seconds,
            }
            for stage in stages()
        },
        "derived": {
            "status": "pending",
            "attempts": [],
            "source_stage_identities": {},
            "created_at": now,
            "updated_at": now,
        },
    }


def _transition(state: dict[str, Any], value: str) -> None:
    now = datetime.now(UTC).isoformat()
    if state.get("state") != value:
        state.setdefault("transitions", []).append(
            {"at": now, "from": state.get("state"), "to": value}
        )
    state["state"] = value
    state["updated_at"] = now


def _valid_success(
    record: dict[str, Any],
    identity: dict[str, Any],
    validate_success: Callable[[dict[str, Any]], bool] | None = None,
) -> bool:
    return (
        record.get("status") == "succeeded"
        and record.get("identity") == identity
        and isinstance(record.get("result"), dict)
        and (validate_success is None or validate_success(record))
    )


def load_or_create(
    path: Path,
    identity: dict[str, Any],
    hardware: dict[str, Any],
    *,
    retry_failed: bool,
) -> dict[str, Any]:
    if not path.exists():
        state = new_state(identity, hardware)
        atomic_write(path, state)
        return state
    try:
        state = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise GateError(f"malformed gate state: {error}") from error
    if (
        state.get("schema_version") != STATE_SCHEMA_VERSION
        or state.get("identity") != identity
    ):
        raise GateError(
            "gate state identity or schema does not match current input/settings/environment"
        )
    if (
        state.get("environment") != environment_identity()
        or state.get("hardware") != hardware
    ):
        raise GateError("gate state environment or hardware identity is stale")
    for record in state.get("stages", {}).values():
        if record.get("status") == "running":
            record["status"] = "interrupted"
            record["failure_reason"] = (
                "parent restarted while isolated child was running"
            )
            attempts = record.get("attempts")
            if (
                isinstance(attempts, list)
                and attempts
                and isinstance(attempts[-1], dict)
            ):
                attempts[-1].update(
                    finished_at=datetime.now(UTC).isoformat(),
                    status="interrupted",
                    failure_reason=record["failure_reason"],
                )
    if retry_failed:
        for stage in state.get("stages", {}).values():
            if stage.get("status") in {
                "failed",
                "timeout",
                "killed",
                "interrupted",
                "stale",
            }:
                stage["status"] = "pending"
        _transition(state, "pending")
    elif any(
        record.get("status") == "interrupted"
        for record in state.get("stages", {}).values()
    ):
        _transition(state, "blocked")
    atomic_write(path, state)
    return state


def import_capability(
    path: Path,
    evidence_path: Path,
    identity: dict[str, Any],
    hardware: dict[str, Any],
    *,
    validate_success: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    """Import only a complete, current standard-runner tau-one result."""
    try:
        rows = [
            json.loads(line) for line in evidence_path.read_text().splitlines() if line
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise GateError(f"malformed capability evidence: {error}") from error
    if len(rows) != 1 or not isinstance(rows[0], dict):
        raise GateError("capability evidence must contain exactly one result row")
    row = rows[0]
    legacy = "hardware" not in row
    capability_evidence = {
        "mode": "hardware_equality",
        "current_hardware": hardware,
        "current_environment": environment_identity(),
    }
    if legacy:
        labels = ("host_label", "environment_label", "environment_fingerprint")
        current_settings = identity.get("settings")
        legacy_settings = row.get("settings")
        expected_legacy_settings = (
            {key: value for key, value in current_settings.items() if key != "threads"}
            if isinstance(current_settings, dict)
            else None
        )
        if (
            not hardware.get("eligible")
            or "gpdctorch_lifecycle" in row
            or any(
                not isinstance(row.get(label), str)
                or row[label] == "unspecified"
                or not row[label].strip()
                for label in labels
            )
            or not _legacy_tigramite_pin(
                row.get("git_commit"), identity["tigramite_pin"]
            )
        ):
            raise GateError("legacy capability evidence cannot be environment-attested")
        if (
            not isinstance(current_settings, dict)
            or current_settings.get("threads") != 1
            or legacy_settings != expected_legacy_settings
        ):
            raise GateError(
                "legacy capability settings require missing threads constrained to 1"
            )
        capability_evidence = {
            "mode": "legacy_environment_attestation",
            "original_labels": {key: row.get(key) for key in labels},
            "original_git_commit": row.get("git_commit"),
            "tigramite_pin": identity["tigramite_pin"],
            "pin_proven": True,
            "legacy_settings": legacy_settings,
            "current_requested_settings": current_settings,
            "threads_provenance": "legacy_absent_constrained_to_default_1",
            "current_hardware": hardware,
            "current_environment": environment_identity(),
        }
    candidate_record = {
        "name": "capability",
        "status": "succeeded",
        "identity": identity,
        "result": row,
        "capability_evidence": capability_evidence,
    }
    if not validate_success(candidate_record):
        raise GateError("capability evidence is stale, mismatched, or malformed")
    state = load_or_create(path, identity, hardware, retry_failed=False)
    record = state["stages"]["capability"]
    if not _valid_success(record, identity, validate_success):
        now = datetime.now(UTC).isoformat()
        attempt = {
            "started_at": now,
            "finished_at": now,
            "status": "succeeded",
            "identity": identity,
            "imported_from": str(evidence_path),
        }
        record.update(status="running", identity=identity)
        record.setdefault("attempts", []).append(attempt)
        _transition(state, "running")
        record.update(
            status="succeeded",
            identity=identity,
            result=row,
            imported_from=str(evidence_path),
            original_git_commit=row.get("git_commit"),
            capability_evidence=capability_evidence,
            failure_reason=None,
        )
        _transition(state, "pending")
        atomic_write(path, state)
    return state


def _legacy_tigramite_pin(git_commit: Any, pin: str) -> bool:
    """Attest a pre-wrapper result against its committed dependency declaration."""
    if not isinstance(git_commit, str) or not git_commit:
        return False
    try:
        result = subprocess.run(
            ["git", "show", f"{git_commit}:pyproject.toml"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except _LEGACY_TIGRAMITE_PIN_ERRORS:
        return False
    return result.returncode == 0 and f'rev = "{pin}"' in result.stdout


def run(
    path: Path,
    identity: dict[str, Any],
    hardware: dict[str, Any],
    execute: Callable[[Stage, str], dict[str, Any]],
    *,
    retry_failed: bool = False,
    validate_success: Callable[[dict[str, Any]], bool] | None = None,
) -> dict[str, Any]:
    """Advance exactly one ordered matrix, persisting before and after every fit."""
    if not hardware.get("eligible"):
        raise GateError(
            f"eligible CUDA GPU required: {hardware.get('reason', 'not eligible')}"
        )
    state = load_or_create(path, identity, hardware, retry_failed=retry_failed)
    for index, stage in enumerate(stages()):
        record = state["stages"].get(stage.name)
        if not isinstance(record, dict):
            raise GateError("gate state stage schema is malformed")
        if _valid_success(record, identity, validate_success):
            continue
        if record.get("status") == "succeeded":
            record.update(
                status="stale", failure_reason="successful evidence no longer validates"
            )
            if retry_failed:
                record["status"] = "pending"
        if record.get("status") in {
            "failed",
            "timeout",
            "killed",
            "interrupted",
            "stale",
        }:
            _transition(state, "blocked")
            atomic_write(path, state)
            return state
        if index and not _valid_success(
            state["stages"][stages()[index - 1].name], identity, validate_success
        ):
            raise GateError("gate predecessor is not a validated success")
        authorization = secrets.token_urlsafe(32)
        attempt = {
            "started_at": datetime.now(UTC).isoformat(),
            "identity": identity,
            "status": "running",
            "authorization_sha256": hashlib.sha256(authorization.encode()).hexdigest(),
        }
        record.update(status="running", identity=identity)
        record.setdefault("attempts", []).append(attempt)
        _transition(state, "running")
        atomic_write(path, state)
        try:
            result = execute(stage, authorization)
        except Exception as error:
            result = {
                "status": "failed",
                "failure_reason": f"executor exception: {type(error).__name__}: {error}",
            }
        status = result.get("status", "failed")
        if status not in {"succeeded", "failed", "timeout", "killed"}:
            status = "failed"
            result = result | {"failure_reason": "child returned an invalid status"}
        attempt.update(
            {
                "finished_at": datetime.now(UTC).isoformat(),
                "status": status,
                "result": result,
            }
        )
        record.update(
            status=status,
            identity=identity,
            result=result,
            failure_reason=result.get("failure_reason"),
        )
        if status == "succeeded" and not _valid_success(
            record, identity, validate_success
        ):
            record.update(
                status="stale",
                failure_reason="successful evidence no longer validates",
            )
            _transition(state, "blocked")
            atomic_write(path, state)
            return state
        _transition(
            state,
            "complete"
            if status == "succeeded" and stage.name == STAGES[-1]
            else ("blocked" if status != "succeeded" else "pending"),
        )
        atomic_write(path, state)
        if status != "succeeded":
            return state
    _transition(state, "complete")
    atomic_write(path, state)
    return state
