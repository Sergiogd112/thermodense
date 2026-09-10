"""Read-only observation of preserved workflow results and optional SGE jobs."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import re
import subprocess
from typing import Any
from xml.etree import ElementTree
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from thermodense.artifacts import ArtifactIntegrityError, recovery_capability, verify_files
from thermodense.checkpoints import fingerprint
from thermodense.workflow_policies import PolicyMismatch, validate_policy

_HOST = re.compile(r"[A-Za-z0-9][A-Za-z0-9.-]*\Z")
_JOB_ID = re.compile(r"[1-9][0-9]*\Z")
_LAYOUTS = {
    "global-polynomial": ("provenance.json", "complete", ("models.npz", "metrics.json")),
    "global-parcorr": (
        "status.json",
        "complete",
        ("pcmci_raw.npz", "eligible_links.csv", "ci_test_trace.jsonl"),
    ),
    "raw-36-parcorr": (
        "provenance.json",
        "completed",
        ("pcmci_results.npz", "eligible_links.csv"),
    ),
    "comparison": ("comparison.json", None, ()),
}
Run = Callable[..., subprocess.CompletedProcess[str]]


def _json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _sha256(path: Path) -> str | None:
    value = fingerprint(path).get("sha256")
    return value if isinstance(value, str) else None


def _without_paths(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _without_paths(item)
            for key, item in value.items()
            if key != "path"
        }
    if isinstance(value, list):
        return [_without_paths(item) for item in value]
    return value


def _embedded_manifest(kind: str, metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
    key = "input_manifest" if kind == "raw-36-parcorr" else "manifest_identity"
    value = metadata.get(key)
    return value if isinstance(value, Mapping) else None


def _verify_local_input(
    kind: str,
    path: Path,
    manifest: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify local input bytes and bind them to completed-result metadata."""
    manifest_digest = _sha256(path)
    if manifest_digest is None:
        return {"integrity": "unverified", "reason": "input_manifest_is_not_a_file"}

    embedded = _embedded_manifest(kind, metadata)
    expected_core: Any = manifest.get("core_sha256")
    if kind == "global-parcorr":
        identity = metadata.get("input_identity")
        if not isinstance(identity, Mapping):
            return {"integrity": "unverified", "reason": "metadata_input_identity_absent"}
        if identity.get("manifest_sha256") != manifest_digest:
            return {"integrity": "unverified", "reason": "input_manifest_digest_mismatch"}
        expected_core = identity.get("core_sha256")
        if manifest.get("core_sha256") != expected_core:
            return {"integrity": "unverified", "reason": "input_core_identity_mismatch"}
    elif kind == "comparison":
        if metadata.get("input_manifest_sha256") != manifest_digest:
            return {"integrity": "unverified", "reason": "input_manifest_digest_mismatch"}
        expected_core = metadata.get("input_core_sha256")
        if manifest.get("core_sha256") != expected_core:
            return {"integrity": "unverified", "reason": "input_core_identity_mismatch"}
    elif embedded is None:
        return {"integrity": "unverified", "reason": "embedded_input_identity_absent"}
    elif kind == "raw-36-parcorr":
        local_identity = _without_paths(manifest)
        embedded_identity = _without_paths(
            {key: embedded.get(key) for key in manifest}
        )
        if local_identity != embedded_identity:
            return {"integrity": "unverified", "reason": "input_manifest_identity_mismatch"}
        expected_core = embedded.get("core_sha256")
    elif _without_paths(manifest) != _without_paths(embedded):
        return {"integrity": "unverified", "reason": "input_manifest_identity_mismatch"}

    core_name = manifest.get("core_npz")
    if not isinstance(core_name, str) or not isinstance(expected_core, str):
        return {"integrity": "unverified", "reason": "input_core_identity_absent"}
    expected_files = {core_name: expected_core}
    if kind == "raw-36-parcorr":
        saber_name = manifest.get("saber_npz")
        expected_saber = manifest.get("saber_sha256")
        if not isinstance(saber_name, str) or not isinstance(expected_saber, str):
            return {"integrity": "unverified", "reason": "input_saber_identity_absent"}
        expected_files[saber_name] = expected_saber
    try:
        verified = verify_files(path.parent, expected_files, required=expected_files)
    except (ArtifactIntegrityError, TypeError) as error:
        return {"integrity": "unverified", "reason": str(error)}
    return {
        "integrity": "verified",
        "manifest_source": "local",
        "manifest_sha256": manifest_digest,
        "files": verified,
    }


def _local(kind: str, run_dir: Path, input_manifest: Path | None) -> dict[str, Any]:
    metadata_name, completed_status, required = _LAYOUTS[kind]
    metadata_path = run_dir if kind == "comparison" and run_dir.suffix == ".json" else run_dir / metadata_name
    metadata = _json(metadata_path)
    state: dict[str, Any] = {"root": "present" if run_dir.is_dir() or metadata_path.is_file() else "missing"}
    if metadata is None:
        return {
            "execution": state,
            "completion": {"declared": False, "reason": "metadata_missing_or_invalid"},
            "artifacts": {"integrity": "unverified", "reason": "metadata_missing_or_invalid"},
            "input": {"integrity": "unverified", "reason": "metadata_missing_or_invalid"},
            "policy": {"matches_contract": False, "status": "policy_unverified", "reason": "metadata_missing_or_invalid"},
        }
    declared = completed_status is None or metadata.get("status") == completed_status
    completion = {"declared": declared, "status": metadata.get("status")}
    hashes = metadata.get("result_hashes", metadata.get("result_files"))
    expected: dict[str, str] = {}
    hashes_valid = isinstance(hashes, Mapping) and bool(hashes)
    if isinstance(hashes, Mapping):
        for name, digest in hashes.items():
            if not isinstance(name, str) or not isinstance(digest, str):
                hashes_valid = False
                break
            expected[name] = digest
    if kind == "comparison":
        artifacts = {
            "integrity": "unverified",
            "reason": "derived_comparison_requires_a_trusted_output_digest_and_explicit_upstream_result_directory",
        }
    elif not hashes_valid:
        artifacts = {"integrity": "unverified", "reason": "result_hashes_or_result_files_absent"}
    else:
        try:
            verified = verify_files(run_dir, expected, required=required)
            artifacts = {"integrity": "verified", "files": verified}
        except (ArtifactIntegrityError, TypeError) as error:
            artifacts = {"integrity": "unverified", "reason": str(error)}
    local_manifest = _json(input_manifest) if input_manifest is not None else None
    embedded = _embedded_manifest(kind, metadata)
    if input_manifest is not None and local_manifest is None:
        input_report = {"integrity": "unverified", "manifest_source": "local_invalid", "reason": "input_manifest_missing_or_invalid"}
        manifest: Mapping[str, Any] | None = None
    elif input_manifest is not None and local_manifest is not None:
        input_report = _verify_local_input(kind, input_manifest, local_manifest, metadata)
        manifest = local_manifest
    elif embedded is not None:
        input_report = {
            "integrity": "declared_only",
            "manifest_source": "embedded",
            "reason": "embedded_identity_has_no_locally_verified_input_bytes",
        }
        manifest = embedded
    else:
        input_report = {"integrity": "unverified", "manifest_source": "absent", "reason": "input_manifest_absent"}
        manifest = None
    try:
        if manifest is None:
            raise PolicyMismatch("input manifest is absent")
        validate_policy(kind, metadata, manifest)
        policy = {"matches_contract": True, "status": "contract_matches"}
    except (PolicyMismatch, ValueError, TypeError) as error:
        policy = {"matches_contract": False, "status": "policy_unverified", "reason": str(error)}
    return {
        "execution": state,
        "completion": completion,
        "artifacts": artifacts,
        "input": input_report,
        "policy": policy,
    }


def _host(value: str) -> str:
    if not _HOST.fullmatch(value) or value.startswith("-") or any(char in value for char in "\r\n\x00"):
        raise ValueError("ssh host must be a plain host alias without control characters")
    return value


def _job(value: int | str) -> int:
    text = str(value)
    if not _JOB_ID.fullmatch(text):
        raise ValueError("job id must be a validated positive integer")
    return int(text)


def _invoke(run: Run, host: str, command: str) -> subprocess.CompletedProcess[str]:
    return run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "--", host, command],
        shell=False,
        text=True,
        capture_output=True,
        timeout=15,
    )


def _time(value: str, timezone: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.strip())
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=ZoneInfo(timezone))
        return parsed.astimezone(UTC)
    except ValueError:
        return None


def _runtime_limit(text: str) -> timedelta | None:
    match = re.search(r"\bh_rt=([0-9]+)(?::([0-9]{2}):([0-9]{2}))?", text)
    if not match:
        return None
    amount, minutes, seconds = match.groups()
    if minutes is None:
        return timedelta(seconds=int(amount))
    return timedelta(hours=int(amount), minutes=int(minutes), seconds=int(seconds))


def _quota(host: str, run: Run) -> dict[str, Any]:
    try:
        result = _invoke(run, host, 'qquota -u "$USER"')
    except (subprocess.TimeoutExpired, OSError) as error:
        return {"observation": type(error).__name__}
    text = (result.stdout or result.stderr).strip()
    match = re.search(r"\bavailable(?:_slots| slots)?\s*[:=]\s*(\d+)\b", text, re.IGNORECASE)
    if not result.returncode and match:
        return {"actual_available_slots": int(match.group(1))}
    return {"observation": text or "qquota_unavailable"}


def _accounting(text: str, job_id: int) -> dict[str, Any]:
    blocks = [block for block in re.split(r"^=+\s*$", text, flags=re.MULTILINE) if block.strip()]
    records: list[dict[str, int]] = []
    for block in blocks:
        job = re.search(r"^jobnumber\s+(\d+)\s*$", block, re.MULTILINE)
        failed = re.search(r"^failed\s+(\d+)\s*$", block, re.MULTILINE)
        exit_status = re.search(r"^exit_status\s+(\d+)\s*$", block, re.MULTILINE)
        if job and int(job.group(1)) == job_id and failed and exit_status:
            records.append({"failed": int(failed.group(1)), "exit_status": int(exit_status.group(1))})
    if len(records) != 1:
        reason = "accounting_record_ambiguous" if len(records) > 1 else "accounting_record_incomplete_or_mismatched"
        return {"state": "unknown", "reason": reason}
    record = records[0]
    return {"state": "failed" if record["failed"] or record["exit_status"] else "completed", **record}


def _sge(host: str, job_id: int, timezone: str, run: Run) -> dict[str, Any]:
    try:
        qstat = _invoke(run, host, 'qstat -xml -u "$USER"')
    except (subprocess.TimeoutExpired, OSError) as error:
        return {"state": "unreachable", "reason": type(error).__name__}
    if qstat.returncode == 255:
        return {"state": "unreachable", "reason": "ssh_exit_255"}
    if qstat.returncode:
        return {"state": "unknown", "reason": qstat.stderr.strip() or "qstat_failed"}
    try:
        root = ElementTree.fromstring(qstat.stdout)
    except ElementTree.ParseError:
        return {"state": "unknown", "reason": "invalid_qstat_xml"}
    job = next((item for item in root.findall(".//job_list") if item.findtext("JB_job_number") == str(job_id)), None)
    if job is not None:
        category = (job.get("state") or "").strip()
        code = (job.findtext("state") or "").strip()
        reason = code or job.findtext("queue_name") or category
        if "E" in code:
            return {"state": "error", "reason": reason, "deadline": "unknown", "quota": _quota(host, run)}
        if category == "running" or code in {"r", "t", "Rr", "Rt"}:
            try:
                details = _invoke(run, host, f"qstat -j {job_id}")
            except (subprocess.TimeoutExpired, OSError):
                details = None
            start = _time(job.findtext("JAT_start_time") or "", timezone)
            limit = _runtime_limit(details.stdout) if details is not None and not details.returncode else None
            deadline = (start + limit).isoformat() if start is not None and limit is not None else None
            return {"state": "running", "reason": reason, "actual_start_at": start.isoformat() if start else None, "deadline_at": deadline, "deadline": "known" if deadline else "unknown", "quota": _quota(host, run)}
        return {"state": "queued", "reason": reason or "scheduler_pending", "deadline": "unknown", "quota": _quota(host, run)}
    try:
        accounting = _invoke(run, host, f"qacct -j {job_id}")
    except (subprocess.TimeoutExpired, OSError) as error:
        return {"state": "unreachable", "reason": type(error).__name__}
    if accounting.returncode == 255:
        return {"state": "unreachable", "reason": "ssh_exit_255"}
    if accounting.returncode or not accounting.stdout.strip():
        return {"state": "unknown", "reason": "accounting_lag_or_no_job"}
    return _accounting(accounting.stdout, job_id)


def monitor(kind: str, run_dir: Path, input_manifest: Path | None = None, *, ssh_host: str | None = None, job_id: int | str | None = None, scheduler_timezone: str = "UTC", run: Run = subprocess.run) -> dict[str, Any]:
    """Return one JSON-safe observation; this function never submits or alters work."""
    if kind not in _LAYOUTS:
        raise ValueError(f"unknown workflow kind {kind!r}")
    report = _local(kind, run_dir, input_manifest)
    report["kind"] = kind
    report["checked_at"] = datetime.now(UTC).isoformat()
    report["recovery"] = {**recovery_capability(kind).__dict__, "resume_readiness": "not_assessed; capability is not checkpoint approval"}
    if (ssh_host is None) != (job_id is None):
        raise ValueError("ssh host and job id must be supplied together")
    if ssh_host is not None and job_id is not None:
        try:
            ZoneInfo(scheduler_timezone)
        except ZoneInfoNotFoundError as error:
            raise ValueError(f"unknown scheduler timezone {scheduler_timezone!r}") from error
        report["scheduler"] = _sge(_host(ssh_host), _job(job_id), scheduler_timezone, run)
    scheduler_state = report.get("scheduler", {}).get("state")
    report["status"] = "verified_complete" if (
        report["completion"]["declared"]
        and report["artifacts"]["integrity"] == "verified"
        and report["input"]["integrity"] == "verified"
        and report["policy"]["matches_contract"]
        and scheduler_state in {None, "completed"}
    ) else "unverified"
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=sorted(_LAYOUTS), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--ssh-host")
    parser.add_argument("--job-id")
    parser.add_argument("--scheduler-timezone", default="UTC")
    args = parser.parse_args(argv)
    print(json.dumps(monitor(args.kind, args.run_dir, args.input_manifest, ssh_host=args.ssh_host, job_id=args.job_id, scheduler_timezone=args.scheduler_timezone), sort_keys=True))


if __name__ == "__main__":
    main()
