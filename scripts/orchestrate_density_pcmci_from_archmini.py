#!/usr/bin/env python3
"""Recoverable Archmini orchestration for direct-density PCMCI cases.

Run this from the Archmini run directory containing runner.py,
analysis_bundle.npz, and run_case_sge.sh.  Recovery is case-level: an outage
inside PCMCI restarts that case from its immutable bundle; completed atomic
artifacts are reused.  Tigramite itself has no in-process checkpoint here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time
from typing import Any

RUN_ID = "density-pcmci-v2-saber"
REMOTE_BASE = f".local/share/thermodense/runs/{RUN_ID}"
OUTPUT_RELATIVE = "outputs/prototypes/density_pcmci_3hour_and_daily"
PLAN = {
    "global_mean-ap": {"host": "spacehopper", "kind": "direct"},
    "global_mean-kp": {"host": "spacehopper", "kind": "direct"},
    "hasdm_all-ap": {"host": "phoenix", "kind": "sge"},
    "hasdm_all-kp": {"host": "phoenix", "kind": "sge"},
    "hasdm_selected-ap": {"host": "kaggle", "kind": "kaggle"},
    "hasdm_selected-kp": {"host": "kaggle", "kind": "kaggle"},
}


def now() -> float:
    return time.time()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def remove_temporary(path: Path) -> None:
    """Remove a temporary retrieval path without following symlinks."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def command(arguments: list[str], *, capture: bool = False) -> str:
    result = subprocess.run(
        arguments, check=True, text=True, stdout=subprocess.PIPE if capture else None
    )
    return result.stdout.strip() if capture else ""


def remote(host: str, shell_command: str, *, capture: bool = False) -> str:
    return command(["ssh", "-o", "BatchMode=yes", host, shell_command], capture=capture)


def paths(run_dir: Path) -> tuple[Path, Path, Path, Path]:
    return (
        run_dir / "runner.py",
        run_dir / "analysis_bundle.npz",
        run_dir / "run_case_sge.sh",
        run_dir / "orchestration_manifest.json",
    )


def expected_tests(case_id: str) -> int:
    if case_id.startswith("global_mean-"):
        return 2 * 10 * 184
    return 17 * (27 if case_id.startswith("hasdm_all-") else 3) * 489


def validate_case(
    directory: Path, case_id: str, bundle_hash: str
) -> dict[str, str] | None:
    """Return file hashes only for a final, production artifact."""
    provenance = directory / "provenance.json"
    tests, retained = (
        directory / "driver_target_tests.csv",
        directory / "retained_links.csv",
    )
    if not all(path.is_file() for path in (provenance, tests, retained)):
        return None
    try:
        record = json.loads(provenance.read_text())
        result_files = record["result_files"]
        count = sum(1 for _ in tests.open()) - 1
    except OSError, ValueError, KeyError:
        return None
    hashes = {path.name: sha256(path) for path in (provenance, tests, retained)}
    if (
        not record.get("production")
        or record.get("case", {}).get("id") != case_id
        or record.get("bundle_sha256") != bundle_hash
        or not record.get("fingerprint")
        or record.get("fdr_family", {}).get("test_count") != expected_tests(case_id)
        or count != expected_tests(case_id)
        or result_files.get(tests.name) != hashes[tests.name]
        or result_files.get(retained.name) != hashes[retained.name]
    ):
        return None
    return hashes


def initial_manifest(run_dir: Path) -> dict[str, Any]:
    runner, bundle, sge, _ = paths(run_dir)
    if not all(path.is_file() for path in (runner, bundle, sge)):
        missing = next(path for path in (runner, bundle, sge) if not path.is_file())
        raise FileNotFoundError(missing)
    hashes = {
        "runner.py": sha256(runner),
        "analysis_bundle.npz": sha256(bundle),
        "run_case_sge.sh": sha256(sge),
    }
    cases = {
        case_id: {
            **assignment,
            "remote_run_dir": f"~/{REMOTE_BASE}",
            "status": "pending",
            "attempts": [],
        }
        for case_id, assignment in PLAN.items()
    }
    return {
        "schema": 1,
        "run_id": RUN_ID,
        "orchestrator": "archmini",
        "created_epoch": now(),
        "plan": PLAN,
        "inputs": hashes,
        "cases": cases,
        "hosts": {},
        "retrieval": {},
        "recovery": "Case-level only: interruption inside PCMCI restarts from the immutable bundle; final atomic provenance artifacts are reused.",
    }


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def save_manifest(manifest: dict[str, Any], path: Path) -> None:
    manifest["updated_epoch"] = now()
    atomic_json(manifest, path)


def mark_host(
    manifest: dict[str, Any], host: str, status: str, detail: str = ""
) -> None:
    manifest.setdefault("hosts", {})[host] = {
        "status": status,
        "detail": detail,
        "epoch": now(),
    }


def stage(
    host: str, run_dir: Path, include_sge: bool, hashes: dict[str, str]
) -> dict[str, str]:
    runner, bundle, sge, _ = paths(run_dir)
    remote(
        host,
        f"mkdir -p ~/{REMOTE_BASE}/{OUTPUT_RELATIVE} ~/{REMOTE_BASE}/{{logs,pids,locks}}",
    )
    files = {
        "runner.py": (runner, "runner.py"),
        "analysis_bundle.npz": (bundle, f"{OUTPUT_RELATIVE}/analysis_bundle.npz"),
        "run_case_sge.sh": (sge, "run_case_sge.sh"),
    }
    selected = ["runner.py", "analysis_bundle.npz"] + (
        ["run_case_sge.sh"] if include_sge else []
    )
    names = " ".join(files[name][1] for name in selected)
    actual = {
        line.split()[1].split("/")[-1]: line.split()[0]
        for line in remote(
            host,
            f"cd ~/{REMOTE_BASE} && for file in {names}; do "
            'test -f "$file" && sha256sum "$file" || printf "MISSING  %s\\n" "$file"; done',
            capture=True,
        ).splitlines()
    }
    wanted = {
        key: value
        for key, value in hashes.items()
        if include_sge or key != "run_case_sge.sh"
    }
    for name in selected:
        if actual.get(name) != wanted[name]:
            path, destination = files[name]
            command(["scp", str(path), f"{host}:{REMOTE_BASE}/{destination}"])
    actual = {
        line.split()[1].split("/")[-1]: line.split()[0]
        for line in remote(
            host, f"cd ~/{REMOTE_BASE} && sha256sum {names}", capture=True
        ).splitlines()
    }
    if actual != wanted:
        raise RuntimeError(f"staged hash mismatch on {host}: {actual} != {wanted}")
    return actual


REMOTE_STATUS_PROGRAM = r"""
import hashlib,json,os,subprocess,sys
base,cases,bundle=sys.argv[1],json.loads(sys.argv[2]),sys.argv[3]
def digest(p):
  with open(p,'rb') as f: return hashlib.sha256(f.read()).hexdigest()
def complete(c):
  d=os.path.join(base,"outputs/prototypes/density_pcmci_3hour_and_daily/cases",c["id"]); p=os.path.join(d,"provenance.json"); t=os.path.join(d,"driver_target_tests.csv"); r=os.path.join(d,"retained_links.csv")
  try:
    x=json.load(open(p)); n=sum(1 for _ in open(t))-1; expected=2*10*184 if c["id"].startswith("global_mean-") else 17*(27 if c["id"].startswith("hasdm_all-") else 3)*489
    return x.get("production") and x.get("case",{}).get("id")==c["id"] and x.get("bundle_sha256")==bundle and bool(x.get("fingerprint")) and x.get("fdr_family",{}).get("test_count")==expected and n==expected and x["result_files"].get("driver_target_tests.csv")==digest(t) and x["result_files"].get("retained_links.csv")==digest(r)
  except (OSError,ValueError,KeyError): return False
out={}
def re_failed(text):
  return any(line.split()[-1] not in ("0", "0.00000") for line in text.splitlines() if line.startswith(("failed", "exit_status")))
for c in cases:
  if complete(c): out[c["id"]]={"status":"complete"}; continue
  if c["kind"]=="direct":
    pid=c.get("pid"); cmd=b""
    try: cmd=open("/proc/%s/cmdline"%pid,"rb").read()
    except (OSError,TypeError): pass
    valid=(b"runner.py\x00run\x00"+c["id"].encode()+b"\x00") in cmd
    out[c["id"]]={"status":"active" if valid else "failed", "detail":"expected runner command" if valid else "pid absent or command mismatch"}
  else:
    job=str(c.get("job_id", "")); q=subprocess.run(["qstat","-j",job],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode if job else 1
    if q==0: out[c["id"]]={"status":"active"}; continue
    a=subprocess.run(["qacct","-j",job],capture_output=True,text=True).stdout if job else ""
    if not a: out[c["id"]]={"status":"missing"}
    elif re_failed(a): out[c["id"]]={"status":"failed", "detail":a}
    else: out[c["id"]]={"status":"exited", "detail":a}
print(json.dumps(out))
"""


def remote_status(host: str, manifest: dict[str, Any]) -> dict[str, dict[str, str]]:
    cases = [
        dict(id=case_id, **data)
        for case_id, data in manifest["cases"].items()
        if data["host"] == host
    ]
    args = " ".join(
        (
            shlex.quote(REMOTE_STATUS_PROGRAM),
            f'"$HOME/{REMOTE_BASE}"',
            shlex.quote(json.dumps(cases)),
            shlex.quote(manifest["inputs"]["analysis_bundle.npz"]),
        )
    )
    return json.loads(remote(host, f"python3 -c {args}", capture=True))


def launch(case_id: str, data: dict[str, Any]) -> dict[str, Any]:
    host, kind = data["host"], data["kind"]
    if kind == "direct":
        shell = (
            f"cd ~/{REMOTE_BASE} && : > logs/{case_id}.log && "
            f"nohup flock -n -E 75 locks/{case_id}.lock env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "
            f"$HOME/.local/share/thermodense/envs/cpu-parcorr-conda/bin/python runner.py run {case_id} --host-label spacehopper "
            f"> logs/{case_id}.log 2>&1 < /dev/null & echo $!"
        )
        return {"pid": int(remote(host, shell, capture=True))}
    name = "dens_" + case_id.replace("-", "_")
    output = remote(
        host,
        f"cd ~/{REMOTE_BASE} && qsub -N {name} -o logs/{case_id}.out -e logs/{case_id}.err run_case_sge.sh {case_id} phoenix-auriga",
        capture=True,
    )
    match = re.search(r"job (\d+)", output)
    if match is None:
        raise RuntimeError(f"could not parse qsub response: {output}")
    return {"job_id": int(match.group(1))}


def selected_hosts(manifest: dict[str, Any], hosts: set[str] | None) -> set[str]:
    # Kaggle cases share the canonical plan but are owned by the Kaggle adapter.
    # Never turn an unfiltered SSH invocation into a Kaggle submission.
    manifest_hosts = {
        data["host"] for data in manifest["cases"].values() if data["kind"] != "kaggle"
    }
    if hosts is None:
        return manifest_hosts
    unknown_hosts = hosts - manifest_hosts
    if unknown_hosts:
        raise ValueError(
            f"requested host(s) not in manifest: {', '.join(sorted(unknown_hosts))}; "
            f"available hosts: {', '.join(sorted(manifest_hosts))}"
        )
    return manifest_hosts & hosts


def resume(
    run_dir: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
    hosts: set[str] | None = None,
) -> bool:
    """Stage/check each host independently and relaunch only non-active cases."""
    healthy = True
    for host in sorted(selected_hosts(manifest, hosts)):
        try:
            staged = stage(host, run_dir, host == "phoenix", manifest["inputs"])
            mark_host(manifest, host, "staged", json.dumps(staged, sort_keys=True))
            save_manifest(manifest, manifest_path)
            observed = remote_status(host, manifest)
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            healthy = False
            mark_host(manifest, host, "unreachable", str(error))
            save_manifest(manifest, manifest_path)
            continue
        for case_id, observed_case in observed.items():
            data, state = manifest["cases"][case_id], observed_case["status"]
            if state == "complete":
                data["status"] = "completed"
                save_manifest(manifest, manifest_path)
                continue
            if state == "active":
                data["status"] = "active"
                save_manifest(manifest, manifest_path)
                continue
            attempt = {"epoch": now(), "reason": state, "status": "launching"}
            data["attempts"].append(attempt)
            data["status"] = "launching"
            save_manifest(manifest, manifest_path)
            try:
                attempt.update(launch(case_id, data))
                attempt["status"] = "submitted"
                data.update(
                    {
                        key: value
                        for key, value in attempt.items()
                        if key in {"pid", "job_id"}
                    }
                )
                data["status"] = "submitted"
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                healthy = False
                attempt.update(status="launch_failed", detail=str(error))
                data["status"] = "failed"
            save_manifest(manifest, manifest_path)
    return healthy


def status(
    manifest: dict[str, Any], hosts: set[str] | None = None
) -> tuple[dict[str, Any], bool]:
    """Query only: do not write the manifest or change remote state."""
    result, healthy = {}, True
    for host in sorted(selected_hosts(manifest, hosts)):
        try:
            result.update(remote_status(host, manifest))
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            healthy = False
            for case_id, data in manifest["cases"].items():
                if data["host"] == host:
                    result[case_id] = {"status": "unreachable", "detail": str(error)}
    return result, healthy


def retrieve(
    run_dir: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
    hosts: set[str] | None = None,
) -> bool:
    selected = selected_hosts(manifest, hosts)
    observed, healthy = status(manifest) if hosts is None else status(manifest, hosts)
    runner, bundle, _sge, _ = paths(run_dir)
    del runner
    destination_base = run_dir / "retrieved"
    destination_base.mkdir(parents=True, exist_ok=True)
    for case_id, observed_case in observed.items():
        if manifest["cases"][case_id]["host"] not in selected:
            continue
        if observed_case["status"] != "complete":
            manifest["retrieval"][case_id] = {"status": "not_complete", "epoch": now()}
            save_manifest(manifest, manifest_path)
            healthy = False
            continue
        data, destination = manifest["cases"][case_id], destination_base / case_id
        temporary = destination_base / f".{case_id}.{os.getpid()}.tmp"
        try:
            remove_temporary(temporary)
            command(
                [
                    "scp",
                    "-r",
                    f"{data['host']}:{REMOTE_BASE}/{OUTPUT_RELATIVE}/cases/{case_id}",
                    str(temporary),
                ]
            )
            hashes = validate_case(temporary, case_id, sha256(bundle))
            if hashes is None:
                raise RuntimeError("refusing partial or altered artifact")
            if destination.exists():
                existing = validate_case(destination, case_id, sha256(bundle))
                if existing != hashes:
                    raise RuntimeError(
                        "refusing to replace retrieved artifact with different bytes"
                )
                # The valid existing tree is authoritative; remove only the temporary copy.
                remove_temporary(temporary)
            else:
                temporary.replace(destination)
            manifest["retrieval"][case_id] = {
                "status": "retrieved",
                "epoch": now(),
                "hashes": hashes,
            }
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            try:
                remove_temporary(temporary)
            except OSError:
                pass
            healthy = False
            manifest["retrieval"][case_id] = {
                "status": "refused",
                "epoch": now(),
                "detail": str(error),
            }
        save_manifest(manifest, manifest_path)
    return healthy


def parser() -> argparse.ArgumentParser:
    command_parser = argparse.ArgumentParser(description=__doc__)
    command_parser.add_argument(
        "command",
        nargs="?",
        choices=("submit", "status", "resume", "retrieve"),
        default="submit",
    )
    command_parser.add_argument("--run-dir", type=Path, default=Path.cwd())
    command_parser.add_argument("--host", action="append")
    return command_parser


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    run_dir = args.run_dir.resolve()
    *_inputs, manifest_path = paths(run_dir)
    if args.command == "submit":
        if not manifest_path.exists():
            manifest = initial_manifest(run_dir)
            save_manifest(manifest, manifest_path)  # durable before any SSH/SCP
        else:
            manifest = load_manifest(manifest_path)
        return (
            0
            if resume(
                run_dir,
                manifest,
                manifest_path,
                set(args.host) if args.host else None,
            )
            else 1
        )
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = load_manifest(manifest_path)
    if args.command == "status":
        payload, healthy = status(manifest, set(args.host) if args.host else None)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if healthy else 1
    return (
        0
        if (
            resume(
                run_dir,
                manifest,
                manifest_path,
                set(args.host) if args.host else None,
            )
            if args.command == "resume"
            else retrieve(
                run_dir,
                manifest,
                manifest_path,
                set(args.host) if args.host else None,
            )
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
