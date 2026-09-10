"""Read-only execution-monitor contracts."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from thermodense.checkpoints import fingerprint
from thermodense.benchmarks import raw_3hour_pcmci
from thermodense.execution_monitor import monitor
from thermodense.workflow_policies import DAILY_ACTIVITY_CLASSIFICATION, DAILY_TRANSFORMATIONS


def _manifest(path: Path, core: bytes = b"immutable input") -> Path:
    core_path = path.parent / "input.npz"
    core_path.write_bytes(core)
    path.write_text(
        json.dumps(
            {
                "input_contract": "global-activity-daily-v1",
                "core_npz": "input.npz",
                "core_sha256": fingerprint(core_path)["sha256"],
                "calendar": {"cadence": "daily UTC"},
                "transformations": DAILY_TRANSFORMATIONS,
                "f107_fit": "F10.7_OBS raw",
                "activity_classification": DAILY_ACTIVITY_CLASSIFICATION,
            }
        )
    )
    return path


def _poly(root: Path, manifest: Path) -> Path:
    root.mkdir()
    (root / "models.npz").write_bytes(b"matrices")
    (root / "metrics.json").write_text("{}")
    hashes = {
        name: fingerprint(root / name)["sha256"]
        for name in ("models.npz", "metrics.json")
    }
    root.joinpath("provenance.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "result_hashes": hashes,
                "config": {"lag": 180},
                "interpretation": "predictive sensitivity benchmark; coefficients are not causal",
                "optimization_status": "bounded epoch budget; neither convergence nor globally optimal ridge is claimed",
                "evaluation_activity": "centered81 only; low <100, moderate 100<=F<160, high >=160; unknown excluded from grouped summaries",
                "manifest_identity": json.loads(manifest.read_text()),
            }
        )
    )
    return root


def _case(tmp_path: Path, name: str = "case") -> tuple[Path, Path]:
    case = tmp_path / name
    case.mkdir()
    manifest = _manifest(case / "manifest.json")
    return _poly(case / "run", manifest), manifest


def _runner(responses, recorded):
    def run(argv, **kwargs):
        recorded.append((argv, kwargs))
        command = argv[-1]
        response = responses.get(command, "")
        if isinstance(response, BaseException):
            raise response
        if isinstance(response, tuple):
            returncode, stdout, stderr = response
            return subprocess.CompletedProcess(argv, returncode, stdout, stderr)
        return subprocess.CompletedProcess(argv, 0, response, "")

    return run


def test_completed_polynomial_requires_verified_input_artifacts_and_contract(tmp_path):
    run, manifest = _case(tmp_path)
    report = monitor("global-polynomial", run, manifest)
    assert report["status"] == "verified_complete"
    assert report["artifacts"]["integrity"] == "verified"
    assert report["input"]["integrity"] == "verified"
    assert report["policy"] == {"matches_contract": True, "status": "contract_matches"}


@pytest.mark.parametrize("change", ("missing", "corrupt"))
def test_declared_completion_with_bad_artifacts_is_not_verified(tmp_path, change):
    run, manifest = _case(tmp_path)
    if change == "missing":
        (run / "models.npz").unlink()
    else:
        (run / "models.npz").write_bytes(b"tampered")
    report = monitor("global-polynomial", run, manifest)
    assert report["completion"]["declared"]
    assert report["artifacts"]["integrity"] == "unverified"
    assert report["status"] == "unverified"


def test_same_policy_but_unrelated_input_cannot_qualify_result(tmp_path):
    run, original = _case(tmp_path, "original")
    other_root = tmp_path / "other"
    other_root.mkdir()
    unrelated = _manifest(other_root / "manifest.json", b"different immutable input")
    report = monitor("global-polynomial", run, unrelated)
    assert report["policy"]["matches_contract"]
    assert report["input"]["integrity"] == "unverified"
    assert report["input"]["reason"] == "input_manifest_identity_mismatch"
    assert report["status"] == "unverified"
    assert original != unrelated


def test_embedded_manifest_is_declared_only_not_byte_verification(tmp_path):
    run, manifest = _case(tmp_path)
    report = monitor("global-polynomial", run)
    assert report["policy"]["status"] == "contract_matches"
    assert report["input"]["integrity"] == "declared_only"
    assert report["status"] == "unverified"
    assert manifest.is_file()


def test_missing_root_and_running_metadata_are_not_verified(tmp_path):
    missing = monitor("global-polynomial", tmp_path / "absent")
    root = tmp_path / "running"
    root.mkdir()
    (root / "provenance.json").write_text('{"status": "running"}')
    running = monitor("global-polynomial", root)
    assert missing["status"] == running["status"] == "unverified"
    assert not running["completion"]["declared"]


def test_required_pcmci_outputs_and_hash_syntax_fail_closed(tmp_path):
    case = tmp_path / "pcmci"
    case.mkdir()
    manifest = _manifest(case / "manifest.json")
    run = case / "run"
    run.mkdir()
    (run / "eligible_links.csv").write_text("source,target\n")
    metadata = {
        "status": "complete",
        "tau_max": 180,
        "interpretation": "retrospective association/sensitivity only; not causal qualification",
        "algorithm": {"ci_test": "analytic ParCorr", "mask_type": "y"},
        "input_identity": {
            "manifest_sha256": fingerprint(manifest)["sha256"],
            "core_sha256": json.loads(manifest.read_text())["core_sha256"],
        },
        "result_files": {"eligible_links.csv": "not-a-digest"},
    }
    (run / "status.json").write_text(json.dumps(metadata))
    report = monitor("global-parcorr", run, manifest)
    assert report["artifacts"]["integrity"] == "unverified"
    assert "required artifact" in report["artifacts"]["reason"]
    assert report["status"] == "unverified"


def test_raw_monitor_verifies_both_core_and_saber_bytes(tmp_path):
    case = tmp_path / "raw"
    case.mkdir()
    core_path = case / "core.npz"
    saber_path = case / "saber.npz"
    core_path.write_bytes(b"core")
    saber_path.write_bytes(b"saber")
    legacy_digest, provenance_digest = "0" * 64, "1" * 64
    manifest_payload = {
        "input_contract": "raw-36-node-native-3hour-v1",
        "input_kind": "canonical",
        "calendar": raw_3hour_pcmci.CANONICAL_CALENDAR,
        "core_npz": core_path.name,
        "core_sha256": fingerprint(core_path)["sha256"],
        "saber_npz": saber_path.name,
        "saber_sha256": fingerprint(saber_path)["sha256"],
        "f10_7": {"representation": "raw_daily_held", "data_type": "OBS only", "non_obs": "unavailable", "interpolation": "none"},
        "transformations": DAILY_TRANSFORMATIONS,
        "source_node_order": list(raw_3hour_pcmci.LEGACY_SOURCE_NODE_ORDER),
        "legacy_prepared_semantics": {
            "prepared_artifact_status": "verified",
            "upstream_semantics_status": "declared_by_bound_legacy_provenance",
            "schema": raw_3hour_pcmci.LEGACY_PROVENANCE_SCHEMA,
            "source_version": raw_3hour_pcmci.LEGACY_PROVENANCE_VERSION,
            "prepared_input_sha256": legacy_digest,
            "provenance_sha256": provenance_digest,
            "channel_names": list(raw_3hour_pcmci.LEGACY_SOURCE_NODE_ORDER),
            "shape": [raw_3hour_pcmci.REMOTE_SLOTS, len(raw_3hour_pcmci.LEGACY_SOURCE_NODE_ORDER)],
            "calendar": {"cadence": raw_3hour_pcmci.CALENDAR_CADENCE},
            "f107_policy": raw_3hour_pcmci.LEGACY_F107_POLICY,
            "co2_missing_policy": raw_3hour_pcmci.LEGACY_CO2_MISSING_POLICY,
            "upstream_source_sha256": {
                "analysis_bundle.npz": "5" * 64,
                "final_arrays.npz": "6" * 64,
                "final_report.json": "7" * 64,
                "co2_daily_mlo.csv": "8" * 64,
            },
        },
        "data_mobility": raw_3hour_pcmci.DATA_MOBILITY_POLICY,
        "sources": {
            "legacy_prepared_npz": {"sha256": legacy_digest, "provenance_sha256": provenance_digest},
            "celestrak_sw_all_csv": {"sha256": "3" * 64, "data_type_policy": raw_3hour_pcmci.F107_SOURCE_POLICY},
            "saber_50x70_npz": {"sha256": "4" * 64, "channels": list(raw_3hour_pcmci.SABER_CHANNELS)},
        },
    }
    manifest = case / "manifest.json"
    manifest.write_text(json.dumps(manifest_payload))
    run = case / "run"
    run.mkdir()
    for name in ("pcmci_results.npz", "eligible_links.csv"):
        (run / name).write_bytes(name.encode())
    metadata = {
        "status": "completed",
        "nodes": raw_3hour_pcmci.node_names(),
        "algorithm": raw_3hour_pcmci.ALGORITHM_POLICY,
        "missing_data_policy": raw_3hour_pcmci.MISSING_DATA_POLICY,
        "link_eligibility": raw_3hour_pcmci.LINK_ELIGIBILITY,
        "input_availability": {name: raw_3hour_pcmci.REMOTE_SLOTS for name in raw_3hour_pcmci.node_names()},
        "code": {"raw_3hour_pcmci_sha256": "2" * 64},
        "versions": {"python": "3.14", "numpy": "2.4", "scipy": "1.17", "statsmodels": "0.14", "tigramite": {"version": "5.2.10.1", "commit": raw_3hour_pcmci.TIGRAMITE_COMMIT}},
        "bh_family": {"method": "Benjamini-Hochberg", "scope": raw_3hour_pcmci.BH_SCOPE, "test_count": raw_3hour_pcmci.bh_test_count()},
        "stationarity": {
            "causal_interpretation_eligible": False,
            "causal_wording_permitted": False,
            "sensitivity_evidence_only": True,
        },
        "interpretation": "association/sensitivity-only; this raw workflow never permits causal wording",
        "input_manifest": manifest_payload,
        "result_files": {
            name: fingerprint(run / name)["sha256"]
            for name in ("pcmci_results.npz", "eligible_links.csv")
        },
    }
    (run / "provenance.json").write_text(json.dumps(metadata))

    assert monitor("raw-36-parcorr", run, manifest)["status"] == "verified_complete"
    fixture_payload = {**manifest_payload, "input_kind": "fixture"}
    fixture_manifest = case / "fixture-manifest.json"
    fixture_manifest.write_text(json.dumps(fixture_payload))
    fixture_report = monitor("raw-36-parcorr", run, fixture_manifest)
    assert fixture_report["status"] == "unverified"
    assert fixture_report["policy"]["status"] == "policy_unverified"
    saber_path.write_bytes(b"tampered")
    report = monitor("raw-36-parcorr", run, manifest)
    assert report["status"] == "unverified"
    assert "saber.npz" in report["input"]["reason"]


def test_comparison_does_not_treat_upstream_hashes_as_derived_file_integrity(tmp_path):
    case = tmp_path / "comparison"
    case.mkdir()
    manifest = _manifest(case / "manifest.json")
    payload = {
        "input_manifest_sha256": fingerprint(manifest)["sha256"],
        "input_core_sha256": json.loads(manifest.read_text())["core_sha256"],
        "result_hashes": {"models.npz": "0" * 64, "metrics.json": "1" * 64},
        "settings": {"lag_days": 180},
        "interpretation": "Predictive result; coefficients are not causal.",
    }
    output = case / "comparison.json"
    output.write_text(json.dumps(payload))
    report = monitor("comparison", output, manifest)
    assert report["completion"]["declared"]
    assert report["policy"]["matches_contract"]
    assert report["artifacts"]["integrity"] == "unverified"
    assert "trusted_output_digest" in report["artifacts"]["reason"]
    assert report["status"] == "unverified"


def test_sge_running_parses_real_xml_seconds_limit_and_only_read_commands(tmp_path):
    run_dir, manifest = _case(tmp_path)
    recorded = []
    responses = {
        'qstat -xml -u "$USER"': (
            "<job_info><queue_info><job_list state='running'>"
            "<JB_job_number>1139</JB_job_number><state>r</state>"
            "<JAT_start_time>2026-09-10T10:00:00+02:00</JAT_start_time>"
            "</job_list></queue_info></job_info>"
        ),
        "qstat -j 1139": "hard resource_list: h_rt=7200,s_rt=7100",
        'qquota -u "$USER"': "available slots: 4",
    }
    report = monitor(
        "global-polynomial",
        run_dir,
        manifest,
        ssh_host="phoenix",
        job_id=1139,
        scheduler_timezone="Europe/Madrid",
        run=_runner(responses, recorded),
    )
    assert report["scheduler"]["state"] == "running"
    assert report["scheduler"]["actual_start_at"] == "2026-09-10T08:00:00+00:00"
    assert report["scheduler"]["deadline_at"] == "2026-09-10T10:00:00+00:00"
    assert report["scheduler"]["quota"] == {"actual_available_slots": 4}
    assert all(kwargs["shell"] is False for _, kwargs in recorded)
    assert all(argv[1:6] == ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "--"] for argv, _ in recorded)
    assert all(not any(word in argv[-1] for word in ("qsub", "qdel", "qalter", "rsync")) for argv, _ in recorded)


def test_queued_and_error_states_never_query_a_deadline(tmp_path):
    queued_run, queued_manifest = _case(tmp_path, "queued")
    queued_recorded = []
    queued = monitor(
        "global-polynomial",
        queued_run,
        queued_manifest,
        ssh_host="phoenix",
        job_id=1139,
        run=_runner(
            {
                'qstat -xml -u "$USER"': "<job_info><job_list state='pending'><JB_job_number>1139</JB_job_number><state>qw</state></job_list></job_info>",
                'qquota -u "$USER"': "unavailable",
            },
            queued_recorded,
        ),
    )
    error_run, error_manifest = _case(tmp_path, "error")
    error = monitor(
        "global-polynomial",
        error_run,
        error_manifest,
        ssh_host="phoenix",
        job_id=1139,
        run=_runner(
            {
                'qstat -xml -u "$USER"': "<job_info><job_list state='pending'><JB_job_number>1139</JB_job_number><state>Eqw</state></job_list></job_info>",
                'qquota -u "$USER"': "unavailable",
            },
            [],
        ),
    )
    assert queued["scheduler"]["state"] == "queued"
    assert queued["scheduler"]["deadline"] == "unknown"
    assert error["scheduler"]["state"] == "error"
    assert all(argv[-1] != "qstat -j 1139" for argv, _ in queued_recorded)
    assert queued["status"] == error["status"] == "unverified"


def test_accounting_requires_one_complete_matching_record(tmp_path):
    valid_run, valid_manifest = _case(tmp_path, "valid")
    valid = monitor(
        "global-polynomial",
        valid_run,
        valid_manifest,
        ssh_host="phoenix",
        job_id=1139,
        run=_runner(
            {
                'qstat -xml -u "$USER"': "<job_info/>",
                "qacct -j 1139": "jobnumber 1139\nfailed 0\nexit_status 0\n",
            },
            [],
        ),
    )
    assert valid["scheduler"] == {"state": "completed", "failed": 0, "exit_status": 0}
    assert valid["status"] == "verified_complete"

    for name, accounting in {
        "missing": "jobnumber 1139\nfailed 0\n",
        "mismatch": "jobnumber 1140\nfailed 0\nexit_status 0\n",
        "ambiguous": "jobnumber 1139\nfailed 0\nexit_status 0\n====\njobnumber 1139\nfailed 0\nexit_status 0\n",
    }.items():
        run_dir, manifest = _case(tmp_path, name)
        report = monitor(
            "global-polynomial",
            run_dir,
            manifest,
            ssh_host="phoenix",
            job_id=1139,
            run=_runner({'qstat -xml -u "$USER"': "<job_info/>", "qacct -j 1139": accounting}, []),
        )
        assert report["scheduler"]["state"] == "unknown"
        assert report["status"] == "unverified"


def test_failed_accounting_unreachable_and_injection_are_distinct(tmp_path):
    run_dir, manifest = _case(tmp_path)
    failed = monitor(
        "global-polynomial",
        run_dir,
        manifest,
        ssh_host="phoenix",
        job_id=1139,
        run=_runner(
            {
                'qstat -xml -u "$USER"': "<job_info/>",
                "qacct -j 1139": "jobnumber 1139\nfailed 37\nexit_status 137\n",
            },
            [],
        ),
    )
    assert failed["scheduler"] == {"state": "failed", "failed": 37, "exit_status": 137}

    unreachable = monitor(
        "global-polynomial",
        run_dir,
        manifest,
        ssh_host="phoenix",
        job_id=1139,
        run=lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 255, "", ""),
    )
    assert unreachable["scheduler"]["state"] == "unreachable"
    with pytest.raises(ValueError):
        monitor("global-polynomial", run_dir, manifest, ssh_host="phoenix; qdel 1", job_id=1139)
    with pytest.raises(ValueError):
        monitor("global-polynomial", run_dir, manifest, ssh_host="phoenix", job_id="1139; qdel 1")
    with pytest.raises(ValueError, match="timezone"):
        monitor("global-polynomial", run_dir, manifest, ssh_host="phoenix", job_id=1139, scheduler_timezone="Mars/Olympus")


def test_qacct_timeout_is_unreachable_not_an_exception(tmp_path):
    run_dir, manifest = _case(tmp_path)
    timeout = subprocess.TimeoutExpired("ssh", 15)
    report = monitor(
        "global-polynomial",
        run_dir,
        manifest,
        ssh_host="phoenix",
        job_id=1139,
        run=_runner({'qstat -xml -u "$USER"': "<job_info/>", "qacct -j 1139": timeout}, []),
    )
    assert report["scheduler"] == {"state": "unreachable", "reason": "TimeoutExpired"}


def test_checkpoint_presence_is_never_resume_approval(tmp_path):
    run_dir, manifest = _case(tmp_path)
    (run_dir / "checkpoint.pt").write_bytes(b"never deserialize this")
    report = monitor("global-polynomial", run_dir, manifest)
    assert report["recovery"]["resume_readiness"] == "not_assessed; capability is not checkpoint approval"
