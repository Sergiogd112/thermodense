"""Synthetic public audit contract; never read the archive."""

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import pytest
from scipy.io import netcdf_file

from scripts.audit_saber_scan_identity import audit_scan_identities
from test_saber_fixed_region import make_file


def audit(tmp_path, start=date(2015, 1, 1), end=date(2015, 1, 1), **kwargs):
    return audit_scan_identities(input_dir=tmp_path / "raw", output_dir=tmp_path / "result",
                                 start=start, end=end, **kwargs)


def test_adjacent_collision_rebased_and_changed_payload(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    make_file(raw, date(2014, 12, 31), scans=[
        {"orbit": 1, "event": 1, "lat": 2.25, "lon": 6.97, "time": 86_445_936},
        {"orbit": 2, "event": 1, "time": 86_400_000},
        {"orbit": 3, "event": 1, "time": 86_400_000},
        {"orbit": 4, "event": 1, "time": 86_400_000},
    ])
    make_file(raw, date(2015, 1, 1), scans=[
        {"orbit": 1, "event": 1, "lat": 2.25, "lon": 6.97, "time": 86_445_936},
        {"orbit": 2, "event": 1, "time": 0},
        {"orbit": 3, "event": 1, "time": 0, "flux": 42},
        {"orbit": 4, "event": 1, "time": 0, "tplatitude_profile": [21, 20, 20]},
    ])
    result = audit(tmp_path)
    groups = {g["orbit"]: g for g in result["duplicate_groups"]}
    assert result["counts"]["archive_records"] == 8
    assert result["counts"]["unique_positive_physical_ids"] == 4
    assert result["counts"]["duplicate_groups"] == 4
    assert groups[1]["classification"] == "identical_raw_payload_different_utc"
    assert groups[1]["utc_deltas_seconds_from_first"] == [0, 86400]
    assert not groups[1]["geo_any"] and not groups[1]["regional_in_utc_window_any"]
    assert groups[2]["classification"] == "rebased_clock_same_utc_identical_measurements"
    assert groups[2]["measurement_payload_equal"] and groups[2]["numeric_measurements_equal"]
    assert groups[3]["classification"] == groups[4]["classification"] == "different_measurement_payload"
    assert {g["orbit"] for g in result["conflicting_regional_groups"]} == {2, 3, 4}
    assert {g["orbit"] for g in result["in_window_regional_issue_groups"]} == {3, 4}
    assert result["counts"]["broad_regional_diagnostic_groups"] == 3
    assert result["counts"]["in_window_regional_issue_groups"] == 2
    assert result["counts"]["regional_absence_check_passes"] == 0
    assert groups[1]["utc_deltas_microseconds_from_first"] == [0, 86_400_000_000]
    assert (tmp_path / "result/COMPLETE").exists()
    assert not (tmp_path / "result/identity.sqlite").exists()


def test_same_file_distant_selection_and_invalid_records(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    make_file(raw, date(2014, 12, 31), scans=[
        {"orbit": 5, "event": 1, "time": 0},
        {"orbit": 5, "event": 1, "time": 86_400_000},
        {"orbit": 0, "event": 1},
        {"orbit": 6, "event": 0, "lat": 50},
        {"orbit": 7, "event": 1, "time": -1},
        {"orbit": 8, "event": 1, "lat": -999},
    ])
    make_file(raw, date(2015, 1, 2), scans=[{"orbit": 5, "event": 1, "lat": 50}])
    result = audit(tmp_path, end=date(2015, 1, 2))
    group = result["duplicate_groups"][0]
    assert group["occurrence"] == "distant" and group["geo_selection_changes"]
    assert group["utc_window_membership_changes"]
    assert group["archive_record_count"] == 3
    assert result["counts"]["invalid_id_records"] == 2
    assert result["counts"]["invalid_id_regional_records"] == 1
    assert result["counts"]["invalid_clock_regional_records"] == 1
    assert result["counts"]["invalid_distinct_records"] == 4
    assert result["counts"]["invalid_geometry_records"] == 1
    assert result["invalid_records"][2]["archive_day"] == "2014-12-31"
    assert result["invalid_records"][2]["conditional_utc"] is None
    assert result["invalid_records"][2]["in_utc_window"] is None
    assert result["invalid_records"][2]["latitude_119km"] == 20
    assert result["invalid_records"][2]["longitude_119km_east"] == 205
    assert result["missing_local_archive_days_including_padding"] == ["2015-01-01"]


def test_version_prefix_empty_region_and_no_overwrite(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    old = make_file(raw, date(2015, 1, 1), version="1.0", scans=[{"orbit": 1, "event": 1}])
    new = make_file(raw, date(2015, 1, 1), version="1.1", scans=[{"orbit": 2, "event": 1, "lat": 50}])
    make_file(raw, date(2015, 1, 2), scans=[{"orbit": 3, "event": 1, "lat": 50}])
    result = audit(tmp_path, end=date(2015, 1, 2), max_files=1)
    assert not result["complete"] and result["files_in_window"] == 2
    assert result["ignored_versions"] == [old.name]
    assert result["source_files"][0]["filename"] == new.name
    assert result["conflicting_regional_groups"] == []
    assert not (tmp_path / "result/COMPLETE").exists()
    assert json.loads((tmp_path / "result/provenance.json").read_text())["complete"] is False
    with pytest.raises(FileExistsError):
        audit(tmp_path)


def test_fractional_time_fills_units_and_schema_fail_closed(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    path = make_file(raw, date(2015, 1, 1), scans=[
        {"orbit": 1, "event": 1, "time": 0.25},
        {"orbit": 1, "event": 1, "time": -999},
    ])
    with netcdf_file(path, "a") as ds:
        ds.variables["time"]._FillValue = -999.0
    result = audit(tmp_path)
    members = result["duplicate_groups"][0]["members"]
    assert members[0]["conditional_utc"] == "2015-01-01T00:00:00.000250+00:00"
    assert members[1]["conditional_utc"] is None
    assert result["duplicate_groups"][0]["classification"] == "invalid_clock_present"
    assert result["duplicate_groups"][0]["has_invalid_clock"]
    assert result["duplicate_groups"][0]["numeric_measurements_equal"]
    assert members[1]["in_utc_window"] is None
    assert len(result["regional_unknown_clock_groups"]) == 1
    with netcdf_file(path, "a") as ds:
        ds.variables["time"].units = "hours"
    with pytest.raises(ValueError, match="msec"):
        audit_scan_identities(input_dir=raw, output_dir=tmp_path / "bad", start=date(2015, 1, 1), end=date(2015, 1, 1))
    assert not (tmp_path / "bad").exists()


def test_same_file_equal_payload_and_regional_clock_collision(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    make_file(raw, date(2015, 1, 1), scans=[
        {"orbit": 1, "event": 1, "time": 0},
        {"orbit": 1, "event": 1, "time": 0},
        {"orbit": 2, "event": 1, "time": 0},
        {"orbit": 2, "event": 1, "time": 1},
    ])
    result = audit(tmp_path)
    groups = {g["orbit"]: g for g in result["duplicate_groups"]}
    assert groups[1]["occurrence"] == "same_file"
    assert groups[1]["classification"] == "identical_raw_payload_same_utc"
    assert groups[2]["classification"] == "valid_different_clocks_same_measurements"
    assert groups[2]["interpreted_utc_equal"] is False
    assert groups[2]["utc_deltas_microseconds_from_first"] == [0, 1000]
    assert groups[2]["numeric_measurements_equal"] and not groups[2]["full_raw_payload_equal"]
    assert result["counts"]["duplicate_archive_records"] == 4
    assert len(result["conflicting_regional_groups"]) == 1


def test_equal_numbers_different_declared_units_are_not_equal_payload(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    make_file(raw, date(2014, 12, 31), scans=[{"orbit": 1, "event": 1, "time": 86_400_000}])
    next_day = make_file(raw, date(2015, 1, 1), scans=[{"orbit": 1, "event": 1, "time": 86_400_000}])
    with netcdf_file(next_day, "a") as ds:
        ds.variables["flux"].units = "W/m2"  # Both labels are accepted by production.
    result = audit(tmp_path)
    group = result["duplicate_groups"][0]
    assert group["classification"] == "different_metadata"
    assert group["numeric_measurements_equal"] and not group["metadata_equal"]
    assert not group["measurement_payload_equal"] and not group["full_raw_payload_equal"]


def test_invalid_clock_and_changed_flux_are_orthogonal(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    make_file(raw, date(2015, 1, 1), scans=[
        {"orbit": 1, "event": 1, "time": 0, "flux": 3},
        {"orbit": 1, "event": 1, "time": -1, "flux": 4},
    ])
    result = audit(tmp_path)
    group = result["duplicate_groups"][0]
    assert group["classification"] == "invalid_clock_present"
    assert group["has_invalid_clock"] and not group["numeric_measurements_equal"]
    assert not group["measurement_payload_equal"] and group["interpreted_utc_equal"] is None
    assert group["utc_window_membership_changes"]
    assert group["members"][1]["in_utc_window"] is None
    assert result["counts"]["regional_unknown_clock_groups"] == 1
    assert len(result["in_window_regional_issue_groups"]) == 1


def test_padding_regional_outside_window_and_in_window_nonregional(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    make_file(raw, date(2014, 12, 31), scans=[{"orbit": 1, "event": 1, "time": 0}])
    make_file(raw, date(2015, 1, 1), scans=[{"orbit": 1, "event": 1, "lat": 50, "time": 0}])
    result = audit(tmp_path)
    group = result["duplicate_groups"][0]
    assert group["geo_any"] and not group["regional_in_utc_window_any"]
    assert group["members"][0]["in_utc_window"] is False
    assert group["members"][1]["in_utc_window"] is True
    assert group["occurrence"] == "adjacent" and group["origin_day_gap_days"] == 1
    assert result["counts"]["broad_regional_diagnostic_groups"] == 1
    assert result["counts"]["in_window_regional_issue_groups"] == 0
    assert result["in_window_regional_issue_groups"] == []
    assert result["counts"]["regional_absence_check_passes"] == 1


def test_invalid_regional_identity_prevents_absence_claim_even_without_groups(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    make_file(raw, date(2015, 1, 1), scans=[{"orbit": 0, "event": 1}])
    result = audit(tmp_path)
    assert result["counts"]["unique_positive_physical_ids"] == 0
    assert result["duplicate_groups"] == [] and result["in_window_regional_issue_groups"] == []
    assert result["counts"]["invalid_id_regional_records"] == 1
    assert result["counts"]["regional_absence_check_passes"] == 0


def test_output_provenance_hash_bindings_and_cli(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    source = make_file(raw, date(2015, 1, 1))
    import scripts.audit_saber_scan_identity as module

    production = Path(module.fixed.__file__)
    before = hashlib.sha256(production.read_bytes()).hexdigest()
    monkeypatch.setattr(sys, "argv", ["audit_saber_scan_identity.py", "--input-dir", str(raw),
                                  "--output-dir", str(tmp_path / "result"), "--start", "2015-01-01", "--end", "2015-01-01"])
    module.main()
    report = json.loads((tmp_path / "result/report.json").read_text())
    provenance = json.loads((tmp_path / "result/provenance.json").read_text())
    assert report["counts"]["archive_records"] == 1
    assert provenance["source_sha256"][source.name] == report["source_files"][0]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert provenance["audit_code_sha256"] == hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    assert provenance["production_code_sha256"] == before == hashlib.sha256(production.read_bytes()).hexdigest()
    assert provenance["report_sha256"] == hashlib.sha256((tmp_path / "result/report.json").read_bytes()).hexdigest()
    assert provenance["limits"]["duplicate_members_in_memory"] == 100_000
    assert provenance["limits"]["invalid_records_in_memory"] == 100_000
    assert (tmp_path / "result/COMPLETE").exists()


@pytest.mark.parametrize("cap,fixture,reason", [
    ("GROUP_LIMIT", [{"orbit": 1, "event": 1}, {"orbit": 1, "event": 1}], "duplicate-group"),
    ("MEMBER_LIMIT", [{"orbit": 1, "event": 1}, {"orbit": 1, "event": 1}], "duplicate-member"),
    ("INVALID_LIMIT", [{"orbit": 0, "event": 1}, {"orbit": 0, "event": 2}], "invalid-record"),
    ("DISK_LIMIT", [{"orbit": 1, "event": 1}], "disk guard"),
])
def test_budget_guards_fail_without_publication(tmp_path, monkeypatch, cap, fixture, reason):
    raw = tmp_path / "raw"
    raw.mkdir()
    make_file(raw, date(2015, 1, 1), scans=fixture)
    import scripts.audit_saber_scan_identity as module

    monkeypatch.setattr(module, cap, 0 if cap == "GROUP_LIMIT" else 1)
    with pytest.raises(RuntimeError, match=reason):
        audit(tmp_path)
    assert not (tmp_path / "result").exists() and not list(tmp_path.glob(".result.*"))


def test_deadline_fails_without_publication(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    make_file(raw, date(2015, 1, 1))
    with pytest.raises(TimeoutError, match="deadline"):
        audit(tmp_path, max_seconds=1e-12)
    assert not (tmp_path / "result").exists() and not list(tmp_path.glob(".result.*"))


def test_duplicate_revisit_detects_source_drift(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    source = make_file(raw, date(2015, 1, 1), scans=[
        {"orbit": 1, "event": 1}, {"orbit": 1, "event": 1},
    ])
    import scripts.audit_saber_scan_identity as module

    original = module._hash
    calls = 0

    def changed(path, limit):
        nonlocal calls
        if path == source:
            calls += 1
            if calls == 3:
                return "0" * 64
        return original(path, limit)

    monkeypatch.setattr(module, "_hash", changed)
    with pytest.raises(ValueError, match="Source changed before duplicate read"):
        audit(tmp_path)
    assert not (tmp_path / "result").exists() and not list(tmp_path.glob(".result.*"))


def test_source_drift_and_failed_publication_leave_no_result(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    path = make_file(raw, date(2015, 1, 1))
    import scripts.audit_saber_scan_identity as module

    original = module._hash
    calls = 0

    def drift(source, deadline):
        nonlocal calls
        if source == path:
            calls += 1
            if calls == 3:
                return "0" * 64
        return original(source, deadline)

    monkeypatch.setattr(module, "_hash", drift)
    with pytest.raises(ValueError, match="Source changed before publication"):
        audit(tmp_path)
    assert not (tmp_path / "result").exists()
    assert not list(tmp_path.glob(".result.*"))
    monkeypatch.setattr(module, "_hash", original)
    monkeypatch.setattr(module.fixed, "_publish_new", lambda *args: (_ for _ in ()).throw(OSError("publication failed")))
    with pytest.raises(OSError, match="publication failed"):
        audit(tmp_path)
    assert not (tmp_path / "result").exists()
    assert not list(tmp_path.glob(".result.*"))
