"""Synthetic NetCDF3 contract tests; never touch the raw archive."""

from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import polars as pl
from scipy.io import netcdf_file

from thermodense.saber_fixed_region import prepare_fixed_region
from scripts.audit_saber_scan_identity import audit_scan_identities


def make_file(root: Path, day: date, *, version: str = "1.0", scans: list[dict] | None = None) -> Path:
    scans = scans or [{"orbit": 7, "event": 1, "lat": 20, "lon": -155, "flux": -2, "time": 0}]
    name = f"SABER_CO2_PROFILE_FLUX_{day.year:04d}{day.timetuple().tm_yday:03d}_V{version}.nc"
    path = root / name
    with netcdf_file(path, "w") as ds:
        ds.Dataset_Name = "SABER_CO2_PROFILE_FLUX"
        ds.Dataset_Version = f"V{version}"
        ds.File_Name = name
        ds.createDimension("scan", len(scans))
        ds.createDimension("alt", 3)
        for key, value in (("year", day.year), ("day", day.timetuple().tm_yday)):
            ds.createVariable(key, "i4", ())[...] = value
        altitude = ds.createVariable("altitude", "f8", ("alt",))
        altitude[:] = [100, 119, 139]
        altitude.units = "km"
        for key in ("orbit", "event", "flux"):
            var = ds.createVariable(key, "f8" if key == "flux" else "i4", ("scan",))
            var[:] = [s.get(key, 1) for s in scans]
            if key == "flux":
                var._FillValue = -999.0
                var.units = "Watts/m^2"
        for key, default in (("tplatitude", 20), ("tplongitude", 205), ("time", 0), ("slt", 0), ("sza", 60), ("CO2cool", -0.25)):
            var = ds.createVariable(key, "f8", ("scan", "alt"))
            var[:] = [s.get(key + "_profile", [s.get({"tplatitude": "lat", "tplongitude": "lon"}.get(key, key), default)] * 3) for s in scans]
            if key in ("time", "slt"):
                var.units = "msec"
            if key == "CO2cool":
                var.units = "Watts/m^3"
            if key in ("tplatitude", "tplongitude", "slt", "sza"):
                var._FillValue = -999.0
    return path


def run(root: Path, start: date, end: date, output: str = "new"):
    return prepare_fixed_region(input_dir=root, output_dir=root.parent / output, start=start, end=end)


def test_roi_clock_negative_fill_and_monthly_sampling(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    make_file(raw, date(2023, 12, 31), scans=[
        {"orbit": 1, "event": 1, "lat": 16, "lon": -160, "time": 86_400_000, "slt": 7_200_000, "flux": -2, "sza": 60},
        {"orbit": 1, "event": 2, "lat": 25, "lon": 210, "time": 90_000_000, "slt": -999, "flux": -999, "sza": -999},
        {"orbit": 2, "event": 1, "lat": 20, "lon": -155, "time": 93_600_000, "slt": 50_400_000, "flux": 4, "sza": 100, "tplatitude_profile": [40, 20, 20]},
        {"orbit": 2, "event": 2, "lat": 20, "lon": -999, "time": 94_000_000, "flux": 100},
        {"orbit": 2, "event": 3, "lat": 26, "lon": 205, "time": 95_000_000, "flux": 100},
        {"orbit": 2, "event": 4, "lat": 20, "lon": 900, "time": 96_000_000, "flux": 100},
    ])
    scans, months = run(raw, date(2024, 1, 1), date(2024, 3, 1))
    assert scans.height == 3
    first, second, third = scans.to_dicts()
    assert first["timestamp_utc"] == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert first["longitude_119km_east"] == 200
    assert first["slt_hours"] == 2
    assert first["co2cool_119km_w_m3"] == -0.25
    assert second["official_flux_w_m2"] is None
    assert set(second["qc_flags"]) == {"missing_slt", "missing_sza", "invalid_flux"}
    assert third["full_profile_in_region"] is False
    jan, feb, mar = months.to_dicts()
    assert (jan["mean_flux_w_m2"], jan["median_flux_w_m2"]) == (1, 1)
    assert (jan["n_geo_selected"], jan["n_valid_flux"], jan["n_distinct_utc_dates"], jan["n_orbits"]) == (3, 2, 1, 2)
    assert (jan["n_day"], jan["n_night"], jan["n_unknown_sza"], jan["n_unknown_slt"]) == (1, 1, 1, 1)
    assert (jan["n_slt_00_04"], jan["n_slt_12_16"], jan["n_full_profile_in_region"]) == (1, 1, 2)
    assert feb["mean_flux_w_m2"] is None and feb["n_geo_selected"] == 0
    assert mar["mean_flux_w_m2"] is None and mar["n_valid_flux"] == 0


def test_leap_rollover_version_dedup_and_fail_closed(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    old = make_file(raw, date(2024, 2, 29), version="1.0", scans=[{"orbit": 8, "event": 1, "time": 86_400_000, "flux": 300}])
    selected = make_file(raw, date(2024, 2, 29), version="1.1", scans=[{"orbit": 8, "event": 1, "time": 86_400_000, "flux": -3}])
    adjacent = make_file(raw, date(2024, 3, 1), scans=[{"orbit": 8, "event": 1, "time": 0, "flux": -3}])
    scans, months = run(raw, date(2024, 3, 1), date(2024, 3, 1))
    row = scans.to_dicts()[0]
    assert scans.height == 1 and row["duplicate_records"] == 2
    assert set(row["duplicate_sources"]) == {selected.name, adjacent.name}
    assert row["official_flux_w_m2"] == -3
    assert months["mean_flux_w_m2"][0] == -3
    provenance = json.loads((tmp_path / "new" / "provenance.json").read_text())
    assert old.name not in [r["filename"] for r in provenance["source_files"]]
    assert provenance["source_files"][0]["sha256"] == hashlib.sha256(selected.read_bytes()).hexdigest()
    assert provenance["n_equal_duplicate_records"] == 1
    with pytest.raises(FileExistsError):
        run(raw, date(2024, 3, 1), date(2024, 3, 1))
    assert old.exists() and adjacent.exists()


@pytest.mark.parametrize("override,error", [
    ({"time": -1}, "clock"), ({"time": 172_800_000}, "clock"),
    ({"lat": -999}, None), ({"lon": 999}, None),
])
def test_clock_and_geometry_ranges(tmp_path: Path, override: dict, error: str | None) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    make_file(raw, date(2024, 1, 1), scans=[{"orbit": 1, "event": 1, **override}])
    if error:
        with pytest.raises(ValueError, match=error):
            run(raw, date(2024, 1, 1), date(2024, 1, 1))
    else:
        scans, months = run(raw, date(2024, 1, 1), date(2024, 1, 1))
        assert scans.height == 0 and months["n_geo_selected"][0] == 0


def test_conflict_and_invalid_nonleap_doy(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    make_file(raw, date(2023, 12, 31), scans=[{"orbit": 1, "event": 1, "time": 86_400_000, "flux": 2}])
    make_file(raw, date(2024, 1, 1), scans=[{"orbit": 1, "event": 1, "time": 0, "flux": 3}])
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        run(raw, date(2024, 1, 1), date(2024, 1, 1))
    assert not (tmp_path / "new").exists()
    (raw / "SABER_CO2_PROFILE_FLUX_2023366_V1.0.nc").write_bytes(b"invalid")
    with pytest.raises(ValueError, match="Invalid archive day"):
        run(raw, date(2023, 12, 31), date(2024, 1, 1))


def test_duplicate_profile_footprint_conflict_even_if_reference_matches(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    make_file(raw, date(2023, 12, 31), scans=[{"orbit": 1, "event": 1, "time": 86_400_000}])
    make_file(raw, date(2024, 1, 1), scans=[{
        "orbit": 1, "event": 1, "time": 0,
        "tplatitude_profile": [21, 20, 20],
    }])
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        run(raw, date(2024, 1, 1), date(2024, 1, 1))


def test_missing_source_days_and_input_manifest_hashes(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    make_file(raw, date(2024, 1, 2))
    (raw / "manifest.json").write_text("{}")
    run(raw, date(2024, 1, 1), date(2024, 1, 3))
    info = json.loads((tmp_path / "new" / "provenance.json").read_text())
    assert info["missing_local_archive_days_including_padding"] == ["2023-12-31", "2024-01-01", "2024-01-03"]
    assert info["input_manifests"][0]["sha256"] == hashlib.sha256(b"{}").hexdigest()


def test_declared_ranges_units_header_and_altitude_are_enforced(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    source = make_file(raw, date(2024, 1, 1), scans=[
        {"orbit": 1, "event": 1, "flux": -1, "slt": 86_400_000, "sza": 60},
        {"orbit": 1, "event": 2, "flux": 5, "slt": 3_600_000, "sza": 40},
    ])
    with netcdf_file(source, "a") as ds:
        ds.variables["flux"].valid_range = np.array([-2, 4], dtype="f8")
        ds.variables["sza"].valid_range = np.array([50, 180], dtype="f8")
    scans, months = run(raw, date(2024, 1, 1), date(2024, 1, 1))
    assert scans["official_flux_w_m2"].to_list() == [-1, None]
    assert scans["slt_hours"].to_list() == [None, 1]
    assert scans["sza_deg"].to_list() == [60, None]
    assert months["mean_flux_w_m2"][0] == -1
    with netcdf_file(source, "a") as ds:
        ds.variables["time"].units = "hours"
    with pytest.raises(ValueError, match="msec"):
        run(raw, date(2024, 1, 1), date(2024, 1, 1), "badunits")
    with netcdf_file(source, "a") as ds:
        ds.variables["time"].units = "msec"
        ds.File_Name = "wrong.nc"
    with pytest.raises(ValueError, match="mismatch"):
        run(raw, date(2024, 1, 1), date(2024, 1, 1), "badheader")
    with netcdf_file(source, "a") as ds:
        ds.File_Name = source.name
        ds.variables["altitude"][:] = [100, 118.4, 139]
    with pytest.raises(ValueError, match="altitude"):
        run(raw, date(2024, 1, 1), date(2024, 1, 1), "badalt")


def test_fractional_milliseconds_and_47h_previous_day(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    make_file(raw, date(2023, 12, 31), scans=[
        {"orbit": 1, "event": 1, "time": 47 * 3_600_000 + 0.25},
    ])
    scans, _ = run(raw, date(2024, 1, 1), date(2024, 2, 1))
    assert scans["timestamp_utc"][0] == datetime(2024, 1, 1, 23, microsecond=250, tzinfo=timezone.utc)
    assert scans.schema["timestamp_utc"] == pl.Datetime("us", "UTC")


def test_empty_month_schema_is_stable_and_publication_complete(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    make_file(raw, date(2024, 1, 1), scans=[{"orbit": 1, "event": 1, "lat": 50}])
    scans, monthly = run(raw, date(2024, 1, 1), date(2024, 2, 1))
    assert scans.schema["official_flux_w_m2"] == pl.Float64
    assert monthly.schema["mean_flux_w_m2"] == pl.Float64
    assert monthly.schema["n_geo_selected"] == pl.Int64
    output = tmp_path / "new"
    metadata = json.loads((output / "provenance.json").read_text())
    assert metadata["complete"] is True
    assert (output / "COMPLETE").is_file()
    for name in ("scans.parquet", "monthly.parquet"):
        assert metadata["outputs"][name]["sha256"] == hashlib.sha256((output / name).read_bytes()).hexdigest()


def test_same_physical_identity_different_timestamp_fails(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    make_file(raw, date(2024, 1, 1), scans=[
        {"orbit": 1, "event": 1, "time": 0},
        {"orbit": 1, "event": 1, "time": 0.25},
    ])
    with pytest.raises(ValueError, match="timestamp"):
        run(raw, date(2024, 1, 1), date(2024, 1, 1))


def test_failed_publication_leaves_no_final_directory(tmp_path: Path, monkeypatch) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    make_file(raw, date(2024, 1, 1))
    def fail_write(*args, **kwargs):
        raise OSError("synthetic write failure")
    monkeypatch.setattr(pl.DataFrame, "write_parquet", fail_write)
    with pytest.raises(OSError, match="synthetic write"):
        run(raw, date(2024, 1, 1), date(2024, 1, 1))
    assert not (tmp_path / "new").exists()
    assert not list(tmp_path.glob(".new.*"))


def test_unknown_flux_units_fail_before_output(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    source = make_file(raw, date(2024, 1, 1))
    with netcdf_file(source, "a") as ds:
        ds.variables["flux"].units = "K/day"
    with pytest.raises(ValueError, match="flux.*units"):
        run(raw, date(2024, 1, 1), date(2024, 1, 1))
    assert not (tmp_path / "new").exists()


def test_unknown_point_rate_units_fail_before_output(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    source = make_file(raw, date(2024, 1, 1))
    with netcdf_file(source, "a") as ds:
        ds.variables["CO2cool"].units = "K/day"
    with pytest.raises(ValueError, match="CO2cool.*units"):
        run(raw, date(2024, 1, 1), date(2024, 1, 1))
    assert not (tmp_path / "new").exists()


def test_ambiguous_identity_across_window_boundary_fails(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    make_file(raw, date(2023, 12, 31), scans=[
        {"orbit": 1, "event": 1, "time": 0},
        {"orbit": 1, "event": 1, "time": 86_400_000},
    ])
    with pytest.raises(ValueError, match="timestamp"):
        run(raw, date(2024, 1, 1), date(2024, 1, 1))
    assert not (tmp_path / "new").exists()


def test_source_changed_before_publication_fails(tmp_path: Path, monkeypatch) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    source = make_file(raw, date(2024, 1, 1))
    original = pl.DataFrame.write_parquet
    def change_after_read(frame, *args, **kwargs):
        result = original(frame, *args, **kwargs)
        with netcdf_file(source, "a") as ds:
            ds.variables["flux"][:] = [40]
        return result
    monkeypatch.setattr(pl.DataFrame, "write_parquet", change_after_read)
    with pytest.raises(ValueError, match="Source changed before publication"):
        run(raw, date(2024, 1, 1), date(2024, 1, 1))
    assert not (tmp_path / "new").exists()
    assert not list(tmp_path.glob(".new.*"))


def test_invalid_nonregional_identity_is_explicitly_quarantined(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    make_file(raw, date(2003, 8, 23), scans=[
        {"orbit": 9239, "event": 0, "lat": -11.2, "lon": 288.4, "time": 0},
        {"orbit": 9240, "event": 1, "lat": 20, "lon": 205, "time": 1},
    ])
    scans, _ = run(raw, date(2003, 8, 23), date(2003, 8, 23))
    assert scans.height == 1 and scans["event"][0] == 1
    info = json.loads((tmp_path / "new/provenance.json").read_text())
    assert info["n_quarantined_nonregional_identity_records"] == 1
    rejected = info["source_files"][0]["quarantined_nonregional_identity_records"]
    assert rejected[0]["scan_index"] == 0 and rejected[0]["event_raw"] == "0"


def test_invalid_regional_identity_still_fails_closed(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    make_file(raw, date(2024, 1, 1), scans=[{"orbit": 1, "event": 0}])
    with pytest.raises(ValueError, match="regional.*identity"):
        run(raw, date(2024, 1, 1), date(2024, 1, 1))
    assert not (tmp_path / "new").exists()


def regional(tmp_path: Path, start: date, end: date, *, audit: bool = True, **kwargs):
    raw = tmp_path / "raw"
    audit_dir = tmp_path / "audit"
    if audit:
        audit_scan_identities(input_dir=raw, output_dir=audit_dir, start=start, end=end)
        kwargs.setdefault("expected_identity_audit_sha256", hashlib.sha256((audit_dir / "report.json").read_bytes()).hexdigest())
    return prepare_fixed_region(input_dir=raw, output_dir=tmp_path / "new", start=start, end=end,
                                validation_scope="regional", identity_audit_dir=audit_dir, **kwargs)


def test_regional_nonregional_reuse_and_quarantine_with_bound_provenance(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    make_file(raw, date(2014, 12, 31), scans=[
        {"orbit": 1, "event": 1, "lat": 50, "time": 86_400_000},
        {"orbit": 2, "event": 1, "lat": 50, "time": 86_400_000},
        {"orbit": 3, "event": 1, "time": 86_400_000, "flux": -2},
    ])
    make_file(raw, date(2015, 1, 1), scans=[
        {"orbit": 1, "event": 1, "lat": 50, "time": 0.25},
        {"orbit": 2, "event": 1, "lat": 50, "time": 0, "flux": 20},
        {"orbit": 4, "event": 1, "lat": 50, "time": -1},
        {"orbit": 5, "event": 0, "lat": 50, "time": 0},
        {"orbit": 6, "event": 1, "time": 0, "flux": -999, "slt": -999, "sza": -999},
    ])
    with pytest.raises(ValueError, match="clock"):
        run(raw, date(2015, 1, 1), date(2015, 1, 1), "global")
    scans, monthly = regional(tmp_path, date(2015, 1, 1), date(2015, 1, 1))
    assert scans["orbit"].to_list() == [3, 6]
    assert monthly["n_geo_selected"][0] == 2 and monthly["mean_flux_w_m2"][0] == -2
    info = json.loads((tmp_path / "new/provenance.json").read_text())
    assert info["fatal_validation_scope"] == "regional"
    assert info["n_quarantined_nonregional_clock_records"] == 1
    assert info["source_files"][1]["quarantined_nonregional_clock_records"][0]["time_msec_119km_raw"] == "-1.0"
    assert info["n_quarantined_nonregional_identity_records"] == 1
    assert {g["orbit"] for g in info["identity_audit"]["nonregional_duplicate_groups"]} == {1, 2}
    assert info["identity_audit"]["global_diagnostic_counts"]["duplicate_groups"] == 2
    assert set(info["identity_audit"]["regional_issue_counts"].values()) == {0}
    assert info["identity_audit"]["report_sha256"] == hashlib.sha256((tmp_path / "audit/report.json").read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        regional(tmp_path, date(2015, 1, 1), date(2015, 1, 1), audit=False)


@pytest.mark.parametrize("scans,reason", [
    ([{"orbit": 1, "event": 1, "time": 0}, {"orbit": 1, "event": 1, "time": 1}], "regional issue"),
    ([{"orbit": 1, "event": 1, "time": 0}, {"orbit": 1, "event": 1, "time": 0, "flux": 4}], "regional issue"),
    ([{"orbit": 0, "event": 1}], "regional issue"),
    ([{"orbit": 1, "event": 1, "time": -1}], "regional issue"),
])
def test_regional_invalid_or_conflicting_selected_rejected(tmp_path: Path, scans: list[dict], reason: str) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    make_file(raw, date(2015, 1, 1), scans=scans)
    with pytest.raises(ValueError, match=reason):
        regional(tmp_path, date(2015, 1, 1), date(2015, 1, 1))
    assert not (tmp_path / "new").exists()


def test_regional_audit_preflight_rejects_missing_incomplete_and_unanchored(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    make_file(raw, date(2015, 1, 1))
    with pytest.raises(ValueError, match="missing or incomplete"):
        regional(tmp_path, date(2015, 1, 1), date(2015, 1, 1), audit=False)
    audit_scan_identities(input_dir=raw, output_dir=tmp_path / "audit", start=date(2015, 1, 1), end=date(2015, 1, 1))
    with pytest.raises(ValueError, match="operator anchor"):
        regional(tmp_path, date(2015, 1, 1), date(2015, 1, 1), audit=False)
    (tmp_path / "audit/COMPLETE").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        regional(tmp_path, date(2015, 1, 1), date(2015, 1, 1), audit=False)


@pytest.mark.parametrize("mutation", ["date", "source", "hash", "complete", "audit_code", "group"])
def test_regional_audit_contract_rejects_tampering(tmp_path: Path, mutation: str) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    source = make_file(raw, date(2015, 1, 1))
    audit_dir = tmp_path / "audit"
    audit_scan_identities(input_dir=raw, output_dir=audit_dir, start=date(2015, 1, 1), end=date(2015, 1, 1))
    report_path, provenance_path = audit_dir / "report.json", audit_dir / "provenance.json"
    report, provenance = json.loads(report_path.read_text()), json.loads(provenance_path.read_text())
    if mutation == "date":
        report["start_utc"] = "2015-01-02"
    elif mutation == "source":
        report["source_files"][0]["filename"] = "wrong.nc"
    elif mutation == "hash":
        with netcdf_file(source, "a") as ds:
            ds.variables["flux"][:] = [12]
    elif mutation == "complete":
        report["complete"] = False
    elif mutation == "audit_code":
        provenance["audit_code_sha256"] = "0" * 64
    else:
        report["duplicate_groups"] = [{"geo_any": True, "classification": "different_measurement_payload"}]
        report["counts"]["duplicate_groups"] = 1
    if mutation != "hash":
        report_path.write_text(json.dumps(report))
        provenance["report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
        provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="audit|Audit"):
        regional(tmp_path, date(2015, 1, 1), date(2015, 1, 1), audit=False,
                 expected_identity_audit_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest())
    assert not (tmp_path / "new").exists()


def test_regional_audit_and_source_drift_before_publication(tmp_path: Path, monkeypatch) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    make_file(raw, date(2015, 1, 1))
    audit_scan_identities(input_dir=raw, output_dir=tmp_path / "audit", start=date(2015, 1, 1), end=date(2015, 1, 1))
    digest = hashlib.sha256((tmp_path / "audit/report.json").read_bytes()).hexdigest()
    original = pl.DataFrame.write_parquet
    def drift(frame, *args, **kwargs):
        result = original(frame, *args, **kwargs)
        (tmp_path / "audit/provenance.json").write_text("{}")
        return result
    monkeypatch.setattr(pl.DataFrame, "write_parquet", drift)
    with pytest.raises(ValueError, match="Identity audit changed"):
        regional(tmp_path, date(2015, 1, 1), date(2015, 1, 1), audit=False, expected_identity_audit_sha256=digest)
    assert not (tmp_path / "new").exists()


def test_regional_audit_prefix_and_changed_highest_version_fail(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    make_file(raw, date(2015, 1, 1))
    make_file(raw, date(2015, 1, 2))
    audit_scan_identities(input_dir=raw, output_dir=tmp_path / "audit", start=date(2015, 1, 1),
                          end=date(2015, 1, 2), max_files=1)
    digest = hashlib.sha256((tmp_path / "audit/report.json").read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="incomplete"):
        regional(tmp_path, date(2015, 1, 1), date(2015, 1, 2), audit=False,
                 expected_identity_audit_sha256=digest)
    audit_scan_identities(input_dir=raw, output_dir=tmp_path / "full", start=date(2015, 1, 1), end=date(2015, 1, 2))
    make_file(raw, date(2015, 1, 2), version="1.1")
    full_digest = hashlib.sha256((tmp_path / "full/report.json").read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="source mismatch"):
        prepare_fixed_region(input_dir=raw, output_dir=tmp_path / "new", start=date(2015, 1, 1),
                             end=date(2015, 1, 2), validation_scope="regional", identity_audit_dir=tmp_path / "full",
                             expected_identity_audit_sha256=full_digest)
    assert not (tmp_path / "new").exists()


def test_regional_source_changed_before_publication(tmp_path: Path, monkeypatch) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    source = make_file(raw, date(2015, 1, 1))
    audit_scan_identities(input_dir=raw, output_dir=tmp_path / "audit", start=date(2015, 1, 1), end=date(2015, 1, 1))
    digest = hashlib.sha256((tmp_path / "audit/report.json").read_bytes()).hexdigest()
    original = pl.DataFrame.write_parquet
    def drift(frame, *args, **kwargs):
        result = original(frame, *args, **kwargs)
        with netcdf_file(source, "a") as ds:
            ds.variables["flux"][:] = [10]
        return result
    monkeypatch.setattr(pl.DataFrame, "write_parquet", drift)
    with pytest.raises(ValueError, match="Source changed before publication"):
        regional(tmp_path, date(2015, 1, 1), date(2015, 1, 1), audit=False, expected_identity_audit_sha256=digest)
    assert not (tmp_path / "new").exists()


def test_regional_cli_operator_anchor(tmp_path: Path, monkeypatch) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    make_file(raw, date(2015, 1, 1))
    audit_dir = tmp_path / "audit"
    audit_scan_identities(input_dir=raw, output_dir=audit_dir, start=date(2015, 1, 1), end=date(2015, 1, 1))
    digest = hashlib.sha256((audit_dir / "report.json").read_bytes()).hexdigest()
    from thermodense import saber_fixed_region as module
    monkeypatch.setattr(sys, "argv", ["saber_fixed_region", "--input-dir", str(raw),
                                   "--output-dir", str(tmp_path / "new"), "--start", "2015-01-01",
                                   "--end", "2015-01-01", "--validation-scope", "regional",
                                   "--identity-audit-dir", str(audit_dir), "--expected-identity-audit-sha256", digest])
    module.main()
    assert json.loads((tmp_path / "new/provenance.json").read_text())["fatal_validation_scope"] == "regional"


def test_nonregional_identity_reused_days_later_fails_in_global_mode(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    make_file(raw, date(2024, 1, 1), scans=[{"orbit": 5, "event": 1, "lat": -60, "lon": 10, "time": 0}])
    make_file(raw, date(2024, 1, 2), scans=[{"orbit": 6, "event": 1, "lat": -60, "lon": 10, "time": 0}])
    make_file(raw, date(2024, 1, 3), scans=[{"orbit": 5, "event": 1, "time": 3_600_000}])
    with pytest.raises(ValueError, match="timestamp"):
        run(raw, date(2024, 1, 1), date(2024, 1, 3))
    assert not (tmp_path / "new").exists()


def test_same_orbit_within_its_pass_is_accepted(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    make_file(raw, date(2024, 1, 1), scans=[
        {"orbit": 5, "event": 1, "time": 0},
        {"orbit": 5, "event": 2, "time": 3_000_000},
    ])
    scans, _ = run(raw, date(2024, 1, 1), date(2024, 1, 1))
    assert scans.height == 2
