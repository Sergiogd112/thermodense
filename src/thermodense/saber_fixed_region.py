"""Descriptive fixed-region SABER CO2 official-flux scan and monthly products."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import re
import shutil
import tempfile
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, SupportsFloat

import numpy as np
import polars as pl
import scipy
from scipy.io import netcdf_file

FILE_RE = re.compile(r"SABER_CO2_PROFILE_FLUX_(\d{4})(\d{3})_V(\d+)\.(\d+)\.nc")
REGION = (16.0, 25.0, 200.0, 210.0)
REFERENCE_KM = 119.0
ALT_TOLERANCE_KM = 0.5
# Audit v1 uses the fixed 119-km geometry above. The report digest is the
# operator-approved full-window result; a different report needs an explicit anchor.
AUDIT_CODE_SHA256 = "ef77f8a3db993ad4c2f1983c442333d195fc6dc2789ca92fb4499de4a2aa7727"
APPROVED_AUDIT_SHA256 = "21fa31e144393d672ad840884c51166bd33db29ef3d0dee1d4c794cb56b2c247"
SCAN_SCHEMA = {
    "source_file": pl.String, "source_sha256": pl.String, "product_version": pl.String,
    "archive_day": pl.Date, "orbit": pl.Int64, "event": pl.Int64, "scan_index": pl.Int64,
    "timestamp_utc": pl.Datetime("us", "UTC"), "reference_altitude_km": pl.Float64,
    "latitude_119km": pl.Float64, "longitude_119km_east": pl.Float64,
    "geo_selected": pl.Boolean, "full_profile_in_region": pl.Boolean,
    "official_flux_w_m2": pl.Float64, "valid_flux": pl.Boolean,
    "slt_hours": pl.Float64, "sza_deg": pl.Float64, "co2cool_119km_w_m3": pl.Float64,
    "qc_flags": pl.List(pl.String), "duplicate_records": pl.Int64,
    "duplicate_sources": pl.List(pl.String),
}
MONTH_SCHEMA = {
    "month": pl.Date, "mean_flux_w_m2": pl.Float64, "median_flux_w_m2": pl.Float64,
    **{name: pl.Int64 for name in (
        "n_geo_selected", "n_valid_flux", "n_distinct_utc_dates", "n_orbits", "n_day",
        "n_night", "n_unknown_sza", "n_unknown_slt", "n_missing_flux", "n_full_profile_in_region",
    )},
    **{f"n_slt_{i * 4:02d}_{(i + 1) * 4:02d}": pl.Int64 for i in range(6)},
}


def _day(match: re.Match[str]) -> date:
    year, doy = int(match[1]), int(match[2])
    first = date(year, 1, 1)
    day = first + timedelta(days=doy - 1)
    if doy < 1 or day.year != year:
        raise ValueError(f"Invalid archive day {year}{doy:03d}")
    return day


def _text(value: object) -> str:
    return value.decode("utf-8").strip() if isinstance(value, bytes) else str(value).strip()


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _values(ds, name: str, shape: tuple[int, ...]) -> np.ndarray:
    if name not in ds.variables:
        raise ValueError(f"Missing required variable {name}")
    variable = ds.variables[name]
    result = np.asarray(variable.data)
    if result.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {result.shape}")
    return result


def _integer(variable, value: SupportsFloat | str | bytes, name: str) -> int:
    number = _number(variable, value, low=1)
    if number is None or not number.is_integer():
        raise ValueError(f"Invalid integer {name}")
    return int(number)


def _unit(variable, name: str, accepted: set[str]) -> str:
    value = _text(getattr(variable, "units", "")).lower().replace(" ", "")
    if value not in accepted:
        raise ValueError(f"{name}: unsupported or missing units {value!r}")
    return value


def _publish_new(staging: Path, output: Path) -> None:
    """Linux atomic no-replace directory publication, including competing writers."""
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        raise RuntimeError("Atomic no-replace directory publication requires renameat2")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(staging), -100, os.fsencode(output), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(output))
    fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _number(variable, value: SupportsFloat | str | bytes, *, low: float | None = None, high: float | None = None) -> float | None:
    """Respect declared fills/ranges before any transformation; never reject by sign."""
    number = float(value)
    if not np.isfinite(number):
        return None
    for attr in ("_FillValue", "missing_value"):
        if hasattr(variable, attr) and np.any(number == np.asarray(getattr(variable, attr), dtype=float)):
            return None
    if hasattr(variable, "valid_range"):
        bounds = np.asarray(variable.valid_range, dtype=float).ravel()
        if len(bounds) != 2:
            raise ValueError("Malformed valid_range")
        low, high = max(low, bounds[0]) if low is not None else bounds[0], min(high, bounds[1]) if high is not None else bounds[1]
    if hasattr(variable, "valid_min"):
        bound = float(variable.valid_min)
        low = max(low, bound) if low is not None else bound
    if hasattr(variable, "valid_max"):
        bound = float(variable.valid_max)
        high = min(high, bound) if high is not None else bound
    return number if (low is None or number >= low) and (high is None or number <= high) else None


def _inside(lat: float | None, lon: float | None) -> bool:
    return lat is not None and lon is not None and REGION[0] <= lat <= REGION[1] and REGION[2] <= lon <= REGION[3]


def _identity_audit(audit_dir: Path, files: list[Path], start: date, end: date,
                    expected_sha256: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("Invalid expected identity audit SHA-256")
    report_path = audit_dir / "report.json"
    provenance_path = audit_dir / "provenance.json"
    if (not (audit_dir / "COMPLETE").is_file()
            or (audit_dir / "COMPLETE").read_text(encoding="utf-8") != "complete identity audit\n"
            or not report_path.is_file() or not provenance_path.is_file()):
        raise ValueError("Identity audit is missing or incomplete")
    report_sha = _hash(report_path)
    provenance_sha = _hash(provenance_path)
    if report_sha != expected_sha256:
        raise ValueError("Identity audit report SHA-256 does not match operator anchor")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if (report.get("complete") is not True or provenance.get("complete") is not True
            or provenance.get("report_sha256") != report_sha
            or provenance.get("audit_code_sha256") != AUDIT_CODE_SHA256
            or not re.fullmatch(r"[0-9a-f]{64}", str(provenance.get("production_code_sha256", "")))
            or report.get("start_utc") != start.isoformat() or report.get("end_utc") != end.isoformat()
            or report.get("files_processed") != len(files) or report.get("files_in_window") != len(files)):
        raise ValueError("Identity audit contract, dates, code or completion mismatch")
    sources = report.get("source_files")
    if not isinstance(sources, list) or len(sources) != len(files):
        raise ValueError("Identity audit source list mismatch")
    for path, source in zip(files, sources):
        match = FILE_RE.fullmatch(path.name)
        assert match is not None
        if (source.get("filename") != path.name or source.get("archive_day") != _day(match).isoformat()
                or source.get("product_version") != f"V{match[3]}.{match[4]}"
                or provenance.get("source_sha256", {}).get(path.name) != source.get("sha256")
                or _hash(path) != source.get("sha256")):
            raise ValueError(f"Identity audit source mismatch: {path.name}")
    if set(provenance["source_sha256"]) != {path.name for path in files}:
        raise ValueError("Identity audit source map mismatch")
    counts = report.get("counts", {})
    gates = ("in_window_regional_issue_groups", "regional_unknown_clock_groups",
             "invalid_id_regional_records", "invalid_clock_regional_records")
    if (any(counts.get(key) != 0 or report.get(key, []) != [] for key in gates[:2])
            or any(counts.get(key) != 0 for key in gates[2:])
            or counts.get("regional_absence_check_passes") != 1):
        raise ValueError("Identity audit regional issue gate failed")
    # The in-window gate alone misses distant reuse involving an outside-window
    # regional member. Never scope such a physical pair down to nonregional.
    groups = report.get("duplicate_groups", [])
    if (counts.get("duplicate_groups") != len(groups)
            or any(group.get("geo_any") and group.get("classification") not in (
                "identical_raw_payload_same_utc", "rebased_clock_same_utc_identical_measurements")
                for group in groups)):
        raise ValueError("Identity audit includes conflicting regional physical identity")
    return {"directory": str(audit_dir), "report_sha256": report_sha,
            "provenance_sha256": provenance_sha, "audit_code_sha256": AUDIT_CODE_SHA256,
            "source_sha256": {s["filename"]: s["sha256"] for s in sources},
            "production_code_sha256_at_audit": provenance["production_code_sha256"],
            "global_diagnostic_counts": counts,
            "nonregional_duplicate_groups": [group for group in groups if not group["geo_any"]],
            "regional_issue_counts": {key: counts[key] for key in gates}}


def _file_rows(path: Path, day: date, version: str, *, validation_scope: str = "global") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    digest = _hash(path)
    with netcdf_file(path, "r", mmap=False) as ds:
        required = {"year", "day", "altitude", "orbit", "event", "flux", "time", "slt", "sza", "CO2cool", "tplatitude", "tplongitude"}
        missing = sorted(required - set(ds.variables))
        if missing:
            raise ValueError(f"{path.name}: missing required variables {missing}")
        if _text(getattr(ds, "File_Name", "")) != path.name or _text(getattr(ds, "Dataset_Version", "")) != version:
            raise ValueError(f"Filename/version header mismatch: {path}")
        if _text(getattr(ds, "Dataset_Name", "")) != "SABER_CO2_PROFILE_FLUX":
            raise ValueError(f"Unexpected dataset: {path}")
        if (np.asarray(ds.variables["year"].data).size != 1 or np.asarray(ds.variables["day"].data).size != 1):
            raise ValueError(f"{path.name}: archive year/day must be scalar")
        if _integer(ds.variables["year"], np.asarray(ds.variables["year"].data).item(), "year") != day.year or _integer(ds.variables["day"], np.asarray(ds.variables["day"].data).item(), "day") != day.timetuple().tm_yday:
            raise ValueError(f"Archive day header mismatch: {path}")
        _unit(ds.variables["altitude"], "altitude", {"km", "kilometers", "kilometres"})
        _unit(ds.variables["flux"], "flux", {"watts/m^2", "w/m^2", "w/m2", "wm-2", "wm^-2"})
        _unit(ds.variables["CO2cool"], "CO2cool", {"watts/m^3", "w/m^3", "w/m3", "wm-3", "wm^-3"})
        alt = np.asarray(ds.variables["altitude"].data, dtype=float)
        if alt.ndim != 1 or not len(alt) or not np.all(np.isfinite(alt)):
            raise ValueError("Invalid native altitudes")
        index = int(np.argmin(np.abs(alt - REFERENCE_KM)))
        if abs(alt[index] - REFERENCE_KM) > ALT_TOLERANCE_KM:
            raise ValueError("No native altitude within 0.5 km of 119 km")
        scans = len(ds.variables["orbit"].data)
        shape = (scans, len(alt))
        scan_dims = ds.variables["orbit"].dimensions
        alt_dims = ds.variables["altitude"].dimensions
        if len(scan_dims) != 1 or len(alt_dims) != 1:
            raise ValueError(f"{path.name}: invalid scan/altitude dimensions")
        for name in ("tplatitude", "tplongitude", "time", "slt", "sza", "CO2cool"):
            if ds.variables[name].dimensions != scan_dims + alt_dims:
                raise ValueError(f"{path.name}: wrong dimension order for {name}")
        for name in ("flux", "event"):
            if ds.variables[name].dimensions != scan_dims:
                raise ValueError(f"{path.name}: wrong scan dimensions for {name}")
        fields = {name: _values(ds, name, shape) for name in ("tplatitude", "tplongitude", "time", "slt", "sza", "CO2cool")}
        flux = _values(ds, "flux", (scans,))
        orbit = _values(ds, "orbit", (scans,))
        event = _values(ds, "event", (scans,))
        if _text(getattr(ds.variables["time"], "units", "")) != "msec" or _text(getattr(ds.variables["slt"], "units", "")) != "msec":
            raise ValueError("time and slt must declare msec units")
        rows = []
        quarantined = []
        quarantined_clocks = []
        for scan in range(scans):
            lat = _number(ds.variables["tplatitude"], fields["tplatitude"][scan, index], low=-90, high=90)
            raw_lon = _number(ds.variables["tplongitude"], fields["tplongitude"][scan, index], low=-360, high=360)
            lon = raw_lon % 360 if raw_lon is not None else None
            geo_selected = _inside(lat, lon)
            try:
                orbit_id = _integer(ds.variables["orbit"], orbit[scan], "orbit")
                event_id = _integer(ds.variables["event"], event[scan], "event")
            except ValueError as error:
                if geo_selected:
                    raise ValueError(f"Invalid regional scan identity: {path.name} scan {scan}") from error
                quarantined.append({"scan_index": scan, "orbit_raw": str(orbit[scan]),
                                    "event_raw": str(event[scan]), "latitude_119km": lat,
                                    "longitude_119km_east": lon,
                                    "time_msec_119km_raw": str(fields["time"][scan, index]),
                                    "reason": str(error)})
                continue
            clock = _number(ds.variables["time"], fields["time"][scan, index], low=0, high=172_800_000 - 1)
            if clock is None:
                # An unplaced scan cannot be assigned an actual UTC month.
                if validation_scope == "regional" and not geo_selected:
                    quarantined_clocks.append({"scan_index": scan, "orbit_raw": str(orbit[scan]),
                                               "event_raw": str(event[scan]), "time_msec_119km_raw": str(fields["time"][scan, index]),
                                               "latitude_119km": lat, "longitude_119km_east": lon,
                                               "reason": "Missing/out-of-range 119-km UTC clock; no UTC month assigned"})
                    continue
                raise ValueError(f"Missing/out-of-range 119-km UTC clock: {path.name} scan {scan}")
            # Preserve fractional milliseconds at explicit microsecond precision.
            timestamp = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc) + timedelta(microseconds=round(clock * 1000))
            slt = _number(ds.variables["slt"], fields["slt"][scan, index], low=0, high=86_400_000 - 1)
            sza = _number(ds.variables["sza"], fields["sza"][scan, index], low=0, high=180)
            official = _number(ds.variables["flux"], flux[scan])
            native = _number(ds.variables["CO2cool"], fields["CO2cool"][scan, index])
            # Full footprints only matter for selected scans. Keeping every
            # global profile would scale memory with millions of excluded scans.
            footprint = tuple(
                (
                    _number(ds.variables["tplatitude"], fields["tplatitude"][scan, j], low=-90, high=90),
                    (value % 360 if value is not None else None),
                )
                for j in range(len(alt))
                for value in [_number(ds.variables["tplongitude"], fields["tplongitude"][scan, j], low=-360, high=360)]
            ) if geo_selected else None
            full = bool(footprint is not None and all(_inside(*point) for point in footprint))
            flags = [name for name, missing in (("missing_geometry", lat is None or lon is None), ("missing_slt", slt is None), ("missing_sza", sza is None), ("invalid_flux", official is None)) if missing]
            rows.append({
                "source_file": path.name, "source_sha256": digest,
                "product_version": version, "archive_day": day,
                "orbit": orbit_id, "event": event_id, "scan_index": scan,
                "timestamp_utc": timestamp, "reference_altitude_km": float(alt[index]),
                "latitude_119km": lat, "longitude_119km_east": lon,
                "geo_selected": geo_selected, "full_profile_in_region": full,
                "official_flux_w_m2": official, "valid_flux": official is not None,
                "slt_hours": slt / 3_600_000 if slt is not None else None,
                "sza_deg": sza, "co2cool_119km_w_m3": native,
                "qc_flags": flags, "duplicate_records": 1, "duplicate_sources": [path.name],
                "_profile_geometry": footprint,
            })
        source = {"filename": path.name, "archive_day": day.isoformat(), "product_version": version,
                  "sha256": digest, "n_scans": scans,
                  "quarantined_nonregional_identity_records": quarantined,
                  "quarantined_nonregional_clock_records": quarantined_clocks,
                  "underlying_saber_version": _text(getattr(ds, "Dataset_SABER_Data_Version", "unreported")),
                  "units": {name: _text(getattr(ds.variables[name], "units", "unreported"))
                            for name in ("altitude", "flux", "CO2cool", "tplatitude", "tplongitude", "time", "slt", "sza")},
                  "coordinate_interpretation": "Named SABER geographic latitude/longitude in degrees per product format; original unit labels retained"}
    if _hash(path) != digest:
        raise ValueError(f"Source changed during read: {path.name}")
    return rows, source


def _monthly(rows: list[dict[str, Any]], start: date, end: date) -> list[dict[str, Any]]:
    buckets: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row["timestamp_utc"].year, row["timestamp_utc"].month)].append(row)
    year, month = start.year, start.month
    result = []
    while (year, month) <= (end.year, end.month):
        selected = buckets[(year, month)]
        good = [r["official_flux_w_m2"] for r in selected if r["valid_flux"]]
        slt_counts = [0] * 6
        for row in selected:
            if row["slt_hours"] is not None:
                slt_counts[int(row["slt_hours"] // 4)] += 1
        result.append({
            "month": date(year, month, 1), "mean_flux_w_m2": float(np.mean(good)) if good else None,
            "median_flux_w_m2": float(np.median(good)) if good else None,
            "n_geo_selected": len(selected), "n_valid_flux": len(good),
            "n_distinct_utc_dates": len({r["timestamp_utc"].date() for r in selected}),
            "n_orbits": len({r["orbit"] for r in selected}),
            "n_day": sum(r["sza_deg"] is not None and r["sza_deg"] < 90 for r in selected),
            "n_night": sum(r["sza_deg"] is not None and r["sza_deg"] >= 90 for r in selected),
            "n_unknown_sza": sum(r["sza_deg"] is None for r in selected),
            "n_unknown_slt": sum(r["slt_hours"] is None for r in selected),
            "n_missing_flux": len(selected) - len(good),
            "n_full_profile_in_region": sum(r["full_profile_in_region"] for r in selected),
            **{f"n_slt_{i * 4:02d}_{(i + 1) * 4:02d}": n for i, n in enumerate(slt_counts)},
        })
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return result


# A real orbit lasts ~97 min. In global validation, a scan claiming an orbit number
# more than this far outside that orbit's observed time span is a reused identity.
ORBIT_IDENTITY_TOLERANCE = timedelta(hours=6)


def prepare_fixed_region(*, input_dir: Path, output_dir: Path, start: date, end: date,
                         validation_scope: str = "global", identity_audit_dir: Path | None = None,
                         expected_identity_audit_sha256: str | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Read local daily archives; publish a new immutable directory, never overwrite."""
    if end < start or not input_dir.is_dir() or not output_dir.parent.is_dir():
        raise ValueError("Invalid dates, input directory or output parent")
    if validation_scope not in ("global", "regional"):
        raise ValueError("Invalid validation_scope")
    if validation_scope == "regional" and identity_audit_dir is None:
        raise ValueError("Regional validation requires --identity-audit-dir")
    if output_dir.exists() or output_dir.resolve().is_relative_to(input_dir.resolve()):
        raise FileExistsError(f"Output must be new and outside inputs: {output_dir}")
    candidates: dict[date, list[tuple[tuple[int, int], Path]]] = defaultdict(list)
    for path in input_dir.glob("SABER_CO2_PROFILE_FLUX_*_V*.nc"):
        match = FILE_RE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unrecognized cooling filename: {path.name}")
        day = _day(match)
        if start - timedelta(days=1) <= day <= end:
            candidates[day].append(((int(match[3]), int(match[4])), path))
    files = [max(versions)[1] for _, versions in sorted(candidates.items())]
    if not files:
        raise ValueError("No matching source files in requested archive window")
    audit = None
    if validation_scope == "regional":
        assert identity_audit_dir is not None
        audit = _identity_audit(identity_audit_dir, files, start, end,
                                expected_identity_audit_sha256 or APPROVED_AUDIT_SHA256)
    rows: list[dict[str, Any]] = []
    provenance_files = []
    seen: dict[tuple[int, int, datetime], dict[str, Any]] = {}
    physical_times: dict[tuple[int, int], dict[str, Any]] = {}
    # Never pruned: compact per-orbit time spans keep distant identity reuse detectable
    # after the two-day row window has dropped the earlier, nonregional scans.
    orbit_spans: dict[int, tuple[datetime, datetime]] = {}
    duplicate_count = 0
    n_missing_geometry = 0
    n_outside_region = 0
    n_outside_utc_window = 0
    for path in files:
        match = FILE_RE.fullmatch(path.name)
        assert match is not None
        day, version = _day(match), f"V{match[3]}.{match[4]}"
        # With nonnegative clocks below 48h, a repeated actual timestamp can
        # occur only in its own or previous archive day. Retain all selected
        # scan identities, but bound excluded-scan bookkeeping to that overlap.
        cutoff = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc)
        seen = {key: row for key, row in seen.items()
                if row["geo_selected"] or row["timestamp_utc"] >= cutoff}
        physical_times = {(row["orbit"], row["event"]): row for row in seen.values()}
        decoded, source = _file_rows(path, day, version, validation_scope=validation_scope)
        if audit is not None and source["sha256"] != audit["source_sha256"][path.name]:
            raise ValueError(f"Identity audit source changed during read: {path.name}")
        provenance_files.append(source)
        for row in decoded:
            in_window = start <= row["timestamp_utc"].date() <= end
            if not in_window:
                n_outside_utc_window += 1
            key = (row["orbit"], row["event"], row["timestamp_utc"])
            physical = (row["orbit"], row["event"])
            prior = physical_times.get(physical)
            if prior is not None and prior["timestamp_utc"] != row["timestamp_utc"] and (audit is None or prior["geo_selected"] or row["geo_selected"]):
                raise ValueError(f"Ambiguous timestamp for physical scan {physical}")
            if prior is None or not prior["geo_selected"]:
                physical_times[physical] = row
            if audit is None:
                stamp = row["timestamp_utc"]
                span = orbit_spans.get(row["orbit"])
                if span is not None and (stamp - span[0] > ORBIT_IDENTITY_TOLERANCE
                                         or span[1] - stamp > ORBIT_IDENTITY_TOLERANCE):
                    raise ValueError(f"Ambiguous timestamp for physical scan {physical}: "
                                     f"orbit {row['orbit']} reappears outside its time span")
                orbit_spans[row["orbit"]] = (min(span[0], stamp), max(span[1], stamp)) if span else (stamp, stamp)
            previous = seen.get(key)
            if previous is not None:
                payload = ("reference_altitude_km", "latitude_119km", "longitude_119km_east", "_profile_geometry", "official_flux_w_m2", "slt_hours", "sza_deg", "co2cool_119km_w_m3", "full_profile_in_region", "qc_flags")
                if any(previous[field] != row[field] for field in payload) and (audit is None or previous["geo_selected"] or row["geo_selected"]):
                    raise ValueError(f"Conflicting duplicate physical scan {key}")
                previous["duplicate_records"] += 1
                previous["duplicate_sources"].append(path.name)
                duplicate_count += 1
            else:
                seen[key] = row
                if not in_window:
                    continue
                if row["geo_selected"]:
                    rows.append(row)
                elif "missing_geometry" in row["qc_flags"]:
                    n_missing_geometry += 1
                else:
                    n_outside_region += 1
    rows.sort(key=lambda r: (r["timestamp_utc"], r["orbit"], r["event"]))
    for row in rows:
        del row["_profile_geometry"]
    months = _monthly(rows, start, end)
    scans_frame = pl.DataFrame(rows, schema=SCAN_SCHEMA)
    monthly_frame = pl.DataFrame(months, schema=MONTH_SCHEMA)
    missing = []
    current = start - timedelta(days=1)
    while current <= end:
        if current not in candidates:
            missing.append(current.isoformat())
        current += timedelta(days=1)
    manifests = [{"name": name, "sha256": _hash(input_dir / name)} for name in ("manifest.json", "remote_inventory.json") if (input_dir / name).is_file()]
    metadata: dict[str, Any] = {
        "region_lat_lon_east": REGION, "reference_altitude_km": REFERENCE_KM,
        "fatal_validation_scope": validation_scope,
        "altitude_tolerance_km": ALT_TOLERANCE_KM, "start_utc": start.isoformat(), "end_utc": end.isoformat(),
        "time_policy": "file day + 119-km msec; UTC inclusive dates, previous archive day padded",
        "fill_policy": "declared fills/ranges, finite values and physical coordinate/clock ranges; negative flux/rates retained",
        "dedup_policy": "highest filename V per archive day; orbit/event/119-km UTC physical key; equal payload counted, conflicts error",
        "product_version_note": "V1.1 documented contact update; not an established radiometric break",
        "source_files": provenance_files, "input_manifests": manifests,
        "ignored_netcdf_files": sorted(p.name for p in input_dir.glob("*.nc") if FILE_RE.fullmatch(p.name) is None),
        "ignored_subdirectories": sorted(p.name for p in input_dir.iterdir() if p.is_dir()),
        "missing_local_archive_days_including_padding": missing,
        "n_equal_duplicate_records": duplicate_count,
        "n_geo_selected": len(rows),
        "n_missing_geometry_in_utc_window": n_missing_geometry,
        "n_outside_region_in_utc_window": n_outside_region,
        "n_outside_utc_window": n_outside_utc_window,
        "n_quarantined_nonregional_identity_records": sum(len(f["quarantined_nonregional_identity_records"]) for f in provenance_files),
        "n_quarantined_nonregional_clock_records": sum(len(f["quarantined_nonregional_clock_records"]) for f in provenance_files),
        "nonregional_outside_utc_window_policy": "Conditional file-origin 119-km clock only; not corrected physical time or an assigned month for unknown clocks",
        "invalid_identity_policy": "Positive integer orbit/event required; invalid reference-region identities fail. Nonregional/unplaceable identities quarantined with raw IDs/geometry, never treated as zero-based or deduplicated.",
        "code_sha256": _hash(Path(__file__)), "python": platform.python_version(),
        "numpy": np.__version__, "scipy": scipy.__version__, "polars": pl.__version__,
        "interpretation": "Descriptive scan-weighted sampling; not a climate or eligibility claim; no gaps filled",
    }
    if audit is not None:
        metadata["identity_audit"] = audit
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        scans_frame.write_parquet(staging / "scans.parquet")
        monthly_frame.write_parquet(staging / "monthly.parquet")
        metadata["outputs"] = {name: {"sha256": _hash(staging / name), "rows": frame.height}
                               for name, frame in (("scans.parquet", scans_frame), ("monthly.parquet", monthly_frame))}
        metadata["complete"] = True
        for source in provenance_files:
            if _hash(input_dir / source["filename"]) != source["sha256"]:
                raise ValueError(f"Source changed before publication: {source['filename']}")
        if audit is not None:
            assert identity_audit_dir is not None
            if (_hash(identity_audit_dir / "report.json") != audit["report_sha256"]
                    or _hash(identity_audit_dir / "provenance.json") != audit["provenance_sha256"]
                    or (identity_audit_dir / "COMPLETE").read_text(encoding="utf-8") != "complete identity audit\n"):
                raise ValueError("Identity audit changed before publication")
        (staging / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        (staging / "COMPLETE").write_text("complete descriptive extraction\n", encoding="utf-8")
        for path in staging.iterdir():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        _publish_new(staging, output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return scans_frame, monthly_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--validation-scope", choices=("global", "regional"), default="global")
    parser.add_argument("--identity-audit-dir", type=Path)
    parser.add_argument("--expected-identity-audit-sha256")
    prepare_fixed_region(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
