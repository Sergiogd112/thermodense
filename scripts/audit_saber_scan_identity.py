"""Read-only, bounded diagnostic of SABER archive orbit/event reuse (not UTC authority)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sqlite3
import sys
import tempfile
import time
from collections import Counter, defaultdict
from contextlib import closing, contextmanager
from datetime import date, datetime, timedelta, timezone
from itertools import groupby
from pathlib import Path

import numpy as np
import scipy
from scipy.io import netcdf_file

from thermodense import saber_fixed_region as fixed

FIELDS = ("time", "tplatitude", "tplongitude", "CO2cool", "slt", "sza", "flux")
VECTORS = FIELDS[:-1]
DISK_LIMIT = 2 * 1024**3
GROUP_LIMIT = 100_000
MEMBER_LIMIT = 100_000
INVALID_LIMIT = 100_000


def _deadline(limit: float | None) -> None:
    if limit is not None and time.monotonic() > limit:
        raise TimeoutError("Audit deadline exceeded; no result published")


def _hash(path: Path, limit: float | None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            _deadline(limit)
            digest.update(chunk)
    return digest.hexdigest()


def _disk_guard(db: Path) -> None:
    size = sum(p.stat().st_size for p in (db, Path(str(db) + "-journal"), Path(str(db) + "-wal"), Path(str(db) + "-shm")) if p.exists())
    if size > DISK_LIMIT:
        raise RuntimeError("SQLite 2 GiB disk guard exceeded; no result published")


@contextmanager
def _source(path: Path, day: date, version: str):
    """Validate production's native schema and headers, without production's early identity gate."""
    ds = netcdf_file(path, "r", mmap=False)
    try:
        required = {"year", "day", "altitude", "orbit", "event", *FIELDS}
        if required - ds.variables.keys():
            raise ValueError(f"{path.name}: missing variables {sorted(required - ds.variables.keys())}")
        if (fixed._text(getattr(ds, "File_Name", "")) != path.name
                or fixed._text(getattr(ds, "Dataset_Version", "")) != version
                or fixed._text(getattr(ds, "Dataset_Name", "")) != "SABER_CO2_PROFILE_FLUX"):
            raise ValueError(f"{path.name}: filename/version/dataset header mismatch")
        for name, value in (("year", day.year), ("day", day.timetuple().tm_yday)):
            data = np.asarray(ds.variables[name].data)
            if data.size != 1 or fixed._integer(ds.variables[name], data.item(), name) != value:
                raise ValueError(f"{path.name}: archive day header mismatch")
        for name, accepted in (("altitude", {"km", "kilometers", "kilometres"}),
                               ("flux", {"watts/m^2", "w/m^2", "w/m2", "wm-2", "wm^-2"}),
                               ("CO2cool", {"watts/m^3", "w/m^3", "w/m3", "wm-3", "wm^-3"})):
            fixed._unit(ds.variables[name], name, accepted)
        for name in ("time", "slt"):
            if fixed._text(getattr(ds.variables[name], "units", "")) != "msec":
                raise ValueError(f"{name} must declare msec units")
        alt = np.asarray(ds.variables["altitude"].data, dtype=float)
        if alt.ndim != 1 or not len(alt) or not np.all(np.isfinite(alt)):
            raise ValueError("Invalid native altitudes")
        ref = int(np.argmin(abs(alt - fixed.REFERENCE_KM)))
        if abs(alt[ref] - fixed.REFERENCE_KM) > fixed.ALT_TOLERANCE_KM:
            raise ValueError("No native altitude within 0.5 km of 119 km")
        scan_dims = ds.variables["orbit"].dimensions
        alt_dims = ds.variables["altitude"].dimensions
        if len(scan_dims) != 1 or len(alt_dims) != 1:
            raise ValueError("Invalid native dimensions")
        scans = len(ds.variables["orbit"].data)
        for name in VECTORS:
            if ds.variables[name].dimensions != scan_dims + alt_dims:
                raise ValueError(f"{name}: wrong dimension order")
            fixed._values(ds, name, (scans, len(alt)))
        for name in ("orbit", "event", "flux"):
            if ds.variables[name].dimensions != scan_dims:
                raise ValueError(f"{name}: wrong scan dimensions")
            fixed._values(ds, name, (scans,))
        yield ds, alt, ref, scans
    finally:
        ds.close()


def _canonical(value):
    """Numeric, endian-independent encoding: canonical NaN and zero; infinities distinct."""
    number = float(value)
    if math.isnan(number):
        return "NaN"
    if math.isinf(number):
        return "+Inf" if number > 0 else "-Inf"
    return 0.0 if number == 0 else number.hex()


def _payload(ds, alt, scan: int) -> tuple[str, str, str, str]:
    metadata = {
        name: {attr: _canonical_list(getattr(ds.variables[name], attr)) if hasattr(ds.variables[name], attr) else None
               for attr in ("units", "_FillValue", "missing_value", "valid_range", "valid_min", "valid_max")}
        for name in ("altitude", *FIELDS)
    }
    values = {name: [_canonical(v) for v in np.asarray(ds.variables[name].data[scan]).flat]
              for name in FIELDS}
    def fingerprint(names, *, include_metadata=True):
        content = {"altitude": [_canonical(v) for v in alt],
                   "values": {n: values[n] for n in names}}
        if include_metadata:
            content["metadata"] = {n: metadata[n] for n in ("altitude", *names)}
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
    metadata_hash = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    return fingerprint(FIELDS), fingerprint(FIELDS[1:]), fingerprint(FIELDS[1:], include_metadata=False), metadata_hash


def _canonical_list(value):
    if isinstance(value, (bytes, str)):
        return fixed._text(value)
    return [_canonical(v) for v in np.asarray(value).flat]


def audit_scan_identities(*, input_dir: Path, output_dir: Path, start: date, end: date,
                          max_seconds: float | None = None, max_files: int | None = None) -> dict:
    """Publish immutable diagnostic; prefix runs are incomplete. Set --max-seconds for a hard operator deadline."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if not (date(2002, 1, 25) <= start <= end <= date(2019, 12, 1)):
        raise ValueError("Dates outside approved inclusive UTC candidate window")
    if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0) or max_files is not None and max_files < 1:
        raise ValueError("Budgets must be positive")
    if not input_dir.is_dir() or not output_dir.parent.is_dir():
        raise ValueError("Input directory or output parent missing")
    if output_dir.exists() or output_dir.resolve().is_relative_to(input_dir.resolve()):
        raise FileExistsError("Output must be new and outside inputs")
    limit = time.monotonic() + max_seconds if max_seconds is not None else None
    candidates = defaultdict(list)
    for path in input_dir.glob("SABER_CO2_PROFILE_FLUX_*_V*.nc"):
        _deadline(limit)
        match = fixed.FILE_RE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unrecognized cooling filename: {path.name}")
        day = fixed._day(match)
        if start - timedelta(days=1) <= day <= end:
            candidates[day].append(((int(match[3]), int(match[4])), path))
    chosen = [max(versions)[1] for _, versions in sorted(candidates.items())]
    if not chosen:
        raise ValueError("No matching source files")
    ignored = sorted(path.name for versions in candidates.values() for _, path in versions if path != max(versions)[1])
    files = chosen[:max_files] if max_files is not None else chosen
    complete = len(files) == len(chosen)
    missing = []
    day = start - timedelta(days=1)
    while day <= end:
        if day not in candidates:
            missing.append(day.isoformat())
        day += timedelta(days=1)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    db = staging / "identity.sqlite"
    try:
        with closing(sqlite3.connect(db)) as conn:
            # Sort/join scratch stays in RAM. Duplicate staging is capped by MEMBER_LIMIT;
            # the scan identity index is on disk. External address-space limits remain essential.
            conn.execute("PRAGMA temp_store=MEMORY")
            if limit is not None:
                conn.set_progress_handler(lambda: int(time.monotonic() > limit), 1000)
            conn.execute("CREATE TABLE scans (file_id INTEGER, scan_index INTEGER, orbit INTEGER, event INTEGER, day TEXT, raw_clock TEXT, utc TEXT, lat REAL, lon REAL, geo INTEGER, window INTEGER, geometry_valid INTEGER, clock_identity TEXT)")
            conn.execute("CREATE INDEX identity ON scans (orbit, event)")
            sources = []
            counters = Counter({"archive_records": 0, "positive_id_archive_records": 0,
                                "unique_positive_physical_ids": 0, "duplicate_groups": 0,
                                "duplicate_archive_records": 0, "invalid_id_regional_records": 0,
                                "invalid_clock_regional_records": 0})
            years = defaultdict(Counter)
            invalid = []
            for file_id, path in enumerate(files):
                _deadline(limit)
                match = fixed.FILE_RE.fullmatch(path.name)
                assert match is not None
                archive_day = fixed._day(match)
                digest = _hash(path, limit)
                with _source(path, archive_day, f"V{match[3]}.{match[4]}") as (ds, alt, ref, scans):
                    source_metadata = {
                        "underlying_saber_version": fixed._text(getattr(ds, "Dataset_SABER_Data_Version", "unreported")),
                        "units": {name: fixed._text(getattr(ds.variables[name], "units", "unreported"))
                                  for name in ("altitude", *FIELDS)},
                        "reference_altitude_km": float(alt[ref]),
                    }
                    for scan in range(scans):
                        if scan % 1024 == 0:
                            _deadline(limit)
                            _disk_guard(db)
                        def number(name, *, low=None, high=None):
                            return fixed._number(ds.variables[name], ds.variables[name].data[scan, ref], low=low, high=high)
                        lat = number("tplatitude", low=-90, high=90)
                        raw_lon = number("tplongitude", low=-360, high=360)
                        lon = raw_lon % 360 if raw_lon is not None else None
                        geo = fixed._inside(lat, lon)
                        clock_raw = ds.variables["time"].data[scan, ref]
                        clock = fixed._number(ds.variables["time"], clock_raw, low=0, high=172_800_000 - 1)
                        utc = (datetime.combine(archive_day, datetime.min.time(), tzinfo=timezone.utc)
                               + timedelta(microseconds=round(clock * 1000))).isoformat(timespec="microseconds") if clock is not None else None
                        window = start.isoformat() <= utc[:10] <= end.isoformat() if utc is not None else None
                        flags = []
                        if lat is None or lon is None:
                            flags.append("invalid_geometry")
                        if clock is None:
                            flags.append("invalid_clock")
                        ids = []
                        for name in ("orbit", "event"):
                            try:
                                ids.append(fixed._integer(ds.variables[name], ds.variables[name].data[scan], name))
                            except ValueError:
                                flags.append("invalid_id")
                                ids.append(None)
                        counters["archive_records"] += 1
                        years[archive_day.year]["archive_records"] += 1
                        for label, selected in (("geo_selected_records", geo), ("in_utc_window_records", window),
                                                ("regional_in_utc_window_records", geo and window)):
                            if selected:
                                counters[label] += 1
                                years[archive_day.year][label] += 1
                        for flag in set(flags):
                            counters[flag + "_records"] += 1
                            years[archive_day.year][flag + "_records"] += 1
                            if geo:
                                counters[flag + "_regional_records"] += 1
                                years[archive_day.year][flag + "_regional_records"] += 1
                        if flags:
                            if len(invalid) >= INVALID_LIMIT:
                                raise RuntimeError("100000 invalid-record memory guard exceeded; no result published")
                            invalid.append({"file": path.name, "archive_day": archive_day.isoformat(), "scan_index": scan,
                                            "orbit_raw": str(ds.variables["orbit"].data[scan]),
                                            "event_raw": str(ds.variables["event"].data[scan]), "raw_clock": str(clock_raw),
                                            "conditional_utc": utc, "latitude_119km": lat, "longitude_119km_east": lon,
                                            "geo_selected": geo, "in_utc_window": window, "flags": sorted(set(flags))})
                        if None not in ids:
                            conn.execute("INSERT INTO scans VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                          (file_id, scan, *ids, archive_day.isoformat(), str(clock_raw), utc, lat, lon,
                                           int(geo), int(window) if window is not None else None,
                                           int(lat is not None and lon is not None), json.dumps(_canonical(clock_raw))))
                            counters["positive_id_archive_records"] += 1
                            years[archive_day.year]["positive_id_archive_records"] += 1
                if _hash(path, limit) != digest:
                    raise ValueError(f"Source changed during read: {path.name}")
                sources.append({"filename": path.name, "archive_day": archive_day.isoformat(), "product_version": f"V{match[3]}.{match[4]}", "sha256": digest, "n_scans": scans, **source_metadata})
                conn.commit()
                _disk_guard(db)
                if (file_id + 1) % 100 == 0:
                    print(f"{file_id + 1}/{len(files)} files", file=sys.stderr)
            conn.execute("CREATE TEMP TABLE dup (orbit INTEGER, event INTEGER)")
            # Stream the covering index: no second 9M-key table or unbounded GROUP BY sort.
            for orbit, event, count in conn.execute("SELECT orbit,event,COUNT(*) FROM scans GROUP BY orbit,event"):
                counters["unique_positive_physical_ids"] += 1
                if count > 1:
                    if counters["duplicate_groups"] >= GROUP_LIMIT:
                        raise RuntimeError("100000 duplicate-group guard exceeded; no result published")
                    counters["duplicate_groups"] += 1
                    conn.execute("INSERT INTO dup VALUES (?,?)", (orbit, event))
                if counters["unique_positive_physical_ids"] % 100_000 == 0:
                    _deadline(limit)
                    _disk_guard(db)
            conn.execute("CREATE INDEX dup_identity ON dup (orbit,event)")
            conn.execute("CREATE TEMP TABLE duplicate_members AS SELECT * FROM scans WHERE 0")
            # One indexed join over all records, never a scan of the full table per source file.
            for member in conn.execute("SELECT scans.* FROM scans JOIN dup USING (orbit,event)"):
                if counters["duplicate_archive_records"] >= MEMBER_LIMIT:
                    raise RuntimeError("100000 duplicate-member memory guard exceeded; no result published")
                conn.execute("INSERT INTO duplicate_members VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", member)
                counters["duplicate_archive_records"] += 1
            conn.execute("CREATE INDEX duplicate_file_scan ON duplicate_members (file_id,scan_index)")
            conn.execute("CREATE INDEX duplicate_identity ON duplicate_members (orbit,event)")
            conn.execute("CREATE TEMP TABLE payload (file_id INTEGER, scan_index INTEGER, full TEXT, measurements TEXT, numeric_measurements TEXT, metadata TEXT)")
            conn.execute("CREATE INDEX payload_file_scan ON payload (file_id,scan_index)")
            for file_id, path in enumerate(files):
                members = [row[0] for row in conn.execute("SELECT scan_index FROM duplicate_members WHERE file_id=?", (file_id,))]
                if not members:
                    continue
                _deadline(limit)
                if _hash(path, limit) != sources[file_id]["sha256"]:
                    raise ValueError(f"Source changed before duplicate read: {path.name}")
                match = fixed.FILE_RE.fullmatch(path.name)
                assert match is not None
                with _source(path, fixed._day(match), f"V{match[3]}.{match[4]}") as (ds, alt, _, _):
                    for scan in members:
                        _deadline(limit)
                        full, measurements, numeric, metadata = _payload(ds, alt, scan)
                        conn.execute("INSERT INTO payload VALUES (?,?,?,?,?,?)", (file_id, scan, full, measurements, numeric, metadata))
                if _hash(path, limit) != sources[file_id]["sha256"]:
                    raise ValueError(f"Source changed during duplicate read: {path.name}")
                conn.commit()
                _disk_guard(db)
            groups = []
            regional_conflicts = []
            in_window_issues = []
            regional_unknown_clocks = []
            query = "SELECT d.*,p.full,p.measurements,p.numeric_measurements,p.metadata FROM duplicate_members d JOIN payload p USING (file_id,scan_index) ORDER BY d.orbit,d.event,d.day,d.file_id,d.scan_index"
            for (orbit, event), rows in groupby(conn.execute(query), lambda r: (r[2], r[3])):
                _deadline(limit)
                members = list(rows)
                fulls = {r[13] for r in members}
                measurements = {r[14] for r in members}
                numeric = {r[15] for r in members}
                metadata = {r[16] for r in members}
                utcs = {r[6] for r in members}
                clocks = {r[12] for r in members}
                has_invalid_clock = None in utcs
                interpreted_utc_equal = None if has_invalid_clock else len(utcs) == 1
                if has_invalid_clock:
                    kind = "invalid_clock_present"
                elif len(fulls) == 1:
                    kind = "identical_raw_payload_same_utc" if interpreted_utc_equal else "identical_raw_payload_different_utc"
                elif len(numeric) > 1:
                    kind = "different_measurement_payload"
                elif len(metadata) > 1:
                    kind = "different_metadata"
                elif len(clocks) > 1 and interpreted_utc_equal and len(measurements) == 1:
                    kind = "rebased_clock_same_utc_identical_measurements"
                elif len(clocks) > 1 and len(measurements) == 1:
                    kind = "valid_different_clocks_same_measurements"
                else:
                    kind = "mixed_other"
                days = [date.fromisoformat(r[4]) for r in members]
                geo = {bool(r[9]) for r in members}
                window = {r[10] for r in members}
                deltas = [(datetime.fromisoformat(r[6]) - datetime.fromisoformat(members[0][6]))
                          if r[6] and members[0][6] else None for r in members]
                group = {"orbit": orbit, "event": event, "classification": kind, "archive_record_count": len(members),
                         "has_invalid_clock": has_invalid_clock, "measurement_payload_equal": len(measurements) == 1,
                         "numeric_measurements_equal": len(numeric) == 1, "metadata_equal": len(metadata) == 1,
                         "full_raw_payload_equal": len(fulls) == 1, "interpreted_utc_equal": interpreted_utc_equal,
                         "origin_day_gap_days": (max(days) - min(days)).days,
                         "occurrence": "same_file" if len({r[0] for r in members}) == 1 else ("adjacent" if (max(days) - min(days)).days <= 1 else "distant"),
                         "utc_deltas_microseconds_from_first": [(d.days * 86_400 + d.seconds) * 1_000_000 + d.microseconds if d is not None else None for d in deltas],
                         "utc_deltas_seconds_from_first": [d.total_seconds() if d is not None else None for d in deltas],
                          "geo_any": any(geo), "geo_all": geo == {True}, "regional_in_utc_window_any": any(r[9] and r[10] for r in members),
                         "geo_selection_changes": len(geo) > 1, "utc_window_membership_changes": len(window) > 1,
                         "members": [{"file": files[r[0]].name, "scan_index": r[1], "archive_day": r[4], "raw_clock": r[5],
                                      "conditional_utc": r[6], "latitude_119km": r[7], "longitude_119km_east": r[8],
                                       "geo_selected": bool(r[9]), "in_utc_window": bool(r[10]) if r[10] is not None else None,
                                       "geometry_valid": bool(r[11])} for r in members]}
                groups.append(group)
                counters[kind + "_groups"] += 1
                for year in {d.year for d in days}:
                    years[year]["duplicate_groups_touching_year"] += 1
                    years[year][kind + "_groups_touching_year"] += 1
                for d in days:
                    years[d.year]["duplicate_archive_records"] += 1
                if group["geo_any"] and kind != "identical_raw_payload_same_utc":
                    regional_conflicts.append(group)
                if any(r[9] and r[6] is None for r in members):
                    regional_unknown_clocks.append(group)
                if group["regional_in_utc_window_any"] and kind not in (
                    "identical_raw_payload_same_utc", "rebased_clock_same_utc_identical_measurements"
                ):
                    in_window_issues.append(group)
            counters["invalid_distinct_records"] = len(invalid)
            counters["broad_regional_diagnostic_groups"] = len(regional_conflicts)
            counters["in_window_regional_issue_groups"] = len(in_window_issues)
            counters["regional_unknown_clock_groups"] = len(regional_unknown_clocks)
            counters["regional_absence_check_passes"] = int(
                complete and not in_window_issues and not regional_unknown_clocks
                and not counters["invalid_id_regional_records"] and not counters["invalid_clock_regional_records"]
            )
            _deadline(limit)
            _disk_guard(db)
        db.unlink()
        report = {"complete": complete, "start_utc": start.isoformat(), "end_utc": end.isoformat(),
                  "scope": "Positive integer orbit/event grouping only; conditional file-origin UTC is not an authoritative physical timestamp. Reference geo is independent of UTC-window selection; origin adjacency is not UTC adjacency.",
                  "interpretation": "Broad regional diagnostic groups are not production failures. An absence claim requires a complete run, zero in-window regional issue groups, zero regional unknown-clock groups and zero invalid-ID/clock regional records. Production fails closed on differing timestamps for a repeated physical ID.",
                  "counts": dict(counters), "per_archive_year": {str(y): dict(c) for y, c in sorted(years.items())},
                  "invalid_records": invalid, "duplicate_groups": groups, "conflicting_regional_groups": regional_conflicts,
                  "broad_regional_diagnostic_groups": regional_conflicts,
                  "in_window_regional_issue_groups": in_window_issues,
                  "regional_unknown_clock_groups": regional_unknown_clocks,
                  "source_files": sources, "ignored_versions": ignored, "missing_local_archive_days_including_padding": missing,
                  "files_processed": len(files), "files_in_window": len(chosen)}
        provenance = {"complete": complete, "source_sha256": {s["filename"]: s["sha256"] for s in sources},
                      "audit_code_sha256": _hash(Path(__file__), limit), "production_code_sha256": _hash(Path(fixed.__file__), limit),
                      "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                      "limits": {"sqlite_disk_bytes_including_sidefiles": DISK_LIMIT, "duplicate_groups": GROUP_LIMIT,
                                 "duplicate_members_in_memory": MEMBER_LIMIT, "invalid_records_in_memory": INVALID_LIMIT,
                                 "max_seconds": max_seconds, "max_files": max_files,
                                 "ram_note": "SQLite TEMP scratch and bounded duplicate/invalid details reside in RAM; use an external address-space limit."}}
        for name, obj in (("report.json", report), ("provenance.json", provenance)):
            _deadline(limit)
            (staging / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for source in sources:
            if _hash(input_dir / source["filename"], limit) != source["sha256"]:
                raise ValueError(f"Source changed before publication: {source['filename']}")
        provenance["report_sha256"] = _hash(staging / "report.json", limit)
        (staging / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        if complete:
            (staging / "COMPLETE").write_text("complete identity audit\n", encoding="utf-8")
        for path in staging.iterdir():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        _deadline(limit)
        fixed._publish_new(staging, output_dir)
        return report
    except sqlite3.OperationalError as error:
        if limit is not None and time.monotonic() > limit:
            raise TimeoutError("Audit deadline exceeded during SQLite query; no result published") from error
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--max-seconds", type=float, help="Optional hard elapsed-time budget; set for full runs (also use an external RAM limit)")
    parser.add_argument("--max-files", type=int, help="Read only the first N selected archive days; result marked incomplete")
    audit_scan_identities(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
