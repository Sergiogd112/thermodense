"""Assemble the canonical raw 36-node 3-hour PCMCI+ input without downloads."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from thermodense.benchmarks import raw_3hour_pcmci as contract

LEGACY_PREPARED_FILENAME = "prepared_input.npz"
LEGACY_CHANNELS = contract.LEGACY_SOURCE_NODE_ORDER
TRANSFORMATIONS = {
    "detrending": "none",
    "deseasonalizing": "none",
    "standardizing": "none",
    "interpolation": "none",
}
LEGACY_PROVENANCE_FILENAME = "prepared_input.provenance.json"
LEGACY_PROVENANCE_VERSION = contract.LEGACY_PROVENANCE_VERSION
LEGACY_PROVENANCE_SCHEMA = contract.LEGACY_PROVENANCE_SCHEMA
LEGACY_F107_POLICY = contract.LEGACY_F107_POLICY
LEGACY_CO2_MISSING_POLICY = contract.LEGACY_CO2_MISSING_POLICY
F107_POLICY = contract.F107_SOURCE_POLICY


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)  # pyright: ignore[reportArgumentType]
    temporary.replace(path)


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _legacy_input(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read only the retained raw legacy channels, rejecting its spline driver."""
    if path.name != LEGACY_PREPARED_FILENAME:
        raise ValueError(f"legacy prepared NPZ must be explicitly named {LEGACY_PREPARED_FILENAME}")
    with np.load(path, allow_pickle=False) as saved:
        required = {"time_ns", "values", "channel_names"}
        if required - set(saved.files):
            raise ValueError("legacy prepared NPZ lacks time_ns, values, or channel_names")
        timestamps = np.asarray(saved["time_ns"], dtype=np.int64).astype("datetime64[ns]")
        values = np.asarray(saved["values"], dtype=float)
        names = tuple(str(value) for value in saved["channel_names"].tolist())
    if names != LEGACY_CHANNELS:
        raise ValueError("legacy prepared NPZ node order is not the expected 36-channel order")
    if values.shape != (len(timestamps), len(LEGACY_CHANNELS)):
        raise ValueError("legacy prepared NPZ values must have shape T x 36")
    _validate_calendar(timestamps)
    if np.any(np.isfinite(values[:, 30]) & (values[:, 30] < 0)):
        raise ValueError("legacy MLO CO2 must encode negative sentinels as missing")
    # Column 27 is explicitly validated through LEGACY_CHANNELS but intentionally
    # never returned: its historical representation is spline-interpolated F10.7.
    return timestamps, values[:, :27], values[:, 28], values[:, 29], values[:, 30]


def _legacy_provenance(
    path: Path,
    digest: str,
    timestamps: np.ndarray,
    *,
    required: bool,
    provenance_digest: str | None = None,
) -> dict[str, Any]:
    provenance_path = path.with_name(LEGACY_PROVENANCE_FILENAME)
    if not provenance_path.is_file():
        if required:
            raise ValueError(f"canonical assembly requires adjacent {LEGACY_PROVENANCE_FILENAME}")
        return {"status": "fixture_unverified", "schema": LEGACY_PROVENANCE_SCHEMA}
    try:
        record = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("legacy prepared provenance is invalid") from error
    if not isinstance(record, dict):
        raise ValueError("legacy prepared provenance must be a JSON object")
    identity = record.get("identity")
    valid_identity = (
        isinstance(identity, dict)
        and identity.get("version") == LEGACY_PROVENANCE_VERSION
        and identity.get("channels") == list(LEGACY_CHANNELS)
    )
    try:
        upstream_hashes = contract.validate_legacy_upstream_hashes(
            identity.get("source_sha256") if isinstance(identity, dict) else None
        )
    except ValueError:
        upstream_hashes = None
    if not (
        valid_identity
        and upstream_hashes is not None
        and record.get("prepared_sha256") == digest
        and record.get("shape") == [len(timestamps), len(LEGACY_CHANNELS)]
        and record.get("cadence") == "3-hour"
        and record.get("f107_policy") == LEGACY_F107_POLICY
        and record.get("co2_policy") == LEGACY_CO2_MISSING_POLICY
    ):
        raise ValueError("legacy prepared provenance does not bind the required raw semantics")
    return {
        "prepared_artifact_status": "verified",
        "upstream_semantics_status": "declared_by_bound_legacy_provenance",
        "schema": LEGACY_PROVENANCE_SCHEMA,
        "source_version": LEGACY_PROVENANCE_VERSION,
        "prepared_input_sha256": digest,
        "provenance_sha256": provenance_digest or _sha256(provenance_path),
        "channel_names": list(LEGACY_CHANNELS),
        "shape": [len(timestamps), len(LEGACY_CHANNELS)],
        "calendar": {"cadence": contract.CALENDAR_CADENCE},
        "f107_policy": LEGACY_F107_POLICY,
        "co2_missing_policy": LEGACY_CO2_MISSING_POLICY,
        "upstream_source_sha256": upstream_hashes,
    }


def _validate_calendar(timestamps: np.ndarray) -> None:
    if (
        timestamps.ndim != 1
        or not len(timestamps)
        or np.isnat(timestamps).any()
        or not np.all(np.diff(timestamps.astype("int64")) == 3 * 3600 * 10**9)
    ):
        raise ValueError("timestamps must be a nonempty consecutive UTC 3-hour calendar")
    day_ns = 24 * 3600 * 10**9
    if timestamps[0].astype("int64") % day_ns:
        raise ValueError("UTC 3-hour calendar must start at midnight for daily holds")


def _f107_daily_hold(path: Path, timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="") as handle:
        rows = csv.DictReader(handle)
        if rows.fieldnames is None or not {"DATE", "F10.7_OBS", "F10.7_DATA_TYPE"}.issubset(rows.fieldnames):
            raise ValueError("CelesTrak compatibility CSV requires DATE, F10.7_OBS, and F10.7_DATA_TYPE columns")
        daily: dict[np.datetime64, float] = {}
        for row in rows:
            try:
                date = np.datetime64(str(row["DATE"]), "D")
            except ValueError as error:
                raise ValueError(f"invalid CelesTrak DATE {row['DATE']!r}") from error
            if row["F10.7_DATA_TYPE"] != "OBS":
                continue
            if date in daily:
                raise ValueError(f"duplicate CelesTrak DATE {date}")
            value = row["F10.7_OBS"]
            daily[date] = float(value) if value not in (None, "") else np.nan
    dates = timestamps.astype("datetime64[D]")
    values = np.array([daily.get(date, np.nan) for date in dates], dtype=float)
    masks = np.isfinite(values)
    contract._daily_held(values, masks, "F10.7")
    return values, masks


def _saber_input(path: Path, timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    with np.load(path, allow_pickle=False) as saved:
        required = {"timestamps", "channels", "values", "masks"}
        if required - set(saved.files):
            raise ValueError("SABER NPZ lacks timestamps, channels, values, or masks")
        source_timestamps = np.asarray(saved["timestamps"]).astype("datetime64[ns]")
        channels = tuple(str(value) for value in saved["channels"].tolist())
        values = np.asarray(saved["values"], dtype=float)
        masks = np.asarray(saved["masks"], dtype=bool)
    if not np.array_equal(source_timestamps, timestamps):
        raise ValueError("SABER timestamps must exactly match legacy UTC timestamps")
    if channels != contract.SABER_CHANNELS:
        raise ValueError("SABER channels must be the exact five-channel 50x70 order")
    if values.shape != (len(contract.SABER_CHANNELS), len(timestamps)) or masks.shape != values.shape:
        raise ValueError("SABER values and masks must have shape 5 x T")
    return source_timestamps, values, masks, channels


def assemble(legacy_prepared_npz: Path, space_weather_csv: Path, saber_npz: Path, output: Path, *, input_kind: str = "canonical") -> Path:
    """Write a new canonical manifest/core pair from three supplied local sources."""
    if input_kind not in {"canonical", "fixture"}:
        raise ValueError("input_kind must be canonical or fixture")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output {output}")
    legacy_prepared_npz, space_weather_csv, saber_npz = (item.resolve(strict=True) for item in (legacy_prepared_npz, space_weather_csv, saber_npz))
    legacy_provenance_path = legacy_prepared_npz.with_name(LEGACY_PROVENANCE_FILENAME)
    source_paths = [legacy_prepared_npz, space_weather_csv, saber_npz]
    if legacy_provenance_path.is_file():
        source_paths.append(legacy_provenance_path)
    source_hashes = {item: _sha256(item) for item in source_paths}
    timestamps, density, ap, kp, co2 = _legacy_input(legacy_prepared_npz)
    f107, f107_masks = _f107_daily_hold(space_weather_csv, timestamps)
    saber_timestamps, saber_values, saber_masks, saber_channels = _saber_input(saber_npz, timestamps)
    density_masks = np.isfinite(density)
    ap_masks, kp_masks, co2_masks = np.isfinite(ap), np.isfinite(kp), np.isfinite(co2)
    contract._daily_held(co2, co2_masks, "MLO CO2")
    calendar = contract.calendar_declaration(timestamps)
    if input_kind == "canonical" and calendar != contract.CANONICAL_CALENDAR:
        raise ValueError("canonical input must retain the exact 2000-01-01 through 2025-07-20 UTC calendar")
    legacy_semantics = _legacy_provenance(
        legacy_prepared_npz,
        source_hashes[legacy_prepared_npz],
        timestamps,
        required=input_kind == "canonical",
        provenance_digest=source_hashes.get(legacy_provenance_path),
    )
    if any(_sha256(item) != digest for item, digest in source_hashes.items()):
        raise ValueError("a source changed while the immutable raw input was assembled")

    output.mkdir(parents=True)
    core_path = output / "core.npz"
    saber_path = output / "saber.npz"
    manifest_path = output / "manifest.json"
    _atomic_npz(core_path, timestamps=timestamps, density_altitudes_km=np.asarray(contract.ALTITUDES_KM), density_log10_values=density, density_masks=density_masks, f107_values=f107, f107_masks=f107_masks, ap_values=ap, ap_masks=ap_masks, kp_values=kp, kp_masks=kp_masks, mlo_co2_values=co2, mlo_co2_masks=co2_masks)
    _atomic_npz(saber_path, timestamps=saber_timestamps, channels=np.asarray(saber_channels), values=saber_values, masks=saber_masks)
    sources = {
        "legacy_prepared_npz": {"path": str(legacy_prepared_npz), "sha256": source_hashes[legacy_prepared_npz], "provenance_path": str(legacy_provenance_path) if legacy_provenance_path in source_hashes else None, "provenance_sha256": source_hashes.get(legacy_provenance_path), "used_channels": [*contract.density_names(), "Ap", "Kp", "co2_mlo_ppm"], "discarded_channel": "F10.7_raw (legacy spline representation)"},
        "celestrak_sw_all_csv": {"path": str(space_weather_csv), "sha256": source_hashes[space_weather_csv], "used_column": "F10.7_OBS", "data_type_policy": F107_POLICY},
        "saber_50x70_npz": {"path": str(saber_npz), "sha256": source_hashes[saber_npz], "channels": list(contract.SABER_CHANNELS)},
    }
    manifest = {"input_contract": contract.VERSION, "input_kind": input_kind, "calendar": calendar, "core_npz": core_path.name, "core_sha256": _sha256(core_path), "saber_npz": saber_path.name, "saber_sha256": _sha256(saber_path), "f10_7": {"representation": "raw_daily_held", "source": "CelesTrak SW-All compatibility CSV F10.7_OBS", "data_type": "OBS only", "non_obs": "unavailable", "interpolation": "none"}, "transformations": TRANSFORMATIONS, "source_node_order": list(LEGACY_CHANNELS), "legacy_prepared_semantics": legacy_semantics, "data_mobility": contract.DATA_MOBILITY_POLICY, "sources": sources}
    _atomic_json(manifest_path, manifest)
    _atomic_json(output / "provenance.json", {"assembly": "canonical raw 36-node native-3hour input" if input_kind == "canonical" else "fixture raw 36-node native-3hour input", "calendar": calendar, "raw_no_transform_declaration": TRANSFORMATIONS, "daily_hold": {"f10_7": F107_POLICY, "mlo_co2": "validated legacy UTC daily hold"}, "legacy_prepared_semantics": legacy_semantics, "source_hashes": sources, "outputs": {"core.npz": _sha256(core_path), "saber.npz": _sha256(saber_path), "manifest.json": _sha256(manifest_path)}})
    return manifest_path


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--legacy-prepared-npz", type=Path, required=True)
    command.add_argument("--space-weather-csv", type=Path, required=True)
    command.add_argument("--saber-50x70-npz", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--fixture", action="store_true", help="allow a short test calendar")
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(assemble(args.legacy_prepared_npz, args.space_weather_csv, args.saber_50x70_npz, args.output, input_kind="fixture" if args.fixture else "canonical"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
