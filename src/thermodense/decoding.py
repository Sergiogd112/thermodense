import gzip
import io
import json
import logging
import os
import re
import traceback
import zipfile
from pathlib import Path
from pprint import pprint
from typing import List, Optional, Tuple

import polars as pl
from filelock import FileLock

OUTPUT_COLUMNS = [
    "source_id",
    "source_name",
    "mission",
    "dataset_family",
    "timestamp",
    "time_system",
    "density_kg_m3",
    "altitude_km",
    "latitude_deg",
    "longitude_deg",
    "local_solar_time_hours",
    "argument_of_latitude_deg",
    "quality_flag",
    "quality_detail",
    "native_file",
    "native_product",
    "native_record_index",
]
OUTPUT_COLUMNS = [
    "source_id",
    "source_name",
    "mission",
    "dataset_family",
    "timestamp",
    "time_system",
    "density_kg_m3",
    "altitude_km",
    "latitude_deg",
    "longitude_deg",
    "local_solar_time_hours",
    "argument_of_latitude_deg",
    "quality_flag",
    "quality_detail",
    "native_file",
    "native_product",
    "native_record_index",
]
TUDELFT_SCHEMAS = {
    "CHAMP": {
        "Date yyyy-mm-dd": pl.Date,
        "Time hh:mm:ss.sss": pl.Time,
        "Time System": pl.Utf8,
        "Altitude (m)": pl.Float64,
        "Longitude (deg)": pl.Float64,
        "Latitude (deg)": pl.Float64,
        "Local Solar Time (hours)": pl.Float64,
        "Argument of Latitude (deg)": pl.Float64,
        "Density (kg/m^3)": pl.Float64,
        "Density Mean (kg/m^3)": pl.Float64,
        "Anomalus Density (kg/m^3)": pl.Float64,
        "Anomalus Density Mean (kg/m^3)": pl.Float64,
    },
    "GRACE": {
        "Date yyyy-mm-dd": pl.Date,
        "Time hh:mm:ss.sss": pl.Time,
        "Time System": pl.Utf8,
        "Altitude (m)": pl.Float64,
        "Longitude (deg)": pl.Float64,
        "Latitude (deg)": pl.Float64,
        "Local Solar Time (hours)": pl.Float64,
        "Argument of Latitude (deg)": pl.Float64,
        "Density (kg/m^3)": pl.Float64,
        "Density Mean (kg/m^3)": pl.Float64,
        "Anomalus Density (kg/m^3)": pl.Float64,
        "Anomalus Density Mean (kg/m^3)": pl.Float64,
    },
    "GRACE_FO": {
        "Date yyyy-mm-dd": pl.Date,
        "Time hh:mm:ss.sss": pl.Time,
        "Time System": pl.Utf8,
        "Altitude (m)": pl.Float64,
        "Longitude (deg)": pl.Float64,
        "Latitude (deg)": pl.Float64,
        "Local Solar Time (hours)": pl.Float64,
        "Argument of Latitude (deg)": pl.Float64,
        "Density (kg/m^3)": pl.Float64,
        "Density Mean (kg/m^3)": pl.Float64,
        "Anomalus Density (kg/m^3)": pl.Float64,
        "Anomalus Density Mean (kg/m^3)": pl.Float64,
    },
    "SWARM": {
        "Date yyyy-mm-dd": pl.Date,
        "Time hh:mm:ss.sss": pl.Time,
        "Time System": pl.Utf8,
        "Altitude (m)": pl.Float64,
        "Longitude (deg)": pl.Float64,
        "Latitude (deg)": pl.Float64,
        "Local Solar Time (hours)": pl.Float64,
        "Argument of Latitude (deg)": pl.Float64,
        "Density (kg/m^3)": pl.Float64,
        "Density Mean (kg/m^3)": pl.Float64,
        "Anomalus Density (kg/m^3)": pl.Float64,
        "Anomalus Density Mean (kg/m^3)": pl.Float64,
    },
    "GOCE": {
        "Date yyyy-mm-dd": pl.Date,
        "Time hh:mm:ss.sss": pl.Time,
        "Time System": pl.Utf8,
        "Altitude (m)": pl.Float64,
        "Longitude (deg)": pl.Float64,
        "Latitude (deg)": pl.Float64,
        "Local Solar Time (hours)": pl.Float64,
        "Argument of Latitude (deg)": pl.Float64,
        "Density (kg/m^3)": pl.Float64,
        "Density Mean (kg/m^3)": pl.Float64,
        "Anomalus Density (kg/m^3)": pl.Float64,
        "Anomalus Density Mean (kg/m^3)": pl.Float64,
        "Degraded Flag Thrusters": pl.Float64,
    },
}
HASDM_SCHEMA = {
    "YYYYMMDDhhmm": pl.Utf8,
    "JulianDay": pl.Float64,
    "HTM": pl.Float64,
    "LAT": pl.Float64,
    "LON": pl.Float64,
    "LST": pl.Float64,
    "RHO": pl.Float64,
}
GLOBAL_DENSITY_COLUMN_RE = re.compile(r"(\d+(?:\.\d+)?)")
GLOBAL_DENSITY_COMMENT_PREFIXES = ("#", ":", "%", "!")


def normalize_whitespace(
    raw: bytes,
    *,
    comment_prefixes: tuple[bytes, ...] = (b"#",),
) -> bytes:
    return b"".join(
        b";".join(line.split()) + b"\n"
        for line in raw.splitlines()
        if line.strip() and not line.startswith(comment_prefixes)
    )


def _read_first_txt_from_zip(sourcepath: str) -> tuple[str, bytes] | None:
    with zipfile.ZipFile(sourcepath, "r") as zip_ref:
        txt_name = next(
            (name for name in zip_ref.namelist() if name.lower().endswith(".txt")),
            None,
        )
        if txt_name is None:
            logging.warning("No .txt file found in %s", sourcepath)
            return None
        with zip_ref.open(txt_name, "r") as f:
            return txt_name, f.read()


def _read_first_matching_from_zip(
    sourcepath: str,
    suffixes: tuple[str, ...],
) -> tuple[str, bytes] | None:
    wanted = tuple(s.lower() for s in suffixes)
    with zipfile.ZipFile(sourcepath, "r") as zip_ref:
        name = next(
            (
                item
                for item in zip_ref.namelist()
                if item.lower().endswith(wanted) and not item.endswith("/")
            ),
            None,
        )
        if name is None:
            logging.warning("No matching text file found in %s", sourcepath)
            return None
        with zip_ref.open(name, "r") as f:
            return name, f.read()


def _read_text_payload(sourcepath: str) -> tuple[str, bytes] | None:
    path = Path(sourcepath)
    if path.suffix.lower() == ".zip":
        return _read_first_matching_from_zip(
            sourcepath,
            (".txt", ".csv", ".dat", ".tsv"),
        )
    if not path.exists():
        logging.warning("Source file does not exist: %s", sourcepath)
        return None
    return path.name, path.read_bytes()


def _iter_data_lines(raw: bytes) -> list[str]:
    lines = raw.decode("utf-8", errors="replace").splitlines()
    return [
        line.strip()
        for line in lines
        if line.strip()
        and not line.lstrip().startswith(GLOBAL_DENSITY_COMMENT_PREFIXES)
    ]


def _detect_separator(lines: list[str]) -> Optional[str]:
    if not lines:
        return None
    sample = lines[min(2, len(lines) - 1)]
    for sep in (",", ";", "\t"):
        if sample.count(sep) >= 1:
            return sep
    return None


def _parse_datetime_column(expr: pl.Expr) -> pl.Expr:
    cast_expr = expr.cast(pl.Utf8, strict=False).str.strip_chars()
    return pl.coalesce(
        cast_expr.str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S", strict=False),
        cast_expr.str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False),
        cast_expr.str.strptime(pl.Datetime, format="%Y/%m/%d %H:%M:%S", strict=False),
        cast_expr.str.strptime(pl.Datetime, format="%Y-%m-%d", strict=False),
        cast_expr.str.strptime(pl.Datetime, format="%Y/%m/%d", strict=False),
        cast_expr.str.strptime(pl.Datetime, format="%Y%m%d", strict=False),
        cast_expr.str.strptime(pl.Datetime, format="%d-%m-%Y", strict=False),
        cast_expr.str.strptime(pl.Datetime, format="%d/%m/%Y", strict=False),
    )


def _normalize_global_density_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _find_date_column(columns: list[str]) -> Optional[str]:
    preferred_tokens = ("timestamp", "datetime", "date", "day", "time", "epoch")
    for token in preferred_tokens:
        for column in columns:
            normalized = _normalize_global_density_name(column)
            if token in normalized:
                return column
    return columns[0] if columns else None


def _find_altitude_column(columns: list[str]) -> Optional[str]:
    for column in columns:
        normalized = _normalize_global_density_name(column)
        if "altitude" in normalized or normalized in {"alt", "height", "heightkm"}:
            return column
    return None


def _find_density_column(columns: list[str], *, exclude: set[str]) -> Optional[str]:
    density_tokens = ("density", "rho", "massdensity")
    for column in columns:
        if column in exclude:
            continue
        normalized = _normalize_global_density_name(column)
        if any(token in normalized for token in density_tokens):
            return column
    return None


def _column_altitude_km(name: str) -> Optional[float]:
    match = GLOBAL_DENSITY_COLUMN_RE.search(name)
    if match is None:
        return None
    return float(match.group(1))


def _global_density_mission_code(
    sourcepath: str, payload_name: str | None = None
) -> str:
    source_name = payload_name or Path(sourcepath).name
    stem = Path(source_name).stem
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")
    return sanitized.upper() or "GLOBAL_DENSITY"


def _read_global_density_table(raw: bytes) -> pl.DataFrame:
    lines = _iter_data_lines(raw)
    if not lines:
        raise ValueError("Global density file contains no readable data lines")

    separator = _detect_separator(lines)
    has_header = any(ch.isalpha() for ch in lines[0])

    if separator is None:
        normalized = (
            "\n".join(";".join(line.split()) for line in lines).encode("utf-8") + b"\n"
        )
        df = pl.read_csv(
            io.BytesIO(normalized),
            separator=";",
            has_header=has_header,
            infer_schema_length=10_000,
            try_parse_dates=False,
        )
    else:
        normalized = "\n".join(lines).encode("utf-8") + b"\n"
        df = pl.read_csv(
            io.BytesIO(normalized),
            separator=separator,
            has_header=has_header,
            infer_schema_length=10_000,
            try_parse_dates=False,
        )

    if not has_header:
        df.columns = [f"column_{i + 1}" for i in range(df.width)]
    return df


def _decode_global_density_frame(df: pl.DataFrame) -> pl.DataFrame:
    date_col = _find_date_column(df.columns)
    if date_col is None:
        raise ValueError("Unable to identify date column in global density file")

    altitude_col = _find_altitude_column(df.columns)
    density_col = _find_density_column(
        df.columns,
        exclude={date_col, altitude_col} if altitude_col else {date_col},
    )

    if altitude_col and density_col:
        altitude_name = _normalize_global_density_name(altitude_col)
        altitude_scale = (
            1_000.0 if "km" in altitude_name or altitude_name == "alt" else 1.0
        )
        decoded = (
            df.with_columns(
                _parse_datetime_column(pl.col(date_col)).alias("timestamp"),
                (
                    pl.col(altitude_col).cast(pl.Float64, strict=False) * altitude_scale
                ).alias("Altitude (m)"),
                pl.col(density_col)
                .cast(pl.Float64, strict=False)
                .alias("Density (kg/m^3)"),
            )
            .select(["timestamp", "Altitude (m)", "Density (kg/m^3)"])
            .drop_nulls()
            .sort(["timestamp", "Altitude (m)"])
        )
        if decoded.is_empty():
            raise ValueError("Decoded long-format global density file is empty")
        return decoded

    altitude_columns: dict[str, float] = {}
    for column in df.columns:
        if column == date_col:
            continue
        altitude_km = _column_altitude_km(column)
        if altitude_km is not None:
            altitude_columns[column] = altitude_km

    if not altitude_columns:
        raise ValueError(
            "Unable to identify altitude columns in global density file. "
            "Expected either explicit altitude/density columns or a wide date-by-altitude table."
        )

    altitude_map = pl.DataFrame(
        {
            "altitude_label": list(altitude_columns.keys()),
            "Altitude (m)": [value * 1_000.0 for value in altitude_columns.values()],
        }
    )

    decoded = (
        df.with_columns(_parse_datetime_column(pl.col(date_col)).alias("timestamp"))
        .select(["timestamp", *altitude_columns.keys()])
        .unpivot(
            index="timestamp",
            variable_name="altitude_label",
            value_name="Density (kg/m^3)",
        )
        .with_columns(pl.col("Density (kg/m^3)").cast(pl.Float64, strict=False))
        .join(altitude_map, on="altitude_label", how="inner")
        .select(["timestamp", "Altitude (m)", "Density (kg/m^3)"])
        .drop_nulls()
        .sort(["timestamp", "Altitude (m)"])
    )
    if decoded.is_empty():
        raise ValueError("Decoded wide-format global density file is empty")
    return decoded


def _append_decode_manifest_entry(
    manifest_path: str,
    *,
    mission: str,
    mission_code: str,
    parquet_path: str,
    sourcepath: str,
) -> None:
    with FileLock(manifest_path + ".lock"):
        if os.path.exists(manifest_path):
            manifest_df = pl.read_csv(manifest_path)
        else:
            manifest_df = pl.DataFrame(
                schema={
                    "mission": pl.Utf8,
                    "mission_code": pl.Utf8,
                    "parquet_path": pl.Utf8,
                    "source_path": pl.Utf8,
                }
            )

        new_entry = pl.DataFrame(
            {
                "mission": [mission],
                "mission_code": [mission_code],
                "parquet_path": [parquet_path],
                "source_path": [sourcepath],
            }
        )

        updated_manifest = pl.concat([manifest_df, new_entry], how="vertical")
        updated_manifest.write_csv(manifest_path)


def decode_tudelft_single(
    mission: str,
    sourcepath: str,
    outfilepath: str,
    manifest_path: str,
) -> Tuple[str, str, str, str] | None:
    schema = TUDELFT_SCHEMAS[mission.upper()]
    parquet_path = str(Path(outfilepath).with_suffix(".parquet"))
    mission_code = str(Path(sourcepath).name).split("_")[0].upper()
    zip_payload = _read_first_txt_from_zip(sourcepath)
    if zip_payload is None:
        return None
    _txt_name, raw = zip_payload

    fixed = normalize_whitespace(raw)
    try:
        df = pl.read_csv(
            io.BytesIO(fixed),
            separator=";",
            comment_prefix="#",
            has_header=False,
            schema=schema,
        )

        if df.is_empty() or df.width < 9:
            logging.warning(
                "Mission %s has an unexpected format and will be skipped.",
                sourcepath,
            )
            return None

        os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
        df.write_parquet(parquet_path, compression="lz4")
        _append_decode_manifest_entry(
            manifest_path,
            mission=mission,
            mission_code=mission_code,
            parquet_path=parquet_path,
            sourcepath=sourcepath,
        )

        return (mission, mission_code, parquet_path, sourcepath)
    except Exception as e:
        logging.error("Error processing %s: %s", sourcepath, e)
        traceback.print_exc()
        # print("\n".join(raw.decode("utf-8", errors="replace").splitlines()[:131]))
        # print(
        #     "\n".join(
        #         [
        #             # color double semicolons in red for better visibility
        #             f"line: {i}, {len(line.split(';'))}, {line.replace(';;', '\033[91m;;\033[0m')}"
        #             for i, line in enumerate(
        #                 fixed.decode("utf-8", errors="replace").splitlines()
        #             )
        #             if line[0] != "#" and len(line.split(";")) != len(schema)
        #         ]
        #     )
        # )
        return None


def decode_tudelft_single_worker(args):
    return decode_tudelft_single(*args)


# VirES marks unavailable densities (e.g. orbit means near data gaps) with this
# sentinel although its HAPI metadata declares no fill value.
VIRES_DENSITY_SENTINEL = 9.99e32
# VirES parameter -> TU Delft decoded column. Columns VirES lacks become null.
VIRES_TO_TUDELFT = {
    "Height_GD": "Altitude (m)",
    "Longitude_GD": "Longitude (deg)",
    "Latitude_GD": "Latitude (deg)",
    "local_solar_time": "Local Solar Time (hours)",
    "density": "Density (kg/m^3)",
    "density_orbitmean": "Density Mean (kg/m^3)",
    "validity_flag": "Anomalus Density (kg/m^3)",
    "validity_flag_orbitmean": "Anomalus Density Mean (kg/m^3)",
}


def read_hapi_csv(sourcepath: str) -> tuple[dict, pl.DataFrame]:
    """Read a HAPI CSV saved with ``include=header``; fill values become null."""
    header_lines: list[bytes] = []
    data_lines: list[bytes] = []
    raw = Path(sourcepath).read_bytes()
    if sourcepath.endswith(".gz"):
        raw = gzip.decompress(raw)
    for line in raw.splitlines(keepends=True):
        (header_lines if line.startswith(b"#") else data_lines).append(line)
    header = json.loads(b"".join(line[1:] for line in header_lines))
    parameters = header["parameters"]
    names = [parameter["name"] for parameter in parameters]
    if not b"".join(data_lines).strip():
        return header, pl.DataFrame(schema={name: pl.Utf8 for name in names})
    df = pl.read_csv(
        io.BytesIO(b"".join(data_lines)),
        has_header=False,
        new_columns=names,
        infer_schema_length=0,
    )
    if df.width != len(names):
        raise ValueError(f"{sourcepath}: {df.width} columns for {len(names)} parameters")
    for parameter in parameters:
        name, fill = parameter["name"], parameter.get("fill")
        if parameter.get("type") == "isotime":
            continue
        value = pl.col(name).cast(pl.Float64)
        if fill is not None:
            value = pl.when(value == float(fill)).then(None).otherwise(value)
        if name in ("density", "density_orbitmean"):
            value = pl.when(value >= VIRES_DENSITY_SENTINEL).then(None).otherwise(value)
        df = df.with_columns(value.alias(name))
    return header, df


def decode_vires_single(
    mission: str,
    sourcepath: str,
    outfilepath: str,
    manifest_path: str,
) -> Tuple[str, str, str, str] | None:
    """Decode one VirES month into the TU Delft decoded schema.

    VirES timestamps are UTC (``Time System`` = ``UTC``) and it does not carry
    the argument of latitude, so that column is null.
    """
    schema = TUDELFT_SCHEMAS[mission.upper()]
    parquet_path = str(Path(outfilepath).with_suffix(".parquet"))
    mission_code = str(Path(sourcepath).name).split("_")[0].upper()
    try:
        header, df = read_hapi_csv(sourcepath)
        if df.is_empty():
            logging.warning("VirES file %s contains no data rows.", sourcepath)
            return None
        timestamp = pl.col("Timestamp").str.strptime(
            pl.Datetime("ms"), "%Y-%m-%dT%H:%M:%S%.fZ"
        )
        columns = [
            timestamp.dt.date().alias("Date yyyy-mm-dd"),
            timestamp.dt.time().alias("Time hh:mm:ss.sss"),
            pl.lit("UTC").alias("Time System"),
            pl.lit(None).alias("Argument of Latitude (deg)"),
        ]
        for source, target in VIRES_TO_TUDELFT.items():
            columns.append(
                (pl.col(source) if source in df.columns else pl.lit(None)).alias(target)
            )
        decoded = df.select(columns).select(
            [pl.col(name).cast(dtype) for name, dtype in schema.items()]
        )
        os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
        decoded.write_parquet(parquet_path, compression="lz4")
        _append_decode_manifest_entry(
            manifest_path,
            mission=mission,
            mission_code=mission_code,
            parquet_path=parquet_path,
            sourcepath=sourcepath,
        )
        logging.info("Decoded %s from %s", header.get("x_dataset"), sourcepath)
        return (mission, mission_code, parquet_path, sourcepath)
    except Exception as e:
        logging.error("Error processing %s: %s", sourcepath, e)
        return None


def _hasdm_mission_code(sourcepath: str, txt_name: str | None = None) -> str:
    source_name = txt_name or Path(sourcepath).name
    match = re.search(r"(\d{4})[_-]?(\d{2})", source_name)
    if match is None:
        raise ValueError(f"Unable to determine HASDM year from {source_name}")
    return f"HASDM_{match.group(1)}"


def _read_hasdm_payload(sourcepath: str) -> tuple[str, bytes] | None:
    path = Path(sourcepath)
    if not path.exists():
        logging.warning("HASDM source file does not exist: %s", sourcepath)
        return None
    if zipfile.is_zipfile(path):
        return _read_first_matching_from_zip(
            sourcepath,
            (".txt", ".csv", ".dat", ".tsv"),
        )

    raw = path.read_bytes()
    if raw.startswith(b"PK"):
        logging.warning(
            "HASDM source %s is not a readable zip archive and will be skipped.",
            sourcepath,
        )
        return None
    return path.name, raw


def _parse_hasdm_timestamp(expr: pl.Expr) -> pl.Expr:
    value = expr.cast(pl.Utf8, strict=False).str.strip_chars()
    normalized = (
        pl.when(value.str.len_bytes() == 10).then(value + "00").otherwise(value)
    )
    return pl.coalesce(
        normalized.str.strptime(pl.Datetime, format="%Y%m%d%H%M", strict=False),
        value.str.strptime(pl.Datetime, format="%Y%m%d", strict=False),
    )


def decode_hasdm_single(
    sourcepath: str,
    outfilepath: str,
    manifest_path: str,
) -> Tuple[str, str, str, str] | None:
    parquet_path = str(Path(outfilepath).with_suffix(".parquet"))

    try:
        payload = _read_hasdm_payload(sourcepath)
        if payload is None:
            return None
        txt_name, raw = payload
        mission = "hasdm"
        mission_code = _hasdm_mission_code(sourcepath, txt_name)
        fixed = normalize_whitespace(raw, comment_prefixes=(b"#", b":"))
        if not fixed.strip():
            logging.warning(
                "HASDM file %s contains no data rows and will be skipped.", sourcepath
            )
            return None

        df = pl.read_csv(
            io.BytesIO(fixed),
            separator=";",
            has_header=False,
            schema=HASDM_SCHEMA,
            null_values=["null", "NULL"],
        )

        if df.is_empty() or df.width < len(HASDM_SCHEMA):
            logging.warning(
                "HASDM file %s has an unexpected format and will be skipped.",
                sourcepath,
            )
            return None

        df = (
            df.with_columns(
                _parse_hasdm_timestamp(pl.col("YYYYMMDDhhmm")).alias("timestamp"),
                (pl.col("HTM") * 1000.0).alias("Altitude (m)"),
                pl.col("LAT").alias("Latitude (deg)"),
                pl.col("LON").alias("Longitude (deg)"),
                pl.col("LST").alias("Local Solar Time (hours)"),
                pl.col("RHO").alias("Density (kg/m^3)"),
            )
            .drop_nulls(subset=["timestamp"])
            .select(
                [
                    "JulianDay",
                    "Altitude (m)",
                    "Longitude (deg)",
                    "Latitude (deg)",
                    "Local Solar Time (hours)",
                    "Density (kg/m^3)",
                    "timestamp",
                ]
            )
            .sort("timestamp")
        )
        if df.is_empty():
            logging.warning(
                "HASDM file %s did not yield any decodable rows and will be skipped.",
                sourcepath,
            )
            return None

        os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
        df.write_parquet(parquet_path, compression="lz4")
        _append_decode_manifest_entry(
            manifest_path,
            mission=mission,
            mission_code=mission_code,
            parquet_path=parquet_path,
            sourcepath=sourcepath,
        )
        return (mission, mission_code, parquet_path, sourcepath)
    except Exception as e:
        logging.error("Error processing HASDM %s: %s", sourcepath, e)
        traceback.print_exc()
        return None


def decode_hasdm_single_worker(args):
    return decode_hasdm_single(*args)


def decode_global_density_single(
    sourcepath: str,
    outfilepath: str,
    manifest_path: str,
) -> Tuple[str, str, str, str] | None:
    parquet_path = str(Path(outfilepath).with_suffix(".parquet"))
    payload = _read_text_payload(sourcepath)
    if payload is None:
        return None

    payload_name, raw = payload
    mission = "global_density"
    mission_code = _global_density_mission_code(sourcepath, payload_name)

    try:
        df = _read_global_density_table(raw)
        df = _decode_global_density_frame(df)
        os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
        df.write_parquet(parquet_path, compression="lz4")
        _append_decode_manifest_entry(
            manifest_path,
            mission=mission,
            mission_code=mission_code,
            parquet_path=parquet_path,
            sourcepath=sourcepath,
        )
        return (mission, mission_code, parquet_path, sourcepath)
    except Exception as e:
        logging.error("Error processing global density file %s: %s", sourcepath, e)
        traceback.print_exc()
        return None


def decode_global_density_single_worker(args):
    return decode_global_density_single(*args)


def merge_parquets(
    parquet_paths: List[str],
    output_path: str,
    manifest_path: str,
):
    dfs = []
    print(f"Merging {len(parquet_paths)} parquet files into {output_path}")
    for path in parquet_paths:
        try:
            df = pl.read_parquet(path)
            dfs.append(df)
        except Exception as e:
            print("Traceback (most recent call last):")
            pprint(parquet_paths)
            pprint("Path: " + path)
            traceback.print_exc()
            logging.error("Error reading %s: %s", path, e)
            raise e
    # create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if dfs:
        merged_df = pl.concat(dfs, how="vertical")
        # Check for Time System values and log if there are multiple
        if "Time System" in merged_df.columns:
            time_systems = merged_df["Time System"].unique()
            if len(time_systems) > 1:
                logging.warning(
                    "Multiple time systems found in %s: %s",
                    output_path,
                    time_systems,
                )
            else:
                logging.info(
                    "Single time system found in %s: %s dropping time_system column",
                    output_path,
                    time_systems[0],
                )
                merged_df = merged_df.drop("Time System")
        # combine date and time columns if they exist by converting to an iso timestamp, decoding the timestamp as UTC and then converting to unix timestamp in seconds
        if (
            "Date yyyy-mm-dd" in merged_df.columns
            and "Time hh:mm:ss.sss" in merged_df.columns
        ):
            merged_df = merged_df.with_columns(
                (
                    pl.col("Date yyyy-mm-dd").cast(pl.Utf8)
                    + "T"
                    + pl.col("Time hh:mm:ss.sss").cast(pl.Utf8)
                    + "Z"
                )
                .str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S%.3fZ")
                .alias("timestamp")
            ).drop(["Date yyyy-mm-dd", "Time hh:mm:ss.sss"])
        if "timestamp" in merged_df.columns:
            merged_df = merged_df.sort("timestamp")

        merged_df.write_parquet(output_path, compression="snappy")
        # Update manifest with locking
        with FileLock(manifest_path + ".lock"):
            if os.path.exists(manifest_path):
                manifest_df = pl.read_csv(manifest_path)
            else:
                manifest_df = pl.DataFrame(
                    schema={
                        "mission_code": pl.Utf8,
                        "parquet_path": pl.Utf8,
                    }
                )

            mission_code = Path(output_path).stem.removesuffix("_merged")
            new_entry = pl.DataFrame(
                {
                    "mission_code": [mission_code],
                    "parquet_path": [output_path],
                }
            )
            updated_manifest = pl.concat([manifest_df, new_entry], how="vertical")
            updated_manifest.write_csv(manifest_path)
        print(f"Finished merging into {output_path}")
    else:
        logging.warning("No valid parquet files to merge for %s", output_path)


def merge_parquets_single_worker(args):
    mission_code, parquet_paths, output_path, manifest_path = args
    merge_parquets(parquet_paths, output_path, manifest_path)
