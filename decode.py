import io
import logging
import os
import re
import traceback
import zipfile
from pathlib import Path
from pprint import pprint
from typing import List, Tuple

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


def _hasdm_mission_code(sourcepath: str, txt_name: str | None = None) -> str:
    source_name = txt_name or Path(sourcepath).name
    match = re.search(r"(\d{4})(\d{2})", source_name)
    if match is None:
        raise ValueError(f"Unable to determine HASDM year from {source_name}")
    return f"HASDM_{match.group(1)}"


def decode_hasdm_single(
    sourcepath: str,
    outfilepath: str,
    manifest_path: str,
) -> Tuple[str, str, str, str] | None:
    parquet_path = str(Path(outfilepath).with_suffix(".parquet"))
    zip_payload = _read_first_txt_from_zip(sourcepath)
    if zip_payload is None:
        return None
    txt_name, raw = zip_payload
    mission = "hasdm"
    mission_code = _hasdm_mission_code(sourcepath, txt_name)

    fixed = normalize_whitespace(raw, comment_prefixes=(b"#", b":"))

    try:
        df = pl.read_csv(
            io.BytesIO(fixed),
            separator=";",
            has_header=False,
            schema=HASDM_SCHEMA,
        )

        if df.is_empty() or df.width < len(HASDM_SCHEMA):
            logging.warning("HASDM file %s has an unexpected format and will be skipped.", sourcepath)
            return None

        df = (
            df.with_columns(
                pl.col("YYYYMMDDhhmm")
                .str.strptime(pl.Datetime, format="%Y%m%d%H%M", strict=True)
                .alias("timestamp"),
                (pl.col("HTM") * 1000.0).alias("Altitude (m)"),
                pl.col("LAT").alias("Latitude (deg)"),
                pl.col("LON").alias("Longitude (deg)"),
                pl.col("LST").alias("Local Solar Time (hours)"),
                pl.col("RHO").alias("Density (kg/m^3)"),
            )
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
