import argparse
import gc
import logging
from multiprocessing import Pool
from pathlib import Path

import polars as pl
from tqdm import tqdm

from thermodense.decoding import (
    decode_hasdm_single_worker,
    merge_parquets_single_worker,
)


def _read_manifest(path: Path) -> pl.DataFrame | None:
    return pl.read_csv(path) if path.exists() else None


def _manifest_results(
    manifest_df: pl.DataFrame | None,
) -> list[tuple[str, str, str, str]]:
    if manifest_df is None:
        return []
    return [
        (row[0], row[1], row[2], row[3])
        for row in manifest_df.select(
            ["mission", "mission_code", "parquet_path", "source_path"]
        ).iter_rows()
    ]


def decode_hasdm_dataset(
    input_dir: str = "data/original/hasdm",
    output_dir: str = "data/decoded/hasdm",
    *,
    workers: int | None = None,
) -> list[tuple[str, str, str, str]]:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    tmp_manifest_path = output_path / "tmp_decode_manifest.csv"
    decoded_manifest_path = output_path / "decoded_manifest.csv"

    tmp_manifest_df = _read_manifest(tmp_manifest_path)
    done_files = (
        set(tmp_manifest_df["source_path"].to_list())
        if tmp_manifest_df is not None
        else set()
    )

    arg_list = [
        (
            str(path),
            str(output_path / "tmp" / f"{path.name}.parquet"),
            str(tmp_manifest_path),
        )
        for path in sorted(input_path.iterdir())
        if path.is_file()
        and path.name.startswith("hasdm_")
        and str(path) not in done_files
    ]

    results: list[tuple[str, str, str, str]] = []
    if arg_list:
        with Pool(processes=workers) as pool:
            results = [
                result
                for result in tqdm(
                    pool.imap(decode_hasdm_single_worker, arg_list),
                    total=len(arg_list),
                )
                if result is not None
            ]
    elif tmp_manifest_df is None:
        logging.warning("No HASDM files were found to decode.")
    else:
        logging.info("All HASDM files were already decoded. Skipping decoding step.")

    all_results = results + _manifest_results(tmp_manifest_df)
    if not all_results:
        return []

    mission_temp_paths: dict[str, dict[str, list[str] | str]] = {}
    for mission, mission_code, parquet_path, _source_path in all_results:
        mission_temp_paths.setdefault(
            mission_code,
            {"mission": mission, "parquet_paths": []},
        )["parquet_paths"].append(parquet_path)

    decoded_manifest_df = _read_manifest(decoded_manifest_path)
    decoded_manifest_sources = (
        set(decoded_manifest_df["mission_code"].to_list())
        if decoded_manifest_df is not None
        else set()
    )
    merge_args = [
        (
            mission_code,
            mission_temp_paths[mission_code]["parquet_paths"],
            str(output_path / f"{mission_code}_merged.parquet"),
            str(decoded_manifest_path),
        )
        for mission_code in sorted(mission_temp_paths)
        if mission_code not in decoded_manifest_sources
    ]

    if merge_args:
        with Pool(processes=workers) as pool:
            list(pool.imap(merge_parquets_single_worker, merge_args))
    else:
        logging.info("No HASDM merged outputs needed updating.")

    gc.collect()
    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode HASDM archives into parquet files."
    )
    parser.add_argument("--input-dir", default="data/original/hasdm")
    parser.add_argument("--output-dir", default="data/decoded/hasdm")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))
    decode_hasdm_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
