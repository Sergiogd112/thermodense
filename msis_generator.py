import csv
import gc
import math
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from multiprocessing import Pool, cpu_count
from pathlib import Path
from pprint import pprint
from time import perf_counter

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from pymsis import msis
from pymsis.utils import get_f107_ap

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

MODELS = ["0", "2.0", "2.1"]
DEFAULT_BATCH_SIZE = 100_000
MANIFEST_FIELDS = [
    "finished_at_utc",
    "status",
    "source_file",
    "output_file",
    "file_index",
    "rows",
    "batches",
    "elapsed_s",
    "batch_size",
    "inner_threads",
    "error",
]


def _parse_file_arg(file_or_tuple):
    if isinstance(file_or_tuple, tuple):
        if len(file_or_tuple) >= 2:
            file, n = file_or_tuple[:2]
            return str(file), n
        return str(file_or_tuple[0]), None
    return str(file_or_tuple), None


def _normalize_file_arg(file_or_tuple):
    return _parse_file_arg(file_or_tuple)[0]


def _file_label(file, n=None):
    name = Path(file).name
    if n is None:
        return name
    return f"[{n}] {name}"


def _output_path_for(file):
    src_path = Path(file)
    return Path(
        f"data/msis/tudelft/{src_path.parent.name}/{src_path.stem.replace("merged", "msis")}.parquet"
    )


def _estimate_num_batches(parquet_file, batch_size):
    metadata = parquet_file.metadata
    total = 0
    for row_group_idx in range(parquet_file.num_row_groups):
        num_rows = metadata.row_group(row_group_idx).num_rows
        total += math.ceil(num_rows / batch_size)
    return total


def _iter_row_group_batches(
    parquet_file: pq.ParquetFile, batch_size=DEFAULT_BATCH_SIZE
):
    # for row_group_idx in range(parquet_file.num_row_groups):
    #     yield from parquet_file.iter_batches(
    #         row_groups=[row_group_idx],
    #         batch_size=batch_size,
    #         use_threads=True,
    #     )
    yield from parquet_file.iter_batches(
        row_groups=list(range(parquet_file.num_row_groups)),
        batch_size=batch_size,
        use_threads=True,
    )


def _manifest_row(
    *,
    status,
    source_file,
    output_file,
    file_index,
    rows,
    batches,
    elapsed_s,
    batch_size,
    inner_threads,
    error="",
):
    return {
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "source_file": str(source_file),
        "output_file": str(output_file),
        "file_index": "" if file_index is None else file_index,
        "rows": rows,
        "batches": batches,
        "elapsed_s": round(float(elapsed_s), 3),
        "batch_size": batch_size,
        "inner_threads": inner_threads,
        "error": error,
    }


def _append_manifest_row(manifest_path, row):
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not manifest_path.exists() or manifest_path.stat().st_size == 0

    with manifest_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in MANIFEST_FIELDS})
        f.flush()
        os.fsync(f.fileno())


def _load_successful_files(manifest_path):
    manifest_path = Path(manifest_path)
    if not manifest_path.exists() or manifest_path.stat().st_size == 0:
        return set()

    completed = set()
    with manifest_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "success" and row.get("source_file"):
                completed.add(row["source_file"])
    return completed


def _filter_pending_files(
    files,
    manifest_path=None,
    resume=True,
    skip_existing_output=True,
):
    completed = set()
    if manifest_path is not None and resume:
        completed = _load_successful_files(manifest_path)

    pending = []
    skipped = []

    for file in files:
        source_file, _ = _parse_file_arg(file)
        output_file = _output_path_for(source_file)

        if source_file in completed:
            if output_file.exists():
                skipped.append((file, "manifest", output_file))
                continue

            print(
                f"Manifest marks done but output is missing, reprocessing: "
                f"{source_file}",
                flush=True,
            )

        if skip_existing_output and output_file.exists():
            skipped.append((file, "output_exists"))
            continue

        pending.append(file)

    return pending, skipped


def _run_msis_job(args):
    (
        timestamps,
        lons,
        lats,
        alts,
        f107,
        f107a,
        aps,
        version,
        column_name,
    ) = args

    density = msis.calculate(
        timestamps,
        lons,
        lats,
        alts,
        f107,
        f107a,
        aps,
        version=version,
    )[:, 0]

    return pl.Series(column_name, density)


def _add_msis_columns(df, inner_threads=1):
    timestamps = df["timestamp"].to_numpy()
    lats = df["Latitude (deg)"].to_numpy()
    lons = df["Longitude (deg)"].to_numpy()
    alts = (df["Altitude (m)"] / 1000.0).to_numpy()

    f107, f107a, aps, f107_type = get_f107_ap(timestamps)

    f107_ct = np.full(f107.shape, 150.0, dtype=np.float64)
    f107a_ct = np.full(f107a.shape, 150.0, dtype=np.float64)
    ap_ct = np.zeros_like(aps, dtype=np.float64)

    jobs = []
    for version in MODELS:
        jobs.append(
            (
                timestamps,
                lons,
                lats,
                alts,
                f107,
                f107a,
                aps,
                version,
                f"msis_density_{version}",
            )
        )

    for version in MODELS:
        jobs.append(
            (
                timestamps,
                lons,
                lats,
                alts,
                f107_ct,
                f107a_ct,
                ap_ct,
                version,
                f"msis_density_ct_{version}",
            )
        )

    if inner_threads > 1:
        max_workers = min(inner_threads, len(jobs))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            densities = list(executor.map(_run_msis_job, jobs))
    else:
        densities = [_run_msis_job(job) for job in jobs]

    out = df.with_columns(
        *densities,
        pl.Series("f107", f107),
        pl.Series("f107a", f107a),
        pl.Series("ap", aps[:, 0]),
        pl.Series("qflag", f107_type),
    )

    del densities
    del jobs
    del f107, f107a, aps, f107_type
    del f107_ct, f107a_ct, ap_ct
    del timestamps, lats, lons, alts
    gc.collect()

    return out


def compute_msis_density(
    file,
    batch_size=DEFAULT_BATCH_SIZE,
    inner_threads=1,
    show_batch_progress=False,
    log_every_n_batches=10,
):
    file, n = _parse_file_arg(file)
    label = _file_label(file, n)

    src_path = Path(file)
    dst_path = _output_path_for(file)
    tmp_path = dst_path.with_suffix(f"{dst_path.suffix}.tmp")
    os.makedirs(tmp_path.parent, exist_ok=True)

    parquet_file = pq.ParquetFile(src_path)
    est_batches = _estimate_num_batches(parquet_file, batch_size)

    print(
        f"START {label} | row_groups={parquet_file.num_row_groups} "
        f"| est_batches={est_batches} | inner_threads={inner_threads}",
        flush=True,
    )

    writer = None
    progress = None
    started_at = perf_counter()
    processed_batches = 0
    processed_rows = 0

    if show_batch_progress and tqdm is not None:
        progress = tqdm(
            total=est_batches,
            desc=label,
            unit="batch",
            leave=True,
        )

    try:
        for batch in _iter_row_group_batches(
            parquet_file,
            batch_size=batch_size,
        ):
            batch_rows = batch.num_rows
            df = pl.from_arrow(batch)
            out_df = _add_msis_columns(df, inner_threads=inner_threads)
            out_table = out_df.to_arrow()

            if writer is None:
                writer = pq.ParquetWriter(
                    tmp_path.as_posix(),
                    out_table.schema,
                    compression="snappy",
                )

            writer.write_table(out_table)

            processed_batches += 1
            processed_rows += batch_rows

            if progress is not None:
                progress.update(1)
            elif (
                log_every_n_batches is not None
                and log_every_n_batches > 0
                and (
                    processed_batches == 1
                    or processed_batches % log_every_n_batches == 0
                    or processed_batches == est_batches
                )
            ):
                elapsed = perf_counter() - started_at
                print(
                    f"PROGRESS {label} | batches={processed_batches}/"
                    f"{est_batches} | rows={processed_rows:,} "
                    f"| elapsed={elapsed:.1f}s"
                    f"| eta={elapsed / processed_batches * (est_batches - processed_batches):.1f}s",
                    flush=True,
                )

            del batch, df, out_df, out_table
            gc.collect()

        if writer is None:
            raise ValueError(f"No rows found in {file}")

        writer.close()
        writer = None

        if progress is not None:
            progress.close()
            progress = None

        tmp_path.replace(dst_path)

        elapsed = perf_counter() - started_at
        print(
            f"DONE {label} | rows={processed_rows:,} | "
            f"batches={processed_batches} | elapsed={elapsed:.1f}s "
            f"| output={dst_path}",
            flush=True,
        )

        return _manifest_row(
            status="success",
            source_file=src_path,
            output_file=dst_path,
            file_index=n,
            rows=processed_rows,
            batches=processed_batches,
            elapsed_s=elapsed,
            batch_size=batch_size,
            inner_threads=inner_threads,
        )

    except Exception as exc:
        if progress is not None:
            progress.close()

        if writer is not None:
            writer.close()

        if tmp_path.exists():
            tmp_path.unlink()

        elapsed = perf_counter() - started_at
        print(
            f"FAILED {label} | elapsed={elapsed:.1f}s | error={exc}",
            flush=True,
        )
        raise


def _compute_msis_density_worker(args):
    (
        file,
        batch_size,
        inner_threads,
        show_batch_progress,
        log_every_n_batches,
    ) = args

    source_file, n = _parse_file_arg(file)
    output_file = _output_path_for(source_file)
    started_at = perf_counter()

    try:
        return compute_msis_density(
            file,
            batch_size=batch_size,
            inner_threads=inner_threads,
            show_batch_progress=show_batch_progress,
            log_every_n_batches=log_every_n_batches,
        )
    except Exception as exc:
        elapsed = perf_counter() - started_at
        return _manifest_row(
            status="failed",
            source_file=source_file,
            output_file=output_file,
            file_index=n,
            rows="",
            batches="",
            elapsed_s=elapsed,
            batch_size=batch_size,
            inner_threads=inner_threads,
            error=f"{type(exc).__name__}: {exc}",
        )


def compute_msis_density_parallel(
    files,
    batch_size=DEFAULT_BATCH_SIZE,
    processes=None,
    inner_threads=1,
    show_progress=True,
    log_every_n_batches=10,
    manifest_path=None,
    resume=True,
    skip_existing_output=True,
):
    if isinstance(files, (str, tuple)):
        pending, skipped = _filter_pending_files(
            [files],
            manifest_path=manifest_path,
            resume=resume,
            skip_existing_output=skip_existing_output,
        )

        if not pending:
            source_file, n = _parse_file_arg(files)
            print(
                f"SKIP {_file_label(source_file, n)} | already completed",
                flush=True,
            )
            return {
                "status": "skipped",
                "source_file": source_file,
                "output_file": str(_output_path_for(source_file)),
            }

        try:
            result = compute_msis_density(
                pending[0],
                batch_size=batch_size,
                inner_threads=inner_threads,
                show_batch_progress=show_progress,
                log_every_n_batches=log_every_n_batches,
            )
            if manifest_path is not None:
                _append_manifest_row(manifest_path, result)
            return result
        except Exception as exc:
            source_file, n = _parse_file_arg(pending[0])
            failure = _manifest_row(
                status="failed",
                source_file=source_file,
                output_file=_output_path_for(source_file),
                file_index=n,
                rows="",
                batches="",
                elapsed_s=0.0,
                batch_size=batch_size,
                inner_threads=inner_threads,
                error=f"{type(exc).__name__}: {exc}",
            )
            if manifest_path is not None:
                _append_manifest_row(manifest_path, failure)
            raise

    pending_files, skipped_files = _filter_pending_files(
        files,
        manifest_path=manifest_path,
        resume=resume,
        skip_existing_output=skip_existing_output,
    )

    if skipped_files:
        print(
            f"Skipping {len(skipped_files)} already-completed files",
            flush=True,
        )

    if not pending_files:
        print("Nothing to do. All files are already complete.", flush=True)
        return [
            {
                "status": "skipped",
                "source_file": file,
                "output_file": skipped_files[i][2],
            }
            for i, file in enumerate(skipped_files)
        ]

    nproc = processes or max(cpu_count() - 1, 1)

    work = [
        (
            file,
            batch_size,
            inner_threads,
            False,
            log_every_n_batches,
        )
        for file in pending_files
    ]

    print(
        f"Starting MSIS generation for {len(work)} files "
        f"with {nproc} worker processes "
        f"(batch_size={batch_size}, inner_threads={inner_threads})",
        flush=True,
    )

    if manifest_path is not None:
        print(f"Manifest: {manifest_path}", flush=True)
    # include skipped files in the results with status "skipped"
    results = [
        {
            "status": "skipped",
            "source_file": file,
            "output_file": skipped_files[i][0][2],
        }
        for i, file in enumerate(skipped_files)
    ]
    succeeded = 0
    failed = 0

    with Pool(processes=nproc) as pool:
        iterator = pool.imap_unordered(_compute_msis_density_worker, work)

        if show_progress and tqdm is not None:
            progress = tqdm(
                iterator,
                total=len(work),
                desc="Files",
                unit="file",
            )

            for result in progress:
                results.append(result)

                if manifest_path is not None:
                    _append_manifest_row(manifest_path, result)

                if result["status"] == "success":
                    succeeded += 1
                else:
                    failed += 1

                progress.set_postfix(
                    success=succeeded,
                    failed=failed,
                )
        else:
            for i, result in enumerate(iterator, start=1):
                results.append(result)

                if manifest_path is not None:
                    _append_manifest_row(manifest_path, result)

                if result["status"] == "success":
                    succeeded += 1
                else:
                    failed += 1

                if show_progress:
                    print(
                        f"Completed {i}/{len(work)} files | "
                        f"success={succeeded} | failed={failed}",
                        flush=True,
                    )

    print(
        f"MSIS generation complete | processed={len(work)} "
        f"| success={succeeded} | failed={failed} "
        f"| skipped={len(skipped_files)}",
        flush=True,
    )

    return results
