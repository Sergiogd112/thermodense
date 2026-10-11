"""Convert the orbit-derived global-mean density table to the wide daily Parquet.

Input: whitespace table with header ``year day log10rho_<km> sigma_<km> ...``
(day = day of year). Output: ``date`` plus every ``log10rho_*``/``sigma_*``
column, unchanged, at the path ``recreate_brown_density_trend_current`` and
``global_mean`` read. Fill values are kept; the trend estimator drops them.
Reproduces the 19 Aug 2026 global-mean trends exactly (max difference 2e-13
%/decade, identical sample counts).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

SOURCE = Path("data/original/orbit_derived_global_mean/orbit-density-ds03-density-values.txt")
OUTPUT = Path("data/decoded/orbit_derived_global_mean/orbit-density-ds03-density-values.parquet")


def convert(text: str) -> pl.DataFrame:
    lines = [line.split() for line in text.splitlines() if line.strip()]
    header, rows = lines[0], lines[1:]
    if header[:2] != ["year", "day"]:
        raise ValueError(f"expected 'year day' leading columns, found {header[:2]}")
    bad = [i for i, row in enumerate(rows, start=2) if len(row) != len(header)]
    if bad:
        raise ValueError(f"rows with a wrong field count, e.g. line {bad[0]}")
    values = [c for c in header[2:] if c.startswith(("log10rho_", "sigma_"))]
    df = (
        pl.DataFrame(rows, schema=header, orient="row")
        .with_columns(pl.col("year", "day").cast(pl.Int32), pl.col(values).cast(pl.Float64))
        .with_columns((pl.date(pl.col("year"), 1, 1) + pl.duration(days=pl.col("day") - 1)).alias("date"))
        .select("date", *values)
    )
    if df.height != df.get_column("date").n_unique():
        raise ValueError("duplicate dates in global-mean table")
    return df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    df = convert(args.source.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.output)
    print(f"{args.output}: {df.height} days, {df['date'].min()}..{df['date'].max()}")


if __name__ == "__main__":
    main()
