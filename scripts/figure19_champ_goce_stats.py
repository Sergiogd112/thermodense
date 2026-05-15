from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl


F107_RANGES = (
    ("<100", pl.col("f107a") < 100),
    ("100-160", (pl.col("f107a") >= 100) & (pl.col("f107a") < 160)),
    (">=160", pl.col("f107a") >= 160),
)
METRICS = (
    "ln_density_ratio_0",
    "ln_density_ratio_2.0",
    "ln_density_ratio_matlab_density",
)


@dataclass(frozen=True)
class MissionConfig:
    name: str
    path: Path
    altitude_range_km: tuple[float, float]


MISSIONS = (
    MissionConfig(
        name="CHAMP",
        path=Path("data/analyzed/tudelft/champ/CH_analyzed.parquet"),
        altitude_range_km=(300, 500),
    ),
    MissionConfig(
        name="GOCE",
        path=Path("data/analyzed/tudelft/goce/GO_analyzed.parquet"),
        altitude_range_km=(225, 300),
    ),
)


def _load_mission(config: MissionConfig, *, exclude_champ_2005: bool) -> pl.DataFrame:
    if not config.path.exists():
        raise FileNotFoundError(f"Missing analyzed parquet: {config.path}")

    df = pl.read_parquet(config.path).sort("timestamp")
    min_alt, max_alt = config.altitude_range_km
    base_filter = (pl.col("Altitude (m)") / 1000).is_between(min_alt, max_alt)

    if config.name == "CHAMP":
        # Figure 19 caption excludes 2006-2009; section 3.1 says 2005-2009.
        solar_minimum_start = date(2005, 1, 1) if exclude_champ_2005 else date(2006, 1, 1)
        return df.filter(
            base_filter
            & (pl.col("timestamp") >= date(2001, 1, 1))
            & (
                (pl.col("timestamp") < solar_minimum_start)
                | (pl.col("timestamp") > date(2009, 12, 31))
            )
            & (pl.col("Anomalus Density (kg/m^3)") == 0)
        )

    if config.name == "GOCE":
        return df.filter(
            base_filter
            & (pl.col("Anomalus Density (kg/m^3)") == 0)
            & (pl.col("Degraded Flag Thrusters") == 0)
        )

    raise ValueError(f"Unsupported mission: {config.name}")


def _with_figure19_bins(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (
            ((pl.col("timestamp").dt.ordinal_day() - 1) / 30.5)
            .floor()
            .clip(0, 11)
            .cast(pl.Int64)
        ).alias("seasonal_bin"),
        pl.col("timestamp").dt.date().alias("date_bin"),
    )


def _mission_summary(df: pl.DataFrame) -> dict[str, object]:
    return df.select(
        pl.len().alias("rows"),
        pl.col("timestamp").dt.date().n_unique().alias("unique_days"),
        pl.col("timestamp").min().alias("start"),
        pl.col("timestamp").max().alias("end"),
        (pl.col("Altitude (m)") / 1000).min().alias("min_alt_km"),
        (pl.col("Altitude (m)") / 1000).max().alias("max_alt_km"),
    ).row(0, named=True)


def _bin_stats(df: pl.DataFrame) -> pl.DataFrame:
    present_metrics = [metric for metric in METRICS if metric in df.columns]
    daily = df.group_by(["seasonal_bin", "date_bin"]).agg(
        pl.col(metric).mean().alias(metric) for metric in present_metrics
    )

    return (
        df.group_by("seasonal_bin")
        .agg(
            pl.len().alias("rows"),
            pl.col("date_bin").n_unique().alias("unique_days"),
            *[
                pl.col(metric).mean().alias(f"{metric}_mean")
                for metric in present_metrics
            ],
            *[
                pl.col(metric).std().alias(f"{metric}_raw_std")
                for metric in present_metrics
            ],
        )
        .join(
            daily.group_by("seasonal_bin").agg(
                *[
                    pl.col(metric).mean().alias(f"{metric}_daily_mean")
                    for metric in present_metrics
                ],
                *[
                    (
                        pl.col(metric).std().fill_null(0)
                        / pl.col(metric).count().sqrt()
                    ).alias(f"{metric}_daily_mean_sem")
                    for metric in present_metrics
                ],
            ),
            on="seasonal_bin",
        )
        .with_columns(
            *[
                (
                    pl.col(f"{metric}_raw_std")
                    / pl.col("unique_days").cast(pl.Float64).sqrt()
                ).alias(f"{metric}_raw_std_over_sqrt_days")
                for metric in present_metrics
            ]
        )
        .sort("seasonal_bin")
    )


def _print_summary(config: MissionConfig, df: pl.DataFrame) -> None:
    summary = _mission_summary(df)
    print(f"\n== {config.name} ==")
    print(
        "rows={rows:,} unique_days={unique_days:,} start={start} end={end} "
        "alt_km={min_alt_km:.1f}-{max_alt_km:.1f}".format(**summary)
    )


def _print_f107_stats(df: pl.DataFrame, *, metric: str) -> None:
    binned_df = _with_figure19_bins(df)
    for label, f107_filter in F107_RANGES:
        filtered = binned_df.filter(f107_filter)
        if filtered.is_empty():
            print(f"\nF10.7a {label}: no rows")
            continue

        stats = _bin_stats(filtered)
        print(f"\nF10.7a {label}")
        print(
            stats.select(
                pl.col("seasonal_bin").alias("bin"),
                "rows",
                pl.col("unique_days").alias("days"),
                pl.col(f"{metric}_mean").alias("raw_mean"),
                pl.col(f"{metric}_daily_mean").alias("daily_mean"),
                pl.col(f"{metric}_daily_mean_sem").alias("daily_sem"),
                pl.col(f"{metric}_raw_std_over_sqrt_days").alias("raw_sem"),
                pl.col(f"{metric}_raw_std").alias("raw_std"),
            )
        )

        maxima = stats.select(
            pl.col(f"{metric}_daily_mean_sem").max().alias("max_daily_mean_sem"),
            pl.col(f"{metric}_raw_std_over_sqrt_days")
            .max()
            .alias("max_raw_std_over_sqrt_days"),
            pl.col(f"{metric}_raw_std").max().alias("max_raw_std"),
        ).row(0, named=True)
        print(
            "max daily_mean_sem={max_daily_mean_sem:.6f} "
            "max raw_std/sqrt(days)={max_raw_std_over_sqrt_days:.6f} "
            "max raw_std={max_raw_std:.6f}".format(**maxima)
        )


def main() -> int:
    pl.Config.set_tbl_rows(20)

    parser = argparse.ArgumentParser(
        description="Print CHAMP/GOCE Figure 19 bin means and uncertainty diagnostics."
    )
    parser.add_argument(
        "--exclude-champ-2005",
        action="store_true",
        help="Use the section 3.1 CHAMP exclusion (2005-2009) instead of the Figure 19 caption exclusion (2006-2009).",
    )
    parser.add_argument(
        "--metric",
        default="ln_density_ratio_2.0",
        choices=METRICS,
        help="Residual column to print in detail.",
    )
    args = parser.parse_args()

    champ_exclusion = "2005-2009" if args.exclude_champ_2005 else "2006-2009"
    print(f"Figure 19 CHAMP/GOCE diagnostics; CHAMP exclusion={champ_exclusion}")
    print(f"Detailed metric={args.metric}")
    print("daily_mean_sem = std(daily bin means) / sqrt(unique days)")
    print("raw_std/sqrt(days) = std(all observations in bin) / sqrt(unique days)")

    for config in MISSIONS:
        df = _load_mission(config, exclude_champ_2005=args.exclude_champ_2005)
        _print_summary(config, df)
        _print_f107_stats(df, metric=args.metric)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
