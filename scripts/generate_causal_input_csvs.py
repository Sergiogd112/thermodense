from pathlib import Path

import numpy as np
import polars as pl
from thermodense.downloader.space_weather import SPACE_WEATHER_CSV_PATH

REPO = Path(__file__).resolve().parents[1]

GLOBAL_MEAN_PATH = (
    REPO
    / "data/decoded/orbit_derived_global_mean/orbit-density-ds03-density-values.parquet"
)
HASDM_WIDE_PATH = (
    REPO
    / "outputs/figures/results/set_hasdm/model_validations/causal_hasdm_saber_maunaloa/hasdm_maunaloa_daily_wide.parquet"
)
MSIS_WIDE_PATH = (
    REPO
    / "outputs/figures/results/hasdm_msis_model_errors/data/hasdm_msis_errors_nearest_timestamp_grid_daily_wide.parquet"
)
SABER_PATH = REPO / "data/decoded/saber/saber_co2_cooling_maunaloa_daily.parquet"
SPACE_WEATHER_PATH = REPO / SPACE_WEATHER_CSV_PATH
CO2_PATH = REPO / "data/original/co2/co2_daily_mlo.csv"

OUTPUT_DIR = REPO / "data/products/causal_discovery"

SABER_ALTITUDES = [100, 119, 139]
MSIS_MODELS = ["nrlmsise_00", "nrlmsis_2p0", "nrlmsis_2p1"]
DENSITY_ALTITUDES = [325, 825]
METRICS = ["mean", "range"]


def load_saber() -> pl.DataFrame:
    long_df = pl.read_parquet(SABER_PATH)
    available = sorted(long_df["altitude_km"].unique().to_list())
    frames = []
    for alt in SABER_ALTITUDES:
        nearest = float(
            available[int(np.argmin(np.abs(np.asarray(available, dtype=float) - alt)))]
        )
        frames.append(
            long_df.filter(pl.col("altitude_km") == nearest).select(
                "date",
                pl.col("co2_cooling_rate_w_m3").alias(f"saber_co2cool_{alt}km"),
            )
        )
    result = frames[0]
    for frame in frames[1:]:
        result = result.join(frame, on="date", how="full", coalesce=True)
    return result.sort("date")


def load_space_weather() -> pl.DataFrame:
    return (
        pl.read_csv(SPACE_WEATHER_PATH)
        .with_columns(pl.col("DATE").str.to_date("%Y-%m-%d").alias("date"))
        .select("date", "F10.7_OBS_CENTER81", "AP_AVG", "KP_SUM")
    )


def load_co2() -> pl.DataFrame:
    schema = {
        "year": pl.Int32,
        "month": pl.Int32,
        "day": pl.Int32,
        "year_decimal": pl.Float32,
        "CO2_ppm": pl.Float64,
    }
    return (
        pl.read_csv(CO2_PATH, has_header=False, schema=schema, comment_prefix="#")
        .with_columns(pl.date("year", "month", "day").alias("date"))
        .select("date", "CO2_ppm")
        .with_columns(
            pl.when(pl.col("CO2_ppm") < 0)
            .then(None)
            .otherwise(pl.col("CO2_ppm"))
            .alias("CO2_ppm")
        )
    )


def as_date_index(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(pl.col("date").cast(pl.Date))
        .filter(pl.col("date").is_not_null())
        .unique(subset="date")
        .sort("date")
    )


def merge_with_drivers(target: pl.DataFrame, include_saber: bool) -> pl.DataFrame:
    sw = as_date_index(load_space_weather())
    co2 = as_date_index(load_co2())
    combined = target.join(sw, on="date", how="full", coalesce=True)
    combined = combined.join(co2, on="date", how="full", coalesce=True)
    if include_saber:
        saber = as_date_index(load_saber())
        combined = combined.join(saber, on="date", how="full", coalesce=True)
    return combined.sort("date")


def generate_global_mean() -> pl.DataFrame:
    df = pl.read_parquet(GLOBAL_MEAN_PATH).select("date", "log10rho_325")
    df = df.with_columns(
        pl.when(pl.col("log10rho_325") < -200)
        .then(None)
        .otherwise(pl.col("log10rho_325"))
        .interpolate()
        .alias("log10rho_325")
    )
    df = as_date_index(df)
    return merge_with_drivers(df, include_saber=False)


def generate_hasdm_density() -> pl.DataFrame:
    cols = []
    for alt in DENSITY_ALTITUDES:
        for metric in METRICS:
            cols.append(f"log10rho_{alt}_daily_{metric}")
    df = pl.read_parquet(HASDM_WIDE_PATH).select("date", *cols)
    df = as_date_index(df)
    return merge_with_drivers(df, include_saber=True)


def generate_msis_residuals() -> pl.DataFrame:
    cols = []
    for model in MSIS_MODELS:
        for alt in DENSITY_ALTITUDES:
            for metric in METRICS:
                cols.append(f"{model}_daily_{metric}_{alt}km")
    df = pl.read_parquet(MSIS_WIDE_PATH).select("date", *cols)
    df = as_date_index(df)
    return merge_with_drivers(df, include_saber=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating global mean input CSV...")
    gm = generate_global_mean()
    gm.write_csv(OUTPUT_DIR / "global_mean_causal_input.csv")
    print(f"  {len(gm)} rows, {gm.columns}")
    print(f"  date range: {gm['date'].min()} to {gm['date'].max()}")

    print("Generating HASDM density input CSV...")
    hd = generate_hasdm_density()
    hd.write_csv(OUTPUT_DIR / "hasdm_density_causal_input.csv")
    print(f"  {len(hd)} rows, {hd.columns}")
    print(f"  date range: {hd['date'].min()} to {hd['date'].max()}")

    print("Generating MSIS residuals input CSV...")
    mr = generate_msis_residuals()
    mr.write_csv(OUTPUT_DIR / "msis_residual_causal_input.csv")
    print(f"  {len(mr)} rows, {mr.columns}")
    print(f"  date range: {mr['date'].min()} to {mr['date'].max()}")

    print("Done.")


if __name__ == "__main__":
    main()
