from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

import matplotlib
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from figure_generator import create_figure_19


CHAMP_PATH = Path("data/analyzed/tudelft/champ/CH_analyzed.parquet")
GOCE_PATH = Path("data/analyzed/tudelft/goce/GO_analyzed.parquet")
OUTPUT_DIR = Path("outputs/figure19_compare")


def load_champ(*, exclude_2005: bool) -> pl.DataFrame:
    solar_minimum_start = date(2005, 1, 1) if exclude_2005 else date(2006, 1, 1)
    return (
        pl.read_parquet(CHAMP_PATH)
        .sort("timestamp")
        .filter(
            ((pl.col("Altitude (m)") / 1000).is_between(300, 500))
            & (pl.col("timestamp") >= date(2001, 1, 1))
            & (
                (pl.col("timestamp") < solar_minimum_start)
                | (pl.col("timestamp") > date(2009, 12, 31))
            )
            & (pl.col("Anomalus Density (kg/m^3)") == 0)
        )
    )


def load_goce() -> pl.DataFrame:
    return (
        pl.read_parquet(GOCE_PATH)
        .sort("timestamp")
        .filter(
            ((pl.col("Altitude (m)") / 1000).is_between(225, 300))
            & (pl.col("Anomalus Density (kg/m^3)") == 0)
            & (pl.col("Degraded Flag Thrusters") == 0)
        )
    )


def save_comparison(
    *,
    output_name: str,
    champ_df: pl.DataFrame,
    goce_df: pl.DataFrame,
    errorbar_mode: str,
    title: str,
) -> None:
    fig = create_figure_19(
        dfs=[champ_df, goce_df],
        mission_names=[
            "CHAMP 300-500 km, 2001-2005, 2010",
            "GOCE 225-300 km, 2009-2013",
        ],
        msis_00_col="ln_density_ratio_0",
        msis_20_col="ln_density_ratio_2.0",
        msis_21_col=None,
        matlab_col="ln_density_ratio_matlab_density",
        errorbar_mode=errorbar_mode,
        figsize=(12, 7),
    )
    fig.suptitle(title, fontsize=14, y=1.02)
    output_path = OUTPUT_DIR / output_name
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(output_path)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    goce_df = load_goce()
    champ_caption_df = load_champ(exclude_2005=False)
    champ_section_df = load_champ(exclude_2005=True)

    save_comparison(
        output_name="champ_goce_daily_mean_sem_caption_2006_2009.png",
        champ_df=champ_caption_df,
        goce_df=goce_df,
        errorbar_mode="uncertainty_of_mean",
        title="CHAMP/GOCE Figure 19: daily-mean SEM, CHAMP excludes 2006-2009",
    )
    save_comparison(
        output_name="champ_goce_daily_mean_sem_section_2005_2009.png",
        champ_df=champ_section_df,
        goce_df=goce_df,
        errorbar_mode="uncertainty_of_mean",
        title="CHAMP/GOCE Figure 19: daily-mean SEM, CHAMP excludes 2005-2009",
    )
    save_comparison(
        output_name="champ_goce_raw_observation_sem_caption_2006_2009.png",
        champ_df=champ_caption_df,
        goce_df=goce_df,
        errorbar_mode="raw_observation_uncertainty",
        title="CHAMP/GOCE Figure 19: raw-observation SEM, CHAMP excludes 2006-2009",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
