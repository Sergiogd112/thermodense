"""
    Figure 19 generator for NRLMSIS comparison.
Recreates the seasonal variation comparison figure from Emmert et al. 2020.
"""

import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence
from tqdm import tqdm


@dataclass(frozen=True)
class ModelSeries:
    """One model log-density-ratio series to render in Figure 19."""

    column: str
    display_name: str
    color: str
    marker: str
    markerfacecolor: Optional[str] = None


def create_figure_19(
    dfs: List[pl.DataFrame],
    mission_names: List[str],
    msis_00_col: str = "ln_density_ratio_0",
    msis_20_col: str = "ln_density_ratio_2.0",
    msis_21_col: Optional[str] = None,
    matlab_col: Optional[str] = None,
    errorbar_mode: str = "uncertainty_of_mean",
    figsize: Tuple[float, float] = None,
    save_path: str = None,
    additional_model_series: Sequence[ModelSeries] = (),
) -> plt.Figure:
    """
    Recreate Figure 19 from Emmert et al. 2020 NRLMSIS 2.0 paper.

    Creates a grid of subplots with n rows (one per mission) and 3 columns
    (for different F10.7 ranges), showing ln(ρ_mod/ρ_obs) vs day of year
    using fixed 30.5-day seasonal bins.

    Parameters
    ----------
    dfs : List[pl.DataFrame]
        List of dataframes, one per mission. Each dataframe must contain:
        - 'timestamp': Datetime column
        - 'f107': F10.7 solar flux values
        - msis_00_col and msis_20_col: ln(ρ_mod/ρ_obs) columns
    mission_names : List[str]
        Names of the missions for row labels (e.g., ["TLE 1971-1985", "GOCE 2010-2013"])
    msis_00_col : str
        Column name for NRLMSISE-00 ln(ρ_mod/ρ_obs) values
    msis_20_col : str
        Column name for MSIS 2.0 ln(ρ_mod/ρ_obs) values
    msis_21_col : str, optional
        Column name for MSIS 2.1 ln(ρ_mod/ρ_obs) values. If None, MSIS 2.1 is not plotted.
    additional_model_series : Sequence[ModelSeries], optional
        Ordered extra model ln(ρ_mod/ρ_obs) series. Each series is included in
        aggregation, error bars, plotting, and the legend.
    errorbar_mode : str
        "uncertainty_of_mean" computes 1σ uncertainty from daily/bin means as
        std(daily_means) / sqrt(n_days), matching the paper convention.
        "raw_observation_uncertainty" computes std(all observations) / sqrt(n_days).
        "sample_std" uses the raw within-bin sample standard deviation.
    figsize : Tuple[float, float], optional
        Figure size. If None, defaults to (12, 3 * n_rows)
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    n_rows = len(dfs)
    n_cols = 3

    if len(mission_names) != n_rows:
        raise ValueError("Number of mission names must match number of dataframes")
    if errorbar_mode not in {
        "uncertainty_of_mean",
        "raw_observation_uncertainty",
        "paper",
        "sample_std",
    }:
        raise ValueError(
            "errorbar_mode must be 'uncertainty_of_mean', "
            "'raw_observation_uncertainty','paper' or 'sample_std'"
        )

    # 81 day average F10.7 ranges for columns and exclude years 2005 to 2009
    f107_ranges = [
        ("$\\bar{F}_{10.7}$ $<$ 100", pl.col("f107a") < 100),
        (
            "100 $<$ $\\bar{F}_{10.7}$ $<$ 160",
            (pl.col("f107a") >= 100) & (pl.col("f107a") < 160),
        ),
        ("$\\bar{F}_{10.7}$ $>$ 160", pl.col("f107a") >= 160),
    ]

    # Create figure
    if figsize is None:
        figsize = (12, 3 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True)

    # Ensure axes is 2D even for single row
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Color scheme matching the paper + blue for MSIS 2.1.
    model_series = [
        ModelSeries(msis_00_col, "NRLMSISE-00", "red", "x"),
        ModelSeries(msis_20_col, "NRLMSIS 2.0", "green", "^", "none"),
    ]
    if msis_21_col:
        model_series.append(
            ModelSeries(msis_21_col, "NRLMSIS 2.1", "blue", "o", "none")
        )

    # Process each mission (row)
    for row_idx, (df, mission_name) in tqdm(
        enumerate(zip(dfs, mission_names)),
        total=n_rows,
        desc="Processing missions for figure 19",
    ):
        has_matlab = matlab_col is not None and matlab_col in df.columns
        mission_model_series = model_series.copy()
        if has_matlab:
            mission_model_series.append(
                ModelSeries(matlab_col, "MATLAB (MSISE-00)", "#EDB120", "s", "none")
            )
        mission_model_series.extend(additional_model_series)
        series_with_aliases = [
            (series, f"model_{index}")
            for index, series in enumerate(mission_model_series)
        ]

        # Build column selection list
        cols_to_select = ["timestamp", "f107a"] + [
            series.column for series, _ in series_with_aliases
        ]
        max_alt = df.select(pl.col("Altitude (m)").max()).item()
        min_alt = df.select(pl.col("Altitude (m)").min()).item()
        timestart = df.select(pl.col("timestamp").min()).item()
        timeend = df.select(pl.col("timestamp").max()).item()
        # Sample if too large
        smaller_df = df.select(cols_to_select)
        # Use 12 fixed 30.5-day bins across the year, matching the paper.
        df_with_bins = smaller_df.with_columns(
            (
                ((pl.col("timestamp").dt.ordinal_day() - 1) / 30.5)
                .floor()
                .clip(0, 11)
                .cast(pl.Int64)
            ).alias("seasonal_bin"),
            pl.col("timestamp").dt.date().alias("date_bin"),
        )

        # Process each F10.7 range (column)
        for col_idx, (col_title, f107_filter) in enumerate(f107_ranges):
            ax = axes[row_idx, col_idx]

            # Filter by F10.7 range
            filtered = df_with_bins.filter(f107_filter)

            if len(filtered) > 0:
                if errorbar_mode == "uncertainty_of_mean":
                    daily = filtered.group_by(["seasonal_bin", "date_bin"]).agg(
                        [
                            pl.col(series.column).mean().alias(alias)
                            for series, alias in series_with_aliases
                        ]
                    )
                    agg_exprs = []
                    for _, alias in series_with_aliases:
                        agg_exprs.extend(
                            [
                                pl.col(alias).mean().alias(f"{alias}_mean"),
                                (
                                    pl.col(alias).std().fill_null(0)
                                    / pl.col(alias).count().sqrt()
                                ).alias(f"{alias}_std"),
                            ]
                        )
                    agg_exprs.append(
                        pl.col(series_with_aliases[0][1]).count().alias("count")
                    )
                    binned = (
                        daily.group_by("seasonal_bin")
                        .agg(agg_exprs)
                        .sort("seasonal_bin")
                    )
                elif errorbar_mode == "raw_observation_uncertainty":
                    agg_exprs = []
                    for series, alias in series_with_aliases:
                        agg_exprs.extend(
                            [
                                pl.col(series.column).mean().alias(f"{alias}_mean"),
                                (
                                    pl.col(series.column).std().fill_null(0)
                                    / pl.col("date_bin").n_unique().sqrt()
                                ).alias(f"{alias}_std"),
                            ]
                        )
                    agg_exprs.append(
                        pl.col(series_with_aliases[0][0].column).count().alias("count")
                    )
                    binned = (
                        filtered.group_by("seasonal_bin")
                        .agg(agg_exprs)
                        .sort("seasonal_bin")
                    )
                elif errorbar_mode == "paper":
                    agg_exprs = []
                    for series, alias in series_with_aliases:
                        agg_exprs.extend(
                            [
                                pl.col(series.column).mean().alias(f"{alias}_mean"),
                                (
                                    pl.col(series.column).pow(2).mean()
                                    - pl.col(series.column).mean().pow(2)
                                )
                                .sqrt()
                                .alias(f"{alias}_std"),
                            ]
                        )
                    agg_exprs.append(
                        pl.col(series_with_aliases[0][0].column).count().alias("count")
                    )
                    binned = (
                        filtered.group_by("seasonal_bin")
                        .agg(agg_exprs)
                        .sort("seasonal_bin")
                    )

                else:
                    agg_exprs = []
                    for series, alias in series_with_aliases:
                        agg_exprs.extend(
                            [
                                pl.col(series.column).mean().alias(f"{alias}_mean"),
                                pl.col(series.column)
                                .std()
                                .fill_null(0)
                                .alias(f"{alias}_std"),
                            ]
                        )
                    agg_exprs.append(
                        pl.col(series_with_aliases[0][0].column).count().alias("count")
                    )
                    binned = (
                        filtered.group_by("seasonal_bin")
                        .agg(agg_exprs)
                        .sort("seasonal_bin")
                    )
                # print(f"Mission: {mission_name}, F10.7 range: {col_title}")
                # display(binned)
                seasonal_bin = binned["seasonal_bin"].to_numpy()
                # Plot each bin at its 30.5-day midpoint on the day-of-year axis.
                x_vals = (seasonal_bin + 0.5) * 30.5

                for series, alias in series_with_aliases:
                    ax.errorbar(
                        x_vals,
                        binned[f"{alias}_mean"].to_numpy(),
                        yerr=binned[f"{alias}_std"].to_numpy(),
                        marker=series.marker,
                        color=series.color,
                        linestyle="-",
                        linewidth=1,
                        markersize=6,
                        label=series.display_name,
                        markerfacecolor=series.markerfacecolor,
                        capsize=3,
                        elinewidth=1,
                    )

            # Add zero line
            ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

            # Set axis limits
            ax.set_xlim(0, 360)
            min_y = -0.2
            max_y = 0.7
            ax.set_ylim(min_y, max_y)

            # Add ticks
            ax.set_xticks([0, 90, 180, 270, 360])
            ax.set_yticks(np.arange(min_y, max_y + 0.1, 0.2))

            # Panel label (a), (b), etc.
            panel_label = chr(ord("a") + row_idx * n_cols + col_idx)
            ax.text(
                0.15,
                0.85,
                f"({panel_label})",
                transform=ax.transAxes,
                fontsize=14,
                fontweight="bold",
                ha="center",
                va="center",
            )

            # Column titles (top row only)
            if row_idx == 0:
                ax.set_title(col_title, fontsize=12)

            # Row labels (right side)
            if col_idx == n_cols - 1:
                row_label = (
                    mission_name
                    if "\n" in mission_name or " km," in mission_name
                    else f"{mission_name}\n{min_alt / 1e3:.0f}-{max_alt / 1e3:.0f} km, {timestart.year}-{timeend.year}"
                )
                ax.text(
                    1.15,
                    0.5,
                    row_label,
                    transform=ax.transAxes,
                    fontsize=11,
                    rotation=90,
                    va="center",
                    ha="center",
                    linespacing=1.2,
                )

            # Grid
            ax.grid(True, which="both", linestyle=":", alpha=0.5)

    # Set common labels
    # X-axis label (bottom row only)
    for col_idx in range(n_cols):
        axes[-1, col_idx].set_xlabel("Day of Year", fontsize=11)

    # Y-axis label (leftmost column only)
    for row_idx in range(n_rows):
        axes[row_idx, 0].set_ylabel("ln($\\rho_{mod}$ / $\\rho_{obs}$)", fontsize=11)

    # Add legend to the GOCE high-F10.7 panel in the TuDelft layout; that panel
    # is empty and keeps the legend from covering Swarm-C data.
    legend_ax = axes[1, -1] if n_rows > 1 else axes[-1, -1]
    legend_model_series = model_series.copy()
    if matlab_col:
        legend_model_series.append(
            ModelSeries(matlab_col, "MATLAB (MSISE-00)", "#EDB120", "s", "none")
        )
    legend_model_series.extend(additional_model_series)
    for series in legend_model_series:
        legend_ax.errorbar(
            [],
            [],
            yerr=[],
            marker=series.marker,
            color=series.color,
            linestyle="-",
            linewidth=1,
            markersize=6,
            markerfacecolor=series.markerfacecolor,
            label=series.display_name,
            capsize=3,
        )
    legend_ax.legend(loc="center", fontsize=10, frameon=True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def create_figure_19_simple(
    df: pl.DataFrame,
    mission_name: str,
    msis_00_col: str = "ln_density_ratio_0",
    msis_20_col: str = "ln_density_ratio_2.0",
    msis_21_col: Optional[str] = None,
    save_path: str = None,
) -> plt.Figure:
    """
    Simplified version for a single dataframe/mission (1 row, 3 columns).
    """
    return create_figure_19(
        dfs=[df],
        mission_names=[mission_name],
        msis_00_col=msis_00_col,
        msis_20_col=msis_20_col,
        msis_21_col=msis_21_col,
        figsize=(12, 3.5),
        save_path=save_path,
    )


if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    n_samples = 5000

    # Create synthetic data
    timestamps = pl.datetime_range(
        start=pl.datetime(2005, 1, 1),
        end=pl.datetime(2005, 12, 31),
        interval="1h",
        eager=True,
    )

    # Take random subset
    timestamps = timestamps.sample(n_samples, with_replacement=True)

    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "f107": np.random.uniform(50, 200, n_samples),
            "ln_density_ratio_0": np.random.normal(0.15, 0.05, n_samples),
            "ln_density_ratio_2.0": np.random.normal(-0.05, 0.05, n_samples),
            "ln_density_ratio_2.1": np.random.normal(0.0, 0.03, n_samples),
        }
    )

    # Create figure without MSIS 2.1
    fig = create_figure_19_simple(df, "Example Mission 2005")
    plt.savefig("figure_19_example.png", dpi=150, bbox_inches="tight")
    print("Example figure (no 2.1) saved to figure_19_example.png")
    plt.close()

    # Create figure with MSIS 2.1
    fig = create_figure_19_simple(
        df, "Example Mission 2005", msis_21_col="ln_density_ratio_2.1"
    )
    plt.savefig("figure_19_example_with_21.png", dpi=150, bbox_inches="tight")
    print("Example figure (with 2.1) saved to figure_19_example_with_21.png")
