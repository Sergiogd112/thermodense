"""
Figure 19 generator for NRLMSIS comparison.
Recreates the seasonal variation comparison figure from Emmert et al. 2020.
"""

import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional
from tqdm import tqdm


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
        Column name for MSISE-00 ln(ρ_mod/ρ_obs) values
    msis_20_col : str
        Column name for MSIS 2.0 ln(ρ_mod/ρ_obs) values
    msis_21_col : str, optional
        Column name for MSIS 2.1 ln(ρ_mod/ρ_obs) values. If None, MSIS 2.1 is not plotted.
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
        ("$\\bar{F}_{10.7}$ < 100", pl.col("f107a") < 100),
        (
            "100 < $\\bar{F}_{10.7}$ < 160",
            (pl.col("f107a") >= 100) & (pl.col("f107a") < 160),
        ),
        ("$\\bar{F}_{10.7}$ > 160", pl.col("f107a") >= 160),
    ]

    # Create figure
    if figsize is None:
        figsize = (12, 5 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True)

    # Ensure axes is 2D even for single row
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Color scheme matching the paper + blue for MSIS 2.1
    msis_00_color = "red"
    msis_20_color = "green"
    msis_21_color = "blue"
    matlab_color = "#EDB120"
    msis_00_marker = "x"
    msis_20_marker = "^"
    msis_21_marker = "o"
    matlab_marker = "s"

    # Process each mission (row)
    for row_idx, (df, mission_name) in tqdm(
        enumerate(zip(dfs, mission_names)),
        total=n_rows,
        desc="Processing missions for figure 19",
    ):
        # Build column selection list
        cols_to_select = ["timestamp", "f107a", msis_00_col, msis_20_col]
        max_alt = df.select(pl.col("Altitude (m)").max()).item()
        min_alt = df.select(pl.col("Altitude (m)").min()).item()
        timestart = df.select(pl.col("timestamp").min()).item()
        timeend = df.select(pl.col("timestamp").max()).item()
        if msis_21_col:
            cols_to_select.append(msis_21_col)
        has_matlab = matlab_col is not None and matlab_col in df.columns
        if has_matlab:
            cols_to_select.append(matlab_col)

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
                metric_cols = [
                    (msis_00_col, "msis_00"),
                    (msis_20_col, "msis_20"),
                ]
                if msis_21_col:
                    metric_cols.append((msis_21_col, "msis_21"))
                if has_matlab:
                    metric_cols.append((matlab_col, "matlab"))

                if errorbar_mode == "uncertainty_of_mean":
                    daily = filtered.group_by(["seasonal_bin", "date_bin"]).agg(
                        [pl.col(col).mean().alias(alias) for col, alias in metric_cols]
                    )
                    agg_exprs = []
                    for _, alias in metric_cols:
                        agg_exprs.extend(
                            [
                                pl.col(alias).mean().alias(f"{alias}_mean"),
                                (
                                    pl.col(alias).std().fill_null(0)
                                    / pl.col(alias).count().sqrt()
                                ).alias(f"{alias}_std"),
                            ]
                        )
                    agg_exprs.append(pl.col(metric_cols[0][1]).count().alias("count"))
                    binned = (
                        daily.group_by("seasonal_bin")
                        .agg(agg_exprs)
                        .sort("seasonal_bin")
                    )
                elif errorbar_mode == "raw_observation_uncertainty":
                    agg_exprs = []
                    for col, alias in metric_cols:
                        agg_exprs.extend(
                            [
                                pl.col(col).mean().alias(f"{alias}_mean"),
                                (
                                    pl.col(col).std().fill_null(0)
                                    / pl.col("date_bin").n_unique().sqrt()
                                ).alias(f"{alias}_std"),
                            ]
                        )
                    agg_exprs.append(pl.col(msis_00_col).count().alias("count"))
                    binned = (
                        filtered.group_by("seasonal_bin")
                        .agg(agg_exprs)
                        .sort("seasonal_bin")
                    )
                elif errorbar_mode == "paper":
                    agg_exprs = []
                    for col, alias in metric_cols:
                        agg_exprs.extend(
                            [
                                pl.col(col).mean().alias(f"{alias}_mean"),
                                (
                                    (pl.col(col).pow(2).mean()-pl.col(col).mean().pow(2))
                                ).sqrt().alias(f"{alias}_std"),
                            ]
                        )
                    agg_exprs.append(pl.col(msis_00_col).count().alias("count"))
                    binned = (
                        filtered.group_by("seasonal_bin")
                        .agg(agg_exprs)
                        .sort("seasonal_bin")
                    )
                
                else:
                    agg_exprs = []
                    for col, alias in metric_cols:
                        agg_exprs.extend(
                            [
                                pl.col(col).mean().alias(f"{alias}_mean"),
                                pl.col(col).std().fill_null(0).alias(f"{alias}_std"),
                            ]
                        )
                    agg_exprs.append(pl.col(msis_00_col).count().alias("count"))
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

                msis_00 = binned["msis_00_mean"].to_numpy()
                msis_00_std = binned["msis_00_std"].to_numpy()
                msis_20 = binned["msis_20_mean"].to_numpy()
                msis_20_std = binned["msis_20_std"].to_numpy()
                if has_matlab:
                    matlab_m = binned["matlab_mean"].to_numpy()
                    matlab_std = binned["matlab_std"].to_numpy()

                # Plot MSISE-00 (red crosses) with errorbars
                ax.errorbar(
                    x_vals,
                    msis_00,
                    yerr=msis_00_std,
                    marker=msis_00_marker,
                    color=msis_00_color,
                    linestyle="-",
                    linewidth=1,
                    markersize=6,
                    label="MSISE-00",
                    capsize=3,
                    elinewidth=1,
                )

                # Plot MSIS 2.0 (green triangles) with errorbars
                ax.errorbar(
                    x_vals,
                    msis_20,
                    yerr=msis_20_std,
                    marker=msis_20_marker,
                    color=msis_20_color,
                    linestyle="-",
                    linewidth=1,
                    markersize=6,
                    label="MSIS 2.0",
                    markerfacecolor="none",
                    capsize=3,
                    elinewidth=1,
                )

                # Plot MSIS 2.1 (blue circles) with errorbars if provided
                if msis_21_col:
                    msis_21 = binned["msis_21_mean"].to_numpy()
                    msis_21_std = binned["msis_21_std"].to_numpy()
                    ax.errorbar(
                        x_vals,
                        msis_21,
                        yerr=msis_21_std,
                        marker=msis_21_marker,
                        color=msis_21_color,
                        linestyle="-",
                        linewidth=1,
                        markersize=6,
                        label="MSIS 2.1",
                        markerfacecolor="none",
                        capsize=3,
                        elinewidth=1,
                    )

                if has_matlab:
                    ax.errorbar(
                        x_vals,
                        matlab_m,
                        yerr=matlab_std,
                        marker=matlab_marker,
                        color=matlab_color,
                        linestyle="-",
                        linewidth=1,
                        markersize=6,
                        label="MATLAB",
                        markerfacecolor="none",
                        capsize=3,
                        elinewidth=1,
                    )

            # Add zero line
            ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

            # Set axis limits
            ax.set_xlim(0, 360)
            min_y = -0.2
            max_y = 0.8
            ax.set_ylim(min_y, max_y)

            # Add ticks
            ax.set_xticks([0, 90, 180, 270, 360])
            ax.set_yticks(np.arange(min_y, max_y + 0.1, 0.1))

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
                    if " km," in mission_name
                    else f"{mission_name} {min_alt / 1e3:.0f}-{max_alt / 1e3:.0f} km, {timestart.year}-{timeend.year}"
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

    # Add legend to the bottom-right panel.
    legend_ax = axes[-1, -1]
    legend_ax.errorbar(
        [],
        [],
        yerr=[],
        marker=msis_00_marker,
        color=msis_00_color,
        linestyle="-",
        linewidth=1,
        markersize=6,
        label="MSISE-00",
        capsize=3,
    )
    legend_ax.errorbar(
        [],
        [],
        yerr=[],
        marker=msis_20_marker,
        color=msis_20_color,
        linestyle="-",
        linewidth=1,
        markersize=6,
        markerfacecolor="none",
        label="MSIS 2.0",
        capsize=3,
    )
    if msis_21_col:
        legend_ax.errorbar(
            [],
            [],
            yerr=[],
            marker=msis_21_marker,
            color=msis_21_color,
            linestyle="-",
            linewidth=1,
            markersize=6,
            markerfacecolor="none",
            label="MSIS 2.1",
            capsize=3,
        )
    if matlab_col:
        legend_ax.errorbar(
            [],
            [],
            yerr=[],
            marker=matlab_marker,
            color=matlab_color,
            linestyle="-",
            linewidth=1,
            markersize=6,
            markerfacecolor="none",
            label="MATLAB (MSISE-00)",
            capsize=3,
        )
    legend_ax.legend(loc="lower right", fontsize=10, frameon=True)

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
