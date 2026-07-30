#!/usr/bin/env python3
"""Create composite heatmap and process-graph panels for completed causal discovery runs."""

import os
import re
from pathlib import Path

from scripts.pgf_config import configure_pgf, fig_size

configure_pgf()
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from tigramite import plotting as tp  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
LIGHTNING_DIR = Path(
    os.environ.get("THERMODENSE_LIGHTNING_RESULTS", "/tmp/opencode/lightning_results")
)
LIGHTNING_MSIS_LINK_DIR = Path(
    os.environ.get(
        "THERMODENSE_LIGHTNING_MSIS_LINKS", "/tmp/opencode/lightning_msis_links"
    )
)
KAGGLE_DIR = Path(os.environ.get("THERMODENSE_KAGGLE_RESULTS", "/tmp/opencode/gm_output"))
PARCORR_DIR = REPO_ROOT / "outputs/causal_discovery/parcorr"
PARCORR_210DAY_DIR = REPO_ROOT / "outputs/causal_discovery/parcorr-210day"
CMIKNN_DIR = REPO_ROOT / "outputs/causal_discovery/cmiknn99"
CMIKNN_TEN_DAY_DIR = REPO_ROOT / "outputs/causal_discovery/cmiknn-ten-day"
REPO_OUTPUTS_DIR = REPO_ROOT / "outputs/figures/results"
OUTPUT_DIR = Path(
    os.environ.get(
        "THERMODENSE_CAUSAL_FIGURE_DIR",
        REPO_ROOT / "outputs/figures/results/causal_composites",
    )
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_SOURCES = [
    PARCORR_DIR,
    PARCORR_210DAY_DIR,
    CMIKNN_DIR,
    CMIKNN_TEN_DAY_DIR,
    LIGHTNING_MSIS_LINK_DIR,
    LIGHTNING_DIR,
    KAGGLE_DIR,
    REPO_OUTPUTS_DIR,
]

VMIN = -1.0
VMAX = 1.0

DENSITY_ALTS = [325, 825]
SABER_ALTS = [100, 119, 139]
GEOMAG_LABELS = {"AP_AVG": "Ap", "KP_SUM": "Kp"}

VAR_ORDER = [
    "F10.7_OBS_CENTER81",
    "AP_AVG",
    "KP_SUM",
    "CO2_ppm",
    "saber_co2cool_100km",
    "saber_co2cool_119km",
    "saber_co2cool_139km",
    "log10rho_325",
    "log10rho_325_daily_mean",
    "log10rho_325_daily_range",
    "log10rho_825_daily_mean",
    "log10rho_825_daily_range",
    "surrogate_white_noise_1",
    "surrogate_white_noise_2",
]
MSIS_MODELS = [
    ("nrlmsis_2p0", "NRLMSIS 2.0"),
]
MSIS_ALTITUDES = [325, 825]
MSIS_SABER_ALTITUDE = 119


def expected_vars(run_id):
    """Construct the full variable list for a run_id, in canonical order."""
    vars = ["F10.7_OBS_CENTER81"]
    if "AP_AVG" in run_id:
        vars.append("AP_AVG")
    elif "KP_SUM" in run_id:
        vars.append("KP_SUM")
    vars.append("CO2_ppm")
    # SABER
    for salt in SABER_ALTS:
        if f"_{salt}km_" in run_id:
            vars.append(f"saber_co2cool_{salt}km")
            break
    # Density
    if run_id.startswith("global_mean"):
        vars.append("log10rho_325")
    elif run_id.startswith("hasdm_"):
        for dalt in DENSITY_ALTS:
            for metric in ["mean", "range"]:
                col = f"log10rho_{dalt}_daily_{metric}"
                if col in run_id:
                    vars.append(col)
                    break
    elif run_id.startswith("msis_mean_"):
        match = re.search(
            r"msis_mean_(nrlmsise_00|nrlmsis_2p0|nrlmsis_2p1)_daily_mean_(\d+)km",
            run_id,
        )
        if match:
            vars.append(f"{match.group(1)}_daily_mean_{match.group(2)}km")
    vars.append("surrogate_white_noise_1")
    vars.append("surrogate_white_noise_2")
    ordered = [v for v in VAR_ORDER if v in vars]
    ordered.extend(v for v in vars if v not in ordered and v not in VAR_ORDER)
    return ordered


VAR_SHORT = {
    "F10.7_OBS_CENTER81": r"$F_{10.7,81}$",
    "AP_AVG": r"$A_p$",
    "KP_SUM": r"$K_p$",
    "CO2_ppm": r"CO$_2$",
    "saber_co2cool_100km": r"CO$_2$ cl. 100 km",
    "saber_co2cool_119km": r"CO$_2$ cl. 119 km",
    "saber_co2cool_139km": r"CO$_2$ cl. 139 km",
    "log10rho_325": r"$\bar{\ell}_\rho$ 325 km",
    "log10rho_325_daily_mean": r"$\bar{\ell}_\rho$ 325 km",
    "log10rho_325_daily_range": r"$\Delta\ell_\rho$ 325 km",
    "log10rho_825_daily_mean": r"$\bar{\ell}_\rho$ 825 km",
    "log10rho_825_daily_range": r"$\Delta\ell_\rho$ 825 km",
    "surrogate_white_noise_1": "WN 1",
    "surrogate_white_noise_2": "WN 2",
}


def short_label(col):
    if col in VAR_SHORT:
        return VAR_SHORT[col]
    match = re.match(r"(nrlmsise_00|nrlmsis_2p0|nrlmsis_2p1)_daily_mean_(\d+)km", col)
    if match:
        model = {
            "nrlmsise_00": "00",
            "nrlmsis_2p0": "2.0",
            "nrlmsis_2p1": "2.1",
        }[match.group(1)]
        return rf"{model} err {match.group(2)} km"
    return col


def find_file(name):
    for src in ALL_SOURCES:
        p = src / name
        if p.exists():
            return p
        matches = list(src.rglob(name)) if src.exists() else []
        if matches:
            return matches[0]
    print("file not found")
    return None


def load_links(run_id, method="parcorr"):
    f = find_file(f"links_{method}_{run_id}.csv")
    if f is None:
        return None
    print(f)
    return pl.read_csv(f)


def build_mci_matrix_and_lags(df, all_vars, vmin=VMIN, vmax=VMAX):
    n = len(all_vars)
    matrix = np.full((n, n), np.nan)
    lag_labels = [["" for _ in all_vars] for _ in all_vars]
    link_types = [["" for _ in all_vars] for _ in all_vars]
    idx = {c: i for i, c in enumerate(all_vars)}
    for row in df.iter_rows(named=True):
        cause = row["cause"]
        target = row["target"]
        if cause not in idx or target not in idx:
            continue
        i = idx[cause]
        j = idx[target]
        value = float(row["mci_value"])
        if not np.isfinite(matrix[i, j]) or abs(value) > abs(matrix[i, j]):
            matrix[i, j] = value
            lag_value = float(row["lag_value"])
            lag_units = row.get("lag_units", "days")
            suffix = {"days": "d", "weeks": "w", "months": "mo"}.get(
                lag_units, lag_units
            )
            formatted_lag = (
                str(int(lag_value)) if lag_value.is_integer() else f"{lag_value:g}"
            )
            lag_labels[i][j] = f"{formatted_lag}{suffix}"
            link_types[i][j] = row["link_type"]
    return np.clip(matrix, vmin, vmax), lag_labels, link_types


def readable_text_color(rgba):
    luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return "black" if luminance > 0.48 else "white"


def plot_heatmap_subplot(
    ax,
    df,
    run_id,
    title="",
    show_xlabels=True,
    show_ylabels=True,
    vmin=VMIN,
    vmax=VMAX,
    cmap=None,
    decimals=2,
):
    all_vars = expected_vars(run_id)
    matrix, lag_labels, _ = build_mci_matrix_and_lags(df, all_vars, vmin, vmax)
    labels = [short_label(v) for v in all_vars]
    cmap = cmap or plt.cm.RdBu_r

    im = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(
        labels if show_xlabels else [], rotation=45, ha="right", fontsize=6
    )
    ax.set_yticklabels(labels if show_ylabels else [], fontsize=6)
    ax.set_xlabel("Target" if show_xlabels else "", fontsize=7)
    ax.set_ylabel("Cause" if show_ylabels else "", fontsize=7)
    if title:
        ax.set_title(title, fontsize=8)

    for i in range(len(all_vars)):
        for j in range(len(all_vars)):
            if np.isfinite(matrix[i, j]):
                rgba = cmap((matrix[i, j] - vmin) / (vmax - vmin))
                tc = readable_text_color(rgba)
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.{decimals}f}\n{lag_labels[i][j]}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color=tc,
                )
    return im


def plot_graph_subplot(
    ax,
    df,
    run_id,
    title="",
    vmin=VMIN,
    vmax=VMAX,
    cmap=None,
    min_edge=0.01,
):
    all_vars = expected_vars(run_id)
    matrix, lag_labels, link_types = build_mci_matrix_and_lags(
        df, all_vars, vmin, vmax
    )
    labels = [short_label(v) for v in all_vars]
    n = len(all_vars)
    cmap = cmap or plt.cm.RdBu_r
    norm = plt.Normalize(vmin, vmax)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2
    positions = np.column_stack([np.cos(angles), np.sin(angles)]) * 0.32
    node_radius = 0.045

    ax.set_xlim(-0.55, 0.55)
    ax.set_ylim(-0.55, 0.55)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=8, pad=2)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            val = matrix[i, j]
            if not np.isfinite(val) or abs(val) < min_edge:
                continue
            color = cmap(norm(val))
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            dist = np.hypot(dx, dy)
            if dist < 1e-6:
                continue
            ux, uy = dx / dist, dy / dist
            start = positions[i] + np.array([ux, uy]) * node_radius
            end = positions[j] - np.array([ux, uy]) * node_radius
            lw = 1.0 + 3.5 * abs(val)
            rad = 0.14 if i < j else -0.14
            unresolved = link_types[i][j] in {"o-o", "x-x"}
            arrow = mpatches.FancyArrowPatch(
                start,
                end,
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-" if unresolved else "->,head_length=6,head_width=4",
                color=color,
                linewidth=lw,
                linestyle="--" if unresolved else "-",
                zorder=2,
            )
            ax.add_patch(arrow)
            label_pos = (start + end) / 2
            label_pos += np.array([-uy, ux]) * rad * dist * 0.75
            ax.text(
                label_pos[0],
                label_pos[1],
                lag_labels[i][j],
                ha="center",
                va="center",
                fontsize=4.2,
                color="black",
                bbox=dict(facecolor="white", alpha=0.82, pad=0.45, edgecolor="none"),
                zorder=9,
            )

    for i in range(n):
        auto_mci = matrix[i, i] if np.isfinite(matrix[i, i]) else 0.0
        color = cmap(norm(auto_mci))
        circle = plt.Circle(
            positions[i], node_radius, color=color, zorder=5, ec="black", lw=0.7
        )
        ax.add_patch(circle)

    for i in range(n):
        direction = positions[i] / np.linalg.norm(positions[i])
        label_pos = positions[i] + direction * (node_radius + 0.06)
        ha = (
            "left"
            if direction[0] > 0.15
            else "right"
            if direction[0] < -0.15
            else "center"
        )
        va = (
            "bottom"
            if direction[1] > 0.15
            else "top"
            if direction[1] < -0.15
            else "center"
        )
        ax.text(
            label_pos[0],
            label_pos[1],
            labels[i],
            ha=ha,
            va=va,
            fontsize=5.5,
            bbox=dict(facecolor="white", alpha=0.75, pad=0.8, edgecolor="none"),
            zorder=10,
        )


def add_shared_colorbar(
    fig, label="Partial correlation", vmin=VMIN, vmax=VMAX, cmap=None
):
    cmap = cmap or plt.cm.RdBu_r
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
    sm.set_array([])
    cax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
    fig.colorbar(sm, cax=cax, label=label)


def build_time_series_matrices(df, all_vars, tau_max=7):
    """Reconstruct Tigramite graph matrices from the retained-link table."""
    n = len(all_vars)
    graph = np.full((n, n, tau_max + 1), "", dtype="<U3")
    values = np.zeros((n, n, tau_max + 1), dtype=float)
    idx = {name: index for index, name in enumerate(all_vars)}
    reverse_types = {
        "-->": "<--",
        "<--": "-->",
        "o-o": "o-o",
        "x-x": "x-x",
        "o->": "<-o",
        "<-o": "o->",
        "x->": "<-x",
        "<-x": "x->",
    }

    for row in df.iter_rows(named=True):
        cause = row["cause"]
        target = row["target"]
        lag = int(row["lag_index"])
        if cause not in idx or target not in idx or lag > tau_max:
            continue
        i, j = idx[cause], idx[target]
        link_type = str(row["link_type"])
        value = float(row["mci_value"])
        graph[i, j, lag] = link_type
        values[i, j, lag] = value
        if lag == 0:
            graph[j, i, lag] = reverse_types.get(link_type, link_type)
            values[j, i, lag] = value
    return graph, values


def make_appendix_time_series_graphs():
    run_specs = [
        (
            "global_mean_log10rho_325_AP_AVG",
            r"Global mean 325 km -- $A_p$",
            "timeseries_global_mean_Ap.pdf",
        ),
        (
            "global_mean_log10rho_325_KP_SUM",
            r"Global mean 325 km -- $K_p$",
            "timeseries_global_mean_Kp.pdf",
        ),
        (
            "hasdm_log10rho_325_daily_mean_119km_AP_AVG",
            r"HASDM $\bar{\ell}_\rho$ 325 km -- $A_p$, SABER 119 km",
            "timeseries_hasdm_mean_325_Ap.pdf",
        ),
        (
            "hasdm_log10rho_825_daily_mean_119km_AP_AVG",
            r"HASDM $\bar{\ell}_\rho$ 825 km -- $A_p$, SABER 119 km",
            "timeseries_hasdm_mean_825_Ap.pdf",
        ),
        (
            "hasdm_log10rho_325_daily_range_119km_AP_AVG",
            r"HASDM $\Delta\ell_\rho$ 325 km -- $A_p$, SABER 119 km",
            "timeseries_hasdm_range_325_Ap.pdf",
        ),
        (
            "hasdm_log10rho_825_daily_range_119km_AP_AVG",
            r"HASDM $\Delta\ell_\rho$ 825 km -- $A_p$, SABER 119 km",
            "timeseries_hasdm_range_825_Ap.pdf",
        ),
        (
            "msis_mean_nrlmsis_2p0_daily_mean_325km_119km_AP_AVG",
            r"NRLMSIS 2.0 error 325 km -- $A_p$, SABER 119 km",
            "timeseries_msis_mean_325_Ap.pdf",
        ),
        (
            "msis_mean_nrlmsis_2p0_daily_mean_825km_119km_AP_AVG",
            r"NRLMSIS 2.0 error 825 km -- $A_p$, SABER 119 km",
            "timeseries_msis_mean_825_Ap.pdf",
        ),
    ]

    for run_id, title, filename in run_specs:
        df = load_links(run_id)
        if df is None:
            print(f"  SKIP expanded graph {run_id} (not found)")
            continue
        all_vars = expected_vars(run_id)
        graph, values = build_time_series_matrices(df, all_vars)
        fig, ax = plt.subplots(figsize=(7.1, 4.7), constrained_layout=False)
        fig.subplots_adjust(left=0.17, right=0.98, top=0.86, bottom=0.13)
        tp.plot_time_series_graph(
            graph=graph,
            val_matrix=values,
            var_names=[short_label(name) for name in all_vars],
            fig_ax=(fig, ax),
            link_colorbar_label="Partial correlation",
            arrow_linewidth=1.6,
            arrowhead_size=8,
            node_size=0.08,
            label_fontsize=7,
            tick_label_size=6,
            vmin_edges=VMIN,
            vmax_edges=VMAX,
            cmap_edges="RdBu_r",
        )
        for patch in ax.patches:
            if isinstance(patch, mpatches.Ellipse):
                patch.set_edgecolor("black")
                patch.set_linewidth(0.5)
        fig.suptitle(title, fontsize=9, y=0.98)
        fig.savefig(OUTPUT_DIR / filename, bbox_inches="tight")
        plt.close(fig)
        print(f"  expanded graph {run_id}: done")


def make_global_mean_composites(
    method="parcorr", output_tag=None, title_suffix=""
):
    file_tag = f"_{output_tag}" if output_tag else ""
    for geomag_col, geomag_label in GEOMAG_LABELS.items():
        run_id = f"global_mean_log10rho_325_{geomag_col}"
        df = load_links(run_id, method=method)
        if df is None:
            print(f"SKIP {run_id} (not found)")
            continue

        # Heatmap
        fig, ax = plt.subplots(figsize=(4.5, 3.5), constrained_layout=False)
        fig.subplots_adjust(left=0.18, right=0.82, top=0.88, bottom=0.22)
        plot_heatmap_subplot(
            ax, df, run_id, f"Global mean 325 km — {geomag_label}{title_suffix}"
        )
        add_shared_colorbar(fig)
        fig.savefig(
            OUTPUT_DIR / f"heatmap{file_tag}_global_mean_{geomag_label}.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)

        # Graph
        fig, ax = plt.subplots(figsize=(4.5, 4), constrained_layout=False)
        fig.subplots_adjust(left=0.05, right=0.82, top=0.92, bottom=0.05)
        plot_graph_subplot(
            ax, df, run_id, f"Global mean 325 km — {geomag_label}{title_suffix}"
        )
        add_shared_colorbar(fig)
        fig.savefig(
            OUTPUT_DIR / f"graph{file_tag}_global_mean_{geomag_label}.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)

        print(f"  global_mean {geomag_label}: done")


def make_hasdm_composites(
    metric, method="parcorr", output_tag=None, title_suffix=""
):
    file_tag = f"_{output_tag}" if output_tag else ""
    metric_col = "mean" if metric == "mean" else "range"
    metric_label = (
        r"$\bar{\ell}_\rho$" if metric == "mean" else r"$\Delta\ell_\rho$"
    )
    metric_prefix = "mean" if metric == "mean" else "range"

    for geomag_col, geomag_label in GEOMAG_LABELS.items():
        plots = {}
        any_found = False
        for salt in SABER_ALTS:
            for dalt in DENSITY_ALTS:
                dcol = f"log10rho_{dalt}_daily_{metric_col}"
                run_id = f"hasdm_{dcol}_{salt}km_{geomag_col}"
                df = load_links(run_id, method=method)
                if df is None:
                    print(f"  SKIP {run_id} (not found)")
                    continue
                any_found = True
                plots[(salt, dalt)] = df

        if not any_found:
            print(f"  No data for HASDM {metric} {geomag_label}")
            continue

        n_rows, n_cols = len(SABER_ALTS), len(DENSITY_ALTS)

        # Heatmap file
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(6, 8), constrained_layout=False
        )
        fig.subplots_adjust(
            left=0.08, right=0.86, top=0.91, bottom=0.02, hspace=0.13, wspace=0.1
        )
        fig.suptitle(
            f"HASDM {metric_label} — {geomag_label}{title_suffix}", fontsize=11
        )

        for i, salt in enumerate(SABER_ALTS):
            for j, dalt in enumerate(DENSITY_ALTS):
                ax = axes[i, j]
                key = (salt, dalt)
                if key in plots:
                    dcol = f"log10rho_{dalt}_daily_{metric_col}"
                    run_id = f"hasdm_{dcol}_{salt}km_{geomag_col}"
                    plot_heatmap_subplot(
                        ax,
                        plots[key],
                        run_id,
                        f"SABER {salt} km, {dalt} km",
                        show_xlabels=i == n_rows - 1,
                        show_ylabels=j == 0,
                    )
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "N/A",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=10,
                        color="gray",
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])

        add_shared_colorbar(fig)
        fig.savefig(
            OUTPUT_DIR
            / f"heatmap{file_tag}_hasdm_{metric_prefix}_{geomag_label}.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)

        # Graph file
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(6, 8), constrained_layout=False
        )
        fig.subplots_adjust(
            left=0.04, right=0.86, top=0.91, bottom=0.02, hspace=0.12, wspace=0.05
        )
        fig.suptitle(
            f"HASDM {metric_label} — {geomag_label}{title_suffix}", fontsize=11
        )

        for i, salt in enumerate(SABER_ALTS):
            for j, dalt in enumerate(DENSITY_ALTS):
                ax = axes[i, j]
                key = (salt, dalt)
                if key in plots:
                    dcol = f"log10rho_{dalt}_daily_{metric_col}"
                    run_id = f"hasdm_{dcol}_{salt}km_{geomag_col}"
                    plot_graph_subplot(
                        ax, plots[key], run_id, f"SABER {salt} km, {dalt} km"
                    )
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "N/A",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=10,
                        color="gray",
                    )
                    ax.set_title(f"SABER {salt} km, {dalt} km", fontsize=8)
                    ax.set_xticks([])
                    ax.set_yticks([])

        add_shared_colorbar(fig)
        fig.savefig(
            OUTPUT_DIR / f"graph{file_tag}_hasdm_{metric_prefix}_{geomag_label}.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)

        print(f"  HASDM {metric} {geomag_label}: done ({len(plots)} subplots)")


def make_msis_residual_composites(
    method="parcorr", output_tag=None, title_suffix=""
):
    file_tag = f"_{output_tag}" if output_tag else ""
    plots = {}
    for model_slug, _ in MSIS_MODELS:
        for altitude in MSIS_ALTITUDES:
            run_id = (
                f"msis_mean_{model_slug}_daily_mean_"
                f"{altitude}km_{MSIS_SABER_ALTITUDE}km_AP_AVG"
            )
            df = load_links(run_id, method=method)
            if df is None:
                print(f"  SKIP {run_id} (not found)")
                continue
            plots[(model_slug, altitude)] = (run_id, df)

    if any(
        (model, altitude) not in plots
        for model, _ in MSIS_MODELS
        for altitude in MSIS_ALTITUDES
    ):
        print("  MSIS residual composite incomplete; skipping")
        return

    n_rows, n_cols = len(MSIS_MODELS), len(MSIS_ALTITUDES)
    for kind, plotter, filename in [
        (
            "heatmap",
            plot_heatmap_subplot,
            f"heatmap{file_tag}_msis_mean_Ap.pdf",
        ),
        ("graph", plot_graph_subplot, f"graph{file_tag}_msis_mean_Ap.pdf"),
    ]:
        figsize = (6, 3.2) if kind == "heatmap" else fig_size(1.0, 0.76)
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=figsize, constrained_layout=False
        )
        axes = np.asarray(axes).reshape(n_rows, n_cols)
        fig.subplots_adjust(
            left=0.08,
            right=0.84,
            top=0.88,
            bottom=0.11 if kind == "heatmap" else 0.05,
            hspace=0.34 if kind == "heatmap" else 0.18,
            wspace=0.22 if kind == "heatmap" else 0.05,
        )
        fig.suptitle(
            f"MSIS residual daily mean — Ap, SABER {MSIS_SABER_ALTITUDE} km"
            f"{title_suffix}",
            fontsize=10,
        )
        for row, (model_slug, model_label) in enumerate(MSIS_MODELS):
            for col, altitude in enumerate(MSIS_ALTITUDES):
                run_id, df = plots[(model_slug, altitude)]
                title = f"{model_label}, {altitude} km"
                if kind == "heatmap":
                    plotter(
                        axes[row, col],
                        df,
                        run_id,
                        title,
                        show_xlabels=row == n_rows - 1,
                        show_ylabels=col == 0,
                    )
                else:
                    plotter(axes[row, col], df, run_id, title)
        add_shared_colorbar(fig)
        fig.savefig(OUTPUT_DIR / filename, bbox_inches="tight")
        plt.close(fig)
        print(f"  MSIS residual {kind}: done")


def make_cmiknn_composite(
    run_specs,
    title,
    filename_prefix,
    n_rows,
    n_cols,
    method="cmiknn",
    output_tag="cmiknn",
):
    plots = []
    for run_id, panel_title in run_specs:
        df = load_links(run_id, method=method)
        if df is None:
            print(f"  CMIknn composite incomplete: missing {run_id}")
            return
        plots.append((run_id, panel_title, df))

    cmi_vmin, cmi_vmax = 0.0, 0.30
    cmi_cmap = plt.cm.viridis
    for kind, filename in [
        ("heatmap", f"heatmap_{output_tag}_{filename_prefix}.pdf"),
        ("graph", f"graph_{output_tag}_{filename_prefix}.pdf"),
    ]:
        figsize = (6.5, 3.6 * n_rows) if kind == "heatmap" else (6.5, 3.5 * n_rows)
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=figsize, constrained_layout=False
        )
        axes = np.asarray(axes).reshape(n_rows, n_cols)
        fig.subplots_adjust(
            left=0.08,
            right=0.86,
            top=0.88,
            bottom=0.13 if kind == "heatmap" else 0.05,
            hspace=0.30,
            wspace=0.18 if kind == "heatmap" else 0.06,
        )
        fig.suptitle(title, fontsize=10)
        for index, (run_id, panel_title, df) in enumerate(plots):
            row, col = divmod(index, n_cols)
            ax = axes[row, col]
            if kind == "heatmap":
                plot_heatmap_subplot(
                    ax,
                    df,
                    run_id,
                    panel_title,
                    show_xlabels=row == n_rows - 1,
                    show_ylabels=col == 0,
                    vmin=cmi_vmin,
                    vmax=cmi_vmax,
                    cmap=cmi_cmap,
                    decimals=3,
                )
            else:
                plot_graph_subplot(
                    ax,
                    df,
                    run_id,
                    panel_title,
                    vmin=cmi_vmin,
                    vmax=cmi_vmax,
                    cmap=cmi_cmap,
                    min_edge=0.0,
                )
        add_shared_colorbar(
            fig,
            label="Conditional mutual information",
            vmin=cmi_vmin,
            vmax=cmi_vmax,
            cmap=cmi_cmap,
        )
        fig.savefig(OUTPUT_DIR / filename, bbox_inches="tight")
        plt.close(fig)
        print(f"  CMIknn {filename_prefix} {kind}: done")


def make_cmiknn_composites():
    make_cmiknn_composite(
        [
            ("global_mean_log10rho_325_AP_AVG", r"Global mean 325 km, $A_p$"),
            ("global_mean_log10rho_325_KP_SUM", r"Global mean 325 km, $K_p$"),
        ],
        "Global mean CMIknn robustness",
        "global_mean",
        1,
        2,
    )
    make_cmiknn_composite(
        [
            (
                "hasdm_log10rho_325_daily_mean_119km_AP_AVG",
                r"$\bar{\ell}_\rho$ 325 km",
            ),
            (
                "hasdm_log10rho_825_daily_mean_119km_AP_AVG",
                r"$\bar{\ell}_\rho$ 825 km",
            ),
            (
                "hasdm_log10rho_325_daily_range_119km_AP_AVG",
                r"$\Delta\ell_\rho$ 325 km",
            ),
            (
                "hasdm_log10rho_825_daily_range_119km_AP_AVG",
                r"$\Delta\ell_\rho$ 825 km",
            ),
        ],
        r"HASDM CMIknn robustness -- $A_p$, SABER 119 km",
        "hasdm",
        2,
        2,
    )
    make_cmiknn_composite(
        [
            (
                "msis_mean_nrlmsis_2p0_daily_mean_325km_119km_AP_AVG",
                "NRLMSIS 2.0 error, 325 km",
            ),
            (
                "msis_mean_nrlmsis_2p0_daily_mean_825km_119km_AP_AVG",
                "NRLMSIS 2.0 error, 825 km",
            ),
        ],
        r"NRLMSIS 2.0 error CMIknn robustness -- $A_p$, SABER 119 km",
        "msis_mean",
        1,
        2,
    )


def make_cmiknn_ten_day_composites():
    common = {"method": "cmiknn_ten_day", "output_tag": "cmiknn_ten_day"}
    make_cmiknn_composite(
        [
            ("global_mean_log10rho_325_AP_AVG", r"Global mean 325 km, $A_p$"),
            ("global_mean_log10rho_325_KP_SUM", r"Global mean 325 km, $K_p$"),
        ],
        "Global mean 10-day CMIknn robustness",
        "global_mean",
        1,
        2,
        **common,
    )
    make_cmiknn_composite(
        [
            (
                "hasdm_log10rho_325_daily_mean_119km_AP_AVG",
                r"$\bar{\ell}_\rho$ 325 km",
            ),
            (
                "hasdm_log10rho_825_daily_mean_119km_AP_AVG",
                r"$\bar{\ell}_\rho$ 825 km",
            ),
            (
                "hasdm_log10rho_325_daily_range_119km_AP_AVG",
                r"$\Delta\ell_\rho$ 325 km",
            ),
            (
                "hasdm_log10rho_825_daily_range_119km_AP_AVG",
                r"$\Delta\ell_\rho$ 825 km",
            ),
        ],
        r"HASDM 10-day CMIknn robustness -- $A_p$, SABER 119 km",
        "hasdm",
        2,
        2,
        **common,
    )
    make_cmiknn_composite(
        [
            (
                "msis_mean_nrlmsis_2p0_daily_mean_325km_119km_AP_AVG",
                "NRLMSIS 2.0 error, 325 km",
            ),
            (
                "msis_mean_nrlmsis_2p0_daily_mean_825km_119km_AP_AVG",
                "NRLMSIS 2.0 error, 825 km",
            ),
        ],
        r"NRLMSIS 2.0 error 10-day CMIknn -- $A_p$, SABER 119 km",
        "msis_mean",
        1,
        2,
        **common,
    )


def make_parcorr_210day_composites():
    common = {
        "method": "parcorr_210day",
        "output_tag": "parcorr_210day",
        "title_suffix": " — 0--210 d",
    }
    make_global_mean_composites(**common)
    make_hasdm_composites("mean", **common)
    make_hasdm_composites("range", **common)
    make_msis_residual_composites(**common)


def main():
    print("=== Creating composites ===")
    print("Global mean:")
    make_global_mean_composites()
    print("HASDM mean:")
    make_hasdm_composites("mean")
    print("HASDM range:")
    make_hasdm_composites("range")
    print("MSIS residual mean:")
    make_msis_residual_composites()
    print("CMIknn robustness:")
    make_cmiknn_composites()
    print("10-day CMIknn robustness:")
    make_cmiknn_ten_day_composites()
    print("210-day ParCorr robustness:")
    make_parcorr_210day_composites()
    print("Appendix lag-resolved graphs:")
    make_appendix_time_series_graphs()
    print(f"\nDone. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
