from __future__ import annotations
# Ruff: configure_pgf() must run before pyplot imports; suppress intentional E402.
# ruff: noqa: E402

import csv
from dataclasses import dataclass
from pathlib import Path

from scripts.pgf_config import configure_pgf

configure_pgf()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
from matplotlib.patches import FancyArrowPatch

ROOT = Path("outputs/figures/results/set_hasdm/causal/gpdctorch")
OUT_DIR = ROOT / "hasdm_msis_residuals_mean_composites"
HASDM_COMPOSITE_DIR = ROOT / "hasdm_density_composites"
GLOBAL_ROOT = Path("outputs/figures/results/global_mean/causal/gpdctorch")
GLOBAL_COMPOSITE_DIR = GLOBAL_ROOT / "altitudes_325" / "daily_7d_composites"

MODELS = [
    ("nrlmsise_00", "NRLMSISE-00"),
    ("nrlmsis_2p0", "NRLMSIS 2.0"),
    ("nrlmsis_2p1", "NRLMSIS 2.1"),
]
ALTITUDES = [500, 825]
VARIABLE_ORDER = [
    "F10.7_OBS_CENTER81",
    "AP_AVG",
    "CO2_ppm",
    "saber_co2cool_139km",
]
SURROGATES = ["surrogate_white_noise_1", "surrogate_white_noise_2"]
HASDM_PANELS = [
    (
        "maunaloa_hasdm_mean_500_saber_139",
        "Density mean",
        "log10rho_500_daily_mean",
    ),
    (
        "maunaloa_hasdm_range_500_saber_139",
        "Density range",
        "log10rho_500_daily_range",
    ),
]
GLOBAL_DATASET = "altitudes_325"
GLOBAL_TARGET = "log10rho_325"
GLOBAL_VARIABLES = [
    "F10.7_OBS_CENTER81",
    "AP_AVG",
    "CO2_ppm",
    GLOBAL_TARGET,
    *SURROGATES,
]


@dataclass(frozen=True)
class Link:
    cause: str
    cause_label: str
    target: str
    target_label: str
    lag: int
    link_type: str
    value: float


def run_dir(model_slug: str, altitude: int) -> Path:
    return (
        ROOT
        / f"maunaloa_hasdm_msis_residuals_mean_{model_slug}_{altitude}_saber_139"
        / "daily_7d"
    )


def links_path(model_slug: str, altitude: int) -> Path:
    dataset = f"maunaloa_hasdm_msis_residuals_mean_{model_slug}_{altitude}_saber_139"
    return (
        run_dir(model_slug, altitude)
        / f"links_gpdctorch_detrended_anomaly_{dataset}_daily_7d.csv"
    )


def hasdm_links_path(dataset: str) -> Path:
    return (
        ROOT
        / dataset
        / "daily_7d"
        / f"links_gpdctorch_detrended_anomaly_{dataset}_daily_7d.csv"
    )


def global_links_path() -> Path:
    return (
        GLOBAL_ROOT
        / GLOBAL_DATASET
        / "daily_7d"
        / f"links_gpdctorch_detrended_anomaly_{GLOBAL_DATASET}_daily_7d.csv"
    )


def load_links(model_slug: str, altitude: int) -> list[Link]:
    path = links_path(model_slug, altitude)
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            Link(
                cause=row["cause"],
                cause_label=row["cause_label"],
                target=row["target"],
                target_label=row["target_label"],
                lag=int(row["lag_value"]),
                link_type=row["link_type"],
                value=max(float(row["mci_value"]), 0.0),
            )
            for row in csv.DictReader(handle)
        ]


def load_links_file(path: Path) -> list[Link]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            Link(
                cause=row["cause"],
                cause_label=row["cause_label"],
                target=row["target"],
                target_label=row["target_label"],
                lag=int(row["lag_value"]),
                link_type=row["link_type"],
                value=max(float(row["mci_value"]), 0.0),
            )
            for row in csv.DictReader(handle)
        ]


def residual_col(model_slug: str, altitude: int) -> str:
    return f"{model_slug}_daily_mean_{altitude}km"


def variables_for_panel(
    model_slug: str, altitude: int, include_surrogates: bool
) -> list[str]:
    variables = [*VARIABLE_ORDER, residual_col(model_slug, altitude)]
    if include_surrogates:
        variables.extend(SURROGATES)
    return variables


def label_for(variable: str, model_label: str, altitude: int) -> str:
    if variable == "F10.7_OBS_CENTER81":
        return "F$_{10.7,81}$"
    if variable == "AP_AVG":
        return "$A_p$"
    if variable == "CO2_ppm":
        return "CO$_2$"
    if variable == "saber_co2cool_139km":
        return "SABER CO$_2$\ncooling\n139 km"
    if variable == "surrogate_white_noise_1":
        return "White\nnoise 1"
    if variable == "surrogate_white_noise_2":
        return "White\nnoise 2"
    if variable == residual_col(model_label_to_slug(model_label), altitude):
        short_model = {
            "NRLMSISE-00": "00",
            "NRLMSIS 2.0": "2.0",
            "NRLMSIS 2.1": "2.1",
        }[model_label]
        return f"{short_model} err at\n{altitude}km"
    return variable.replace("_", "\n")


def hasdm_label_for(variable: str) -> str:
    if variable == "F10.7_OBS_CENTER81":
        return "F$_{10.7,81}$"
    if variable == "AP_AVG":
        return "$A_p$"
    if variable == "CO2_ppm":
        return "CO$_2$"
    if variable == "saber_co2cool_139km":
        return "CO$_2$ cooling\n139 km"
    if variable == "log10rho_500_daily_mean":
        return "$\\bar{\\ell}_\\rho$\n500 km"
    if variable == "log10rho_500_daily_range":
        return "$\\Delta\\ell_\\rho$\n500 km"
    if variable == "surrogate_white_noise_1":
        return "White\nnoise 1"
    if variable == "surrogate_white_noise_2":
        return "White\nnoise 2"
    return variable.replace("_", "\n")


def global_label_for(variable: str) -> str:
    if variable == "F10.7_OBS_CENTER81":
        return "F$_{10.7,81}$"
    if variable == "AP_AVG":
        return "$A_p$"
    if variable == "CO2_ppm":
        return "CO$_2$"
    if variable == GLOBAL_TARGET:
        return "$\\bar{\\ell}_\\rho$\n325 km"
    if variable == "surrogate_white_noise_1":
        return "White\nnoise 1"
    if variable == "surrogate_white_noise_2":
        return "White\nnoise 2"
    return variable.replace("_", "\n")


def model_label_to_slug(model_label: str) -> str:
    for slug, label in MODELS:
        if label == model_label:
            return slug
    raise ValueError(model_label)


def aggregate_pair_links(
    links: list[Link], variables: list[str]
) -> dict[tuple[str, str], tuple[float, list[int]]]:
    allowed = set(variables)
    grouped: dict[tuple[str, str], list[Link]] = {}
    for link in links:
        if link.cause not in allowed or link.target not in allowed:
            continue
        grouped.setdefault((link.cause, link.target), []).append(link)

    aggregated: dict[tuple[str, str], tuple[float, list[int]]] = {}
    for pair, pair_links in grouped.items():
        strongest = max(link.value for link in pair_links)
        lags = sorted({link.lag for link in pair_links})
        aggregated[pair] = strongest, lags
    return aggregated


def lag_text(lags: list[int]) -> str:
    if not lags:
        return ""
    return ",".join(str(lag) for lag in lags)


def readable_text_color(rgba: tuple[float, float, float, float]) -> str:
    red, green, blue, _ = rgba
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "black" if luminance > 0.48 else "white"


def panel_title(row: int, col: int) -> str:
    pieces = []
    if row == 0:
        pieces.append(f"{ALTITUDES[col]} km")
    if col == 0:
        pieces.append(MODELS[row][1])
    return " | ".join(pieces)


def global_heatmap_vmax() -> float:
    values = []
    for model_slug, _ in MODELS:
        for altitude in ALTITUDES:
            variables = variables_for_panel(
                model_slug, altitude, include_surrogates=False
            )
            links = load_links(model_slug, altitude)
            values.extend(
                value for value, _ in aggregate_pair_links(links, variables).values()
            )
    return max(max(values), 0.01)


def plot_heatmap_composite() -> None:
    vmax = global_heatmap_vmax()
    cmap = plt.get_cmap("viridis")
    norm = colors.Normalize(vmin=0.0, vmax=vmax)
    fig, axes = plt.subplots(3, 2, figsize=(8.4, 11.6), constrained_layout=False)

    for row, (model_slug, model_label) in enumerate(MODELS):
        for col, altitude in enumerate(ALTITUDES):
            ax = axes[row, col]
            variables = variables_for_panel(
                model_slug, altitude, include_surrogates=False
            )
            labels = [
                label_for(variable, model_label, altitude) for variable in variables
            ]
            aggregated = aggregate_pair_links(
                load_links(model_slug, altitude), variables
            )
            matrix = np.full((len(variables), len(variables)), np.nan)
            lag_labels = [["" for _ in variables] for _ in variables]
            index = {variable: idx for idx, variable in enumerate(variables)}
            for (cause, target), (value, lags) in aggregated.items():
                matrix[index[cause], index[target]] = value
                lag_labels[index[cause]][index[target]] = lag_text(lags)

            ax.imshow(matrix, cmap=cmap, norm=norm)
            ax.set_xticks(
                np.arange(len(variables)), labels, rotation=35, ha="right", fontsize=8.5
            )
            ax.set_yticks(np.arange(len(variables)), labels, fontsize=8.5)
            ax.set_xlabel("")
            ax.set_ylabel("")
            title = panel_title(row, col)
            if title:
                ax.set_title(title, fontsize=10, pad=6)
            for i in range(len(variables)):
                for j in range(len(variables)):
                    if not np.isfinite(matrix[i, j]):
                        continue
                    text_color = readable_text_color(cmap(norm(matrix[i, j])))
                    ax.text(
                        j,
                        i,
                        f"{matrix[i, j]:.2f}\n{lag_labels[i][j]}d",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=text_color,
                    )

    fig.subplots_adjust(
        left=0.15, right=0.86, top=0.96, bottom=0.08, wspace=0.34, hspace=0.5
    )
    cax = fig.add_axes([0.9, 0.18, 0.025, 0.64])
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.set_label("MCI GPDCtorch distance correlation", fontsize=9)
    cbar.ax.tick_params(labelsize=8.5)
    save_figure(fig, "hasdm_msis_residuals_mean_mci_heatmaps_500_825.pgf")


def hasdm_variables(target: str) -> list[str]:
    return [*VARIABLE_ORDER, target, *SURROGATES]


def hasdm_heatmap_vmax() -> float:
    values = []
    for dataset, _, target in HASDM_PANELS:
        variables = hasdm_variables(target)
        values.extend(
            value
            for value, _ in aggregate_pair_links(
                load_links_file(hasdm_links_path(dataset)), variables
            ).values()
        )
    return max(max(values), 0.01)


def plot_hasdm_density_heatmap_composite() -> None:
    vmax = hasdm_heatmap_vmax()
    cmap = plt.get_cmap("viridis")
    norm = colors.Normalize(vmin=0.0, vmax=vmax)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4), constrained_layout=False)

    for ax, (dataset, title, target) in zip(axes, HASDM_PANELS, strict=True):
        variables = hasdm_variables(target)
        labels = [hasdm_label_for(variable) for variable in variables]
        aggregated = aggregate_pair_links(
            load_links_file(hasdm_links_path(dataset)), variables
        )
        matrix = np.full((len(variables), len(variables)), np.nan)
        lag_labels = [["" for _ in variables] for _ in variables]
        index = {variable: idx for idx, variable in enumerate(variables)}
        for (cause, target_variable), (value, lags) in aggregated.items():
            matrix[index[cause], index[target_variable]] = value
            lag_labels[index[cause]][index[target_variable]] = lag_text(lags)

        ax.imshow(matrix, cmap=cmap, norm=norm)
        ax.set_xticks(
            np.arange(len(variables)), labels, rotation=35, ha="right", fontsize=9
        )
        ax.set_yticks(np.arange(len(variables)), labels, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(title, fontsize=10, pad=6)
        for i in range(len(variables)):
            for j in range(len(variables)):
                if not np.isfinite(matrix[i, j]):
                    continue
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.2f}\n{lag_labels[i][j]}d",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=readable_text_color(cmap(norm(matrix[i, j]))),
                )

    fig.subplots_adjust(left=0.1, right=0.88, top=0.88, bottom=0.18, wspace=0.34)
    cax = fig.add_axes([0.91, 0.25, 0.025, 0.58])
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.set_label("MCI GPDCtorch distance correlation", fontsize=9)
    cbar.ax.tick_params(labelsize=8.5)
    save_figure_to_output(
        fig,
        "hasdm_density_range_mci_heatmaps_500km_with_noise.pgf",
        HASDM_COMPOSITE_DIR,
    )


def plot_hasdm_graph_panel(
    ax: plt.Axes,
    dataset: str,
    title: str,
    target: str,
    edge_norm: colors.Normalize,
    node_norm: colors.Normalize,
) -> None:
    variables = hasdm_variables(target)
    labels = {variable: hasdm_label_for(variable) for variable in variables}
    positions = graph_positions(variables)
    aggregated = aggregate_pair_links(
        load_links_file(hasdm_links_path(dataset)), variables
    )
    edge_cmap = plt.get_cmap("viridis")
    node_cmap = plt.get_cmap("magma")

    pair_counter: dict[frozenset[str], int] = {}
    for (cause, target_variable), (value, lags) in sorted(
        aggregated.items(), key=lambda item: item[1][0]
    ):
        if cause == target_variable:
            continue
        key = frozenset((cause, target_variable))
        pair_counter[key] = pair_counter.get(key, 0) + 1
        rad = 0.14 if pair_counter[key] % 2 else -0.14
        draw_curved_arrow(
            ax,
            positions[cause],
            positions[target_variable],
            edge_cmap(edge_norm(value)),
            1.0 + 7.0 * edge_norm(value),
            rad,
        )
        sx, sy = positions[cause]
        tx, ty = positions[target_variable]
        ax.text(
            (sx + tx) / 2,
            (sy + ty) / 2,
            lag_text(lags),
            fontsize=7.5,
            ha="center",
            va="center",
            color="black",
        )

    node_values = {variable: 0.0 for variable in variables}
    for (cause, target_variable), (value, _) in aggregated.items():
        if cause == target_variable:
            node_values[cause] = max(node_values[cause], value)
            draw_self_loop(
                ax,
                positions[cause],
                node_cmap(node_norm(value)),
                1.0 + 6.0 * node_norm(value),
            )

    for variable in variables:
        x, y = positions[variable]
        ax.scatter(
            [x],
            [y],
            s=700,
            c=[node_cmap(node_norm(node_values[variable]))],
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
        )
        tx, ty, ha, va = label_position(variable, positions[variable])
        ax.text(
            tx,
            ty,
            labels[variable],
            ha=ha,
            va=va,
            fontsize=8.5,
            zorder=6,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.2},
        )

    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.set_aspect("equal")


def hasdm_graph_vmaxes() -> tuple[float, float]:
    edge_values = []
    node_values = []
    for dataset, _, target in HASDM_PANELS:
        variables = hasdm_variables(target)
        aggregated = aggregate_pair_links(
            load_links_file(hasdm_links_path(dataset)), variables
        )
        for (cause, target_variable), (value, _) in aggregated.items():
            if cause == target_variable:
                node_values.append(value)
            else:
                edge_values.append(value)
    return max(max(edge_values), 0.01), max(max(node_values), 0.01)


def plot_hasdm_density_graph_composite() -> None:
    edge_vmax, node_vmax = hasdm_graph_vmaxes()
    edge_norm = colors.Normalize(vmin=0.0, vmax=edge_vmax)
    node_norm = colors.Normalize(vmin=0.0, vmax=node_vmax)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.7), constrained_layout=False)
    for ax, (dataset, title, target) in zip(axes, HASDM_PANELS, strict=True):
        plot_hasdm_graph_panel(ax, dataset, title, target, edge_norm, node_norm)

    fig.subplots_adjust(left=0.03, right=0.84, top=0.9, bottom=0.05, wspace=0.02)
    edge_cax = fig.add_axes([0.88, 0.56, 0.025, 0.32])
    node_cax = fig.add_axes([0.88, 0.14, 0.025, 0.32])
    edge_cbar = fig.colorbar(
        cm.ScalarMappable(norm=edge_norm, cmap="viridis"), cax=edge_cax
    )
    edge_cbar.set_label("edge MCI", fontsize=9)
    edge_cbar.ax.tick_params(labelsize=8.5)
    node_cbar = fig.colorbar(
        cm.ScalarMappable(norm=node_norm, cmap="magma"), cax=node_cax
    )
    node_cbar.set_label("auto-MCI", fontsize=9)
    node_cbar.ax.tick_params(labelsize=8.5)
    save_figure_to_output(
        fig,
        "hasdm_density_range_process_graphs_with_noise_500km.pgf",
        HASDM_COMPOSITE_DIR,
    )


def global_heatmap_vmax_compact() -> float:
    values = [
        value
        for value, _ in aggregate_pair_links(
            load_links_file(global_links_path()), GLOBAL_VARIABLES
        ).values()
    ]
    return max(max(values), 0.01)


def plot_global_mean_heatmap_composite() -> None:
    vmax = global_heatmap_vmax_compact()
    cmap = plt.get_cmap("viridis")
    norm = colors.Normalize(vmin=0.0, vmax=vmax)
    fig, ax = plt.subplots(figsize=(4.9, 4.4), constrained_layout=False)
    labels = [global_label_for(variable) for variable in GLOBAL_VARIABLES]
    aggregated = aggregate_pair_links(
        load_links_file(global_links_path()), GLOBAL_VARIABLES
    )
    matrix = np.full((len(GLOBAL_VARIABLES), len(GLOBAL_VARIABLES)), np.nan)
    lag_labels = [["" for _ in GLOBAL_VARIABLES] for _ in GLOBAL_VARIABLES]
    index = {variable: idx for idx, variable in enumerate(GLOBAL_VARIABLES)}

    for (cause, target_variable), (value, lags) in aggregated.items():
        matrix[index[cause], index[target_variable]] = value
        lag_labels[index[cause]][index[target_variable]] = lag_text(lags)

    ax.imshow(matrix, cmap=cmap, norm=norm)
    ax.set_xticks(
        np.arange(len(GLOBAL_VARIABLES)), labels, rotation=35, ha="right", fontsize=9
    )
    ax.set_yticks(np.arange(len(GLOBAL_VARIABLES)), labels, fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("")
    for i in range(len(GLOBAL_VARIABLES)):
        for j in range(len(GLOBAL_VARIABLES)):
            if not np.isfinite(matrix[i, j]):
                continue
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}\n{lag_labels[i][j]}d",
                ha="center",
                va="center",
                fontsize=8,
                color=readable_text_color(cmap(norm(matrix[i, j]))),
            )

    fig.subplots_adjust(left=0.2, right=0.83, top=0.94, bottom=0.2)
    cax = fig.add_axes([0.86, 0.25, 0.035, 0.58])
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.set_label("MCI GPDCtorch distance correlation", fontsize=9)
    cbar.ax.tick_params(labelsize=8.5)
    save_figure_to_output(
        fig,
        "global_mean_density_mci_heatmap_325km_with_noise.pgf",
        GLOBAL_COMPOSITE_DIR,
    )


def global_graph_vmaxes() -> tuple[float, float]:
    edge_values = []
    node_values = []
    aggregated = aggregate_pair_links(
        load_links_file(global_links_path()), GLOBAL_VARIABLES
    )
    for (cause, target_variable), (value, _) in aggregated.items():
        if cause == target_variable:
            node_values.append(value)
        else:
            edge_values.append(value)
    return max(max(edge_values), 0.01), max(max(node_values), 0.01)


def plot_global_mean_graph_composite() -> None:
    edge_vmax, node_vmax = global_graph_vmaxes()
    edge_norm = colors.Normalize(vmin=0.0, vmax=edge_vmax)
    node_norm = colors.Normalize(vmin=0.0, vmax=node_vmax)
    fig, ax = plt.subplots(figsize=(5.0, 4.7), constrained_layout=False)
    labels = {variable: global_label_for(variable) for variable in GLOBAL_VARIABLES}
    positions = graph_positions(GLOBAL_VARIABLES)
    aggregated = aggregate_pair_links(
        load_links_file(global_links_path()), GLOBAL_VARIABLES
    )
    edge_cmap = plt.get_cmap("viridis")
    node_cmap = plt.get_cmap("magma")

    pair_counter: dict[frozenset[str], int] = {}
    for (cause, target_variable), (value, lags) in sorted(
        aggregated.items(), key=lambda item: item[1][0]
    ):
        if cause == target_variable:
            continue
        key = frozenset((cause, target_variable))
        pair_counter[key] = pair_counter.get(key, 0) + 1
        rad = 0.14 if pair_counter[key] % 2 else -0.14
        draw_curved_arrow(
            ax,
            positions[cause],
            positions[target_variable],
            edge_cmap(edge_norm(value)),
            1.0 + 7.0 * edge_norm(value),
            rad,
        )
        sx, sy = positions[cause]
        tx, ty = positions[target_variable]
        ax.text(
            (sx + tx) / 2,
            (sy + ty) / 2,
            lag_text(lags),
            fontsize=7.5,
            ha="center",
            va="center",
            color="black",
        )

    node_values = {variable: 0.0 for variable in GLOBAL_VARIABLES}
    for (cause, target_variable), (value, _) in aggregated.items():
        if cause == target_variable:
            node_values[cause] = max(node_values[cause], value)
            draw_self_loop(
                ax,
                positions[cause],
                node_cmap(node_norm(value)),
                1.0 + 6.0 * node_norm(value),
            )

    for variable in GLOBAL_VARIABLES:
        x, y = positions[variable]
        ax.scatter(
            [x],
            [y],
            s=700,
            c=[node_cmap(node_norm(node_values[variable]))],
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
        )
        tx, ty, ha, va = label_position(variable, positions[variable])
        ax.text(
            tx,
            ty,
            labels[variable],
            ha=ha,
            va=va,
            fontsize=8.5,
            zorder=6,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.2},
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.set_aspect("equal")

    fig.subplots_adjust(left=0.03, right=0.78, top=0.96, bottom=0.05)
    edge_cax = fig.add_axes([0.84, 0.56, 0.035, 0.32])
    node_cax = fig.add_axes([0.84, 0.14, 0.035, 0.32])
    edge_cbar = fig.colorbar(
        cm.ScalarMappable(norm=edge_norm, cmap="viridis"), cax=edge_cax
    )
    edge_cbar.set_label("edge MCI", fontsize=9)
    edge_cbar.ax.tick_params(labelsize=8.5)
    node_cbar = fig.colorbar(
        cm.ScalarMappable(norm=node_norm, cmap="magma"), cax=node_cax
    )
    node_cbar.set_label("auto-MCI", fontsize=9)
    node_cbar.ax.tick_params(labelsize=8.5)
    save_figure_to_output(
        fig,
        "global_mean_density_process_graph_325km_with_noise.pgf",
        GLOBAL_COMPOSITE_DIR,
    )


def graph_positions(variables: list[str]) -> dict[str, tuple[float, float]]:
    base = {
        "F10.7_OBS_CENTER81": (0.82, 0.82),
        "AP_AVG": (0.5, 0.88),
        "CO2_ppm": (0.16, 0.68),
        "saber_co2cool_139km": (0.18, 0.24),
        "surrogate_white_noise_1": (0.5, 0.12),
        "surrogate_white_noise_2": (0.82, 0.22),
    }
    residual = [variable for variable in variables if variable not in base][0]
    base[residual] = (0.62, 0.46)
    return {variable: base[variable] for variable in variables}


def graph_vmaxes() -> tuple[float, float]:
    edge_values = []
    node_values = []
    for model_slug, _ in MODELS:
        for altitude in ALTITUDES:
            variables = variables_for_panel(
                model_slug, altitude, include_surrogates=True
            )
            aggregated = aggregate_pair_links(
                load_links(model_slug, altitude), variables
            )
            for (cause, target), (value, _) in aggregated.items():
                if cause == target:
                    node_values.append(value)
                else:
                    edge_values.append(value)
    return max(max(edge_values), 0.01), max(max(node_values), 0.01)


def draw_curved_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: tuple[float, float, float, float],
    width: float,
    rad: float,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=width,
        color=color,
        alpha=0.9,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=22,
        shrinkB=22,
    )
    ax.add_patch(arrow)


def draw_self_loop(
    ax: plt.Axes,
    center: tuple[float, float],
    color: tuple[float, float, float, float],
    width: float,
) -> None:
    x, y = center
    arrow = FancyArrowPatch(
        (x - 0.045, y + 0.055),
        (x + 0.045, y + 0.055),
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=width,
        color=color,
        alpha=0.9,
        connectionstyle="arc3,rad=1.6",
        shrinkA=5,
        shrinkB=5,
    )
    ax.add_patch(arrow)


def label_position(
    variable: str, position: tuple[float, float]
) -> tuple[float, float, str, str]:
    x, y = position
    offsets = {
        "F10.7_OBS_CENTER81": (0.0, 0.095, "center", "bottom"),
        "AP_AVG": (0.0, 0.095, "center", "bottom"),
        "CO2_ppm": (-0.085, 0.0, "right", "center"),
        "saber_co2cool_139km": (-0.085, 0.0, "right", "center"),
        "surrogate_white_noise_1": (0.0, -0.095, "center", "top"),
        "surrogate_white_noise_2": (0.085, 0.0, "left", "center"),
    }
    dx, dy, ha, va = offsets.get(variable, (0.085, 0.0, "left", "center"))
    return x + dx, y + dy, ha, va


def plot_graph_panel(
    ax: plt.Axes,
    model_slug: str,
    model_label: str,
    altitude: int,
    edge_norm: colors.Normalize,
    node_norm: colors.Normalize,
) -> None:
    variables = variables_for_panel(model_slug, altitude, include_surrogates=True)
    labels = {
        variable: label_for(variable, model_label, altitude) for variable in variables
    }
    positions = graph_positions(variables)
    aggregated = aggregate_pair_links(load_links(model_slug, altitude), variables)
    edge_cmap = plt.get_cmap("viridis")
    node_cmap = plt.get_cmap("magma")

    pair_counter: dict[frozenset[str], int] = {}
    for (cause, target), (value, lags) in sorted(
        aggregated.items(), key=lambda item: item[1][0]
    ):
        if cause == target:
            continue
        key = frozenset((cause, target))
        pair_counter[key] = pair_counter.get(key, 0) + 1
        rad = 0.14 if pair_counter[key] % 2 else -0.14
        color = edge_cmap(edge_norm(value))
        width = 1.0 + 7.0 * edge_norm(value)
        draw_curved_arrow(ax, positions[cause], positions[target], color, width, rad)
        sx, sy = positions[cause]
        tx, ty = positions[target]
        mx, my = (sx + tx) / 2, (sy + ty) / 2
        ax.text(
            mx,
            my,
            lag_text(lags),
            fontsize=7.5,
            ha="center",
            va="center",
            color="black",
        )

    node_values = {variable: 0.0 for variable in variables}
    for (cause, target), (value, lags) in aggregated.items():
        if cause == target:
            node_values[cause] = max(node_values[cause], value)
            draw_self_loop(
                ax,
                positions[cause],
                node_cmap(node_norm(value)),
                1.0 + 6.0 * node_norm(value),
            )

    for variable in variables:
        x, y = positions[variable]
        ax.scatter(
            [x],
            [y],
            s=700,
            c=[node_cmap(node_norm(node_values[variable]))],
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
        )
        tx, ty, ha, va = label_position(variable, positions[variable])
        ax.text(
            tx,
            ty,
            labels[variable],
            ha=ha,
            va=va,
            fontsize=8.5,
            zorder=6,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.2},
        )

    title = panel_title(
        MODELS.index((model_slug, model_label)), ALTITUDES.index(altitude)
    )
    if title:
        ax.set_title(title, fontsize=10, pad=6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.set_aspect("equal")


def plot_graph_composite() -> None:
    edge_vmax, node_vmax = graph_vmaxes()
    edge_norm = colors.Normalize(vmin=0.0, vmax=edge_vmax)
    node_norm = colors.Normalize(vmin=0.0, vmax=node_vmax)
    fig, axes = plt.subplots(3, 2, figsize=(8.4, 11.6), constrained_layout=False)
    for row, (model_slug, model_label) in enumerate(MODELS):
        for col, altitude in enumerate(ALTITUDES):
            plot_graph_panel(
                axes[row, col], model_slug, model_label, altitude, edge_norm, node_norm
            )

    fig.subplots_adjust(
        left=0.03, right=0.85, top=0.96, bottom=0.04, wspace=0.02, hspace=0.14
    )
    edge_cax = fig.add_axes([0.89, 0.56, 0.025, 0.32])
    node_cax = fig.add_axes([0.89, 0.14, 0.025, 0.32])
    edge_cbar = fig.colorbar(
        cm.ScalarMappable(norm=edge_norm, cmap="viridis"), cax=edge_cax
    )
    edge_cbar.set_label("edge MCI", fontsize=9)
    edge_cbar.ax.tick_params(labelsize=8.5)
    node_cbar = fig.colorbar(
        cm.ScalarMappable(norm=node_norm, cmap="magma"), cax=node_cax
    )
    node_cbar.set_label("auto-MCI", fontsize=9)
    node_cbar.ax.tick_params(labelsize=8.5)
    save_figure(fig, "hasdm_msis_residuals_mean_process_graphs_with_noise_500_825.pgf")


def save_figure(fig: plt.Figure, filename: str) -> None:
    save_figure_to_output(fig, filename, OUT_DIR)


def save_figure_to_output(fig: plt.Figure, filename: str, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / filename, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plot_heatmap_composite()
    plot_graph_composite()
    plot_hasdm_density_heatmap_composite()
    plot_hasdm_density_graph_composite()
    plot_global_mean_heatmap_composite()
    plot_global_mean_graph_composite()


if __name__ == "__main__":
    main()
