"""Shared graph policy and selected-link extraction for augmented PCMCI+ runs."""

from __future__ import annotations

import numpy as np


def build_link_assumptions(
    node_names: list[str], tau_max: int, f107_node: str = "f10_7_center81"
) -> dict[int, dict[tuple[int, int], str]]:
    """Apply the F10.7-exogenous policy to an arbitrary augmented node list."""
    f107_index = node_names.index(f107_node)
    assumptions: dict[int, dict[tuple[int, int], str]] = {
        target: {} for target in range(len(node_names))
    }
    for target in range(len(node_names)):
        for cause in range(len(node_names)):
            for lag in range(1, tau_max + 1):
                if target == f107_index and cause != f107_index:
                    continue
                assumptions[target][(cause, -lag)] = "-?>"
    for cause in range(len(node_names)):
        for target in range(cause + 1, len(node_names)):
            if cause == f107_index:
                assumptions[target][(f107_index, 0)] = "-?>"
            elif target == f107_index:
                assumptions[cause][(f107_index, 0)] = "-?>"
            else:
                assumptions[target][(cause, 0)] = "o?o"
    return assumptions


def selected_control_links(
    results: dict[str, np.ndarray], node_names: list[str], control_names: set[str]
) -> list[dict[str, str | int | float]]:
    """Return canonical selected graph rows involving at least one control.

    The graph orientation and contemporaneous de-duplication follow the existing
    surrogate robustness output exactly.
    """
    rows: list[dict[str, str | int | float]] = []
    graph, p_matrix, val_matrix = (
        results["graph"],
        results["p_matrix"],
        results["val_matrix"],
    )
    for cause_index, cause in enumerate(node_names):
        for target_index, target in enumerate(node_names):
            if cause not in control_names and target not in control_names:
                continue
            for lag in range(graph.shape[2]):
                graph_mark = str(graph[cause_index, target_index, lag])
                if not graph_mark or (lag == 0 and graph_mark == "<--"):
                    continue
                if (
                    lag == 0
                    and graph_mark in {"o-o", "x-x"}
                    and cause_index > target_index
                ):
                    continue
                if lag > 0 and graph_mark != "-->":
                    continue
                if cause in control_names and target in control_names:
                    relation = "surrogate↔surrogate"
                elif cause in control_names:
                    relation = "surrogate→physical"
                else:
                    relation = "physical→surrogate"
                rows.append(
                    {
                        "relation": relation,
                        "cause": cause,
                        "target": target,
                        "lag": lag,
                        "graph_mark": graph_mark,
                        "p_value": float(p_matrix[cause_index, target_index, lag]),
                        "val": float(val_matrix[cause_index, target_index, lag]),
                    }
                )
    return rows
