import json
from pathlib import Path

import polars as pl
import pytest

from scripts.maunaloa_global_figures import (
    PAPER_CANDIDATE_FIGURE_3_COLUMNS,
    PAPER_CANDIDATE_FIGURE_3_ROWS,
    PAPER_CANDIDATE_FIGURE_4_CAUSES,
    paper_candidate_figure_3_layout,
    plot_paper_candidate_figure_3,
    require_columns,
)


def load_plan() -> dict:
    path = (
        Path(__file__).resolve().parents[1]
        / "configs/paper/first-paper-figures-v1.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_first_paper_plan_has_six_unique_ordered_candidates():
    plan = load_plan()

    assert plan["main_figure_target"] == "5-6"
    assert [figure["number"] for figure in plan["figures"]] == [1, 2, 3, 4, 5, 6]
    assert len({figure["id"] for figure in plan["figures"]}) == 6


def test_first_paper_plan_records_resolved_figure_decisions():
    figures = {figure["number"]: figure for figure in load_plan()["figures"]}

    assert figures[3]["panel_layout_contract"] == (
        "Exact 4x4 scatter composite. Rows: mean log density at 175 and 825 km, "
        "then daily log-density range at 175 and 825 km. Columns: F10.7 centered "
        "81-day mean, Ap, tropospheric CO2, SABER CO2 cooling at 139 km."
    )
    assert "paired JB2006/JB2008" in figures[2]["panel_layout_contract"]
    assert (
        "WACCM-X is explicitly excluded" in figures[2]["implementation_status_blockers"]
    )
    assert "paper-organizing result" in figures[6]["role_message"]


def test_selected_figure_layout_helpers_are_fixed_to_the_plan_contract():
    assert paper_candidate_figure_3_layout() == (
        [
            "log10rho_175_daily_mean",
            "log10rho_825_daily_mean",
            "log10rho_175_daily_range",
            "log10rho_825_daily_range",
        ],
        ["F10.7_OBS_CENTER81", "AP_AVG", "CO2_ppm", "saber_co2cool_max_alt"],
    )
    assert PAPER_CANDIDATE_FIGURE_3_ROWS[0].endswith("175_daily_mean")
    assert PAPER_CANDIDATE_FIGURE_3_COLUMNS == PAPER_CANDIDATE_FIGURE_4_CAUSES


def test_selected_figure_column_guard_names_missing_columns():
    with pytest.raises(
        ValueError, match="Paper candidate Figure 3 requires columns: b"
    ):
        require_columns(
            pl.DataFrame({"a": [1]}), ["a", "b"], "Paper candidate Figure 3"
        )


def test_selected_figure_requires_both_approved_altitudes_before_rendering():
    with pytest.raises(ValueError, match="requires HASDM altitudes: 825 km"):
        plot_paper_candidate_figure_3(pl.DataFrame(), [175])
