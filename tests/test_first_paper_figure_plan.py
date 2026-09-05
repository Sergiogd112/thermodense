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


def test_figure_six_requires_brown_digitized_profile_provenance_before_rendering():
    figure_six = next(
        figure for figure in load_plan()["figures"] if figure["number"] == 6
    )
    plan_text = (
        Path(__file__).resolve().parents[1] / "docs/first-paper-figure-plan.md"
    ).read_text(encoding="utf-8")

    assert (
        "Panel A is rendered from plot-precision values vector-extracted from Brown et al. 2024 Figure 2"
        in figure_six["panel_layout_contract"]
    )
    assert (
        "Panel B shows Thermodense's updated solar-adjusted"
        in figure_six["panel_layout_contract"]
    )
    assert (
        "brown_2024_figure2_digitized.csv"
        in figure_six["implementation_status_blockers"]
    )
    assert "checksum" in figure_six["implementation_status_blockers"]
    assert (
        "not replacements for original study data"
        in figure_six["implementation_status_blockers"]
    )
    assert (
        "do not infer altitude profiles from Brown's 400-km tables"
        in figure_six["implementation_status_blockers"]
    )
    assert "presentation-derived digitized CSV" in plan_text
    assert "not replacements for the original study data" in plan_text


def test_figure_six_plan_declares_vector_compositor_and_verified_source_checksums():
    figure_six = next(
        figure for figure in load_plan()["figures"] if figure["number"] == 6
    )

    assert "compose_density_trend_figure6.py" in figure_six["compositor_contract"]
    assert "--require-jb" in figure_six["compositor_contract"]
    assert (
        "jb2006_log10rho_daily_mean_<alt>km"
        in figure_six["implementation_status_blockers"]
    )
    assert (
        "Final output may never include only one JB model"
        in figure_six["implementation_status_blockers"]
    )
    assert figure_six["brown_pdf_sha256"] == (
        "ac2f2097d3ee28b85bce2e7d082af7e4203459c87e16408480fbdfefa9c392ea"
    )
    assert figure_six["brown_digitized_csv_sha256"] == (
        "1fafa2718250adcd01677d4c9257cef4f72d3e7d654a7a14accc2c8cdc216583"
    )
    assert figure_six["brown_digitized_presentation_source_sha256"] == (
        "1bd91d049f801edba688aabf49952cf8a7a553a5e4b9c47c5ba59909d6a5a7e2"
    )
    assert "--brown-pdf" not in figure_six["compositor_contract"]
    assert figure_six["output_contract"] == [
        "density_trend_figure6.png",
        "density_trend_figure6.pdf",
        "density_trend_figure6_caption.txt",
        "density_trend_figure6_alt_text.txt",
        "density_trend_figure6_provenance.json",
    ]


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
