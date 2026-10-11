"""Focused tests for the Figure 6 Brown vector compositor."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import pytest

import scripts.compose_density_trend_figure6 as figure6


def brown_csv_path() -> Path:
    return Path("data/derived/literature/brown_2024_figure2_digitized.csv")


def trend_data() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "dataset": ["Global mean thermospheric density"],
            "altitude_km": [400.0],
            "trend_percent_per_decade": [-5.0],
            "trend_percent_per_decade_hac_95_ci_lower": [-7.0],
            "trend_percent_per_decade_hac_95_ci_upper": [-3.0],
            "duration_bin_years": [11],
        }
    )


def paired_jb_trend_data() -> pl.DataFrame:
    return pl.concat(
        [
            trend_data(),
            pl.DataFrame(
                {
                    "dataset": [
                        "JB2006 Mauna Loa baseline",
                        "JB2008 Mauna Loa baseline",
                    ],
                    "altitude_km": [400.0, 425.0],
                    "trend_percent_per_decade": [-2.0, -3.0],
                    "trend_percent_per_decade_hac_95_ci_lower": [-3.0, -4.0],
                    "trend_percent_per_decade_hac_95_ci_upper": [-1.0, -2.0],
                    "duration_bin_years": [11, 11],
                }
            ),
        ]
    )


def test_bundled_brown_csv_has_authoritative_hash_shape_and_studies():
    data = figure6.load_brown_literature(brown_csv_path())

    assert figure6.sha256_file(brown_csv_path()) == figure6.BROWN_DATA_SHA256
    assert data.height == 427
    assert data.get_column("study").n_unique() == 16


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("series_type", "curve", "series_type"),
        ("line_style", "dashdot", "line_style"),
        ("density_trend_pct_per_decade", "not-a-number", "non-numeric"),
        ("include_in_plot", "perhaps", "invalid boolean"),
    ],
)
def test_load_brown_literature_rejects_malformed_values(
    tmp_path: Path, column: str, value: str, message: str
):
    data = pl.read_csv(brown_csv_path(), schema_overrides={column: pl.String})
    data = data.with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.lit(value))
        .otherwise(pl.col(column))
        .alias(column)
    )
    path = tmp_path / "brown.csv"
    data.write_csv(path)

    with pytest.raises(ValueError, match=message):
        figure6.load_brown_literature(path)


def test_load_brown_literature_rejects_duplicate_sequences(tmp_path: Path):
    data = pl.read_csv(brown_csv_path())
    data = data.with_columns(
        pl.when(pl.int_range(pl.len()) == 1)
        .then(pl.lit(0))
        .otherwise(pl.col("sequence"))
        .alias("sequence")
    )
    path = tmp_path / "brown.csv"
    data.write_csv(path)

    with pytest.raises(ValueError, match="unique within study and variant"):
        figure6.load_brown_literature(path)


def test_load_brown_literature_rejects_nonfinite_included_coordinates(tmp_path: Path):
    data = pl.read_csv(brown_csv_path(), schema_overrides={"altitude_km": pl.String})
    data = data.with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.lit("nan"))
        .otherwise(pl.col("altitude_km"))
        .alias("altitude_km")
    )
    path = tmp_path / "brown.csv"
    data.write_csv(path)

    with pytest.raises(ValueError, match="finite trend and altitude"):
        figure6.load_brown_literature(path)


def test_plot_brown_literature_draws_profiles_points_errors_and_one_legend_entry_per_study():
    data = figure6.load_brown_literature(brown_csv_path())
    fig, ax = plt.subplots()

    handles = figure6.plot_brown_literature(ax, data)

    assert ax.get_xlim() == figure6.BROWN_X_LIMITS
    assert len(ax.lines) >= 6  # Profile lines plus point uncertainty-bar artists.
    assert (
        len(ax.collections)
        >= data.filter(pl.col("include_in_plot") & pl.col("marker")).height
    )
    assert [handle.get_label() for handle in handles] == list(figure6.BROWN_STUDY_ORDER)
    assert len(handles) == data.get_column("study").n_unique()
    assert {handle.get_marker() for handle in handles} >= {
        "o",
        "s",
        "x",
        "*",
        "D",
        "p",
        "h",
        "^",
        "v",
    }
    plt.close(fig)


def test_compose_panels_shares_altitude_axis_and_only_left_y_labels():
    fig = plt.figure(figsize=(15.5, 7.5), layout="constrained")
    ax_a, ax_b = figure6.compose_panels(
        fig, figure6.load_brown_literature(brown_csv_path()), trend_data()
    )

    assert ax_a.get_shared_y_axes().joined(ax_a, ax_b)
    assert ax_a.get_ylim() == (0.0, 850.0)
    assert ax_b.get_ylim() == (0.0, 850.0)
    assert ax_a.get_ylabel() == "Altitude (km)"
    assert ax_b.get_ylabel() == ""
    assert not any(label.get_visible() for label in ax_b.get_yticklabels())
    assert len(fig.axes) == 2
    assert ax_a.get_legend() is not None
    assert ax_a.get_legend()._ncols == 4
    assert ax_b.get_position().x0 - ax_a.get_position().x1 < 0.08
    plt.close(fig)


def test_caption_and_provenance_use_digitized_vector_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    trend_csv = tmp_path / "trends.csv"
    trend_data().write_csv(trend_csv)
    monkeypatch.setattr(
        "matplotlib.figure.Figure.savefig",
        lambda _fig, path, **_kwargs: Path(path).touch(),
    )

    provenance = figure6.compose_figure(
        brown_csv_path(), trend_csv, tmp_path, ["test-figure6"]
    )

    assert provenance["schema_version"] == 4
    assert provenance["brown"]["digitized_csv"]["sha256"] == figure6.BROWN_DATA_SHA256
    assert (
        provenance["brown"]["digitized_csv"][
            "presentation_source_sha256_before_lf_normalization"
        ]
        == figure6.BROWN_PRESENTATION_SOURCE_SHA256
    )
    assert provenance["brown"]["rows"] == 427
    assert provenance["brown"]["studies"] == 16
    assert provenance["shared_altitude_limits"] == [0.0, 850.0]
    assert "digitized third-party figure geometry" in provenance["disclosure"]
    assert "not replacements for the original study data" in figure6.figure_caption()
    assert provenance["jb_required"] is False
    assert provenance["jb_included"] is False
    assert "Draft: paired JB2006/JB2008" in figure6.figure_caption()
    written = json.loads(
        (tmp_path / "density_trend_figure6_provenance.json").read_text()
    )
    assert "raster_dpi" not in written["brown"]
    assert "plot_crop_fractions" not in written["brown"]


def test_parse_args_defaults_to_bundled_brown_csv():
    args = figure6.parse_args([])

    assert args.brown_data == figure6.DEFAULT_BROWN_DATA


def test_compositor_rejects_singleton_jb_even_for_draft(tmp_path: Path):
    trends = trend_data().with_columns(
        pl.lit("JB2006 Mauna Loa baseline").alias("dataset")
    )

    with pytest.raises(ValueError, match="paired JB2006/JB2008"):
        figure6.validate_jb_pair(trends)


def test_compositor_requires_and_records_paired_jb(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    trend_csv = tmp_path / "trends.csv"
    paired_jb_trend_data().write_csv(trend_csv)
    monkeypatch.setattr(
        "matplotlib.figure.Figure.savefig",
        lambda _fig, path, **_kwargs: Path(path).touch(),
    )

    provenance = figure6.compose_figure(
        brown_csv_path(), trend_csv, tmp_path, ["test-figure6", "--require-jb"], True
    )

    assert provenance["jb_required"] is True
    assert provenance["jb_included"] is True
    assert provenance["jb_canonical_pair"] == list(figure6.JB_DATASETS)
    assert "externally generated provider-model outputs" in figure6.figure_caption(True)
    assert (
        "JB inputs are externally generated provider-model outputs"
        in provenance["disclosure"]
    )


def drawn_color(line) -> str:
    """Errorbar caps are marker-only lines drawn with their edge color."""
    return line.get_markeredgecolor() if line.get_linestyle() in ("None", "") else line.get_color()


def test_muted_brown_draws_everything_gray_with_one_legend_entry():
    data = figure6.load_brown_literature(brown_csv_path())
    fig, ax = plt.subplots()

    handles = figure6.plot_brown_literature(ax, data, muted=True)

    assert len(handles) == 1
    assert "Brown et al. (2024)" in handles[0].get_label()
    assert {drawn_color(line) for line in ax.lines} == {figure6.BROWN_MUTED_COLOR}
    assert ax.get_xlim() != figure6.BROWN_X_LIMITS  # axis styling is left to the caller
    plt.close(fig)


def test_overlay_puts_brown_behind_current_trends_on_one_axis(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    fig = plt.figure(figsize=(10, 7.5), layout="constrained")
    ax = figure6.compose_overlay(
        fig, figure6.load_brown_literature(brown_csv_path()), paired_jb_trend_data()
    )

    assert len(fig.axes) == 1
    assert ax.get_ylim() == (0.0, 850.0)
    gray = [line for line in ax.lines if drawn_color(line) == figure6.BROWN_MUTED_COLOR]
    colored = [
        line
        for line in ax.lines
        if drawn_color(line) not in (figure6.BROWN_MUTED_COLOR, "black")  # zero line is reference
        and line.get_linestyle() not in ("None", "")
    ]
    assert gray and colored
    assert max(line.get_zorder() for line in gray) < min(line.get_zorder() for line in colored)
    plt.close(fig)

    trend_csv = tmp_path / "trends.csv"
    paired_jb_trend_data().write_csv(trend_csv)
    monkeypatch.setattr(
        "matplotlib.figure.Figure.savefig",
        lambda _fig, path, **_kwargs: Path(path).touch(),
    )
    provenance = figure6.compose_figure(
        brown_csv_path(), trend_csv, tmp_path, ["test"], require_jb=True, layout="overlay"
    )
    assert provenance["layout"] == "overlay"
    assert (tmp_path / "density_trend_figure6_overlay_provenance.json").is_file()
    assert "Light gray" in (tmp_path / "density_trend_figure6_overlay_caption.txt").read_text()
    with pytest.raises(ValueError):
        figure6.compose_figure(brown_csv_path(), trend_csv, tmp_path, layout="stacked")


def test_extra_trend_csv_adds_waccmx_with_caption_and_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    trend_csv = tmp_path / "trends.csv"
    paired_jb_trend_data().write_csv(trend_csv)
    extra = tmp_path / "waccmx.csv"
    pl.DataFrame(
        {
            "dataset": list(figure6.WACCMX_DATASETS),
            "altitude_km": [398.5, 398.5],
            "trend_percent_per_decade": [-5.9, -6.6],
            "trend_percent_per_decade_hac_95_ci_lower": [-9.3, -10.8],
            "trend_percent_per_decade_hac_95_ci_upper": [-2.5, -2.2],
            "duration_bin_years": [11, 11],
            "day_coverage": [1.0, 1.0],
        }
    ).write_csv(extra)
    monkeypatch.setattr(
        "matplotlib.figure.Figure.savefig",
        lambda _fig, path, **_kwargs: Path(path).touch(),
    )

    provenance = figure6.compose_figure(
        brown_csv_path(), trend_csv, tmp_path, ["test"], require_jb=True,
        layout="overlay", extra_trend_csvs=(extra,),
    )

    assert provenance["waccmx_included"] is True
    assert provenance["extra_trend_csvs"][0]["sha256"] == figure6.sha256_file(extra)
    caption = (tmp_path / "density_trend_figure6_overlay_caption.txt").read_text()
    assert "WACCM-X" in caption and "not an observation" in caption
    with pytest.raises(FileNotFoundError):
        figure6.compose_figure(brown_csv_path(), trend_csv, tmp_path,
                               extra_trend_csvs=(tmp_path / "missing.csv",))


def test_overlay_limits_keep_every_brown_point_and_error_bar():
    near_zero = pl.DataFrame(
        {
            "dataset": ["NRLMSIS 2.1 Mauna Loa baseline"],
            "altitude_km": [400.0],
            "trend_percent_per_decade": [-0.2],
            "trend_percent_per_decade_hac_95_ci_lower": [-0.4],
            "trend_percent_per_decade_hac_95_ci_upper": [0.0],
            "duration_bin_years": [55],
        }
    )
    brown = figure6.load_brown_literature(brown_csv_path())
    fig = plt.figure(figsize=(10, 7.5), layout="constrained")
    ax = figure6.compose_overlay(fig, brown, near_zero)
    low, high = figure6.brown_x_extent(brown)
    assert ax.get_xlim()[0] <= low and ax.get_xlim()[1] >= high
    plt.close(fig)


def test_overlay_disclosure_describes_one_shared_axis(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    trend_csv = tmp_path / "trends.csv"
    paired_jb_trend_data().write_csv(trend_csv)
    monkeypatch.setattr(
        "matplotlib.figure.Figure.savefig",
        lambda _fig, path, **_kwargs: Path(path).touch(),
    )
    provenance = figure6.compose_figure(
        brown_csv_path(), trend_csv, tmp_path, ["test"], require_jb=True, layout="overlay"
    )
    assert "Panel" not in provenance["disclosure"]
    assert "same axis" in provenance["disclosure"] and "CC BY 4.0" in provenance["disclosure"]
