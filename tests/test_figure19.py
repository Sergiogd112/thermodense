from datetime import datetime

import matplotlib
import polars as pl
import pytest

matplotlib.use("Agg")

from thermodense.figure19 import ModelSeries, create_figure_19


@pytest.mark.parametrize(
    "errorbar_mode",
    [
        "uncertainty_of_mean",
        "raw_observation_uncertainty",
        "paper",
        "sample_std",
    ],
)
def test_additional_model_series_are_aggregated_plotted_and_in_legend(errorbar_mode):
    df = pl.DataFrame(
        {
            "timestamp": [datetime(2010, 1, 1), datetime(2010, 1, 2)],
            "f107a": [90.0, 90.0],
            "Altitude (m)": [300_000.0, 300_000.0],
            "ln_density_ratio_0": [0.1, 0.2],
            "ln_density_ratio_2.0": [0.2, 0.3],
            "ln_density_ratio_jb2006": [0.3, 0.4],
            "ln_density_ratio_jb2008": [0.4, 0.5],
        }
    )

    fig = create_figure_19(
        [df],
        ["Test mission"],
        errorbar_mode=errorbar_mode,
        additional_model_series=(
            ModelSeries("ln_density_ratio_jb2006", "JB2006", "purple", "D"),
            ModelSeries("ln_density_ratio_jb2008", "JB2008", "orange", "P"),
        ),
    )

    data_ax = fig.axes[0]
    plotted_series = {container.get_label(): container for container in data_ax.containers}
    assert set(plotted_series) == {
        "NRLMSISE-00",
        "NRLMSIS 2.0",
        "JB2006",
        "JB2008",
    }
    assert plotted_series["JB2006"].lines[0].get_ydata()[0] == pytest.approx(0.35)
    assert plotted_series["JB2008"].lines[0].get_ydata()[0] == pytest.approx(0.45)
    assert [text.get_text() for text in fig.legends] == []
    assert [text.get_text() for text in fig.axes[-1].get_legend().get_texts()] == [
        "NRLMSISE-00",
        "NRLMSIS 2.0",
        "JB2006",
        "JB2008",
    ]
