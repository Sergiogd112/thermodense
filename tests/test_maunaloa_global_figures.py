from unittest.mock import MagicMock

import pytest

from scripts import maunaloa_global_figures as figures


@pytest.mark.parametrize(
    ("requested_title", "expected_title"),
    [
        ("Custom composite title", "Custom composite title"),
        (None, "Mauna Loa correlation by HASDM altitude"),
    ],
)
def test_plot_correlation_by_altitude_uses_composite_title(
    monkeypatch, requested_title, expected_title
):
    fig = MagicMock()
    axes = [MagicMock(), MagicMock()]
    line = MagicMock()
    line.get_color.return_value = "black"
    for ax in axes:
        ax.plot.return_value = (line,)
    monkeypatch.setattr(figures.plt, "subplots", lambda *args, **kwargs: (fig, axes))
    monkeypatch.setattr(figures, "require_columns", lambda *args: None)
    monkeypatch.setattr(
        figures,
        "correlation_and_duration",
        lambda *args: (0.5, 0.4, 0.6, None),
    )
    monkeypatch.setattr(figures, "add_correlation_effect_size_bands", lambda ax: None)
    monkeypatch.setattr(figures, "add_record_length_legend", lambda *args: None)
    monkeypatch.setattr(figures, "save_and_close", lambda *args: None)

    kwargs = {} if requested_title is None else {"title": requested_title}
    figures.plot_correlation_by_altitude(MagicMock(), [175], causes=["cause"], **kwargs)

    fig.suptitle.assert_called_once_with(expected_title, fontsize=11)
