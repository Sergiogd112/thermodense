"""Shared PGF backend configuration for thesis figure generation.

All figure-generating scripts should import this module at the top,
before any matplotlib plotting calls:

    from scripts.pgf_config import configure_pgf, TEXTWIDTH_IN, fig_size

    configure_pgf()

This switches matplotlib to the PGF backend so figures use the same
fonts as the LaTeX document, and provides textwidth-relative figure
sizing helpers.
"""

from __future__ import annotations

import matplotlib

# Thesis text block dimensions. The width is measured from the document
# template with layouts' \printinunitsof{in}\prntlen{\textwidth}.
TEXTWIDTH_IN = 5.90666
TEXTHEIGHT_IN = 22.0 / 2.54


def configure_pgf() -> None:
    """Activate the PGF backend with pdflatex-compatible settings."""
    matplotlib.use("pgf")
    matplotlib.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": False,
            "pgf.preamble": (
                r"\usepackage{amssymb}"
                r"\usepackage{siunitx}"
                r"\providecommand{\mathdefault}{\mathrm}"
                r"\usepackage[strings]{underscore}"
            ),
            # DPI for rasterized elements within PGF figures
            "savefig.dpi": 300,
            "figure.dpi": 300,
        }
    )


def fig_size(scale: float = 1.0, aspect_ratio: float = 0.75) -> tuple[float, float]:
    """Return (width, height) in inches relative to thesis textwidth.

    Args:
        scale: Fraction of textwidth to use (default 1.0 = full width).
        aspect_ratio: height / width ratio (default 0.75 = 4:3 landscape).
    """
    width = TEXTWIDTH_IN * scale
    height = width * aspect_ratio
    return width, height


def page_fig_size(
    scale: float = 1.0,
    aspect_ratio: float = 0.75,
    max_height_scale: float = 1.0,
) -> tuple[float, float]:
    """Return a textwidth-relative size capped by the usable text height."""
    width, height = fig_size(scale, aspect_ratio)
    return width, min(height, TEXTHEIGHT_IN * max_height_scale)
