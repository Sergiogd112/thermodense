"""Compose publication-draft Figure 6 from Brown Figure 2 and current trends."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from scripts.recreate_brown_density_trend_current import plot_trends

BROWN_TITLE = (
    "Future Climate Change in the Thermosphere Under Varying Solar Activity Conditions"
)
BROWN_CITATION = (
    "Brown, M. K., H. G. Lewis, A. J. Kavanagh, I. Cnossen, and S. Elvidge "
    "(2024), Journal of Geophysical Research: Space Physics, 129(9), e2024JA032659."
)
BROWN_DOI = "10.1029/2024JA032659"
BROWN_SOURCE_URL = "https://doi.org/10.1029/2024JA032659"
BROWN_PDF_REFERENCE_URL = (
    "https://agupubs.onlinelibrary.wiley.com/doi/pdfdirect/10.1029/2024JA032659"
)
BROWN_ACCEPTED_MANUSCRIPT_URL = (
    "https://nora.nerc.ac.uk/id/eprint/537943/1/JGR%20Space%20Physics%20-%202024%20"
    "-%20Brown%20-%20Future%20Climate%20Change%20in%20the%20Thermosphere%20Under%20"
    "Varying%20Solar%20Activity%20Conditions.pdf"
)
BROWN_PDF_SHA256 = "ac2f2097d3ee28b85bce2e7d082af7e4203459c87e16408480fbdfefa9c392ea"
BROWN_LICENSE = "CC BY 4.0"
BROWN_PRESENTATION_SOURCE_SHA256 = (
    "1bd91d049f801edba688aabf49952cf8a7a553a5e4b9c47c5ba59909d6a5a7e2"
)
BROWN_DATA_SHA256 = "1fafa2718250adcd01677d4c9257cef4f72d3e7d654a7a14accc2c8cdc216583"
DEFAULT_BROWN_DATA = Path("data/derived/literature/brown_2024_figure2_digitized.csv")
DEFAULT_TREND_CSV = Path(
    "outputs/figures/results/current_density_trends/"
    "current_density_trends_by_dataset_altitude.csv"
)
DEFAULT_OUTPUT_DIR = Path("outputs/figures/results/density_trend_figure6")
BROWN_X_LIMITS = (-7.0, 1.0)
SHARED_ALTITUDE_LIMITS = (0.0, 850.0)
JB_DATASETS = ("JB2006 Mauna Loa baseline", "JB2008 Mauna Loa baseline")
BROWN_REQUIRED_COLUMNS = {
    "study",
    "variant",
    "series_type",
    "sequence",
    "density_trend_pct_per_decade",
    "altitude_km",
    "uncertainty_minus_pct_per_decade",
    "uncertainty_plus_pct_per_decade",
    "marker",
    "line_style",
    "color_hex",
    "extraction_basis",
    "source_reference",
    "source_url",
    "notes",
    "include_in_plot",
}
BROWN_NUMERIC_COLUMNS = (
    "density_trend_pct_per_decade",
    "altitude_km",
    "uncertainty_minus_pct_per_decade",
    "uncertainty_plus_pct_per_decade",
)
BROWN_STUDY_ORDER = (
    "Keating 2000",
    "Emmert 2004",
    "Marcos 2005",
    "Qian 2006",
    "Akmaev 2006",
    "Emmert 2008",
    "Cnossen 2009",
    "Qian 2011",
    "Saunders 2011",
    "Emmert 2011",
    "Emmert 2015",
    "Solomon 2015",
    "Solomon 2018",
    "Solomon 2019",
    "Weng 2020",
    "Cnossen 2020",
)
BROWN_MARKERS = {
    "Keating 2000": "o",
    "Emmert 2004": "s",
    "Marcos 2005": "o",
    "Qian 2006": "o",
    "Akmaev 2006": "x",
    "Emmert 2008": "o",
    "Cnossen 2009": "*",
    "Qian 2011": "D",
    "Saunders 2011": "D",
    "Emmert 2011": "o",
    "Emmert 2015": "p",
    "Solomon 2015": "h",
    "Solomon 2018": "^",
    "Solomon 2019": "v",
    "Weng 2020": "D",
    "Cnossen 2020": "o",
}
BROWN_LINESTYLES = {"solid": "-", "dash": "--", "dot": ":", "none": "None"}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _invalid_values(data: pl.DataFrame, column: str) -> list[str]:
    """Return distinct non-empty source values that failed numeric conversion."""

    source = f"_{column}_source"
    return (
        data.filter(pl.col(source).str.strip_chars().ne("") & pl.col(column).is_null())
        .get_column(source)
        .unique()
        .to_list()
    )


def load_brown_literature(path: Path) -> pl.DataFrame:
    """Load and validate the vector-extracted Brown Figure 2 reconstruction."""

    if not path.is_file():
        raise FileNotFoundError(f"Brown digitized CSV does not exist: {path}")
    data = pl.read_csv(
        path, schema_overrides={column: pl.String for column in BROWN_REQUIRED_COLUMNS}
    )
    missing = sorted(BROWN_REQUIRED_COLUMNS - set(data.columns))
    if missing:
        raise ValueError(
            "Brown digitized CSV is missing required columns: " + ", ".join(missing)
        )
    if data.height < 427:
        raise ValueError("Brown digitized CSV must contain at least 427 rows")

    data = data.with_columns(
        [pl.col(column).alias(f"_{column}_source") for column in BROWN_NUMERIC_COLUMNS]
    ).with_columns(
        [
            pl.col(column).cast(pl.Float64, strict=False).alias(column)
            for column in BROWN_NUMERIC_COLUMNS
        ]
        + [pl.col("sequence").cast(pl.Int64, strict=False).alias("sequence")]
        + [
            pl.col(column)
            .str.to_lowercase()
            .replace_strict({"true": True, "false": False}, default=None)
            .alias(column)
            for column in ("marker", "include_in_plot")
        ]
    )
    for column in BROWN_NUMERIC_COLUMNS:
        invalid = _invalid_values(data, column)
        if invalid:
            raise ValueError(f"Brown digitized CSV has non-numeric {column}: {invalid}")
    if data.get_column("sequence").null_count():
        raise ValueError("Brown digitized CSV has non-integer sequence values")
    for column in ("marker", "include_in_plot"):
        if data.get_column(column).null_count():
            raise ValueError(f"Brown digitized CSV has invalid boolean {column} values")
    invalid_series = data.filter(~pl.col("series_type").is_in(["profile", "point"]))
    if invalid_series.height:
        raise ValueError("Brown digitized CSV series_type must be profile or point")
    invalid_styles = data.filter(~pl.col("line_style").is_in(BROWN_LINESTYLES))
    if invalid_styles.height:
        raise ValueError(
            "Brown digitized CSV line_style must be solid, dash, dot, or none"
        )
    if data.select(pl.col("study").n_unique()).item() < 16:
        raise ValueError("Brown digitized CSV must contain at least 16 studies")
    duplicates = (
        data.group_by("study", "variant", "sequence").len().filter(pl.col("len") > 1)
    )
    if duplicates.height:
        raise ValueError(
            "Brown digitized CSV sequence must be unique within study and variant"
        )
    included = data.filter(pl.col("include_in_plot"))
    if included.is_empty():
        raise ValueError("Brown digitized CSV has no included rows")
    coordinates = included.select(BROWN_NUMERIC_COLUMNS[:2]).to_numpy().astype(float)
    if not np.isfinite(coordinates).all():
        raise ValueError(
            "Brown digitized CSV included rows require finite trend and altitude"
        )
    return data.drop([f"_{column}_source" for column in BROWN_NUMERIC_COLUMNS])


def plot_brown_literature(ax: object, data: pl.DataFrame) -> list[object]:
    """Plot the vector-extracted Brown Figure 2 geometry and study legend handles."""

    from matplotlib.lines import Line2D

    included = data.filter(pl.col("include_in_plot"))
    for group in included.filter(pl.col("series_type") == "profile").partition_by(
        ["study", "variant"], as_dict=False
    ):
        group = group.sort("sequence")
        study = group.item(0, "study")
        ax.plot(
            group.get_column("density_trend_pct_per_decade"),
            group.get_column("altitude_km"),
            color=group.item(0, "color_hex"),
            linestyle=BROWN_LINESTYLES[group.item(0, "line_style")],
            linewidth=1.25,
            zorder=2,
        )

    marked = included.filter(pl.col("marker"))
    for row in marked.iter_rows(named=True):
        ax.scatter(
            row["density_trend_pct_per_decade"],
            row["altitude_km"],
            marker=BROWN_MARKERS.get(row["study"], "o"),
            s=34,
            color=row["color_hex"],
            edgecolor="black" if BROWN_MARKERS.get(row["study"], "o") != "x" else None,
            linewidth=0.35,
            zorder=3,
        )

    for row in included.filter(pl.col("series_type") == "point").iter_rows(named=True):
        lower, upper = (
            row["uncertainty_minus_pct_per_decade"],
            row["uncertainty_plus_pct_per_decade"],
        )
        if lower is not None or upper is not None:
            lower = 0.0 if lower is None else lower
            upper = 0.0 if upper is None else upper
            ax.errorbar(
                row["density_trend_pct_per_decade"],
                row["altitude_km"],
                xerr=[[lower], [upper]],
                fmt="none",
                ecolor=row["color_hex"],
                elinewidth=0.9,
                capsize=2.0,
                zorder=1,
            )

    handles = []
    for study in BROWN_STUDY_ORDER:
        study_rows = included.filter(pl.col("study") == study)
        if study_rows.is_empty():
            continue
        first = study_rows.sort("sequence").row(0, named=True)
        handles.append(
            Line2D(
                [0],
                [0],
                color=first["color_hex"],
                linestyle=BROWN_LINESTYLES[first["line_style"]],
                marker=BROWN_MARKERS[study],
                markersize=6,
                linewidth=1.25,
                label=study,
            )
        )
    ax.set_xlim(*BROWN_X_LIMITS)
    ax.set_xlabel("Reported density trend (%/decade)")
    ax.grid(True, which="major", alpha=0.25)
    ax.minorticks_on()
    ax.grid(True, which="minor", alpha=0.10)
    ax.set_title("(a) Literature synthesis (Brown et al., 2024)", fontsize=13, pad=10)
    return handles


def validate_jb_pair(trends: pl.DataFrame, require_jb: bool = False) -> bool:
    """Require JB2006/JB2008 to be present together when either is supplied."""

    datasets = set(trends.get_column("dataset").to_list())
    included = [dataset in datasets for dataset in JB_DATASETS]
    if any(included) and not all(included):
        raise ValueError(
            "Trend CSV must include paired JB2006/JB2008 baselines together"
        )
    if require_jb and not all(included):
        raise ValueError(
            "--require-jb requires paired JB2006/JB2008 baselines in Trend CSV"
        )
    return all(included)


def figure_caption(jb_included: bool = False) -> str:
    """Return the Figure 6 caption supplied with the rendered artifacts."""

    jb_caption = (
        "Paired JB2006/JB2008 Mauna Loa curves are externally generated provider-model "
        "outputs and empirical model baselines rather than observed density trends."
        if jb_included
        else "Draft: paired JB2006/JB2008 baselines are pending licensed external model outputs."
    )
    return (
        "Figure 6. (a) Literature synthesis reconstructed from plot-precision values "
        "vector-extracted from Brown et al. (2024), Figure 2, under CC BY 4.0; these "
        "are not replacements for the original study data. Both panels share a 0–850 km "
        "altitude axis. (b) Updated solar-adjusted density trends: OLS log10 density with "
        "time, centered 81-day F10.7, and squared F10.7; error bars are 27-day "
        "Newey-West/HAC 95% intervals. Colors and line styles identify individual products; "
        "NRLMSISE-00, NRLMSIS 2.0, and NRLMSIS 2.1 are empirical climatology baselines "
        "rather than observed density trends. Markers encode record-length bins. "
        + jb_caption
    )


def figure_alt_text(jb_included: bool = False) -> str:
    """Return concise non-causal alt text for the composite."""

    jb_text = (
        " JB2006 stays near zero to minus one percent per decade, whereas JB2008 "
        "becomes progressively more negative with altitude, reaching roughly minus "
        "fourteen percent per decade."
        if jb_included
        else ""
    )
    return (
        "Side-by-side 0–850 km altitude profiles of density trends. At left, a vector "
        "reconstruction of Brown et al.'s literature synthesis shows mostly negative reported "
        "trends across altitude. At right, project-derived global mean and HASDM estimates are "
        "generally about minus four to minus seven percent per decade, empirical climatology "
        "baselines are near minus one percent per decade, and short TU Delft records have broad "
        "intervals that span zero."
        + jb_text
    )


def compose_panels(
    fig: object, brown_data: pl.DataFrame, trends: pl.DataFrame
) -> tuple[object, object]:
    """Add vector Brown and current-trend panels to ``fig`` with one shared y-axis."""

    gridspec = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.0), wspace=0.12)
    ax_a = fig.add_subplot(gridspec[0, 0])
    ax_b = fig.add_subplot(gridspec[0, 1], sharey=ax_a)
    brown_handles = plot_brown_literature(ax_a, brown_data)
    ax_a.set_ylim(*SHARED_ALTITUDE_LIMITS)
    ax_a.set_ylabel("Altitude (km)")
    ax_a.legend(
        handles=brown_handles,
        title="Study",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        fontsize=5.5,
        title_fontsize=6,
        framealpha=0.92,
        labelspacing=0.2,
        columnspacing=0.7,
        handlelength=1.2,
    )

    dataset_handles, duration_handles = plot_trends(
        ax_b, trends, altitude_limits=SHARED_ALTITUDE_LIMITS
    )
    ax_b.tick_params(labelleft=False)
    ax_b.set_ylabel("")
    ax_b.set_title("(b) Updated solar-adjusted estimates", fontsize=13, pad=10)
    dataset_legend = ax_b.legend(
        handles=dataset_handles,
        title="Dataset",
        loc="upper right",
        ncol=2,
        fontsize=7,
        title_fontsize=8,
        framealpha=0.92,
    )
    ax_b.add_artist(dataset_legend)
    ax_b.legend(
        handles=duration_handles,
        title="Record length",
        loc="lower right",
        ncol=2,
        fontsize=7,
        title_fontsize=8,
        framealpha=0.92,
    )
    return ax_a, ax_b


def compose_figure(
    brown_data_path: Path,
    trend_csv: Path,
    output_dir: Path,
    invocation: list[str] | None = None,
    require_jb: bool = False,
) -> dict[str, object]:
    """Render Figure 6 and write its caption, alt text, and provenance sidecar."""

    brown_data = load_brown_literature(brown_data_path)
    if not trend_csv.is_file():
        raise FileNotFoundError(f"Trend CSV does not exist: {trend_csv}")
    trends = pl.read_csv(trend_csv)
    jb_included = validate_jb_pair(trends, require_jb=require_jb)
    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16.5, 7.5), layout="constrained")
    compose_panels(fig, brown_data, trends)
    png_path = output_dir / "density_trend_figure6.png"
    pdf_path = output_dir / "density_trend_figure6.pdf"
    caption_path = output_dir / "density_trend_figure6_caption.txt"
    alt_text_path = output_dir / "density_trend_figure6_alt_text.txt"
    provenance_path = output_dir / "density_trend_figure6_provenance.json"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    caption_path.write_text(figure_caption(jb_included) + "\n", encoding="utf-8")
    alt_text_path.write_text(figure_alt_text(jb_included) + "\n", encoding="utf-8")

    output_paths = [png_path, pdf_path, caption_path, alt_text_path, provenance_path]
    provenance: dict[str, object] = {
        "schema_version": 4,
        "generated_utc": datetime.now(UTC).isoformat(),
        "script_path": str(Path(__file__).resolve().relative_to(Path.cwd().resolve())),
        "invocation": invocation if invocation is not None else sys.argv,
        "brown": {
            "title": BROWN_TITLE,
            "citation": BROWN_CITATION,
            "doi": BROWN_DOI,
            "source_url": BROWN_SOURCE_URL,
            "license": BROWN_LICENSE,
            "external_publisher_pdf_sha256": BROWN_PDF_SHA256,
            "external_publisher_pdf_reference_url": BROWN_PDF_REFERENCE_URL,
            "accepted_manuscript_url": BROWN_ACCEPTED_MANUSCRIPT_URL,
            "digitized_csv": {
                "path": str(brown_data_path),
                "sha256": sha256_file(brown_data_path),
                "presentation_source_sha256_before_lf_normalization": BROWN_PRESENTATION_SOURCE_SHA256,
            },
            "extraction_basis": "Plot-precision values vector-extracted from Brown et al. (2024) Figure 2 PDF.",
            "disclaimer": "Digitized values are not replacements for the original study data.",
            "rows": brown_data.height,
            "studies": brown_data.get_column("study").n_unique(),
        },
        "shared_altitude_limits": list(SHARED_ALTITUDE_LIMITS),
        "trend_csv": {"path": str(trend_csv), "sha256": sha256_file(trend_csv)},
        "jb_required": require_jb,
        "jb_included": jb_included,
        "jb_canonical_pair": list(JB_DATASETS),
        "estimator": "OLS log10 density with time, centered 81-day F10.7, and squared F10.7; 27-day Bartlett Newey-West/HAC 95% intervals.",
        "outputs": [str(path) for path in output_paths],
        "disclosure": (
            "Panel A is generated from digitized third-party figure geometry under "
            "CC BY 4.0; Panel B is generated from project data. JB inputs are externally "
            "generated provider-model outputs and are included as a canonical pair."
            if jb_included
            else "Panel A is generated from digitized third-party figure geometry under "
            "CC BY 4.0; Panel B is generated from project data. This draft omits pending "
            "paired JB provider-model outputs."
        ),
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return provenance


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the Figure 6 compositor command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brown-data", type=Path, default=DEFAULT_BROWN_DATA)
    parser.add_argument("--trend-csv", type=Path, default=DEFAULT_TREND_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--require-jb", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the Figure 6 compositor from the command line."""

    args = parse_args(argv)
    provenance = compose_figure(
        args.brown_data,
        args.trend_csv,
        args.output_dir,
        [Path(__file__).name, *(argv if argv is not None else sys.argv[1:])],
        args.require_jb,
    )
    print(
        f"Wrote Figure 6 artifacts to {args.output_dir} ({provenance['brown']['digitized_csv']['sha256']})"
    )


if __name__ == "__main__":
    main()
