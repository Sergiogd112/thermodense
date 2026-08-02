"""PROTOTYPE — generate sample review figures from real PCMCI+ results.

This is throwaway code for the issue-15 figure-review workbench prototype.
It reads the committed real results under benchmarks/pcmci-methods/results/real-1/local/
and writes figures/*.png plus data.json (figure records with provenance,
claims, and journal profiles) that index.html consumes.

Regenerate any time:  PYTHONPATH=src python -m thermodense.prototypes.figure_review.make_figures
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

# Keep the publication artifact friendly to pdfLaTeX/LuaLaTeX: vector graphics
# with embedded TrueType fonts rather than Matplotlib's default Type 3 glyphs.
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[3] / "benchmarks" / "pcmci-methods" / "results" / "real-1" / "local"
FIG_DIR = HERE / "figures"

NODES = [
    "f10_7_center81",
    "ap_avg",
    "co2_ppm",
    "log10rho_325_daily_mean",
    "log10rho_825_daily_mean",
]
SHORT = {
    "f10_7_center81": "F10.7 (81-d mean)",
    "ap_avg": "Ap index",
    "co2_ppm": "CO$_2$ (ppm)",
    "log10rho_325_daily_mean": r"$\log_{10}\rho$ 325 km",
    "log10rho_825_daily_mean": r"$\log_{10}\rho$ 825 km",
}
DRIVER = {"f10_7_center81", "ap_avg", "co2_ppm"}

POS = {
    "f10_7_center81": (0.0, 1.0),
    "ap_avg": (1.0, 1.0),
    "co2_ppm": (2.0, 1.0),
    "log10rho_325_daily_mean": (0.5, 0.0),
    "log10rho_825_daily_mean": (1.5, 0.0),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_graph_links(npz_path: Path, n_phys: int) -> dict:
    """Decode a tigramite string graph into {(i, j, tau): (mark, val)} for the first n_phys nodes."""
    a = np.load(npz_path)
    g = a["graph"]
    val = a["val_matrix"]
    links = {}
    for i in range(n_phys):
        for j in range(n_phys):
            for tau in range(g.shape[2]):
                m = str(g[i, j, tau]).strip()
                if m:
                    links[(i, j, tau)] = (m, float(val[i, j, tau]))
    return links


def load_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def run_record(name: str) -> dict:
    return json.loads((RESULTS / name).read_text().splitlines()[0])


def cross_stats(artifact_dir: str) -> tuple:
    """Count physical->surrogate control links and their median/max |val|."""
    rows = [
        r
        for r in load_rows(RESULTS / artifact_dir / "control_links.jsonl")
        if r.get("relation") == "physical→surrogate"
    ]
    n = len(rows)
    if n:
        av = [abs(r["val"]) for r in rows]
        return n, float(np.median(av)), float(np.max(av))
    return 0, 0.0, 0.0


def cross_links(links: dict) -> dict:
    """Lagged (tau>0) links between different physical nodes, keyed by (i, j)."""
    out = {}
    for (i, j, tau), (m, v) in links.items():
        if tau > 0 and i != j:
            out.setdefault((i, j), []).append((tau, m, v))
    return out


def draw_graph(ax, links: dict, retained=None, title=""):
    """Draw the 5-node physical graph. retained: set of (i,j,tau) kept, else all solid."""
    G = nx.DiGraph()
    G.add_nodes_from(NODES)
    node_color = ["#5b8def" if n in DRIVER else "#e67e22" for n in G.nodes()]
    nx.draw_networkx_nodes(G, POS, ax=ax, node_color=node_color, node_size=2600)
    nx.draw_networkx_labels(
        G, POS, ax=ax,
        labels={n: SHORT[n] for n in NODES},
        font_size=9, font_color="white",
    )
    for (i, j, tau), (m, v) in links.items():
        if tau == 0 and m == "x-x" and i < j:
            ax.plot(
                [POS[NODES[i]][0], POS[NODES[j]][0]],
                [POS[NODES[i]][1], POS[NODES[j]][1]],
                ls="--", c="#7f8c8d", lw=1.6, zorder=1,
            )
            ax.text(
                (POS[NODES[i]][0] + POS[NODES[j]][0]) / 2,
                (POS[NODES[i]][1] + POS[NODES[j]][1]) / 2 + 0.07,
                f"{v:+.2f}", ha="center", fontsize=8, color="#34495e",
            )
    for (i, j), ts in cross_links(links).items():
        keep = retained is None or all((i, j, t) in retained for t, m, v in ts)
        style = dict(ls="-", lw=1.8, zorder=2) if keep else dict(
            ls=(0, (4, 2)), lw=1.2, zorder=2, alpha=0.35
        )
        c = "#c0392b" if np.mean([v for t, m, v in ts]) >= 0 else "#2980b9"
        if not keep:
            c = "#bdc3c7"
        ax.annotate(
            "", xy=POS[NODES[i]], xytext=POS[NODES[j]],
            arrowprops=dict(arrowstyle="-|>", color=c, **style),
        )
        ax.text(
            (POS[NODES[i]][0] + POS[NODES[j]][0]) / 2 + 0.06,
            (POS[NODES[i]][1] + POS[NODES[j]][1]) / 2 + 0.06,
            "τ=" + ",".join(str(t) for t in sorted(t for t, m, v in ts)),
            fontsize=7.5, color=c, ha="center",
        )
    for i, n in enumerate(NODES):
        selfs = sorted(t for (ii, jj, t), (m, v) in links.items() if ii == jj == i and t > 0)
        if selfs:
            ax.text(
                POS[n][0], POS[n][1] - 0.14,
                "self: " + ",".join(str(t) for t in selfs),
                ha="center", fontsize=6.8, color="#7f8c8d",
            )
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.45, 1.45)
    ax.axis("off")


def figure_01(primary, iaaft_kept, shift_kept, iaaft_cross, shift_cross):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    draw_graph(axes[0], primary, title="(a) Primary PCMCI+ graph (42 links)")
    draw_graph(
        axes[1], primary, retained=iaaft_kept,
        title=f"(b) IAAFT control: {len(iaaft_kept)}/42 retained · {iaaft_cross} control→physical",
    )
    draw_graph(
        axes[2], primary, retained=shift_kept,
        title=f"(c) Circular-shift control: {len(shift_kept)}/42 retained · {shift_cross} control→physical",
    )
    fig.suptitle("Structural-control retention of the canonical Ap–density links", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def figure_02(counts, cross):
    """counts: dict family -> (retained, added); cross: dict family -> (n, median, max)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    families = ["Primary", "IAAFT", "Shift", "Combined"]
    retained = [counts[f][0] for f in families]
    added = [0] + [counts[f][1] for f in families[1:]]
    x = np.arange(len(families))
    axes[0].bar(x, retained, 0.5, label="primary links retained", color="#27ae60")
    axes[0].bar(x, added, 0.5, bottom=retained, label="links added", color="#f39c12")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(families)
    axes[0].set_ylabel("physical links")
    axes[0].set_title("(a) Physical links under structural control")
    axes[0].legend(fontsize=8)
    fams = list(cross)
    axes[1].bar(np.arange(len(fams)), [cross[f][0] for f in fams], 0.5, color="#8e44ad")
    for k, f in enumerate(fams):
        axes[1].text(
            k, cross[f][0] + 0.2,
            f"med |r| {cross[f][1]:.3f}\nmax |r| {cross[f][2]:.3f}",
            ha="center", fontsize=8,
        )
    axes[1].set_xticks(range(len(fams)))
    axes[1].set_xticklabels(fams)
    axes[1].set_ylabel("control→physical links")
    axes[1].set_title("(b) Spurious control→physical links")
    fig.suptitle("Retention by structural-control family", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def figure_03(rel_counts, fam_counts, surrogate_phys_count):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    rels = list(rel_counts)
    axes[0].bar(np.arange(len(rels)), [rel_counts[r] for r in rels], 0.5, color="#16a085")
    axes[0].axhline(42, ls="--", c="#c0392b", lw=1.2, label="primary (42)")
    axes[0].set_yscale("log")
    axes[0].set_xticks(range(len(rels)))
    axes[0].set_xticklabels([r.replace("\u2194", "↔") for r in rels], fontsize=8)
    axes[0].set_ylabel("links (log)")
    axes[0].set_title("(a) Surrogate-run link relations")
    axes[0].legend(fontsize=8)
    fams = list(fam_counts)
    axes[1].bar(np.arange(len(fams)), [fam_counts[f] for f in fams], 0.45, color="#d35400")
    axes[1].bar(
        len(fams), surrogate_phys_count, 0.45, color="#27ae60",
        label="physical links in surrogate run",
    )
    axes[1].set_xticks(list(range(len(fams) + 1)))
    axes[1].set_xticklabels(fams + ["phys"], fontsize=9)
    axes[1].set_ylabel("links")
    axes[1].set_title("(b) Surrogate links by family")
    axes[1].legend(fontsize=8)
    fig.suptitle("White-noise and periodic surrogates", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)

    rec_primary = run_record("results-parcorr.jsonl")
    rec_iaaft = run_record("results-parcorr-controls-iaaft.jsonl")
    rec_shift = run_record("results-parcorr-controls-shift.jsonl")
    rec_comb = run_record("results-parcorr-controls.jsonl")
    rec_surr = run_record("results-parcorr-surrogates.jsonl")

    primary = load_graph_links(RESULTS / "results-parcorr_artifacts" / "parcorr.npz", 5)
    iaaft_links = load_graph_links(
        RESULTS / "results-parcorr-controls-iaaft_artifacts" / "parcorr.npz", 5
    )
    shift_links = load_graph_links(
        RESULTS / "results-parcorr-controls-shift_artifacts" / "parcorr.npz", 5
    )
    comb_links = load_graph_links(
        RESULTS / "results-parcorr-controls_artifacts" / "parcorr.npz", 5
    )
    surr_phys = load_graph_links(
        RESULTS / "results-parcorr-surrogates_artifacts" / "parcorr.npz", 5
    )

    primary_set = set(primary)
    iaaft_kept = primary_set & set(iaaft_links)
    shift_kept = primary_set & set(shift_links)
    comb_kept = primary_set & set(comb_links)
    iaaft_added = set(iaaft_links) - primary_set
    shift_added = set(shift_links) - primary_set
    comb_added = set(comb_links) - primary_set

    iaaft_cross = cross_stats("results-parcorr-controls-iaaft_artifacts")
    shift_cross = cross_stats("results-parcorr-controls-shift_artifacts")
    comb_cross_by_family = {}
    for r in load_rows(RESULTS / "results-parcorr-controls_artifacts" / "control_links.jsonl"):
        if r["relation"] == "physical→surrogate":
            fam = r["target_family"]
            comb_cross_by_family.setdefault(fam, []).append(abs(r["val"]))
    comb_cross = {
        fam: (len(av), float(np.median(av)), float(np.max(av)))
        for fam, av in comb_cross_by_family.items()
    }

    counts = {
        "Primary": (len(primary_set), 0),
        "IAAFT": (len(iaaft_kept), len(iaaft_added)),
        "Shift": (len(shift_kept), len(shift_added)),
        "Combined": (len(comb_kept), len(comb_added)),
    }
    cross = {"IAAFT": iaaft_cross, "Shift": shift_cross, **comb_cross}

    rel_counts = {}
    fam_counts = {}
    for r in load_rows(RESULTS / "results-parcorr-surrogates_artifacts" / "surrogate_links.jsonl"):
        rel_counts[r["relation"]] = rel_counts.get(r["relation"], 0) + 1
        tgt = r["target"]
        if tgt.startswith("surrogate_"):
            fam = tgt.split("_")[1]
            fam_counts[fam] = fam_counts.get(fam, 0) + 1

    figs = {
        "figure-01-graph.png": figure_01(
            primary, iaaft_kept, shift_kept, iaaft_cross[0], shift_cross[0]
        ),
        "figure-02-retention.png": figure_02(counts, cross),
        "figure-03-surrogates.png": figure_03(rel_counts, fam_counts, len(surr_phys)),
    }
    for name, fig in figs.items():
        path = FIG_DIR / name
        pdf_path = path.with_suffix(".pdf")
        fig.savefig(path, dpi=150)
        fig.savefig(
            pdf_path,
            format="pdf",
            bbox_inches="tight",
            metadata={
                "Creator": "thermodense figure-review prototype",
                "CreationDate": None,
                "ModDate": None,
            },
        )
        plt.close(fig)
        print(f"wrote {path}  sha256={sha256(path)[:16]}…")
        print(f"wrote {pdf_path}  sha256={sha256(pdf_path)[:16]}…")

    artifacts = {
        "fig-01": [
            ("results-parcorr.jsonl", rec_primary["artifact"]["sha256"]),
            ("results-parcorr-controls-iaaft.jsonl", rec_iaaft["artifact"]["sha256"]),
            ("results-parcorr-controls-shift.jsonl", rec_shift["artifact"]["sha256"]),
        ],
        "fig-02": [
            ("results-parcorr-controls-iaaft.jsonl", rec_iaaft["artifact"]["sha256"]),
            ("results-parcorr-controls-shift.jsonl", rec_shift["artifact"]["sha256"]),
            ("results-parcorr-controls.jsonl", rec_comb["artifact"]["sha256"]),
        ],
        "fig-03": [
            ("results-parcorr-surrogates.jsonl", rec_surr["artifact"]["sha256"]),
        ],
    }
    titles = {
        "fig-01": "Primary PCMCI+ graph and structural-control retention",
        "fig-02": "Physical-link retention and spurious control→physical links",
        "fig-03": "White-noise and periodic surrogate evidence",
    }
    captions = {
        "fig-01": (
            "Sample review figure generated from the committed real ParCorr results "
            "(prototype renderer). (a) primary 42-link graph; (b) links retained under the "
            "IAAFT structural control; (c) links retained under the circular-shift control. "
            "Dashed grey contemporaneous edges are labelled with the partial correlation; "
            "arrows are lagged cross links with lag τ."
        ),
        "fig-02": (
            "(a) physical links retained vs added under each structural-control family; "
            "(b) spurious control→physical links with median and max |r|."
        ),
        "fig-03": (
            "(a) relation types in the combined white-noise/periodic surrogate run "
            "(log scale, dashed line = 42 primary links); (b) surrogate links by family "
            "and the physical-subgraph link count of the surrogate run."
        ),
    }
    panels = {
        "fig-01": [
            ("a", "Primary ParCorr graph"),
            ("b", "IAAFT control retention"),
            ("c", "Circular-shift control retention"),
        ],
        "fig-02": [
            ("a", "Retention by family"),
            ("b", "Spurious cross links"),
        ],
        "fig-03": [
            ("a", "Surrogate relation types"),
            ("b", "Surrogate links by family"),
        ],
    }

    figures = []
    for fid, fname in [
        ("fig-01", "figure-01-graph.png"),
        ("fig-02", "figure-02-retention.png"),
        ("fig-03", "figure-03-surrogates.png"),
    ]:
        path = FIG_DIR / fname
        pdf_path = path.with_suffix(".pdf")
        figures.append({
            "id": fid,
            "title": titles[fid],
            "src": f"figures/{fname}",
            "sha256": sha256(path),
            "publicationSrc": f"figures/{pdf_path.name}",
            "publicationSha256": sha256(pdf_path),
            "publicationFormat": "application/pdf",
            "caption": captions[fid],
            "panels": [{"id": p, "label": lab} for p, lab in panels[fid]],
            "claimCardIds": ["claim-01", "claim-02"] if fid in ("fig-01", "fig-02") else [],
            "provenance": {
                "sourceArtifacts": [
                    {
                        "name": n,
                        "path": f"benchmarks/pcmci-methods/results/real-1/local/{n}",
                        "sha256": h,
                    }
                    for n, h in artifacts[fid]
                ],
                "renderer": "prototype make_figures.py (issue #15)",
                "advisory": (
                    "Prototype renderer; figures are interface samples, "
                    "not final paper figures."
                ),
            },
        })

    data = {
        "prototype": True,
        "figureSetVersion": "real-1-local-v2",
        "figures": figures,
        "claims": [
            {
                "id": "claim-01",
                "text": (
                    "Canonical Ap–density links (contemporaneous Ap↔ρ₃₂₅, Ap↔ρ₈₂₅; "
                    "lagged Ap→ρ₃₂₅ at 53 d) are retained under family-separated "
                    "structural controls (IAAFT, circular shift)."
                ),
                "evidenceFigureIds": ["fig-01", "fig-02"],
            },
            {
                "id": "claim-02",
                "text": (
                    "Spurious control→physical links are few (7 IAAFT, 4 shift) and weak "
                    "(median |r| ≈ 0.04) relative to the canonical links."
                ),
                "evidenceFigureIds": ["fig-02"],
            },
        ],
        "profiles": {
            "agu-wiley": {
                "name": "AGU / Wiley",
                "warnings": [
                    "No introduce / remove / move / obscure of image content",
                    "Data and software availability statement required",
                    "Preferred 300 dpi for line art",
                ],
            },
            "copernicus": {
                "name": "Copernicus",
                "warnings": [
                    "Minimum figure width 8 cm",
                    "300 dpi resolution",
                    "Fonts embedded in figure files",
                    "Captions in the text, not inside figure files",
                ],
            },
        },
    }
    (HERE / "data.json").write_text(json.dumps(data, indent=1))
    print("wrote data.json")


if __name__ == "__main__":
    main()
