"""Forest plot of paired CO2 gains (pp of MAPE) with 95% paired-month intervals (paper figure).

Values are copied from the receipts: docs/research/karman-bounded-closeout-2026-10-05.md,
co2-paper-execution-2026-10-06.md, full25-closeout-2026-10-10.md and
cheap-control-results-2026-10-10.md. Run: ``uv run python -m scripts.plot_prediction_gains``.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("outputs/figures/results/prediction_gains")

rows = [  # label, all-21-month (mid, lo, hi) or None, original-16-month (mid, lo, hi)
    ("Neural, JB2008+OMNI (to 2019)", None, (0.072, -0.108, 0.271)),
    ("Neural, JB2008+OMNI (to 2025)", (0.122, 0.028, 0.209), (0.123, 0.028, 0.211)),
    ("Neural, OMNI residual head", (-0.037, -0.214, 0.144), (0.005, -0.189, 0.213)),
    ("Neural, static residual", (-0.180, -0.290, -0.065), (-0.170, -0.300, -0.041)),
    ("Ridge, D−C (CO₂ beyond spline)", (-0.057, -0.142, 0.036), (-0.084, -0.175, 0.011)),
    ("Boosting, D−C", (-0.025, -0.159, 0.074), (-0.054, -0.221, 0.057)),
    ("Ridge, B−P (CO₂ vs placebo)", (-0.026, -0.218, 0.151), None),
    ("Boosting, B−P", (-0.068, -0.286, 0.113), None),
]
style = {"font.family": "sans-serif", "font.size": 8.5}
with matplotlib.rc_context(style):
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.axvline(0, color="#1f1f1e", lw=1)
    for i, (label, a, b) in enumerate(rows):
        y = len(rows) - 1 - i
        for vals, dy, color, name in ((a, 0.14, "#1f77b4", "all 21 test months"), (b, -0.14, "#ff7f0e", "16 original months")):
            if vals:
                m, lo, hi = vals
                ax.errorbar(m, y + dy, xerr=[[m - lo], [hi - m]], fmt="o", color=color, ms=4, capsize=2, lw=1.2,
                            label=name if i in (1,) else None)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    ax.set_xlabel("gain from adding CO₂, MAPE without − with (percentage points)")
    ax.grid(axis="x", color="#e6e5df")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "prediction_gains.pdf")
    fig.savefig(OUT / "prediction_gains.png", dpi=180)
