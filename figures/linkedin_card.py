"""Render the results card for LinkedIn: figures/jev_vs_spacy.png (1600x900).

uv run --with matplotlib python figures/linkedin_card.py
"""

from pathlib import Path

import matplotlib.pyplot as plt

SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
# Reference palette, categorical slots 1-3 in fixed order (validated all-pairs for three series).
CONTESTANTS = [
    ("Jev, zero-shot (0 labels)", "#2a78d6"),
    ("spaCy, 20 labels per class", "#eb6834"),
    ("spaCy, full training set (11k–24k labels)", "#1baf7a"),
]
PANELS = [
    ("Spam detection · Enron-Spam\n2 classes, 300 emails", [98.0, 73.7, 98.0]),
    ("Support routing · Bitext\n11 classes, 275 messages", [97.8, 94.2, 100.0]),
    ("Swedish routing · MASSIVE sv\n18 classes, 360 requests", [88.9, 68.3, 84.4]),
]
LATENCY = "Median latency: Jev ~300 ms per call (API) · spaCy 0.4–2 ms (local CPU)"

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})
fig, axes = plt.subplots(1, 3, figsize=(16, 9), dpi=100, facecolor=SURFACE)
fig.subplots_adjust(left=0.04, right=0.97, top=0.68, bottom=0.2, wspace=0.14)

fig.text(
    0.04,
    0.92,
    "Zero-shot Jev matched spaCy on English spam and beat it in Swedish",
    fontsize=26,
    weight="bold",
    color=INK,
)
fig.text(
    0.04,
    0.86,
    "Accuracy on held-out test sets, same rows and scorers for every model · "
    "Swedish spaCy uses sv_core_news_md vectors",
    fontsize=17,
    color=INK_2,
)

for ax, (title, values) in zip(axes, PANELS, strict=True):
    ax.set_facecolor(SURFACE)
    rows = range(len(CONTESTANTS))[::-1]
    ax.barh(list(rows), values, height=0.6, color=[c for _, c in CONTESTANTS], edgecolor=SURFACE, linewidth=2)
    for y, v in zip(rows, values, strict=True):
        ax.text(v - 1.5, y, f"{v:.1f}%", va="center", ha="right", fontsize=19, weight="bold", color="#ffffff")
    ax.set_xlim(0, 100)
    ax.set_title(title, loc="left", fontsize=18, color=INK, pad=12, linespacing=1.4)
    ax.set_yticks([])
    ax.set_xticks([0, 50, 100], ["0", "50", "100%"], fontsize=14, color=MUTED)
    ax.tick_params(length=0)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ax.spines.values():
        side.set_visible(False)

handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in CONTESTANTS]
fig.legend(
    handles,
    [n for n, _ in CONTESTANTS],
    loc="lower left",
    bbox_to_anchor=(0.035, 0.085),
    ncol=3,
    frameon=False,
    fontsize=17,
    handlelength=1.2,
    columnspacing=2.2,
)
fig.text(0.04, 0.035, LATENCY + "   ·   Braintrust + Pydantic AI eval, Sept 2026", fontsize=15, color=INK_2)

out = Path(__file__).with_name("jev_vs_spacy.png")
fig.savefig(out, facecolor=SURFACE)
print(out)
