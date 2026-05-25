"""
Visualize top important tokens per record from iter_dpmlm explanation logs.
Reads explanations.md, extracts peak token scores from [refreshed] steps,
and produces a publication-quality figure for paper appendix.
"""

import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

LOG_FILE = "explanations.md"
OUTPUT_FILE = "docs/paper/plots/tab_top_tokens.pdf"
TOP_K = 10

# ── parse ─────────────────────────────────────────────────────────────────────

line_re = re.compile(
    r"record=(?P<rec>\S+)\s+step=(?P<step>\d+)\s+(?P<refresh>\[refreshed\]\s+)?.*?"
    r"top5=\[(?P<top5>[^\]]+)\]"
)
token_re = re.compile(r"'([^']+)\(([0-9.]+)\)'")

records = {}   # record_id -> {token -> max_score}
record_order = []

with open(LOG_FILE) as f:
    for raw in f:
        m = line_re.search(raw)
        if not m:
            continue
        rec = m.group("rec")
        is_refresh = bool(m.group("refresh"))
        top5_str = m.group("top5")

        if rec not in records:
            records[rec] = {}
            record_order.append(rec)

        if not is_refresh:
            continue  # only use refreshed scores (most accurate)

        for tok, score in token_re.findall(top5_str):
            score = float(score)
            records[rec][tok] = max(records[rec].get(tok, 0.0), score)

# ── token categorise ──────────────────────────────────────────────────────────

STOP = {
    "the", "The", "a", "an", "of", "in", "to", "and", "by", "on", "with",
    "for", "was", "were", "is", "are", "that", "this", "its", "their",
    "had", "has", "at", "it", "as", "be", "from", "or", "not", "which",
    "between", "no.", '"the', "each", "two", "four", "one", "three",
}

def token_color(tok):
    if tok in ("DATE_TIME",):
        return "#4C72B0"   # blue
    if tok in ("LOCATION",):
        return "#DD8452"   # orange
    if tok in ("PERSON",):
        return "#55A868"   # green
    return "#937860"       # brown for regular tokens

def is_stop(tok):
    return tok.lower() in {s.lower() for s in STOP}

def top_tokens(rec_scores, k, exclude_stop=True):
    items = [(t, s) for t, s in rec_scores.items()
             if not (exclude_stop and is_stop(t))]
    items.sort(key=lambda x: -x[1])
    return items[:k]

# ── record labels ─────────────────────────────────────────────────────────────

RECORD_LABELS = {
    "001-61807": "001-61807\n(Turkey – Sinop detention)",
    "001-61253": "001-61253\n(UK – doctor-patient privilege)",
    "001-66929": "001-66929\n(Turkey – Izmir Security Court)",
    "001-159913": "001-159913\n(Poland – Opole/Lubelskie Prison)",
    "001-61227": "001-61227\n(UK – HM Prison Craiginches)",
    "001-90749": "001-90749\n(Cyprus – property rights)",
}

# ── plot ──────────────────────────────────────────────────────────────────────

n = len(record_order)
ncols = 2
nrows = (n + 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(13, nrows * 3.4))
axes = axes.flatten()

for ax, rec in zip(axes, record_order):
    items = top_tokens(records[rec], TOP_K)
    if not items:
        ax.axis("off")
        continue

    tokens, scores = zip(*items)
    colors = [token_color(t) for t in tokens]

    y = np.arange(len(tokens))
    bars = ax.barh(y, scores, color=colors, height=0.65, edgecolor="white", linewidth=0.4)

    # value labels
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + max(scores) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{score:.4f}", va="center", ha="left", fontsize=7.5, color="#333333")

    ax.set_yticks(y)
    ax.set_yticklabels(tokens, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Peak importance score", fontsize=8)
    ax.set_title(RECORD_LABELS.get(rec, rec), fontsize=9, fontweight="bold", pad=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.3f"))
    ax.set_xlim(0, max(scores) * 1.22)

# hide unused axes
for ax in axes[n:]:
    ax.axis("off")

# legend
legend_patches = [
    mpatches.Patch(color="#4C72B0", label="DATE_TIME (anonymized)"),
    mpatches.Patch(color="#DD8452", label="LOCATION (anonymized)"),
    # mpatches.Patch(color="#55A868", label="PERSON (anonymized)"),
    mpatches.Patch(color="#937860", label="Surface token"),
]
fig.legend(handles=legend_patches, loc="lower center", ncol=4,
           fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.04, 1, 1])

import os
os.makedirs("docs/paper/plots", exist_ok=True)
plt.savefig(OUTPUT_FILE, bbox_inches="tight", dpi=200)
plt.savefig(OUTPUT_FILE.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
print(f"Saved → {OUTPUT_FILE}")
print(f"Saved → {OUTPUT_FILE.replace('.pdf', '.png')}")
