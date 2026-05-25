"""
Plot SHAP token attributions per record from data/{ds}/tri_risk/shap.jsonl.
For each record shows three groups (top-to-bottom):
  • top 5 highest-scoring tokens  (positive SHAP → risk-increasing)
  • 2 tokens closest to zero      (neutral)
  • 5 lowest-scoring tokens       (negative SHAP → risk-reducing)
De-identified token text is extracted from presidio/{ds}.jsonl.

Usage:
    python plot_shap_tokens.py          # both datasets
    python plot_shap_tokens.py tab
    python plot_shap_tokens.py db_bio
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")
from dp.loaders import get_adapter

# ── constants ──────────────────────────────────────────────────────────────────

N_HIGH  = 5
N_ZERO  = 2
N_LOW   = 5
NCOLS   = 2

STOP_LOWER = {
    "the", "a", "an", "of", "in", "to", "and", "by", "on", "with",
    "for", "was", "were", "is", "are", "that", "this", "its", "their",
    "had", "has", "at", "it", "as", "be", "from", "or", "not", "which",
    "between", "no.", "each", "two", "four", "one", "three",
}

ENTITY_COLORS = {
    "DATE_TIME":    "#4C72B0",
    "LOCATION":     "#DD8452",
    "PERSON":       "#55A868",
    "NRP":          "#8172B2",
    "ORGANIZATION": "#C44E52",
}
DEFAULT_COLOR  = "#937860"
ALPHA_POS      = 0.90
ALPHA_NEAR     = 0.55
ALPHA_NEG      = 0.45


def _color(tok, alpha):
    base = ENTITY_COLORS.get(tok, DEFAULT_COLOR)
    r, g, b = int(base[1:3], 16)/255, int(base[3:5], 16)/255, int(base[5:7], 16)/255
    return (r, g, b, alpha)


def _is_stop(tok):
    return tok.lower() in STOP_LOWER


# ── dataset configs ────────────────────────────────────────────────────────────

DATASETS = {
    "tab": {
        "data_in": "data/tab",
        "presidio": "presidio/tab.jsonl",
        "shap":    "data/tab/tri_risk/shap.jsonl",
        "output":  "docs/paper/plots/tab_top_tokens",
        "selected_uids": [
            "001-61807",
            "001-61253",
            "001-66929",
            "001-159913",
            "001-61227",
            "001-90749",
            "001-82311",
            "001-77315",
        ],
        "labels": {
            "001-61807":  "001-61807  (Turkey – Sinop detention)",
            "001-61253":  "001-61253  (UK – doctor-patient privilege)",
            "001-66929":  "001-66929  (Turkey – Izmir Security Court)",
            "001-159913": "001-159913  (Poland – Opole/Lubelskie Prison)",
            "001-61227":  "001-61227  (UK – HM Prison Craiginches)",
            "001-90749":  "001-90749  (Cyprus – property rights)",
            "001-82311":  "001-82311  (Turkey – gendarmerie / acquittal)",
            "001-77315":  "001-77315  (Poland – Sieradz Regional)",
        },
    },
    "db_bio": {
        "data_in": "data/db_bio",
        "presidio": "presidio/db_bio.jsonl",
        "shap":    "data/db_bio/tri_risk/shap.jsonl",
        "output":  "docs/paper/plots/db_bio_top_tokens",
        "selected_uids": [
            "Antonio_Barbalonga",
            "Daniel_Huger",
            "Amanda_Elliott",
            "William_Oliver_(physician)",
            "Andreas_Dahl",
            "Yosef_Tekoah",
            "Daniel_Sturridge",
            "Elseid_Hysaj",
        ],
        "labels": {
            "Antonio_Barbalonga":         "Antonio Barbalonga  (Baroque painter)",
            "Daniel_Huger":               "Daniel Huger  (SC statesman)",
            "Amanda_Elliott":             "Amanda Elliott  (Koper/Slovenia)",
            "William_Oliver_(physician)": "William Oliver  (physician)",
            "Andreas_Dahl":               "Andreas Dahl  (Scandinavian football)",
            "Yosef_Tekoah":               "Yosef Tekoah  (Ben-Gurion / 1975–81)",
            "Daniel_Sturridge":           "Daniel Sturridge  (footballer)",
            "Elseid_Hysaj":               "Elseid Hysaj  (Coppa Italia)",
        },
    },
}


# ── data loading ───────────────────────────────────────────────────────────────

def load_presidio(path):
    table = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            table[r["idx"]] = r["text"]
    return table


def load_shap(path):
    """uid → (offsets, scores), scores already sorted descending."""
    table = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            table[r["uid"]] = (r["offsets"], r["scores"])
    return table


def build_uid_to_idx(dataset, data_in):
    adapter = get_adapter(dataset, data=dataset, data_in=data_in)
    return {str(rec.uid): i for i, rec in enumerate(adapter.iter_records())}


def select_token_groups(uid, uid_to_idx, pres_table, shap_table):
    """Return (high, near_zero, low) each a list of (token_str, score).

    high     – N_HIGH tokens with highest scores (descending)
    near_zero– N_ZERO tokens closest to 0 from the remainder (descending)
    low      – N_LOW  tokens with lowest scores (descending, so least-neg first)
    """
    idx = uid_to_idx.get(uid)
    if idx is None or idx not in pres_table:
        return [], [], []
    text   = pres_table[idx]
    offs, scores = shap_table.get(uid, ([], []))

    # collect non-stop tokens; shap already sorted descending
    all_toks = []
    for (s, e), sc in zip(offs, scores):
        tok = text[s:e]
        if not _is_stop(tok) and len(tok) > 1:
            all_toks.append((tok, sc))

    if not all_toks:
        return [], [], []

    # high: first N_HIGH
    high = all_toks[:N_HIGH]

    # low: last N_LOW (scores already ascending at end since sorted desc)
    low_raw = all_toks[max(N_HIGH, len(all_toks) - N_LOW):]
    # reverse so least-negative comes first (top of the low group)
    low = list(reversed(low_raw))

    # near-zero: from the middle, pick N_ZERO closest to 0
    used = set(range(N_HIGH)) | set(range(len(all_toks) - N_LOW, len(all_toks)))
    middle = [(i, tok, sc) for i, (tok, sc) in enumerate(all_toks)
              if i not in used]
    middle.sort(key=lambda x: abs(x[2]))
    near_zero_raw = [(tok, sc) for _, tok, sc in middle[:N_ZERO]]
    # sort descending so positive near-zero is above negative near-zero
    near_zero = sorted(near_zero_raw, key=lambda x: -x[1])

    return high, near_zero, low


# ── plotting ───────────────────────────────────────────────────────────────────

def _draw_panel(ax, high, near_zero, low, title):
    """Draw one diverging-bar panel for a single record."""
    # flatten into display order: high → near_zero → low
    groups = [
        (high,      ALPHA_POS),
        (near_zero, ALPHA_NEAR),
        (low,       ALPHA_NEG),
    ]
    display = []
    group_ends = []  # y-indices where each group ends (for separator lines)
    for items, alpha in groups:
        for tok, sc in items:
            display.append((tok, sc, alpha))
        group_ends.append(len(display) - 1)

    if not display:
        ax.axis("off")
        return

    tokens = [tok for tok, _, _ in display]
    scores  = np.array([sc  for _, sc, _ in display])
    alphas  = [a   for _, _, a in display]
    colors  = [_color(tok, alpha) for (tok, _, alpha) in display]

    y = np.arange(len(display))
    bars = ax.barh(y, scores, color=colors, height=0.65, edgecolor="white", linewidth=0.3)

    # zero reference line
    ax.axvline(x=0, color="#444444", linewidth=0.6, zorder=3)

    # score value labels
    abs_max = max(abs(scores)) if len(scores) else 1.0
    label_pad = abs_max * 0.03
    for bar, (tok, sc, _) in zip(bars, display):
        if sc >= 0:
            lx, ha = sc + label_pad, "left"
        else:
            lx, ha = sc - label_pad, "right"
        ax.text(lx, bar.get_y() + bar.get_height() / 2,
                f"{sc:+.4f}", va="center", ha=ha, fontsize=7, color="#333333")

    # group separator lines with count label
    group_sizes = [len(g) for g, _ in groups]
    for i, end_idx in enumerate(group_ends[:-1]):
        y_sep = end_idx + 0.5
        count = group_sizes[i]
        ax.axhline(y=y_sep, color="#aaaaaa", linewidth=0.5, linestyle="--", zorder=2)
        ax.text(0, y_sep, f"({count})", ha="center", va="center",
                fontsize=6.5, color="#777777",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5), zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(tokens, fontsize=8.5)
    ax.invert_yaxis()

    # symmetric limits: zero centred, equal space on each side
    half = abs_max * 1.45
    ax.set_xlim(-half, half)

    ax.set_xlabel("SHAP attribution score", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=6)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)   # remove tick marks, keep labels
    ax.tick_params(axis="x", labelsize=7.5)
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.3f"))


def plot_dataset(ds_name, cfg):
    uid_to_idx = build_uid_to_idx(ds_name, cfg["data_in"])
    pres_table = load_presidio(cfg["presidio"])
    shap_table = load_shap(cfg["shap"])

    selected = cfg["selected_uids"]
    labels   = cfg["labels"]
    n        = len(selected)
    nrows    = (n + NCOLS - 1) // NCOLS
    n_rows_per_panel = N_HIGH + N_ZERO + N_LOW  # 12

    fig, axes = plt.subplots(nrows, NCOLS,
                              figsize=(13, nrows * (n_rows_per_panel * 0.32 + 1.1)))
    axes = axes.flatten()

    for ax, uid in zip(axes, selected):
        high, near_zero, low = select_token_groups(uid, uid_to_idx, pres_table, shap_table)
        _draw_panel(ax, high, near_zero, low, labels.get(uid, uid))

    for ax in axes[n:]:
        ax.axis("off")

    # legend: entity types + alpha-coded direction key
    entity_patches = [
        mpatches.Patch(color=ENTITY_COLORS["DATE_TIME"],    label="DATE_TIME (anon.)"),
        mpatches.Patch(color=ENTITY_COLORS["LOCATION"],     label="LOCATION (anon.)"),
        mpatches.Patch(color=ENTITY_COLORS["PERSON"],       label="PERSON (anon.)"),
        mpatches.Patch(color=ENTITY_COLORS["NRP"],          label="NRP (anon.)"),
        mpatches.Patch(color=DEFAULT_COLOR,                  label="Surface token"),
    ]
    direction_patches = [
        mpatches.Patch(facecolor="#888888", alpha=ALPHA_POS,  label="Risk-increasing (top 5)"),
        mpatches.Patch(facecolor="#888888", alpha=ALPHA_NEAR, label="Near-zero (2)"),
        mpatches.Patch(facecolor="#888888", alpha=ALPHA_NEG,  label="Risk-reducing (bottom 5)"),
    ]
    fig.legend(handles=entity_patches + direction_patches,
               loc="lower center", ncol=4,
               fontsize=7.8, frameon=False, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.045, 1, 1])

    out = cfg["output"]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(f"{out}.pdf", bbox_inches="tight", dpi=200)
    plt.savefig(f"{out}.png", bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved → {out}.pdf / .png")


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    requested = sys.argv[1:] if len(sys.argv) > 1 else list(DATASETS.keys())
    for ds in requested:
        if ds not in DATASETS:
            print(f"Unknown dataset '{ds}'. Choose from: {list(DATASETS.keys())}")
            sys.exit(1)
        print(f"Processing {ds}…")
        plot_dataset(ds, DATASETS[ds])
