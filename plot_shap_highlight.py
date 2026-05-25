"""
In-context SHAP text highlighting for thesis.

Renders Presidio-anonymized text with word-level background colors
proportional to SHAP attribution (sqrt-normalized per record):
  red   = risk-increasing (positive SHAP)
  white = neutral / near-zero
  blue  = risk-reducing   (negative SHAP)

DB-Bio: full text shown (no windowing — records are short enough).
TAB:    900-char window centred on the cluster of highest-scoring tokens.

Output: docs/thesis/images/explanations/{dataset}_shap_highlight.pdf / .png

Usage:
    python plot_shap_highlight.py          # both datasets
    python plot_shap_highlight.py tab
    python plot_shap_highlight.py db_bio
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
import numpy as np

sys.path.insert(0, ".")
from dp.loaders import get_adapter

# ── Layout ─────────────────────────────────────────────────────────────────────

CHARS_PER_LINE = 72
FONTSIZE       = 8.5      # pt, monospace
FIG_WIDTH      = 8.5      # inches

# Per-dataset settings
DATASET_CFG = {
    "db_bio": {"window": None, "max_lines": 9},   # None = full text
    "tab":    {"window": 520,  "max_lines": 7},
}

# ── Dataset configs ────────────────────────────────────────────────────────────

CONFIGS = {
    "tab": {
        "data_in":  "data/tab",
        "presidio": "presidio/tab.jsonl",
        "shap":     "data/tab/tri_risk/shap.jsonl",
        "output":   "docs/thesis/images/explanations/tab_shap_highlight",
        "records": [
            ("001-61807",   "001-61807 · Turkey – Sinop Assize Court"),
            ("001-66929",   "001-66929 · Turkey – Aydın Magistrate's Court"),
            ("001-90749",   "001-90749 · Cyprus – property plots"),
            ("001-77315",   "001-77315 · Poland – Sieradz Regional Court"),
        ],
    },
    "db_bio": {
        "data_in":  "data/db_bio",
        "presidio": "presidio/db_bio.jsonl",
        "shap":     "data/db_bio/tri_risk/shap.jsonl",
        "output":   "docs/thesis/images/explanations/db_bio_shap_highlight",
        "records": [
            ("Antonio_Barbalonga",         "Antonio Barbalonga · Baroque painter"),
            ("William_Oliver_(physician)", "William Oliver · physician / Madron"),
            ("Yosef_Tekoah",               "Yosef Tekoah · Ben-Gurion diplomat"),
            ("Elseid_Hysaj",               "Elseid Hysaj · footballer / Coppa Italia"),
        ],
    },
}

# ── Data loading ───────────────────────────────────────────────────────────────

def load_presidio(path):
    table = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            table[r["idx"]] = r["text"]
    return table


def load_shap(path):
    table = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            table[r["uid"]] = (r["offsets"], r["scores"])
    return table


def build_uid_to_idx(ds_name, data_in):
    adapter = get_adapter(ds_name, data=ds_name, data_in=data_in)
    return {str(rec.uid): i for i, rec in enumerate(adapter.iter_records())}

# ── Window selection ───────────────────────────────────────────────────────────

def select_window(text, offsets, scores, window):
    """Return (win_start, win_end). window=None → full text."""
    if window is None or not scores:
        return 0, len(text)
    ranked = sorted(zip(offsets, scores), key=lambda x: -abs(x[1]))
    top_starts = [s for (s, _e), _ in ranked[:5]]
    center = int(np.median(top_starts))
    win_start = max(0, center - window // 3)
    win_end   = min(len(text), win_start + window)
    if win_end - win_start < window:
        win_start = max(0, win_end - window)
    return win_start, win_end

# ── Tokenisation ───────────────────────────────────────────────────────────────

def tokenize_segment(text, win_start, win_end, offsets, scores):
    """
    Return list of (token_str, shap_score_or_None) for the text segment.
    All whitespace-delimited words are returned; score=None means unscored.
    """
    segment = text[win_start:win_end]

    # char-level score map (relative to segment start)
    score_map = {}
    for (s, e), sc in zip(offsets, scores):
        for c in range(max(s, win_start), min(e, win_end)):
            score_map[c - win_start] = sc

    tokens = []
    i = 0
    while i < len(segment):
        if segment[i] in (" ", "\n", "\t"):
            i += 1
            continue
        j = i
        while j < len(segment) and segment[j] not in (" ", "\n", "\t"):
            j += 1
        tok = segment[i:j]
        span_sc = [score_map[c] for c in range(i, j) if c in score_map]
        sc = max(span_sc, key=abs) if span_sc else None
        tokens.append((tok, sc))
        i = j
    return tokens

# ── Color mapping (sqrt-normalised) ───────────────────────────────────────────

def score_to_color(score, max_abs):
    """
    Return (r, g, b, alpha).  Uses sqrt compression so secondary-peak tokens
    show visible tint rather than collapsing to white under a dominant outlier.
    """
    if max_abs == 0 or score is None:
        return (1.0, 1.0, 1.0, 0.0)
    t = float(np.sign(score)) * float(np.sqrt(abs(score) / max_abs))
    t = max(-1.0, min(1.0, t))
    alpha = abs(t) * 0.80 + 0.10
    if t >= 0:
        r, g, b = 1.0, 1.0 - 0.75 * t, 1.0 - 0.75 * t
    else:
        r, g, b = 1.0 + 0.75 * t, 1.0 + 0.75 * t, 1.0
    return (r, g, b, alpha)

# ── Record renderer ────────────────────────────────────────────────────────────

def draw_record(ax, tokens, max_abs, title, max_lines):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title bar
    ax.add_patch(FancyBboxPatch(
        (0.0, 0.90), 1.0, 0.10,
        boxstyle="square,pad=0.0", facecolor="#e4e4e4", edgecolor="none",
        transform=ax.transAxes, zorder=0,
    ))
    ax.text(0.010, 0.950, title,
            transform=ax.transAxes, fontsize=FONTSIZE + 0.5,
            fontfamily="monospace", fontweight="bold",
            va="center", ha="left", color="#111111", zorder=1)

    text_top   = 0.875
    text_left  = 0.008
    text_right = 0.992
    line_h     = text_top / max_lines
    char_w     = (text_right - text_left) / CHARS_PER_LINE
    gap        = char_w * 0.45   # inter-token gap

    x, line = text_left, 0

    for tok, sc in tokens:
        if line >= max_lines:
            break
        tok_w = len(tok) * char_w
        if x + tok_w > text_right and x > text_left:
            line += 1
            x = text_left
            if line >= max_lines:
                break

        y_top = text_top - line * line_h
        y_bot = y_top - line_h * 0.88
        y_mid = (y_top + y_bot) / 2

        if sc is not None:
            r, g, b, alpha = score_to_color(sc, max_abs)
            pad = char_w * 0.12
            ax.add_patch(FancyBboxPatch(
                (x - pad, y_bot), tok_w + 2 * pad, y_top - y_bot,
                boxstyle="square,pad=0.0",
                facecolor=(r, g, b), alpha=alpha,
                edgecolor="none", zorder=1,
                transform=ax.transAxes, clip_on=True,
            ))

        ax.text(x, y_mid, tok,
                transform=ax.transAxes,
                fontsize=FONTSIZE, fontfamily="monospace",
                va="center", ha="left", color="#111111", zorder=2)

        x += tok_w + gap

# ── Main ───────────────────────────────────────────────────────────────────────

def plot_dataset(ds_name, cfg):
    dcfg       = DATASET_CFG[ds_name]
    window     = dcfg["window"]
    max_lines  = dcfg["max_lines"]

    pres_table = load_presidio(cfg["presidio"])
    shap_table = load_shap(cfg["shap"])
    uid_to_idx = build_uid_to_idx(ds_name, cfg["data_in"])
    records    = cfg["records"]

    n = len(records)
    # Row height scales with max_lines so text doesn't overlap
    row_h = max_lines * (FONTSIZE / 72) * 1.55 + 0.40
    fig, axes = plt.subplots(n, 1, figsize=(FIG_WIDTH, n * row_h))
    if n == 1:
        axes = [axes]
    fig.subplots_adjust(hspace=0.06, left=0.005, right=0.995, top=0.99, bottom=0.05)

    for ax, (uid, label) in zip(axes, records):
        idx = uid_to_idx.get(uid)
        if idx is None or idx not in pres_table:
            ax.axis("off")
            continue

        text = pres_table[idx]
        offsets, scores = shap_table.get(uid, ([], []))

        win_s, win_e = select_window(text, offsets, scores, window)
        tokens = tokenize_segment(text, win_s, win_e, offsets, scores)

        all_sc = [abs(sc) for _, sc in tokens if sc is not None]
        max_abs = max(all_sc) if all_sc else 1.0

        draw_record(ax, tokens, max_abs, label, max_lines)

    # Colorbar
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    sm   = ScalarMappable(cmap="RdBu_r", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(
        sm, ax=axes, orientation="horizontal",
        fraction=0.018, pad=0.005, aspect=60, shrink=0.60,
    )
    cbar.set_label(
        "SHAP attribution (sqrt-compressed, normalized per record;  "
        "red = risk-increasing,  blue = risk-reducing)",
        fontsize=7.5,
    )
    cbar.ax.tick_params(labelsize=7)

    out = cfg["output"]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(f"{out}.pdf", bbox_inches="tight", dpi=200)
    plt.savefig(f"{out}.png", bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved → {out}.pdf / .png")


if __name__ == "__main__":
    requested = sys.argv[1:] if len(sys.argv) > 1 else list(CONFIGS.keys())
    for ds in requested:
        if ds not in CONFIGS:
            print(f"Unknown dataset '{ds}'. Choose from: {list(CONFIGS.keys())}")
            sys.exit(1)
        print(f"Processing {ds}…")
        plot_dataset(ds, CONFIGS[ds])
