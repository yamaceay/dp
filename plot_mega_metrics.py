from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from plot_mega_shared import (
    GROUP_ORDER,
    MDS_DIR,
    VariantStats,
    compute_variant_stats,
    load_meta,
    read_dataset_labels,
)

OUTPUT_DIR = Path("images/mega_metrics")


def _entry_label(params: dict) -> str:
    parts: list[str] = []
    if "epsilon" in params:
        parts.append(f"ε={int(params['epsilon'])}")
    if "k" in params:
        parts.append(f"k={int(params['k'])}")
    if "rho" in params:
        parts.append(f"ρ={float(params['rho']):.2g}")
    if "lambda" in params:
        parts.append(f"λ={float(params['lambda']):.2g}")
    return ", ".join(parts)


METRICS: list[tuple[str, str]] = [
    ("P",    "Privacy risk (P ↓ better)"),
    ("TRIR", "Re-identification rate (TRIR ↓ better)"),
    ("U",    "Utility (U ↑ better)"),
    ("PU",   "Pseudo-utility (PU ↑ better)"),
    ("D_text","Divergence (D ↓ better)"),
    ("T",    "Runtime (T, s ↓ better)"),
    ("GAIN", "GAIN (↑ better)"),
    ("relative_gain", "Relative gain (↑ better)"),
]


def _plot_panel(
    ax: plt.Axes,
    stats: list[VariantStats],
    dataset_label: str,
    x_min: float,
    x_max: float,
) -> None:
    groups_present = [g for g in GROUP_ORDER if any(s.variant.group == g for s in stats)]
    n_total = len(stats) + len(groups_present)

    y = n_total - 1
    y_ticks: list[float] = []
    y_labels: list[str] = []
    y_bold: list[bool] = []
    header_y: list[int] = []
    prev_group: Optional[str] = None

    for s in stats:
        if s.variant.group != prev_group:
            if prev_group is not None:
                ax.axhline(y + 0.5, color="#bbbbbb", linewidth=1.0, zorder=1)
            header_y.append(y)
            y_ticks.append(y)
            y_labels.append(s.variant.group)
            y_bold.append(True)
            y -= 1
            prev_group = s.variant.group

        color = s.variant.color
        if s.not_applicable:
            ax.text(
                (x_min + x_max) / 2,
                y,
                "n/a",
                va="center", ha="center", fontsize=7, color="#aaaaaa", style="italic",
            )
        elif s.n == 1:
            ax.scatter(s.mean, y, s=80, color=color, zorder=5, linewidths=0.5, edgecolors="white")
            ax.text(
                x_max + (x_max - x_min) * 0.02,
                y,
                f"{s.mean:.3f}",
                va="center", ha="left", fontsize=8, color=color, fontweight="bold",
            )
        else:
            ax.hlines(y, s.lo, s.hi, colors=color, linewidth=4.0, alpha=0.45, zorder=3)
            ax.scatter(s.lo, y, s=60, color=color, marker="|", zorder=4, linewidths=2.0)
            ax.scatter(s.hi, y, s=60, color=color, marker="|", zorder=4, linewidths=2.0)
            ax.scatter(s.mean, y, s=80, color=color, zorder=5, linewidths=0.8, edgecolors="white")
            x_span = x_max - x_min
            lo_lbl = _entry_label(s.entries[-1][0])
            hi_lbl = _entry_label(s.entries[0][0])
            if lo_lbl:
                ax.text(s.lo, y - 0.30, lo_lbl,
                        ha="center", va="top", fontsize=5.5, color="#888888", zorder=6)
            if hi_lbl and (s.hi - s.lo) > x_span * 0.06:
                ax.text(s.hi, y - 0.30, hi_lbl,
                        ha="center", va="top", fontsize=5.5, color="#888888", zorder=6)
            ax.text(
                x_max + (x_max - x_min) * 0.02,
                y,
                f"{s.mean:.3f} n={s.n}",
                va="center", ha="left", fontsize=8, color=color, fontweight="bold",
            )

        y_ticks.append(y)
        marker = " *" if (s.not_applicable or s.footnote) else ""
        y_labels.append(f"  {s.variant.label}{marker}")
        y_bold.append(False)
        y -= 1

    for hy in header_y:
        ax.axhspan(hy - 0.45, hy + 0.45, color="#eeeeee", zorder=0, linewidth=0)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=9.0)
    ax.set_ylim(-0.8, n_total - 0.2)
    ax.set_xlim(x_min - (x_max - x_min) * 0.05, x_max + (x_max - x_min) * 0.25)
    ax.set_title(dataset_label, fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.35, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    for tick, bold in zip(ax.get_yticklabels(), y_bold):
        if bold:
            tick.set_fontweight("bold")
            tick.set_color("#333333")
            tick.set_fontsize(9.0)


def _legend_handles(footnote_note: str | None = None) -> list[Line2D]:
    handles = [
        Line2D([0], [0], color="#888888", linewidth=5.0, alpha=0.5, label="Range (min–max)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#333333", markersize=9, label="Mean"),
    ]
    if footnote_note:
        handles.append(Line2D([], [], color="none", label=footnote_note))
    return handles


def _plot_metric(
    metric_col: str,
    metric_label: str,
    panel_data: dict[str, list[VariantStats]],
    dataset_labels: dict[str, str],
) -> Path:
    all_vals = [v for stats in panel_data.values() for s in stats for v in s.values]
    x_min, x_max = float(np.min(all_vals)), float(np.max(all_vals))

    has_footnote = any(s.footnote for stats in panel_data.values() for s in stats)
    has_na = any(s.not_applicable for stats in panel_data.values() for s in stats)
    footnote_parts = []
    if has_footnote:
        footnote_parts.append("* D_pp excluded")
    if has_na:
        footnote_parts.append("* not applicable")
    footnote_note = "  |  ".join(footnote_parts) if footnote_parts else None

    n_groups = len(GROUP_ORDER)
    n_rows = max(len(s) + n_groups for s in panel_data.values())
    n_datasets = len(panel_data)

    fig, axes = plt.subplots(
        1, n_datasets,
        figsize=(8.5 * n_datasets, 0.52 * n_rows + 1.2),
        squeeze=False,
        sharey=False,
    )
    fig.subplots_adjust(wspace=0.5)

    for ax, (ds, stats) in zip(axes[0], panel_data.items()):
        _plot_panel(ax, stats, dataset_labels.get(ds, ds), x_min, x_max)
        ax.set_xlabel(metric_label, fontsize=10)
        if metric_col in ("GAIN", "relative_gain") and x_min < 0 < x_max:
            ax.axvline(0.0, color="black", linewidth=1.2, linestyle="--", alpha=0.5, zorder=2)

    fig.legend(
        handles=_legend_handles(footnote_note),
        loc="lower center",
        ncol=2 + (1 if footnote_note else 0),
        fontsize=8,
        frameon=True,
        bbox_to_anchor=(0.5, -0.04),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"{metric_col.lower()}.pdf"
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="*", dest="datasets", metavar="DATASET")
    parser.add_argument("--metric", nargs="*", dest="metrics", metavar="METRIC",
                        choices=[m for m, _ in METRICS] + [m.lower() for m, _ in METRICS])
    args = parser.parse_args()

    dataset_labels = read_dataset_labels()
    all_datasets = sorted(p.name for p in MDS_DIR.iterdir() if p.is_dir())
    selected_datasets = args.datasets if args.datasets else all_datasets

    selected_metrics = METRICS
    if args.metrics:
        requested = {m.lower() for m in args.metrics}
        selected_metrics = [(m, lbl) for m, lbl in METRICS if m.lower() in requested]

    raw: dict[str, object] = {}
    for ds in selected_datasets:
        df = load_meta(ds)
        if df is not None:
            raw[ds] = df

    if not raw:
        print("No meta_logs.csv found. Run transform_logs.py first.")
        return

    all_panel_data: dict[str, dict[str, list[VariantStats]]] = {}
    for metric_col, metric_label in selected_metrics:
        panel_data: dict[str, list[VariantStats]] = {}
        for ds, df in raw.items():
            if metric_col not in df.columns:
                continue
            stats = compute_variant_stats(df, metric_col)
            if stats:
                panel_data[ds] = stats
        if not panel_data:
            print(f"No data for metric {metric_col}, skipping.")
            continue
        all_panel_data[metric_col] = panel_data
        out = _plot_metric(metric_col, metric_label, panel_data, dataset_labels)
        print(f"Saved {out}")

    if all_panel_data:
        _write_manifest(all_panel_data, dataset_labels)


def _write_manifest(
    all_panel_data: dict[str, dict[str, list[VariantStats]]],
    dataset_labels: dict[str, str],
) -> None:
    import json

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for metric_col, panel_data in all_panel_data.items():
        manifest: dict[str, list[dict]] = {}
        for ds, stats in panel_data.items():
            ds_label = dataset_labels.get(ds, ds)
            manifest[ds_label] = [
                {
                    "group":           s.variant.group,
                    "label":           s.variant.label,
                    "not_applicable":  s.not_applicable,
                    "n":               s.n,
                    "mean":            None if s.not_applicable else round(s.mean, 6),
                    "min":             None if s.not_applicable else round(s.lo,   6),
                    "max":             None if s.not_applicable else round(s.hi,   6),
                    "combinations":    [] if s.not_applicable else [
                        {**params, metric_col: round(value, 6)}
                        for params, value in s.entries
                    ],
                }
                for s in stats
            ]
        out = OUTPUT_DIR / f"{metric_col.lower()}_manifest.json"
        with open(out, "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
