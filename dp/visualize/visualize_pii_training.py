#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

CACHE_ROOT = Path(".cache")
MPL_ROOT = Path(".matplotlib")
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
MPL_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT.resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(MPL_ROOT.resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_STYLE = "seaborn-v0_8-paper"
RECALL_COLUMNS = ("strict_recall", "exact_recall", "partial_recall")


@dataclass(frozen=True)
class Trial:
    selection_metric: str
    batch_size: int
    learning_rate: float
    strict_recall: float
    exact_recall: float
    partial_recall: float

    def mean_recall(self) -> float:
        return (self.strict_recall + self.exact_recall + self.partial_recall) / 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize pii_training.csv sweeps.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("pii_training.csv"),
        help="Path to pii_training.csv.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("imgs/pii_training"),
        help="Directory to write figures.",
    )
    parser.add_argument(
        "--format",
        choices=("png", "svg", "pdf"),
        default="png",
        help="Output format.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="How many top configs to show in ranking plot.",
    )
    return parser.parse_args()


def read_trials(path: Path) -> List[Trial]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"selection_metric", "batch_size", "learning_rate", *RECALL_COLUMNS}
        if reader.fieldnames is None:
            raise ValueError(f"Missing header in {path}")
        missing = sorted(required.difference(set(reader.fieldnames)))
        if missing:
            raise ValueError(f"Missing columns in {path}: {missing}")

        trials: List[Trial] = []
        for row in reader:
            trials.append(
                Trial(
                    selection_metric=str(row["selection_metric"]).strip(),
                    batch_size=int(float(row["batch_size"])),
                    learning_rate=float(row["learning_rate"]),
                    strict_recall=float(row["strict_recall"]),
                    exact_recall=float(row["exact_recall"]),
                    partial_recall=float(row["partial_recall"]),
                )
            )
    if not trials:
        raise ValueError(f"No rows found in {path}")
    return trials


def stable_order(values: Iterable[str]) -> List[str]:
    preferred = ["strict_recall", "exact_recall", "partial_recall"]
    rest = sorted(set(values).difference(preferred))
    ordered = [v for v in preferred if v in set(values)] + rest
    return ordered


def format_lr(value: float) -> str:
    if value == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10**exp)
    if abs(mantissa - 1.0) < 1e-12:
        return f"1e{exp}"
    return f"{mantissa:.1f}e{exp}"


def trials_index(trials: Sequence[Trial]) -> Dict[Tuple[int, float, str], Trial]:
    index: Dict[Tuple[int, float, str], Trial] = {}
    for trial in trials:
        key = (trial.batch_size, trial.learning_rate, trial.selection_metric)
        if key in index:
            raise ValueError(f"Duplicate row for batch={trial.batch_size}, lr={trial.learning_rate}, sel={trial.selection_metric}")
        index[key] = trial
    return index


def plot_heatmaps(
    trials: Sequence[Trial],
    outpath: Path,
    style: str,
) -> None:
    batch_sizes = sorted({t.batch_size for t in trials})
    learning_rates = sorted({t.learning_rate for t in trials})
    selection_metrics = stable_order({t.selection_metric for t in trials})
    idx = trials_index(trials)

    fig, axes = plt.subplots(
        nrows=len(batch_sizes),
        ncols=len(RECALL_COLUMNS),
        figsize=(4.6 * len(RECALL_COLUMNS), 2.6 * len(batch_sizes)),
        constrained_layout=True,
    )

    if len(batch_sizes) == 1 and len(RECALL_COLUMNS) == 1:
        axes = np.array([[axes]])
    elif len(batch_sizes) == 1:
        axes = np.array([axes])
    elif len(RECALL_COLUMNS) == 1:
        axes = np.array([[ax] for ax in axes])

    global_min = min(getattr(t, col) for t in trials for col in RECALL_COLUMNS)
    global_max = max(getattr(t, col) for t in trials for col in RECALL_COLUMNS)

    for row_i, batch_size in enumerate(batch_sizes):
        for col_i, recall_col in enumerate(RECALL_COLUMNS):
            ax = axes[row_i, col_i]
            values = np.full((len(selection_metrics), len(learning_rates)), np.nan, dtype=float)
            for sel_i, sel in enumerate(selection_metrics):
                for lr_i, lr in enumerate(learning_rates):
                    trial = idx.get((batch_size, lr, sel))
                    if trial is None:
                        continue
                    values[sel_i, lr_i] = float(getattr(trial, recall_col))

            im = ax.imshow(
                values,
                vmin=global_min,
                vmax=global_max,
                cmap="viridis",
                aspect="auto",
            )

            ax.set_title(f"batch={batch_size} · {recall_col}", loc="left", fontsize=11, weight="bold")
            ax.set_xticks(list(range(len(learning_rates))), [format_lr(v) for v in learning_rates])
            ax.set_yticks(list(range(len(selection_metrics))), selection_metrics)
            ax.tick_params(axis="x", labelrotation=0)

            best_val = np.nanmax(values)
            if np.isfinite(best_val):
                best_pos = np.argwhere(values == best_val)
            else:
                best_pos = np.zeros((0, 2), dtype=int)

            for sel_i in range(values.shape[0]):
                for lr_i in range(values.shape[1]):
                    v = values[sel_i, lr_i]
                    if not np.isfinite(v):
                        continue
                    ax.text(
                        lr_i,
                        sel_i,
                        f"{v:.3f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="white" if v >= (global_min + global_max) / 2 else "black",
                    )

            for sel_i, lr_i in best_pos:
                ax.add_patch(
                    plt.Rectangle(
                        (lr_i - 0.5, sel_i - 0.5),
                        1.0,
                        1.0,
                        fill=False,
                        linewidth=2.0,
                        edgecolor="#ffffff",
                    )
                )

    fig.suptitle("PII training sweep: recall heatmaps", fontsize=14, weight="bold")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.92, pad=0.02)
    cbar.set_label("recall", rotation=90)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_top_configs(
    trials: Sequence[Trial],
    outpath: Path,
    top_n: int,
) -> None:
    ranked = sorted(trials, key=lambda t: t.mean_recall(), reverse=True)
    shown = ranked[: max(1, min(top_n, len(ranked)))]

    labels = [f"{t.selection_metric} | bs={t.batch_size} | lr={format_lr(t.learning_rate)}" for t in shown]
    means = [t.mean_recall() for t in shown]

    fig, ax = plt.subplots(figsize=(10.8, 0.55 * len(shown) + 1.6), constrained_layout=True)
    y = np.arange(len(shown))
    ax.barh(y, means, color="#386cb0")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("mean recall")
    ax.set_title("Top configs by mean recall", loc="left", fontsize=13, weight="bold")
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.6)

    lo = min(means)
    hi = max(means)
    pad = 0.02 if hi > lo else 0.05
    ax.set_xlim(max(0.0, lo - pad), min(1.0, hi + pad))

    for yi, v in zip(y, means):
        ax.text(v + 0.002, yi, f"{v:.4f}", va="center", ha="left", fontsize=9)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_tradeoffs(
    trials: Sequence[Trial],
    outpath: Path,
) -> None:
    selection_metrics = stable_order({t.selection_metric for t in trials})
    batch_sizes = sorted({t.batch_size for t in trials})
    learning_rates = sorted({t.learning_rate for t in trials})

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(13.2, 3.8), constrained_layout=True)

    colors = {
        "strict_recall": "#1b9e77",
        "exact_recall": "#d95f02",
        "partial_recall": "#7570b3",
    }
    marker_cycle = ["o", "s", "D", "^", "v", "P"]
    markers = {
        batch_size: marker_cycle[i % len(marker_cycle)]
        for i, batch_size in enumerate(batch_sizes)
    }

    for ax, recall_col in zip(axes, RECALL_COLUMNS):
        for sel in selection_metrics:
            for batch_size in batch_sizes:
                xs = []
                ys = []
                for lr in learning_rates:
                    match = [
                        t
                        for t in trials
                        if t.selection_metric == sel and t.batch_size == batch_size and t.learning_rate == lr
                    ]
                    if not match:
                        continue
                    xs.append(lr)
                    ys.append(float(getattr(match[0], recall_col)))

                if not xs:
                    continue

                ax.plot(
                    xs,
                    ys,
                    marker=markers.get(batch_size, "o"),
                    linewidth=1.2,
                    alpha=0.9,
                    color=colors.get(sel, "#616161"),
                    label=f"{sel} (bs={batch_size})",
                )

        ax.set_xscale("log")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
        ax.set_title(recall_col, loc="left", fontsize=12, weight="bold")
        ax.set_xlabel("learning rate")
        ax.set_ylabel("recall")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(6, len(labels)), frameon=False)
    fig.suptitle("Recall vs learning rate (faceted)", fontsize=14, weight="bold")

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    trials = read_trials(args.input)

    try:
        plt.style.use(DEFAULT_STYLE)
    except OSError:
        plt.style.use("default")

    args.outdir.mkdir(parents=True, exist_ok=True)
    plot_heatmaps(trials, args.outdir / f"pii_training_heatmaps.{args.format}", DEFAULT_STYLE)
    plot_top_configs(trials, args.outdir / f"pii_training_top_configs.{args.format}", args.top_n)
    plot_tradeoffs(trials, args.outdir / f"pii_training_tradeoffs.{args.format}")


if __name__ == "__main__":
    main()
