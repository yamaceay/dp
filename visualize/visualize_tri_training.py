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


@dataclass(frozen=True)
class Trial:
    dataset: str
    selection_metric: str
    batch_size: int
    learning_rate: float
    finetuning_epochs: int
    use_pretraining: bool
    accuracy: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize TRI training sweep results for TRI (tab) and reddit datasets."
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="*",
        default=[Path("tri_training_tab.csv"), Path("tri_training_reddit.csv")],
        help="Input CSV paths. Defaults to tri_training_tab.csv and tri_training_reddit.csv.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("imgs/tri_training"),
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


def stable_order(values: Iterable[str]) -> List[str]:
    ordered = list(dict.fromkeys(values))
    if "accuracy" in ordered:
        ordered.remove("accuracy")
        ordered.insert(0, "accuracy")
    return ordered


def format_lr(value: float) -> str:
    if value == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10**exp)
    if abs(mantissa - 1.0) < 1e-12:
        return f"1e{exp}"
    return f"{mantissa:.1f}e{exp}"


def _dataset_label_from_path(path: Path) -> str:
    stem = path.stem
    # tri_training_tab -> tri ; tri_training_reddit -> reddit
    if stem.startswith("tri_training_"):
        suffix = stem[len("tri_training_") :]
        if suffix == "tab":
            return "tri"
        return suffix
    if stem == "tab":
        return "tri"
    return stem


def _read_csv_rows(path: Path) -> Tuple[List[str], List[List[str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty CSV: {path}")
        rows = [row for row in reader if any(cell.strip() for cell in row)]
    return [h.strip() for h in header], rows


def read_trials(path: Path, dataset: str) -> List[Trial]:
    header, rows = _read_csv_rows(path)
    if not rows:
        raise ValueError(f"No rows found in {path}")

    first_len = len(rows[0])
    if first_len != len(header):
        # Handle a known malformed case (tri_training_reddit.csv) where the header
        # includes selection_metric but the data rows do not.
        if "selection_metric" in header and first_len == len(header) - 1:
            header = [h for h in header if h != "selection_metric"]
        else:
            raise ValueError(
                f"Header/data width mismatch in {path}: header={len(header)} row={first_len}"
            )

    normalized_header = [h.strip() for h in header]

    required = {"batch_size", "learning_rate", "finetuning_epochs", "use_pretraining", "accuracy"}
    missing = sorted(required.difference(set(normalized_header)))
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    trials: List[Trial] = []
    for row in rows:
        if len(row) != len(normalized_header):
            raise ValueError(
                f"Inconsistent row width in {path}: expected={len(normalized_header)} got={len(row)}"
            )
        record = {k: v for k, v in zip(normalized_header, row)}

        selection_metric = str(record.get("selection_metric", "accuracy")).strip() or "accuracy"
        trials.append(
            Trial(
                dataset=dataset,
                selection_metric=selection_metric,
                batch_size=int(float(record["batch_size"])),
                learning_rate=float(record["learning_rate"]),
                finetuning_epochs=int(float(record["finetuning_epochs"])),
                use_pretraining=str(record["use_pretraining"]).strip().lower() in {"true", "1", "yes"},
                accuracy=float(record["accuracy"]),
            )
        )

    return trials


def trials_index(trials: Sequence[Trial]) -> Dict[Tuple[int, float, int, bool, str], Trial]:
    index: Dict[Tuple[int, float, int, bool, str], Trial] = {}
    for trial in trials:
        key = (
            trial.batch_size,
            trial.learning_rate,
            trial.finetuning_epochs,
            trial.use_pretraining,
            trial.selection_metric,
        )
        if key in index:
            raise ValueError(
                "Duplicate row for "
                f"bs={trial.batch_size}, lr={trial.learning_rate}, "
                f"epochs={trial.finetuning_epochs}, pretrain={trial.use_pretraining}, sel={trial.selection_metric}"
            )
        index[key] = trial
    return index


def plot_heatmaps(
    trials: Sequence[Trial],
    outpath: Path,
    dataset_label: str,
) -> None:
    batch_sizes = sorted({t.batch_size for t in trials})
    learning_rates = sorted({t.learning_rate for t in trials})
    epochs = sorted({t.finetuning_epochs for t in trials})
    selection_metrics = stable_order({t.selection_metric for t in trials})
    pretraining_values = [False, True]

    if len(epochs) != 1 or len(selection_metrics) != 1:
        # Keep the visualization simple; if multiple epochs/metrics exist, we still plot
        # the first (stable-ordered) selection metric and the max epoch.
        selection_metric = selection_metrics[0]
        finetuning_epochs = epochs[-1]
    else:
        selection_metric = selection_metrics[0]
        finetuning_epochs = epochs[0]

    idx = trials_index(trials)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(pretraining_values),
        figsize=(5.4 * len(pretraining_values), 3.9),
        constrained_layout=True,
    )
    if len(pretraining_values) == 1:
        axes = [axes]

    # global min/max across plotted slice
    plotted_vals: List[float] = []
    for batch_size in batch_sizes:
        for lr in learning_rates:
            for pretrain in pretraining_values:
                trial = idx.get((batch_size, lr, finetuning_epochs, pretrain, selection_metric))
                if trial is not None:
                    plotted_vals.append(trial.accuracy)

    if not plotted_vals:
        raise ValueError("No values to plot (check filtering)")

    global_min = float(min(plotted_vals))
    global_max = float(max(plotted_vals))

    im = None
    for ax, pretrain in zip(axes, pretraining_values):
        values = np.full((len(batch_sizes), len(learning_rates)), np.nan, dtype=float)
        for bi, batch_size in enumerate(batch_sizes):
            for li, lr in enumerate(learning_rates):
                trial = idx.get((batch_size, lr, finetuning_epochs, pretrain, selection_metric))
                if trial is None:
                    continue
                values[bi, li] = float(trial.accuracy)

        im = ax.imshow(
            values,
            vmin=global_min,
            vmax=global_max,
            cmap="viridis",
            aspect="auto",
        )

        ax.set_title(
            f"pretraining={pretrain} · epochs={finetuning_epochs}",
            loc="left",
            fontsize=11,
            weight="bold",
        )
        ax.set_xticks(list(range(len(learning_rates))), [format_lr(v) for v in learning_rates])
        ax.set_yticks(list(range(len(batch_sizes))), [str(v) for v in batch_sizes])
        ax.set_xlabel("learning rate")
        ax.set_ylabel("batch size")

        for bi in range(values.shape[0]):
            for li in range(values.shape[1]):
                v = values[bi, li]
                if not np.isfinite(v):
                    continue
                ax.text(
                    li,
                    bi,
                    f"{v:.3f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if v >= (global_min + global_max) / 2 else "black",
                )

    fig.suptitle(
        f"TRI training sweep ({dataset_label}): accuracy heatmaps",
        fontsize=14,
        weight="bold",
    )
    if im is not None:
        cbar = fig.colorbar(im, ax=list(axes), shrink=0.92, pad=0.02)
        cbar.set_label("accuracy", rotation=90)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_top_configs(
    trials: Sequence[Trial],
    outpath: Path,
    dataset_label: str,
    top_n: int,
) -> None:
    ranked = sorted(trials, key=lambda t: t.accuracy, reverse=True)
    shown = ranked[: max(1, min(top_n, len(ranked)))]

    labels = [
        f"pre={t.use_pretraining} | bs={t.batch_size} | lr={format_lr(t.learning_rate)} | ep={t.finetuning_epochs}"
        for t in shown
    ]
    values = [t.accuracy for t in shown]

    fig, ax = plt.subplots(figsize=(11.2, 0.55 * len(shown) + 1.6), constrained_layout=True)
    y = np.arange(len(shown))
    ax.barh(y, values, color="#386cb0")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("accuracy")
    ax.set_title(
        f"Top configs by accuracy ({dataset_label})",
        loc="left",
        fontsize=13,
        weight="bold",
    )
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.6)

    lo = min(values)
    hi = max(values)
    pad = 0.02 if hi > lo else 0.05
    ax.set_xlim(max(0.0, lo - pad), min(1.0, hi + pad))

    for yi, v in zip(y, values):
        ax.text(v + 0.002, yi, f"{v:.4f}", va="center", ha="left", fontsize=9)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_tradeoffs(
    trials: Sequence[Trial],
    outpath: Path,
    dataset_label: str,
) -> None:
    batch_sizes = sorted({t.batch_size for t in trials})
    learning_rates = sorted({t.learning_rate for t in trials})
    epochs = sorted({t.finetuning_epochs for t in trials})
    selection_metrics = stable_order({t.selection_metric for t in trials})

    selection_metric = selection_metrics[0] if selection_metrics else "accuracy"
    finetuning_epochs = epochs[-1] if epochs else 0

    idx = trials_index(trials)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12.4, 3.9), constrained_layout=True)
    pretraining_values = [False, True]

    marker_cycle = ["o", "s", "D", "^", "v", "P"]
    markers = {bs: marker_cycle[i % len(marker_cycle)] for i, bs in enumerate(batch_sizes)}

    for ax, pretrain in zip(axes, pretraining_values):
        for batch_size in batch_sizes:
            xs = []
            ys = []
            for lr in learning_rates:
                trial = idx.get((batch_size, lr, finetuning_epochs, pretrain, selection_metric))
                if trial is None:
                    continue
                xs.append(lr)
                ys.append(trial.accuracy)

            if not xs:
                continue

            ax.plot(
                xs,
                ys,
                marker=markers.get(batch_size, "o"),
                linewidth=1.2,
                alpha=0.9,
                label=f"bs={batch_size}",
            )

        ax.set_xscale("log")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
        ax.set_title(
            f"pretraining={pretrain} · epochs={finetuning_epochs}",
            loc="left",
            fontsize=12,
            weight="bold",
        )
        ax.set_xlabel("learning rate")
        ax.set_ylabel("accuracy")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(6, len(labels)), frameon=False)
    fig.suptitle(
        f"Accuracy vs learning rate ({dataset_label})",
        fontsize=14,
        weight="bold",
    )

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    try:
        plt.style.use(DEFAULT_STYLE)
    except OSError:
        plt.style.use("default")

    args.outdir.mkdir(parents=True, exist_ok=True)

    for input_path in args.inputs:
        dataset_label = _dataset_label_from_path(input_path)
        trials = read_trials(input_path, dataset_label)

        prefix = f"tri_training_{dataset_label}"
        plot_heatmaps(trials, args.outdir / f"{prefix}_heatmaps.{args.format}", dataset_label)
        plot_top_configs(
            trials,
            args.outdir / f"{prefix}_top_configs.{args.format}",
            dataset_label,
            args.top_n,
        )
        plot_tradeoffs(trials, args.outdir / f"{prefix}_tradeoffs.{args.format}", dataset_label)


if __name__ == "__main__":
    main()
