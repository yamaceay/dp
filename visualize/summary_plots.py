from __future__ import annotations

import argparse
import json
from math import ceil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_flat(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def filter_dataset(rows: Sequence[Dict[str, Any]], dataset: str) -> List[Dict[str, Any]]:
    return [r for r in rows if r.get("dataset") == dataset]


def unique_sorted(values: Iterable[Any]) -> List[Any]:
    def sort_key(item: Any) -> Tuple[int, Any]:
        text = str(item)
        numeric = text.replace(".", "", 1).isdigit()
        if numeric:
            return (0, float(text))
        return (1, text)
    return sorted(set(values), key=sort_key)


def method_palette(methods: Sequence[str]) -> Dict[str, Any]:
    colormap = matplotlib.colormaps.get_cmap("tab20")
    size = max(1, len(methods))
    return {method_name: colormap(index / size) for index, method_name in enumerate(methods)}


def plot_grid_by_param(dataset: str, flat_path: Path, output_dir: Path, metric: str, param_name: str) -> None:
    flat_rows = filter_dataset(load_flat(flat_path), dataset)
    method_names = sorted({row["method"] for row in flat_rows})
    palette = method_palette(method_names)
    param_values = unique_sorted(row["params"].get(param_name) for row in flat_rows if param_name in row.get("params", {}))
    if not param_values:
        return
    cols_count = min(4, len(param_values))
    rows_count = ceil(len(param_values) / cols_count)
    figure, axes_grid = plt.subplots(rows_count, cols_count, figsize=(5 * cols_count, 4 * rows_count), squeeze=False, sharey=True)
    for index, param_value in enumerate(param_values):
        axis = axes_grid[index // cols_count][index % cols_count]
        subset = [row for row in flat_rows if row.get("params", {}).get(param_name) == param_value]
        base_x: Dict[str, float] = {method: float(pos) for pos, method in enumerate(method_names)}
        grouped: Dict[str, List[Dict[str, Any]]] = {method: [] for method in method_names}
        for row in subset:
            grouped[row["method"]].append(row)
        x_positions: List[float] = []
        y_values: List[float] = []
        colors: List[Any] = []
        bar_width = 0.7
        for method_name in method_names:
            entries = grouped[method_name]
            if not entries:
                continue
            entry_count = len(entries)
            if entry_count == 1:
                x_positions.append(base_x[method_name])
                y_values.append(float(entries[0][metric]))
                colors.append(palette[method_name])
            else:
                offset_step = bar_width / max(1, entry_count)
                start_x = base_x[method_name] - bar_width / 2 + offset_step / 2
                for entry_index, entry in enumerate(entries):
                    x_positions.append(start_x + entry_index * offset_step)
                    y_values.append(float(entry[metric]))
                    colors.append(palette[method_name])
        axis.bar(x_positions, y_values, color=colors, width=bar_width / 2)
        axis.set_xticks(list(base_x.values()))
        axis.set_xticklabels(method_names, rotation=30, ha="right")
        axis.set_title(f"{param_name}={param_value}")
    last_index = index
    for empty_index in range(last_index + 1, rows_count * cols_count):
        figure.delaxes(axes_grid[empty_index // cols_count][empty_index % cols_count])
    axes_grid[0][0].set_ylabel(metric)
    figure.suptitle(f"{dataset}: {metric} by {param_name}")
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    ensure_dir(output_dir)
    figure.savefig(output_dir / f"{dataset}_grid_by_{param_name}.png", dpi=220)
    plt.close(figure)


def discover_params(flat_path: Path, dataset: str) -> List[str]:
    flat_rows = filter_dataset(load_flat(flat_path), dataset)
    names = {name for row in flat_rows for name in (row.get("params") or {}).keys()}
    preferred = ["epsilon", "k", "rho", "lambda"]
    ordered = [n for n in preferred if n in names]
    remaining = sorted(names - set(ordered))
    return ordered + remaining


def param_value_palettes(flat_path: Path, dataset: str) -> Dict[str, Dict[float, Any]]:
    rows = filter_dataset(load_flat(flat_path), dataset)
    names = {name for row in rows for name in (row.get("params") or {}).keys()}
    palettes: Dict[str, Dict[float, Any]] = {}
    for name in names:
        vals = [float(row["params"][name]) for row in rows if name in row.get("params", {})]
        uniq = unique_sorted(vals)
        colormap = matplotlib.colormaps.get_cmap({"epsilon": "Blues", "k": "Oranges", "rho": "Greens", "lambda": "Reds"}.get(name, "Purples"))
        if not uniq:
            palettes[name] = {}
            continue
        positions = [i / max(1, len(uniq) - 1) for i in range(len(uniq))]
        palettes[name] = {float(v): colormap(0.25 + 0.6 * p) for v, p in zip(uniq, positions)}
    return palettes


def plot_full(dataset: str, flat_path: Path, output_dir: Path, metric: str) -> None:
    rows = filter_dataset(load_flat(flat_path), dataset)
    if not rows:
        return
    methods = sorted({row["method"] for row in rows})
    base_x: Dict[str, float] = {method: float(i) for i, method in enumerate(methods)}
    grouped: Dict[str, List[Dict[str, Any]]] = {method: [] for method in methods}
    for row in rows:
        grouped[row["method"]].append(row)
    palettes = param_value_palettes(flat_path, dataset)
    precedence = ["epsilon", "k", "rho", "lambda"]
    total_entries = sum(max(1, len(grouped[name])) for name in methods)
    figure_width = max(14, int(0.6 * total_entries))
    figure, axis = plt.subplots(figsize=(figure_width, 8))
    x_positions: List[float] = []
    y_values: List[float] = []
    colors: List[Any] = []
    labels: List[str] = []
    widths: List[float] = []
    for method in methods:
        entries = grouped[method]
        bar_group_width = 0.82
        count = max(1, len(entries))
        step = bar_group_width / count
        start_x = base_x[method] - bar_group_width / 2 + step / 2
        for index_in_group, entry in enumerate(entries):
            params = entry.get("params") or {}
            label_text = "\n".join(f"{key}={params[key]}" for key in sorted(params.keys())) or "baseline"
            primary = next((name for name in precedence if name in params), next(iter(params.keys()), None))
            if primary is None:
                color = (0.7, 0.7, 0.7, 1.0)
            else:
                value = float(params[primary])
                color = palettes.get(primary, {}).get(value, (0.5, 0.5, 0.5, 1.0))
            x_positions.append(start_x + index_in_group * step)
            y_values.append(float(entry[metric]))
            colors.append(color)
            labels.append(label_text)
            widths.append(step * 0.85)
    axis.bar(x_positions, y_values, color=colors, width=widths, edgecolor="black", linewidth=0.3)
    axis.set_xticks(list(base_x.values()))
    axis.set_xticklabels(methods, rotation=30, ha="right")
    axis.set_ylabel(metric)
    axis.set_title(f"{dataset}: {metric} full view")
    if y_values:
        ymax = max(y_values)
        axis.set_ylim(0, ymax * 1.18)
    for x_val, y_val, text in zip(x_positions, y_values, labels):
        axis.annotate(text, xy=(x_val, y_val), xytext=(0, 3), textcoords="offset points", rotation=90, ha="center", va="bottom", fontsize=7, clip_on=False)
    handles = []
    names = discover_params(flat_path, dataset)
    for name in names:
        sample_values = list(palettes.get(name, {}).values())
        if not sample_values:
            continue
        handles.append(plt.Line2D([0], [0], color=sample_values[-1], lw=6))
    if handles:
        axis.legend(handles, names, title="hyperparam (shade=value)", fontsize=9, ncol=min(4, len(handles)))
    ensure_dir(output_dir)
    figure.tight_layout()
    figure.savefig(output_dir / f"{dataset}_full.png", dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["grid-by-param", "grid-all-params", "full"]) 
    parser.add_argument("--dataset", default="reddit")
    parser.add_argument("--metric", default="privacy_mean_rank_change")
    parser.add_argument("--flat", default="visualize/reddit_summary.json")
    parser.add_argument("--out", default="visualize/plots")
    parser.add_argument("--param", default="epsilon")
    args = parser.parse_args()
    output_dir = Path(args.out)
    if args.action == "grid-by-param":
        plot_grid_by_param(args.dataset, Path(args.flat), output_dir, args.metric, args.param)
        return
    if args.action == "grid-all-params":
        for name in discover_params(Path(args.flat), args.dataset):
            plot_grid_by_param(args.dataset, Path(args.flat), output_dir, args.metric, name)
        return
    if args.action == "full":
        plot_full(args.dataset, Path(args.flat), output_dir, args.metric)
        return


if __name__ == "__main__":
    main()
