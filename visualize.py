#!/usr/bin/env python3
"""Plot privacy/divergence/utility metrics with anonymization-aware coloring."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

CACHE_ROOT = Path(".cache")
MPL_ROOT = Path(".matplotlib")
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
MPL_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT.resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(MPL_ROOT.resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

PARAM_RULES = {
    "epsilon": {"color": "#1b9e77", "direction": "low"},
    "rho": {"color": "#386cb0", "direction": "low"},
    "lambda": {"color": "#d95f02", "direction": "low"},
    "k": {"color": "#7570b3", "direction": "high"},
}
PREFIXES = ("privacy", "divergence", "utility")
PREFIX_BACKGROUNDS = {
    "privacy": "#edf7ed",
    "divergence": "#e8f1fb",
    "utility": "#fff3e0",
}
DEFAULT_COLOR = "#616161"
DEFAULT_STYLE = "seaborn-v0_8-paper"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize merged metric results.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("logs/merged_results.jsonl"),
        help="Path to the JSONL file containing merged results.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/merged_results.svg"),
        help="Destination for the combined rendered plot.",
    )
    parser.add_argument(
        "--format",
        choices=("svg", "png", "pdf"),
        default="svg",
        help="Figure format for saved plots.",
    )
    parser.add_argument(
        "--per-metric-dir",
        type=Path,
        help="Optional directory to export per-metric plots (files saved as PREFIX_metric.<format>).",
    )
    return parser.parse_args()


def load_records(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No records found in {path}")
    return rows


def parse_name(full_name: str) -> Tuple[str, Dict[str, str]]:
    if "?" not in full_name:
        return full_name, {}
    base, payload = full_name.split("?", 1)
    params: Dict[str, str] = {}
    for chunk in payload.split("&"):
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        params[key] = value
    return base, params


def to_number(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def infer_param_type(params: Dict[str, str]) -> Tuple[Optional[str], Optional[float]]:
    for target in PARAM_RULES.keys():
        for key, value in params.items():
            if target in key:
                numeric = to_number(value)
                return target, numeric
    return None, None


def blend_color(hex_color: str, strength: float) -> str:
    base_rgb = mcolors.to_rgb(hex_color)
    white = (1.0, 1.0, 1.0)
    strength = min(max(strength, 0.0), 1.0)
    mix = 0.35 + 0.65 * strength
    blended = tuple(white[i] + (base_rgb[i] - white[i]) * mix for i in range(3))
    return mcolors.to_hex(blended)


def assign_colors(rows: List[Dict[str, float]]) -> List[Dict[str, object]]:
    enriched: List[Dict[str, object]] = []
    param_values: Dict[str, List[float]] = defaultdict(list)

    for row in rows:
        base_name, params = parse_name(row.get("name", "unknown"))
        param_type, value = infer_param_type(params)
        if param_type and value is not None:
            param_values[param_type].append(value)
        enriched.append(
            {
                "raw": row,
                "label": row.get("name", base_name),
                "base": base_name,
                "param_type": param_type,
                "param_value": value,
            }
        )

    ranges: Dict[str, Tuple[float, float]] = {}
    for param_type, values in param_values.items():
        if values:
            ranges[param_type] = (min(values), max(values))

    for entry in enriched:
        param_type = entry["param_type"]
        param_value = entry["param_value"]
        if isinstance(param_type, str) and isinstance(param_value, float):
            min_val, max_val = ranges.get(param_type, (param_value, param_value))
            if math.isclose(max_val, min_val):
                normalized = 0.5
            else:
                normalized = (param_value - min_val) / (max_val - min_val)
            direction = PARAM_RULES.get(param_type, {}).get("direction", "low")
            strength = 1.0 - normalized if direction == "low" else normalized
            base_color = PARAM_RULES.get(param_type, {}).get("color", DEFAULT_COLOR)
            entry["color"] = blend_color(base_color, strength)
        else:
            entry["color"] = DEFAULT_COLOR
    return enriched


def sanitize_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", text)


def collect_metrics(
    entries: List[Dict[str, object]],
) -> List[Tuple[str, str, List[Tuple[str, float, str]]]]:
    grouped: Dict[str, Dict[str, List[Tuple[str, float, str]]]] = {
        prefix: {} for prefix in PREFIXES
    }
    for entry in entries:
        row = entry["raw"]
        label = entry["label"]
        color = entry["color"]
        for key, value in row.items():
            if key == "name" or not isinstance(value, (int, float)):
                continue
            for prefix in PREFIXES:
                if key.startswith(prefix):
                    grouped[prefix].setdefault(key, []).append(
                        (label, float(value), color)
                    )
    panels: List[Tuple[str, str, List[Tuple[str, float, str]]]] = []
    for prefix, bucket in grouped.items():
        for metric, values in bucket.items():
            panels.append((prefix, metric, values))
    return panels


def sort_records(records: Iterable[Tuple[str, float, str]]) -> List[Tuple[str, float, str]]:
    def sort_key(item):
        _, value, _ = item
        if math.isnan(value):
            return (1, 0.0)
        return (0, -value)

    return sorted(records, key=sort_key)


def draw_panel(ax, prefix: str, metric: str, records: Sequence[Tuple[str, float, str]]) -> None:
    ordered = sort_records(records)
    labels = [label for label, _, _ in ordered]
    values = [value if not math.isnan(value) else 0.0 for _, value, _ in ordered]
    colors = [
        color if not math.isnan(value) else "#bdbdbd" for (_, value, color) in ordered
    ]

    bars = ax.barh(labels, values, color=colors, edgecolor="#303030", linewidth=0.5)
    display_metric = (
        metric[len(prefix) + 1 :] if metric.startswith(f"{prefix}_") else metric
    )
    ax.set_title(
        f"{display_metric} ({prefix.title()})",
        loc="left",
        fontsize=12,
        weight="bold",
    )
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.invert_yaxis()
    ax.tick_params(labelsize=9)

    max_value = max([val for val in values if not math.isnan(val)] or [1.0])
    ax.set_xlim(0, max_value * 1.1)

    for bar, (_, value, _) in zip(bars, ordered):
        label_x = bar.get_width()
        if math.isnan(value):
            ax.text(
                0.01,
                bar.get_y() + bar.get_height() / 2,
                "NaN",
                va="center",
                ha="left",
                color="#555555",
                fontsize=8,
            )
        else:
            ax.text(
                label_x,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                ha="left",
                color="#1a1a1a",
                fontsize=8,
            )


def render_plots(
    panels: Sequence[Tuple[str, str, List[Tuple[str, float, str]]]],
    output_path: Path,
    file_format: str,
) -> None:
    if not panels:
        raise ValueError("No metrics to plot.")

    plt.style.use(DEFAULT_STYLE)
    height_ratios = [max(1, len(values)) for _, _, values in panels]
    total_rows = sum(height_ratios)
    fig_height = max(4, 0.32 * total_rows)
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(14, fig_height),
        squeeze=False,
        gridspec_kw={"height_ratios": height_ratios},
    )
    axes = axes.ravel()

    for ax, (prefix, metric, records) in zip(axes, panels):
        ax.set_facecolor(PREFIX_BACKGROUNDS.get(prefix, "#ffffff"))
        draw_panel(ax, prefix, metric, records)

    fig.suptitle(
        "Privacy, Divergence, and Utility Metrics by Experiment",
        fontsize=16,
        weight="bold",
    )
    fig.tight_layout(rect=(0.03, 0.03, 0.98, 0.97))
    dpi = 250 if file_format == "png" else None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format=file_format, dpi=dpi)
    plt.close(fig)


def export_per_metric(
    panels: Sequence[Tuple[str, str, List[Tuple[str, float, str]]]],
    directory: Path,
    file_format: str,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for prefix, metric, records in panels:
        width = 10
        height = max(2.5, 0.25 * len(records))
        fig, ax = plt.subplots(figsize=(width, height))
        ax.set_facecolor(PREFIX_BACKGROUNDS.get(prefix, "#ffffff"))
        draw_panel(ax, prefix, metric, records)
        fig.tight_layout()
        display_metric = (
            metric[len(prefix) + 1 :] if metric.startswith(f"{prefix}_") else metric
        )
        slug = sanitize_filename(f"{prefix}_{display_metric}")
        target = directory / f"{slug}.{file_format}"
        dpi = 250 if file_format == "png" else None
        fig.savefig(target, format=file_format, dpi=dpi)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = load_records(args.input)
    enriched = assign_colors(rows)
    panels = collect_metrics(enriched)
    render_plots(panels, args.output, args.format)
    if args.per_metric_dir:
        export_per_metric(panels, args.per_metric_dir, args.format)


if __name__ == "__main__":
    main()
