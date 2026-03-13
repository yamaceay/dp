from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import patheffects
import pandas as pd
import yaml


INPUT_DIR = Path("mds")
DATASET_CONFIG_PATH = Path("configs/visualize/datasets.yaml")
METHODS_CONFIG_PATH = Path("configs/visualize/methods.yaml")
PARAMS_CONFIG_PATH = Path("configs/visualize/params.yaml")
OUTPUT_DIR = Path("images")
ENABLED_DATASETS = {"db_bio", "tab"}

_palette = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#4C72B0",
    "#55A868",
    "#8172B3",
    "#C44E52",
]


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def parse_params(params_value: object) -> dict[str, object]:
    if isinstance(params_value, dict):
        return params_value
    if not isinstance(params_value, str):
        return {}
    try:
        parsed = ast.literal_eval(params_value)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def read_param_specs() -> dict[str, dict[str, Any]]:
    config = read_yaml(PARAMS_CONFIG_PATH)
    return {
        item["name"]: item
        for item in config.get("param_sets", [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }


def read_dataset_specs() -> dict[str, dict[str, Any]]:
    config = read_yaml(DATASET_CONFIG_PATH)
    result: dict[str, dict[str, Any]] = {}
    for dataset_set in config.get("dataset_sets", []):
        if not isinstance(dataset_set, dict):
            continue
        dataset_names: list[tuple[str, str]] = []
        scope = dataset_set.get("names")
        if isinstance(scope, list):
            for entry in scope:
                if isinstance(entry, dict) and isinstance(entry.get("name"), str):
                    dataset_names.append(
                        (
                            entry["name"],
                            str(entry.get("print_as") or entry["name"].replace("_", "-").upper()),
                        )
                    )
                elif isinstance(entry, str):
                    dataset_names.append((entry, entry.replace("_", "-").upper()))
        elif isinstance(dataset_set.get("name"), str):
            dataset_name = str(dataset_set["name"])
            dataset_names.append(
                (
                    dataset_name,
                    str(dataset_set.get("print_as") or dataset_name.replace("_", "-").upper()),
                )
            )
        scores = []
        for score in dataset_set.get("scores", []):
            if not isinstance(score, dict):
                continue
            name = score.get("name")
            if not isinstance(name, str):
                continue
            scores.append(
                {
                    "name": name,
                    "label": str(score.get("rename_as") or score.get("print_as") or name),
                    "best": score.get("best"),
                    "worst": score.get("worst"),
                    "category": metric_category(name),
                }
            )
        for dataset_name, dataset_label in dataset_names:
            result[dataset_name] = {
                "label": dataset_label,
                "scores": list(scores),
            }
    return result


def read_method_specs() -> list[dict[str, Any]]:
    config = read_yaml(METHODS_CONFIG_PATH)
    return [
        item
        for item in config.get("method_sets", [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    ]


def read_method_labels() -> dict[str, str]:
    labels: dict[str, str] = {"baseline": "Original"}
    for method_set in read_method_specs():
        for method_spec in method_set.get("methods", []):
            if not isinstance(method_spec, dict):
                continue
            method = method_spec.get("method")
            print_as = method_spec.get("print_as")
            if isinstance(method, str) and isinstance(print_as, str) and method not in labels:
                labels[method] = print_as
    return labels


def metric_category(metric_name: str) -> str:
    for prefix in ("privacy_", "utility_", "supervised_divergence_", "divergence_", "runtime_"):
        if metric_name.startswith(prefix):
            return prefix[:-1]
    raise ValueError(f"Unsupported metric category: {metric_name}")


def metric_output_name(metric_spec: dict[str, Any]) -> str:
    label = str(metric_spec.get("label") or metric_spec.get("name") or "metric")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_")
    return slug or "metric"


def metric_direction(metric_spec: dict[str, Any]) -> str:
    best = numeric_bound(metric_spec.get("best"))
    worst = numeric_bound(metric_spec.get("worst"))
    if best is None or worst is None:
        raise ValueError(f"Metric spec missing numeric best/worst: {metric_spec}")
    return "max" if best > worst else "min"


def numeric_bound(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == 'float("inf")' or normalized == "float('inf')" or normalized == "inf":
            return float("inf")
        if normalized == 'float("-inf")' or normalized == "float('-inf')" or normalized == "-inf":
            return float("-inf")
        try:
            return float(normalized)
        except ValueError:
            return None
    return None


def finite_metric_bounds(metric_spec: dict[str, Any]) -> tuple[float, float] | None:
    best = numeric_bound(metric_spec.get("best"))
    worst = numeric_bound(metric_spec.get("worst"))
    if best is None or worst is None:
        return None
    if not math.isfinite(best) or not math.isfinite(worst):
        return None
    lower = min(best, worst)
    upper = max(best, worst)
    if lower == upper:
        return None
    return lower, upper


def sort_param_value(value: object) -> tuple[int, object]:
    if isinstance(value, (int, float)):
        return (0, float(value))
    try:
        return (0, float(value))
    except Exception:
        return (1, str(value))


def sort_param_items(params: dict[str, object], param_specs: dict[str, dict[str, Any]]) -> list[tuple[str, object]]:
    ordered = sorted(
        params.items(),
        key=lambda item: (
            list(param_specs.keys()).index(item[0]) if item[0] in param_specs else len(param_specs),
            sort_param_value(item[1]),
        ),
    )
    return ordered


def format_param_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def compact_params(
    params: dict[str, object],
    param_specs: dict[str, dict[str, Any]],
    exclude: set[str] | None = None,
) -> str:
    excluded = exclude or set()
    parts: list[str] = []
    for key, value in sort_param_items(params, param_specs):
        if key in excluded:
            continue
        key_label = str(param_specs.get(key, {}).get("print_as") or key)
        parts.append(f"{key_label}={format_param_value(value)}")
    return ", ".join(parts)


def method_color(method: str, color_cache: dict[str, tuple[float, float, float]]) -> tuple[float, float, float]:
    if method == "baseline":
        return mcolors.to_rgb("#7f7f7f")
    if method not in color_cache:
        color_cache[method] = mcolors.to_rgb(_palette[len(color_cache) % len(_palette)])
    return color_cache[method]


def load_dataset_frame(dataset_name: str) -> pd.DataFrame:
    path = INPUT_DIR / f"{dataset_name}_logs.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if "params" in frame.columns:
        frame["params_dict"] = frame["params"].apply(parse_params)
    else:
        frame["params_dict"] = [{} for _ in range(len(frame))]
    return frame


def values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) == float(right)
    return left == right


def dataset_matches_scope(dataset: str, datasets_scope: Any) -> bool:
    if datasets_scope is None:
        return True
    if isinstance(datasets_scope, list):
        return dataset in datasets_scope
    raise ValueError(f"Unsupported datasets scope format: {type(datasets_scope)}")


def params_one_run_keys(params_one_run: Any) -> set[str]:
    if isinstance(params_one_run, dict):
        return set(params_one_run.keys())
    if isinstance(params_one_run, list):
        return set(params_one_run)
    raise ValueError(f"Unsupported params_one_run format: {type(params_one_run)}")


def matches_params_one_run_values(params: dict[str, Any], params_one_run: Any) -> bool:
    if isinstance(params_one_run, list):
        return True
    if not isinstance(params_one_run, dict):
        raise ValueError(f"Unsupported params_one_run format: {type(params_one_run)}")
    for param_name, allowed_values in params_one_run.items():
        if param_name not in params:
            return False
        if not isinstance(allowed_values, list):
            raise ValueError(f"params_one_run values must be lists. Got {param_name}={type(allowed_values)}")
        if not any(values_equal(params[param_name], allowed_value) for allowed_value in allowed_values):
            return False
    return True


def matches_method_spec(dataset: str, params: dict[str, Any], method: str, method_spec: dict[str, Any]) -> bool:
    if not dataset_matches_scope(dataset, method_spec.get("datasets")):
        return False
    expected_method = method_spec.get("method")
    if method != expected_method:
        return False
    params_one_run = method_spec.get("params_one_run", [])
    expected_param_keys = set(method_spec.get("params", [])) | params_one_run_keys(params_one_run)
    if set(params.keys()) != expected_param_keys:
        return False
    return matches_params_one_run_values(params, params_one_run)


def filter_method_set_rows(dataset: str, frame: pd.DataFrame, method_set: dict[str, Any]) -> pd.DataFrame:
    if not dataset_matches_scope(dataset, method_set.get("datasets")):
        return frame.iloc[0:0].copy()
    methods = method_set.get("methods", [])
    mask = frame.apply(
        lambda row: any(
            matches_method_spec(dataset, row["params_dict"], str(row["method"]), method_spec)
            for method_spec in methods
            if isinstance(method_spec, dict)
        ),
        axis=1,
    )
    return frame[mask].copy()


def representative_rows(frame: pd.DataFrame, metric_name: str, direction: str) -> pd.DataFrame:
    working = frame.copy()
    working[metric_name] = pd.to_numeric(working[metric_name], errors="coerce")
    working = working.dropna(subset=[metric_name])
    if working.empty:
        return working
    ascending = direction == "min"
    representatives = []
    for _, group in working.groupby("method", sort=False):
        group_sorted = group.sort_values(metric_name, ascending=ascending, kind="mergesort")
        representatives.append(group_sorted.iloc[0])
    result = pd.DataFrame(representatives)
    baseline = result[result["method"] == "baseline"]
    others = result[result["method"] != "baseline"].sort_values(metric_name, ascending=ascending, kind="mergesort")
    if not baseline.empty:
        return pd.concat([baseline, others], ignore_index=True)
    return others.reset_index(drop=True)


def add_value_annotations(
    ax: plt.Axes,
    positions: list[int],
    values: list[float],
    notes: list[str],
) -> None:
    value_min = min(values)
    value_max = max(values)
    span = value_max - value_min
    offset = span * 0.02 if span > 0 else max(abs(value_max) * 0.02, 0.01)
    for position, value, note in zip(positions, values, notes):
        label = f"{value:.4g}"
        if note:
            label = f"{label} | {note}"
        ax.text(value + offset, position, label, va="center", ha="left", fontsize=9)


def zoom_axis_limits(ax: plt.Axes, values: list[float]) -> None:
    value_min = min(values)
    value_max = max(values)
    span = value_max - value_min
    if span == 0:
        padding = max(abs(value_max) * 0.1, 0.05)
    else:
        padding = span * 0.08
    ax.set_xlim(value_min - padding, value_max + padding * 2.5)


def ordered_param_values(values: set[object], param_name: str, param_specs: dict[str, dict[str, Any]]) -> list[object]:
    ordered = sorted(values, key=sort_param_value)
    if param_specs.get(param_name, {}).get("print_order") == "high_to_low":
        ordered.reverse()
    return ordered


def save_manifest(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path.with_suffix(".csv"), index=False)
    with path.with_suffix(".json").open("w") as f:
        json.dump(frame.to_dict(orient="records"), f, indent=2)


def heatmap_secondary_param(frame: pd.DataFrame) -> str | None:
    secondary_params: set[str] = set()
    methods: set[str] = set()
    for row in frame.itertuples():
        params_dict = row.params_dict if isinstance(row.params_dict, dict) else {}
        if "epsilon" not in params_dict or len(params_dict) != 2:
            continue
        methods.add(str(row.method))
        for key in params_dict:
            if key != "epsilon":
                secondary_params.add(key)
    if len(secondary_params) != 1:
        return None
    if len(methods) != 1:
        return None
    return next(iter(secondary_params))


def heatmap_frame(frame: pd.DataFrame, metric_name: str, secondary_param: str) -> pd.DataFrame:
    working = frame.copy()
    working[metric_name] = pd.to_numeric(working[metric_name], errors="coerce")
    working = working.dropna(subset=[metric_name])
    mask = working["params_dict"].apply(
        lambda params: isinstance(params, dict)
        and "epsilon" in params
        and secondary_param in params
        and len(params) == 2
    )
    return working[mask].copy()


def plot_summary(
    dataset_name: str,
    dataset_label: str,
    metric_spec: dict[str, Any],
    frame: pd.DataFrame,
    method_labels: dict[str, str],
    param_specs: dict[str, dict[str, Any]],
) -> None:
    metric_name = str(metric_spec["name"])
    metric_output = metric_output_name(metric_spec)
    direction = metric_direction(metric_spec)
    reps = representative_rows(frame, metric_name, direction)
    if reps.empty:
        return
    metric_label = str(metric_spec["label"])
    plot_rows = reps.copy()
    plot_rows["method_label"] = plot_rows["method"].apply(lambda method: method_labels.get(str(method), str(method)))
    plot_rows["annotation"] = plot_rows.apply(
        lambda row: compact_params(row["params_dict"], param_specs) if row["method"] != "baseline" else "",
        axis=1,
    )
    plot_rows["rank"] = range(1, len(plot_rows) + 1)

    output_base = OUTPUT_DIR / "summary" / dataset_name / metric_spec["category"] / metric_output
    output_base.parent.mkdir(parents=True, exist_ok=True)
    manifest = plot_rows[["rank", "method", "method_label", metric_name, "params"]].copy()
    save_manifest(output_base.with_name(f"{output_base.name}_manifest"), manifest)

    fig_height = max(4.0, 0.65 * len(plot_rows) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    color_cache: dict[str, tuple[float, float, float]] = {}
    positions = list(range(len(plot_rows)))
    values = [float(value) for value in plot_rows[metric_name]]
    colors = [method_color(str(method), color_cache) for method in plot_rows["method"]]
    bars = ax.barh(positions, values, color=colors, alpha=0.92)
    if not plot_rows[plot_rows["method"] == "baseline"].empty:
        baseline_index = int(plot_rows.index[plot_rows["method"] == "baseline"][0])
        bars[baseline_index].set_hatch("//")
        if len(plot_rows) > 1:
            ax.axhline(baseline_index + 0.5, color="#444444", linewidth=1.0, alpha=0.8)
    labels = [f"{row.method_label}" for row in plot_rows.itertuples()]
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    zoom_axis_limits(ax, values)
    add_value_annotations(ax, positions, values, plot_rows["annotation"].tolist())
    ax.set_xlabel(metric_label)
    ax.set_title(f"{dataset_label} | {metric_label}", fontsize=14, pad=12)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=220)
    plt.close(fig)


def facet_epsilon_values(frame: pd.DataFrame, param_specs: dict[str, dict[str, Any]]) -> list[object]:
    values = {
        params["epsilon"]
        for params in frame["params_dict"]
        if isinstance(params, dict) and "epsilon" in params and len(params) >= 2
    }
    if not values:
        return []
    ordered = sorted(values, key=sort_param_value)
    epsilon_spec = param_specs.get("epsilon", {})
    if epsilon_spec.get("print_order") == "high_to_low":
        ordered.reverse()
    return ordered


def variant_label(
    method: str,
    params: dict[str, object],
    method_labels: dict[str, str],
    param_specs: dict[str, dict[str, Any]],
    exclude: set[str] | None = None,
) -> str:
    base = method_labels.get(method, method)
    params_label = compact_params(params, param_specs, exclude=exclude)
    if not params_label:
        return base
    return f"{base} ({params_label})"


def panel_rows_for_epsilon(frame: pd.DataFrame, epsilon_value: object) -> pd.DataFrame:
    if epsilon_value is None:
        return frame.copy()
    mask = frame["params_dict"].apply(
        lambda params: not isinstance(params, dict)
        or "epsilon" not in params
        or values_equal(params["epsilon"], epsilon_value)
    )
    return frame[mask].copy()


def plot_drilldown(
    dataset_name: str,
    dataset_label: str,
    metric_spec: dict[str, Any],
    frame: pd.DataFrame,
    method_set: dict[str, Any],
    method_labels: dict[str, str],
    param_specs: dict[str, dict[str, Any]],
) -> None:
    metric_name = str(metric_spec["name"])
    metric_output = metric_output_name(metric_spec)
    working = frame.copy()
    working[metric_name] = pd.to_numeric(working[metric_name], errors="coerce")
    working = working.dropna(subset=[metric_name])
    if working.empty:
        return

    epsilon_values = facet_epsilon_values(working, param_specs)
    panels = epsilon_values if epsilon_values else [None]
    fig, axes = create_bar_axes(len(panels), max(4.5, 0.45 * max(len(working), 6)))
    color_cache: dict[str, tuple[float, float, float]] = {}
    direction = metric_direction(metric_spec)
    ascending = direction == "min"
    metric_label = str(metric_spec["label"])

    for ax, epsilon_value in zip(axes, panels):
        panel = panel_rows_for_epsilon(working, epsilon_value)
        if panel.empty:
            ax.set_visible(False)
            continue
        labels = []
        values = []
        colors = []
        notes = []
        order_frame = panel.sort_values(metric_name, ascending=ascending, kind="mergesort")
        baseline = order_frame[order_frame["method"] == "baseline"]
        others = order_frame[order_frame["method"] != "baseline"]
        if not baseline.empty:
            order_frame = pd.concat([baseline, others], ignore_index=True)
        for row in order_frame.itertuples():
            params_dict = row.params_dict if isinstance(row.params_dict, dict) else {}
            exclude = {"epsilon"} if epsilon_value is not None else set()
            label = variant_label(str(row.method), params_dict, method_labels, param_specs, exclude=exclude)
            labels.append(label)
            values.append(float(getattr(row, metric_name)))
            colors.append(method_color(str(row.method), color_cache))
            if epsilon_value is not None and isinstance(params_dict, dict) and "epsilon" in params_dict:
                notes.append("")
            else:
                notes.append("")
        positions = list(range(len(order_frame)))
        bars = ax.barh(positions, values, color=colors, alpha=0.92)
        if not baseline.empty:
            bars[0].set_hatch("//")
            if len(order_frame) > 1:
                ax.axhline(0.5, color="#444444", linewidth=1.0, alpha=0.8)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        zoom_axis_limits(ax, values)
        add_value_annotations(ax, positions, values, notes)
        ax.grid(axis="x", alpha=0.25)
        if epsilon_value is None:
            ax.set_title(metric_label, fontsize=12)
        else:
            epsilon_label = str(param_specs.get("epsilon", {}).get("print_as") or "epsilon")
            ax.set_title(f"{epsilon_label}={format_param_value(epsilon_value)}", fontsize=12)
        ax.set_xlabel(metric_label)

    method_set_label = str(method_set.get("print_as") or method_set["name"])
    fig.suptitle(f"{dataset_label} | {metric_label} | {method_set_label}", fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = OUTPUT_DIR / "drilldown" / dataset_name / str(method_set["name"]) / f"{metric_output}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_heatmap(
    dataset_name: str,
    dataset_label: str,
    metric_spec: dict[str, Any],
    frame: pd.DataFrame,
    method_set: dict[str, Any],
    method_labels: dict[str, str],
    param_specs: dict[str, dict[str, Any]],
) -> None:
    metric_name = str(metric_spec["name"])
    metric_output = metric_output_name(metric_spec)
    secondary_param = heatmap_secondary_param(frame)
    if secondary_param is None:
        return
    working = heatmap_frame(frame, metric_name, secondary_param)
    if working.empty:
        return

    epsilon_values = ordered_param_values(
        {params["epsilon"] for params in working["params_dict"] if isinstance(params, dict)},
        "epsilon",
        param_specs,
    )
    secondary_values = ordered_param_values(
        {params[secondary_param] for params in working["params_dict"] if isinstance(params, dict)},
        secondary_param,
        param_specs,
    )
    grid = pd.DataFrame(index=secondary_values, columns=epsilon_values, dtype=float)
    params_lookup: dict[tuple[object, object], dict[str, object]] = {}
    for row in working.itertuples():
        params_dict = row.params_dict if isinstance(row.params_dict, dict) else {}
        epsilon_value = params_dict["epsilon"]
        secondary_value = params_dict[secondary_param]
        grid.loc[secondary_value, epsilon_value] = float(getattr(row, metric_name))
        params_lookup[(secondary_value, epsilon_value)] = params_dict

    method_name = str(working.iloc[0]["method"])
    method_label = method_labels.get(method_name, method_name)
    metric_label = str(metric_spec["label"])
    x_tick_labels = [format_param_value(value) for value in epsilon_values]
    y_tick_labels = [format_param_value(value) for value in secondary_values]
    secondary_label = str(param_specs.get(secondary_param, {}).get("print_as") or secondary_param)
    epsilon_label = str(param_specs.get("epsilon", {}).get("print_as") or "epsilon")

    bounds = finite_metric_bounds(metric_spec)
    values = grid.to_numpy(dtype=float)
    finite_values = values[~pd.isna(values)]
    if finite_values.size == 0:
        return
    if bounds is None:
        vmin = float(finite_values.min())
        vmax = float(finite_values.max())
        if vmin == vmax:
            delta = max(abs(vmin) * 0.05, 0.01)
            vmin -= delta
            vmax += delta
    else:
        vmin, vmax = bounds

    fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * len(epsilon_values) + 2.5), max(4.5, 1.0 * len(secondary_values) + 1.8)))
    image = ax.imshow(values, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(epsilon_values)))
    ax.set_xticklabels(x_tick_labels)
    ax.set_yticks(range(len(secondary_values)))
    ax.set_yticklabels(y_tick_labels)
    ax.set_xlabel(epsilon_label)
    ax.set_ylabel(secondary_label)
    ax.set_title(f"{dataset_label} | {method_label} | {metric_label}", fontsize=14, pad=12)
    for row_index, secondary_value in enumerate(secondary_values):
        for col_index, epsilon_value in enumerate(epsilon_values):
            cell_value = grid.loc[secondary_value, epsilon_value]
            if pd.isna(cell_value):
                continue
            text = ax.text(
                col_index,
                row_index,
                f"{float(cell_value):.3g}",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
            )
            text.set_path_effects(
                [
                    patheffects.Stroke(linewidth=2.2, foreground="black", alpha=0.85),
                    patheffects.Normal(),
                ]
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.ax.set_ylabel(metric_label, rotation=270, labelpad=14)
    fig.tight_layout()

    output_path = OUTPUT_DIR / "heatmap" / dataset_name / str(method_set["name"]) / f"{metric_output}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    manifest_rows = []
    for secondary_value in secondary_values:
        for epsilon_value in epsilon_values:
            if (secondary_value, epsilon_value) not in params_lookup:
                continue
            manifest_rows.append(
                {
                    "method": method_name,
                    "epsilon": epsilon_value,
                    secondary_param: secondary_value,
                    metric_name: float(grid.loc[secondary_value, epsilon_value]),
                    "params": params_lookup[(secondary_value, epsilon_value)],
                }
            )
    manifest = pd.DataFrame(manifest_rows)
    save_manifest(output_path.with_name(f"{output_path.stem}_manifest"), manifest)


def create_bar_axes(num_panels: int, panel_height: float) -> tuple[plt.Figure, list[plt.Axes]]:
    if num_panels == 5:
        fig = plt.figure(figsize=(15, panel_height * 2.1))
        grid = fig.add_gridspec(2, 6)
        first = fig.add_subplot(grid[0, 1:3])
        second = fig.add_subplot(grid[0, 3:5], sharex=first)
        third = fig.add_subplot(grid[1, 0:2])
        fourth = fig.add_subplot(grid[1, 2:4], sharex=third)
        fifth = fig.add_subplot(grid[1, 4:6], sharex=third)
        return fig, [first, second, third, fourth, fifth]

    if num_panels <= 3:
        fig, axes = plt.subplots(1, num_panels, figsize=(max(8.0, 6.0 * num_panels), panel_height), squeeze=False)
        return fig, list(axes[0])

    ncols = 3
    nrows = math.ceil(num_panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, panel_height * nrows), squeeze=False)
    flat_axes = list(axes.flat)
    for extra_ax in flat_axes[num_panels:]:
        extra_ax.set_visible(False)
    return fig, flat_axes[:num_panels]


def resolve_datasets(args: argparse.Namespace, dataset_specs: dict[str, dict[str, Any]]) -> list[str]:
    if args.datasets:
        return [dataset for dataset in args.datasets if dataset in dataset_specs and dataset in ENABLED_DATASETS]
    return [dataset for dataset in dataset_specs if dataset in ENABLED_DATASETS]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate single-metric ranking and drill-down plots.")
    parser.add_argument("--datasets", nargs="*", default=None)
    args = parser.parse_args()

    dataset_specs = read_dataset_specs()
    method_labels = read_method_labels()
    method_sets = read_method_specs()
    param_specs = read_param_specs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_name in resolve_datasets(args, dataset_specs):
        dataset_spec = dataset_specs[dataset_name]
        dataset_label = str(dataset_spec["label"])
        frame = load_dataset_frame(dataset_name)
        for metric_spec in dataset_spec["scores"]:
            metric_name = str(metric_spec["name"])
            if metric_name not in frame.columns:
                continue
            plot_summary(dataset_name, dataset_label, metric_spec, frame, method_labels, param_specs)
            for method_set in method_sets:
                filtered = filter_method_set_rows(dataset_name, frame, method_set)
                if filtered.empty:
                    continue
                plot_drilldown(
                    dataset_name,
                    dataset_label,
                    metric_spec,
                    filtered,
                    method_set,
                    method_labels,
                    param_specs,
                )
                plot_heatmap(
                    dataset_name,
                    dataset_label,
                    metric_spec,
                    filtered,
                    method_set,
                    method_labels,
                    param_specs,
                )


if __name__ == "__main__":
    main()
