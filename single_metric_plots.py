from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path
from typing import Any
from collections.abc import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import patheffects
import pandas as pd
import yaml

from plot_layout import (
    HYPERPARAM_AXIS_ORDER,
    LAYOUT,
    SUBTITLE_Y,
    SUPTITLE_Y,
    TIGHT_LAYOUT_RECT,
    create_panel_axes,
    param_directed_sort_key,
    style_x_axis,
)

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

RUNTIME_STACK_PARTS = [
    ("runtime_deid_total_time_s", "De-identification", "#E69F00"),
    ("runtime_bk_gen_total_time_s", "TRI preprocessing", "#4C72B0"),
    ("runtime_train_total_time_s", "TRI training", "#55A868"),
    ("runtime_risk_total_time_s", "TRI risk precomputation", "#C44E52"),
    ("runtime_real_anon_total_time_s", "Anonymization", "#8172B3"),
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
                    "exclude": score.get("exclude", []),
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
    labels: dict[str, str] = {"baseline": "Original", "dummy": "Dummy"}
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


def normalized_excluded_methods(raw_methods: Any) -> set[str]:
    if not isinstance(raw_methods, list):
        return set()
    normalized: set[str] = set()
    for item in raw_methods:
        if not isinstance(item, str):
            continue
        token = item.strip().lower()
        if not token:
            continue
        normalized.add(token)
    return normalized


def method_is_excluded(method: str, excluded_methods: set[str]) -> bool:
    normalized_method = method.strip().lower()
    if normalized_method in excluded_methods:
        return True
    return any(normalized_method.startswith(f"{excluded}_") for excluded in excluded_methods)


def apply_metric_exclusions(frame: pd.DataFrame, metric_spec: dict[str, Any]) -> pd.DataFrame:
    excluded_methods = normalized_excluded_methods(metric_spec.get("exclude", []))
    if not excluded_methods:
        return frame
    mask = frame["method"].astype(str).apply(lambda method: not method_is_excluded(method, excluded_methods))
    return frame[mask].copy()


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


def metric_heatmap_bounds(
    metric_name: str,
    metric_spec: dict[str, Any],
    finite_values: Any,
) -> tuple[float, float]:
    normalized_name = metric_name.strip().lower()
    label = str(metric_spec.get("label") or metric_spec.get("name") or metric_name).strip().lower()

    is_mae_metric = "mae" in normalized_name or "mae" in label
    is_unit_interval_metric = (
        "acc" in normalized_name
        or "f1" in normalized_name
        or "mrr" in normalized_name
        or "reciprocal_rank" in normalized_name
        or "acc" in label
        or "f1" in label
        or "mrr" in label
    )

    values_max = float(finite_values.max())
    values_min = float(finite_values.min())

    if is_mae_metric:
        return 0.0, max(3.0, values_max)
    if is_unit_interval_metric:
        return 0.0, 1.0

    bounds = finite_metric_bounds(metric_spec)
    if bounds is not None:
        return bounds

    vmin = values_min
    vmax = values_max
    if vmin == vmax:
        delta = max(abs(vmin) * 0.05, 0.01)
        vmin -= delta
        vmax += delta
    return vmin, vmax


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
            param_directed_sort_key(item[0], item[1], param_specs),
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
    if method == "dummy":
        return mcolors.to_rgb("#a9a9a9")
    if method not in color_cache:
        color_cache[method] = mcolors.to_rgb(_palette[len(color_cache) % len(_palette)])
    return color_cache[method]


def load_dataset_frame(dataset_name: str) -> pd.DataFrame:
    path = INPUT_DIR / dataset_name / "logs.csv"
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


def constrained_params_from_spec(params_field: Any) -> dict[str, list[Any]]:
    if isinstance(params_field, dict):
        constrained: dict[str, list[Any]] = {}
        for key, allowed in params_field.items():
            if not isinstance(key, str):
                continue
            if isinstance(allowed, list):
                constrained[key] = allowed
            else:
                constrained[key] = [allowed]
        return constrained
    return {}


def required_param_keys_from_spec(method_spec: dict[str, Any]) -> set[str]:
    params_field = method_spec.get("params", [])
    if isinstance(params_field, list):
        declared_params = {param for param in params_field if isinstance(param, str)}
    elif isinstance(params_field, dict):
        declared_params = {key for key in params_field.keys() if isinstance(key, str)}
    else:
        declared_params = set()
    params_one_run = method_spec.get("params_one_run", [])
    return declared_params | params_one_run_keys(params_one_run)


def method_spec_value_constraints(method_spec: dict[str, Any]) -> dict[str, list[Any]]:
    constraints = constrained_params_from_spec(method_spec.get("params", []))
    params_one_run = method_spec.get("params_one_run", [])
    if isinstance(params_one_run, dict):
        for key, allowed_values in params_one_run.items():
            if isinstance(key, str) and isinstance(allowed_values, list):
                constraints[key] = allowed_values
    return constraints


def matches_method_spec(dataset: str, params: dict[str, Any], method: str, method_spec: dict[str, Any]) -> bool:
    if not dataset_matches_scope(dataset, method_spec.get("datasets")):
        return False
    expected_method = method_spec.get("method")
    if method != expected_method:
        return False
    expected_param_keys = required_param_keys_from_spec(method_spec)
    if not expected_param_keys.issubset(set(params.keys())):
        return False
    ordered_hparams = {name.strip().lower() for name in HYPERPARAM_AXIS_ORDER}
    present_hparams = {key for key in params if key.strip().lower() in ordered_hparams}
    expected_hparams = {key for key in expected_param_keys if key.strip().lower() in ordered_hparams}
    if present_hparams != expected_hparams:
        return False
    for param_name, allowed_values in method_spec_value_constraints(method_spec).items():
        if param_name not in params:
            return False
        if not any(values_equal(params[param_name], allowed_value) for allowed_value in allowed_values):
            return False
    params_one_run = method_spec.get("params_one_run", [])
    return matches_params_one_run_values(params, params_one_run)


def method_spec_specificity(method_spec: dict[str, Any]) -> tuple[int, int]:
    expected_param_keys = required_param_keys_from_spec(method_spec)
    constrained_keys = set(method_spec_value_constraints(method_spec).keys())
    return len(constrained_keys), len(expected_param_keys)


def filter_method_set_rows(dataset: str, frame: pd.DataFrame, method_set: dict[str, Any]) -> pd.DataFrame:
    if not dataset_matches_scope(dataset, method_set.get("datasets")):
        return frame.iloc[0:0].copy()
    methods = method_set.get("methods", [])
    always_include_methods = {"dummy"}
    mask = frame.apply(
        lambda row: (
            str(row["method"]) in always_include_methods
            or any(
                matches_method_spec(dataset, row["params_dict"], str(row["method"]), method_spec)
                for method_spec in methods
                if isinstance(method_spec, dict)
            )
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


def zoom_limits(values: list[float]) -> tuple[float, float]:
    value_min = min(values)
    value_max = max(values)
    span = value_max - value_min
    if span == 0:
        padding = max(abs(value_max) * 0.1, 0.05)
    else:
        padding = span * 0.08
    return value_min - padding, value_max + padding * 2.5


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
    base_label: str | None = None,
) -> str:
    base = base_label if isinstance(base_label, str) else method_labels.get(method, method)
    params_label = compact_params(params, param_specs, exclude=exclude)
    if not params_label:
        return base
    return f"{base} ({params_label})"


def global_variant_labels(
    frame: pd.DataFrame,
    method_labels: dict[str, str],
    param_specs: dict[str, dict[str, Any]],
    exclude: set[str] | None = None,
    base_label_resolver: Callable[[str, dict[str, object]], str] | None = None,
) -> list[str]:
    excluded = exclude or set()
    filtered_param_key_sets: list[set[str]] = []
    methods_with_filtered_params: dict[str, set[str]] = {}
    ordered_hparam_keys = {name.strip().lower() for name in HYPERPARAM_AXIS_ORDER}
    for row in frame.itertuples():
        params_dict = row.params_dict if isinstance(row.params_dict, dict) else {}
        filtered_params = {key for key in params_dict.keys() if key not in excluded}
        filtered_param_key_sets.append(filtered_params)
        method_name = str(row.method)
        method_filtered = methods_with_filtered_params.setdefault(method_name, set())
        method_filtered.update(filtered_params)

    filtered_hparams = sorted(
        {key for keys in filtered_param_key_sets for key in keys if key.strip().lower() in ordered_hparam_keys},
        key=lambda key: list(HYPERPARAM_AXIS_ORDER.keys()).index(key.strip().lower())
        if key.strip().lower() in HYPERPARAM_AXIS_ORDER
        else len(HYPERPARAM_AXIS_ORDER),
    )
    active_threshold_param = filtered_hparams[0] if len(filtered_hparams) == 1 else None
    methods_with_threshold = {
        method_name
        for method_name, method_params in methods_with_filtered_params.items()
        if active_threshold_param is not None and active_threshold_param in method_params
    }

    seen: dict[str, tuple[int, str, int, tuple[tuple[str, tuple[int, object]], ...], str]] = {}
    for row in frame.itertuples():
        params_dict = row.params_dict if isinstance(row.params_dict, dict) else {}
        method_name = str(row.method)
        resolved_base_label = (
            base_label_resolver(method_name, params_dict)
            if base_label_resolver is not None
            else method_labels.get(method_name, method_name)
        )
        label = variant_label(
            method_name,
            params_dict,
            method_labels,
            param_specs,
            exclude=excluded,
            base_label=resolved_base_label,
        )
        base_label = resolved_base_label
        params_label = compact_params(params_dict, param_specs, exclude=excluded)
        filtered_params = {key: value for key, value in params_dict.items() if key not in excluded}
        has_ordered_hparam = any(str(key).lower() in HYPERPARAM_AXIS_ORDER for key in filtered_params)
        params_signature = tuple(
            (key, param_directed_sort_key(key, value, param_specs))
            for key, value in sort_param_items(filtered_params, param_specs)
        )
        method_has_threshold_family = (
            active_threshold_param is not None
            and method_name in methods_with_threshold
            and method_name not in {"baseline", "presidio"}
        )
        is_threshold_variant = (
            method_has_threshold_family
            and active_threshold_param in filtered_params
        )
        is_full_variant = (
            method_has_threshold_family
            and active_threshold_param not in filtered_params
        )

        if method_name == "baseline":
            rank = 0
        elif method_name == "dummy":
            rank = 1
        elif active_threshold_param is not None and method_name == "presidio" and methods_with_threshold:
            rank = 2
        elif is_threshold_variant:
            rank = 3
        elif is_full_variant:
            rank = 4
        elif method_name == "presidio":
            rank = 2
        else:
            rank = 3
        hparam_rank = 0 if has_ordered_hparam else 1
        if label not in seen:
            seen[label] = (rank, base_label, hparam_rank, params_signature, params_label)
    return [
        label
        for label, _ in sorted(
            seen.items(),
            key=lambda item: (
                item[1][0],
                item[1][1].lower(),
                item[1][2],
                item[1][3],
                item[1][4].lower(),
                item[0].lower(),
            ),
        )
    ]


def method_set_display_label(
    dataset_name: str,
    method_set: dict[str, Any],
    method: str,
    params: dict[str, Any],
    method_labels: dict[str, str],
) -> str:
    best_label: str | None = None
    best_specificity: tuple[int, int] = (-1, -1)
    for method_spec in method_set.get("methods", []):
        if not isinstance(method_spec, dict):
            continue
        if not matches_method_spec(dataset_name, params, method, method_spec):
            continue
        print_as = method_spec.get("print_as")
        if not isinstance(print_as, str) or not print_as.strip():
            continue
        specificity = method_spec_specificity(method_spec)
        if specificity > best_specificity:
            best_specificity = specificity
            best_label = print_as.strip()
    if best_label is not None:
        return best_label
    return method_labels.get(method, method)


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
    fig, axes = create_bar_axes(len(panels))
    if len(panels) > 1:
        fig.set_size_inches(fig.get_figwidth() * 1.45, fig.get_figheight(), forward=True)
    color_cache: dict[str, tuple[float, float, float]] = {}
    direction = metric_direction(metric_spec)
    ascending = direction == "min"
    metric_label = str(metric_spec["label"])
    method_set_label = str(method_set.get("print_as") or method_set["name"])
    all_values: list[float] = []
    excluded_params = {"epsilon"} if epsilon_values else set()
    label_resolver = lambda method, params: method_set_display_label(
        dataset_name,
        method_set,
        method,
        params,
        method_labels,
    )
    ordered_labels = global_variant_labels(
        working,
        method_labels,
        param_specs,
        exclude=excluded_params,
        base_label_resolver=label_resolver,
    )
    baseline_label = method_labels.get("baseline", "baseline")
    dummy_label = method_labels.get("dummy", "dummy")

    panel_count = len(panels)
    for panel_index, (ax, epsilon_value) in enumerate(zip(axes, panels)):
        panel = panel_rows_for_epsilon(working, epsilon_value)
        if panel.empty:
            ax.set_visible(False)
            continue
        panel_values_by_label: dict[str, tuple[float, str]] = {}
        panel = panel.sort_values(metric_name, ascending=ascending, kind="mergesort")
        for row in panel.itertuples():
            params_dict = row.params_dict if isinstance(row.params_dict, dict) else {}
            label = variant_label(
                str(row.method),
                params_dict,
                method_labels,
                param_specs,
                exclude=excluded_params,
                base_label=label_resolver(str(row.method), params_dict),
            )
            value = float(getattr(row, metric_name))
            current = panel_values_by_label.get(label)
            if current is None:
                panel_values_by_label[label] = (value, str(row.method))
                continue
            current_value = current[0]
            if (ascending and value < current_value) or (not ascending and value > current_value):
                panel_values_by_label[label] = (value, str(row.method))

        panel_positions: list[int] = []
        panel_values: list[float] = []
        panel_colors: list[tuple[float, float, float]] = []
        panel_methods: list[str] = []
        for index, label in enumerate(ordered_labels):
            if label not in panel_values_by_label:
                continue
            value, method_name = panel_values_by_label[label]
            panel_positions.append(index)
            panel_values.append(value)
            panel_methods.append(method_name)
            panel_colors.append(method_color(method_name, color_cache))

        bars = ax.barh(panel_positions, panel_values, color=panel_colors, alpha=0.92)
        all_values.extend(panel_values)
        for bar_index, method_name in enumerate(panel_methods):
            if method_name == "baseline":
                bars[bar_index].set_hatch("//")
            elif method_name == "dummy":
                bars[bar_index].set_hatch("\\\\")
                bars[bar_index].set_linewidth(1.0)
                bars[bar_index].set_edgecolor("#555555")
        anchor_count = 0
        for label in ordered_labels:
            if label.startswith(baseline_label) or label.startswith(dummy_label):
                anchor_count += 1
            else:
                break
        if anchor_count > 0:
            ax.axhline(anchor_count - 0.5, color="#444444", linewidth=1.0, alpha=0.8)

        ax.set_yticks(list(range(len(ordered_labels))))
        ax.set_yticklabels(ordered_labels, fontsize=9)
        if show_y_labels(panel_index, panel_count):
            ax.tick_params(axis="y", length=3)
        else:
            for tick_label in ax.get_yticklabels():
                tick_label.set_visible(False)
            ax.tick_params(axis="y", length=0)

        ax.invert_yaxis()
        if panel_values:
            add_value_annotations(ax, panel_positions, panel_values, [""] * len(panel_values))
        ax.grid(axis="x", alpha=0.25)
        if len(panels) > 1:
            if epsilon_value is None:
                ax.set_title("all epsilon", fontsize=11, pad=12)
            else:
                epsilon_label = str(param_specs.get("epsilon", {}).get("print_as") or "epsilon")
                ax.set_title(f"{epsilon_label}={format_param_value(epsilon_value)}", fontsize=11, pad=12)
        else:
            if epsilon_value is None:
                ax.set_title(f"{dataset_label} | {method_set_label} | {metric_label}", fontsize=12, pad=14)
            else:
                epsilon_label = str(param_specs.get("epsilon", {}).get("print_as") or "epsilon")
                ax.set_title(
                    f"{dataset_label} | {method_set_label} | {metric_label} | "
                    f"{epsilon_label}={format_param_value(epsilon_value)}",
                    fontsize=12,
                    pad=14,
                )
        ax.set_xlabel(metric_label)

    if all_values:
        x_left, x_right = zoom_limits(all_values)
        for ax in axes:
            if ax.get_visible():
                ax.set_xlim(x_left, x_right)

    assert_identical_panel_xlimits(axes)
    assert_identical_panel_ylabels(axes)

    if len(panels) > 1:
        fig.suptitle(f"{dataset_label} | {method_set_label} | {metric_label}", fontsize=14, y=SUPTITLE_Y)
        fig.tight_layout(rect=[0.22, 0.03, 0.98, TIGHT_LAYOUT_RECT[3]])
    else:
        fig.tight_layout(rect=[0.22, 0.03, 0.98, 0.98])

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

    values = grid.to_numpy(dtype=float)
    finite_values = values[~pd.isna(values)]
    if finite_values.size == 0:
        return
    vmin, vmax = metric_heatmap_bounds(metric_name, metric_spec, finite_values)

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


def runtime_stack_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required_columns = [column for column, _, _ in RUNTIME_STACK_PARTS]
    working = frame.copy()
    for column in required_columns:
        if column not in working.columns:
            working[column] = 0.0
        working[column] = pd.to_numeric(working[column], errors="coerce").fillna(0.0)
    working["runtime_total_time_s"] = working[required_columns].sum(axis=1)
    return working[working["runtime_total_time_s"] > 0].copy()

def assert_identical_panel_ylabels(axes: list[plt.Axes]) -> None:
    visible_axes = [ax for ax in axes if ax.get_visible()]
    if len(visible_axes) <= 1:
        return
    reference_ticks = visible_axes[0].get_yticks().tolist()
    reference_labels = [tick.get_text() for tick in visible_axes[0].get_yticklabels() if tick.get_text()]
    for ax in visible_axes[1:]:
        current_ticks = ax.get_yticks().tolist()
        assert current_ticks == reference_ticks, "Y-axis tick positions differ across epsilon panels"
        current_labels = [tick.get_text() for tick in ax.get_yticklabels() if tick.get_text()]
        if current_labels:
            assert current_labels == reference_labels, "Y-label order differs across epsilon panels"


def assert_identical_panel_xlimits(axes: list[plt.Axes]) -> None:
    visible_axes = [ax for ax in axes if ax.get_visible()]
    if len(visible_axes) <= 1:
        return
    ref_left, ref_right = visible_axes[0].get_xlim()
    for ax in visible_axes[1:]:
        left, right = ax.get_xlim()
        assert math.isclose(left, ref_left, rel_tol=0.0, abs_tol=1e-12)
        assert math.isclose(right, ref_right, rel_tol=0.0, abs_tol=1e-12)

def add_total_annotations(ax: plt.Axes, positions: list[int], totals: list[float]) -> None:
    total_max = max(totals)
    offset = total_max * 0.02 if total_max > 0 else 0.01
    for position, total in zip(positions, totals):
        ax.text(total + offset, position, f"{total:.4g}", va="center", ha="left", fontsize=9)

def show_y_labels(panel_index: int, panel_count: int) -> bool:
    if panel_count <= 1:
        return True
    if panel_count == 5:
        return panel_index in {0, 2}
    if panel_count <= 3:
        return panel_index == 0
    return panel_index % 3 == 0


def plot_runtime_stacked(
    dataset_name: str,
    dataset_label: str,
    frame: pd.DataFrame,
    method_set: dict[str, Any],
    method_labels: dict[str, str],
    param_specs: dict[str, dict[str, Any]],
) -> None:
    working = runtime_stack_frame(frame)
    if working.empty:
        return

    epsilon_values = facet_epsilon_values(working, param_specs)
    panels = epsilon_values if epsilon_values else [None]
    fig, axes = create_bar_axes(len(panels))
    excluded_params = {"epsilon"} if epsilon_values else set()
    label_resolver = lambda method, params: method_set_display_label(
        dataset_name,
        method_set,
        method,
        params,
        method_labels,
    )
    ordered_labels = global_variant_labels(
        working,
        method_labels,
        param_specs,
        exclude=excluded_params,
        base_label_resolver=label_resolver,
    )
    method_set_label = str(method_set.get("print_as") or method_set["name"])

    all_totals: list[float] = []
    panel_count = len(panels)
    for panel_index, (ax, epsilon_value) in enumerate(zip(axes, panels)):
        panel = panel_rows_for_epsilon(working, epsilon_value)
        if panel.empty:
            ax.set_visible(False)
            continue
        panel = panel.sort_values("runtime_total_time_s", ascending=True, kind="mergesort").reset_index(drop=True)
        panel_rows_by_label: dict[str, Any] = {}
        for row in panel.itertuples():
            params_dict = row.params_dict if isinstance(row.params_dict, dict) else {}
            label = variant_label(
                str(row.method),
                params_dict,
                method_labels,
                param_specs,
                exclude=excluded_params,
                base_label=label_resolver(str(row.method), params_dict),
            )
            if label not in panel_rows_by_label:
                panel_rows_by_label[label] = row

        positions = list(range(len(ordered_labels)))
        left = [0.0] * len(ordered_labels)
        for column, stage_label, color in RUNTIME_STACK_PARTS:
            values = []
            for label in ordered_labels:
                row = panel_rows_by_label.get(label)
                if row is None:
                    values.append(0.0)
                else:
                    values.append(float(getattr(row, column)))
            ax.barh(positions, values, left=left, color=color, alpha=0.92, label=stage_label)
            left = [current_left + value for current_left, value in zip(left, values)]
        present_positions = [index for index, label in enumerate(ordered_labels) if label in panel_rows_by_label]
        present_totals = [left[index] for index in present_positions]
        all_totals.extend(present_totals)
        ax.set_yticks(list(range(len(ordered_labels))))
        ax.set_yticklabels(ordered_labels, fontsize=9)
        if show_y_labels(panel_index, panel_count):
            ax.tick_params(axis="y", length=3)
        else:
            for tick_label in ax.get_yticklabels():
                tick_label.set_visible(False)
            ax.tick_params(axis="y", length=0)
        ax.invert_yaxis()
        if present_positions:
            add_total_annotations(ax, present_positions, present_totals)
        ax.grid(axis="x", alpha=0.25)
        if epsilon_value is None:
            ax.set_title("Pipeline runtime", fontsize=12, pad=14)
        else:
            epsilon_label = str(param_specs.get("epsilon", {}).get("print_as") or "epsilon")
            ax.set_title(f"{epsilon_label}={format_param_value(epsilon_value)}", fontsize=12, pad=14)
        ax.set_xlabel("Pipeline runtime (s)")

    if all_totals:
        for ax in axes:
            if ax.get_visible():
                style_x_axis(ax, use_engineering=True, max_ticks=4)

    assert_identical_panel_xlimits(axes)
    assert_identical_panel_ylabels(axes)

    legend_source_axis = next((ax for ax in axes if ax.get_visible()), axes[0])
    handles, labels = legend_source_axis.get_legend_handles_labels()
    dedup_handles: list[Any] = []
    dedup_labels: list[str] = []
    seen_labels: set[str] = set()
    for handle, label in zip(handles, labels):
        if label in seen_labels:
            continue
        seen_labels.add(label)
        dedup_handles.append(handle)
        dedup_labels.append(label)
    if len(panels) == 5:
        legend_axis_index = 1
    elif len(panels) <= 3:
        legend_axis_index = len(panels) - 1
    else:
        legend_axis_index = 2
    legend = axes[legend_axis_index].legend(
        dedup_handles,
        dedup_labels,
        title="Stage",
        bbox_to_anchor=(1.02, 0.98),
        loc="upper left",
        ncol=1,
        fontsize=9,
        frameon=True,
    )
    legend.get_title().set_fontsize(10)
    fig.suptitle(f"{dataset_label} | Runtime assessment", fontsize=14, y=SUPTITLE_Y)
    fig.text(0.5, SUBTITLE_Y, method_set_label, ha="center", va="center", fontsize=10)
    fig.tight_layout(rect=[0.03, 0.03, 0.90, TIGHT_LAYOUT_RECT[3]])

    output_path = OUTPUT_DIR / "stacked" / dataset_name / str(method_set["name"]) / "runtime_pipeline.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    manifest_columns = ["method", "params", "runtime_total_time_s"] + [column for column, _, _ in RUNTIME_STACK_PARTS]
    save_manifest(output_path.with_name(f"{output_path.stem}_manifest"), working[manifest_columns].copy())


def create_bar_axes(num_panels: int) -> tuple[plt.Figure, list[plt.Axes]]:
    fig, axes = create_panel_axes(
        num_panels,
        sharex=True,
        sharey=True,
        row_size=(8.0, 7.0),
        grid_size=(8.0, 6.0),
    )
    if num_panels == 1:
        fig.set_size_inches(fig.get_figwidth() * 1.45, fig.get_figheight(), forward=True)
    return fig, axes


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
        for method_set in method_sets:
            filtered = filter_method_set_rows(dataset_name, frame, method_set)
            if filtered.empty:
                continue
            plot_runtime_stacked(
                dataset_name,
                dataset_label,
                filtered,
                method_set,
                method_labels,
                param_specs,
            )
        for metric_spec in dataset_spec["scores"]:
            metric_name = str(metric_spec["name"])
            if metric_name not in frame.columns:
                continue
            if str(metric_spec.get("category")) == "runtime":
                continue
            metric_frame = apply_metric_exclusions(frame, metric_spec)
            if metric_frame.empty:
                continue
            for method_set in method_sets:
                filtered = filter_method_set_rows(dataset_name, metric_frame, method_set)
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
