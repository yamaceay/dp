from typing import Optional
import math
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
import pandas as pd
from pathlib import Path
import ast
import yaml
import argparse
import re

from plot_layout import (
    LAYOUT,
    SUBTITLE_Y,
    SUPTITLE_Y,
    TIGHT_LAYOUT_RECT,
    create_panel_axes,
    order_is_descending,
)

X_AXIS_OPTIONS = {
    "p_exact": "P_exact",
    "p_more": "P_more",
    "p_full": "P_full",
}

Y_AXIS_OPTIONS = {
    "u_nominal": "U_nominal_acc",
    "u_ordinal": "U_ordinal_mae",
    "sd_nominal": "SD_nominal_acc",
    "sd_ordinal": "SD_ordinal_mae",
}

dataset_names = [
    "db_bio",
    "tab",
]
PARAMS_CONFIG_PATH = Path("configs/visualize/params.yaml")
DATASET_CONFIG_PATH = Path("configs/visualize/datasets.yaml")
METHODS_CONFIG_PATH = Path("configs/visualize/methods.yaml")
OUTPUT_DIR = Path("images/tradeoff")

_palette = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#000000",  # black
    "#F0E442",  # yellow
    "#C44E52",
    "#4C72B0",
    "#55A868",
    "#8172B3",
]
_marker_palette = [
    "o",
    "s",
    "^",
    "D",
    "P",
    "X",
    "v",
    "<",
    ">",
    "h",
    "*",
]
PRESIDIO_COLOR = "#8F2D56"
_param_label_map = {
    "epsilon": "ε",
    "rho": "ρ",
    "lambda": "λ",
}


def read_param_specs() -> dict[str, dict[str, object]]:
    with open(PARAMS_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    param_specs: dict[str, dict[str, object]] = {}
    for item in config.get("param_sets", []):
        if "name" in item:
            param_specs[item["name"]] = item
    return param_specs


def read_dataset_metric_bounds() -> dict[str, dict[str, tuple[float, float]]]:
    with open(DATASET_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    bounds_by_dataset: dict[str, dict[str, tuple[float, float]]] = {}
    for dataset_set in config.get("dataset_sets", []):
        dataset_scope = dataset_set.get("names")
        dataset_names: list[str] = []
        if isinstance(dataset_scope, list):
            for entry in dataset_scope:
                if isinstance(entry, str):
                    dataset_names.append(entry)
                    continue
                if isinstance(entry, dict):
                    dataset_name = entry.get("name")
                    if isinstance(dataset_name, str):
                        dataset_names.append(dataset_name)
        if not dataset_names:
            dataset_name = dataset_set.get("name")
            if isinstance(dataset_name, str):
                dataset_names = [dataset_name]
        score_bounds: dict[str, tuple[float, float]] = {}
        for score in dataset_set.get("scores", []):
            score_name = score.get("rename_as", score.get("print_as", score.get("name")))
            best_value = score.get("best")
            worst_value = score.get("worst")
            if not isinstance(score_name, str):
                continue
            if not isinstance(best_value, (int, float)) or not isinstance(worst_value, (int, float)):
                continue
            score_bounds[score_name] = (float(best_value), float(worst_value))
        for dataset_name in dataset_names:
            bounds_by_dataset[dataset_name] = dict(score_bounds)
    return bounds_by_dataset


def normalized_excluded_methods(raw_methods: object) -> set[str]:
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


def read_dataset_metric_exclusions() -> dict[str, dict[str, set[str]]]:
    with open(DATASET_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    exclusions_by_dataset: dict[str, dict[str, set[str]]] = {}
    for dataset_set in config.get("dataset_sets", []):
        dataset_scope = dataset_set.get("names")
        dataset_names: list[str] = []
        if isinstance(dataset_scope, list):
            for entry in dataset_scope:
                if isinstance(entry, str):
                    dataset_names.append(entry)
                    continue
                if isinstance(entry, dict):
                    dataset_name = entry.get("name")
                    if isinstance(dataset_name, str):
                        dataset_names.append(dataset_name)
        if not dataset_names:
            dataset_name = dataset_set.get("name")
            if isinstance(dataset_name, str):
                dataset_names = [dataset_name]
        score_exclusions: dict[str, set[str]] = {}
        for score in dataset_set.get("scores", []):
            if not isinstance(score, dict):
                continue
            excluded_methods = normalized_excluded_methods(score.get("exclude", []))
            if not excluded_methods:
                continue
            for metric_key in (score.get("name"), score.get("rename_as"), score.get("print_as")):
                if isinstance(metric_key, str) and metric_key.strip():
                    score_exclusions[metric_key] = set(excluded_methods)
        for dataset_name in dataset_names:
            exclusions_by_dataset[dataset_name] = dict(score_exclusions)
    return exclusions_by_dataset


def read_dataset_print_labels() -> dict[str, str]:
    with open(DATASET_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    labels: dict[str, str] = {}
    for dataset_set in config.get("dataset_sets", []):
        dataset_scope = dataset_set.get("names")
        if isinstance(dataset_scope, list):
            for entry in dataset_scope:
                if isinstance(entry, dict):
                    dataset_name = entry.get("name")
                    dataset_print_as = entry.get("print_as")
                    if isinstance(dataset_name, str) and isinstance(dataset_print_as, str) and dataset_print_as.strip():
                        labels[dataset_name] = dataset_print_as.strip()
                elif isinstance(entry, str):
                    labels.setdefault(entry, entry.replace("_", "-").upper())
            continue
        dataset_name = dataset_set.get("name")
        dataset_print_as = dataset_set.get("print_as")
        if isinstance(dataset_name, str):
            if isinstance(dataset_print_as, str) and dataset_print_as.strip():
                labels[dataset_name] = dataset_print_as.strip()
            else:
                labels.setdefault(dataset_name, dataset_name.replace("_", "-").upper())
    return labels


def read_method_group_labels() -> dict[str, str]:
    with open(METHODS_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    labels: dict[str, str] = {}
    for method_set in config.get("method_sets", []):
        name = method_set.get("name")
        print_as = method_set.get("print_as")
        if isinstance(name, str) and isinstance(print_as, str) and print_as.strip():
            labels[name] = print_as.strip()
    return labels


def read_method_display_specs() -> dict[str, list[dict[str, object]]]:
    with open(METHODS_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    specs: dict[str, list[dict[str, object]]] = {}
    for method_set in config.get("method_sets", []):
        group_name = method_set.get("name")
        methods = method_set.get("methods", [])
        if not isinstance(group_name, str) or not isinstance(methods, list):
            continue
        group_specs: list[dict[str, object]] = []
        for method_spec in methods:
            if isinstance(method_spec, dict):
                group_specs.append(method_spec)
        specs[group_name] = group_specs
    return specs


def configured_method_group_names() -> list[str]:
    with open(METHODS_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    names: list[str] = []
    for method_set in config.get("method_sets", []):
        name = method_set.get("name")
        if isinstance(name, str):
            names.append(name)
    return names


def metric_label_with_unit(metric_name: str) -> str:
    name, unit = metric_name_and_unit(metric_name)
    if unit is None:
        return name
    return f"{name} ({unit})"


def metric_direction_arrow(metric_name: str, metric_bounds: dict[str, tuple[float, float]]) -> str:
    bounds = metric_bounds.get(metric_name)
    if bounds is None:
        return ""
    best_value, worst_value = bounds
    assert best_value != worst_value, f"Best and worst values are the same for {metric_name}"
    if best_value > worst_value:
        return "↑"
    else:
        return "↓"


def metric_label_with_unit_and_direction(
    metric_name: str,
    metric_bounds: dict[str, tuple[float, float]],
) -> str:
    name, unit = metric_name_and_unit(metric_name)

    direction = metric_direction_arrow(metric_name, metric_bounds)
    if unit is None and not direction:
        return name
    if unit is None:
        return f"{name} ({direction})"
    if not direction:
        return f"{name} ({unit})"
    return f"{name} ({unit} {direction})"


def metric_name_and_unit(metric_name: str) -> tuple[str, str | None]:
    if metric_name in {"P_exact", "P_more", "P_full"}:
        return metric_name, "MRR"
    if metric_name in {"U_nominal_acc", "SD_nominal_acc", "U_nominal_raw", "SD_nominal_raw"}:
        return metric_name.replace("_raw", "").replace("_acc", ""), "ACC"
    if metric_name in {"U_ordinal_mae", "SD_ordinal_mae", "U_ordinal_raw", "SD_ordinal_raw"}:
        return metric_name.replace("_raw", "").replace("_mae", ""), "MAE"
    return metric_name, None


def y_axis_assessment_label(y_column: str, metric_bounds: dict[str, tuple[float, float]]) -> str:
    metric_label = metric_label_with_unit_and_direction(y_column, metric_bounds)
    if y_column.startswith("U_"):
        return f"Utility ({metric_label})"
    if y_column.startswith("SD_"):
        return f"Supervised divergence ({metric_label})"
    return metric_label


def y_metric_context(y_column: str) -> tuple[str, str] | None:
    if y_column in {"U_nominal_acc", "U_nominal_raw"}:
        return ("utility", "acc")
    if y_column in {"U_ordinal_mae", "U_ordinal_raw"}:
        return ("utility", "mae")
    if y_column in {"SD_nominal_acc", "SD_nominal_raw"}:
        return ("supervised_divergence", "acc")
    if y_column in {"SD_ordinal_mae", "SD_ordinal_raw"}:
        return ("supervised_divergence", "mae")
    return None


def _first_numeric(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.iloc[0])


def extract_dummy_strategy(df: pd.DataFrame, y_column: str) -> str | None:
    metric_ctx = y_metric_context(y_column)
    if metric_ctx is None:
        return None
    section_prefix = metric_ctx[0]
    strategy_col = f"{section_prefix}_utility_dummy_strategy"
    if strategy_col not in df.columns:
        return None
    strategies = df[strategy_col].dropna().unique()
    if len(strategies) == 0:
        return None
    return str(strategies[0])


def reference_y_values(reference_df: pd.DataFrame, y_column: str) -> list[float]:
    metric_ctx = y_metric_context(y_column)
    if metric_ctx is None:
        return []
    section_prefix, abs_metric_name = metric_ctx
    values: list[float] = []

    baseline_columns = [
        col for col in reference_df.columns
        if re.match(rf"^{section_prefix}_.+_baseline_{abs_metric_name}$", col)
    ]
    for baseline_col in baseline_columns:
        value = _first_numeric(reference_df[baseline_col])
        if value is not None:
            values.append(value)
            break

    dummy_columns = [
        col for col in reference_df.columns
        if re.match(rf"^{section_prefix}_.+_dummy_{abs_metric_name}$", col)
    ]
    for dummy_col in dummy_columns:
        value = _first_numeric(reference_df[dummy_col])
        if value is not None:
            values.append(value)
            break

    return values


def reference_method_point(
    reference_df: pd.DataFrame | None,
    method_name: str,
    x_column: str,
    y_column: str,
) -> tuple[float, float] | None:
    if reference_df is None:
        return None
    if "method" not in reference_df.columns or x_column not in reference_df.columns or y_column not in reference_df.columns:
        return None
    method_rows = reference_df[reference_df["method"] == method_name].copy()
    if method_rows.empty:
        return None
    method_rows["P_plot"] = pd.to_numeric(method_rows[x_column], errors="coerce")
    method_rows["U_plot"] = pd.to_numeric(method_rows[y_column], errors="coerce")
    method_rows = method_rows.dropna(subset=["P_plot", "U_plot"])
    if method_rows.empty:
        return None
    row = method_rows.iloc[0]
    return float(row["P_plot"]), float(row["U_plot"])


def raw_metric_column(metric_name: str) -> str | None:
    raw_columns = {
        "P_exact": "privacy_exact_mean_reciprocal_rank",
        "P_more": "privacy_more_mean_reciprocal_rank",
        "P_full": "privacy_full_mean_reciprocal_rank",
        "U_nominal_acc": "utility_utility_nominal_raw_acc",
        "U_ordinal_mae": "utility_utility_ordinal_raw_mae",
        "SD_nominal_acc": "supervised_divergence_utility_nominal_raw_acc",
        "SD_ordinal_mae": "supervised_divergence_utility_ordinal_raw_mae",
    }
    return raw_columns.get(metric_name)


def method_point_from_any_metric_schema(
    reference_df: pd.DataFrame | None,
    method_name: str,
    x_column: str,
    y_column: str,
) -> tuple[float, float] | None:
    point = reference_method_point(reference_df, method_name, x_column, y_column)
    if point is not None:
        return point
    raw_x_column = raw_metric_column(x_column)
    raw_y_column = raw_metric_column(y_column)
    if raw_x_column is None or raw_y_column is None:
        return None
    return reference_method_point(reference_df, method_name, raw_x_column, raw_y_column)


def transformed_method_point(
    dataset_name: str,
    method_name: str,
    x_column: str,
    y_column: str,
    fallback_frames: list[pd.DataFrame] | None = None,
) -> tuple[float, float] | None:
    if fallback_frames:
        for frame in fallback_frames:
            point = method_point_from_any_metric_schema(frame, method_name, x_column, y_column)
            if point is not None:
                return point
    for path in sorted(Path("mds").glob(f"{dataset_name}_logs_*.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        point = method_point_from_any_metric_schema(frame, method_name, x_column, y_column)
        if point is not None:
            return point
    return None


def add_central_tendency_line(
    ax: plt.Axes,
    reference_df: pd.DataFrame,
    y_column: str,
) -> None:
    metric_ctx = y_metric_context(y_column)
    if metric_ctx is None:
        return
    section_prefix, abs_metric_name = metric_ctx
    search_df = (
        reference_df[~reference_df["method"].isin(["baseline", "dummy"])]
        if "method" in reference_df.columns
        else reference_df
    )

    baseline_columns = [
        col for col in search_df.columns
        if re.match(rf"^{section_prefix}_.+_baseline_{abs_metric_name}$", col)
    ]
    baseline_value: float | None = None
    for baseline_col in baseline_columns:
        baseline_value = _first_numeric(search_df[baseline_col])
        if baseline_value is not None:
            break
    if baseline_value is not None:
        ax.axhline(
            baseline_value,
            color="#2ca02c",
            linestyle="--",
            linewidth=1.3,
            alpha=0.9,
            label=f"baseline clf ({baseline_value:.3f})",
        )

    dummy_columns = [
        col for col in search_df.columns
        if re.match(rf"^{section_prefix}_.+_dummy_{abs_metric_name}$", col)
    ]
    dummy_value: float | None = None
    for dummy_col in dummy_columns:
        dummy_value = _first_numeric(search_df[dummy_col])
        if dummy_value is not None:
            break
    if dummy_value is None:
        return
    strategy = extract_dummy_strategy(search_df, y_column)
    label = strategy if strategy else "dummy clf"
    ax.axhline(
        dummy_value,
        color="#d62728",
        linestyle="-.",
        linewidth=1.3,
        alpha=0.85,
        label=f"{label} ({dummy_value:.3f})",
    )

def read_csv_file(path: Path, x_column: str, y_column: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
        if x_column not in df.columns:
            print(f"Skipping file without {x_column}: {path}")
            return None
        if y_column not in df.columns:
            print(f"Skipping file without {y_column}: {path}")
            return None
        filtered = df.copy()
        filtered["P_plot"] = pd.to_numeric(filtered[x_column], errors="coerce")
        filtered["U_plot"] = pd.to_numeric(filtered[y_column], errors="coerce")
        filtered = filtered.dropna(subset=["P_plot", "U_plot"])
        if filtered.empty:
            print(f"Skipping file without valid {x_column}/{y_column} values: {path}")
            return None
        return filtered
    except Exception as e:
        print(f"Error reading CSV file {path}")
        return None


def read_dataset_reference_frame(dataset_name: str) -> Optional[pd.DataFrame]:
    path = Path(f"mds/{dataset_name}_logs.csv")
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _prompt_choice(prompt: str, options: list[tuple[str, str]]) -> tuple[str, str]:
    print(prompt)
    for idx, (_, label) in enumerate(options, start=1):
        print(f"({idx}) {label}")
    while True:
        choice = input("Enter number: ").strip()
        if not choice.isdigit():
            print("Please enter a valid number.")
            continue
        index = int(choice)
        if 1 <= index <= len(options):
            return options[index - 1]
        print("Choice out of range.")


def resolve_plot_targets(args: argparse.Namespace) -> list[tuple[str, str, str, str]]:
    if args.all:
        targets: list[tuple[str, str, str, str]] = []
        for x_name, x_column in X_AXIS_OPTIONS.items():
            for y_name, y_column in Y_AXIS_OPTIONS.items():
                targets.append((x_name, x_column, y_name, y_column))
        return targets

    if args.x_axis and args.y_axis:
        return [
            (
                args.x_axis,
                X_AXIS_OPTIONS[args.x_axis],
                args.y_axis,
                Y_AXIS_OPTIONS[args.y_axis],
            )
        ]

    x_options = [
        ("p_exact", "p_exact"),
        ("p_more", "p_more"),
        ("p_full", "p_full"),
    ]
    y_options = [
        ("u_nominal", "u_nominal"),
        ("u_ordinal", "u_ordinal"),
        ("sd_nominal", "sd_nominal"),
        ("sd_ordinal", "sd_ordinal"),
    ]
    x_name, _ = _prompt_choice("select x-axis:", x_options)
    y_name, _ = _prompt_choice("select y-axis:", y_options)
    return [(x_name, X_AXIS_OPTIONS[x_name], y_name, Y_AXIS_OPTIONS[y_name])]


def discover_method_names(dataset_name: str) -> list[str]:
    prefix = f"{dataset_name}_logs_"
    suffix = ".csv"
    methods: list[str] = []
    for path in sorted(Path("mds").glob(f"{dataset_name}_logs_*.csv")):
        name = path.name
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        method_name = name[len(prefix):-len(suffix)]
        methods.append(method_name)
    return methods


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


def param_keys_signature(params: dict[str, object]) -> str:
    if not params:
        return "none"
    return "_".join(sorted(params.keys()))


def param_values_signature(params: dict[str, object], keys: list[str]) -> str:
    return "|".join(f"{key}={params.get(key)}" for key in keys)


def get_base_color(
    color_key: str,
    color_cache: dict[str, tuple[float, float, float]],
) -> tuple[float, float, float]:
    if color_key not in color_cache:
        color_cache[color_key] = mcolors.to_rgb(_palette[len(color_cache) % len(_palette)])
    return color_cache[color_key]


def get_group_marker(
    group_key: str,
    marker_cache: dict[str, str],
) -> str:
    if group_key not in marker_cache:
        marker_cache[group_key] = _marker_palette[len(marker_cache) % len(_marker_palette)]
    return marker_cache[group_key]


def shade_color(color: tuple[float, float, float], level: float) -> tuple[float, float, float]:
    rgb = mcolors.to_rgb(color)
    hsv = mcolors.rgb_to_hsv(rgb)
    level = max(0.0, min(1.0, level))
    hsv[1] = max(0.65, hsv[1])
    hsv[2] = 0.98 - 0.38 * level
    return tuple(mcolors.hsv_to_rgb(hsv))


def sort_param_value_for_shading(value: object) -> tuple[int, object]:
    if isinstance(value, (int, float)):
        return (0, float(value))
    try:
        return (0, float(value))
    except Exception:
        return (1, str(value))


def sort_param_values(values: set[object], param_name: str | None, param_specs: dict[str, dict[str, object]]) -> list[object]:
    ordered = sorted(values, key=sort_param_value_for_shading)
    if order_is_descending(param_name, param_specs):
        ordered.reverse()
    return ordered


def select_shade_param(params: dict[str, object]) -> str | None:
    keys = sorted(params.keys())
    if not keys:
        return None
    if len(keys) >= 2 and "epsilon" in keys:
        return "epsilon"
    if len(keys) == 1:
        return keys[0]
    return keys[0]


def base_color_group_key(method: str, params_dict: dict[str, object], param_keys_sig: str) -> str:
    keys = sorted(params_dict.keys())
    if len(keys) >= 2 and "epsilon" in keys:
        non_epsilon_keys = [key for key in keys if key != "epsilon"]
        return f"{method}+{param_keys_sig}+{param_values_signature(params_dict, non_epsilon_keys)}"
    return f"{method}+{param_keys_sig}"


def configured_param_keys(method_spec: dict[str, object]) -> set[str]:
    keys: set[str] = set()
    for field_name in ("params", "params_one_run"):
        field_value = method_spec.get(field_name)
        if isinstance(field_value, list):
            for entry in field_value:
                if isinstance(entry, str):
                    keys.add(entry)
        elif isinstance(field_value, dict):
            for key in field_value.keys():
                if isinstance(key, str):
                    keys.add(key)
    return keys


def configured_method_display_name(
    method_group_name: str,
    method_name: str,
    params_dict: dict[str, object],
    shade_param: str | None,
    method_display_specs: dict[str, list[dict[str, object]]],
) -> str:
    group_specs = method_display_specs.get(method_group_name, [])
    candidates = [
        spec for spec in group_specs
        if isinstance(spec.get("method"), str) and spec.get("method") == method_name
    ]
    if not candidates:
        return method_name

    conservative_keys = {"rho", "lambda", "k"}
    active_keys = {key for key in params_dict.keys() if isinstance(key, str)}
    if isinstance(shade_param, str):
        active_keys.add(shade_param)

    def score(spec: dict[str, object]) -> tuple[int, int]:
        spec_keys = configured_param_keys(spec)
        matching_keys = len(spec_keys & active_keys)
        conservative_match = len((spec_keys & conservative_keys) & active_keys)
        return (conservative_match, matching_keys)

    best_spec = max(candidates, key=score)
    print_as = best_spec.get("print_as")
    if isinstance(print_as, str) and print_as.strip():
        return print_as.strip()
    return method_name


def facet_values_by_epsilon(
    df: pd.DataFrame,
    param_specs: dict[str, dict[str, object]],
) -> list[object]:
    values = {
        params["epsilon"]
        for params in df["params_dict"]
        if isinstance(params, dict) and "epsilon" in params and len(params) >= 2
    }
    if not values:
        return []
    return sort_param_values(values, "epsilon", param_specs)


def panel_df_for_epsilon(df: pd.DataFrame, epsilon_value: object | None) -> pd.DataFrame:
    if epsilon_value is None:
        panel = df.copy()
        panel["plot_params_dict"] = panel["params_dict"]
        return panel
    mask = df["params_dict"].apply(
        lambda params: not isinstance(params, dict)
        or "epsilon" not in params
        or params.get("epsilon") == epsilon_value
    )
    panel = df[mask].copy()
    panel["plot_params_dict"] = panel["params_dict"].apply(
        lambda params: {k: v for k, v in params.items() if k != "epsilon"} if isinstance(params, dict) else {}
    )
    return panel


def format_param_label(params_dict: dict[str, object]) -> str:
    if not params_dict:
        return ""
    parts = []
    for key, value in params_dict.items():
        pretty_key = _param_label_map.get(key, key)
        parts.append(f"{pretty_key}={value}")
    return ", ".join(parts)


def series_legend_label(
    method_display_name: str,
    params_dict: dict[str, object],
    shade_param: str | None,
) -> str:
    fixed_params = dict(params_dict)
    if isinstance(shade_param, str) and shade_param in fixed_params:
        fixed_params.pop(shade_param)
    params_label = format_param_label(fixed_params)
    if isinstance(shade_param, str) and shade_param in params_dict:
        sweep_label = f"{_param_label_map.get(shade_param, shade_param)} sweep"
        return f"{method_display_name} ({params_label}; {sweep_label})" if params_label else f"{method_display_name}"
    if params_label:
        return f"{method_display_name}({params_label})"
    return method_display_name


def annotate_series_endpoints(
    ax: plt.Axes,
    ordered_rows: pd.DataFrame,
    shade_param: str | None,
    color: tuple[float, float, float],
) -> None:
    if not isinstance(shade_param, str):
        return
    rows = [ordered_rows.iloc[0]]
    if len(ordered_rows) > 1:
        rows.append(ordered_rows.iloc[-1])
    seen_values: set[str] = set()
    x_values = [float(value) for value in ordered_rows["P_plot"]]
    y_values = [float(value) for value in ordered_rows["U_plot"]]
    x_span = max(x_values) - min(x_values)
    y_span = max(y_values) - min(y_values)
    x_offset = x_span * 0.02 if x_span > 0 else 0.01
    y_offset = y_span * 0.02 if y_span > 0 else 0.01
    for row_index, row in enumerate(rows):
        params_dict = row["plot_params_dict"]
        if not isinstance(params_dict, dict) or shade_param not in params_dict:
            continue
        label = f"{_param_label_map.get(shade_param, shade_param)}={params_dict[shade_param]}"
        if label in seen_values:
            continue
        seen_values.add(label)
        label_x = float(row["P_plot"]) + x_offset
        label_y = float(row["U_plot"]) + y_offset
        if row_index == 0 and shade_param in {"rho", "lambda", "k"}:
            label_x = float(row["P_plot"]) - (x_offset * 1.8 if x_offset > 0 else 0.02)
            label_y = float(row["U_plot"]) - (y_offset * 1.2 if y_offset > 0 else 0.015)
        ax.annotate(
            label,
            (float(row["P_plot"]), float(row["U_plot"])),
            xytext=(label_x, label_y),
            textcoords="data",
            color=color,
            fontsize=8,
        )


def zoomed_limits_for_df(
    panel_df: pd.DataFrame,
    extra_y_values: list[float] | None = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    x_values = [float(value) for value in panel_df["P_plot"]]
    y_values = [float(value) for value in panel_df["U_plot"]]
    if extra_y_values:
        y_values.extend(float(value) for value in extra_y_values)
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    x_span = x_max - x_min
    y_span = y_max - y_min
    x_padding = x_span * 0.08 if x_span > 0 else max(abs(x_max) * 0.05, 0.01)
    y_padding = y_span * 0.08 if y_span > 0 else max(abs(y_max) * 0.05, 0.01)
    return (x_min - x_padding, x_max + x_padding), (y_min - y_padding, y_max + y_padding)


def set_zoomed_limits(
    ax: plt.Axes,
    panel_df: pd.DataFrame,
    extra_y_values: list[float] | None = None,
) -> None:
    x_limits, y_limits = zoomed_limits_for_df(panel_df, extra_y_values=extra_y_values)
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)


def create_tradeoff_axes(num_panels: int) -> tuple[plt.Figure, list[plt.Axes]]:
    fig, axes = create_panel_axes(
        num_panels,
        sharex=True,
        sharey=True,
        row_size=(8.0, 7.0),
        grid_size=(8.0, 6.0),
    )
    return fig, axes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate privacy-utility tradeoff plots")
    parser.add_argument("--all", action="store_true", help="Generate all x/y tradeoff combinations")
    parser.add_argument("--x-axis", choices=sorted(X_AXIS_OPTIONS.keys()), help="x-axis metric")
    parser.add_argument("--y-axis", choices=sorted(Y_AXIS_OPTIONS.keys()), help="y-axis metric")
    args = parser.parse_args()

    if (args.x_axis and not args.y_axis) or (args.y_axis and not args.x_axis):
        raise ValueError("Both --x-axis and --y-axis must be provided together.")
    if args.all and (args.x_axis or args.y_axis):
        raise ValueError("Use either --all or --x-axis/--y-axis, not both.")

    targets = resolve_plot_targets(args)

    param_specs = read_param_specs()
    dataset_metric_bounds = read_dataset_metric_bounds()
    dataset_metric_exclusions = read_dataset_metric_exclusions()
    dataset_print_labels = read_dataset_print_labels()
    method_group_labels = read_method_group_labels()
    method_display_specs = read_method_display_specs()
    method_group_names = configured_method_group_names()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for dataset_name in dataset_names:
        metric_bounds = dataset_metric_bounds.get(dataset_name, {})
        reference_df = read_dataset_reference_frame(dataset_name)
        method_names = [name for name in discover_method_names(dataset_name) if name in method_group_names]
        for method_name in method_names:
            method_group_label = method_group_labels.get(method_name, method_name)
            dataset_label = dataset_print_labels.get(dataset_name, dataset_name.replace("_", "-").upper())
            file_name = f"mds/{dataset_name}_logs_{method_name}.csv"
            file_path = Path(file_name)
            for x_name, x_column, y_name, y_column in targets:
                raw_df = read_csv_file(file_path, x_column, y_column)
                if raw_df is None:
                    continue
                metric_exclusions = dataset_metric_exclusions.get(dataset_name, {})
                excluded_methods = set(metric_exclusions.get(x_column, set())) | set(metric_exclusions.get(y_column, set()))
                if excluded_methods:
                    exclusion_mask = raw_df["method"].astype(str).apply(
                        lambda method: not method_is_excluded(method, excluded_methods)
                    )
                    raw_df = raw_df[exclusion_mask].copy()
                    if raw_df.empty:
                        continue
                print(f"Dataset: {dataset_name}, Method Group: {method_group_label}, X: {x_column}, Y: {y_column}")

                df = raw_df.copy()
                df["params_dict"] = df["params"].apply(parse_params)
                df = df[df["method"] != "baseline"].copy()
                if df.empty:
                    continue
                epsilon_panels = facet_values_by_epsilon(df, param_specs)
                panel_values: list[object | None] = epsilon_panels if epsilon_panels else [None]
                fig, axes = create_tradeoff_axes(len(panel_values))
                color_cache: dict[str, tuple[float, float, float]] = {}
                marker_cache: dict[str, str] = {}
                panel_frames: list[pd.DataFrame] = [panel_df_for_epsilon(df, epsilon_value) for epsilon_value in panel_values]
                reference_source_df = reference_df if reference_df is not None else df
                extra_y_values = reference_y_values(reference_source_df, y_column)

                for panel_index, (ax, epsilon_value, panel_df) in enumerate(zip(axes, panel_values, panel_frames)):
                    panel_df["param_keys_signature"] = panel_df["plot_params_dict"].apply(param_keys_signature)
                    panel_df["method_param_key"] = panel_df.apply(
                        lambda row: f"{row['method']}+{row['param_keys_signature']}",
                        axis=1,
                    )
                    panel_df["shade_param"] = panel_df["plot_params_dict"].apply(select_shade_param)
                    panel_df["base_color_group"] = panel_df.apply(
                        lambda row: base_color_group_key(row["method"], row["plot_params_dict"], row["param_keys_signature"]),
                        axis=1,
                    )

                    shade_level_map: dict[str, dict[object, float]] = {}
                    for base_group, group_df in panel_df.groupby("base_color_group", sort=False):
                        shade_param = group_df["shade_param"].iloc[0]
                        if not isinstance(shade_param, str):
                            shade_level_map[base_group] = {}
                            continue
                        shade_values = []
                        for params_dict in group_df["plot_params_dict"]:
                            if shade_param in params_dict:
                                shade_values.append(params_dict[shade_param])
                        unique_values = sort_param_values(set(shade_values), shade_param, param_specs)
                        if not unique_values:
                            shade_level_map[base_group] = {}
                        elif len(unique_values) == 1:
                            shade_level_map[base_group] = {unique_values[0]: 0.45}
                        else:
                            shade_level_map[base_group] = {
                                value: idx / (len(unique_values) - 1)
                                for idx, value in enumerate(unique_values)
                            }

                    for base_group, group_df in panel_df.groupby("base_color_group", sort=False):
                        first_row = group_df.iloc[0]
                        grouped_method_name = str(first_row["method"])
                        shade_param = first_row["shade_param"]
                        method_display_name = configured_method_display_name(
                            method_name,
                            grouped_method_name,
                            first_row["plot_params_dict"],
                            shade_param,
                            method_display_specs,
                        )
                        base_color = get_base_color(base_group, color_cache)
                        method_marker = get_group_marker(base_group, marker_cache)

                        ordered_rows = group_df.copy()
                        if isinstance(shade_param, str):
                            ordered_rows["__shade_sort__"] = ordered_rows["plot_params_dict"].apply(
                                lambda params: sort_param_value_for_shading(params.get(shade_param))
                            )
                            ordered_rows = ordered_rows.sort_values("__shade_sort__")
                            if order_is_descending(shade_param, param_specs):
                                ordered_rows = ordered_rows.iloc[::-1]

                        has_line = len(ordered_rows) >= 2
                        series_label = series_legend_label(
                            method_display_name,
                            first_row["plot_params_dict"],
                            shade_param,
                        )
                        if has_line:
                            ax.plot(
                                ordered_rows["P_plot"],
                                ordered_rows["U_plot"],
                                color=base_color,
                                linewidth=1.8,
                                alpha=0.9,
                                label=series_label,
                            )

                        first_point = True
                        for _, row in ordered_rows.iterrows():
                            params_dict = row["plot_params_dict"]
                            shade_value = None
                            if isinstance(shade_param, str):
                                shade_value = params_dict.get(shade_param)
                            shade_level = shade_level_map.get(base_group, {}).get(shade_value, 0.45)
                            point_color = shade_color(base_color, shade_level)

                            ax.scatter(
                                row["P_plot"],
                                row["U_plot"],
                                marker=method_marker,
                                color=point_color,
                                edgecolor=base_color,
                                linewidth=0.8,
                                alpha=0.98,
                                s=90,
                                label=series_label if first_point and not has_line else "_nolegend_",
                            )
                            first_point = False

                        annotate_series_endpoints(ax, ordered_rows, shade_param, base_color)

                    add_central_tendency_line(ax, reference_source_df, y_column)
                    ax.set_xlabel(f"Privacy ({x_column})")
                    ax.set_ylabel(y_axis_assessment_label(y_column, metric_bounds))
                    if len(panel_values) > 1:
                        if epsilon_value is not None:
                            ax.set_title(f"ε={epsilon_value}", fontsize=12, pad=14)
                    else:
                        if epsilon_value is None:
                            ax.set_title(
                                f"Privacy-Utility Tradeoff in {dataset_label} | {method_group_label}",
                                fontsize=12,
                                pad=14,
                            )
                        else:
                            ax.set_title(
                                f"Privacy-Utility Tradeoff in {dataset_label} | "
                                f"{method_group_label} | ε={epsilon_value}",
                                fontsize=12,
                                pad=14,
                            )
                    ax.grid(alpha=0.25)

                    handles, labels = ax.get_legend_handles_labels()
                    dedup_handles = []
                    dedup_labels = []
                    seen_labels = set()
                    for handle, label in zip(handles, labels):
                        if label == "_nolegend_" or label in seen_labels:
                            continue
                        seen_labels.add(label)
                        dedup_handles.append(handle)
                        dedup_labels.append(label)
                    should_show_legend = True
                    if len(panel_values) == 5:
                        should_show_legend = panel_index == 1

                    if dedup_labels and should_show_legend:
                        legend = ax.legend(
                            dedup_handles,
                            dedup_labels,
                            title="Series",
                            bbox_to_anchor=(1.02, 1),
                            loc="upper left",
                            fontsize=9,
                            ncol=1,
                        )
                        legend.get_title().set_fontsize(10)

                if len(panel_values) > 1:
                    combined_df = pd.concat(panel_frames, ignore_index=True)
                    x_limits, y_limits = zoomed_limits_for_df(
                        combined_df,
                        extra_y_values=extra_y_values,
                    )
                    for ax in axes:
                        ax.set_xlim(*x_limits)
                        ax.set_ylim(*y_limits)
                else:
                    set_zoomed_limits(
                        axes[0],
                        panel_frames[0],
                        extra_y_values=extra_y_values,
                    )

                if len(panel_values) > 1:
                    fig.suptitle(f"Privacy-Utility Tradeoff in {dataset_label}", fontsize=14, y=SUPTITLE_Y)
                    fig.text(0.5, SUBTITLE_Y, method_group_label, ha="center", va="center", fontsize=10)
                    fig.tight_layout(rect=TIGHT_LAYOUT_RECT)
                else:
                    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.98])
                output_dir = OUTPUT_DIR / method_name / dataset_name / x_name
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{y_name}.png"
                fig.savefig(output_path, dpi=200)
                print(f"Saved plot to {output_path}")
                plt.close(fig)
