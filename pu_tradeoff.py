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


def format_metric_bounds_summary(
    x_column: str,
    y_column: str,
    metric_bounds: dict[str, tuple[float, float]],
) -> str:
    lines: list[str] = []
    x_bounds = metric_bounds.get(x_column)
    if x_bounds is not None:
        lines.append(f"{x_column}: best={x_bounds[0]:g} | worst={x_bounds[1]:g}")
    y_bounds = metric_bounds.get(y_column)
    if y_bounds is not None:
        lines.append(f"{y_column}: best={y_bounds[0]:g} | worst={y_bounds[1]:g}")
    return "\n".join(lines)


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
    if not isinstance(param_name, str):
        return ordered
    spec = param_specs.get(param_name, {})
    if spec.get("print_order") == "high_to_low":
        ordered.reverse()
    return ordered


def endpoint_row_for_baseline_extension(
    ordered_rows: pd.DataFrame,
    shade_param: str | None,
    param_specs: dict[str, dict[str, object]],
) -> pd.Series:
    if len(ordered_rows) == 0:
        raise ValueError("ordered_rows must not be empty")
    return ordered_rows.iloc[-1]


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


def format_param_label(params_dict: dict[str, object]) -> str:
    if not params_dict:
        return ""
    parts = []
    for key, value in params_dict.items():
        pretty_key = _param_label_map.get(key, key)
        parts.append(f"{pretty_key}={value}")
    return ", ".join(parts)


def point_legend_label(method: str, params_dict: dict[str, object]) -> str:
    params_label = format_param_label(params_dict)
    if not params_label:
        return method
    return f"{method}({params_label})"


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
    ordered = sort_param_values(values, "epsilon", param_specs)
    return ordered


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


def series_legend_label(
    method: str,
    params_dict: dict[str, object],
    shade_param: str | None,
) -> str:
    fixed_params = dict(params_dict)
    if isinstance(shade_param, str) and shade_param in fixed_params:
        fixed_params.pop(shade_param)
    params_label = format_param_label(fixed_params)
    if isinstance(shade_param, str) and shade_param in params_dict:
        sweep_label = f"{_param_label_map.get(shade_param, shade_param)} sweep"
        return f"{method} ({params_label}; {sweep_label})" if params_label else f"{method} ({sweep_label})"
    if params_label:
        return f"{method}({params_label})"
    return method


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
    for row in rows:
        params_dict = row["plot_params_dict"]
        if not isinstance(params_dict, dict) or shade_param not in params_dict:
            continue
        label = f"{_param_label_map.get(shade_param, shade_param)}={params_dict[shade_param]}"
        if label in seen_values:
            continue
        seen_values.add(label)
        ax.annotate(
            label,
            (float(row["P_plot"]), float(row["U_plot"])),
            xytext=(float(row["P_plot"]) + x_offset, float(row["U_plot"]) + y_offset),
            textcoords="data",
            color=color,
            fontsize=8,
        )


def set_zoomed_limits(ax: plt.Axes, panel_df: pd.DataFrame, extra_y_values: list[float] | None = None) -> None:
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
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)


def zoomed_limits_for_df(panel_df: pd.DataFrame, extra_y_values: list[float] | None = None) -> tuple[tuple[float, float], tuple[float, float]]:
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


def draw_threshold_bridge(
    ax: plt.Axes,
    panel_df: pd.DataFrame,
    ordered_rows: pd.DataFrame,
    grouped_method_name: str,
    shade_param: str | None,
    base_color: tuple[float, float, float],
    param_specs: dict[str, dict[str, object]],
) -> None:
    if not isinstance(shade_param, str):
        return
    if len(ordered_rows) == 0:
        return
    base_candidates = panel_df[
        (panel_df["method"] == grouped_method_name)
        & (panel_df["plot_params_dict"].apply(lambda params: isinstance(params, dict) and len(params) == 0))
    ]
    if base_candidates.empty:
        return
    endpoint = ordered_rows.iloc[-1]
    base_row = base_candidates.iloc[0]
    ax.plot(
        [float(endpoint["P_plot"]), float(base_row["P_plot"])],
        [float(endpoint["U_plot"]), float(base_row["U_plot"])],
        color=base_color,
        linewidth=1.6,
        alpha=0.85,
        linestyle="--",
        label="_nolegend_",
    )
    ax.annotate(
        "full",
        (float(base_row["P_plot"]), float(base_row["U_plot"])),
        xytext=(6, 6),
        textcoords="offset points",
        color=base_color,
        fontsize=8,
    )


def create_tradeoff_axes(num_panels: int) -> tuple[plt.Figure, list[plt.Axes]]:
    if num_panels == 5:
        fig = plt.figure(figsize=(18, 12))
        grid = fig.add_gridspec(2, 6)
        first = fig.add_subplot(grid[0, 1:3])
        second = fig.add_subplot(grid[0, 3:5], sharex=first, sharey=first)
        third = fig.add_subplot(grid[1, 0:2], sharex=first, sharey=first)
        fourth = fig.add_subplot(grid[1, 2:4], sharex=first, sharey=first)
        fifth = fig.add_subplot(grid[1, 4:6], sharex=first, sharey=first)
        return fig, [first, second, third, fourth, fifth]

    if num_panels <= 3:
        fig, axes = plt.subplots(1, num_panels, figsize=(8 * num_panels, 7), squeeze=False, sharex=True, sharey=True)
        return fig, list(axes[0])

    ncols = 3
    nrows = math.ceil(num_panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * nrows), squeeze=False, sharex=True, sharey=True)
    flat_axes = list(axes.flat)
    for extra_ax in flat_axes[num_panels:]:
        extra_ax.set_visible(False)
    return fig, flat_axes[:num_panels]

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
    dataset_print_labels = read_dataset_print_labels()
    method_group_labels = read_method_group_labels()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for dataset_name in dataset_names:
        metric_bounds = dataset_metric_bounds.get(dataset_name, {})
        reference_df = read_dataset_reference_frame(dataset_name)
        method_names = discover_method_names(dataset_name)
        for method_name in method_names:
            method_group_label = method_group_labels.get(method_name, method_name)
            dataset_label = dataset_print_labels.get(dataset_name, dataset_name.replace("_", "-").upper())
            file_name = f"mds/{dataset_name}_logs_{method_name}.csv"
            file_path = Path(file_name)
            for x_name, x_column, y_name, y_column in targets:
                df = read_csv_file(file_path, x_column, y_column)
                if df is None:
                    continue
                print(f"Dataset: {dataset_name}, Method Group: {method_group_label}, X: {x_column}, Y: {y_column}")

                df["params_dict"] = df["params"].apply(parse_params)
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
                    baseline_df = panel_df[panel_df["method"] == "baseline"]
                    baseline_point: tuple[float, float] | None = None
                    if not baseline_df.empty:
                        baseline_row = baseline_df.iloc[0]
                        baseline_point = (float(baseline_row["P_plot"]), float(baseline_row["U_plot"]))

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
                        base_color = get_base_color(base_group, color_cache)
                        method_marker = get_group_marker(base_group, marker_cache)

                        ordered_rows = group_df.copy()
                        if isinstance(shade_param, str):
                            ordered_rows["__shade_sort__"] = ordered_rows["plot_params_dict"].apply(
                                lambda params: sort_param_value_for_shading(params.get(shade_param))
                            )
                            ordered_rows = ordered_rows.sort_values("__shade_sort__")
                            if param_specs.get(shade_param, {}).get("print_order") == "high_to_low":
                                ordered_rows = ordered_rows.iloc[::-1]

                        has_line = len(ordered_rows) >= 2
                        series_label = series_legend_label(grouped_method_name, first_row["plot_params_dict"], shade_param)
                        if has_line:
                            ax.plot(
                                ordered_rows["P_plot"],
                                ordered_rows["U_plot"],
                                color=base_color,
                                linewidth=1.8,
                                alpha=0.9,
                                label=series_label,
                            )

                        if has_line and baseline_point is not None and grouped_method_name != "baseline":
                            endpoint = endpoint_row_for_baseline_extension(ordered_rows, shade_param, param_specs)
                            ax.plot(
                                [float(endpoint["P_plot"]), baseline_point[0]],
                                [float(endpoint["U_plot"]), baseline_point[1]],
                                color=base_color,
                                linewidth=1.4,
                                alpha=0.8,
                                linestyle=":",
                                label="_nolegend_",
                            )

                        draw_threshold_bridge(
                            ax,
                            panel_df,
                            ordered_rows,
                            grouped_method_name,
                            shade_param,
                            base_color,
                            param_specs,
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
                    if epsilon_value is not None:
                        ax.set_title(f"ε={epsilon_value}", fontsize=12, pad=14)
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
                    x_limits, y_limits = zoomed_limits_for_df(combined_df, extra_y_values=extra_y_values)
                    for ax in axes:
                        ax.set_xlim(*x_limits)
                        ax.set_ylim(*y_limits)
                else:
                    set_zoomed_limits(axes[0], panel_frames[0], extra_y_values=extra_y_values)

                fig.suptitle(f"Privacy-Utility Tradeoff in {dataset_label}", fontsize=14, y=0.98)
                fig.text(0.5, 0.94, method_group_label, ha="center", va="center", fontsize=10)
                fig.tight_layout(rect=[0.03, 0.03, 0.9, 0.93])
                output_dir = OUTPUT_DIR / method_name / dataset_name / x_name
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{y_name}.png"
                fig.savefig(output_path, dpi=200)
                print(f"Saved plot to {output_path}")
                plt.close(fig)
