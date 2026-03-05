from typing import Optional
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
    "u_nominal": "U_nominal",
    "u_ordinal": "U_ordinal",
    "sd_nominal": "SD_nominal",
    "sd_ordinal": "SD_ordinal",
}

dataset_names = [
    "db_bio",
    "tab",
]
PARAMS_CONFIG_PATH = Path("configs/visualize/params.yaml")
DATASET_CONFIG_PATH = Path("configs/visualize/datasets.yaml")
METHODS_CONFIG_PATH = Path("configs/visualize/methods.yaml")
OUTPUT_DIR = Path("images")

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


def metric_name_and_unit(metric_name: str) -> tuple[str, str | None]:
    if metric_name in {"P_exact", "P_more", "P_full"}:
        return metric_name, "MRR"
    if metric_name in {"U_nominal", "SD_nominal"}:
        return metric_name, "ACC"
    if metric_name in {"U_ordinal", "SD_ordinal"}:
        return metric_name, "MAE"
    return metric_name, None


def y_axis_assessment_label(y_column: str) -> str:
    if y_column.startswith("U_"):
        return f"Utility ({y_column})"
    if y_column.startswith("SD_"):
        return f"Supervised divergence ({y_column})"
    return y_column


def y_metric_context(y_column: str) -> tuple[str, str, str, str] | None:
    if y_column == "U_nominal":
        return ("utility", "acc", "r", "a")
    if y_column == "U_ordinal":
        return ("utility", "mae", "r", "m")
    if y_column == "SD_nominal":
        return ("supervised_divergence", "acc", "r", "a")
    if y_column == "SD_ordinal":
        return ("supervised_divergence", "mae", "r", "m")
    return None


def _first_numeric(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.iloc[0])


def _first_count(df: pd.DataFrame, section_prefix: str, feature: str) -> float:
    for count_column in (f"{section_prefix}__{feature}_count", f"{section_prefix}_{feature}_count"):
        if count_column not in df.columns:
            continue
        value = _first_numeric(df[count_column])
        if value is not None and value > 0:
            return value
    return 1.0


def utility_anchor_values(
    df: pd.DataFrame,
    section_prefix: str,
    abs_metric_name: str,
) -> tuple[float, float] | None:
    baseline_pattern = re.compile(rf"^{section_prefix}_(.+)_baseline_{abs_metric_name}$")
    weighted_baseline = 0.0
    weighted_dummy = 0.0
    weight_total = 0.0

    for column in df.columns:
        match = baseline_pattern.match(column)
        if not match:
            continue
        feature = match.group(1)
        dummy_column = f"{section_prefix}_{feature}_dummy_{abs_metric_name}"
        if dummy_column not in df.columns:
            continue
        baseline_value = _first_numeric(df[column])
        dummy_value = _first_numeric(df[dummy_column])
        if baseline_value is None or dummy_value is None:
            continue
        weight = _first_count(df, section_prefix, feature)
        weighted_baseline += baseline_value * weight
        weighted_dummy += dummy_value * weight
        weight_total += weight

    if weight_total <= 0:
        return None
    return (weighted_baseline / weight_total, weighted_dummy / weight_total)


def build_absolute_recovery_block(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    metric_bounds: dict[str, tuple[float, float]],
) -> str:
    rows: list[tuple[str, float, float]] = []
    baseline_rows = df[df["method"] == "baseline"] if "method" in df.columns else pd.DataFrame()
    if not baseline_rows.empty and x_column in baseline_rows.columns:
        baseline_privacy = _first_numeric(baseline_rows[x_column])
        if baseline_privacy is not None:
            rows.append((x_column, baseline_privacy, 0.0))

    metric_ctx = y_metric_context(y_column)
    formula_lines: list[str] = []
    if metric_ctx is not None:
        section_prefix, abs_metric_name, rel_symbol, abs_symbol = metric_ctx
        utility_anchors = utility_anchor_values(df, section_prefix, abs_metric_name)
        if utility_anchors is not None:
            rows.append((y_column, utility_anchors[0], utility_anchors[1]))
            formula_lines.append("to obtain the absolute scores:")
            formula_lines.append(f"{abs_symbol}(r) = r * (b - d) + d")

    if not rows and not formula_lines:
        return ""

    metric_name_width = len("Metric")
    for metric_name, _, _ in rows:
        raw_name, _ = metric_name_and_unit(metric_name)
        metric_name_width = max(metric_name_width, len(raw_name))
    metric_width = len("Metric")

    formatted_rows: list[tuple[str, float, float]] = []
    for metric_name, baseline_value, dummy_value in rows:
        raw_name, unit = metric_name_and_unit(metric_name)
        if unit is None:
            formatted_metric = raw_name
        else:
            formatted_metric = f"{raw_name.ljust(metric_name_width)} ({unit})"
        metric_width = max(metric_width, len(formatted_metric))
        formatted_rows.append((formatted_metric, baseline_value, dummy_value))

    b_header = "b (baseline)"
    d_header = "d (dummy)"
    b_width = max(len(b_header), 7)
    d_width = max(len(d_header), 7)

    lines: list[str] = []
    lines.append(f"{'Metric'.ljust(metric_width)} | {b_header.ljust(b_width)} | {d_header.ljust(d_width)}")
    lines.append(f"{'-' * metric_width}-+-{'-' * b_width}-+-{'-' * d_width}")
    for metric_name, baseline_value, dummy_value in formatted_rows:
        lines.append(
            f"{metric_name.ljust(metric_width)} | {baseline_value:>{b_width}.3f} | {dummy_value:>{d_width}.3f}"
        )
    if formula_lines:
        lines.append("")
        lines.extend(formula_lines)
    return "\n".join(lines)


def add_metric_bounds_to_plot(
    ax: plt.Axes,
    x_column: str,
    y_column: str,
    metric_bounds: dict[str, tuple[float, float]],
) -> str:
    labels: list[str] = []
    x_bounds = metric_bounds.get(x_column)
    if x_bounds is not None:
        x_best, x_worst = x_bounds
        ax.axvline(x_best, color="0.55", linestyle="--", linewidth=1.1, alpha=0.8, label="_nolegend_")
        ax.axvline(x_worst, color="0.55", linestyle=":", linewidth=1.1, alpha=0.8, label="_nolegend_")
        labels.append(f"{x_column}[best={x_best:g}, worst={x_worst:g}]")

    y_bounds = metric_bounds.get(y_column)
    if y_bounds is not None:
        y_best, y_worst = y_bounds
        ax.axhline(y_best, color="0.55", linestyle="--", linewidth=1.1, alpha=0.8, label="_nolegend_")
        ax.axhline(y_worst, color="0.55", linestyle=":", linewidth=1.1, alpha=0.8, label="_nolegend_")
        labels.append(f"{y_column}[best={y_best:g}, worst={y_worst:g}]")

    return " | ".join(labels)


def normalize_ordinal_plot_scores(
    df: pd.DataFrame,
    y_column: str,
    metric_bounds: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    if y_column not in {"U_ordinal", "SD_ordinal"}:
        return df
    bounds = metric_bounds.get(y_column)
    if bounds is None:
        return df
    best_value, worst_value = bounds
    if best_value < worst_value:
        df = df.copy()
        df["U_plot"] = 1.0 - df["U_plot"]
    return df

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
    if not isinstance(shade_param, str):
        return ordered_rows.iloc[-1]
    if shade_param == "epsilon":
        return ordered_rows.iloc[-1]
    spec = param_specs.get(shade_param, {})
    if spec.get("print_order") == "high_to_low":
        return ordered_rows.iloc[0]
    return ordered_rows.iloc[0]


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
                df = normalize_ordinal_plot_scores(df, y_column, metric_bounds)
                print(f"Dataset: {dataset_name}, Method Group: {method_group_label}, X: {x_column}, Y: {y_column}")

                fig, ax = plt.subplots(figsize=(12, 8))
                ax.set_box_aspect(1)
                color_cache: dict[str, tuple[float, float, float]] = {}

                df["params_dict"] = df["params"].apply(parse_params)
                df["param_keys_signature"] = df["params_dict"].apply(param_keys_signature)
                df["method_param_key"] = df.apply(
                    lambda row: f"{row['method']}+{row['param_keys_signature']}",
                    axis=1,
                )
                df["shade_param"] = df["params_dict"].apply(select_shade_param)
                df["base_color_group"] = df.apply(
                    lambda row: base_color_group_key(row["method"], row["params_dict"], row["param_keys_signature"]),
                    axis=1,
                )
                baseline_df = df[df["method"] == "baseline"]
                baseline_point: tuple[float, float] | None = None
                if not baseline_df.empty:
                    baseline_row = baseline_df.iloc[0]
                    baseline_point = (float(baseline_row["P_plot"]), float(baseline_row["U_plot"]))

                shade_level_map: dict[str, dict[object, float]] = {}
                for base_group, group_df in df.groupby("base_color_group", sort=False):
                    shade_param = group_df["shade_param"].iloc[0]
                    if not isinstance(shade_param, str):
                        shade_level_map[base_group] = {}
                        continue
                    shade_values = []
                    for params_dict in group_df["params_dict"]:
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

                for base_group, group_df in df.groupby("base_color_group", sort=False):
                    first_row = group_df.iloc[0]
                    grouped_method_name = str(first_row["method"])
                    shade_param = first_row["shade_param"]
                    base_color = get_base_color(base_group, color_cache)

                    ordered_rows = group_df.copy()
                    if isinstance(shade_param, str):
                        ordered_rows["__shade_sort__"] = ordered_rows["params_dict"].apply(
                            lambda params: sort_param_value_for_shading(params.get(shade_param))
                        )
                        ordered_rows = ordered_rows.sort_values("__shade_sort__")
                        if param_specs.get(shade_param, {}).get("print_order") == "high_to_low":
                            ordered_rows = ordered_rows.iloc[::-1]

                    has_line = len(ordered_rows) >= 2
                    if has_line:
                        ax.plot(
                            ordered_rows["P_plot"],
                            ordered_rows["U_plot"],
                            color=base_color,
                            linewidth=1.8,
                            alpha=0.9,
                            label="_nolegend_",
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

                    for _, row in ordered_rows.iterrows():
                        params_dict = row["params_dict"]
                        shade_value = None
                        if isinstance(shade_param, str):
                            shade_value = params_dict.get(shade_param)
                        shade_level = shade_level_map.get(base_group, {}).get(shade_value, 0.45)
                        point_color = shade_color(base_color, shade_level)

                        ax.scatter(
                                row["P_plot"],
                            row["U_plot"],
                            color=point_color,
                            edgecolor=base_color,
                            linewidth=0.8,
                            alpha=0.98,
                            s=90,
                            label=point_legend_label(grouped_method_name, params_dict),
                        )

                info_block = build_absolute_recovery_block(df, x_column, y_column, metric_bounds)
                ax.set_xlabel(f"Privacy ({x_column})")
                ax.set_ylabel(y_axis_assessment_label(y_column))
                ax.set_title(f"Privacy-Utility Tradeoff in {dataset_label}", fontsize=14, pad=24)
                ax.text(
                    0.5,
                    1.01,
                    method_group_label,
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )
                ax.grid()

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
                legend = ax.legend(
                    dedup_handles,
                    dedup_labels,
                    title="Method Variants",
                    bbox_to_anchor=(1.02, 1),
                    loc="upper left",
                    fontsize=9,
                    ncol=1 if len(dedup_labels) <= 5 else 2,
                )
                legend.get_title().set_fontsize(10)

                if info_block:
                    ax.text(
                        1.05,
                        0.0,
                        info_block,
                        transform=ax.transAxes,
                        ha="left",
                        va="bottom",
                        fontsize=10,
                        fontfamily="monospace",
                        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
                    )

                fig.tight_layout(rect=[0.06, 0.03, 0.73, 0.98])
                output_path = OUTPUT_DIR / f"pu_tradeoff_{dataset_name}_{method_name}_{x_name}_{y_name}.png"
                fig.savefig(output_path, dpi=200)
                print(f"Saved plot to {output_path}")
                plt.close(fig)
