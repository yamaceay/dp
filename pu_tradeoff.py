from typing import Optional
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
import pandas as pd
from pathlib import Path
import ast
import yaml

privacy_columns = ["P_exact"]
utility_columns = ["U_nominal"]
method_columns = ["method", "params"]

dataset_names = [
    "db_bio",
    "tab",
    "reddit",
]

method_names = [
    "simple_maskers",
    "simple_maskers_with_thresholds",
    "record_level_rewriters",
    "token_level_rewriters",
    "token_level_rewriters_with_threshold_k",
    "token_level_rewriters_with_threshold_rho",
    "token_level_rewriters_with_threshold_lambda",
]
PARAMS_CONFIG_PATH = Path("configs/visualize/params.yaml")
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

def read_csv_file(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path, usecols=method_columns + privacy_columns + utility_columns)
        return df
    except Exception as e:
        print(f"Error reading CSV file {path}")
        return None


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
    param_specs = read_param_specs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for dataset_name in dataset_names:
        for method_name in method_names:
            file_name = f"mds/{dataset_name}_logs_{method_name}.csv"
            file_path = Path(file_name)
            df = read_csv_file(file_path)
            if df is None:
                continue
            print(f"Dataset: {dataset_name}, Method: {method_name}")
            
            fig, ax = plt.subplots(figsize=(12, 6))
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
                baseline_point = (float(baseline_row["P_exact"]), float(baseline_row["U_nominal"]))

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
                method_name = str(first_row["method"])
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
                        ordered_rows["P_exact"],
                        ordered_rows["U_nominal"],
                        color=base_color,
                        linewidth=1.8,
                        alpha=0.9,
                        label="_nolegend_",
                    )

                if has_line and baseline_point is not None and method_name != "baseline":
                    endpoint = endpoint_row_for_baseline_extension(ordered_rows, shade_param, param_specs)
                    ax.plot(
                        [float(endpoint["P_exact"]), baseline_point[0]],
                        [float(endpoint["U_nominal"]), baseline_point[1]],
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
                        row["P_exact"],
                        row["U_nominal"],
                        color=point_color,
                        edgecolor=base_color,
                        linewidth=0.8,
                        alpha=0.98,
                        s=90,
                        label=point_legend_label(method_name, params_dict),
                    )

            ax.set_xlabel("Privacy (P_exact)")
            ax.set_ylabel("Utility (U_nominal)")
            ax.set_title(f"Privacy vs Utility for {dataset_name} - {method_name}")
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
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                fontsize=9,
                ncol=1 if len(dedup_labels) <= 5 else 2
            )
            legend.get_title().set_fontsize(10)
            
            fig.tight_layout()
            output_path = OUTPUT_DIR / f"pu_tradeoff_{dataset_name}_{method_name}.png"
            fig.savefig(output_path, dpi=200, bbox_inches="tight")
            print(f"Saved plot to {output_path}")
            plt.close(fig)
