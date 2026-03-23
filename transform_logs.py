import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


INPUT_LOGS_PATH = Path("merged_logs.json")
OUTPUT_DIR = Path("mds")
REDDIT_UTILITY_OUTPUT_PATH = OUTPUT_DIR / "reddit_utility.json"
METHOD_SETS_CONFIG_PATH = Path("configs/visualize/methods.yaml")
DATASET_SETS_CONFIG_PATH = Path("configs/visualize/datasets.yaml")
PARAM_SETS_CONFIG_PATH = Path("configs/visualize/params.yaml")
PARAMS_MANIFEST_PATH = OUTPUT_DIR / "params_manifest.json"
TYPST_TABLES_MANIFEST_PATH = OUTPUT_DIR / "tables_manifest.json"
INCLUDE_GROUPED_ROWS = False
HIERARCHICAL_SPLIT_TABLES = True
ENABLED_DATASETS = {"db_bio", "tab"}

dataset_lengths = {
    "db_bio": 2419,
    "tab": 1268,
}


def read_logs(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r") as f:
        return json.load(f)


class LogMetricsEnricher:
    def __init__(self, data: list[dict[str, Any]]):
        self.data = data

    def aggregate(self) -> list[dict[str, Any]]:
        enriched: list[dict[str, Any]] = []
        for item in self.data:
            item_copy = dict(item)
            for section in ("utility", "supervised_divergence"):
                section_metrics = item_copy.get(section)
                if not isinstance(section_metrics, dict):
                    continue
                item_copy[section] = self._augment_utility_scores(
                    dict(section_metrics),
                    allow_missing_rmae=True,
                    dataset=item_copy.get("dataset"),
                )
            enriched.append(item_copy)
        return enriched

    def _extract_feature_counts(self, section_metrics: dict[str, Any], dataset: str) -> dict[str, int]:
        default_count = dataset_lengths.get(dataset)
        if default_count is None:
            raise ValueError(f"Unknown dataset: {dataset}")
        suffixes = ("_mae", "_rmae", "_acc", "_racc", "_count")
        features: set[str] = set()
        for metric_name in section_metrics.keys():
            if not isinstance(metric_name, str):
                continue
            for suffix in suffixes:
                if metric_name.endswith(suffix):
                    feature = metric_name[: -len(suffix)]
                    if feature and "_baseline" not in feature and "_dummy" not in feature:
                        features.add(feature)
                    break

        feature_counts: dict[str, int] = {}
        for feature in sorted(features):
            raw_count = section_metrics.get(f"{feature}_count")
            if isinstance(raw_count, (int, float)) and raw_count > 0:
                feature_counts[feature] = int(raw_count)
            else:
                feature_counts[feature] = default_count
        return feature_counts

    def _augment_utility_scores(
        self,
        utility_metrics: dict[str, Any],
        allow_missing_rmae: bool = False,
        dataset: str = None,
    ) -> dict[str, Any]:
        feature_counts = self._extract_feature_counts(utility_metrics, dataset)
        weighted_sum = 0.0
        weighted_count = 0
        ordinal_weighted_sum = 0.0
        ordinal_weighted_count = 0
        nominal_weighted_sum = 0.0
        nominal_weighted_count = 0
        ordinal_raw_sum = 0.0
        ordinal_raw_count = 0
        nominal_raw_sum = 0.0
        nominal_raw_count = 0
        for feature, count in feature_counts.items():
            if not feature:
                continue
            score: float | None = None
            mae_key = f"{feature}_mae"
            rmae_key = f"{feature}_rmae"
            acc_key = f"{feature}_acc"
            racc_key = f"{feature}_racc"
            if mae_key in utility_metrics:
                mae_value = float(utility_metrics[mae_key])
                ordinal_raw_sum += mae_value * count
                ordinal_raw_count += count
                if rmae_key in utility_metrics:
                    score = float(utility_metrics[rmae_key])
                elif not allow_missing_rmae:
                    raise ValueError(
                        f"Feature '{feature}' has mae but missing ordinal utility score '{rmae_key}'"
                    )
                if score is not None:
                    ordinal_weighted_sum += score * count
                    ordinal_weighted_count += count
            elif acc_key in utility_metrics:
                acc_value = float(utility_metrics[acc_key])
                nominal_raw_sum += acc_value * count
                nominal_raw_count += count
                if racc_key in utility_metrics:
                    score = float(utility_metrics[racc_key])
                elif not allow_missing_rmae:
                    raise ValueError(
                        f"Feature '{feature}' has acc but missing nominal utility score '{racc_key}'"
                    )
                if score is not None:
                    nominal_weighted_sum += score * count
                    nominal_weighted_count += count
            else:
                if not allow_missing_rmae:
                    raise ValueError(
                        f"Feature '{feature}' has neither ordinal mae nor nominal acc metric"
                    )
                continue
            if score is None:
                continue
            utility_metrics[f"{feature}_score"] = score
            weighted_sum += score * count
            weighted_count += count
        if weighted_count > 0:
            utility_metrics["utility_weighted_score"] = weighted_sum / weighted_count
            utility_metrics["utility_total_score"] = utility_metrics["utility_weighted_score"]
            utility_metrics["_utility_score_count"] = weighted_count
        if ordinal_weighted_count > 0:
            utility_metrics["utility_ordinal_weighted_score"] = ordinal_weighted_sum / ordinal_weighted_count
            utility_metrics["_utility_ordinal_score_count"] = ordinal_weighted_count
        if nominal_weighted_count > 0:
            utility_metrics["utility_nominal_weighted_score"] = nominal_weighted_sum / nominal_weighted_count
            utility_metrics["_utility_nominal_score_count"] = nominal_weighted_count
        if ordinal_raw_count > 0:
            utility_metrics["utility_ordinal_raw_mae"] = ordinal_raw_sum / ordinal_raw_count
            utility_metrics["_utility_ordinal_raw_count"] = ordinal_raw_count
        if nominal_raw_count > 0:
            utility_metrics["utility_nominal_raw_acc"] = nominal_raw_sum / nominal_raw_count
            utility_metrics["_utility_nominal_raw_count"] = nominal_raw_count
        return utility_metrics

class RedditUtilityByHardness:
    def __init__(self, data: list[dict[str, Any]]):
        self.data = data
        self.filtered_data = self.filter()
        self.grouped_data = self.group()

    def filter(self):
        filtered_data = []
        for item in self.data:
            if item['dataset'] == 'reddit' and 'group' in item and 'utility' in item:
                filtered_data.append(item)
        return filtered_data

    def group(self):
        groups = {}
        for item in self.filtered_data:
            group = item['group']
            metrics = item['utility']
            method_and_params = item["method"]
            if item["params"]:
                method_and_params += "?" + "&".join(f"{k}={v}" for k, v in item["params"].items())
            for feature, metric_value in metrics.items():
                if feature.startswith("_"):
                    feature = feature[1:]
                keys = ["macro_f1", "f1", "macro_mae", "mae", "acc", "count"]
                for key in keys:
                    if feature.endswith("_" + key):
                        feature = feature[:-len(key) - 1]
                        groups.setdefault(group, {}).setdefault(feature, {}).setdefault(key, {}).setdefault(method_and_params, metric_value)
                        break
        return groups


class FlatDatasetLogs:
    def __init__(self, data: list[dict[str, Any]], include_grouped: bool = False):
        self.data = data
        self.include_grouped = include_grouped
        self.filtered_data = self.filter()
        self.flattened_data = self.flatten()

    def filter(self) -> list[dict[str, Any]]:
        filtered_data: list[dict[str, Any]] = []
        for item in self.data:
            if item.get("dataset") not in ENABLED_DATASETS:
                continue
            has_group = "group" in item
            if self.include_grouped or not has_group:
                filtered_data.append(item)
        return filtered_data

    def flatten(self) -> list[dict[str, Any]]:
        flattened: list[dict[str, Any]] = []
        for item in self.filtered_data:
            params = item["params"] or {}
            if not isinstance(params, dict):
                params = {}
            flat_item: dict[str, Any] = {
                "dataset": item["dataset"],
                "method": item["method"],
                "params": order_params_dict(params),
                "split": item.get("split"),
            }
            for section in ("privacy", "utility", "supervised_divergence", "divergence", "runtime"):
                for metric, value in item.get(section, {}).items():
                    flat_item[f"{section}_{metric}"] = value
            if item["method"] == "baseline":
                flat_item.setdefault("divergence_bertscore", 0.0)
                flat_item.setdefault("divergence_cosine", 0.0)
            flattened.append(flat_item)
        sorted_flattened = self.sort_by_method_order(flattened)
        return sorted_flattened

    def by_dataset(self) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in self.flattened_data:
            grouped.setdefault(item["dataset"], []).append(item)
        return grouped
    
    def sort_by_method_order(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        param_specs = read_param_specs_config(PARAM_SETS_CONFIG_PATH)
        param_print_order = {
            spec["name"]: spec.get("print_order", "low_to_high")
            for spec in param_specs
        }
        method_order = [
            "baseline",
            "dummy",
            "presidio",
            "spacy",
            "manual",
            "baroud",
            "petre",
            "risk",
            "dpbart",
            "dpparaphrase",
            "dpprompt",
            "dpmlm_uniform",
            "dpmlm_shap"
        ]
        for item in items:
            method = item["method"]
            params = item["params"] or {}
            method_index = method_order.index(method) if method in method_order else len(method_order)

            ordered_params = sorted(params.items(), key=lambda kv: kv[0].lower())
            normalized_params: list[tuple[str, tuple[int, int | float | str]]] = []
            for param_name, raw_value in ordered_params:
                value = sortable_param_value(raw_value)
                if isinstance(value, (int, float)) and param_print_order.get(param_name) == "high_to_low":
                    value = -value
                value_type_rank = 0 if isinstance(value, (int, float)) else 1
                normalized_params.append((param_name, (value_type_rank, value)))

            item["_sort_key"] = (method_index, tuple(normalized_params))
        
        sorted_items = sorted(items, key=lambda x: x["_sort_key"])
        for item in sorted_items:
            del item["_sort_key"]
        return sorted_items


class MarkdownDatasetLogsWriter:
    def __init__(
        self,
        data: list[dict[str, Any]],
        output_dir: str | Path = ".",
        include_grouped: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.flat_logs = FlatDatasetLogs(data, include_grouped=include_grouped)
        self.dataset_projection_config = self._read_dataset_projection_config(DATASET_SETS_CONFIG_PATH)

    def _read_dataset_projection_config(self, path: str | Path) -> dict[str, list[dict[str, Any]]]:
        return read_dataset_projection_config(path)

    def _dataset_frames(self) -> dict[str, pd.DataFrame]:
        frames: dict[str, pd.DataFrame] = {}
        for dataset, logs in self.flat_logs.by_dataset().items():
            frames[dataset] = pd.DataFrame(logs)
        return frames

    def _write_frame(self, frame: pd.DataFrame, base_path: Path) -> list[Path]:
        base_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path = base_path.with_suffix(".md")
        csv_path = base_path.with_suffix(".csv")
        json_path = base_path.with_suffix(".json")
        with open(markdown_path, "w") as f:
            f.write(frame.to_markdown(index=False) + "\n")
        frame.to_csv(csv_path, index=False)
        with open(json_path, "w") as f:
            json.dump(
                {
                    "columns": list(frame.columns),
                    "rows": frame.to_dict(orient="records"),
                },
                f,
                indent=2,
            )
        return [markdown_path, csv_path, json_path]

    def _project_method_set_frame(self, dataset: str, frame: pd.DataFrame) -> pd.DataFrame:
        scores = self.dataset_projection_config.get(dataset)
        required_columns = ["method", "params"]
        columns: list[str] = list(required_columns)
        rename_map: dict[str, str] = {}
        for score in scores or []:
            score_name = score["name"]
            if score_name not in frame.columns:
                frame[score_name] = None
            columns.append(score_name)
            rename_map[score_name] = score.get("rename_as", score.get("print_as", score_name))

        missing_required = [column for column in required_columns if column not in frame.columns]
        if missing_required:
            raise ValueError(f"Missing required columns for dataset '{dataset}' projection: {missing_required}")

        projected = frame[columns].copy()
        projected = projected.rename(columns=rename_map)
        for score in scores or []:
            renamed_col = rename_map.get(score["name"], score["name"])
            exclude_methods = score.get("exclude", [])
            if exclude_methods and renamed_col in projected.columns:
                mask = projected["method"].isin(exclude_methods)
                projected.loc[mask, renamed_col] = None
        return projected

    def write_dataset_tables(self) -> list[Path]:
        written: list[Path] = []
        for dataset, frame in self._dataset_frames().items():
            written.extend(self._write_frame(frame, self.output_dir / dataset / "logs"))
        return written

    def write_method_set_tables(self, method_sets_file: str | Path) -> list[Path]:
        with open(method_sets_file, "r") as f:
            config = yaml.safe_load(f)
        method_sets = config.get("method_sets", [])

        written: list[Path] = []
        for dataset, logs in self.flat_logs.by_dataset().items():
            for method_set in method_sets:
                if not self._dataset_enabled_for_method_set(dataset, method_set):
                    continue
                set_name = method_set["name"]
                methods = method_set.get("methods", [])
                filtered_logs = [item for item in logs if self._matches_any_method_spec(item, methods)]
                if not filtered_logs:
                    continue
                filtered_logs = self._sort_by_method_set_order(filtered_logs, methods)
                frame = pd.DataFrame(filtered_logs)
                frame = self._project_method_set_frame(dataset, frame)
                written.extend(self._write_frame(frame, self.output_dir / dataset / set_name))
        return written

    def _sort_by_method_set_order(
        self, items: list[dict[str, Any]], method_specs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        def spec_index(item: dict[str, Any]) -> int:
            for i, spec in enumerate(method_specs):
                if self._matches_method_spec(item, spec):
                    return i
            return len(method_specs)

        return sorted(items, key=spec_index)

    def _matches_any_method_spec(
        self,
        item: dict[str, Any],
        method_specs: list[dict[str, Any]],
    ) -> bool:
        return any(self._matches_method_spec(item, method_spec) for method_spec in method_specs)

    def _dataset_enabled_for_method_set(self, dataset: str, method_set: dict[str, Any]) -> bool:
        return self._dataset_matches_scope(dataset, method_set.get("datasets"))

    def _dataset_enabled_for_method_spec(self, dataset: str, method_spec: dict[str, Any]) -> bool:
        return self._dataset_matches_scope(dataset, method_spec.get("datasets"))

    def _dataset_matches_scope(self, dataset: str, datasets_scope: Any) -> bool:
        if datasets_scope is None:
            return True
        if isinstance(datasets_scope, list):
            return dataset in datasets_scope
        raise ValueError(f"Unsupported datasets scope format: {type(datasets_scope)}")

    def _matches_method_spec(self, item: dict[str, Any], method_spec: dict[str, Any]) -> bool:
        if not self._dataset_enabled_for_method_spec(item["dataset"], method_spec):
            return False
        expected_method = method_spec["method"]
        if item["method"] != expected_method:
            return False

        split_spec = method_spec.get("split")
        item_split = item.get("split")
        if split_spec is not None and item_split is not None:
            if split_spec != item_split:
                return False
        elif split_spec is not None or item_split is not None:
            return False

        params = item["params"] or {}
        if not isinstance(params, dict):
            return False

        params_one_run = method_spec.get("params_one_run", [])
        params_one_run_keys = self._params_one_run_keys(params_one_run)
        expected_param_keys = set(method_spec.get("params", [])) | params_one_run_keys
        if set(params.keys()) != expected_param_keys:
            return False

        return self._matches_params_one_run_values(params, params_one_run)

    def _params_one_run_keys(self, params_one_run: Any) -> set[str]:
        if isinstance(params_one_run, dict):
            return set(params_one_run.keys())
        if isinstance(params_one_run, list):
            return set(params_one_run)
        raise ValueError(f"Unsupported params_one_run format: {type(params_one_run)}")

    def _matches_params_one_run_values(self, params: dict[str, Any], params_one_run: Any) -> bool:
        if isinstance(params_one_run, list):
            return True
        if not isinstance(params_one_run, dict):
            raise ValueError(f"Unsupported params_one_run format: {type(params_one_run)}")

        for param_name, allowed_values in params_one_run.items():
            if param_name not in params:
                return False
            if not isinstance(allowed_values, list):
                raise ValueError(
                    f"params_one_run values must be lists. Got {param_name}={type(allowed_values)}"
                )
            if not any(self._values_equal(params[param_name], allowed_value) for allowed_value in allowed_values):
                return False
        return True

    def _values_equal(self, left: Any, right: Any) -> bool:
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return float(left) == float(right)
        return left == right


def build_typst_tables_manifest(output_dir: Path) -> list[dict[str, Any]]:
    dataset_projection_config = read_dataset_projection_config(DATASET_SETS_CONFIG_PATH)
    dataset_print_labels = read_dataset_print_labels_config(DATASET_SETS_CONFIG_PATH)
    entries: list[dict[str, str]] = []
    for path in sorted(output_dir.glob("*/*.md")):
        dataset = path.parent.name
        section = path.stem
        if dataset not in ENABLED_DATASETS:
            continue
        if section == "logs":
            continue
        heatmap_columns = []
        for score in dataset_projection_config.get(dataset, []):
            heatmap_columns.append(
                {
                    "name": score.get("rename_as", score.get("print_as", score["name"])),
                    "source_name": score["name"],
                    "best": score.get("best"),
                    "worst": score.get("worst"),
                }
            )
        json_path = path.with_suffix(".json")
        if json_path.exists():
            with open(json_path, "r") as f:
                table_json = json.load(f)
            if "relative_gain" in table_json.get("columns", []):
                heatmap_columns.append(
                    {
                        "name": "relative_gain",
                        "source_name": "relative_gain",
                        "best": 1,
                        "worst": -1,
                    }
                )
        entries.append(
            {
                "file": str(path.relative_to(output_dir)),
                "csv_file": str(path.with_suffix(".csv").relative_to(output_dir)),
                "json_file": str(path.with_suffix(".json").relative_to(output_dir)),
                "dataset": dataset,
                "section": section,
                "title": typst_table_title(dataset, section, dataset_print_labels),
                "heatmap_columns": heatmap_columns,
                "render_mode": "hierarchical" if HIERARCHICAL_SPLIT_TABLES else "flat",
            }
        )
    return entries


def typst_table_title(dataset: str, section: str, dataset_print_labels: dict[str, str] | None = None) -> str:
    dataset_label = (dataset_print_labels or {}).get(dataset, dataset.replace("_", "-").upper())
    if section == "all":
        return f"{dataset_label} results (all methods)"
    return f"{dataset_label} results ({section.replace('_', ' ')})"


def _dataset_name_entries(dataset_set: dict[str, Any]) -> list[tuple[str, str | None]]:
    entries = dataset_set.get("names")
    if isinstance(entries, list):
        result: list[tuple[str, str | None]] = []
        for entry in entries:
            if isinstance(entry, str):
                result.append((entry, None))
                continue
            if isinstance(entry, dict):
                dataset_name = entry.get("name")
                dataset_print_as = entry.get("print_as")
                if isinstance(dataset_name, str):
                    result.append((dataset_name, dataset_print_as if isinstance(dataset_print_as, str) else None))
        if result:
            return result
    dataset_name = dataset_set.get("name")
    dataset_print_as = dataset_set.get("print_as")
    if isinstance(dataset_name, str):
        return [(dataset_name, dataset_print_as if isinstance(dataset_print_as, str) else None)]
    return []


def read_dataset_print_labels_config(path: str | Path) -> dict[str, str]:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    labels: dict[str, str] = {}
    for dataset_set in config.get("dataset_sets", []):
        for dataset_name, dataset_print_as in _dataset_name_entries(dataset_set):
            if dataset_print_as:
                labels[dataset_name] = dataset_print_as
    return labels


def read_dataset_projection_config(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    projections: dict[str, list[dict[str, Any]]] = {}
    for dataset_set in config.get("dataset_sets", []):
        for dataset_name, _ in _dataset_name_entries(dataset_set):
            projections[dataset_name] = dataset_set.get("scores", [])
    return projections


def read_param_specs_config(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config.get("param_sets", [])


def order_params_dict(params: dict[str, Any]) -> dict[str, Any]:
    ordered_keys = sorted(params.keys(), key=lambda key: key.lower())
    return {key: params[key] for key in ordered_keys}


def sortable_param_value(value: Any) -> int | float | str:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return value
    try:
        if "." in str(value):
            return float(value)
        return int(value)
    except (TypeError, ValueError):
        try:
            return float(value)
        except (TypeError, ValueError):
            return str(value)

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = LogMetricsEnricher(read_logs(INPUT_LOGS_PATH)).aggregate()
    data = [item for item in data if item.get("dataset") in ENABLED_DATASETS]

    writer = MarkdownDatasetLogsWriter(
        data,
        output_dir=OUTPUT_DIR,
        include_grouped=INCLUDE_GROUPED_ROWS,
    )
    writer.write_dataset_tables()
    writer.write_method_set_tables(METHOD_SETS_CONFIG_PATH)
    with open(PARAMS_MANIFEST_PATH, "w") as f:
        json.dump(read_param_specs_config(PARAM_SETS_CONFIG_PATH), f, indent=2)
    with open(TYPST_TABLES_MANIFEST_PATH, "w") as f:
        json.dump(build_typst_tables_manifest(OUTPUT_DIR), f, indent=2)
