import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def read_logs(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r") as f:
        return json.load(f)


class TaskResultWeightedAggregator:
    def __init__(self, data: list[dict[str, Any]]):
        self.data = data

    def aggregate(self) -> list[dict[str, Any]]:
        aggregated: list[dict[str, Any]] = []
        for item in self.data:
            task_results = item.get("task_results")
            if not isinstance(task_results, list) or not task_results:
                aggregated.append(item)
                continue

            merged_item: dict[str, Any] = {}
            for key, value in item.items():
                if key in {"privacy", "utility", "divergence"}:
                    continue
                merged_item[key] = value

            for section in ("privacy", "utility", "divergence"):
                metrics = self._aggregate_section(task_results, section)
                if metrics:
                    merged_item[section] = metrics

            aggregated.append(merged_item)
        return aggregated

    def _aggregate_section(self, task_results: list[dict[str, Any]], section: str) -> dict[str, Any]:
        weighted_sums: dict[str, float] = {}
        weight_sums: dict[str, float] = {}
        count_sums: dict[str, int] = {}
        tasks = 0

        for task in task_results:
            section_metrics = task.get(section)
            if not isinstance(section_metrics, dict):
                continue
            tasks += 1
            feature_counts = self._extract_feature_counts(section_metrics)
            default_count = feature_counts.get("") or 1
            for metric_name, metric_value in section_metrics.items():
                if metric_name.startswith("_"):
                    count_sums[metric_name] = count_sums.get(metric_name, 0) + int(metric_value)
                    continue
                feature_name = self._feature_name(metric_name, feature_counts)
                metric_count = feature_counts.get(feature_name, default_count)
                weighted_sums[metric_name] = weighted_sums.get(metric_name, 0.0) + float(metric_value) * float(metric_count)
                weight_sums[metric_name] = weight_sums.get(metric_name, 0.0) + float(metric_count)

        if tasks == 0:
            return {}

        aggregated: dict[str, Any] = {}
        for metric_name, metric_sum in weighted_sums.items():
            weight = weight_sums.get(metric_name, 0.0)
            if weight <= 0.0:
                continue
            aggregated[metric_name] = metric_sum / weight
        for metric_name, metric_sum in count_sums.items():
            aggregated[metric_name] = metric_sum
        aggregated["_tasks"] = tasks
        return aggregated

    def _extract_feature_counts(self, section_metrics: dict[str, Any]) -> dict[str, int]:
        feature_counts: dict[str, int] = {}
        for metric_name, metric_value in section_metrics.items():
            if metric_name == "_count":
                feature_counts[""] = int(metric_value)
                continue
            match = re.match(r"^_(.+)_count$", metric_name)
            if match:
                feature_counts[match.group(1)] = int(metric_value)
        return feature_counts

    def _feature_name(self, metric_name: str, feature_counts: dict[str, int]) -> str:
        candidate_features = sorted(
            [feature for feature in feature_counts.keys() if feature],
            key=len,
            reverse=True,
        )
        for feature in candidate_features:
            if metric_name.startswith(feature + "_"):
                return feature
        return ""

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
            has_group = "group" in item
            if self.include_grouped or not has_group:
                filtered_data.append(item)
        return filtered_data

    def flatten(self) -> list[dict[str, Any]]:
        flattened: list[dict[str, Any]] = []
        for item in self.filtered_data:
            flat_item: dict[str, Any] = {
                "dataset": item["dataset"],
                "method": item["method"],
                "params": item["params"],
            }
            for section in ("privacy", "utility", "divergence"):
                for metric, value in item.get(section, {}).items():
                    flat_item[f"{section}_{metric}"] = value
            flattened.append(flat_item)
        sorted_flattened = self.sort_by_method_order(flattened)
        return sorted_flattened

    def by_dataset(self) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in self.flattened_data:
            grouped.setdefault(item["dataset"], []).append(item)
        return grouped
    
    def sort_by_method_order(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        method_order = [
            "baseline",
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
        param_order = [
            "epsilon",
            "k",
            "rho",
            "lambda",
        ]
        for item in items:
            method = item["method"]
            params = item["params"] or {}
            method_index = method_order.index(method) if method in method_order else len(method_order)
            
            param_indices = [param_order.index(k) if k in param_order else len(param_order) for k in params.keys()]
            
            param_values: list[int | float] = []
            for param in param_order:
                if param in params:
                    if param == "k" or param == "epsilon":
                        param_values.append(int(params[param]))
                    else:
                        param_values.append(float(params[param]))
                else:
                    param_values.append(0)
            item["_sort_key"] = (method_index, param_indices, tuple(param_values))
        
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

    def render(self) -> dict[str, str]:
        tables: dict[str, str] = {}
        grouped = self.flat_logs.by_dataset()
        for dataset, logs in grouped.items():
            frame = pd.DataFrame(logs)
            tables[dataset] = frame.to_markdown(index=False)
        return tables

    def write(self) -> list[Path]:
        written: list[Path] = []
        tables = self.render()
        for dataset, table_markdown in tables.items():
            output_path = self.output_dir / f"{dataset}_logs.md"
            with open(output_path, "w") as f:
                f.write(table_markdown + "\n")
            written.append(output_path)
        return written

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process merged logs.")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="merged_logs.json",
        help="Path to input merged logs JSON",
    )
    parser.add_argument(
        "--mode",
        choices=["reddit_utility", "markdown", "all"],
        default="all",
        help="What to generate",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="reddit_utility.json",
        help="Output path for reddit utility JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for per-dataset markdown tables",
    )
    parser.add_argument(
        "--include-grouped",
        action="store_true",
        help="Include records that contain `group` (default: only non-grouped rows)",
    )
    args = parser.parse_args()

    data = TaskResultWeightedAggregator(read_logs(args.input)).aggregate()

    if args.mode in {"reddit_utility", "all"}:
        utility = RedditUtilityByHardness(data)
        with open(args.output, "w") as f:
            json.dump(utility.grouped_data, f, indent=4)

    if args.mode in {"markdown", "all"}:
        writer = MarkdownDatasetLogsWriter(
            data,
            output_dir=args.output_dir,
            include_grouped=args.include_grouped,
        )
        writer.write()
