import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_logs(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r") as f:
        return json.load(f)

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

    data = read_logs(args.input)

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
