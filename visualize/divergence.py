from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from visualize.common import normalize_source_key

FILES: List[Tuple[str, str]] = [
    ("logs/reddit_div_cosine_exp.jsonl", "reddit"),
    ("logs/tab_div_cosine_exp.jsonl", "tab"),
    ("logs/db_bio_div_cosine_exp.jsonl", "db_bio"),
    ("logs/reddit_div_bertscore_exp.jsonl", "reddit"),
    ("logs/tab_div_bertscore_exp.jsonl", "tab"),
    ("logs/db_bio_div_bertscore_exp.jsonl", "db_bio"),
]

def params_to_str_for_sort(params: Mapping[str, Any]) -> str:
    parts: List[Tuple[str, str]] = []
    for k, v in params.items():
        if k == "epsilon":
            parts.append((k, f"{1000 - int(v):03d}"))
        elif k == "k":
            parts.append((k, f"{int(v):02d}"))
        elif k in {"rho", "lambda"}:
            parts.append((k, f"{int(100 - float(v) * 100):03d}"))
    return "&".join(f"{k}={v}" for k, v in parts)

def params_to_str(params: Mapping[str, Any]) -> str:
    return "&".join(f"{k}={v}" for k, v in params.items())

def read_data(files: Sequence[Tuple[str, str]]) -> Iterable[Dict[str, Any]]:
    for file_path, dataset_name in files:
        fp = Path(file_path)
        if not fp.exists():
            continue
        metric_name = ""
        with fp.open() as file:
            for line in file:
                data = json.loads(line)
                if data.get("type") == "experiment":
                    metric_name = str(data.get("metric_name", "")) or metric_name
                    continue
                if data.get("type") != "evaluation":
                    continue
                key = normalize_source_key(str(data.get("source", "")))
                params: Dict[str, Any] = {}
                if "?" in key:
                    key, params_str = key.split("?", 1)
                    for param in params_str.split("&"):
                        if "=" not in param:
                            raise ValueError(f"Unexpected hyperparameter format: {param}")
                        name, value = param.split("=", 1)
                        if name in {"epsilon", "k"}:
                            value = int(value)
                        elif name in {"rho", "lambda"}:
                            value = float(value) / 100.0
                        params[name] = value
                params = dict(sorted(params.items(), key=lambda item: item[0]))
                res = data["summary"]
                values = {
                    "divergence_mean": res["divergence_mean"],
                }
                yield {"method": key, "params": params, "dataset": dataset_name, "metric": metric_name, **values}

def build_summaries() -> None:
    entries = list(read_data(FILES))
    entries = sorted(entries, key=lambda x: (x["method"], params_to_str_for_sort(x["params"])))
    by_dataset_by_metric: Dict[str, List[Dict[str, Any]]] = {}
    for entry in entries:
        dataset = entry["dataset"]
        metric = entry["metric"]
        by_dataset_by_metric.setdefault(dataset, {}).setdefault(metric, []).append(entry)
    Path("visualize/pretty").mkdir(parents=True, exist_ok=True)
    for dataset, dataset_entries in by_dataset_by_metric.items():
        Path(f"visualize/pretty/{dataset}").mkdir(parents=True, exist_ok=True)
        for metric_name, dataset_metric_entries in dataset_entries.items():
            with open(f"visualize/pretty/{dataset}/divergence/{metric_name}.json", "w", encoding="utf-8") as f:
                json.dump(dataset_metric_entries, f, indent=2)

if __name__ == "__main__":
    build_summaries()
