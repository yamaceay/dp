from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

FILES: List[Tuple[str, str]] = [
    ("logs/reddit_div_exp.jsonl", "reddit"),
    ("logs/tab_div_exp.jsonl", "tab"),
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
        with fp.open() as file:
            for line in file:
                data = json.loads(line)
                if data.get("type") != "evaluation":
                    continue
                key = re.sub(r"outputs/[a-z]+/[a-z]+/[0-9]{8}_[0-9]{6}_[a-z]+_(.*?).jsonl", r"\1", data["source"])
                key = re.sub(r"_eps_[0-9]{3}(\?.*?)", r"\1", key)
                key = re.sub(r"(?:_k|_risk|_pii)(\?.*?)", r"\1", key)
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
                    "divergence_mean_bertscore": res["divergence_mean"],
                }
                yield {"method": key, "params": params, "dataset": dataset_name, **values}

def build_summaries() -> None:
    entries = list(read_data(FILES))
    entries = sorted(entries, key=lambda x: (x["method"], params_to_str_for_sort(x["params"])))
    by_dataset_by_method: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for entry in entries:
        e = entry.copy()
        dataset = e.pop("dataset")
        method = e.pop("method")
        by_dataset_by_method.setdefault(dataset, {}).setdefault(method, []).append(e)
    by_dataset_by_params: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for entry in entries:
        e = entry.copy()
        dataset = e.pop("dataset")
        params = params_to_str(e.pop("params"))
        by_dataset_by_params.setdefault(dataset, {}).setdefault(params, []).append(e)
    Path("visualize/pretty").mkdir(parents=True, exist_ok=True)
    with open("visualize/pretty/divergence.json", "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    for dataset in by_dataset_by_method:
        with open(f"visualize/pretty/divergence_by_method_{dataset}.json", "w", encoding="utf-8") as f:
            json.dump(by_dataset_by_method[dataset], f, indent=2)
    for dataset in by_dataset_by_params:
        with open(f"visualize/pretty/divergence_by_params_{dataset}.json", "w", encoding="utf-8") as f:
            json.dump(by_dataset_by_params[dataset], f, indent=2)

if __name__ == "__main__":
    build_summaries()