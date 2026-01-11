from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


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


def parse_params_from_key(key: str) -> Tuple[str, Dict[str, Any]]:
    params: Dict[str, Any] = {}
    method = key
    if "?" in key:
        method, params_str = key.split("?", 1)
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
    return method, params


def normalize_source_key(source_path: str) -> str:
    key = re.sub(r"outputs/[a-z]+/[a-z]+/[0-9]{8}_[0-9]{6}_[a-z]+_(.*?).jsonl", r"\1", source_path)
    key = re.sub(r"_eps_[0-9]{3}(\?.*?)", r"\1", key)
    key = re.sub(r"(?:_k|_risk|_pii)(\?.*?)", r"\1", key)
    return key


def read_jsonl_entries(files: Sequence[Tuple[str, ...]] , kind: str) -> Iterable[Dict[str, Any]]:
    if kind not in {"privacy", "utility"}:
        raise ValueError("kind must be 'privacy' or 'utility'")
    for spec in files:
        fp = Path(spec[0])
        if not fp.exists():
            continue
        with fp.open() as file:
            for line in file:
                data = json.loads(line)
                if kind == "utility":
                    dataset_name, feature = spec[1], spec[2]
                    if data.get("type") == "experiment":
                        baseline_metrics = data["baseline_metrics"]
                        baseline_result = {
                            "method": "baseline",
                            "params": {},
                            "dataset": dataset_name,
                            "feature": feature,
                        }
                        for metric_name, metric_value in baseline_metrics.items():
                            baseline_result[f"utility_{metric_name}_{feature}"] = metric_value
                        yield baseline_result
                        continue
                    if data.get("type") != "evaluation":
                        continue
                    key = normalize_source_key(str(data.get("source", "")))
                    method, params = parse_params_from_key(key)
                    res = data["metrics"]
                    values = {}
                    for metric_name, metric_value in res.items():
                        values[f"utility_{metric_name}_{feature}"] = metric_value
                    yield {"method": method, "params": params, "dataset": dataset_name, "feature": feature, **values}
                else:
                    dataset_name = spec[1]
                    if data.get("type") == "experiment":
                        yield {
                            "method": "baseline",
                            "params": {},
                            "dataset": dataset_name,
                            **{f"privacy_{metric}": value for metric, value in data["score"].items()},
                        }
                        continue
                    if data.get("type") != "evaluation":
                        continue
                    key = normalize_source_key(str(data.get("source", "")))
                    method, params = parse_params_from_key(key)
                    res = data["summary"]
                    values = {
                        "privacy_mean_rank_change": res["mean"],
                        "privacy_median_rank_change": res["median"],
                        "privacy_num_rank_increased": res["improved"],
                        "privacy_num_rank_decreased": res["degraded"],
                        **{f"privacy_{metric}": value for metric, value in res.items()},
                    }
                    yield {"method": method, "params": params, "dataset": dataset_name, **values}
