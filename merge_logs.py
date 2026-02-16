from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

def parse_params_from_key(key: str) -> tuple[str, Dict[str, Any]]:
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


def normalize_source_key(source_path: str, dataset: str) -> str:
    key = re.sub(
        rf"outputs/[a-z_]+/[a-z_]+/[0-9]{{8}}_[0-9]{{6}}_{re.escape(dataset)}_(.*?)\.jsonl",
        r"\1",
        source_path,
    )
    key = re.sub(r"_eps_[0-9]{3}(\?.*?)", r"\1", key)
    key = re.sub(r"(?:_k|_risk|_pii)(\?.*?)", r"\1", key)
    return key

def filter_utility_metrics(result: Dict[str, Any], metrics: List[str], feature: str) -> Dict[str, Any]:
    return {feature + "_" + k: result[k] for k in metrics if k in result}

class ExperimentLogParser:
    def parse(log_file: str, dataset: str, *args, **kwargs) -> List[Dict[str, Any]]:
        raise NotImplementedError

class UtilityExperimentLogParser(ExperimentLogParser):
    def parse(log_file: str, dataset: str, feature: str, metrics: List[str]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        with Path(log_file).open("r", encoding="utf-8") as f:
            for line in f:
                result = json.loads(line)
                record_type = result.get("type")
                if record_type == "experiment":
                    results.append(
                        {
                            "dataset": dataset,
                            "feature": feature,
                            "method": "baseline",
                            "params": {},
                            "utility": filter_utility_metrics(result["baseline_overall_metrics"], metrics, feature)
                        }
                    )
                    continue
                if record_type != "evaluation":
                    continue
                source = str(result.get("source", ""))
                key = normalize_source_key(source, dataset)
                method, params = parse_params_from_key(key)
                results.append(
                    {
                        "dataset": dataset,
                        "feature": feature,
                        "method": method,
                        "params": params,
                        "utility": filter_utility_metrics(result["overall_results"]["metrics"], metrics, feature),
                    }
                )

                if "grouped_results" in result:
                    for group_name, group_result in result["grouped_results"].items():
                        results.append(
                            {
                                "dataset": dataset,
                                "feature": feature,
                                "method": method,
                                "params": params,
                                "group": group_name,
                                "utility": filter_utility_metrics(group_result["metrics"], metrics, feature),
                            }
                        )
        return results

class PrivacyExperimentLogParser(ExperimentLogParser):
    def parse(log_file: str, dataset: str) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        with Path(log_file).open("r", encoding="utf-8") as f:
            for line in f:
                result = json.loads(line)
                record_type = result.get("type")
                if record_type == "experiment":
                    score = result["score"]
                    results.append(
                        {
                            "dataset": dataset,
                            "method": "baseline",
                            "params": {},
                            "privacy": {
                                "mean_reciprocal_rank": score["mean_reciprocal_rank"],
                                "accuracy": score["accuracy"],
                            },
                        }
                    )
                    continue
                if record_type != "evaluation":
                    continue
                source = str(result.get("source", ""))
                key = normalize_source_key(source, dataset)
                method, params = parse_params_from_key(key)
                summary = result["summary"]
                results.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "params": params,
                        "privacy": {
                            "mean_reciprocal_rank": summary["mean_reciprocal_rank"],
                            "accuracy": summary["accuracy"],
                        },
                    }
                )
        return results

class DivergenceExperimentLogParser(ExperimentLogParser):
    def parse(log_file: str, dataset: str, metric: str) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        with Path(log_file).open("r", encoding="utf-8") as f:
            for line in f:
                result = json.loads(line)
                record_type = result.get("type")
                if record_type != "evaluation":
                    continue
                source = str(result.get("source", ""))
                key = normalize_source_key(source, dataset)
                method, params = parse_params_from_key(key)
                divergence = result["summary"]
                results.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "params": params,
                        "divergence": {
                            metric: divergence["divergence_mean"],
                        },
                    }
                )
        return results

def maybe_feature(result: Dict[str, Any]) -> str | None:
    return result.get("feature")

def maybe_group(result: Dict[str, Any]) -> str | None:
    return result.get("group")

def experiment_type(result: Dict[str, Any]) -> str:
    for field in ["privacy", "utility", "divergence"]:
        if field in result:
            return field
    raise ValueError(f"Could not determine experiment type for result: {result}")

class LogGrouper:
    def group(results: List[Dict[str, Any]]) -> Dict[tuple, List[Dict[str, Any]]]:
        grouped: Dict[tuple, List[Dict[str, Any]]] = {}
        for result in results:
            assert all(field in result for field in ["dataset", "method", "params"]), f"Missing required fields in result: {result}"
            assert any(field in result for field in ["privacy", "utility", "divergence"]), f"Missing privacy/utility/divergence field in result: {result}"
            
            identifiers = {k: result[k] for k in ["dataset", "method", "params"]}
            group = maybe_group(result)
            if group:
                identifiers["group"] = group
            feature = maybe_feature(result)
            if feature:
                identifiers["feature"] = feature

            key = (identifiers["dataset"], identifiers["method"], frozenset(identifiers["params"].items()), identifiers.get("group"))

            type_of_experiment = experiment_type(result)
            grouped.setdefault(key, identifiers).setdefault(type_of_experiment, {}).update(result[type_of_experiment])

        return list(grouped.values())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Merge experiment logs into a single JSON file.")
    parser.add_argument("-o", "--output", type=str, help="Output file for merged logs")
    args = parser.parse_args()

    all_privacy_logs = list(Path("logs").glob("*_priv_exp.jsonl"))
    all_divergence_logs = list(Path("logs").glob("*_div_*_exp.jsonl"))
    all_utility_logs = list(set(Path("logs").glob("*_*_exp.jsonl")) - set(all_privacy_logs) - set(all_divergence_logs))

    find_dataset = lambda log_file: re.match(r"(db_bio|reddit|tab)_(.*?).jsonl", log_file.name).group(1)
    divergence_find_metric = lambda log_file, dataset: re.search(rf"{re.escape(dataset)}_div_(.*?)_exp.jsonl", log_file.name).group(1)
    utility_find_feature = lambda log_file, dataset: re.search(rf"{re.escape(dataset)}_(.*?)_exp.jsonl", log_file.name).group(1)

    privacy_logs = [
        (log_file, dataset) for dataset, log_file in [(find_dataset(log_file), log_file) for log_file in all_privacy_logs]
    ]
    utility_logs = [
        (log_file, dataset, utility_find_feature(log_file, dataset)) for dataset, log_file in [(find_dataset(log_file), log_file) for log_file in all_utility_logs]
    ]
    divergence_logs = [
        (log_file, dataset, divergence_find_metric(log_file, dataset)) for dataset, log_file in [(find_dataset(log_file), log_file) for log_file in all_divergence_logs]
    ]

    utility_metrics = ["acc", "macro_mae", "mae", "macro_f1", "f1"]
    
    all_results: List[Dict[str, Any]] = []
    for log_file, dataset in privacy_logs:
        all_results.extend(PrivacyExperimentLogParser.parse(log_file, dataset))
    for log_file, dataset, feature in utility_logs:
        all_results.extend(UtilityExperimentLogParser.parse(log_file, dataset, feature, utility_metrics))
    for log_file, dataset, metric in divergence_logs:
        all_results.extend(DivergenceExperimentLogParser.parse(log_file, dataset, metric))
    
    grouped_results = LogGrouper.group(all_results)
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(grouped_results, f, indent=2)
    else:
        print(json.dumps(grouped_results, indent=2))
        
