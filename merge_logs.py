from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


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
    name = Path(source_path).name
    if name.endswith(".jsonl"):
        name = name[:-6]
    key = re.sub(rf"^\d{{8}}_\d{{6}}_{re.escape(dataset)}_", "", name)
    method, sep, params = key.partition("?")
    method = re.sub(r"_eps_[0-9]{3}$", "", method)
    method = re.sub(r"(?:_k|_risk|_pii)$", "", method)
    return method + ((sep + params) if sep else "")

def parse_privacy_profile_from_log_name(log_name: str) -> str | None:
    if re.search(r"^more\.jsonl$", log_name):
        return "more"
    if re.search(r"^exact\.jsonl$", log_name):
        return "exact"
    if re.search(r"^full\.jsonl$", log_name):
        return "full"
    return None

def filter_utility_metrics(result: Dict[str, Any], metrics: List[str], feature: str) -> Dict[str, Any]:
    return {feature + "_" + k: result[k] for k in metrics if k in result}


def extract_num_classes_from_utility_result(overall_results: Dict[str, Any]) -> int | None:
    confusion_matrix = overall_results.get("confusion_matrix")
    if not isinstance(confusion_matrix, list) or not confusion_matrix:
        return None
    if not isinstance(confusion_matrix[0], list):
        return None
    rows = len(confusion_matrix)
    cols = len(confusion_matrix[0])
    if rows <= 0 or cols <= 0:
        return None
    return rows if rows >= cols else cols


class ExperimentLogParser:
    def parse(log_file: str, dataset: str, *args, **kwargs) -> List[Dict[str, Any]]:
        raise NotImplementedError

class UtilityExperimentLogParser(ExperimentLogParser):
    def parse(
        log_file: str,
        dataset: str,
        feature: str,
        metrics: List[str],
        section_name: str = "utility",
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        experiment_result = None
        total_count = None
        with Path(log_file).open("r", encoding="utf-8") as f:
            for line in f:
                result = json.loads(line)
                record_type = result.get("type")
                if record_type == "experiment":
                    baseline_utility = filter_utility_metrics(result["baseline_overall_metrics"], metrics, feature)
                    experiment_result = {
                        "dataset": dataset,
                        "feature": feature,
                        "method": "baseline",
                        "params": {},
                        section_name: baseline_utility,
                    }
                    continue
                if record_type != "evaluation":
                    continue
                count = result["overall_results"]["total"]
                num_classes = extract_num_classes_from_utility_result(result["overall_results"])
                if experiment_result and total_count is None:
                    total_count = count
                    experiment_result["count"] = count
                    if num_classes is not None:
                        experiment_result[section_name][f"_{feature}_num_classes"] = num_classes
                    results.append(experiment_result)
                source = str(result.get("source", ""))
                key = normalize_source_key(source, dataset)
                method, params = parse_params_from_key(key)
                utility_metrics = filter_utility_metrics(result["overall_results"]["metrics"], metrics, feature)
                results.append(
                    {
                        "dataset": dataset,
                        "feature": feature,
                        "method": method,
                        "params": params,
                        "count": result["overall_results"]["total"],
                        section_name: {
                            **utility_metrics,
                            **({f"_{feature}_num_classes": num_classes} if num_classes is not None else {}),
                        },
                    }
                )

                if "grouped_results" in result:
                    for group_name, group_result in result["grouped_results"].items():
                        grouped_metrics = filter_utility_metrics(group_result["metrics"], metrics, feature)
                        grouped_num_classes = extract_num_classes_from_utility_result({"confusion_matrix": group_result.get("confusion_matrix")})
                        results.append(
                            {
                                "dataset": dataset,
                                "feature": feature,
                                "method": method,
                                "params": params,
                                "group": group_name,
                                "count": group_result["total"],
                                section_name: {
                                    **grouped_metrics,
                                    **({f"_{feature}_num_classes": grouped_num_classes} if grouped_num_classes is not None else {}),
                                },
                            }
                        )
        return results

class PrivacyExperimentLogParser(ExperimentLogParser):
    def parse(log_file: str, dataset: str) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        log_name = Path(log_file).name
        privacy_profile = parse_privacy_profile_from_log_name(log_name)
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
                            "privacy_profile": privacy_profile,
                            "count": result["original_record_count"],
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
                        "privacy_profile": privacy_profile,
                        "count": summary["count"],
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
                        "count": divergence["count"],
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
    for field in ["privacy", "utility", "supervised_divergence", "divergence"]:
        if field in result:
            return field
    raise ValueError(f"Could not determine experiment type for result: {result}")

class LogGrouper:
    def group(results: List[Dict[str, Any]]) -> Dict[tuple, List[Dict[str, Any]]]:
        grouped: Dict[Tuple[str, str, frozenset, str | None], Dict[str, Any]] = {}
        for result in results:
            assert all(field in result for field in ["dataset", "method", "params"]), f"Missing required fields in result: {result}"
            assert any(field in result for field in ["privacy", "utility", "supervised_divergence", "divergence"]), f"Missing privacy/utility/divergence field in result: {result}"
            
            identifiers = {k: result[k] for k in ["dataset", "method", "params"]}
            group = maybe_group(result)
            if group:
                identifiers["group"] = group

            key = (identifiers["dataset"], identifiers["method"], frozenset(identifiers["params"].items()), identifiers.get("group"))
            grouped.setdefault(key, {"dataset": identifiers["dataset"], "method": identifiers["method"], "params": identifiers["params"]})
            if "group" in identifiers:
                grouped[key]["group"] = identifiers["group"]

            type_of_experiment = experiment_type(result)
            metrics = dict(result[type_of_experiment])
            feature = maybe_feature(result)
            if feature:
                metrics["_{}_count".format(feature)] = result["count"]
            else:
                metrics["_count"] = result["count"]
            if type_of_experiment == "privacy":
                profile = str(result.get("privacy_profile") or "exact")
                prefixed_metrics: Dict[str, Any] = {}
                for metric_name, metric_value in metrics.items():
                    if metric_name.startswith("_"):
                        prefixed_metrics[f"_{profile}{metric_name}"] = metric_value
                    else:
                        prefixed_metrics[f"{profile}_{metric_name}"] = metric_value
                metrics = prefixed_metrics

            section = grouped[key].setdefault(type_of_experiment, {})
            assert isinstance(section, dict), f"Unexpected section type for key {type_of_experiment}"
            for metric_name, metric_value in metrics.items():
                if metric_name in section:
                    assert section[metric_name] == metric_value, f"Conflicting value for {metric_name}: {section[metric_name]} vs {metric_value}"
                    continue
                section[metric_name] = metric_value

        return list(grouped.values())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Merge experiment logs into a single JSON file.")
    parser.add_argument("-o", "--output", type=str, help="Output file for merged logs")
    args = parser.parse_args()

    def iter_type_dataset_logs(type_name: str) -> List[tuple[Path, str]]:
        root = Path("logs") / type_name
        if not root.exists():
            return []
        rows: List[tuple[Path, str]] = []
        for dataset_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
            dataset = dataset_dir.name
            for log_file in sorted(dataset_dir.glob("*.jsonl")):
                rows.append((log_file, dataset))
        return rows

    def feature_from_log_name(log_name: str) -> str:
        match = re.match(r"(.+?)\.jsonl$", log_name)
        if not match:
            raise ValueError(f"Could not parse feature from utility log name: {log_name}")
        return match.group(1)

    def divergence_metric_from_log_name(log_name: str) -> str:
        match = re.match(r"([a-z0-9_]+)\.jsonl$", log_name)
        if not match:
            raise ValueError(f"Could not parse divergence metric from log name: {log_name}")
        return match.group(1)

    privacy_logs = iter_type_dataset_logs("privacy")
    utility_logs = [(log_file, dataset, feature_from_log_name(log_file.name)) for log_file, dataset in iter_type_dataset_logs("utility")]
    supervised_divergence_logs = [
        (log_file, dataset, feature_from_log_name(log_file.name)) for log_file, dataset in iter_type_dataset_logs("supervised_divergence")
    ]
    divergence_logs = [
        (log_file, dataset, divergence_metric_from_log_name(log_file.name)) for log_file, dataset in iter_type_dataset_logs("divergence")
    ]

    utility_metrics = ["acc", "macro_mae", "mae", "macro_f1", "f1", "exp_rmae"]
    
    all_results: List[Dict[str, Any]] = []
    for log_file, dataset in privacy_logs:
        all_results.extend(PrivacyExperimentLogParser.parse(log_file, dataset))
    for log_file, dataset, feature in utility_logs:
        all_results.extend(UtilityExperimentLogParser.parse(log_file, dataset, feature, utility_metrics, section_name="utility"))
    for log_file, dataset, feature in supervised_divergence_logs:
        all_results.extend(UtilityExperimentLogParser.parse(log_file, dataset, feature, utility_metrics, section_name="supervised_divergence"))
    for log_file, dataset, metric in divergence_logs:
        all_results.extend(DivergenceExperimentLogParser.parse(log_file, dataset, metric))
    
    grouped_results = LogGrouper.group(all_results)
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(grouped_results, f, indent=2)
    else:
        print(json.dumps(grouped_results, indent=2))
        
