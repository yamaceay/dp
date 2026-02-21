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
    key = re.sub(
        rf"outputs/[a-z_]+/[a-z_]+/[0-9]{{8}}_[0-9]{{6}}_{re.escape(dataset)}_(.*?)\.jsonl",
        r"\1",
        source_path,
    )
    key = re.sub(r"_task_[0-9]+", "", key)
    key = re.sub(r"_eps_[0-9]{3}(\?.*?)", r"\1", key)
    key = re.sub(r"(?:_k|_risk|_pii)(\?.*?)", r"\1", key)
    return key

def parse_task_id_from_text(value: str) -> int | None:
    match = re.search(r"_task_([0-9]+)", value)
    if not match:
        return None
    return int(match.group(1))

def parse_task_id_from_log_name(log_name: str) -> int | None:
    match = re.search(r"_exp_task_([0-9]+)\.jsonl$", log_name)
    if not match:
        return None
    return int(match.group(1))

def parse_privacy_profile_from_log_name(log_name: str) -> str | None:
    if re.search(r"_priv_more_exp_task_[0-9]+\.jsonl$", log_name):
        return "more"
    if re.search(r"_priv_exp_task_[0-9]+\.jsonl$", log_name):
        return "exact"
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
    def parse(log_file: str, dataset: str, feature: str, metrics: List[str]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        experiment_result = None
        total_count = None
        default_task_id = parse_task_id_from_log_name(Path(log_file).name)
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
                        "task_id": default_task_id,
                        "utility": baseline_utility,
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
                        experiment_result["utility"][f"_{feature}_num_classes"] = num_classes
                    results.append(experiment_result)
                source = str(result.get("source", ""))
                task_id = parse_task_id_from_text(source)
                if task_id is None:
                    task_id = default_task_id
                key = normalize_source_key(source, dataset)
                method, params = parse_params_from_key(key)
                utility_metrics = filter_utility_metrics(result["overall_results"]["metrics"], metrics, feature)
                results.append(
                    {
                        "dataset": dataset,
                        "feature": feature,
                        "method": method,
                        "params": params,
                        "task_id": task_id,
                        "count": result["overall_results"]["total"],
                        "utility": {
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
                                "task_id": task_id,
                                "group": group_name,
                                "count": group_result["total"],
                                "utility": {
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
        default_task_id = parse_task_id_from_log_name(log_name)
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
                            "task_id": default_task_id,
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
                task_id = parse_task_id_from_text(source)
                if task_id is None:
                    task_id = default_task_id
                key = normalize_source_key(source, dataset)
                method, params = parse_params_from_key(key)
                summary = result["summary"]
                results.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "params": params,
                        "task_id": task_id,
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
        default_task_id = parse_task_id_from_log_name(Path(log_file).name)
        with Path(log_file).open("r", encoding="utf-8") as f:
            for line in f:
                result = json.loads(line)
                record_type = result.get("type")
                if record_type != "evaluation":
                    continue
                source = str(result.get("source", ""))
                task_id = parse_task_id_from_text(source)
                if task_id is None:
                    task_id = default_task_id
                key = normalize_source_key(source, dataset)
                method, params = parse_params_from_key(key)
                divergence = result["summary"]
                results.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "params": params,
                        "task_id": task_id,
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
    for field in ["privacy", "utility", "divergence"]:
        if field in result:
            return field
    raise ValueError(f"Could not determine experiment type for result: {result}")

class LogGrouper:
    def group(results: List[Dict[str, Any]]) -> Dict[tuple, List[Dict[str, Any]]]:
        grouped: Dict[Tuple[str, str, frozenset, str | None], Dict[str, Any]] = {}
        for result in results:
            assert all(field in result for field in ["dataset", "method", "params"]), f"Missing required fields in result: {result}"
            assert any(field in result for field in ["privacy", "utility", "divergence"]), f"Missing privacy/utility/divergence field in result: {result}"
            
            identifiers = {k: result[k] for k in ["dataset", "method", "params"]}
            group = maybe_group(result)
            if group:
                identifiers["group"] = group

            key = (identifiers["dataset"], identifiers["method"], frozenset(identifiers["params"].items()), identifiers.get("group"))
            grouped.setdefault(key, {"dataset": identifiers["dataset"], "method": identifiers["method"], "params": identifiers["params"], "task_results": []})
            if "group" in identifiers:
                grouped[key]["group"] = identifiers["group"]

            type_of_experiment = experiment_type(result)
            metrics = dict(result[type_of_experiment])
            feature = maybe_feature(result)
            if feature:
                metrics["_{}_count".format(feature)] = result["count"]
            else:
                metrics["_count"] = result["count"]
            task_id = result.get("task_id")
            if task_id is None:
                task_id = -1
            task_result_entry: Dict[str, Any] = {"task_id": task_id}
            if type_of_experiment == "privacy":
                profile = str(result.get("privacy_profile") or "exact")
                prefixed_metrics: Dict[str, Any] = {}
                for metric_name, metric_value in metrics.items():
                    if metric_name.startswith("_"):
                        prefixed_metrics[f"_{profile}{metric_name}"] = metric_value
                    else:
                        prefixed_metrics[f"{profile}_{metric_name}"] = metric_value
                task_result_entry[type_of_experiment] = prefixed_metrics
            else:
                task_result_entry[type_of_experiment] = metrics
            grouped[key]["task_results"].append(task_result_entry)

        final_results: List[Dict[str, Any]] = []
        for entry in grouped.values():
            task_results = entry.pop("task_results")
            tasks = sorted(task_results, key=lambda task: int(task.get("task_id", -1)))
            for experiment_name in ["privacy", "utility", "divergence"]:
                numeric_sums: Dict[str, float] = {}
                numeric_counts: Dict[str, int] = {}
                count_sums: Dict[str, int] = {}
                class_max: Dict[str, int] = {}
                task_count = 0
                for task in tasks:
                    if experiment_name not in task:
                        continue
                    task_count += 1
                    for metric_name, metric_value in task[experiment_name].items():
                        if metric_name.startswith("_"):
                            if metric_name.endswith("_num_classes"):
                                class_max[metric_name] = max(class_max.get(metric_name, 0), int(metric_value))
                            elif metric_name.endswith("_count"):
                                count_sums[metric_name] = count_sums.get(metric_name, 0) + int(metric_value)
                            continue
                        numeric_sums[metric_name] = numeric_sums.get(metric_name, 0.0) + float(metric_value)
                        numeric_counts[metric_name] = numeric_counts.get(metric_name, 0) + 1
                if task_count == 0:
                    continue
                mean_metrics: Dict[str, Any] = {}
                for metric_name, metric_sum in numeric_sums.items():
                    mean_metrics[metric_name] = metric_sum / numeric_counts[metric_name]
                for metric_name, metric_sum in count_sums.items():
                    mean_metrics[metric_name] = metric_sum
                for metric_name, metric_sum in class_max.items():
                    mean_metrics[metric_name] = metric_sum
                mean_metrics["_tasks"] = task_count
                entry[experiment_name] = mean_metrics
            entry["task_results"] = tasks
            final_results.append(entry)

        return final_results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Merge experiment logs into a single JSON file.")
    parser.add_argument("-o", "--output", type=str, help="Output file for merged logs")
    args = parser.parse_args()

    all_logs = list(Path("logs").glob("*_exp_task_*.jsonl"))
    all_privacy_logs = [log for log in all_logs if re.search(r"_priv(?:_[a-z_]+)?_exp_task_[0-9]+\.jsonl$", log.name)]
    all_divergence_logs = [log for log in all_logs if re.search(r"_div_[a-z0-9_]+_exp_task_[0-9]+\.jsonl$", log.name)]
    all_utility_logs = list(set(all_logs) - set(all_privacy_logs) - set(all_divergence_logs))

    find_dataset = lambda log_file: re.match(r"(db_bio|reddit|tab)_(.*?).jsonl", log_file.name).group(1)
    divergence_find_metric = lambda log_file, dataset: re.search(rf"{re.escape(dataset)}_div_(.*?)_exp_task_[0-9]+\.jsonl", log_file.name).group(1)
    utility_find_feature = lambda log_file, dataset: re.search(rf"{re.escape(dataset)}_(.*?)_exp_task_[0-9]+\.jsonl", log_file.name).group(1)

    privacy_logs = [
        (log_file, dataset) for dataset, log_file in [(find_dataset(log_file), log_file) for log_file in all_privacy_logs]
    ]
    utility_logs = [
        (log_file, dataset, utility_find_feature(log_file, dataset)) for dataset, log_file in [(find_dataset(log_file), log_file) for log_file in all_utility_logs]
    ]
    divergence_logs = [
        (log_file, dataset, divergence_find_metric(log_file, dataset)) for dataset, log_file in [(find_dataset(log_file), log_file) for log_file in all_divergence_logs]
    ]

    utility_metrics = ["acc", "macro_mae", "mae", "macro_f1", "f1", "exp_rmae"]
    
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
        
