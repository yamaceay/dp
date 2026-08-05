from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dp.utils.log_keys import normalize_source_key, parse_params_from_key

SEQUENTIAL_PROGRESS_PARAMS = {"k", "rho", "lambda"}
CONSTANT_RUNTIME_METHODS = {"baroud", "risk_shap"}

dataset_lengths = {
    "db_bio": 2419,
    "tab": 1268,
    "rat_bench": 300,
}

ignored_methods: list[str] = []

def parse_privacy_profile_from_log_name(log_name: str) -> str | None:
    if re.search(r"^more\.jsonl$", log_name):
        return "more"
    if re.search(r"^exact\.jsonl$", log_name):
        return "exact"
    if re.search(r"^full\.jsonl$", log_name):
        return "full"
    match = re.match(r"^([a-z0-9]+)_subset\.jsonl$", log_name)
    if match:
        return f"{match.group(1)}_subset"
    match = re.match(r"^([a-z0-9]+)_k\.jsonl$", log_name)
    if match:
        return match.group(1)
    # Fallback for ablation-suffixed names (e.g. rat_bench's full_deid.jsonl,
    # full_nobart.jsonl, full_deid_nobart.jsonl): use the stem verbatim as the
    # profile, instead of silently defaulting to "exact" downstream.
    match = re.match(r"^([a-z0-9_]+)\.jsonl$", log_name)
    if match:
        return match.group(1)
    return None


def _load_test_indices(dataset: str) -> set[int]:
    path = Path("indices") / dataset / "test.txt"
    if not path.exists():
        return set()
    values: set[int] = set()
    for line in path.read_text().splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            values.add(int(text))
        except ValueError:
            continue
    return values


def _summary_from_rank_rows(rows: List[Dict[str, Any]], test_indices: set[int]) -> Dict[str, Any] | None:
    filtered: List[Dict[str, Any]] = []
    for row in rows:
        index_value = row.get("index")
        rank_value = row.get("rank")
        if not isinstance(index_value, int) or not isinstance(rank_value, (int, float)):
            continue
        # Log "index" fields are 1-based (dp/experiments/privacy_annotations.py builds
        # them via enumerate(..., start=1)), while indices/{dataset}/test.txt holds
        # 0-based dataset row positions (the convention used everywhere else, e.g.
        # dp/loaders/base.py's split filtering). Convert before comparing.
        if test_indices and (index_value - 1) not in test_indices:
            continue
        if rank_value <= 0:
            continue
        filtered.append(row)
    if not filtered:
        return None
    reciprocal_ranks = [1.0 / float(row["rank"]) for row in filtered]
    accuracy = sum(1 for row in filtered if float(row["rank"]) == 1.0) / float(len(filtered))
    return {
        "count": len(filtered),
        "mean_reciprocal_rank": sum(reciprocal_ranks) / float(len(filtered)),
        "accuracy": accuracy,
    }

def filter_utility_metrics(result: Dict[str, Any], metrics: List[str], feature: str) -> Dict[str, Any]:
    return {feature + "_" + k: result[k] for k in metrics if k in result}


def _extract_baseline_metric_overall(experiment_row: Dict[str, Any], metric_name: str) -> float | None:
    baseline_raw = experiment_row.get("baseline_overall_metrics")
    if isinstance(baseline_raw, dict) and isinstance(baseline_raw.get(metric_name), (int, float)):
        return float(baseline_raw.get(metric_name))
    fallback_key = f"baseline_{metric_name}_overall"
    fallback_value = experiment_row.get(fallback_key)
    if isinstance(fallback_value, (int, float)):
        return float(fallback_value)
    return None


def _extract_dummy_metric_overall(experiment_row: Dict[str, Any]) -> tuple[str, float, str | None] | None:
    dummy = experiment_row.get("baseline_dummy")
    if not isinstance(dummy, dict):
        return None
    strategy: str | None = dummy.get("strategy") if isinstance(dummy.get("strategy"), str) else None
    for metric_name in ("mae", "acc", "mse"):
        metric_payload = dummy.get(metric_name)
        if isinstance(metric_payload, dict) and isinstance(metric_payload.get("overall"), (int, float)):
            return metric_name, float(metric_payload.get("overall")), strategy
    median = experiment_row.get("baseline_median_dummy_mae")
    if isinstance(median, dict) and isinstance(median.get("overall"), (int, float)):
        return "mae", float(median.get("overall")), strategy or "median"
    return None


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
        split: str | None = None
        actual_feature = feature
        if feature.endswith("_subset"):
            actual_feature = feature[: -len("_subset")]
            split = "test"
        elif feature.endswith("_k"):
            actual_feature = feature[: -len("_k")]
        results: List[Dict[str, Any]] = []
        baseline_metric_overall: Dict[str, float] = {}
        dummy_metric_name: str | None = None
        dummy_metric_overall: float | None = None
        dummy_strategy: str | None = None
        baseline_row_metrics: Dict[str, Any] | None = None
        dummy_row_metrics: Dict[str, Any] | None = None
        with Path(log_file).open("r", encoding="utf-8") as f:
            for line in f:
                result = json.loads(line)
                record_type = result.get("type")
                if record_type == "experiment":
                    baseline_metric_overall = {}
                    for metric_name in metrics:
                        baseline_value = _extract_baseline_metric_overall(result, metric_name)
                        if baseline_value is not None:
                            baseline_metric_overall[metric_name] = baseline_value
                    dummy_metric = _extract_dummy_metric_overall(result)
                    if dummy_metric is None:
                        dummy_metric_name = None
                        dummy_metric_overall = None
                        dummy_strategy = None
                    else:
                        dummy_metric_name, dummy_metric_overall, dummy_strategy = dummy_metric
                    baseline_row_metrics = filter_utility_metrics(baseline_metric_overall, metrics, actual_feature)
                    if dummy_metric_name is not None and dummy_metric_overall is not None:
                        dummy_row_metrics = {
                            f"{actual_feature}_{dummy_metric_name}": dummy_metric_overall,
                        }
                        if dummy_strategy is not None:
                            dummy_row_metrics[f"{actual_feature}_dummy_strategy"] = dummy_strategy
                    else:
                        dummy_row_metrics = None
                    continue
                if record_type != "evaluation":
                    continue
                source = str(result.get("source", ""))
                key = normalize_source_key(source, dataset)
                method, params = parse_params_from_key(key)
                if method in ignored_methods:
                    continue
                utility_metrics = filter_utility_metrics(result["overall_results"]["metrics"], metrics, actual_feature)
                for metric_name, baseline_value in baseline_metric_overall.items():
                    utility_metrics[f"{actual_feature}_baseline_{metric_name}"] = baseline_value
                if dummy_metric_name is not None and dummy_metric_overall is not None:
                    utility_metrics[f"{actual_feature}_dummy_{dummy_metric_name}"] = dummy_metric_overall
                if dummy_strategy is not None:
                    utility_metrics[f"{actual_feature}_dummy_strategy"] = dummy_strategy
                row: Dict[str, Any] = {
                    "dataset": dataset,
                    "feature": actual_feature,
                    "method": method,
                    "params": params,
                    section_name: {
                        **utility_metrics,
                    },
                }
                if split is not None:
                    row["split"] = split
                results.append(row)
        if baseline_row_metrics:
            row = {
                "dataset": dataset,
                "feature": actual_feature,
                "method": "baseline",
                "params": {},
                section_name: dict(baseline_row_metrics),
            }
            if split is not None:
                row["split"] = split
            results.append(row)
        if dummy_row_metrics:
            row = {
                "dataset": dataset,
                "feature": actual_feature,
                "method": "dummy",
                "params": {},
                section_name: dict(dummy_row_metrics),
            }
            if split is not None:
                row["split"] = split
            results.append(row)
        return results

class PrivacyExperimentLogParser(ExperimentLogParser):
    def parse(log_file: str, dataset: str) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        log_name = Path(log_file).name
        privacy_profile = parse_privacy_profile_from_log_name(log_name)
        split: str | None = None
        if privacy_profile and privacy_profile.endswith("_subset"):
            privacy_profile = privacy_profile[: -len("_subset")]
            split = "test"
        # Any "full_*" profile (full, full_deid, full_nobart, full_deid_nobart, ...) is a
        # whole-corpus run and needs the same held-out test-set filtering as "full" itself,
        # unlike "exact"/"more" which are natively pre-generated over only the test subset.
        is_full_corpus_profile = bool(privacy_profile) and privacy_profile.startswith("full")
        test_indices = _load_test_indices(dataset) if is_full_corpus_profile else set()
        original_rank_rows: List[Dict[str, Any]] = []
        evaluation_rank_rows: Dict[str, List[Dict[str, Any]]] = {}
        evaluation_rows: List[Dict[str, Any]] = []
        experiment_row: Dict[str, Any] | None = None
        with Path(log_file).open("r", encoding="utf-8") as f:
            for line in f:
                result = json.loads(line)
                record_type = result.get("type")
                if record_type == "experiment":
                    experiment_row = result
                    continue
                if record_type == "original_rank":
                    original_rank_rows.append(result)
                    continue
                if record_type == "evaluation_rank":
                    evaluation_name = str(result.get("evaluation") or "")
                    if evaluation_name:
                        evaluation_rank_rows.setdefault(evaluation_name, []).append(result)
                    continue
                if record_type != "evaluation":
                    continue
                evaluation_rows.append(result)

        if experiment_row is not None:
            if is_full_corpus_profile:
                summary = _summary_from_rank_rows(original_rank_rows, test_indices)
                if summary is None:
                    score = experiment_row.get("score", {})
                    summary = {
                        "count": int(experiment_row.get("original_record_count", 0)),
                        "mean_reciprocal_rank": float(score.get("mean_reciprocal_rank", 0.0)),
                        "accuracy": float(score.get("accuracy", 0.0)),
                    }
            else:
                score = experiment_row.get("score", {})
                summary = {
                    "count": int(experiment_row.get("original_record_count", 0)),
                    "mean_reciprocal_rank": float(score.get("mean_reciprocal_rank", 0.0)),
                    "accuracy": float(score.get("accuracy", 0.0)),
                }
            row: Dict[str, Any] = {
                "dataset": dataset,
                "method": "baseline",
                "params": {},
                "privacy_profile": privacy_profile,
                "count": summary["count"],
                "privacy": {
                    "mean_reciprocal_rank": summary["mean_reciprocal_rank"],
                    "accuracy": summary["accuracy"],
                },
            }
            if split is not None:
                row["split"] = split
            results.append(row)

        for result in evaluation_rows:
            source = str(result.get("source", ""))
            key = normalize_source_key(source, dataset)
            method, params = parse_params_from_key(key)
            if method in ignored_methods:
                continue
            if is_full_corpus_profile:
                evaluation_name = str(result.get("name") or "")
                rank_rows = evaluation_rank_rows.get(evaluation_name, [])
                summary = _summary_from_rank_rows(rank_rows, test_indices)
                if summary is None:
                    raw_summary = result.get("summary", {})
                    summary = {
                        "count": int(raw_summary.get("count", result.get("record_count", 0))),
                        "mean_reciprocal_rank": float(raw_summary.get("mean_reciprocal_rank", 0.0)),
                        "accuracy": float(raw_summary.get("accuracy", 0.0)),
                    }
            else:
                raw_summary = result.get("summary", {})
                summary = {
                    "count": int(raw_summary.get("count", result.get("record_count", 0))),
                    "mean_reciprocal_rank": float(raw_summary.get("mean_reciprocal_rank", 0.0)),
                    "accuracy": float(raw_summary.get("accuracy", 0.0)),
                }
            row = {
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
            if split is not None:
                row["split"] = split
            results.append(row)
        return results

class DivergenceExperimentLogParser(ExperimentLogParser):
    def parse(log_file: str, dataset: str, metric: str) -> List[Dict[str, Any]]:
        split: str | None = None
        actual_metric = metric
        if metric.endswith("_subset"):
            actual_metric = metric[: -len("_subset")]
            split = "test"
        elif metric.endswith("_k"):
            actual_metric = metric[: -len("_k")]
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
                if method in ignored_methods:
                    continue
                divergence = result["summary"]
                row: Dict[str, Any] = {
                    "dataset": dataset,
                    "method": method,
                    "params": params,
                    "count": divergence["count"],
                    "divergence": {
                        actual_metric: divergence["divergence_mean"],
                    },
                }
                if split is not None:
                    row["split"] = split
                results.append(row)
        return results

def maybe_feature(result: Dict[str, Any]) -> str | None:
    return result.get("feature")

def maybe_group(result: Dict[str, Any]) -> str | None:
    return result.get("group")

def experiment_type(result: Dict[str, Any]) -> str:
    for field in ["privacy", "utility", "pseudo_utility", "divergence"]:
        if field in result:
            return field
    raise ValueError(f"Could not determine experiment type for result: {result}")


def _group_key_without_progress_params(dataset: str, method: str, params: Dict[str, Any]) -> tuple:
    progress_param = next((p for p in SEQUENTIAL_PROGRESS_PARAMS if p in params), None)
    filtered_params = {k: v for k, v in params.items() if k not in SEQUENTIAL_PROGRESS_PARAMS}
    return dataset, method, frozenset(filtered_params.items()), progress_param


def add_real_anonymization_runtime(grouped_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    max_pp_by_group: Dict[tuple, float] = {}

    for row in grouped_results:
        divergence = row.get("divergence")
        params = row.get("params")
        method = str(row.get("method"))
        if not isinstance(divergence, dict) or not isinstance(params, dict):
            continue
        if method in CONSTANT_RUNTIME_METHODS:
            continue
        if not any(param_name in params for param_name in SEQUENTIAL_PROGRESS_PARAMS):
            continue
        pp = divergence.get("pp")
        if not isinstance(pp, (int, float)):
            continue
        key = _group_key_without_progress_params(str(row.get("dataset")), method, params)
        max_pp_by_group[key] = max(float(pp), max_pp_by_group.get(key, float("-inf")))

    for row in grouped_results:
        runtime = row.get("runtime")
        if not isinstance(runtime, dict):
            continue
        anon_total_time_s = runtime.get("anon_total_time_s")
        if not isinstance(anon_total_time_s, (int, float)):
            continue

        real_anon_total_time_s = float(anon_total_time_s)
        method = str(row.get("method"))
        params = row.get("params")
        divergence = row.get("divergence")
        if method not in CONSTANT_RUNTIME_METHODS and isinstance(params, dict) and isinstance(divergence, dict) and any(param_name in params for param_name in SEQUENTIAL_PROGRESS_PARAMS):
            pp = divergence.get("pp")
            if isinstance(pp, (int, float)):
                key = _group_key_without_progress_params(str(row.get("dataset")), method, params)
                max_pp = max_pp_by_group.get(key)
                if isinstance(max_pp, float) and max_pp > 0.0:
                    real_anon_total_time_s = float(anon_total_time_s) * (float(pp) / max_pp)

        # Per-record pipeline costs added on top of the raw anonymization time:
        # - Presidio de-identification (run once per record before anonymization)
        # - Risk precomputation (SHAP scores, run once per record)
        # BK preprocessing and TRI training are excluded: they are one-time constants.
        for time_key in ("deid_total_time_s", "risk_total_time_s"):
            t = runtime.get(time_key)
            if isinstance(t, (int, float)):
                real_anon_total_time_s += float(t)

        runtime["real_anon_total_time_s"] = real_anon_total_time_s
        anon_texts_processed = runtime.get("anon_texts_processed")
        if isinstance(anon_texts_processed, (int, float)) and float(anon_texts_processed) > 0.0:
            runtime["avg_anon_time_s"] = real_anon_total_time_s / float(anon_texts_processed)

    return grouped_results

class LogGrouper:
    def group(results: List[Dict[str, Any]]) -> Dict[tuple, List[Dict[str, Any]]]:
        grouped: Dict[Tuple[str, str, frozenset, str | None, str | None], Dict[str, Any]] = {}
        valid_types = ["privacy", "utility", "pseudo_utility", "divergence", "runtime"]
        for result in results:
            assert all(field in result for field in ["dataset", "method", "params"]), f"Missing required fields in result: {result}"
            assert any(field in result for field in valid_types), f"Missing privacy/utility/divergence/runtime field in result: {result}"

            identifiers = {k: result[k] for k in ["dataset", "method", "params"]}
            group = maybe_group(result)
            if group:
                identifiers["group"] = group
            split: str | None = result.get("split")

            key = (identifiers["dataset"], identifiers["method"], frozenset(identifiers["params"].items()), identifiers.get("group"), split)
            grouped.setdefault(key, {"dataset": identifiers["dataset"], "method": identifiers["method"], "params": identifiers["params"]})
            if "group" in identifiers:
                grouped[key]["group"] = identifiers["group"]
            if split is not None:
                grouped[key]["split"] = split

            for type_of_experiment in valid_types:
                if type_of_experiment in result:
                    break
            else:
                raise ValueError(f"Could not determine experiment type for result: {result}")

            metrics = dict(result[type_of_experiment])
            # feature = maybe_feature(result)
            # if feature and type_of_experiment != "runtime":
            #     metrics["_{}_count".format(feature)] = result.get("count")
            # elif type_of_experiment != "runtime":
            #     metrics["_count"] = result.get("count")
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
                section[metric_name] = metric_value  # last-in wins; _k logs sort after base logs

        return list(grouped.values())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Merge experiment logs into a single JSON file.")
    parser.add_argument("-o", "--output", type=str, default="merged_logs.json", help="Output file for merged logs")
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
    pseudo_utility_logs = [
        (log_file, dataset, feature_from_log_name(log_file.name)) for log_file, dataset in iter_type_dataset_logs("pseudo_utility")
    ]
    divergence_logs = [
        (log_file, dataset, divergence_metric_from_log_name(log_file.name)) for log_file, dataset in iter_type_dataset_logs("divergence")
    ]

    def iter_runtime_logs() -> List[tuple[Path, str]]:
        root = Path("logs/runtime")
        if not root.exists():
            return []
        rows: List[tuple[Path, str]] = []
        for dataset_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
            dataset = dataset_dir.name
            for log_file in sorted(dataset_dir.glob("*.jsonl")):
                rows.append((log_file, dataset))
        return rows

    runtime_logs = iter_runtime_logs()

    utility_metrics = ["acc", "macro_mae", "mae", "macro_f1", "f1", "rmae"]

    all_results: List[Dict[str, Any]] = []
    for log_file, dataset in privacy_logs:
        all_results.extend(PrivacyExperimentLogParser.parse(log_file, dataset))
    for log_file, dataset, feature in utility_logs:
        all_results.extend(UtilityExperimentLogParser.parse(log_file, dataset, feature, utility_metrics, section_name="utility"))
    for log_file, dataset, feature in pseudo_utility_logs:
        all_results.extend(UtilityExperimentLogParser.parse(log_file, dataset, feature, utility_metrics, section_name="pseudo_utility"))
    for log_file, dataset, metric in divergence_logs:
        all_results.extend(DivergenceExperimentLogParser.parse(log_file, dataset, metric))

    for log_file, dataset in runtime_logs:
        with log_file.open("r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                method, params = entry.pop("method"), entry.pop("params")
                if method in ignored_methods:
                    continue
                all_results.append({
                    "dataset": dataset,
                    "method": method,
                    "params": params,
                    "runtime": {
                        **entry
                    },
                    "type": "runtime"
                })

    grouped_results = LogGrouper.group(all_results)
    grouped_results = add_real_anonymization_runtime(grouped_results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(grouped_results, f, indent=2)
    else:
        print(json.dumps(grouped_results, indent=2))
        
