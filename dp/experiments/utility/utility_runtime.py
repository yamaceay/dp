from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from dp.experiments import ExperimentResult
from dp.experiments.utility.base import UtilityTarget
from dp.experiments.utility.io import iter_utility_evaluation_texts
from dp.experiments.utility.models import UtilitySpec
from dp.loaders.base import DatasetRecord


@dataclass(frozen=True)
class UtilityConfig:
    random_state: int = 42
    tune_param: Optional[str] = None
    tune_values: Optional[List[Any]] = None


def _normalize_label(value: Any, mode: UtilityTarget.Mode) -> Optional[Any]:
    if value is None:
        return None
    if mode is UtilityTarget.Mode.CARDINAL:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    return text


def _fit_model(
    model: Any,
    train_x: Any,
    train_y: Sequence[Any],
    eval_x: Optional[Any],
    eval_y: Optional[Sequence[Any]],
) -> None:
    model_name = str(getattr(model, "name", "")).lower()
    if model_name.startswith("bert_"):
        use_eval_x = train_x if eval_x is None else eval_x
        use_eval_y = train_y if eval_y is None else eval_y
        model.fit(train_x, train_y, use_eval_x, use_eval_y)
        return
    model.fit(train_x, train_y)


def _score_difference(baseline: Dict[str, float], current: Dict[str, float]) -> Dict[str, float]:
    drops: Dict[str, float] = {}
    for name, value in baseline.items():
        if name in current:
            drops[f"{name}_drop"] = float(value - current[name])
    return drops


def _metric_is_minimized(metric_name: str) -> bool:
    name = str(metric_name or "").lower()
    return any(token in name for token in ("loss", "mae", "rmse", "mse", "error"))


def _subset_by_keys(
    keys: Sequence[str],
    text_by_key: Dict[str, str],
    label_by_key: Dict[str, Any],
) -> Tuple[List[str], List[str], List[Any]]:
    out_keys: List[str] = []
    out_texts: List[str] = []
    out_labels: List[Any] = []
    for key in keys:
        if key not in text_by_key or key not in label_by_key:
            continue
        text = text_by_key[key]
        if not text:
            continue
        out_keys.append(key)
        out_texts.append(text)
        out_labels.append(label_by_key[key])
    return out_keys, out_texts, out_labels


def _fit_and_eval_single(
    *,
    spec: UtilitySpec,
    vectorizer_name: Optional[str],
    vectorizer_kwargs: Dict[str, Any],
    head_name: Optional[str],
    head_kwargs: Dict[str, Any],
    identifier: Optional[str],
    train_texts: Sequence[str],
    train_labels: Sequence[Any],
    val_texts: Sequence[str],
    val_labels: Sequence[Any],
    test_texts: Sequence[str],
    test_labels: Sequence[Any],
    overall_texts: Sequence[str],
    overall_labels: Sequence[Any],
) -> Dict[str, Any]:
    vectorizer, model = spec.build_components(
        vectorizer_name=vectorizer_name,
        vectorizer_kwargs=vectorizer_kwargs,
        head_name=head_name,
        head_kwargs=head_kwargs,
        identifier=identifier,
    )
    try:
        if hasattr(model, "set_label_order") and spec.target.label_order:
            model.set_label_order(spec.target.label_order)
        model.setup()
        vectorizer.fit(list(train_texts))
        x_train = vectorizer.transform(list(train_texts))
        x_val = vectorizer.transform(list(val_texts)) if val_texts else x_train
        y_val = list(val_labels) if val_labels else list(train_labels)
        _fit_model(model, x_train, list(train_labels), x_val, y_val)
        train_metrics = {k: float(v) for k, v in model.evaluate(x_train, list(train_labels)).items()}
        val_metrics = {k: float(v) for k, v in model.evaluate(x_val, y_val).items()}
        x_test = vectorizer.transform(list(test_texts)) if test_texts else None
        test_metrics = {k: float(v) for k, v in model.evaluate(x_test, list(test_labels)).items()} if x_test is not None else {}
        x_overall = vectorizer.transform(list(overall_texts)) if overall_texts else None
        overall_metrics = {k: float(v) for k, v in model.evaluate(x_overall, list(overall_labels)).items()} if x_overall is not None else {}
        return {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
            "overall": overall_metrics,
        }
    finally:
        try:
            model.cleanup()
        finally:
            vectorizer.cleanup()


def _pick_tuned_head_kwargs(
    *,
    base_head_kwargs: Dict[str, Any],
    tune_param: Optional[str],
    tune_values: Optional[Sequence[Any]],
    spec: UtilitySpec,
    vectorizer_name: Optional[str],
    vectorizer_kwargs: Dict[str, Any],
    head_name: Optional[str],
    identifier: Optional[str],
    primary_metric: str,
    train_texts: Sequence[str],
    train_labels: Sequence[Any],
    val_texts: Sequence[str],
    val_labels: Sequence[Any],
    test_texts: Sequence[str],
    test_labels: Sequence[Any],
    overall_texts: Sequence[str],
    overall_labels: Sequence[Any],
) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, Any], List[Dict[str, Any]]]:
    values = list(tune_values or [])
    if not tune_param or not values:
        eval_result = _fit_and_eval_single(
            spec=spec,
            vectorizer_name=vectorizer_name,
            vectorizer_kwargs=vectorizer_kwargs,
            head_name=head_name,
            head_kwargs=base_head_kwargs,
            identifier=identifier,
            train_texts=train_texts,
            train_labels=train_labels,
            val_texts=val_texts,
            val_labels=val_labels,
            test_texts=test_texts,
            test_labels=test_labels,
            overall_texts=overall_texts,
            overall_labels=overall_labels,
        )
        return dict(base_head_kwargs), eval_result["val"], eval_result, []

    minimize = _metric_is_minimized(primary_metric)
    trials: List[Dict[str, Any]] = []
    best_kwargs: Optional[Dict[str, Any]] = None
    best_val: Optional[float] = None
    best_eval: Optional[Dict[str, Any]] = None
    for value in values:
        candidate_kwargs = dict(base_head_kwargs)
        candidate_kwargs[tune_param] = value
        eval_result = _fit_and_eval_single(
            spec=spec,
            vectorizer_name=vectorizer_name,
            vectorizer_kwargs=vectorizer_kwargs,
            head_name=head_name,
            head_kwargs=candidate_kwargs,
            identifier=identifier,
            train_texts=train_texts,
            train_labels=train_labels,
            val_texts=val_texts,
            val_labels=val_labels,
            test_texts=test_texts,
            test_labels=test_labels,
            overall_texts=overall_texts,
            overall_labels=overall_labels,
        )
        val_metrics = eval_result.get("val", {})
        score = float(val_metrics.get(primary_metric, 0.0))
        trials.append({"value": value, "val_metrics": dict(val_metrics), "score": score})
        if best_val is None:
            best_val = score
            best_kwargs = candidate_kwargs
            best_eval = eval_result
            continue
        is_better = score < best_val if minimize else score > best_val
        if is_better:
            best_val = score
            best_kwargs = candidate_kwargs
            best_eval = eval_result
    if best_kwargs is None or best_eval is None:
        raise ValueError("failed to resolve tuned head params")
    return best_kwargs, best_eval.get("val", {}), best_eval, trials


def run_utility_experiment(
    *,
    spec: UtilitySpec,
    records: Sequence[DatasetRecord],
    evaluation_texts: Optional[Dict[str, Dict[str, str]]],
    evaluation_sources: Optional[Dict[str, Path]],
    index_to_key: Optional[Dict[int, str]],
    vectorizer_name: Optional[str],
    vectorizer_kwargs: Dict[str, Any],
    head_name: Optional[str],
    head_kwargs: Dict[str, Any],
    identifier: Optional[str],
    config: UtilityConfig,
    model_name: str,
    primary_metric: str,
    split_keys: Dict[str, List[str]],
    protocol: str,
) -> ExperimentResult:
    label_by_key: Dict[str, Any] = {}
    original_text_by_key: Dict[str, str] = {}
    for idx, record in enumerate(records):
        if not record.text:
            continue
        key = str(record.uid or record.name or f"record_{idx + 1}")
        normalized = _normalize_label(spec.target.value(record), spec.target.mode)
        if normalized is None:
            continue
        label_by_key[key] = normalized
        original_text_by_key[key] = record.text

    train_keys = list(split_keys.get("train", []))
    val_keys = list(split_keys.get("val", []))
    test_keys = list(split_keys.get("test", []))
    if not train_keys or not test_keys:
        raise ValueError("fixed split requires non-empty train and test splits")

    train_keys, original_train_texts, original_train_labels = _subset_by_keys(train_keys, original_text_by_key, label_by_key)
    val_keys, original_val_texts, original_val_labels = _subset_by_keys(val_keys, original_text_by_key, label_by_key)
    test_keys, original_test_texts, original_test_labels = _subset_by_keys(test_keys, original_text_by_key, label_by_key)
    all_keys_order = [key for key in (train_keys + val_keys + test_keys) if key in label_by_key and key in original_text_by_key]
    all_texts = [original_text_by_key[key] for key in all_keys_order]
    all_labels = [label_by_key[key] for key in all_keys_order]
    if not original_train_texts or not original_test_texts:
        raise ValueError("not enough labeled records in fixed train/test splits")

    selected_head_kwargs, selected_val_metrics, baseline_eval, tuning_trials = _pick_tuned_head_kwargs(
        base_head_kwargs=head_kwargs,
        tune_param=config.tune_param,
        tune_values=config.tune_values,
        spec=spec,
        vectorizer_name=vectorizer_name,
        vectorizer_kwargs=vectorizer_kwargs,
        head_name=head_name,
        identifier=identifier,
        primary_metric=primary_metric,
        train_texts=original_train_texts,
        train_labels=original_train_labels,
        val_texts=original_val_texts,
        val_labels=original_val_labels,
        test_texts=original_test_texts,
        test_labels=original_test_labels,
        overall_texts=all_texts,
        overall_labels=all_labels,
    )

    baseline_train_metrics = dict(baseline_eval.get("train", {}))
    baseline_val_metrics = dict(baseline_eval.get("val", {}))
    baseline_test_metrics = dict(baseline_eval.get("test", {}))
    baseline_overall_metrics = dict(baseline_eval.get("overall", {}))

    if evaluation_texts is None and (evaluation_sources is None or index_to_key is None):
        raise ValueError("evaluation input is required")

    evaluations: Dict[str, Dict[str, Any]] = {}
    primary_scores: List[float] = []
    shared_train_metrics: Dict[str, float] = {}
    shared_val_metrics: Dict[str, float] = {}
    shared_vectorizer: Optional[Any] = None
    shared_model: Optional[Any] = None

    selected_keys = set(label_by_key.keys())

    def _iter_evaluations() -> Iterator[Tuple[str, Dict[str, str]]]:
        if evaluation_texts is not None:
            for dataset_name in sorted(evaluation_texts.keys()):
                yield dataset_name, evaluation_texts[dataset_name]
            return
        if evaluation_sources is None or index_to_key is None:
            return
        yield from iter_utility_evaluation_texts(index_to_key, evaluation_sources, selected_keys=selected_keys)

    if protocol == "supervised_divergence":
        shared_vectorizer, shared_model = spec.build_components(
            vectorizer_name=vectorizer_name,
            vectorizer_kwargs=vectorizer_kwargs,
            head_name=head_name,
            head_kwargs=selected_head_kwargs,
            identifier=identifier,
        )
        if hasattr(shared_model, "set_label_order") and spec.target.label_order:
            shared_model.set_label_order(spec.target.label_order)
        shared_model.setup()
        shared_vectorizer.fit(list(original_train_texts))
        shared_x_train = shared_vectorizer.transform(list(original_train_texts))
        shared_x_val = shared_vectorizer.transform(list(original_val_texts)) if original_val_texts else shared_x_train
        shared_y_val = list(original_val_labels) if original_val_labels else list(original_train_labels)
        _fit_model(shared_model, shared_x_train, list(original_train_labels), shared_x_val, shared_y_val)
        shared_train_metrics = {k: float(v) for k, v in shared_model.evaluate(shared_x_train, list(original_train_labels)).items()}
        shared_val_metrics = {k: float(v) for k, v in shared_model.evaluate(shared_x_val, shared_y_val).items()}

    try:
        for name, mapping in _iter_evaluations():
            train_keys_a, anon_train_texts, anon_train_labels = _subset_by_keys(train_keys, mapping, label_by_key)
            val_keys_a, anon_val_texts, anon_val_labels = _subset_by_keys(val_keys, mapping, label_by_key)
            test_keys_a, anon_test_texts, anon_test_labels = _subset_by_keys(test_keys, mapping, label_by_key)
            all_keys_a = [key for key in (train_keys_a + val_keys_a + test_keys_a) if key in mapping and key in label_by_key]
            anon_all_texts = [mapping[key] for key in all_keys_a]
            anon_all_labels = [label_by_key[key] for key in all_keys_a]

            if protocol == "supervised_divergence":
                if shared_vectorizer is None or shared_model is None:
                    raise RuntimeError("supervised_divergence model is not initialized")
                train_metrics = dict(shared_train_metrics)
                val_metrics = dict(shared_val_metrics)
                if anon_test_texts:
                    x_test = shared_vectorizer.transform(list(anon_test_texts))
                    test_metrics = {k: float(v) for k, v in shared_model.evaluate(x_test, list(anon_test_labels)).items()}
                else:
                    test_metrics = {}
                if anon_all_texts:
                    x_overall = shared_vectorizer.transform(list(anon_all_texts))
                    overall_metrics = {k: float(v) for k, v in shared_model.evaluate(x_overall, list(anon_all_labels)).items()}
                else:
                    overall_metrics = {}
            else:
                if not anon_train_texts or not anon_test_texts:
                    evaluations[name] = {
                        "metrics": {},
                        "drops": {},
                        "train_matched": len(anon_train_texts),
                        "train_total": len(train_keys),
                        "test_matched": len(anon_test_texts),
                        "test_total": len(test_keys),
                        "available": len(anon_all_texts),
                        "valid": False,
                        "train_results": {"metrics": {}, "drops": {}, "matched": len(anon_train_texts), "total": len(train_keys)},
                        "val_results": {"metrics": {}, "drops": {}, "matched": len(anon_val_texts), "total": len(val_keys)},
                        "test_results": {"metrics": {}, "drops": {}, "matched": len(anon_test_texts), "total": len(test_keys)},
                        "overall_results": {"metrics": {}, "drops": {}, "matched": len(anon_all_texts), "total": len(all_keys_order)},
                        "grouped_results": {},
                        "utility": {"error": "missing_anonymized_split_records"},
                    }
                    continue
                eval_result = _fit_and_eval_single(
                    spec=spec,
                    vectorizer_name=vectorizer_name,
                    vectorizer_kwargs=vectorizer_kwargs,
                    head_name=head_name,
                    head_kwargs=selected_head_kwargs,
                    identifier=identifier,
                    train_texts=anon_train_texts,
                    train_labels=anon_train_labels,
                    val_texts=anon_val_texts if anon_val_texts else original_val_texts,
                    val_labels=anon_val_labels if anon_val_labels else original_val_labels,
                    test_texts=anon_test_texts,
                    test_labels=anon_test_labels,
                    overall_texts=anon_all_texts,
                    overall_labels=anon_all_labels,
                )
                train_metrics = dict(eval_result.get("train", {}))
                val_metrics = dict(eval_result.get("val", {}))
                test_metrics = dict(eval_result.get("test", {}))
                overall_metrics = dict(eval_result.get("overall", {}))

            metrics_primary = dict(test_metrics)
            drops_primary = _score_difference(baseline_test_metrics, metrics_primary) if metrics_primary else {}
            if primary_metric and primary_metric in metrics_primary:
                primary_scores.append(float(metrics_primary[primary_metric]))

            evaluations[name] = {
                "metrics": metrics_primary,
                "drops": drops_primary,
                "train_matched": len(train_keys) if protocol == "supervised_divergence" else len(anon_train_texts),
                "train_total": len(train_keys),
                "test_matched": len(anon_test_texts),
                "test_total": len(test_keys),
                "available": len(anon_all_texts),
                "valid": bool(metrics_primary),
                "train_results": {
                    "metrics": train_metrics,
                    "drops": _score_difference(baseline_train_metrics, train_metrics),
                    "matched": len(train_keys) if protocol == "supervised_divergence" else len(anon_train_texts),
                    "total": len(train_keys),
                },
                "val_results": {
                    "metrics": val_metrics,
                    "drops": _score_difference(baseline_val_metrics, val_metrics),
                    "matched": len(val_keys) if protocol == "supervised_divergence" else len(anon_val_texts),
                    "total": len(val_keys),
                },
                "test_results": {
                    "metrics": test_metrics,
                    "drops": _score_difference(baseline_test_metrics, test_metrics),
                    "matched": len(anon_test_texts),
                    "total": len(test_keys),
                },
                "overall_results": {
                    "metrics": overall_metrics,
                    "drops": _score_difference(baseline_overall_metrics, overall_metrics),
                    "matched": len(anon_all_texts),
                    "total": len(all_keys_order),
                },
                "grouped_results": {},
                "utility": {
                    "protocol": str(protocol),
                    "split_mode": "fixed_train_val_test",
                },
            }
    finally:
        if shared_model is not None:
            try:
                shared_model.cleanup()
            finally:
                if shared_vectorizer is not None:
                    shared_vectorizer.cleanup()
        elif shared_vectorizer is not None:
            shared_vectorizer.cleanup()

    if not evaluations:
        raise ValueError("no anonymized texts aligned with dataset records")

    metrics_payload: Dict[str, Any] = {
        "model": str(model_name),
        "primary_metric": str(primary_metric),
        "baseline": {
            "metrics": dict(baseline_test_metrics),
            "train_size": len(train_keys),
            "test_size": len(test_keys),
            "train_metrics": dict(baseline_train_metrics),
            "test_metrics": dict(baseline_test_metrics),
            "overall_metrics": dict(baseline_overall_metrics),
            "dummy": {},
            "median_dummy_mae": {},
            "val_metrics": dict(baseline_val_metrics),
            "selected_head_params": dict(selected_head_kwargs),
            "selected_val_metrics": dict(selected_val_metrics),
            "tuning_trials": tuning_trials,
        },
        "evaluations": evaluations,
        "records": {},
        "protocol": str(protocol),
        "utility": {
            "split_mode": "fixed_train_val_test",
            "random_state": int(config.random_state),
            "tune_param": config.tune_param,
            "tune_values": list(config.tune_values or []),
        },
    }
    score = float(np.mean(primary_scores)) if primary_scores else 0.0
    return ExperimentResult(score=score, metrics=metrics_payload)
