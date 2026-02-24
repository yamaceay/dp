from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from dp.experiments import ExperimentResult
from dp.experiments.utility.base import UtilityTarget
from dp.experiments.utility.models import UtilitySpec
from dp.loaders.base import DatasetRecord


@dataclass(frozen=True)
class InternalUtilityConfig:
    n_folds: int
    eval_fold_offset: int
    random_state: int
    max_rounds: Optional[int]


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


def _aggregate_metrics(values: Sequence[Dict[str, float]]) -> Dict[str, Any]:
    if not values:
        return {"mean": {}, "std": {}, "values": []}
    names: List[str] = sorted({name for row in values for name in row.keys()})
    mean: Dict[str, float] = {}
    std: Dict[str, float] = {}
    for name in names:
        arr = np.asarray([float(row[name]) for row in values if name in row], dtype=float)
        if arr.size == 0:
            continue
        mean[name] = float(arr.mean())
        std[name] = float(arr.std(ddof=0))
    return {
        "mean": mean,
        "std": std,
        "values": [{k: float(v) for k, v in row.items()} for row in values],
    }


def _build_balanced_folds(
    labels: Sequence[Any],
    mode: UtilityTarget.Mode,
    n_folds: int,
    random_state: int,
) -> List[List[int]]:
    if n_folds < 3:
        raise ValueError("internal_utility requires n_folds >= 3")
    rng = np.random.default_rng(random_state)
    folds: List[List[int]] = [[] for _ in range(n_folds)]
    indices = np.arange(len(labels), dtype=int)
    if mode is UtilityTarget.Mode.CARDINAL:
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        for i, idx in enumerate(shuffled.tolist()):
            folds[i % n_folds].append(int(idx))
        return folds
    labels_list = list(labels)
    labels_arr = np.asarray(labels_list, dtype=object)
    unique_labels = sorted({str(v) for v in labels_list})
    for label in unique_labels:
        class_indices = indices[np.asarray([str(v) == label for v in labels_arr], dtype=bool)].copy()
        rng.shuffle(class_indices)
        for i, idx in enumerate(class_indices.tolist()):
            folds[i % n_folds].append(int(idx))
    return folds


def _labels_subset_present(train_labels: Sequence[Any], other_labels: Sequence[Any]) -> bool:
    return set(other_labels).issubset(set(train_labels))


def _cv_rounds_from_folds(
    folds: Sequence[Sequence[int]],
    max_rounds: Optional[int],
) -> Tuple[List[Tuple[int, List[int]]], int]:
    non_empty = [idx for idx, fold in enumerate(folds) if fold]
    if len(non_empty) < 2:
        return [], 0
    rounds: List[Tuple[int, List[int]]] = []
    skipped = 0
    ordered = list(non_empty)
    for test_fold in ordered:
        train_folds = [f for f in ordered if f != test_fold]
        if not train_folds:
            skipped += 1
            continue
        rounds.append((test_fold, train_folds))
        if max_rounds is not None and len(rounds) >= max_rounds:
            break
    return rounds, skipped


def _cv_round_valid(
    *,
    train_idx: Sequence[int],
    test_idx: Sequence[int],
    labels: Sequence[Any],
    mode: UtilityTarget.Mode,
) -> bool:
    if not train_idx or not test_idx:
        return False
    train_labels = [labels[i] for i in train_idx]
    test_labels = [labels[i] for i in test_idx]
    if mode is not UtilityTarget.Mode.CARDINAL and len(set(train_labels)) < 2:
        return False
    if mode is UtilityTarget.Mode.CARDINAL:
        return True
    return _labels_subset_present(train_labels, test_labels)


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


def _evaluate_cv_arrays(
    *,
    spec: UtilitySpec,
    texts: Sequence[str],
    labels: Sequence[Any],
    vectorizer_name: Optional[str],
    vectorizer_kwargs: Dict[str, Any],
    head_name: Optional[str],
    head_kwargs: Dict[str, Any],
    identifier: Optional[str],
    config: InternalUtilityConfig,
) -> Dict[str, Any]:
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have equal length")
    if len(texts) < 3:
        return {
            "metrics": {},
            "train_results": {"metrics": {}, "std": {}, "fold_metrics": [], "matched": 0, "total": 0},
            "test_results": {"metrics": {}, "std": {}, "fold_metrics": [], "matched": 0, "total": 0},
            "overall_results": {"metrics": {}, "std": {}, "fold_metrics": [], "matched": len(texts), "total": len(texts)},
            "train_matched": 0,
            "test_matched": 0,
            "valid": False,
            "internal_utility": {"error": "not_enough_available_records"},
        }
    folds = _build_balanced_folds(
        labels=labels,
        mode=spec.target.mode,
        n_folds=config.n_folds,
        random_state=config.random_state,
    )
    rounds, skipped_rounds = _cv_rounds_from_folds(folds=folds, max_rounds=config.max_rounds)
    fold_rows: List[Dict[str, Any]] = []
    train_metrics_rows: List[Dict[str, float]] = []
    test_metrics_rows: List[Dict[str, float]] = []
    overall_metrics_rows: List[Dict[str, float]] = []
    train_sizes: List[int] = []
    test_sizes: List[int] = []
    texts_list = list(texts)
    labels_list = list(labels)
    for round_idx, (test_fold, train_folds) in enumerate(rounds):
        train_idx = [i for f in train_folds for i in folds[f]]
        test_idx = list(folds[test_fold])
        if not _cv_round_valid(
            train_idx=train_idx,
            test_idx=test_idx,
            labels=labels_list,
            mode=spec.target.mode,
        ):
            continue
        train_texts = [texts_list[i] for i in train_idx]
        test_texts = [texts_list[i] for i in test_idx]
        train_labels = [labels_list[i] for i in train_idx]
        test_labels = [labels_list[i] for i in test_idx]
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
            vectorizer.fit(train_texts)
            x_train = vectorizer.transform(train_texts)
            x_test = vectorizer.transform(test_texts)
            x_all = vectorizer.transform(texts_list)
            _fit_model(model, x_train, train_labels, None, None)
            train_metrics = {k: float(v) for k, v in model.evaluate(x_train, train_labels).items()}
            test_metrics = {k: float(v) for k, v in model.evaluate(x_test, test_labels).items()}
            overall_metrics = {k: float(v) for k, v in model.evaluate(x_all, labels_list).items()}
        finally:
            try:
                model.cleanup()
            finally:
                vectorizer.cleanup()
        train_metrics_rows.append(train_metrics)
        test_metrics_rows.append(test_metrics)
        overall_metrics_rows.append(overall_metrics)
        train_sizes.append(len(train_idx))
        test_sizes.append(len(test_idx))
        fold_rows.append(
            {
                "round": int(round_idx),
                "test_fold": int(test_fold),
                "train_folds": [int(f) for f in train_folds],
                "sizes": {
                    "train": len(train_idx),
                    "test": len(test_idx),
                    "overall": len(texts_list),
                },
                "metrics": {
                    "train": train_metrics,
                    "test": test_metrics,
                    "overall": overall_metrics,
                },
            }
        )
    train_agg = _aggregate_metrics(train_metrics_rows)
    test_agg = _aggregate_metrics(test_metrics_rows)
    overall_agg = _aggregate_metrics(overall_metrics_rows)
    valid_rounds = len(test_metrics_rows)
    mean_train = int(round(float(np.mean(train_sizes)))) if train_sizes else 0
    mean_test = int(round(float(np.mean(test_sizes)))) if test_sizes else 0
    metrics_mean = test_agg["mean"] if isinstance(test_agg.get("mean"), dict) else {}
    return {
        "metrics": {k: float(v) for k, v in metrics_mean.items()},
        "train_results": {
            "metrics": train_agg["mean"],
            "std": train_agg["std"],
            "fold_metrics": train_agg["values"],
            "matched": mean_train,
            "total": mean_train,
        },
        "test_results": {
            "metrics": test_agg["mean"],
            "std": test_agg["std"],
            "fold_metrics": test_agg["values"],
            "matched": mean_test,
            "total": mean_test,
        },
        "overall_results": {
            "metrics": overall_agg["mean"],
            "std": overall_agg["std"],
            "fold_metrics": overall_agg["values"],
            "matched": len(texts_list),
            "total": len(texts_list),
        },
        "train_matched": mean_train,
        "test_matched": mean_test,
        "valid": bool(valid_rounds),
        "internal_utility": {
            "protocol": "internal_utility",
            "cv_mode": "kfold_train_test",
            "n_folds_requested": int(config.n_folds),
            "n_folds_non_empty": int(sum(1 for fold in folds if fold)),
            "rounds_attempted": int(len(rounds) + skipped_rounds),
            "rounds_completed": int(valid_rounds),
            "rounds_skipped": int(skipped_rounds + max(0, len(rounds) - valid_rounds)),
            "folds": fold_rows,
        },
    }


def compute_internal_utility_baselines(
    *,
    spec: UtilitySpec,
    records: Sequence[DatasetRecord],
    evaluation_texts: Dict[str, Dict[str, str]],
    vectorizer_name: Optional[str],
    vectorizer_kwargs: Dict[str, Any],
    head_name: Optional[str],
    head_kwargs: Dict[str, Any],
    identifier: Optional[str],
    config: InternalUtilityConfig,
) -> Dict[str, Any]:
    filtered_keys: List[str] = []
    filtered_labels: List[Any] = []
    original_text_by_key: Dict[str, str] = {}
    for idx, record in enumerate(records):
        if not record.text:
            continue
        key = str(record.uid or f"record_{idx + 1}")
        value = spec.target.value(record)
        normalized = _normalize_label(value, spec.target.mode)
        if normalized is None:
            continue
        filtered_keys.append(key)
        filtered_labels.append(normalized)
        original_text_by_key[key] = record.text
    if len(filtered_keys) < 3:
        raise ValueError("not enough labeled records for internal_utility baseline")
    label_by_key = {k: filtered_labels[i] for i, k in enumerate(filtered_keys)}
    global_original_texts = [original_text_by_key[k] for k in filtered_keys]
    global_labels = [label_by_key[k] for k in filtered_keys]
    global_summary = _evaluate_cv_arrays(
        spec=spec,
        texts=global_original_texts,
        labels=global_labels,
        vectorizer_name=vectorizer_name,
        vectorizer_kwargs=vectorizer_kwargs,
        head_name=head_name,
        head_kwargs=head_kwargs,
        identifier=identifier,
        config=config,
    )
    per_evaluation: Dict[str, Dict[str, Any]] = {}
    for name in sorted(evaluation_texts.keys()):
        available_keys = [k for k in filtered_keys if k in evaluation_texts[name] and evaluation_texts[name][k]]
        original_texts = [original_text_by_key[k] for k in available_keys]
        labels = [label_by_key[k] for k in available_keys]
        per_evaluation[name] = _evaluate_cv_arrays(
            spec=spec,
            texts=original_texts,
            labels=labels,
            vectorizer_name=vectorizer_name,
            vectorizer_kwargs=vectorizer_kwargs,
            head_name=head_name,
            head_kwargs=head_kwargs,
            identifier=identifier,
            config=config,
        )
    return {
        "global": global_summary,
        "per_evaluation": per_evaluation,
    }


def run_internal_utility(
    *,
    spec: UtilitySpec,
    records: Sequence[DatasetRecord],
    evaluation_texts: Dict[str, Dict[str, str]],
    vectorizer_name: Optional[str],
    vectorizer_kwargs: Dict[str, Any],
    head_name: Optional[str],
    head_kwargs: Dict[str, Any],
    identifier: Optional[str],
    config: InternalUtilityConfig,
    model_name: str,
    primary_metric: str,
) -> ExperimentResult:
    filtered_keys: List[str] = []
    filtered_labels: List[Any] = []
    original_text_by_key: Dict[str, str] = {}
    for idx, record in enumerate(records):
        if not record.text:
            continue
        key = str(record.uid or f"record_{idx + 1}")
        value = spec.target.value(record)
        normalized = _normalize_label(value, spec.target.mode)
        if normalized is None:
            continue
        filtered_keys.append(key)
        filtered_labels.append(normalized)
        original_text_by_key[key] = record.text
    if len(filtered_keys) < 3:
        raise ValueError("not enough labeled records for internal_utility")
    label_by_key = {k: filtered_labels[i] for i, k in enumerate(filtered_keys)}
    baseline_bundle = compute_internal_utility_baselines(
        spec=spec,
        records=records,
        evaluation_texts=evaluation_texts,
        vectorizer_name=vectorizer_name,
        vectorizer_kwargs=vectorizer_kwargs,
        head_name=head_name,
        head_kwargs=head_kwargs,
        identifier=identifier,
        config=config,
    )
    global_baseline = baseline_bundle["global"]
    per_eval_baseline = baseline_bundle["per_evaluation"]
    evaluations: Dict[str, Dict[str, Any]] = {}
    primary_scores: List[float] = []
    for name in sorted(evaluation_texts.keys()):
        mapping = evaluation_texts[name]
        available_keys = [k for k in filtered_keys if k in mapping and mapping[k]]
        texts = [mapping[k] for k in available_keys]
        labels = [label_by_key[k] for k in available_keys]
        anon_summary = _evaluate_cv_arrays(
            spec=spec,
            texts=texts,
            labels=labels,
            vectorizer_name=vectorizer_name,
            vectorizer_kwargs=vectorizer_kwargs,
            head_name=head_name,
            head_kwargs=head_kwargs,
            identifier=identifier,
            config=config,
        )
        baseline_summary = per_eval_baseline.get(name, {})
        evaluation_primary = dict(anon_summary.get("metrics", {}) or {})
        evaluation_drops = _score_difference(
            baseline_summary.get("metrics", {}) or {},
            evaluation_primary,
        ) if evaluation_primary else {}
        if primary_metric and primary_metric in evaluation_primary:
            primary_scores.append(float(evaluation_primary[primary_metric]))
        train_results = dict(anon_summary.get("train_results", {}) or {})
        test_results = dict(anon_summary.get("test_results", {}) or {})
        overall_results = dict(anon_summary.get("overall_results", {}) or {})
        baseline_train = (baseline_summary.get("train_results", {}) or {}).get("metrics", {}) or {}
        baseline_test = (baseline_summary.get("test_results", {}) or {}).get("metrics", {}) or {}
        baseline_overall = (baseline_summary.get("overall_results", {}) or {}).get("metrics", {}) or {}
        train_results["drops"] = _score_difference(baseline_train, train_results.get("metrics", {}) or {})
        test_results["drops"] = _score_difference(baseline_test, test_results.get("metrics", {}) or {})
        overall_results["drops"] = _score_difference(baseline_overall, overall_results.get("metrics", {}) or {})
        train_results["baseline_metrics"] = baseline_train
        test_results["baseline_metrics"] = baseline_test
        overall_results["baseline_metrics"] = baseline_overall
        evaluations[name] = {
            "metrics": evaluation_primary,
            "drops": evaluation_drops,
            "train_matched": int(anon_summary.get("train_matched", 0)),
            "train_total": int(anon_summary.get("train_matched", 0)),
            "test_matched": int(anon_summary.get("test_matched", 0)),
            "test_total": int(anon_summary.get("test_matched", 0)),
            "available": len(texts),
            "valid": bool(anon_summary.get("valid", False)),
            "train_results": train_results,
            "val_results": {
                "metrics": {},
                "drops": {},
                "matched": 0,
                "total": 0,
            },
            "test_results": test_results,
            "overall_results": overall_results,
            "grouped_results": {},
            "internal_utility": dict(anon_summary.get("internal_utility", {}) or {}),
        }
    metrics_payload: Dict[str, Any] = {
        "model": str(model_name),
        "primary_metric": str(primary_metric),
        "baseline": {
            "metrics": dict(global_baseline.get("metrics", {}) or {}),
            "train_size": int(global_baseline.get("train_matched", 0)),
            "test_size": int(global_baseline.get("test_matched", 0)),
            "train_metrics": dict((global_baseline.get("train_results", {}) or {}).get("metrics", {}) or {}),
            "test_metrics": dict((global_baseline.get("test_results", {}) or {}).get("metrics", {}) or {}),
            "overall_metrics": dict((global_baseline.get("overall_results", {}) or {}).get("metrics", {}) or {}),
            "median_dummy_mae": {},
        },
        "evaluations": evaluations,
        "records": {},
        "protocol": "internal_utility",
        "internal_utility": {
            "n_folds": int(config.n_folds),
            "eval_fold_offset": int(config.eval_fold_offset),
            "random_state": int(config.random_state),
            "max_rounds": None if config.max_rounds is None else int(config.max_rounds),
        },
    }
    score = float(np.mean(primary_scores)) if primary_scores else 0.0
    return ExperimentResult(score=score, metrics=metrics_payload)
