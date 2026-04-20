from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import f1_score, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

from dp.loaders.base import DatasetRecord
from dp.experiments import Experiment, ExperimentResult
from dp.experiments.utility.vectorizer import SelfSupervisedFeatureExtractor
from dp.bert import SupervisedDownstreamHead

class UtilityTarget:
    class Mode(Enum):
        BINARY = "binary"
        NOMINAL = "nominal"
        ORDINAL = "ordinal"
        CARDINAL = "cardinal"

    def __init__(self, name: str, source: str, mode: "UtilityTarget.Mode | str", getter: Callable[[DatasetRecord], Any], label_order: Optional[List[str]] = None):
        if not name:
            raise ValueError("target name is required")
        if not source:
            raise ValueError("target source is required")
        if not callable(getter):
            raise ValueError("target getter is required")
        self.name = name
        self.source = source
        if isinstance(mode, UtilityTarget.Mode):
            self.mode = mode
        else:
            self.mode = UtilityTarget.Mode(mode)
        self.getter = getter
        self.label_order = label_order

    def value(self, record: DatasetRecord) -> Any:
        return self.getter(record)


def split_indices(
    size: int,
    test_size: float,
    random_state: int,
    labels: Sequence[Any],
    stratify: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    if size < 2:
        raise ValueError("at least two records required")
    if not 0 <= test_size <= 1:
        raise ValueError("test_size must be between 0 and 1 inclusive")
    indices = np.arange(size)
    if test_size == 0:
        return indices, np.array([], dtype=int)
    if stratify and len(set(labels)) > 1 and 0 < test_size < 1:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(indices, labels))
        return train_idx, test_idx
    rng = np.random.default_rng(random_state)
    rng.shuffle(indices)
    split = int(round(size * (1.0 - test_size)))
    if split <= 0:
        split = 1
    if split >= size:
        split = size - 1
    train_idx = indices[:split]
    test_idx = indices[split:]
    return train_idx, test_idx


class TextUtilityExperiment(Experiment):
    def __init__(
        self,
        test_size: float = 0.0,
        random_state: int = 42,
        include_confusion_matrix: bool = False,
    ):
        super().__init__()
        if not 0 <= test_size <= 1:
            raise ValueError("test_size must be between 0 and 1 inclusive")
        self.test_size = test_size
        self.random_state = random_state
        self.include_confusion_matrix = include_confusion_matrix
        self._vectorizer: Optional[SelfSupervisedFeatureExtractor] = None
        self._model: Optional[SupervisedDownstreamHead] = None
        self._target: Optional[UtilityTarget] = None
        self._records: List[DatasetRecord] = []
        self._keys: List[str] = []
        self._labels: List[Any] = []
        self._train_idx: Optional[np.ndarray] = None
        self._test_idx: Optional[np.ndarray] = None
        self._baseline_metrics: Optional[Dict[str, float]] = None
        self._baseline_train_metrics: Optional[Dict[str, float]] = None
        self._baseline_overall_metrics: Optional[Dict[str, float]] = None
        self._label_by_key: Dict[str, Any] = {}
        self._train_keys: List[str] = []
        self._test_keys: List[str] = []
        self._train_key_set: set[str] = set()
        self._test_key_set: set[str] = set()
        self._all_key_set: set[str] = set()
        self._record_info: Dict[str, Dict[str, Any]] = {}
        self._train_override_texts: Optional[Sequence[str]] = None
        self._train_override_labels: Optional[Sequence[Any]] = None
        self._group_by: Optional[str] = None
        self._group_keys: Dict[str, List[str]] = {}
        self._baseline_group_metrics: Dict[str, Dict[str, float]] = {}
        self._median_dummy_mae_train: Optional[float] = None
        self._median_dummy_mae_test: Optional[float] = None
        self._median_dummy_mae_overall: Optional[float] = None
        self._median_dummy_mae_group: Dict[str, float] = {}
        self._dummy_baseline_train_metrics: Dict[str, float] = {}
        self._dummy_baseline_test_metrics: Dict[str, float] = {}
        self._dummy_baseline_overall_metrics: Dict[str, float] = {}
        self._dummy_baseline_group_metrics: Dict[str, Dict[str, float]] = {}

    def setup(
        self,
        target: UtilityTarget,
        records: Sequence[DatasetRecord],
        vectorizer: SelfSupervisedFeatureExtractor,
        model: SupervisedDownstreamHead,
        train_texts_override: Optional[Sequence[str]] = None,
        train_labels_override: Optional[Sequence[Any]] = None,
        test_texts_override: Optional[Sequence[str]] = None,
        test_labels_override: Optional[Sequence[Any]] = None,
        maybe_group_mapping: Optional[Dict[str, Any]] = None,
        group_by: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if not records:
            raise ValueError("records cannot be empty")
        self._target = target
        self._vectorizer = vectorizer
        self._model = model
        if hasattr(self._model, "set_label_order") and target.label_order:
            self._model.set_label_order(target.label_order)
        self._model.setup()
        filtered_records: List[DatasetRecord] = []
        keys: List[str] = []
        labels: List[Any] = []
        for index, record in enumerate(records):
            if not record.text:
                continue
            key = record.uid or f"record_{index + 1}"
            value = target.value(record)
            normalized = self._normalize_label(value, target.mode)
            if normalized is None:
                continue
            filtered_records.append(record)
            keys.append(key)
            labels.append(normalized)
        if len(filtered_records) < 2:
            raise ValueError("not enough records for experiment")
        self._records = filtered_records
        self._keys = keys
        self._labels = labels
        self._group_by = str(group_by) if group_by else None
        
        has_train_override = (
            train_texts_override is not None
            and train_labels_override is not None
            and len(train_texts_override) == len(train_labels_override)
            and len(train_texts_override) > 0
        )
        has_test_override = (
            test_texts_override is not None
            and test_labels_override is not None
            and len(test_texts_override) == len(test_labels_override)
            and len(test_texts_override) > 0
        )
        
        if has_train_override and not has_test_override:
            self._train_idx = np.array([], dtype=int)
            self._test_idx = np.arange(len(self._records), dtype=int)
        elif has_test_override and not has_train_override:
            self._train_idx = np.arange(len(self._records), dtype=int)
            self._test_idx = np.array([], dtype=int)
        elif has_train_override and has_test_override:
            self._train_idx = np.array([], dtype=int)
            self._test_idx = np.array([], dtype=int)
        else:
            self._train_idx, self._test_idx = split_indices(
                size=len(self._records),
                test_size=self.test_size,
                random_state=self.random_state,
                labels=self._labels,
                stratify=target.mode is not UtilityTarget.Mode.CARDINAL,
            )
        self._train_keys = [self._keys[i] for i in self._train_idx]
        self._test_keys = [self._keys[i] for i in self._test_idx]
        self._train_key_set = set(self._train_keys)
        self._test_key_set = set(self._test_keys)
        self._all_key_set = set(self._keys)
        self._train_texts = [self._records[i].text for i in self._train_idx]
        self._test_texts = [self._records[i].text for i in self._test_idx]
        self._train_labels = [self._labels[i] for i in self._train_idx]
        self._test_labels = [self._labels[i] for i in self._test_idx]
        
        actual_train_texts = list(train_texts_override) if has_train_override else self._train_texts
        actual_train_labels = list(train_labels_override) if has_train_override else self._train_labels
        actual_test_texts = list(test_texts_override) if has_test_override else self._test_texts
        actual_test_labels = list(test_labels_override) if has_test_override else self._test_labels
        
        if has_train_override:
            self._train_override_texts = actual_train_texts
            self._train_override_labels = actual_train_labels
        
        if not actual_train_texts:
            raise ValueError("no training data available")
        
        self._vectorizer.fit(actual_train_texts)
        self._x_train = self._vectorizer.transform(actual_train_texts)
        
        if actual_test_texts:
            self._model.fit(actual_train_texts, actual_train_labels, actual_test_texts, actual_test_labels)
        else:
            self._model.fit(self._x_train, actual_train_labels)
        
        baseline_train_raw = self._model.evaluate(self._x_train, actual_train_labels)
        self._median_dummy_mae_train = self._compute_median_dummy_mae(actual_train_labels)

        if actual_test_texts:
            x_test = self._vectorizer.transform(actual_test_texts)
            baseline_test_raw = self._model.evaluate(x_test, actual_test_labels)
            self._median_dummy_mae_test = self._compute_median_dummy_mae(actual_test_labels)
        else:
            baseline_test_raw = {}
            self._median_dummy_mae_test = None

        all_texts = [r.text for r in self._records]
        all_labels = list(self._labels)
        x_all_eval = self._vectorizer.transform(all_texts)
        baseline_overall_raw = self._model.evaluate(x_all_eval, all_labels)
        self._median_dummy_mae_overall = self._compute_median_dummy_mae(all_labels)
        self._baseline_train_metrics = {k: float(v) for k, v in baseline_train_raw.items()}
        self._baseline_metrics = {k: float(v) for k, v in baseline_test_raw.items()} if baseline_test_raw else {}
        self._baseline_overall_metrics = self._with_utility_metrics(baseline_overall_raw, self._median_dummy_mae_overall)
        self._dummy_baseline_train_metrics = self._compute_dummy_metrics(actual_train_labels, actual_train_labels)
        self._dummy_baseline_test_metrics = self._compute_dummy_metrics(actual_train_labels, actual_test_labels) if actual_test_labels else {}
        self._dummy_baseline_overall_metrics = self._compute_dummy_metrics(actual_train_labels, all_labels)
        self._label_by_key = {key: self._labels[idx] for idx, key in enumerate(self._keys)}
        self._record_info = {}
        self._group_keys = {}
        for idx, key in enumerate(self._keys):
            row = {
                "index": idx + 1,
                "split": "train" if key in self._train_key_set else "test",
                "label": self._labels[idx],
            }
            if maybe_group_mapping is not None and key in maybe_group_mapping:
                group_value = maybe_group_mapping[key]
                group_key = str(group_value)
                row["group"] = group_key
                if self._group_by:
                    row["group_by"] = self._group_by
                self._group_keys.setdefault(group_key, []).append(key)
            self._record_info[key] = row
        self._baseline_group_metrics = {}
        if self._group_keys:
            key_to_index = {key: idx for idx, key in enumerate(self._keys)}
            for group_key, group_keys in self._group_keys.items():
                if not group_keys:
                    continue
                group_indices = [key_to_index[key] for key in group_keys if key in key_to_index]
                group_texts = [self._records[i].text for i in group_indices]
                group_labels = [self._labels[i] for i in group_indices]
                x_group = self._vectorizer.transform(group_texts)
                group_metrics_raw = self._model.evaluate(x_group, group_labels)
                group_median_dummy_mae = self._compute_median_dummy_mae(group_labels)
                if group_median_dummy_mae is not None:
                    self._median_dummy_mae_group[group_key] = group_median_dummy_mae
                self._baseline_group_metrics[group_key] = {k: float(v) for k, v in group_metrics_raw.items()}
                self._dummy_baseline_group_metrics[group_key] = self._compute_dummy_metrics(group_labels, group_labels)
        super().setup(**kwargs)

    def run(self, evaluation_texts: Dict[str, Dict[str, str]], **kwargs: Any) -> ExperimentResult:
        if self._model is None or self._vectorizer is None or self._baseline_metrics is None or self._target is None:
            raise RuntimeError("setup must be completed before run")
        evaluations: Dict[str, Dict[str, Any]] = {}
        for name, mapping in tqdm(evaluation_texts.items(), desc="Evaluating datasets"):
            def _eval_subset(keys_subset: Sequence[str]) -> Tuple[Dict[str, float], int, Optional[List[List[int]]]]:
                subset_keys = [key for key in keys_subset if key in mapping]
                if not subset_keys:
                    return {}, 0, None
                texts = [mapping[key] for key in subset_keys]
                labels = [self._label_by_key[key] for key in subset_keys]
                x_eval = self._vectorizer.transform(texts)
                metrics = self._model.evaluate(x_eval, labels)
                cm = None
                if self.include_confusion_matrix:
                    from sklearn.metrics import confusion_matrix
                    preds = self._model.predict(x_eval)
                    label_order = getattr(self._model, "_label_order", None)
                    if not label_order:
                        label_order = sorted(list(set(self._labels)))
                    cm_array = confusion_matrix(labels, preds, labels=label_order)
                    cm = cm_array.tolist()
                return metrics, len(subset_keys), cm

            if not self._test_keys:
                overall_metrics_raw, overall_matched, overall_cm = _eval_subset(list(self._all_key_set))
                overall_metrics = self._with_utility_metrics(overall_metrics_raw, self._median_dummy_mae_overall)
                metrics = dict(overall_metrics)
                drops = self._score_difference(self._baseline_overall_metrics or {}, metrics) if metrics else {}
                overall_res = {
                    "metrics": overall_metrics,
                    "drops": self._score_difference(self._baseline_overall_metrics or {}, overall_metrics) if overall_metrics else {},
                    "matched": overall_matched,
                    "total": len(self._keys),
                }
                if overall_cm is not None:
                    overall_res["confusion_matrix"] = overall_cm
                evaluations[name] = {
                    "metrics": metrics,
                    "drops": drops,
                    "train_matched": 0,
                    "train_total": len(self._train_keys),
                    "test_matched": 0,
                    "test_total": len(self._test_keys),
                    "available": overall_matched,
                    "valid": bool(overall_matched),
                    "train_results": {"metrics": {}, "drops": {}, "matched": 0, "total": len(self._train_keys)},
                    "test_results": {"metrics": {}, "drops": {}, "matched": 0, "total": len(self._test_keys)},
                    "overall_results": overall_res,
                    "val_results": {"metrics": {}, "drops": {}, "matched": 0, "total": 0},
                }
            else:
                train_metrics_raw, train_matched, train_cm = _eval_subset(list(self._train_key_set))
                test_metrics_raw, test_matched, test_cm = _eval_subset(list(self._test_key_set))
                overall_metrics_raw, overall_matched, overall_cm = _eval_subset(list(self._all_key_set))
                train_metrics = {k: float(v) for k, v in train_metrics_raw.items()}
                test_metrics = {k: float(v) for k, v in test_metrics_raw.items()}
                overall_metrics = self._with_utility_metrics(overall_metrics_raw, self._median_dummy_mae_overall)

                metrics = dict(test_metrics) if test_metrics else dict(overall_metrics)
                drops = self._score_difference(self._baseline_metrics, metrics) if metrics else {}
                train_drops = self._score_difference(self._baseline_train_metrics or {}, train_metrics) if train_metrics else {}
                overall_drops = self._score_difference(self._baseline_overall_metrics or {}, overall_metrics) if overall_metrics else {}

                train_res = {
                    "metrics": train_metrics,
                    "drops": train_drops,
                    "matched": train_matched,
                    "total": len(self._train_keys),
                }
                if train_cm is not None:
                    train_res["confusion_matrix"] = train_cm
                test_res = {
                    "metrics": test_metrics,
                    "drops": drops,
                    "matched": test_matched,
                    "total": len(self._test_keys),
                }
                if test_cm is not None:
                    test_res["confusion_matrix"] = test_cm
                overall_res = {
                    "metrics": overall_metrics,
                    "drops": overall_drops,
                    "matched": overall_matched,
                    "total": len(self._keys),
                }
                if overall_cm is not None:
                    overall_res["confusion_matrix"] = overall_cm

                evaluations[name] = {
                    "metrics": metrics,
                    "drops": drops,
                    "train_matched": train_matched,
                    "train_total": len(self._train_keys),
                    "test_matched": test_matched,
                    "test_total": len(self._test_keys),
                    "available": overall_matched,
                    "valid": bool(test_matched or train_matched or overall_matched),
                    "train_results": train_res,
                    "test_results": test_res,
                    "overall_results": overall_res,
                    "val_results": {
                        "metrics": {},
                        "drops": {},
                        "matched": 0,
                        "total": 0,
                    },
                }
            grouped_results: Dict[str, Dict[str, Any]] = {}
            if self._group_keys:
                for group_key, group_keys in sorted(self._group_keys.items(), key=lambda item: item[0]):
                    group_metrics_raw, group_matched, group_cm = _eval_subset(group_keys)
                    baseline_group = self._baseline_group_metrics.get(group_key, {})
                    group_metrics = {k: float(v) for k, v in group_metrics_raw.items()}
                    group_drops = {}
                    if group_metrics:
                        group_drops = self._score_difference(baseline_group, group_metrics)
                    group_payload: Dict[str, Any] = {
                        "metrics": group_metrics,
                        "drops": group_drops,
                        "matched": group_matched,
                        "total": len(group_keys),
                        "valid": bool(group_matched),
                    }
                    if self._group_by:
                        group_payload["group_by"] = self._group_by
                    if group_cm is not None:
                        group_payload["confusion_matrix"] = group_cm
                    grouped_results[group_key] = group_payload
            evaluations[name]["grouped_results"] = grouped_results
        has_test = bool(self._test_keys)
        baseline_primary = self._baseline_metrics if has_test else (self._baseline_overall_metrics or {})
        baseline_test_metrics = self._baseline_metrics if has_test else {}
        metrics_payload: Dict[str, Any] = {
            "model": self._model.name,
            "primary_metric": self._model.primary_metric,
            "baseline": {
                "metrics": baseline_primary,
                "train_size": len(self._train_keys),
                "test_size": len(self._test_keys),
                "train_metrics": self._baseline_train_metrics or {},
                "test_metrics": baseline_test_metrics,
                "overall_metrics": self._baseline_overall_metrics or {},
                "dummy": {
                    "strategy": self._dummy_strategy_name(),
                    "train_metrics": self._dummy_baseline_train_metrics,
                    "test_metrics": self._dummy_baseline_test_metrics,
                    "overall_metrics": self._dummy_baseline_overall_metrics,
                    "group_metrics": self._dummy_baseline_group_metrics,
                },
                "median_dummy_mae": {
                    "train": self._median_dummy_mae_train,
                    "test": self._median_dummy_mae_test,
                    "overall": self._median_dummy_mae_overall,
                    "group": self._median_dummy_mae_group,
                },
            },
            "evaluations": evaluations,
            "records": self._record_info,
        }
        if self._group_by:
            metrics_payload["group_by"] = self._group_by
        if self._baseline_group_metrics:
            metrics_payload["group_baseline"] = self._baseline_group_metrics
        score = float(self._baseline_metrics.get(self._model.primary_metric, 0.0))
        return ExperimentResult(score=score, metrics=metrics_payload)

    def cleanup(self) -> None:
        if self._model is not None:
            self._model.cleanup()
        self._vectorizer = None
        self._model = None
        self._target = None
        self._records = []
        self._keys = []
        self._labels = []
        self._train_idx = None
        self._test_idx = None
        self._baseline_metrics = None
        self._label_by_key = {}
        self._train_keys = []
        self._test_keys = []
        self._train_key_set = set()
        self._test_key_set = set()
        self._record_info = {}
        self._group_by = None
        self._group_keys = {}
        self._baseline_group_metrics = {}
        self._median_dummy_mae_train = None
        self._median_dummy_mae_test = None
        self._median_dummy_mae_overall = None
        self._median_dummy_mae_group = {}
        self._dummy_baseline_train_metrics = {}
        self._dummy_baseline_test_metrics = {}
        self._dummy_baseline_overall_metrics = {}
        self._dummy_baseline_group_metrics = {}
        super().cleanup()

    def _clone_vectorizer(self) -> SelfSupervisedFeatureExtractor:
        if not self._vectorizer:
            raise RuntimeError("vectorizer is not initialized")
        return self._vectorizer

    def _normalize_label(self, value: Any, mode: UtilityTarget.Mode) -> Optional[Any]:
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

    def _score_difference(self, baseline: Dict[str, float], current: Dict[str, float]) -> Dict[str, float]:
        drops: Dict[str, float] = {}
        for name, value in baseline.items():
            if name in current:
                drops[f"{name}_drop"] = float(value - current[name])
        return drops

    def _with_utility_metrics(self, metrics: Dict[str, float], median_dummy_mae: Optional[float]) -> Dict[str, float]:
        return {k: float(v) for k, v in metrics.items()}

    def _compute_median_dummy_mae(self, labels: Sequence[Any]) -> Optional[float]:
        if self._target is None or self._target.mode is not UtilityTarget.Mode.ORDINAL:
            return None
        if not labels:
            return None
        label_order = self._target.label_order
        if not label_order:
            raise ValueError("ordinal target requires label_order")
        label_to_rank = {str(label): index for index, label in enumerate(label_order)}
        encoded: List[float] = []
        for label in labels:
            key = str(label)
            if key not in label_to_rank:
                raise ValueError(f"Unknown ordinal label '{key}' not in target enum")
            encoded.append(float(label_to_rank[key]))
        values = np.asarray(encoded, dtype=float)
        median_value = float(np.median(values))
        mae = float(np.mean(np.abs(values - median_value)))
        if mae <= 0:
            return None
        return mae

    def _dummy_strategy_name(self) -> str:
        if self._target is None:
            return "unknown"
        if self._target.mode in {UtilityTarget.Mode.BINARY, UtilityTarget.Mode.NOMINAL}:
            return "mode"
        if self._target.mode is UtilityTarget.Mode.ORDINAL:
            return "median"
        if self._target.mode is UtilityTarget.Mode.CARDINAL:
            return "mean"
        return "unknown"

    def _compute_dummy_metrics(self, fit_labels: Sequence[Any], eval_labels: Sequence[Any]) -> Dict[str, float]:
        if self._target is None:
            return {}
        if not fit_labels or not eval_labels:
            return {}
        mode = self._target.mode
        if mode in {UtilityTarget.Mode.BINARY, UtilityTarget.Mode.NOMINAL}:
            fit_values = [str(v) for v in fit_labels]
            eval_values = [str(v) for v in eval_labels]
            counts: Dict[str, int] = {}
            first_seen: Dict[str, int] = {}
            for idx, value in enumerate(fit_values):
                counts[value] = counts.get(value, 0) + 1
                if value not in first_seen:
                    first_seen[value] = idx
            best_label = min(counts.keys(), key=lambda key: (-counts[key], first_seen[key]))
            preds = [best_label] * len(eval_values)
            unique_eval = sorted(set(eval_values))
            macro_f1 = float(f1_score(eval_values, preds, average="macro", zero_division=0))
            acc = float(np.mean(np.asarray(eval_values, dtype=object) == np.asarray(preds, dtype=object)))
            per_class_recalls: List[float] = []
            for label in unique_eval:
                mask = np.asarray([v == label for v in eval_values], dtype=bool)
                if int(mask.sum()) <= 0:
                    continue
                pred_arr = np.asarray(preds, dtype=object)
                eval_arr = np.asarray(eval_values, dtype=object)
                per_class_recalls.append(float((pred_arr[mask] == eval_arr[mask]).mean()))
            balanced_acc = float(np.mean(per_class_recalls)) if per_class_recalls else 0.0
            return {
                "macro_f1": macro_f1,
                "f1": macro_f1,
                "acc": acc,
                "balanced_acc": balanced_acc,
            }
        if mode is UtilityTarget.Mode.ORDINAL:
            label_order = self._target.label_order or []
            if not label_order:
                raise ValueError("ordinal target requires label_order")
            label_to_rank = {str(label): index for index, label in enumerate(label_order)}
            fit_encoded = np.asarray([label_to_rank[str(v)] for v in fit_labels], dtype=float)
            eval_encoded = np.asarray([label_to_rank[str(v)] for v in eval_labels], dtype=int)
            median_rank = int(np.rint(float(np.median(fit_encoded))))
            median_rank = max(0, min(median_rank, len(label_order) - 1))
            preds = np.full(eval_encoded.shape, median_rank, dtype=int)
            unique_classes = sorted(set(eval_encoded.tolist()))
            per_class_mae: List[float] = []
            per_class_recall: List[float] = []
            per_class_within1: List[float] = []
            for cls in unique_classes:
                mask = eval_encoded == cls
                if int(mask.sum()) <= 0:
                    continue
                abs_err = np.abs(eval_encoded[mask] - preds[mask])
                per_class_mae.append(float(abs_err.mean()))
                per_class_recall.append(float((eval_encoded[mask] == preds[mask]).mean()))
                per_class_within1.append(float((abs_err <= 1).mean()))
            abs_err_all = np.abs(eval_encoded - preds)
            return {
                "macro_mae": float(np.mean(per_class_mae)) if per_class_mae else float("inf"),
                "macro_within1": float(np.mean(per_class_within1)) if per_class_within1 else 0.0,
                "worst_recall": float(np.min(per_class_recall)) if per_class_recall else 0.0,
                "mae": float(abs_err_all.mean()),
                "acc": float((eval_encoded == preds).mean()),
                "within1": float((abs_err_all <= 1).mean()),
            }
        fit_vals = np.asarray([float(v) for v in fit_labels], dtype=float)
        eval_vals = np.asarray([float(v) for v in eval_labels], dtype=float)
        pred_value = float(np.mean(fit_vals))
        preds = np.full(eval_vals.shape, pred_value, dtype=float)
        mse = float(mean_squared_error(eval_vals, preds))
        try:
            r2 = float(r2_score(eval_vals, preds))
        except Exception:
            r2 = 0.0
        return {
            "rmse": float(np.sqrt(mse)),
            "r2": r2,
            "mae": float(np.mean(np.abs(eval_vals - preds))),
        }
