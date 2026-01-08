from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

from dp.loaders.base import DatasetRecord
from dp.experiments import Experiment, ExperimentResult
from dp.experiments.utility.vectorizer import SelfSupervisedFeatureExtractor
from dp.experiments.utility.downstream import SupervisedDownstreamHead

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
    ):
        super().__init__()
        if not 0 <= test_size <= 1:
            raise ValueError("test_size must be between 0 and 1 inclusive")
        self.test_size = test_size
        self.random_state = random_state
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

    def setup(
        self,
        target: UtilityTarget,
        records: Sequence[DatasetRecord],
        vectorizer: SelfSupervisedFeatureExtractor,
        model: SupervisedDownstreamHead,
        train_texts_override: Optional[Sequence[str]] = None,
        train_labels_override: Optional[Sequence[Any]] = None,
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
        override_mode = (
            train_texts_override is not None
            and train_labels_override is not None
            and len(train_texts_override) == len(train_labels_override)
            and len(train_texts_override) > 0
        )
        if override_mode:
            size = len(self._records)
            if self.test_size <= 0:
                test_idx = np.arange(size, dtype=int)
            elif self.test_size >= 1:
                test_idx = np.arange(size, dtype=int)
            else:
                count = max(1, int(round(size * float(self.test_size))))
                indices = np.arange(size)
                if target.mode is not UtilityTarget.Mode.CARDINAL and len(set(self._labels)) > 1:
                    splitter = StratifiedShuffleSplit(n_splits=1, test_size=count, random_state=self.random_state)
                    try:
                        _, test_idx = next(splitter.split(indices, self._labels))
                    except Exception:
                        splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=float(self.test_size), random_state=self.random_state)
                        _, test_idx = next(splitter2.split(indices, self._labels))
                else:
                    rng = np.random.default_rng(self.random_state)
                    rng.shuffle(indices)
                    test_idx = indices[:count]
            self._train_idx = np.array([], dtype=int)
            self._test_idx = np.asarray(test_idx, dtype=int)
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
        if override_mode:
            self._train_override_texts = list(train_texts_override)  # type: ignore[arg-type]
            self._train_override_labels = list(train_labels_override)  # type: ignore[arg-type]
            self._vectorizer.fit(self._train_override_texts)
            self._x_train = self._vectorizer.transform(self._train_override_texts)
            if self._test_texts:
                self._model.fit(self._train_override_texts, self._train_override_labels, self._test_texts, self._test_labels)
            else:
                self._model.fit(self._x_train, self._train_override_labels)
        else:
            self._vectorizer.fit(self._train_texts)
            self._x_train = self._vectorizer.transform(self._train_texts)
            self._model.fit(self._x_train, self._train_labels)
        if self._test_texts:
            x_test = self._vectorizer.transform(self._test_texts)
            self._baseline_metrics = self._model.evaluate(x_test, self._test_labels)
        else:
            if self._train_texts:
                x_train_eval = self._vectorizer.transform(self._train_texts)
                self._baseline_metrics = self._model.evaluate(x_train_eval, self._train_labels)
            else:
                self._baseline_metrics = {}
        if self._train_texts:
            x_train_eval = self._vectorizer.transform(self._train_texts)
            self._baseline_train_metrics = self._model.evaluate(x_train_eval, self._train_labels)
        else:
            self._baseline_train_metrics = {}
        all_texts = [r.text for r in self._records]
        all_labels = list(self._labels)
        x_all_eval = self._vectorizer.transform(all_texts)
        self._baseline_overall_metrics = self._model.evaluate(x_all_eval, all_labels)
        self._label_by_key = {key: self._labels[idx] for idx, key in enumerate(self._keys)}
        self._record_info = {
            key: {
                "index": idx + 1,
                "split": "train" if key in self._train_key_set else "test",
                "label": self._labels[idx],
            }
            for idx, key in enumerate(self._keys)
        }
        super().setup(**kwargs)

    def run(self, evaluation_texts: Dict[str, Dict[str, str]], **kwargs: Any) -> ExperimentResult:
        if self._model is None or self._vectorizer is None or self._baseline_metrics is None or self._target is None:
            raise RuntimeError("setup must be completed before run")
        evaluations: Dict[str, Dict[str, Any]] = {}
        for name, mapping in tqdm(evaluation_texts.items(), desc="Evaluating datasets"):
            def _eval_subset(keys_subset: Sequence[str]) -> Tuple[Dict[str, float], int]:
                subset_keys = [key for key in keys_subset if key in mapping]
                if not subset_keys:
                    return {}, 0
                texts = [mapping[key] for key in subset_keys]
                labels = [self._label_by_key[key] for key in subset_keys]
                x_eval = self._vectorizer.transform(texts)
                metrics = self._model.evaluate(x_eval, labels)
                return metrics, len(subset_keys)

            if not self._test_keys:
                overall_metrics, overall_matched = _eval_subset(list(self._all_key_set))
                metrics = dict(overall_metrics)
                drops = self._score_difference(self._baseline_overall_metrics or {}, metrics) if metrics else {}
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
                    "overall_results": {
                        "metrics": overall_metrics,
                        "drops": self._score_difference(self._baseline_overall_metrics or {}, overall_metrics) if overall_metrics else {},
                        "matched": overall_matched,
                        "total": len(self._keys),
                    },
                    "val_results": {"metrics": {}, "drops": {}, "matched": 0, "total": 0},
                }
            else:
                train_metrics, train_matched = _eval_subset(list(self._train_key_set))
                test_metrics, test_matched = _eval_subset(list(self._test_key_set))
                overall_metrics, overall_matched = _eval_subset(list(self._all_key_set))

                metrics = dict(test_metrics) if test_metrics else dict(overall_metrics)
                drops = self._score_difference(self._baseline_metrics, metrics) if metrics else {}
                train_drops = self._score_difference(self._baseline_train_metrics or {}, train_metrics) if train_metrics else {}
                overall_drops = self._score_difference(self._baseline_overall_metrics or {}, overall_metrics) if overall_metrics else {}

                evaluations[name] = {
                    "metrics": metrics,
                    "drops": drops,
                    "train_matched": train_matched,
                    "train_total": len(self._train_keys),
                    "test_matched": test_matched,
                    "test_total": len(self._test_keys),
                    "available": overall_matched,
                    "valid": bool(test_matched or train_matched or overall_matched),
                    "train_results": {
                        "metrics": train_metrics,
                        "drops": train_drops,
                        "matched": train_matched,
                        "total": len(self._train_keys),
                    },
                    "test_results": {
                        "metrics": test_metrics,
                        "drops": drops,
                        "matched": test_matched,
                        "total": len(self._test_keys),
                    },
                    "overall_results": {
                        "metrics": overall_metrics,
                        "drops": overall_drops,
                        "matched": overall_matched,
                        "total": len(self._keys),
                    },
                    "val_results": {
                        "metrics": {},
                        "drops": {},
                        "matched": 0,
                        "total": 0,
                    },
                }
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
            },
            "evaluations": evaluations,
            "records": self._record_info,
        }
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
