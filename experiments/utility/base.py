from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit

from experiments import Experiment, ExperimentResult
from dp.loaders.base import DatasetRecord


def split_indices(size: int, test_size: float, random_state: int, labels: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    if size == 0:
        raise ValueError("Empty dataset")
    if size < 2:
        raise ValueError("At least two records required")
    if test_size <= 0 or test_size >= 1:
        raise ValueError("test_size must be in (0, 1)")
    indices = np.arange(size)
    unique_labels = np.unique(labels)
    if unique_labels.size == 1:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)
        split = int(size * (1 - test_size))
        return indices[:split], indices[split:]

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    try:
        train_idx, test_idx = next(splitter.split(indices, labels))
    except ValueError:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)
        split = int(size * (1 - test_size))
        if split == 0 or split == size:
            split = max(1, size - 1)
        train_idx = indices[:split]
        test_idx = indices[split:]
    return train_idx, test_idx


class TextUtilityExperiment(Experiment):
    def __init__(
        self,
        records: List[DatasetRecord],
        max_records: Optional[int] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        vectorizer: Optional[TfidfVectorizer] = None,
        classifier: Optional[LogisticRegression] = None,
    ):
        super().__init__()
        self.records = list(records)
        self.max_records = max_records
        self.test_size = test_size
        self.random_state = random_state
        self._vectorizer = vectorizer or TfidfVectorizer(ngram_range=(1, 2), max_features=20000, min_df=2)
        self._classifier = classifier or LogisticRegression(max_iter=1000, class_weight="balanced")
        self.evaluation_texts: Dict[str, Dict[str, str]] = {}
        self._texts: List[str] = []
        self._labels: List[str] = []
        self._keys: List[str] = []
        self._train_idx: Optional[np.ndarray] = None
        self._test_idx: Optional[np.ndarray] = None
        self._baseline_score: Optional[float] = None
        self._selected_records: List[DatasetRecord] = []
        self.record_info: Dict[str, Dict[str, Any]] = {}

    def set_evaluation_texts(self, evaluation_texts: Dict[str, Dict[str, str]]) -> None:
        self.evaluation_texts = evaluation_texts or {}

    def get_label(self, record: DatasetRecord) -> Optional[str]:
        raise NotImplementedError

    def filter_records(self, records: Iterable[DatasetRecord]) -> Tuple[List[str], List[str], List[str], List[DatasetRecord]]:
        texts: List[str] = []
        labels: List[str] = []
        keys: List[str] = []
        selected: List[DatasetRecord] = []
        for global_index, record in enumerate(records):
            if self.max_records is not None and len(texts) >= self.max_records:
                break
            label = self.get_label(record)
            if label is None:
                continue
            if not record.text:
                continue
            texts.append(record.text)
            labels.append(label)
            key = record.uid or f"record_{global_index}"
            keys.append(key)
            selected.append(record)
        if not texts:
            raise ValueError("No valid records")
        return texts, labels, keys, selected

    def setup(self, **kwargs: Any) -> None:  # type: ignore[override]
        texts, labels, keys, selected = self.filter_records(self.records)
        self._texts = texts
        self._labels = labels
        self._keys = keys
        self._selected_records = selected
        train_idx, test_idx = split_indices(len(texts), self.test_size, self.random_state, labels)
        self._train_idx = train_idx
        self._test_idx = test_idx
        train_texts = [texts[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        train_labels = [labels[i] for i in train_idx]
        test_labels = [labels[i] for i in test_idx]
        self._baseline_score = self.fit_and_score(train_texts, train_labels, test_texts, test_labels)
        self.record_info = {
            key: {
                "index": idx + 1,
                "label": labels[idx],
                "persona_uid": selected[idx].metadata.get("persona_uid"),
            }
            for idx, key in enumerate(keys)
        }
        super().setup(**kwargs)

    def fit_and_score(self, train_texts: Sequence[str], train_labels: Sequence[str], test_texts: Sequence[str], test_labels: Sequence[str]) -> float:
        vectorizer = self._clone_vectorizer()
        classifier = self._clone_classifier()
        x_train = vectorizer.fit_transform(train_texts)
        classifier.fit(x_train, train_labels)
        x_test = vectorizer.transform(test_texts)
        predictions = classifier.predict(x_test)
        return float(f1_score(test_labels, predictions, average="weighted"))

    def _clone_vectorizer(self) -> TfidfVectorizer:
        params = self._vectorizer.get_params(deep=True)
        return TfidfVectorizer(**params)

    def _clone_classifier(self) -> LogisticRegression:
        params = self._classifier.get_params(deep=True)
        return LogisticRegression(**params)

    def run(self, **kwargs: Any) -> ExperimentResult:  # type: ignore[override]
        if self._baseline_score is None or self._train_idx is None or self._test_idx is None:
            raise RuntimeError("setup must be completed before run")
        if not self.evaluation_texts:
            raise RuntimeError("No evaluation datasets provided")
        evaluations: Dict[str, Dict[str, Any]] = {}
        losses: List[float] = []
        train_total = len(self._train_idx)
        test_total = len(self._test_idx)
        for name, texts_map in sorted(self.evaluation_texts.items()):
            matched_train_idx = [idx for idx in self._train_idx if self._keys[idx] in texts_map]
            matched_test_idx = [idx for idx in self._test_idx if self._keys[idx] in texts_map]
            evaluation: Dict[str, Any] = {
                "f1": None,
                "loss": None,
                "train_matched": len(matched_train_idx),
                "test_matched": len(matched_test_idx),
                "train_total": train_total,
                "test_total": test_total,
                "available": len(texts_map),
                "valid": False,
            }
            if matched_train_idx and matched_test_idx:
                train_texts = [texts_map[self._keys[idx]] for idx in matched_train_idx]
                train_labels = [self._labels[idx] for idx in matched_train_idx]
                test_texts = [texts_map[self._keys[idx]] for idx in matched_test_idx]
                test_labels = [self._labels[idx] for idx in matched_test_idx]
                score = self.fit_and_score(train_texts, train_labels, test_texts, test_labels)
                loss = self._baseline_score - score
                evaluation["f1"] = float(score)
                evaluation["loss"] = float(loss)
                evaluation["valid"] = True
                losses.append(loss)
            evaluations[name] = evaluation
        metrics = {
            "baseline": {
                "f1": float(self._baseline_score),
                "train_size": train_total,
                "test_size": test_total,
            },
            "evaluations": evaluations,
            "records": self.record_info,
        }
        overall_loss = max(losses) if losses else 0.0
        return ExperimentResult(score=float(overall_loss), metrics=metrics)
