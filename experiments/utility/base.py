from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from experiments import Experiment, ExperimentResult
from dp.loaders.base import DatasetRecord


@dataclass
class UtilityScores:
    original: float
    anonymized: float
    loss: float


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
    from sklearn.model_selection import StratifiedShuffleSplit

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
        anonymize: Callable[[str], str],
        max_records: Optional[int] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        vectorizer: Optional[TfidfVectorizer] = None,
        classifier: Optional[LogisticRegression] = None,
    ):
        super().__init__()
        self.anonymize = anonymize
        self.max_records = max_records
        self.test_size = test_size
        self.random_state = random_state
        self._vectorizer = vectorizer or TfidfVectorizer(ngram_range=(1, 2), max_features=20000, min_df=2)
        self._classifier = classifier or LogisticRegression(max_iter=1000, class_weight="balanced")

    def load_records(self) -> Iterable[DatasetRecord]:
        raise NotImplementedError

    def get_label(self, record: DatasetRecord) -> Optional[str]:
        raise NotImplementedError

    def filter_records(self, records: Iterable[DatasetRecord]) -> Tuple[List[str], List[str]]:
        texts: List[str] = []
        labels: List[str] = []
        for idx, record in enumerate(records):
            if self.max_records is not None and idx >= self.max_records:
                break
            label = self.get_label(record)
            if label is None:
                continue
            if not record.text:
                continue
            texts.append(record.text)
            labels.append(label)
        if not texts:
            raise ValueError("No valid records")
        return texts, labels

    def fit_and_score(self, train_texts: Sequence[str], train_labels: Sequence[str], test_texts: Sequence[str], test_labels: Sequence[str]) -> float:
        vectorizer = self._clone_vectorizer()
        classifier = self._clone_classifier()
        x_train = vectorizer.fit_transform(train_texts)
        classifier.fit(x_train, train_labels)
        x_test = vectorizer.transform(test_texts)
        predictions = classifier.predict(x_test)
        return float(f1_score(test_labels, predictions, average="weighted"))

    def compute_scores(self) -> Tuple[float, float]:
        records = list(self.load_records())
        texts, labels = self.filter_records(records)
        train_idx, test_idx = split_indices(len(texts), self.test_size, self.random_state, labels)
        train_texts = [texts[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        train_labels = [labels[i] for i in train_idx]
        test_labels = [labels[i] for i in test_idx]
        original_score = self.fit_and_score(train_texts, train_labels, test_texts, test_labels)
        anonymized_texts = [self.anonymize(text) for text in texts]
        anonymized_train = [anonymized_texts[i] for i in train_idx]
        anonymized_test = [anonymized_texts[i] for i in test_idx]
        anonymized_score = self.fit_and_score(anonymized_train, train_labels, anonymized_test, test_labels)
        return original_score, anonymized_score

    def _clone_vectorizer(self) -> TfidfVectorizer:
        params = self._vectorizer.get_params(deep=True)
        return TfidfVectorizer(**params)

    def _clone_classifier(self) -> LogisticRegression:
        params = self._classifier.get_params(deep=True)
        return LogisticRegression(**params)

    def run(self) -> ExperimentResult:
        original_score, anonymized_score = self.compute_scores()
        scores = UtilityScores(original=original_score, anonymized=anonymized_score, loss=original_score - anonymized_score)
        metrics = {
            "original_f1": scores.original,
            "anonymized_f1": scores.anonymized,
            "utility_loss": scores.loss,
            "loss": scores.loss,
        }
        return ExperimentResult(score=scores.loss, metrics=metrics)
