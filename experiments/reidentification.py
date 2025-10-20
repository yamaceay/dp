from __future__ import annotations

from typing import Callable, Iterable, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from experiments import Experiment, ExperimentResult
from dp.loaders.base import DatasetRecord


class ReidentificationRiskExperiment(Experiment):
    def __init__(
        self,
        records: Iterable[DatasetRecord],
        anonymize: Callable[[str], str],
        max_records: Optional[int] = None,
        vectorizer: Optional[TfidfVectorizer] = None,
    ):
        super().__init__()
        self.anonymize = anonymize
        self.max_records = max_records
        self._vectorizer = vectorizer or TfidfVectorizer(ngram_range=(1, 2), max_features=20000, min_df=2)
        self._texts = self._collect_texts(records)

    def _collect_texts(self, records: Iterable[DatasetRecord]) -> list[str]:
        texts: list[str] = []
        count = 0
        for record in records:
            if self.max_records is not None and count >= self.max_records:
                break
            text = record.text if isinstance(record, DatasetRecord) else str(record)
            if not text:
                continue
            texts.append(text)
            count += 1
        if not texts:
            raise ValueError("No valid records")
        return texts

    def _clone_vectorizer(self) -> TfidfVectorizer:
        params = self._vectorizer.get_params(deep=True)
        return TfidfVectorizer(**params)

    def run(self) -> ExperimentResult:
        original_texts = list(self._texts)
        vectorizer = self._clone_vectorizer()
        reference_matrix = vectorizer.fit_transform(original_texts)
        original_similarity = cosine_similarity(reference_matrix, reference_matrix)
        original_matches = original_similarity.argmax(axis=1)
        original_risk = float(np.mean(original_matches == np.arange(len(original_texts))))
        anonymized_texts = [self.anonymize(text) for text in original_texts]
        anonymized_matrix = vectorizer.transform(anonymized_texts)
        anonymized_similarity = cosine_similarity(anonymized_matrix, reference_matrix)
        anonymized_matches = anonymized_similarity.argmax(axis=1)
        anonymized_risk = float(np.mean(anonymized_matches == np.arange(len(original_texts))))
        privacy_loss = anonymized_risk - original_risk
        metrics = {
            "original_risk": original_risk,
            "anonymized_risk": anonymized_risk,
            "privacy_loss": privacy_loss,
            "loss": privacy_loss,
        }
        return ExperimentResult(score=privacy_loss, metrics=metrics)
