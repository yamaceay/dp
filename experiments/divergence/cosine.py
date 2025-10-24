from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from dp.experiments.divergence.base import DivergenceMetric, TextDivergenceExperiment
from dp.experiments.utility.vectorizer import TextVectorizer, TfidfTextVectorizer


class CosineSimilarityMetric(DivergenceMetric):
    def __init__(self, vectorizer: TextVectorizer):
        super().__init__("cosine")
        self._template = vectorizer.clone()
        self._vectorizer: Optional[TextVectorizer] = None

    def clone(self) -> "CosineSimilarityMetric":
        return CosineSimilarityMetric(self._template.clone())

    def prepare(self, references: Dict[str, str]) -> None:
        vectorizer = self._template.clone()
        vectorizer.fit(list(references.values()))
        self._vectorizer = vectorizer

    def similarities(self, references: Sequence[str], candidates: Sequence[str]) -> List[float]:
        if self._vectorizer is None:
            raise RuntimeError("cosine similarity metric is not prepared")
        ref_matrix = self._vectorizer.transform(list(references))
        cand_matrix = self._vectorizer.transform(list(candidates))
        matrix = cosine_similarity(cand_matrix, ref_matrix)
        diagonal = np.diag(matrix)
        return [float(value) for value in diagonal.tolist()]

    def metadata(self) -> Dict[str, Any]:
        description = self._template.describe()
        payload: Dict[str, Any] = {"name": self.name}
        if description:
            payload["vectorizer"] = description
        return payload

    def cleanup(self) -> None:
        self._vectorizer = None


class CosineSimilarityDivergence(TextDivergenceExperiment):
    def __init__(self, vectorizer: TextVectorizer):
        metric = CosineSimilarityMetric(vectorizer)
        super().__init__(metric)
