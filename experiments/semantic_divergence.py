from __future__ import annotations

from abc import ABC, abstractmethod
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from bert_score import score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from dp.experiments import Experiment, ExperimentResult


class DivergenceMetric(ABC):
    def __init__(self, name: str):
        if not name:
            raise ValueError("metric name is required")
        self.name = name

    @abstractmethod
    def clone(self) -> "DivergenceMetric":
        raise NotImplementedError

    def prepare(self, references: Dict[str, str]) -> None:
        return

    @abstractmethod
    def similarities(self, references: Sequence[str], candidates: Sequence[str]) -> List[float]:
        raise NotImplementedError

    def metadata(self) -> Dict[str, Any]:
        return {"name": self.name}

    def cleanup(self) -> None:
        return


class TextDivergenceExperiment(Experiment, ABC):
    def __init__(self, metric: DivergenceMetric):
        super().__init__()
        self.metric_template = metric
        self.metric: Optional[DivergenceMetric] = None
        self.original_texts: Dict[str, str] = {}
        self.evaluation_datasets: Dict[str, Dict[str, Any]] = {}
        self.record_info: Dict[str, Dict[str, Any]] = {}
        self.metric_metadata: Dict[str, Any] = {}

    def setup(
        self,
        original_texts: Dict[str, str],
        evaluation_datasets: Dict[str, Dict[str, Any]],
        record_info: Dict[str, Dict[str, Any]],
        **kwargs,
    ) -> None:
        if not original_texts:
            raise ValueError("original_texts cannot be empty")
        if not evaluation_datasets:
            raise ValueError("evaluation_datasets cannot be empty")
        filtered = {}
        for name, payload in evaluation_datasets.items():
            texts = dict(payload.get("texts", {}))
            if not texts:
                continue
            total = int(payload.get("total", len(texts)))
            filtered[name] = {
                "texts": texts,
                "total": total,
            }
        if not filtered:
            raise ValueError("evaluation_datasets must contain at least one non-empty dataset")
        self.original_texts = dict(original_texts)
        self.evaluation_datasets = filtered
        self.record_info = dict(record_info)
        self.metric = self.metric_template.clone()
        self.metric.prepare(self.original_texts)
        self.metric_metadata = self.metric.metadata()
        super().setup(**kwargs)

    def run(self, **kwargs: Any) -> ExperimentResult:
        if not self.metric:
            raise RuntimeError("setup must be completed before run")
        evaluations: Dict[str, Dict[str, Any]] = {}
        divergence_means: List[float] = []
        total_records = len(self.original_texts)
        for name, payload in self.evaluation_datasets.items():
            texts = payload["texts"]
            total = payload["total"]
            matched_keys = [key for key in self.original_texts if key in texts]
            if not matched_keys:
                evaluations[name] = {
                    "similarity": {},
                    "divergence": {},
                    "summary": None,
                    "matched": 0,
                    "total": total,
                    "missing": total_records,
                }
                continue
            references = [self.original_texts[key] for key in matched_keys]
            candidates = [texts[key] for key in matched_keys]
            similarities = self.metric.similarities(references, candidates)
            divergence_values = [1.0 - value for value in similarities]
            similarity_map = {key: float(similarities[idx]) for idx, key in enumerate(matched_keys)}
            divergence_map = {key: float(divergence_values[idx]) for idx, key in enumerate(matched_keys)}
            summary = self._summarize(similarities, divergence_values)
            evaluations[name] = {
                "similarity": similarity_map,
                "divergence": divergence_map,
                "summary": summary,
                "matched": len(matched_keys),
                "total": total,
                "missing": max(total_records - len(matched_keys), 0),
            }
            if summary:
                divergence_means.append(summary["divergence_mean"])
        score_value = float(sum(divergence_means) / len(divergence_means)) if divergence_means else 0.0
        metrics = {
            "records": self.record_info,
            "original": {
                "count": total_records,
            },
            "evaluations": evaluations,
            "metric": self.metric_metadata.get("name"),
            "metric_metadata": self.metric_metadata,
        }
        metadata = dict(self.metric_metadata)
        return ExperimentResult(score=score_value, metrics=metrics, metadata=metadata)

    def cleanup(self) -> None:
        if self.metric:
            self.metric.cleanup()
        self.metric = None
        self.original_texts = {}
        self.evaluation_datasets = {}
        self.record_info = {}
        self.metric_metadata = {}
        super().cleanup()

    def _summarize(
        self,
        similarities: Iterable[float],
        divergences: Iterable[float],
    ) -> Optional[Dict[str, float]]:
        similarity_values = list(similarities)
        divergence_values = list(divergences)
        if not similarity_values or not divergence_values:
            return None
        return {
            "count": len(similarity_values),
            "similarity_mean": float(mean(similarity_values)),
            "similarity_median": float(median(similarity_values)),
            "similarity_min": float(min(similarity_values)),
            "similarity_max": float(max(similarity_values)),
            "divergence_mean": float(mean(divergence_values)),
            "divergence_median": float(median(divergence_values)),
            "divergence_min": float(min(divergence_values)),
            "divergence_max": float(max(divergence_values)),
        }


class BERTScoreMetric(DivergenceMetric):
    def __init__(
        self,
        model_type: Optional[str] = None,
        language: Optional[str] = None,
        batch_size: int = 16,
        device: Optional[str] = None,
        rescale_with_baseline: bool = False,
    ):
        super().__init__("bertscore")
        self.model_type = model_type
        self.language = language
        self.batch_size = batch_size
        self.device = device
        self.rescale_with_baseline = rescale_with_baseline

    def clone(self) -> "BERTScoreMetric":
        return BERTScoreMetric(
            model_type=self.model_type,
            language=self.language,
            batch_size=self.batch_size,
            device=self.device,
            rescale_with_baseline=self.rescale_with_baseline,
        )

    def similarities(self, references: Sequence[str], candidates: Sequence[str]) -> List[float]:
        kwargs: Dict[str, Any] = {"batch_size": self.batch_size, "rescale_with_baseline": self.rescale_with_baseline}
        if self.model_type is not None:
            kwargs["model_type"] = self.model_type
        if self.language is not None:
            kwargs["lang"] = self.language
        if self.device is not None:
            kwargs["device"] = self.device
        _, _, f1 = score(list(candidates), list(references), **kwargs)
        return [float(value) for value in f1.tolist()]

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model_type": self.model_type,
            "language": self.language,
            "batch_size": self.batch_size,
            "device": self.device,
            "rescale_with_baseline": self.rescale_with_baseline,
        }


class CosineSimilarityMetric(DivergenceMetric):
    def __init__(self, vectorizer: Optional[TfidfVectorizer] = None):
        super().__init__("cosine")
        template = vectorizer or TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self._vectorizer_params = template.get_params(deep=False)
        self._vectorizer: Optional[TfidfVectorizer] = None

    def clone(self) -> "CosineSimilarityMetric":
        return CosineSimilarityMetric(TfidfVectorizer(**self._vectorizer_params))

    def prepare(self, references: Dict[str, str]) -> None:
        vectorizer = TfidfVectorizer(**self._vectorizer_params)
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
        return {
            "name": self.name,
            "vectorizer": {
                "ngram_range": self._vectorizer_params.get("ngram_range"),
                "max_features": self._vectorizer_params.get("max_features"),
                "min_df": self._vectorizer_params.get("min_df"),
            },
        }

    def cleanup(self) -> None:
        self._vectorizer = None


class BERTScoreDivergence(TextDivergenceExperiment):
    def __init__(
        self,
        model_type: Optional[str] = None,
        language: Optional[str] = None,
        batch_size: int = 16,
        device: Optional[str] = None,
        rescale_with_baseline: bool = False,
    ):
        metric = BERTScoreMetric(
            model_type=model_type,
            language=language,
            batch_size=batch_size,
            device=device,
            rescale_with_baseline=rescale_with_baseline,
        )
        super().__init__(metric)


class CosineSimilarityDivergence(TextDivergenceExperiment):
    def __init__(self, vectorizer: Optional[TfidfVectorizer] = None):
        metric = CosineSimilarityMetric(vectorizer)
        super().__init__(metric)
