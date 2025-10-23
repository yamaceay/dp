from __future__ import annotations

from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional

from dp.experiments import Experiment, ExperimentResult


class SemanticDivergenceExperiment(Experiment):
    def __init__(
        self,
        original_texts: Dict[str, str],
        evaluation_datasets: Dict[str, Dict[str, Any]],
        record_info: Dict[str, Dict[str, Any]],
        model_type: Optional[str] = None,
        language: Optional[str] = None,
        batch_size: int = 16,
        device: Optional[str] = None,
        rescale_with_baseline: bool = False,
        metric: str = "bertscore",
    ):
        super().__init__()
        if not original_texts:
            raise ValueError("original_texts cannot be empty")
        if not evaluation_datasets:
            raise ValueError("evaluation_datasets cannot be empty")
        if metric != "bertscore":
            raise ValueError(f"Unsupported divergence metric '{metric}'")
        self.original_texts = dict(original_texts)
        self.evaluation_datasets = {
            name: {
                "texts": dict(payload.get("texts", {})),
                "total": int(payload.get("total", len(payload.get("texts", {})))),
            }
            for name, payload in evaluation_datasets.items()
            if payload.get("texts")
        }
        if not self.evaluation_datasets:
            raise ValueError("evaluation_datasets must contain at least one non-empty dataset")
        self.record_info = record_info
        self.model_type = model_type
        self.language = language
        self.batch_size = batch_size
        self.device = device
        self.rescale_with_baseline = rescale_with_baseline
        self.metric = metric

    def run(self, **kwargs: Any) -> ExperimentResult:
        try:
            from bert_score import score  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("bert-score package is required for SemanticDivergenceExperiment") from exc
        evaluations: Dict[str, Dict[str, Any]] = {}
        divergence_means: List[float] = []
        config = self._build_score_kwargs()
        total_records = len(self.original_texts)
        for name, payload in self.evaluation_datasets.items():
            texts = payload["texts"]
            total = payload.get("total", len(texts))
            matched_keys = [key for key in self.original_texts if key in texts]
            if not matched_keys:
                evaluations[name] = {
                    "similarity": {},
                    "divergence": {},
                    "summary": None,
                    "matched": 0,
                    "total": total,
                }
                continue
            references = [self.original_texts[key] for key in matched_keys]
            candidates = [texts[key] for key in matched_keys]
            _, _, f1 = score(candidates, references, **config)
            similarities = [float(value) for value in f1.tolist()]
            divergence_values = [1.0 - value for value in similarities]
            similarity_map = {key: sim for key, sim in zip(matched_keys, similarities)}
            divergence_map = {key: div for key, div in zip(matched_keys, divergence_values)}
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
        }
        metadata = {
            "metric": self.metric,
            "model_type": self.model_type,
            "language": self.language,
            "batch_size": self.batch_size,
            "device": self.device,
            "rescale_with_baseline": self.rescale_with_baseline,
        }
        return ExperimentResult(score=score_value, metrics=metrics, metadata=metadata)

    def _build_score_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if self.model_type is not None:
            kwargs["model_type"] = self.model_type
        if self.language is not None:
            kwargs["lang"] = self.language
        kwargs["batch_size"] = self.batch_size
        if self.device is not None:
            kwargs["device"] = self.device
        kwargs["rescale_with_baseline"] = self.rescale_with_baseline
        return kwargs

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
