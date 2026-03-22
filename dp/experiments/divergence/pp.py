from __future__ import annotations

from statistics import mean, median, pvariance
from typing import Any, Dict, List, Optional

from dp.experiments import ExperimentResult
from dp.experiments.divergence.base import DivergenceMetric, TextDivergenceExperiment


class PerturbationPercentageMetric(DivergenceMetric):
    def __init__(self) -> None:
        super().__init__("pp")

    def clone(self) -> "PerturbationPercentageMetric":
        return PerturbationPercentageMetric()

    def similarities(self, references: List[str], candidates: List[str]) -> List[float]:
        return []

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "definition": "pp=perturbed/total",
            "reliability": "variance(pp)",
        }


class PerturbationPercentageDivergence(TextDivergenceExperiment):
    def __init__(self) -> None:
        super().__init__(PerturbationPercentageMetric())

    @staticmethod
    def _as_non_negative_int(value: Any) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return 0
        return max(parsed, 0)

    @staticmethod
    def _compute_pp(metadata: Dict[str, Any]) -> float:
        perturbed = PerturbationPercentageDivergence._as_non_negative_int(metadata.get("perturbed", metadata.get("masked")))
        total = PerturbationPercentageDivergence._as_non_negative_int(metadata.get("total"))
        if total <= 0:
            return None
        value = float(perturbed) / float(total)
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value

    def run(self, **kwargs: Any) -> ExperimentResult:
        if not self.metric:
            raise RuntimeError("setup must be completed before run")
        evaluations: Dict[str, Dict[str, Any]] = {}
        divergence_means: List[float] = []
        total_records = len(self.original_texts)

        for name, payload in self.evaluation_datasets.items():
            if name in ["spacy", "presidio", "manual", "dpbart", "dpprompt", "dpparaphrase"]:
                print(f"Skipping evaluation for {name} as it is a known method with expected metadata format.")
                continue
            metadata_by_key: Dict[str, Dict[str, Any]] = dict(payload.get("metadata", {}))
            total = int(payload.get("total", len(metadata_by_key)))
            matched_keys = [key for key in self.original_texts if key in metadata_by_key]
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

            pp_values = [self._compute_pp(metadata_by_key[key]) for key in matched_keys]
            similarities = [1.0 - value for value in pp_values]
            similarity_map = {key: float(similarities[idx]) for idx, key in enumerate(matched_keys)}
            divergence_map = {key: float(pp_values[idx]) for idx, key in enumerate(matched_keys)}
            summary = self._summarize_pp(pp_values)
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

    @staticmethod
    def _summarize_pp(pp_values: List[float]) -> Optional[Dict[str, float]]:
        if not pp_values:
            return None
        similarities = [1.0 - value for value in pp_values]
        variance = float(pvariance(pp_values)) if len(pp_values) > 1 else 0.0
        return {
            "count": len(pp_values),
            "similarity_mean": float(mean(similarities)),
            "similarity_median": float(median(similarities)),
            "similarity_min": float(min(similarities)),
            "similarity_max": float(max(similarities)),
            "divergence_mean": float(mean(pp_values)),
            "divergence_median": float(median(pp_values)),
            "divergence_min": float(min(pp_values)),
            "divergence_max": float(max(pp_values)),
            "divergence_variance": variance,
            "pp_variance": variance,
        }
