from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from bert_score import score

from dp.experiments.divergence.base import DivergenceMetric, TextDivergenceExperiment


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
