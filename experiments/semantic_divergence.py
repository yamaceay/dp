from __future__ import annotations

from typing import Callable, Iterable, Optional

from experiments import Experiment, ExperimentResult
from dp.loaders.base import DatasetRecord


class SemanticDivergenceExperiment(Experiment):
    def __init__(
        self,
        records: Iterable[DatasetRecord],
        anonymize: Callable[[str], str],
        max_records: Optional[int] = None,
        model_type: Optional[str] = None,
        language: Optional[str] = None,
        batch_size: int = 16,
        device: Optional[str] = None,
        rescale_with_baseline: bool = False,
    ):
        super().__init__()
        self.anonymize = anonymize
        self.max_records = max_records
        self.model_type = model_type
        self.language = language
        self.batch_size = batch_size
        self.device = device
        self.rescale_with_baseline = rescale_with_baseline
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
        if len(texts) < 2:
            raise ValueError("At least two records required")
        return texts

    def run(self) -> ExperimentResult:
        try:
            from bert_score import score
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("bert-score package is required for SemanticDivergenceExperiment") from exc
        original = list(self._texts)
        anonymized = [self.anonymize(text) for text in original]
        kwargs = {}
        if self.model_type is not None:
            kwargs["model_type"] = self.model_type
        if self.language is not None:
            kwargs["lang"] = self.language
        kwargs["batch_size"] = self.batch_size
        if self.device is not None:
            kwargs["device"] = self.device
        kwargs["rescale_with_baseline"] = self.rescale_with_baseline
        _, _, f1 = score(anonymized, original, **kwargs)
        divergence = 1.0 - float(f1.mean().item())
        metrics = {
            "semantic_similarity": 1.0 - divergence,
            "divergence_loss": divergence,
            "loss": divergence,
        }
        return ExperimentResult(score=divergence, metrics=metrics)
