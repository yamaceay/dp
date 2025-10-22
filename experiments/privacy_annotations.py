from __future__ import annotations

from statistics import mean, median
from typing import Any, Dict, List, Optional
from tqdm import tqdm

from dp.experiments import Experiment, ExperimentResult
from dp.loaders.base import DatasetRecord
from dp.utils.tri_detector import TRIDetector


class AnnotationPrivacyExperiment(Experiment):
    def __init__(
        self,
        dataset_name: str,
        original_dataset: List[DatasetRecord],
        evaluation_datasets: Dict[str, List[DatasetRecord]],
        tri_pipeline: str,
        tri_max_length: int = 512,
        tri_device: str = "auto",
    ):
        super().__init__()
        if not dataset_name:
            raise ValueError("dataset_name is required")
        if not original_dataset:
            raise ValueError("original_dataset cannot be empty")
        if not evaluation_datasets:
            raise ValueError("evaluation_datasets cannot be empty")
        if not tri_pipeline:
            raise ValueError("tri_pipeline is required")
        self.dataset_name = dataset_name
        self.original_dataset = list(original_dataset)
        self.evaluation_datasets = {
            key: list(value) for key, value in evaluation_datasets.items() if value
        }
        if not self.evaluation_datasets:
            raise ValueError("evaluation_datasets must contain at least one non-empty dataset")
        self.tri_pipeline = tri_pipeline
        self.tri_max_length = tri_max_length
        self.tri_device = tri_device
        self.detector: Optional[TRIDetector] = None
        self.original_ranks: Dict[str, int] = {}

    def setup(self, **kwargs) -> None:
        self.detector = TRIDetector(
            dataset_name=self.dataset_name,
            max_length=self.tri_max_length,
            device=self.tri_device,
        )
        self.detector.load(self.tri_pipeline)
        self.original_ranks = self._compute_ranks(self.original_dataset, **kwargs)
        super().setup()

    def run(self, **kwargs) -> ExperimentResult:
        if not self.detector or not self.original_ranks:
            raise RuntimeError("setup must be completed before run")
        evaluations: Dict[str, Dict[str, Any]] = {}
        for name, records in self.evaluation_datasets.items():
            ranks = self._compute_ranks(records, **kwargs)
            deltas = self._compute_rank_deltas(ranks)
            evaluations[name] = {
                "ranks": ranks,
                "rank_deltas": deltas,
            }
            summary = self._summarize_rank_deltas(deltas)
            if summary:
                evaluations[name]["rank_delta_summary"] = summary
        metrics: Dict[str, Any] = {
            "original_ranks": self.original_ranks,
            "evaluations": evaluations,
            "uids": {record.uid: record.name for record in self.original_dataset},
        }
        metadata = {
            "dataset": self.dataset_name,
            "tri_pipeline": self.tri_pipeline,
            "tri_max_length": self.tri_max_length,
            "tri_device": str(self.detector.device),
        }
        return ExperimentResult(score=0.0, metrics=metrics, metadata=metadata)

    def cleanup(self) -> None:
        self.detector = None
        super().cleanup()

    def _compute_ranks(self, records: List[DatasetRecord], **kwargs) -> Dict[str, int]:
        if not self.detector:
            raise RuntimeError("detector is not initialized")
        predictions = self.detector.predict(records)
        return self._extract_ranks(records, predictions, **kwargs)

    def _extract_ranks(
        self,
        records: List[DatasetRecord],
        predictions: Dict[str, Dict[str, float]],
        progress: bool = False,
    ) -> Dict[str, int]:
        if not self.detector:
            raise RuntimeError("detector is not initialized")
        uid_to_name = {record.uid: record.name for record in records}
        ranks: Dict[str, int] = {}
        iterator = predictions.items()
        if progress:
            iterator = tqdm(iterator, desc="Computing ranks")
        for uid, scores in iterator:
            name = uid_to_name.get(uid)
            if not name:
                continue
            if name not in self.detector.name_to_label:
                continue
            label_id = self.detector.name_to_label[name]
            ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            rank = self._find_rank(label_id, ordered)
            if rank is not None:
                ranks[uid] = rank
        return ranks

    def _find_rank(self, label_id: int, ordered: List[tuple[str, float]]) -> Optional[int]:
        for position, (label, _) in enumerate(ordered, start=1):
            parsed = self._parse_label(label)
            if parsed == label_id:
                return position
        return None

    def _parse_label(self, label: str) -> Optional[int]:
        if not label:
            return None
        if "_" not in label:
            return None
        _, value = label.rsplit("_", 1)
        if not value.isdigit():
            return None
        return int(value)

    def _compute_rank_deltas(self, anonymized_ranks: Dict[str, int]) -> Dict[str, int]:
        deltas: Dict[str, int] = {}
        for uid, original_rank in self.original_ranks.items():
            if uid in anonymized_ranks:
                deltas[uid] = anonymized_ranks[uid] - original_rank
        return deltas

    def _summarize_rank_deltas(self, deltas: Dict[str, int]) -> Optional[Dict[str, Any]]:
        if not deltas:
            return None
        values = list(deltas.values())
        improved = sum(1 for value in values if value > 0)
        degraded = sum(1 for value in values if value < 0)
        unchanged = sum(1 for value in values if value == 0)
        return {
            "count": len(values),
            "mean": float(mean(values)),
            "median": float(median(values)),
            "min": min(values),
            "max": max(values),
            "improved": improved,
            "degraded": degraded,
            "unchanged": unchanged,
        }
