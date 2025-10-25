from __future__ import annotations

from statistics import mean, median
from typing import Any, Dict, List, Optional
from tqdm import tqdm

from dp.experiments import Experiment, ExperimentResult
from dp.loaders.base import DatasetRecord
from dp.utils.tri_detector import TRIDetector


class TextPrivacyExperiment(Experiment):
    def __init__(
        self,
        tri_pipeline: str,
        tri_max_length: int = 512,
        tri_device: str = "auto",
    ):
        super().__init__()
        if not tri_pipeline:
            raise ValueError("tri_pipeline is required")
        self.tri_pipeline = tri_pipeline
        self.tri_max_length = tri_max_length
        self.tri_device = tri_device
        self.dataset_name: Optional[str] = None
        self.original_dataset: List[DatasetRecord] = []
        self.evaluation_datasets: Dict[str, List[DatasetRecord]] = {}
        self.detector: Optional[TRIDetector] = None
        self.original_ranks: Dict[str, int] = {}
        self.record_keys: List[str] = []
        self.record_info: Dict[str, Dict[str, Any]] = {}

    def setup(
        self,
        dataset_name: str,
        original_dataset: List[DatasetRecord],
        evaluation_datasets: Dict[str, List[DatasetRecord]],
        **kwargs,
    ) -> None:
        if not dataset_name:
            raise ValueError("dataset_name is required")
        if not original_dataset:
            raise ValueError("original_dataset cannot be empty")
        if not evaluation_datasets:
            raise ValueError("evaluation_datasets cannot be empty")
        filtered_evaluations = {
            key: list(value) for key, value in evaluation_datasets.items() if value
        }
        if not filtered_evaluations:
            raise ValueError("evaluation_datasets must contain at least one non-empty dataset")
        self.dataset_name = dataset_name
        self.original_dataset = list(original_dataset)
        self.evaluation_datasets = filtered_evaluations
        self.detector = TRIDetector(
            dataset_name=self.dataset_name,
            max_length=self.tri_max_length,
            device=self.tri_device,
        )
        self.detector.load(self.tri_pipeline)
        self.record_keys = self._build_record_keys(self.original_dataset)
        self.original_ranks = self._compute_ranks(
            self.original_dataset,
            keys=self.record_keys,
            **kwargs,
        )
        self.record_info = {}
        for idx, (key, record) in enumerate(
            zip(self.record_keys, self.original_dataset),
            start=1,
        ):
            self.record_info[key] = {
                "name": record.name,
                "index": record.metadata.get("record_index", idx),
            }
        super().setup(**kwargs)

    def run(self, **kwargs) -> ExperimentResult:
        if not self.detector:
            raise RuntimeError("setup must be completed before run")
        evaluations: Dict[str, Dict[str, Any]] = {}
        for name, records in self.evaluation_datasets.items():
            ranks = self._compute_ranks(records, keys=self.record_keys, **kwargs)
            deltas = self._compute_rank_deltas(ranks)
            evaluations[name] = {
                "ranks": ranks,
                "rank_deltas": deltas,
            }
            summary = self._summarize_rank_deltas(deltas)
            if not summary:
                summary = {
                    "count": 0,
                    "mean": 0.0,
                    "median": 0.0,
                    "min": 0,
                    "max": 0,
                    "improved": 0,
                    "degraded": 0,
                    "unchanged": 0,
                }
            evaluations[name]["rank_delta_summary"] = summary
        metrics: Dict[str, Any] = {
            "original_ranks": self.original_ranks,
            "evaluations": evaluations,
            "records": self.record_info,
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
        self.dataset_name = None
        self.original_dataset = []
        self.evaluation_datasets = {}
        self.original_ranks = {}
        self.record_keys = []
        self.record_info = {}
        super().cleanup()

    def _build_record_keys(self, records: List[DatasetRecord]) -> List[str]:
        return [record.uid for record in records]

    def _compute_ranks(
        self,
        records: List[DatasetRecord],
        keys: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, int]:
        if not self.detector:
            raise RuntimeError("detector is not initialized")
        if keys is None:
            keys = self._build_record_keys(records)
        predictions = self.detector.predict(records)
        return self._extract_ranks(records, keys, predictions, **kwargs)

    def _extract_ranks(
        self,
        records: List[DatasetRecord],
        keys: List[str],
        predictions: Dict[str, Dict[str, float]],
        progress: bool = False,
    ) -> Dict[str, int]:
        if not self.detector:
            raise RuntimeError("detector is not initialized")
        ranks: Dict[str, int] = {}
        iterator = zip(keys, records)
        if progress:
            iterator = tqdm(iterator, total=len(keys), desc="Computing ranks")
        for key, record in iterator:
            name = record.name
            if not name or not self.detector:
                continue
            if name not in self.detector.name_to_label:
                continue
            scores = predictions.get(record.uid)
            if not scores:
                continue
            ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            rank = self._rank_for_name(name, ordered)
            if rank is not None:
                ranks[key] = rank
        return ranks

    def _rank_for_name(self, name: str, ordered: List[tuple[str, float]]) -> Optional[int]:
        for position, (candidate, _) in enumerate(ordered, start=1):
            if candidate == name:
                return position
        return None

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