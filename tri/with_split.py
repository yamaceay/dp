from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union

from dp.loaders.base import DatasetRecord
from dp.tri.base import TRIDetector, TRIDataset


class TRIDetectorWithSplit(TRIDetector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_records: Optional[List[DatasetRecord]] = None
        self.test_records: Optional[List[DatasetRecord]] = None

    def setup(
        self,
        records: List[DatasetRecord],
        train_fraction: float = 0.7,
        val_fraction: float = 0.15,
        test_fraction: float = 0.15,
        seed: int = 0,
        stratified: bool = True,
    ) -> None:
        splits = self._split_records(
            records,
            train_fraction,
            val_fraction,
            test_fraction,
            seed,
            stratified,
        )
        self.train_records = splits["train"]
        self.eval_records = splits["val"]
        self.test_records = splits["test"]
        self.build_label_mappings()

    def get_eval_dataset(
        self,
        best_metric_dataset: Optional[str] = None,
        per_step: Optional[int] = None,
    ) -> Tuple[Union[TRIDataset, Dict[str, TRIDataset]], Dict[str, Any]]:
        if not self.eval_records:
            raise ValueError("Validation split is empty")
        eval_dataset = TRIDataset(self.eval_records, self.tokenizer, self.name_to_label, self.max_length)
        eval_kwargs: Dict[str, Any] = {
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_Accuracy",
            "greater_is_better": True,
        }
        if per_step:
            eval_kwargs.update({
                "eval_strategy": "steps",
                "eval_steps": per_step,
            })
        else:
            eval_kwargs.update({
                "eval_strategy": "epoch",
            })
        return eval_dataset, eval_kwargs

    def _split_records(
        self,
        records: List[DatasetRecord],
        train_fraction: float,
        val_fraction: float,
        test_fraction: float,
        seed: int,
        stratified: bool,
    ) -> Dict[str, List[DatasetRecord]]:
        if not records:
            raise ValueError("records cannot be empty")
        if train_fraction < 0 or val_fraction < 0 or test_fraction < 0:
            raise ValueError("Split fractions must be non-negative")
        total_fraction = train_fraction + val_fraction + test_fraction
        if total_fraction <= 0:
            raise ValueError("Split fractions sum must be positive")
        normalized = (
            train_fraction / total_fraction,
            val_fraction / total_fraction,
            test_fraction / total_fraction,
        )
        if stratified:
            return self._stratified_split(records, normalized, seed)
        return self._flat_split(records, normalized, seed)

    def _flat_split(
        self,
        records: List[DatasetRecord],
        fractions: Tuple[float, float, float],
        seed: int,
    ) -> Dict[str, List[DatasetRecord]]:
        pool = list(records)
        random.Random(seed).shuffle(pool)
        counts = self._distribute_counts(len(pool), fractions)
        train_cut = counts[0]
        val_cut = train_cut + counts[1]
        train_split = pool[:train_cut]
        val_split = pool[train_cut:val_cut]
        test_split = pool[val_cut:]
        if not train_split or not val_split or not test_split:
            raise ValueError("Splits are empty; adjust fractions or provide more data")
        self._validate_label_coverage(train_split)
        return {
            "train": train_split,
            "val": val_split,
            "test": test_split,
        }

    def _stratified_split(
        self,
        records: List[DatasetRecord],
        fractions: Tuple[float, float, float],
        seed: int,
    ) -> Dict[str, List[DatasetRecord]]:
        grouped: Dict[str, List[DatasetRecord]] = {}
        for record in records:
            if not record.name:
                raise ValueError("Each record must have a name for stratified splits")
            grouped.setdefault(record.name, []).append(record)
        rng = random.Random(seed)
        splits: Dict[str, List[DatasetRecord]] = {"train": [], "val": [], "test": []}
        for group in grouped.values():
            bucket = list(group)
            rng.shuffle(bucket)
            counts = self._distribute_counts(len(bucket), fractions)
            train_count, val_count, test_count = self._ensure_train_presence(counts)
            idx = 0
            splits["train"].extend(bucket[idx: idx + train_count])
            idx += train_count
            splits["val"].extend(bucket[idx: idx + val_count])
            idx += val_count
            splits["test"].extend(bucket[idx: idx + test_count])
        if not splits["train"] or not splits["val"] or not splits["test"]:
            raise ValueError("Stratified splits produced empty partitions; adjust fractions or data size")
        self._validate_label_coverage(splits["train"])
        return splits

    def _distribute_counts(self, total: int, fractions: Tuple[float, float, float]) -> Tuple[int, int, int]:
        if total <= 0:
            return 0, 0, 0
        weighted = [fractions[i] * total for i in range(3)]
        counts = [math.floor(value) for value in weighted]
        remainder = total - sum(counts)
        order = sorted(range(3), key=lambda idx: weighted[idx] - counts[idx], reverse=True)
        for idx in order:
            if remainder == 0:
                break
            counts[idx] += 1
            remainder -= 1
        return counts[0], counts[1], counts[2]

    def _ensure_train_presence(self, counts: Tuple[int, int, int]) -> Tuple[int, int, int]:
        train, val, test = counts
        total = train + val + test
        if total == 0:
            return 0, 0, 0
        if train == 0:
            if val > 0:
                val -= 1
                train += 1
            elif test > 0:
                test -= 1
                train += 1
            else:
                train = 1
        current = train + val + test
        if current != total:
            test += total - current
        return train, val, test

    def _validate_label_coverage(self, train_records: List[DatasetRecord]) -> None:
        labels = {record.name for record in train_records if record.name}
        if not labels:
            raise ValueError("Training split has no labeled records")
