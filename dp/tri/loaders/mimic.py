from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import tqdm

from dp.loaders.base import DatasetRecord
from dp.loaders import get_adapter
from dp.tri.loaders.base import AttackerDatasetAdapter
from dp.tri.loaders.base import AttackerDatasetRecord, merge_records

class MIMICAttackerDatasetAdapter(AttackerDatasetAdapter):
    def __init__(
        self,
        **data_kwargs,
    ):
        adapter = get_adapter("mimic", **data_kwargs)
        super().__init__(adapter=adapter)

    def iter_records(self, progress: bool = False) -> Iterable[AttackerDatasetRecord]:
        records_list = list(self.adapter.iter_records())
        train_texts_grouped: Dict[str, List[str]] = {}
        eval_texts_grouped: Dict[str, List[str]] = {}
        for record in records_list:
            split = record.metadata.get("split")
            if split == "train":
                train_texts_grouped.setdefault(record.name, []).append(record.text)
            elif split == "val":
                eval_texts_grouped.setdefault(record.name, []).append(record.text)
        iterator = merge_records(train_texts_grouped, eval_texts_grouped)
        if progress:
            iterator = tqdm.tqdm(iterator, desc="Processing attacker records", total=len(iterator))
        for attacker_record in iterator:
            train_texts = [r for r in attacker_record.train_texts]
            eval_texts = [r for r in attacker_record.eval_texts]
            yield AttackerDatasetRecord(
                name=attacker_record.name,
                train_texts=train_texts,
                eval_texts=eval_texts,
            )